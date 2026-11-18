# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from argparse import Namespace
from io import BytesIO
from unittest import TestCase
import json
import hashlib
import os.path
import shutil
import tempfile

from hwp5.attachments import build_inventory
from hwp5.attachments import inventory_from_hwp5file
from hwp5.proc import attachments as attachments_command
from hwp5.binmodel import BinData
from hwp5.xmlmodel import Hwp5File

from .fixtures import get_fixture_path


def make_bindata_model(seqno, storage_type, **bindata):
    flags = BinData.Flags(0)
    flags = int(storage_type) | (int(flags) & ~0x7)
    return {
        'type': BinData,
        'seqno': seqno,
        'level': 1,
        'content': dict(flags=flags, bindata=bindata),
    }


class MemoryStream(object):
    def __init__(self, data):
        self.data = data

    def open(self):
        return BytesIO(self.data)


class MemoryStorage(object):
    def __init__(self, **items):
        self.items = items

    def __iter__(self):
        return iter(sorted(self.items))

    def __getitem__(self, name):
        return self.items[name]


def sample_model(bindata_id):
    return {
        'type': type(str('ShapePicture'), (object,), {}),
        'seqno': bindata_id,
        'level': 3,
        'content': dict(picture=dict(bindata_id=bindata_id)),
    }


class BuildInventoryTest(TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def write(self, name, data):
        path = os.path.join(self.tmpdir, name)
        directory = os.path.dirname(path)
        if not os.path.isdir(directory):
            os.makedirs(directory)
        with open(path, 'wb') as f:
            f.write(data)
        return path

    def test_duplicate_records_are_not_merged(self):
        # The same storage identity declared twice, with the same content.
        models = [
            make_bindata_model(2, BinData.StorageType.EMBEDDING,
                               storage_id=1, ext='png'),
            make_bindata_model(3, BinData.StorageType.EMBEDDING,
                               storage_id=1, ext='png'),
        ]
        storage = MemoryStorage(**{'BIN0001.png': MemoryStream(b'image1')})
        inventory = build_inventory(
            models, bindata_storage=storage, base_dir=self.tmpdir,
        )
        self.assertEqual(2, len(inventory['attachments']))
        first, second = inventory['attachments']
        self.assertEqual(1, first['bindata_id'])
        self.assertEqual(2, second['bindata_id'])
        self.assertEqual('BinData/BIN0001.png', first['identity'])
        self.assertEqual('BinData/BIN0001.png', second['identity'])
        self.assertEqual([2], first['shared']['storage_with'])
        self.assertEqual([], inventory['name_collisions'])
        self.assertEqual([], first['references'])
        self.assertEqual([], second['references'])
    def test_shared_content_keeps_separate_storage(self):
        # Different storage identities, byte-identical content.
        models = [
            make_bindata_model(2, BinData.StorageType.EMBEDDING,
                               storage_id=1, ext='png'),
            make_bindata_model(3, BinData.StorageType.EMBEDDING,
                               storage_id=2, ext='png'),
        ]
        storage = MemoryStorage(**{
            'BIN0001.png': MemoryStream(b'same'),
            'BIN0002.png': MemoryStream(b'same'),
        })
        inventory = build_inventory(
            models, bindata_storage=storage, base_dir=self.tmpdir,
        )
        first, second = inventory['attachments']
        self.assertNotEqual(first['identity'], second['identity'])
        self.assertEqual([2], first['shared']['content_with'])
        self.assertNotIn('storage_with', first)
        self.assertEqual(
            hashlib.sha256(b'same').hexdigest(),
            first['content']['sha256'],
        )

    def test_references_are_kept_per_occurrence(self):
        models = [
            make_bindata_model(2, BinData.StorageType.EMBEDDING,
                               storage_id=1, ext='png'),
        ]
        storage = MemoryStorage(**{'BIN0001.png': MemoryStream(b'x')})
        sections = [{
            'part': 'BodyText',
            'index': 0,
            'stream': 'BodyText/Section0',
            'models': [sample_model(1), sample_model(1), sample_model(1)],
        }]
        inventory = build_inventory(
            models, sections=sections, bindata_storage=storage,
            base_dir=self.tmpdir,
        )
        entry = inventory['attachments'][0]
        self.assertEqual(3, len(entry['references']))
        self.assertEqual([0, 0, 0],
                         [ref['section'] for ref in entry['references']])

    def test_unresolved_link_is_pending_not_substituted(self):
        # The document declares a link to image1, but the only supplied
        # stream with that basename is an attachment declared separately.
        models = [
            make_bindata_model(2, BinData.StorageType.LINK,
                               abspath=r'C:\docs\image1.png',
                               relpath='image1.png'),
            make_bindata_model(3, BinData.StorageType.EMBEDDING,
                               storage_id=1, ext='png'),
        ]
        storage = MemoryStorage(**{'BIN0001.png': MemoryStream(b'other')})
        inventory = build_inventory(
            models, bindata_storage=storage, base_dir=self.tmpdir,
        )
        link, embedded = inventory['attachments']
        self.assertEqual('link', link['storage_type'])
        self.assertEqual('pending', link['status'])
        self.assertTrue(link['external']['retrieval_required'])
        self.assertEqual('available', embedded['status'])
        self.assertEqual('image1.png', link['name'])
        self.assertEqual('BIN0001.png', embedded['name'])
        self.assertNotEqual(link['identity'], embedded['identity'])
        self.assertEqual('pending', link['status'])
        self.assertEqual(1, len(inventory['pending_links']))
        self.assertEqual(1, inventory['summary']['pending_links'])
        # The other supplied stream is never used to satisfy the link.
        self.assertIsNone(link['content'])

    def test_same_named_links_are_a_collision(self):
        models = [
            make_bindata_model(2, BinData.StorageType.LINK,
                               abspath=r'C:\a\image1.png',
                               relpath=r'a\image1.png'),
            make_bindata_model(3, BinData.StorageType.LINK,
                               abspath=r'C:\b\image1.png',
                               relpath=r'b\image1.png'),
        ]
        inventory = build_inventory(models, base_dir=self.tmpdir)
        self.assertEqual(1, len(inventory['name_collisions']))
        collision = inventory['name_collisions'][0]
        self.assertEqual('image1.png', collision['name'])
        self.assertEqual([1, 2], collision['bindata_ids'])
        self.assertEqual(
            ['link:a/image1.png', 'link:b/image1.png'],
            collision['identities'],
        )
        for entry in inventory['attachments']:
            self.assertTrue(entry['name_collision'])
            self.assertNotEqual(entry['identity'],
                                [x for x in inventory['attachments']
                                 if x is not entry][0]['identity'])

    def test_relative_link_uses_document_directory(self):
        path = self.write('image1.png', b'external')
        models = [
            make_bindata_model(2, BinData.StorageType.LINK,
                               abspath=r'C:\elsewhere\image1.png',
                               relpath='image1.png'),
        ]
        inventory = build_inventory(
            models, document_path=os.path.join(self.tmpdir, 'doc.hwp'),
        )
        entry = inventory['attachments'][0]
        self.assertEqual(path, entry['external']['resolved_path'])
        self.assertEqual('document-relative',
                         entry['external']['resolution'])
        self.assertEqual('available', entry['status'])

    def test_absolute_link_is_not_joined_to_document_directory(self):
        models = [
            make_bindata_model(2, BinData.StorageType.LINK,
                               abspath=r'C:\docs\image1.png',
                               relpath=r'C:\docs\image1.png'),
        ]
        inventory = build_inventory(
            models, base_dir=self.tmpdir,
        )
        entry = inventory['attachments'][0]
        self.assertEqual('C:/docs/image1.png',
                         entry['external']['resolved_path'])
        self.assertEqual('pending', entry['status'])

    def test_missing_embedded_stream_is_not_reported_as_present(self):
        models = [
            make_bindata_model(2, BinData.StorageType.EMBEDDING,
                               storage_id=1, ext='png'),
        ]
        storage = MemoryStorage(**{'BIN0002.png': MemoryStream(b'other')})
        inventory = build_inventory(
            models, bindata_storage=storage, base_dir=self.tmpdir,
        )
        entry = inventory['attachments'][0]
        self.assertEqual('missing', entry['status'])
        self.assertIsNone(entry['content'])

    def test_undeclared_stream_is_listed_separately(self):
        models = [
            make_bindata_model(2, BinData.StorageType.EMBEDDING,
                               storage_id=1, ext='png'),
        ]
        storage = MemoryStorage(**{
            'BIN0001.png': MemoryStream(b'declared'),
            'BIN0002.png': MemoryStream(b'extra'),
        })
        inventory = build_inventory(
            models, bindata_storage=storage, base_dir=self.tmpdir,
        )
        names = [x['name'] for x in inventory['unreferenced_streams']]
        self.assertEqual(['BIN0002.png'], names)
        self.assertFalse(inventory['unreferenced_streams'][0]['declared'])


class FixtureInventoryTest(TestCase):

    def test_sample_5017(self):
        path = get_fixture_path('sample-5017.hwp')
        hwp5file = Hwp5File(path)
        inventory = inventory_from_hwp5file(hwp5file, path)
        self.assertEqual(2, len(inventory['attachments']))
        first = inventory['attachments'][0]
        self.assertEqual('BIN0002', first['storage_id'])
        self.assertEqual('BIN0002.jpg', first['name'])
        self.assertEqual(1, len(first['references']))
        self.assertEqual('BodyText/Section0',
                         first['references'][0]['stream'])
        self.assertEqual('available', first['status'])
        self.assertFalse(inventory['document']['declared_bindata_count']
                         != len(inventory['attachments']))

    def test_sample_5017_pics_shared_content(self):
        path = get_fixture_path('sample-5017-pics.hwp')
        hwp5file = Hwp5File(path)
        inventory = inventory_from_hwp5file(hwp5file, path)
        self.assertEqual(2, len(inventory['attachments']))
        self.assertEqual(11, inventory['summary']['references'])
        for entry in inventory['attachments']:
            self.assertIn('content_with', entry['shared'])

    def test_borderfill_indirect_reference(self):
        path = get_fixture_path('borderfill.hwp')
        hwp5file = Hwp5File(path)
        inventory = inventory_from_hwp5file(hwp5file, path)
        entry = inventory['attachments'][0]
        self.assertEqual(3, len(entry['references']))
        self.assertEqual({'borderfill'},
                         set(ref['kind'] for ref in entry['references']))
        self.assertEqual(3, len(entry['definition_references']))


class CommandTest(TestCase):

    def test_json_output(self):
        path = get_fixture_path('sample-5017.hwp')
        output = tempfile.NamedTemporaryFile(delete=False)
        output.close()
        try:
            args = Namespace(
                hwp5file=path,
                format='json',
                output=output.name,
                base_dir=None,
                no_hash=False,
            )
            attachments_command.main(args)
            with open(output.name, 'rb') as f:
                inventory = json.loads(f.read().decode('utf-8'))
        finally:
            os.unlink(output.name)

        self.assertEqual(2, len(inventory['attachments']))
        self.assertEqual('available',
                         inventory['attachments'][0]['status'])

    def test_table_output(self):
        path = get_fixture_path('sample-5017.hwp')
        output = tempfile.NamedTemporaryFile(delete=False)
        output.close()
        try:
            args = Namespace(
                hwp5file=path,
                format='table',
                output=output.name,
                base_dir=None,
                no_hash=True,
            )
            attachments_command.main(args)
            with open(output.name, 'rb') as f:
                text = f.read().decode('utf-8')
        finally:
            os.unlink(output.name)

        lines = text.splitlines()
        self.assertTrue(lines[0].startswith('ID'))
        self.assertIn('BIN0002.jpg', lines[1])
        self.assertIn('available', lines[1])


class ExtraReferenceTest(TestCase):

    def test_unmatched_bindata_id_is_reported_not_dropped(self):
        models = [
            make_bindata_model(2, BinData.StorageType.EMBEDDING,
                               storage_id=1, ext='png'),
        ]
        storage = MemoryStorage(**{'BIN0001.png': MemoryStream(b'x')})
        sections = [{
            'part': 'BodyText',
            'index': 0,
            'stream': 'BodyText/Section0',
            'models': [sample_model(9)],
        }]
        inventory = build_inventory(
            models, sections=sections, bindata_storage=storage,
            base_dir='/tmp',
        )
        self.assertEqual([], inventory['attachments'][0]['references'])
        self.assertEqual(1, len(inventory['unresolved_references']))
        unresolved = inventory['unresolved_references'][0]
        self.assertEqual(9, unresolved['bindata_id'])
        self.assertIn('no HWPTAG_BIN_DATA record',
                      unresolved['reason'])

    def test_ole_storage_reference_is_linked_by_identity(self):
        models = [
            make_bindata_model(2, BinData.StorageType.EMBEDDING,
                               storage_id=7, ext='bin'),
            make_bindata_model(3, BinData.StorageType.EMBEDDING,
                               storage_id=7, ext='bin'),
        ]
        storage = MemoryStorage(**{'BIN0007.bin': MemoryStream(b'x')})
        ole_model = {
            'type': type(str('ShapeOLE'), (object,), {}),
            'seqno': 4,
            'level': 2,
            'content': dict(storage_id=7),
        }
        sections = [{
            'part': 'BodyText',
            'index': 0,
            'stream': 'BodyText/Section0',
            'models': [ole_model],
        }]
        inventory = build_inventory(
            models, sections=sections, bindata_storage=storage,
            base_dir='/tmp',
        )
        self.assertEqual(1, len(inventory['storage_references']))
        reference = inventory['storage_references'][0]
        self.assertEqual('BIN0007', reference['storage_id'])
        self.assertEqual([1, 2], reference['attachment_bindata_ids'])
        # The OLE reference is not a bindata-id occurrence, so it does not
        # invent a body reference for either declared attachment.
        for entry in inventory['attachments']:
            self.assertEqual([], entry['references'])


class LinkPathDeclarationTest(TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_relative_link_wins_over_absolute_declaration(self):
        with open(os.path.join(self.tmpdir, 'image1.png'), 'wb') as f:
            f.write(b'relative')
        models = [
            make_bindata_model(2, BinData.StorageType.LINK,
                               abspath=r'C:\\elsewhere\\image1.png',
                               relpath='image1.png'),
        ]
        inventory = build_inventory(models, base_dir=self.tmpdir)
        external = inventory['attachments'][0]['external']
        self.assertEqual(
            os.path.join(self.tmpdir, 'image1.png'),
            external['resolved_path'],
        )
        self.assertEqual('document-relative', external['resolution'])
        self.assertTrue(external['exists'])

    def test_link_without_base_is_unresolved_not_guessed(self):
        models = [
            make_bindata_model(2, BinData.StorageType.LINK,
                               abspath=r'C:\\docs\\image1.png',
                               relpath='image1.png'),
        ]
        inventory = build_inventory(models)
        entry = inventory['attachments'][0]
        self.assertEqual('pending', entry['status'])
        self.assertEqual('unresolved-relative',
                         entry['external']['resolution'])
        self.assertFalse(entry['external']['exists'])
        self.assertTrue(entry['external']['retrieval_required'])


class StorageTypeTest(TestCase):

    def test_storage_resolves_unique_extension(self):
        models = [
            make_bindata_model(2, BinData.StorageType.STORAGE,
                               storage_id=7),
        ]
        storage = MemoryStorage(**{'BIN0007.bin': MemoryStream(b'x')})
        inventory = build_inventory(
            models, bindata_storage=storage, base_dir='/tmp',
        )
        entry = inventory['attachments'][0]
        self.assertEqual('storage', entry['storage_type'])
        self.assertEqual('BinData/BIN0007.bin', entry['stream'])
        self.assertEqual('available', entry['status'])

    def test_storage_ambiguous_extension_is_missing(self):
        models = [
            make_bindata_model(2, BinData.StorageType.STORAGE,
                               storage_id=7),
        ]
        storage = MemoryStorage(**{
            'BIN0007.jpg': MemoryStream(b'a'),
            'BIN0007.png': MemoryStream(b'b'),
        })
        inventory = build_inventory(
            models, bindata_storage=storage, base_dir='/tmp',
        )
        entry = inventory['attachments'][0]
        self.assertEqual('missing', entry['status'])
        self.assertIsNone(entry['stream'])
