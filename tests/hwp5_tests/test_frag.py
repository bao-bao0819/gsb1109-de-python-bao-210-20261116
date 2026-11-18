# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from unittest import TestCase
import json
import os
import os.path
import shutil
import tempfile

from hwp5.binmodel import Hwp5File
from hwp5.frag import FragmentError
from hwp5.frag import extract_fragment
from hwp5.frag import validate_fragment
from hwp5.recordstream import Hwp5File as RecordHwp5File
from hwp5.recordstream import Record
from hwp5.recordstream import dump_record
from hwp5.recordstream import read_records
from hwp5.storage import unpack
from hwp5.storage.fs import FileSystemStorage
from hwp5.tagids import HWPTAG_BEGIN
from hwp5.tagids import HWPTAG_TABLE

from .fixtures import get_fixture_path


class TestBase(TestCase):

    def setUp(self):
        self.workdir = tempfile.mkdtemp(prefix='hwp5frag-')
        self.addCleanup(shutil.rmtree, self.workdir)

    def fixture(self, filename):
        return get_fixture_path(filename)

    def outdir(self, name):
        return os.path.join(self.workdir, name)

    def read_manifest(self, outdir):
        with open(os.path.join(outdir, 'manifest.json')) as f:
            return json.load(f)

    def paragraph_texts(self, hwpfile, indexes=None):
        section = hwpfile.bodytext.section(0)
        texts = []
        for idx, group in enumerate(section.models_treegrouped()):
            if indexes is not None and idx not in indexes:
                continue
            text = ''
            for model in group:
                if model['tagname'] == 'HWPTAG_PARA_TEXT':
                    chunks = model['content']['chunks']
                    text += ''.join(chunk[1] for chunk in chunks
                                    if isinstance(chunk[1], str))
            texts.append(text)
        return texts


class TestExtractTable(TestBase):

    def test_extract_paragraphs_with_table(self):
        source = self.fixture('table.hwp')
        outdir = self.outdir('table.frag')
        manifest = extract_fragment(source, outdir, [0, 1])

        # the package is self-sufficient
        self.assertTrue(manifest['complete'])
        self.assertEqual([], manifest['incomplete_reasons'])
        self.assertEqual([], validate_fragment(outdir))

        # the table of the selected paragraph is included
        frag = Hwp5File(FileSystemStorage(outdir))
        tagids = [model['tagid']
                  for group in frag.bodytext.section(0).models_treegrouped()
                  for model in group]
        self.assertIn(HWPTAG_TABLE, tagids)

        # paragraph texts are preserved
        original = Hwp5File(source)
        self.assertEqual(self.paragraph_texts(original, {0, 1}),
                         self.paragraph_texts(frag))

        # manifest.json is stored in the package and matches the return value
        stored = self.read_manifest(outdir)
        self.assertEqual(manifest, stored)
        self.assertEqual([0, 1], stored['paragraphs'])

    def test_docinfo_contains_only_needed_resources(self):
        source = self.fixture('table.hwp')
        outdir = self.outdir('table.frag')
        manifest = extract_fragment(source, outdir, [0, 1])

        frag = Hwp5File(FileSystemStorage(outdir))
        idmappings = None
        counts = {}
        for model in frag.docinfo.models():
            if model['tagname'] == 'HWPTAG_ID_MAPPINGS':
                idmappings = model['content']
            tagname = model['tagname']
            counts[tagname] = counts.get(tagname, 0) + 1

        resources = manifest['resources']
        self.assertEqual(idmappings['charshapes'],
                         len(resources['charshape']))
        self.assertEqual(idmappings['parashapes'],
                         len(resources['parashape']))
        self.assertEqual(idmappings['styles'],
                         len(resources['style']))
        self.assertEqual(idmappings['borderfills'],
                         len(resources['borderfill']))
        self.assertEqual(counts.get('HWPTAG_CHAR_SHAPE', 0),
                         len(resources['charshape']))
        # unreferenced resources are not collected
        original = Hwp5File(source)
        n_original_charshapes = len([
            model for model in original.docinfo.models()
            if model['tagname'] == 'HWPTAG_CHAR_SHAPE'])
        self.assertTrue(len(resources['charshape']) < n_original_charshapes)


class TestExtractBinData(TestBase):

    def test_shared_bindata_collected_once(self):
        source = self.fixture('sample-5017-pics.hwp')
        outdir = self.outdir('pics.frag')
        manifest = extract_fragment(source, outdir, [0, 1, 2])

        self.assertTrue(manifest['complete'])
        self.assertEqual([], validate_fragment(outdir))

        # the three selected paragraphs share a single embedded picture:
        # it is collected only once
        bindata_dir = os.path.join(outdir, 'BinData')
        names = os.listdir(bindata_dir)
        self.assertEqual(['BIN0001.jpg'], names)

        # attachments of unselected paragraphs are not included
        self.assertNotIn('BIN000B.jpg', names)
        self.assertNotIn('BIN000C.jpg', names)

        # the stream content is preserved
        original = Hwp5File(source)
        original_bytes = original['BinData']['BIN000B.jpg'].open().read()
        with open(os.path.join(bindata_dir, 'BIN0001.jpg'), 'rb') as f:
            self.assertEqual(original_bytes, f.read())

        # the manifest keeps the correspondence to the original resource
        (bindata,) = manifest['resources']['bindata']
        self.assertEqual(2, bindata['original'])
        self.assertEqual(1, bindata['fragment'])
        self.assertEqual({'original': 11, 'fragment': 1},
                         bindata['storage_id'])
        self.assertEqual('BinData/BIN000B.jpg',
                         bindata['original_stream'])

        # picture references are remapped to the new bindata id
        frag = Hwp5File(FileSystemStorage(outdir))
        for group in frag.bodytext.section(0).models_treegrouped():
            for model in group:
                if model['tagname'] == 'HWPTAG_SHAPE_COMPONENT_PICTURE':
                    bindata_id = model['content']['picture']['bindata_id']
                    self.assertEqual(1, bindata_id)

    def test_no_bindata_without_picture_paragraphs(self):
        source = self.fixture('sample-5017-pics.hwp')
        outdir = self.outdir('text.frag')
        manifest = extract_fragment(source, outdir, [2, 4, 6])
        self.assertTrue(manifest['complete'])
        self.assertNotIn('bindata', manifest['resources'])
        self.assertFalse(os.path.exists(os.path.join(outdir, 'BinData')))
        self.assertEqual([], validate_fragment(outdir))


class TestExtractFonts(TestBase):

    def test_fonts_collected_per_language(self):
        source = self.fixture('sample-5017-pics.hwp')
        outdir = self.outdir('pics.frag')
        extract_fragment(source, outdir, [0, 1, 2])

        original_names = [
            model['content']['name']
            for model in Hwp5File(source).docinfo.models()
            if model['tagname'] == 'HWPTAG_FACE_NAME'
        ]
        frag = Hwp5File(FileSystemStorage(outdir))
        frag_names = [
            model['content']['name']
            for model in frag.docinfo.models()
            if model['tagname'] == 'HWPTAG_FACE_NAME'
        ]
        # one font per language, the one actually used (original id 1
        # in each per-language list)
        self.assertEqual(7, len(frag_names))
        for i, name in enumerate(frag_names):
            self.assertEqual(original_names[i * 2 + 1], name)


class TestExtractNumbering(TestBase):

    def test_bullet_dependency(self):
        source = self.fixture('lists-bullet.hwp')
        outdir = self.outdir('lists-bullet.frag')
        manifest = extract_fragment(source, outdir, [0, 1, 2])
        self.assertTrue(manifest['complete'])
        self.assertEqual([], validate_fragment(outdir))
        self.assertIn('bullet', manifest['resources'])

    def test_numbering_dependency(self):
        source = self.fixture('lists.hwp')
        outdir = self.outdir('lists.frag')
        manifest = extract_fragment(source, outdir, [0, 1, 2])
        self.assertTrue(manifest['complete'])
        self.assertEqual([], validate_fragment(outdir))


class TestOleBinStorage(TestBase):

    def make_source_with_ole_object(self):
        '''Build a directory-based source where the picture of paragraph 1
        is replaced by an OLE object referencing BinData storage id 11.'''
        import struct
        from hwp5.tagids import HWPTAG_SHAPE_COMPONENT_OLE
        from hwp5.tagids import HWPTAG_SHAPE_COMPONENT_PICTURE
        srcdir = os.path.join(self.workdir, 'src')
        os.makedirs(srcdir)
        hwp5file = RecordHwp5File(self.fixture('sample-5017-pics.hwp'))
        unpack(hwp5file, srcdir)
        section_path = os.path.join(srcdir, 'BodyText', 'Section0')
        with open(section_path, 'rb') as f:
            records = list(read_records(f))
        # ShapeOLE payload: flags, extent_x, extent_y, storage_id, border
        payload = struct.pack('<IiiH', 0, 100, 100, 11) + (b'\x00' * 12)
        ole_record = Record(HWPTAG_SHAPE_COMPONENT_OLE, 2, payload)
        replaced = False
        for idx, record in enumerate(records):
            if record['tagid'] == HWPTAG_SHAPE_COMPONENT_PICTURE:
                records[idx] = ole_record
                replaced = True
                break
        self.assertTrue(replaced)
        with open(section_path, 'wb') as f:
            for record in records:
                dump_record(f, record)
        return srcdir

    def test_ole_storage_id_dependency(self):
        srcdir = self.make_source_with_ole_object()
        outdir = self.outdir('ole.frag')
        manifest = extract_fragment(srcdir, outdir, [0, 1, 2])
        self.assertTrue(manifest['complete'])
        self.assertEqual([], validate_fragment(outdir))

        # the BinData record owning storage id 11 is collected
        (bindata,) = manifest['resources']['bindata']
        self.assertEqual({'original': 11, 'fragment': 1},
                         bindata['storage_id'])
        self.assertEqual(['BIN0001.jpg'],
                         os.listdir(os.path.join(outdir, 'BinData')))

        # the OLE object reference is remapped to the new storage id
        frag = Hwp5File(FileSystemStorage(outdir))
        from hwp5.tagids import HWPTAG_SHAPE_COMPONENT_OLE
        found = False
        for group in frag.bodytext.section(0).models_treegrouped():
            for model in group:
                if model['tagid'] == HWPTAG_SHAPE_COMPONENT_OLE:
                    self.assertEqual(1, model['content']['storage_id'])
                    found = True
        self.assertTrue(found)


class TestIncomplete(TestBase):
    def make_source_with_unknown_record(self):
        '''Build a directory-based source with an unknown record injected
        into the first paragraph.'''
        srcdir = os.path.join(self.workdir, 'src')
        os.makedirs(srcdir)
        hwp5file = RecordHwp5File(self.fixture('table.hwp'))
        unpack(hwp5file, srcdir)
        section_path = os.path.join(srcdir, 'BodyText', 'Section0')
        with open(section_path, 'rb') as f:
            records = list(read_records(f))
        unknown_tagid = HWPTAG_BEGIN + 90
        fake = Record(unknown_tagid, 1, b'\x01\x02\x03\x04')
        records.insert(4, fake)
        with open(section_path, 'wb') as f:
            for record in records:
                dump_record(f, record)
        return srcdir

    def test_unknown_record_marks_incomplete(self):
        srcdir = self.make_source_with_unknown_record()
        outdir = self.outdir('incomplete.frag')
        manifest = extract_fragment(srcdir, outdir, [0])

        # readable text does not make the package self-sufficient:
        # an unknown record may reference uncollected resources
        self.assertFalse(manifest['complete'])
        self.assertEqual(1, len(manifest['incomplete_reasons']))
        self.assertIn('unknown record', manifest['incomplete_reasons'][0])

        # the package is still interpretable
        self.assertEqual([], validate_fragment(outdir))

    def test_unknown_control_marks_incomplete(self):
        srcdir = os.path.join(self.workdir, 'src')
        os.makedirs(srcdir)
        hwp5file = RecordHwp5File(self.fixture('table.hwp'))
        unpack(hwp5file, srcdir)
        section_path = os.path.join(srcdir, 'BodyText', 'Section0')
        with open(section_path, 'rb') as f:
            records = list(read_records(f))
        # an unknown control header ('abcd' chid, stored little-endian)
        from hwp5.tagids import HWPTAG_CTRL_HEADER
        fake = Record(HWPTAG_CTRL_HEADER, 1, b'dcba')
        records.insert(4, fake)
        with open(section_path, 'wb') as f:
            for record in records:
                dump_record(f, record)
        outdir = self.outdir('incomplete-ctrl.frag')
        manifest = extract_fragment(srcdir, outdir, [0])
        self.assertFalse(manifest['complete'])
        self.assertIn('unknown control',
                      manifest['incomplete_reasons'][0])


class TestErrors(TestBase):

    def test_outdir_must_not_exist(self):
        source = self.fixture('table.hwp')
        outdir = self.outdir('existing.frag')
        os.makedirs(outdir)
        with self.assertRaises(FragmentError):
            extract_fragment(source, outdir, [0, 1])

    def test_paragraph_index_out_of_range(self):
        source = self.fixture('table.hwp')
        outdir = self.outdir('outofrange.frag')
        with self.assertRaises(FragmentError):
            extract_fragment(source, outdir, [0, 1, 100])
