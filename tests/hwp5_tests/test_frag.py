# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from unittest import TestCase
import json
import os
import shutil
import struct
import tempfile

from hwp5 import frag
from hwp5.recordstream import encode_record_header
from hwp5.tagids import HWPTAG_BIN_DATA
from hwp5.tagids import HWPTAG_CTRL_HEADER
from hwp5.tagids import HWPTAG_MEMO_LIST
from hwp5.tagids import HWPTAG_PARA_CHAR_SHAPE
from hwp5.tagids import HWPTAG_PARA_HEADER
from hwp5.tagids import HWPTAG_PARA_LINE_SEG
from hwp5.tagids import HWPTAG_PARA_TEXT
from hwp5.tagids import HWPTAG_SHAPE_COMPONENT_PICTURE


def _rec(tagid, payload, level=0):
    return encode_record_header(dict(tagid=tagid, level=level,
                                     payload=payload)) + payload


def _ctrl(chid, command='', field_id=0, flags=0):
    payload = bytearray()
    payload += bytes(bytearray((ord(chid[3]), ord(chid[2]),
                                ord(chid[1]), ord(chid[0]))))
    payload += struct.pack('<I', flags)
    payload += b'\x00'
    encoded = command.encode('utf-16le')
    payload += struct.pack('<H', len(command))
    payload += encoded
    payload += struct.pack('<I', field_id)
    payload += b'\x00\x00\x00\x00'
    return bytes(payload)


def _chunk(code, chid=None, param_id=None):
    payload = bytearray(struct.pack('<H', code))
    if chid is not None:
        payload += bytes(bytearray((ord(chid[3]), ord(chid[2]),
                                    ord(chid[1]), ord(chid[0]))))
        if param_id is not None:
            payload += struct.pack('<II', param_id, 0)
        else:
            payload += b'\x00' * 10
    while len(payload) < 16:
        payload += b'\x00'
    return bytes(payload)


def _para_text(chunks):
    payload = bytearray()
    for item in chunks:
        if isinstance(item, str):
            payload += item.encode('utf-16le')
        elif isinstance(item, int):
            payload += _chunk(item)
        elif isinstance(item, dict):
            payload += _chunk(item['code'], item.get('chid'),
                              item.get('param_id'))
        else:
            payload += _chunk(*item)
    return bytes(payload)


def _para_header(text_units, charshapes=1, linesegs=1):
    value = struct.pack('<I', text_units & 0x7fffffff)
    value += struct.pack('<I', 0)
    value += struct.pack('<H', 0)
    value += b'\x00'
    value += b'\x00'
    value += struct.pack('<H', charshapes)
    value += struct.pack('<H', 0)
    value += struct.pack('<H', linesegs)
    value += struct.pack('<I', 0)
    return value


def _char_shape(pairs):
    return b''.join(struct.pack('<II', pos, shape)
                    for pos, shape in pairs)


def _line_seg(chposisions):
    return b''.join(struct.pack('<iiiiiiiiHH', pos, 0, 0, 0, 0, 0, 0, 0,
                                0, 0)
                    for pos in chposisions)


def _memo_body(body_id, text):
    payload = struct.pack('<I', body_id) + b'\x00' * 12
    return _rec(HWPTAG_MEMO_LIST, payload, level=1) + \
        _rec(HWPTAG_PARA_HEADER, _para_header(len(text) + 1), level=2) + \
        _rec(HWPTAG_PARA_TEXT, _para_text([text, (13,)]), level=3)


def make_fragment(path):
    section = bytearray()
    section += _rec(HWPTAG_PARA_HEADER, _para_header(45))
    section += _rec(
        HWPTAG_PARA_TEXT,
        _para_text([
            'before ',
            (3, '%%me'),
            'annotated text',
            (4, '\x00%me'),
            ' after',
            (13,),
        ]),
        level=1)
    section += _rec(HWPTAG_PARA_CHAR_SHAPE,
                    _char_shape([(0, 0), (7, 1), (21, 0), (28, 0)]),
                    level=1)
    section += _rec(HWPTAG_PARA_LINE_SEG, _line_seg([0]), level=1)
    section += _rec(HWPTAG_CTRL_HEADER,
                    _ctrl('%%me', 'memo', 42), level=1)
    section += _memo_body(42, 'the memo body')

    manifest = {
        'format': frag.FRAGMENT_FORMAT,
        'complete': True,
        'incomplete_reasons': [],
        'paragraphs': [0],
        'resources': {'bindata': []},
        'section': 0,
        'source': {'file': 'synthetic.hwp', 'version': [5, 0, 1, 7]},
        'warnings': [],
    }
    if not os.path.exists(path):
        os.makedirs(os.path.join(path, 'BodyText'))
    with open(os.path.join(path, 'FileHeader'), 'wb') as f:
        f.write(b'HWP Document File' + b'\x00' * 240)
    with open(os.path.join(path, 'DocInfo'), 'wb') as f:
        pass
    with open(os.path.join(path, 'BodyText', 'Section0'), 'wb') as f:
        f.write(bytes(section))
    with open(os.path.join(path, 'manifest.json'), 'w') as f:
        json.dump(manifest, f)


class FragTestBase(TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.archive = os.path.join(self.tmpdir, 'fragment')
        make_fragment(self.archive)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)


class TestInventory(FragTestBase):

    def test_inventory_lists_annotation_text_and_markers(self):
        inv = frag.Inventory(frag.open_fragment(self.archive))
        self.assertEqual(1, len(inv.annotations))
        annotation = inv.annotations[0]
        self.assertEqual('annotated text', annotation.text)
        self.assertEqual('%%me', annotation.start.chid)
        self.assertEqual('\x00%me', annotation.end.chid)
        self.assertEqual(42, annotation.start.field_id)
        self.assertEqual(42, annotation.body_id)
        self.assertEqual('the memo body', annotation.body.text)

        start = annotation.start.to_dict(0)
        end = annotation.end.to_dict(0)
        # Extended controls occupy eight UTF-16 code units; inline end
        # markers occupy eight as well (code + CHID + parameter).
        self.assertEqual(8, start['code_units'][1] -
                            start['code_units'][0])
        self.assertEqual(8, end['code_units'][1] -
                            end['code_units'][0])
        self.assertEqual('\x00%me', end['chid'])
        self.assertEqual([], annotation.resources)

    def test_inventory_json_is_serialisable(self):
        inv = frag.Inventory(frag.open_fragment(self.archive))
        data = inv.to_dict()
        json.dumps(data)
        self.assertEqual(1, data['counts']['annotations'])
        self.assertEqual(1, data['counts']['deletable'])


class TestDeletion(FragTestBase):

    def test_deletes_only_markers_and_memo_body(self):
        out = os.path.join(self.tmpdir, 'out')
        report = frag.delete_annotations(self.archive, out, select_all=True)
        self.assertEqual(['ann-0001'], report['deleted'])

        before = open(os.path.join(self.archive, 'BodyText', 'Section0'),
                      'rb').read()
        after = open(os.path.join(out, 'BodyText', 'Section0'), 'rb').read()
        after_text = after.decode('utf-16le', 'ignore')
        self.assertIn('annotated text', after_text)
        self.assertIn('before ', after_text)
        self.assertIn(' after', after_text)
        self.assertNotIn(b'%%me', after)
        self.assertNotIn(b'\x00%me', after)
        self.assertLess(len(after), len(before))

    def test_manifest_records_deletion(self):
        out = os.path.join(self.tmpdir, 'out')
        frag.delete_annotations(self.archive, out, select_all=True)
        with open(os.path.join(out, 'manifest.json')) as f:
            manifest = json.load(f)
        self.assertEqual(['ann-0001'], manifest['deleted_annotations'])
        self.assertEqual(1, len(manifest['annotations']))

    def test_dry_run_writes_nothing(self):
        out = os.path.join(self.tmpdir, 'out')
        report = frag.delete_annotations(self.archive, out, select_all=True,
                                         dry_run=True)
        self.assertEqual(['ann-0001'], report['deleted'])
        self.assertFalse(os.path.exists(out))

    def test_body_text_survives_marker_removal(self):
        out = os.path.join(self.tmpdir, 'out')
        frag.delete_annotations(self.archive, out, select_all=True)
        inv = frag.Inventory(frag.open_fragment(out))
        self.assertEqual([], inv.annotations)


class TestRefusal(FragTestBase):

    def _rewrite_section(self, transform):
        path = os.path.join(self.archive, 'BodyText', 'Section0')
        with open(path, 'rb') as f:
            data = f.read()
        data = transform(data)
        with open(path, 'wb') as f:
            f.write(data)

    def test_missing_end_is_refused(self):
        def drop_end(data):
            out = bytearray()
            for record in frag.iter_records(data):
                if record.tagid == HWPTAG_PARA_TEXT:
                    chunks = list(frag.ParaTextChunks.parse_chunks(
                        record.payload))
                    payload = _para_text([
                        item for (bounds, item) in chunks
                        if not (isinstance(item, dict) and
                                item.get('code') == 4)
                    ])
                    out += encode_record_header(dict(
                        tagid=record.tagid, level=record.level,
                        payload=payload)) + payload
                else:
                    out += record.raw(data)
            return bytes(out)

        self._rewrite_section(drop_end)
        inv = frag.Inventory(frag.open_fragment(self.archive))
        self.assertTrue(inv.refusals)
        self.assertEqual(0, len([a for a in inv.annotations if a.deletable]))

    def test_start_without_end_is_refused(self):
        def make_unpaired(data):
            out = bytearray()
            for record in frag.iter_records(data):
                if record.tagid == HWPTAG_PARA_TEXT:
                    payload = _para_text([
                        'a',
                        (3, '%%me'),
                        'no end',
                        (13,),
                    ])
                    out += encode_record_header(dict(
                        tagid=record.tagid, level=record.level,
                        payload=payload)) + payload
                else:
                    out += record.raw(data)
            return bytes(out)

        self._rewrite_section(make_unpaired)
        inv = frag.Inventory(frag.open_fragment(self.archive))
        self.assertEqual(0, len(inv.annotations))
        self.assertTrue(inv.refusals)
        report = frag.plan_annotation_deletion(inv, select_all=True)
        self.assertEqual([], report.deletions)

    def test_shared_body_is_not_deleted(self):
        # Two annotations that point at the same memo body ID.
        def duplicate(data):
            out = bytearray()
            for record in frag.iter_records(data):
                if record.tagid == HWPTAG_PARA_TEXT:
                    payload = _para_text([
                        'x',
                        (3, '%%me'),
                        'one',
                        (4, '\x00%me'),
                        (3, '%%me'),
                        'two',
                        (4, '\x00%me'),
                        (13,),
                    ])
                    out += encode_record_header(dict(
                        tagid=record.tagid, level=record.level,
                        payload=payload)) + payload
                    out += _rec(HWPTAG_CTRL_HEADER,
                                _ctrl('%%me', '', 42), level=1)
                    out += _rec(HWPTAG_CTRL_HEADER,
                                _ctrl('%%me', '', 42), level=1)
                elif record.tagid == HWPTAG_CTRL_HEADER:
                    continue
                else:
                    out += record.raw(data)
            return bytes(out)

        self._rewrite_section(duplicate)
        inv = frag.Inventory(frag.open_fragment(self.archive))
        self.assertEqual(2, len(inv.annotations))
        self.assertEqual(inv.annotations[0].body_id,
                         inv.annotations[1].body_id)
        # Both annotations are individually deletable.  Deleting only one
        # must leave the shared memo body in place for the other anchor.
        self.assertTrue(all(a.deletable for a in inv.annotations))
        out = os.path.join(self.tmpdir, 'out')
        frag.delete_annotations(self.archive, out, selection=['ann-0001'])
        after = frag.Inventory(frag.open_fragment(out))
        self.assertEqual(1, len(after.annotations))
        self.assertEqual(1, len(after.bodies))

        # Deleting both releases the body, so it may be removed.
        out2 = os.path.join(self.tmpdir, 'out2')
        frag.delete_annotations(self.archive, out2, select_all=True)
        after2 = frag.Inventory(frag.open_fragment(out2))
        self.assertEqual([], after2.annotations)
        self.assertEqual(0, len(after2.bodies))

    def test_unpaired_marker_inside_range_is_refused(self):
        # Start A, start B, end A: B is unpaired and sits inside A.  A's
        # end could equally belong to B, so A is refused too.
        section = bytearray()
        section += _rec(HWPTAG_PARA_HEADER, _para_header(40))
        section += _rec(
            HWPTAG_PARA_TEXT,
            _para_text([
                'a',
                {'code': 3, 'chid': '%%me', 'param_id': 1},
                'b',
                {'code': 3, 'chid': '%%me', 'param_id': 2},
                'c',
                {'code': 4, 'chid': '\x00%me', 'param_id': 1},
                (13,),
            ]),
            level=1)
        section += _rec(HWPTAG_CTRL_HEADER, _ctrl('%%me', '', 1), level=1)
        section += _rec(HWPTAG_CTRL_HEADER, _ctrl('%%me', '', 2), level=1)
        section += _memo_body(1, 'body one')
        section += _memo_body(2, 'body two')
        path = os.path.join(self.archive, 'BodyText', 'Section0')
        with open(path, 'wb') as f:
            f.write(bytes(section))

        inv = frag.Inventory(frag.open_fragment(self.archive))
        self.assertEqual(1, len(inv.annotations))
        self.assertFalse(inv.annotations[0].deletable)
        self.assertTrue(any('falls inside' in reason
                            for reason in inv.annotations[0].reasons))

    def test_nested_annotations_are_allowed(self):
        # start A, start B, end B, end A (LIFO nesting) is one reading.
        section = bytearray()
        section += _rec(HWPTAG_PARA_HEADER, _para_header(40))
        section += _rec(
            HWPTAG_PARA_TEXT,
            _para_text([
                'a',
                {'code': 3, 'chid': '%%me', 'param_id': 11},
                'b',
                {'code': 3, 'chid': '%%me', 'param_id': 12},
                'c',
                {'code': 4, 'chid': '\x00%me', 'param_id': 12},
                'd',
                {'code': 4, 'chid': '\x00%me', 'param_id': 11},
                (13,),
            ]),
            level=1)
        section += _rec(HWPTAG_CTRL_HEADER, _ctrl('%%me', '', 11), level=1)
        section += _rec(HWPTAG_CTRL_HEADER, _ctrl('%%me', '', 12), level=1)
        section += _memo_body(11, 'body eleven')
        section += _memo_body(12, 'body twelve')
        path = os.path.join(self.archive, 'BodyText', 'Section0')
        with open(path, 'wb') as f:
            f.write(bytes(section))

        inv = frag.Inventory(frag.open_fragment(self.archive))
        self.assertEqual(2, len(inv.annotations))
        self.assertEqual(['bcd', 'c'], sorted(a.text
                                              for a in inv.annotations))
        self.assertTrue(all(a.deletable for a in inv.annotations))

    def test_crossing_annotations_are_refused(self):
        # start A, start B, end A, end B: two distinct ids cross.
        section = bytearray()
        section += _rec(HWPTAG_PARA_HEADER, _para_header(40))
        section += _rec(
            HWPTAG_PARA_TEXT,
            _para_text([
                'a',
                {'code': 3, 'chid': '%%me', 'param_id': 11},
                'b',
                {'code': 3, 'chid': '%%me', 'param_id': 12},
                'c',
                {'code': 4, 'chid': '\x00%me', 'param_id': 11},
                'd',
                {'code': 4, 'chid': '\x00%me', 'param_id': 12},
                (13,),
            ]),
            level=1)
        section += _rec(HWPTAG_CTRL_HEADER, _ctrl('%%me', '', 11), level=1)
        section += _rec(HWPTAG_CTRL_HEADER, _ctrl('%%me', '', 12), level=1)
        section += _memo_body(11, 'body eleven')
        section += _memo_body(12, 'body twelve')
        path = os.path.join(self.archive, 'BodyText', 'Section0')
        with open(path, 'wb') as f:
            f.write(bytes(section))

        inv = frag.Inventory(frag.open_fragment(self.archive))
        self.assertEqual(2, len(inv.annotations))
        self.assertFalse(any(a.deletable for a in inv.annotations))
        report = frag.plan_annotation_deletion(inv, select_all=True)
        self.assertEqual([], report.deletions)

    def test_cross_paragraph_annotation_text(self):
        section = bytearray()
        section += _rec(HWPTAG_PARA_HEADER, _para_header(20))
        section += _rec(
            HWPTAG_PARA_TEXT,
            _para_text(['first ', (3, '%%me'), 'part', (13,)]),
            level=1)
        section += _rec(HWPTAG_CTRL_HEADER, _ctrl('%%me', '', 7), level=1)
        section += _rec(HWPTAG_PARA_HEADER, _para_header(12))
        section += _rec(
            HWPTAG_PARA_TEXT,
            _para_text(['second', (4, '\x00%me'), ' tail', (13,)]),
            level=1)
        section += _memo_body(7, 'cross paragraph body')
        path = os.path.join(self.archive, 'BodyText', 'Section0')
        with open(path, 'wb') as f:
            f.write(bytes(section))

        inv = frag.Inventory(frag.open_fragment(self.archive))
        self.assertEqual(1, len(inv.annotations))
        self.assertEqual('part\nsecond', inv.annotations[0].text)


class TestResourceImpact(FragTestBase):

    def _add_resource(self, referenced_by_memo=False):
        docinfo = bytearray()
        bindata = struct.pack('<H', 1) + struct.pack('<H', 1) + \
            struct.pack('<H', 3) + b'jpg'
        docinfo += _rec(HWPTAG_BIN_DATA, bindata)
        with open(os.path.join(self.archive, 'DocInfo'), 'wb') as f:
            f.write(bytes(docinfo))

        picture = bytearray(73)
        struct.pack_into('<H', picture, 71, 1)
        section_path = os.path.join(self.archive, 'BodyText', 'Section0')
        with open(section_path, 'ab') as f:
            f.write(_rec(HWPTAG_SHAPE_COMPONENT_PICTURE, bytes(picture),
                         level=1))
        os.makedirs(os.path.join(self.archive, 'BinData'))
        with open(os.path.join(self.archive, 'BinData', 'BIN0001.jpg'),
                  'wb') as f:
            f.write(b'JPEGDATA')

    def test_body_resource_survives_note_deletion(self):
        self._add_resource()
        inv = frag.Inventory(frag.open_fragment(self.archive))
        # The picture is ordinary body content, not part of the note body.
        self.assertIn(1, inv.external_resource_refs)
        out = os.path.join(self.tmpdir, 'out')
        report = frag.delete_annotations(self.archive, out, select_all=True)
        self.assertEqual([], report['plan']['resources']['removed'])
        self.assertTrue(os.path.exists(
            os.path.join(out, 'BinData', 'BIN0001.jpg')))

    def test_resource_removed_only_when_explicitly_requested(self):
        self._add_resource()
        # Register the picture inside the memo body span instead of the
        # main section, so it is referenced only by the note.
        path = os.path.join(self.archive, 'BodyText', 'Section0')
        with open(path, 'rb') as f:
            data = f.read()
        records = list(frag.iter_records(data))
        picture = [r for r in records
                   if r.tagid == HWPTAG_SHAPE_COMPONENT_PICTURE][0]
        without = data[:picture.header_start] + data[picture.end:]
        # Re-append the picture at the end of the memo body (level 4).
        payload = bytes(picture.payload)
        memo_picture = encode_record_header(dict(
            tagid=HWPTAG_SHAPE_COMPONENT_PICTURE, level=4,
            payload=payload)) + payload
        with open(path, 'wb') as f:
            f.write(without + memo_picture)

        inv = frag.Inventory(frag.open_fragment(self.archive))
        self.assertNotIn(1, inv.external_resource_refs)
        report = frag.delete_annotations(
            self.archive, os.path.join(self.tmpdir, 'kept'),
            select_all=True)
        # Reported as unreferenced, but the stream is kept by default.
        self.assertEqual([1], report['plan']['resources']['removed'])
        self.assertTrue(os.path.exists(
            os.path.join(self.tmpdir, 'kept', 'BinData', 'BIN0001.jpg')))

        out = os.path.join(self.tmpdir, 'pruned')
        frag.delete_annotations(
            self.archive, out, select_all=True,
            delete_unreferenced_resources=True)
        self.assertFalse(os.path.exists(
            os.path.join(out, 'BinData', 'BIN0001.jpg')))


if __name__ == '__main__':
    import unittest
    unittest.main()
