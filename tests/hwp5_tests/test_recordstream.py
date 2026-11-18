# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from io import BytesIO
import json
from unittest import TestCase

from hwp5 import recordstream as RS
from hwp5.recordstream import Record
from hwp5.recordstream import RecordStream
from hwp5.recordstream import dump_record
from hwp5.recordstream import link_records
from hwp5.recordstream import read_record
from hwp5.recordstream import record_to_json
from hwp5.storage import ExtraItemStorage
from hwp5.tagids import HWPTAG_DOCUMENT_PROPERTIES
from hwp5.tagids import HWPTAG_ID_MAPPINGS
from hwp5.tagids import HWPTAG_PARA_HEADER
from hwp5.utils import cached_property

from . import test_filestructure


def make_records(levels):
    records = [Record(tagid=HWPTAG_PARA_HEADER + seqno, level=level,
                      payload=b'', seqno=seqno)
               for seqno, level in enumerate(levels)]
    for record, level in zip(records, levels):
        record['label'] = 'r%d@%d' % (record['seqno'], level)
    return records


def edge_of(record):
    return (
        record['label'],
        (record.get('parent') or {}).get('label'),
        (record.get('sister') or {}).get('label'),
        record.get('parent_missing', False),
    )


class TestLinkRecords(TestCase):

    def test_wellformed_links_keep_proven_edges(self):
        records = make_records([0, 0, 1, 2, 1, 2, 0])
        result = [edge_of(r) for r in link_records(iter(records))]
        self.assertEqual(result, [
            ('r0@0', None, None, False),
            ('r1@0', None, 'r0@0', False),
            ('r2@1', 'r1@0', None, False),
            ('r3@2', 'r2@1', None, False),
            ('r4@1', 'r1@0', None, False),
            ('r5@2', 'r4@1', None, False),
            ('r6@0', None, None, False),
        ])

    def test_level_jump_marks_missing_parent_instead_of_guessing(self):
        records = make_records([1, 3, 1, 2])
        result = [edge_of(r) for r in link_records(iter(records))]
        self.assertEqual(result, [
            ('r0@1', None, None, False),
            ('r1@3', None, None, True),   # no fabricated parent
            ('r2@1', None, None, False),  # sibling stays at top, not adopted
            ('r3@2', 'r2@1', None, False),
        ])

    def test_orphan_siblings_share_missing_parent_not_each_other(self):
        records = make_records([0, 2, 2, 1])
        result = [edge_of(r) for r in link_records(iter(records))]
        self.assertEqual(result, [
            ('r0@0', None, None, False),
            ('r1@2', None, None, True),
            ('r2@2', None, 'r1@2', True),
            ('r3@1', 'r0@0', None, False),
        ])

    def test_nonzero_baselevel_without_gap_is_unchanged(self):
        records = make_records([7, 8, 9, 7])
        result = [edge_of(r) for r in link_records(iter(records))]
        self.assertEqual(result, [
            ('r0@7', None, None, False),
            ('r1@8', 'r0@7', None, False),
            ('r2@9', 'r1@8', None, False),
            ('r3@7', None, None, False),
        ])

    def test_empty_and_below_baselevel_do_not_crash(self):
        self.assertEqual([], list(link_records(iter([]))))

        records = make_records([1, 0])
        result = [edge_of(r) for r in link_records(iter(records))]
        self.assertEqual(2, len(result))
        self.assertFalse(any(row[3] for row in result))


class TestBase(test_filestructure.TestBase):

    @property
    def hwp5file_rec(self):
        return RS.Hwp5File(self.olestg)

    hwp5file = hwp5file_rec


class TestRecord(TestBase):

    def test_read_record(self):
        docinfo_stream = self.hwp5file['DocInfo']

        record = read_record(docinfo_stream.open(), 0)
        self.assertEqual(HWPTAG_DOCUMENT_PROPERTIES, record['tagid'])

    def test_dump_record(self):
        docinfo_stream = self.hwp5file['DocInfo']
        record = read_record(docinfo_stream.open(), 0)
        stream = BytesIO()
        dump_record(stream, record)
        stream.seek(0)
        record2 = read_record(stream, 0)
        self.assertEqual(record2, record)


class TestRecordStream(TestBase):

    @cached_property
    def docinfo(self):
        return RecordStream(self.hwp5file_fs['DocInfo'],
                            self.hwp5file_fs.header.version)

    def test_records(self):
        self.assertEqual(67, len(list(self.docinfo.records())))

    def test_records_kwargs_treegroup(self):
        records = self.docinfo.records(treegroup=1)
        self.assertEqual(66, len(records))
        self.assertEqual(HWPTAG_ID_MAPPINGS, records[0]['tagid'])

        records = self.docinfo.records(treegroup=0)
        self.assertEqual(1, len(records))
        self.assertEqual(HWPTAG_DOCUMENT_PROPERTIES, records[0]['tagid'])

        records = self.bodytext.section(0).records(treegroup=5)
        self.assertEqual(26, records[0]['seqno'])
        self.assertEqual(37, len(records))

    def test_record(self):
        record = self.docinfo.record(0)
        self.assertEqual(0, record['seqno'])

        record = self.docinfo.record(10)
        self.assertEqual(10, record['seqno'])

    def test_records_treegrouped(self):
        groups = self.docinfo.records_treegrouped()
        document_properties_treerecords = next(groups)
        self.assertEqual(1, len(document_properties_treerecords))
        idmappings_treerecords = next(groups)
        self.assertEqual(66, len(idmappings_treerecords))

        section = self.bodytext.section(0)
        for group_idx, records in enumerate(section.records_treegrouped()):
            # print group_idx, records[0]['seqno'], len(records)
            self.assertEqual(HWPTAG_PARA_HEADER, records[0]['tagid'])

    def test_records_treegrouped_as_iterable(self):
        groups = self.docinfo.records_treegrouped(group_as_list=False)
        group = next(groups)
        self.assertFalse(isinstance(group, list))

    def test_records_treegroped_as_list(self):
        groups = self.docinfo.records_treegrouped()
        group = next(groups)
        self.assertTrue(isinstance(group, list))

    def test_records_treegroup(self):
        records = self.docinfo.records_treegroup(1)
        self.assertEqual(66, len(records))
        self.assertEqual(HWPTAG_ID_MAPPINGS, records[0]['tagid'])

        records = self.docinfo.records_treegroup(0)
        self.assertEqual(1, len(records))
        self.assertEqual(HWPTAG_DOCUMENT_PROPERTIES, records[0]['tagid'])

        records = self.bodytext.section(0).records_treegroup(5)
        self.assertEqual(26, records[0]['seqno'])
        self.assertEqual(37, len(records))


class TestHwp5File(TestBase):

    def test_if_hwp5file_contains_other_formats(self):
        stg = ExtraItemStorage(self.hwp5file)
        self.assertTrue('DocInfo.records' in list(stg))

    def test_docinfo(self):
        docinfo = self.hwp5file.docinfo
        self.assertTrue(isinstance(docinfo, RS.RecordStream))
        records = list(docinfo.records())
        self.assertEqual(67, len(records))

    def test_bodytext(self):
        bodytext = self.hwp5file.bodytext
        self.assertTrue(isinstance(bodytext, RS.Sections))
        stg = ExtraItemStorage(bodytext)
        self.assertEqual(['Section0', 'Section0.records'], list(stg))


class TestJson(TestBase):
    def test_record_to_json(self):
        record = next(self.hwp5file.docinfo.records())
        json_string = record_to_json(record)
        jsonobject = json.loads(json_string)
        self.assertEqual(16, jsonobject['tagid'])
        self.assertEqual(0, jsonobject['level'])
        self.assertEqual(26, jsonobject['size'])
        self.assertEqual(['01 00 01 00 01 00 01 00 01 00 01 00 01 00 00 00',
                          '00 00 07 00 00 00 05 00 00 00'],
                         jsonobject['payload'])
        self.assertEqual(0, jsonobject['seqno'])
        self.assertEqual('HWPTAG_DOCUMENT_PROPERTIES', jsonobject['tagname'])

    def test_generate_json(self):
        records_json = self.hwp5file.docinfo.records_json()
        json_string = ''.join(records_json.generate())

        jsonobject = json.loads(json_string)
        self.assertEqual(67, len(jsonobject))
