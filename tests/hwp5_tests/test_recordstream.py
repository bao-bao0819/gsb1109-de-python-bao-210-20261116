# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from io import BytesIO
from unittest import TestCase
import json
import os
import zlib

from hwp5 import recordstream as RS
from hwp5.recordstream import RecordStream
from hwp5.recordstream import decompress_records
from hwp5.recordstream import dump_record
from hwp5.recordstream import read_record
from hwp5.recordstream import read_records
from hwp5.recordstream import read_records_with_tail
from hwp5.recordstream import record_to_json
from hwp5.recordstream import split_records
from hwp5.storage import ExtraItemStorage
from hwp5.tagids import HWPTAG_DOCUMENT_PROPERTIES
from hwp5.tagids import HWPTAG_ID_MAPPINGS
from hwp5.tagids import HWPTAG_PARA_HEADER
from hwp5.utils import cached_property

from . import test_filestructure


class TestTruncatedRecords(TestCase):

    @cached_property
    def records(self):
        return [
            RS.Record(HWPTAG_PARA_HEADER, 0, os.urandom(5)),
            RS.Record(HWPTAG_PARA_HEADER + 1, 1, os.urandom(500)),
            RS.Record(HWPTAG_PARA_HEADER + 2, 1, b''),
            # payload big enough to require the extended (8-byte) header
            RS.Record(HWPTAG_PARA_HEADER + 3, 0, os.urandom(5000)),
            RS.Record(HWPTAG_PARA_HEADER + 4, 0, os.urandom(4)),
        ]

    @cached_property
    def records_data(self):
        f = BytesIO()
        for record in self.records:
            dump_record(f, record)
        return f.getvalue()

    @cached_property
    def compressed_data(self):
        c = zlib.compressobj(9, zlib.DEFLATED, -15)
        return c.compress(self.records_data) + c.flush()

    def record_keys(self, records):
        return [(rec['tagid'], rec['level'], rec['size'], rec['payload'],
                 rec['seqno']) for rec in records]

    def offset_of(self, idx):
        f = BytesIO()
        for record in self.records[:idx]:
            dump_record(f, record)
        return len(f.getvalue())

    def test_split_records_of_a_complete_buffer(self):
        records, tail = split_records(self.records_data)
        self.assertEqual(len(self.records), len(records))
        self.assertEqual(b'', tail)
        for rec, expected in zip(records, self.records):
            self.assertEqual(expected['tagid'], rec['tagid'])
            self.assertEqual(expected['level'], rec['level'])
            self.assertEqual(expected['payload'], rec['payload'])
        self.assertEqual(list(range(len(records))),
                         [rec['seqno'] for rec in records])

    def test_split_records_of_a_truncated_payload(self):
        # cut in the middle of the 4th record's payload
        cut = len(self.records_data) - 2500
        records, tail = split_records(self.records_data[:cut])
        self.assertEqual(3, len(records))
        # the partially present record stays in the undetermined tail
        self.assertEqual(self.records_data[self.offset_of(3):cut], tail)
        self.assertTrue(len(tail) > 0)

    def test_split_records_of_a_truncated_header(self):
        # cut 2 bytes into the 4-byte header of the last record
        cut = self.offset_of(4) + 2
        records, tail = split_records(self.records_data[:cut])
        self.assertEqual(4, len(records))
        self.assertEqual(self.records_data[self.offset_of(4):cut], tail)
        self.assertEqual(2, len(tail))

    def test_split_records_of_a_truncated_extended_header(self):
        # cut in the middle of the 8-byte extended header of the 4th record
        cut = self.offset_of(3) + 6
        records, tail = split_records(self.records_data[:cut])
        self.assertEqual(3, len(records))
        self.assertEqual(self.records_data[self.offset_of(3):cut], tail)
        self.assertEqual(6, len(tail))

    def test_read_records_with_tail(self):
        cut = len(self.records_data) - 2500
        records, tail = read_records_with_tail(
            BytesIO(self.records_data[:cut]))
        self.assertEqual(3, len(records))
        self.assertEqual(self.records_data[self.offset_of(3):cut], tail)

    def test_read_records_stops_at_a_truncated_record(self):
        cut = len(self.records_data) - 2500
        records = list(read_records(BytesIO(self.records_data[:cut])))
        self.assertEqual(3, len(records))

    def test_decompress_records_of_a_complete_stream(self):
        for bufsize in (1, 2, 3, 7, 4096, 65536):
            records, tail, status = decompress_records(
                BytesIO(self.compressed_data), bufsize)
            self.assertEqual(len(self.records), len(records))
            self.assertEqual(b'', tail)
            self.assertTrue(status.eof)
            self.assertFalse(status.truncated)

    def test_decompress_records_of_a_truncated_stream(self):
        # truncate the compressed stream: some records are complete, the
        # decompressed half of the next record stays in the tail
        truncated = self.compressed_data[:-100]
        results = []
        for bufsize in (1, 2, 3, 7, 4096, 65536):
            records, tail, status = decompress_records(
                BytesIO(truncated), bufsize)
            results.append((self.record_keys(records), tail, status))

        # decompression status and record completeness are reported
        # separately
        for keys, tail, status in results:
            self.assertTrue(status.input_exhausted)
            self.assertFalse(status.eof)
            self.assertTrue(status.truncated)
            self.assertTrue(len(keys) > 0)
            self.assertTrue(len(tail) > 0)

        # the determined records and their source bytes do not depend on
        # the input chunk size
        for keys, tail, status in results:
            self.assertEqual(results[0][0], keys)
            self.assertEqual(results[0][1], tail)

        # only structurally complete records are accepted
        keys = results[0][0]
        complete, _ = split_records(self.records_data)
        self.assertEqual(self.record_keys(complete)[:len(keys)], keys)

    def test_decompress_records_matches_read_records(self):
        records, tail, status = decompress_records(
            BytesIO(self.compressed_data), 4096)
        streamed = list(read_records(BytesIO(self.records_data)))
        self.assertEqual(self.record_keys(streamed),
                         self.record_keys(records))


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
