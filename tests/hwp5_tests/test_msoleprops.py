# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from io import BytesIO
from unittest import TestCase
from uuid import UUID
import json
import struct

from hwp5.filestructure import HwpSummaryInfo
from hwp5.msoleprops import PIDSI_CREATE_DTM
from hwp5.msoleprops import PIDSI_PAGECOUNT
from hwp5.msoleprops import PIDSI_TITLE
from hwp5.msoleprops import PropertySetFormat
from hwp5.msoleprops import PropertySetStreamReader
from hwp5.msoleprops import RawTextValue
from hwp5.msoleprops import UnreadableValue
from hwp5.summaryinfo import FMTID_HWP_SUMMARY_INFORMATION
from hwp5.summaryinfo import HWP_PROPERTIES
from hwp5.summaryinfo import HwpSummaryInfoTextFormatter
from hwp5.summaryinfo import HWPPIDSI_DATE_STR


HWP_FMTID = UUID('9fa2b660-1061-11d4-b4c6-006097c09d8c')

VT_I4 = 0x0003
VT_LPSTR = 0x001E
VT_LPWSTR = 0x001F
VT_FILETIME = 0x0040


def i4(value):
    return struct.pack('<i', value)


def lpstr(raw):
    return struct.pack('<I', len(raw)) + raw


def lpwstr(text):
    data = (text + '\x00').encode('utf-16le')
    codeunit_count = len(text) + 1
    return struct.pack('<I', codeunit_count) + data


def filetime(value):
    return struct.pack('<Q', value)


def build_property_set_stream(entries, clsid=HWP_FMTID, fmtid=HWP_FMTID):
    ''' Build a minimal [MS-OLEPS] PropertySetStream in memory.

        entries is a list of ``(property_id, type_code, payload)`` triples,
        where payload is the value bytes following the 4-byte type indicator.
    '''
    body = bytearray(struct.pack('<II', 0, len(entries)))
    body.extend(b'\x00' * (8 * len(entries)))
    descriptors = []
    for property_id, type_code, payload in entries:
        body.extend(b'\x00' * ((-len(body)) % 4))
        offset = len(body)
        descriptors.append((property_id, offset))
        body.extend(struct.pack('<I', type_code))
        body.extend(payload)

    struct.pack_into('<I', body, 0, len(body))
    for index, (property_id, offset) in enumerate(descriptors):
        struct.pack_into('<II', body, 8 + index * 8, property_id, offset)

    stream = BytesIO()
    stream.write(struct.pack('<HHI', 0xFFFE, 0, 0x00020105))
    stream.write(clsid.bytes_le)
    stream.write(struct.pack('<I', 1))
    property_set_offset = 48
    stream.write(fmtid.bytes_le)
    stream.write(struct.pack('<I', property_set_offset))
    stream.write(bytes(body))
    stream.seek(0)
    return stream


def read_stream(entries):
    reader = PropertySetStreamReader([
        PropertySetFormat(FMTID_HWP_SUMMARY_INFORMATION, HWP_PROPERTIES)
    ])
    return reader.read(build_property_set_stream(entries))


class PropertySetStreamReaderTest(TestCase):

    def test_lpstr_is_decoded_with_the_set_codepage(self):
        raw = '한글'.encode('cp949') + b'\x00'
        stream = read_stream([
            (1, VT_I4, i4(949)),            # PID_CODEPAGE
            (PIDSI_TITLE.id, VT_LPSTR, lpstr(raw)),
        ])
        property_set = stream.propertysets[0]
        title = property_set[PIDSI_TITLE]
        self.assertEqual('한글', title)

        title_property = [
            p for p in property_set.properties if p.id == PIDSI_TITLE.id
        ][0]
        self.assertEqual(949, title_property.codepage)
        self.assertEqual('VT_LPSTR', title_property.typeLabel)
        self.assertEqual('0x001e', title_property.typeCode)
        self.assertEqual(48, title_property.propsetDesc.offset)
        start, end = title_property.byteRange
        self.assertEqual(start + 4, end - len(raw) - 4)
        self.assertEqual(start, title_property.streamOffset)

    def test_unknown_codepage_keeps_raw_bytes_exports_rest(self):
        stream = read_stream([
            (1, VT_I4, i4(99999)),          # unknown PID_CODEPAGE
            (PIDSI_TITLE.id, VT_LPSTR, lpstr(b'\xff\xfe\x00')),
            (PIDSI_PAGECOUNT.id, VT_I4, i4(7)),
        ])
        property_set = stream.propertysets[0]

        title = property_set[PIDSI_TITLE]
        self.assertIsInstance(title, RawTextValue)
        self.assertEqual('unknown-codepage', title.reason)
        self.assertEqual(99999, title.codepage)
        self.assertEqual(b'\xff\xfe', title.raw)

        # readable properties of other types are still exported
        self.assertEqual(7, property_set[PIDSI_PAGECOUNT])
        self.assertEqual(
            len(property_set.properties), 3,
            'the raw-text property must not be dropped',
        )

    def test_valid_codepage_with_invalid_bytes_is_retained_raw(self):
        # 0x81 is not assigned in Windows-1252
        stream = read_stream([
            (1, VT_I4, i4(1252)),
            (PIDSI_TITLE.id, VT_LPSTR, lpstr(b'\x81\x00')),
        ])
        title = stream.propertysets[0][PIDSI_TITLE]
        self.assertIsInstance(title, RawTextValue)
        self.assertEqual('undecodable', title.reason)
        self.assertEqual(b'\x81', title.raw)

    def test_duplicate_identifiers_are_both_retained(self):
        earlier = 129225153735400000
        later = earlier + 10 * 10 * 1000 * 1000
        stream = read_stream([
            (PIDSI_CREATE_DTM.id, VT_FILETIME, filetime(earlier)),
            (PIDSI_CREATE_DTM.id, VT_FILETIME, filetime(later)),
        ])
        property_set = stream.propertysets[0]
        values = property_set.findall(PIDSI_CREATE_DTM)
        self.assertEqual([earlier, later], [v.value for v in values])

        create_properties = [
            p for p in property_set.properties
            if p.id == PIDSI_CREATE_DTM.id
        ]
        self.assertEqual(2, len(create_properties))
        self.assertNotEqual(
            create_properties[0].byteRange,
            create_properties[1].byteRange,
        )

    def test_unknown_variant_type_does_not_block_other_properties(self):
        stream = read_stream([
            (99, 0x9999, b'\x00\x00\x00\x00'),
            (PIDSI_PAGECOUNT.id, VT_I4, i4(5)),
        ])
        property_set = stream.propertysets[0]
        unknown = [p for p in property_set.properties if p.id == 99][0]
        self.assertIsInstance(unknown.value, UnreadableValue)
        self.assertEqual('VT_UNKNOWN_0x9999', unknown.typeLabel)
        self.assertEqual(5, property_set[PIDSI_PAGECOUNT])


class HwpSummaryInfoExplanationTest(TestCase):

    def _summary_info(self, entries):
        stream = read_stream(entries)
        info = HwpSummaryInfo.__new__(HwpSummaryInfo)
        info.propertySetStream = stream
        info.propertySet = stream.propertysets[0]
        return info

    def test_explanation_records_type_set_codepage_and_byte_range(self):
        raw = '한글'.encode('cp949') + b'\x00'
        info = self._summary_info([
            (1, VT_I4, i4(949)),
            (PIDSI_TITLE.id, VT_LPSTR, lpstr(raw)),
            (PIDSI_CREATE_DTM.id, VT_FILETIME, filetime(129225153735400000)),
            (HWPPIDSI_DATE_STR.id, VT_LPWSTR, lpwstr('2010년 7월 2일')),
        ])
        explanation = info.explanation()

        # the whole explanation has to be JSON serializable
        encoded = json.dumps(explanation, ensure_ascii=False)
        self.assertIn('한글', encoded)

        property_set = explanation['property_sets'][0]
        self.assertEqual(str(HWP_FMTID), property_set['fmtid'])

        title_record = [
            p for p in property_set['properties']
            if p['id'] == PIDSI_TITLE.id
        ][0]
        self.assertEqual('VT_LPSTR', title_record['type'])
        self.assertEqual('0x001e', title_record['type_code'])
        self.assertEqual(949, title_record['codepage'])
        self.assertEqual(str(HWP_FMTID), title_record['property_set'])
        self.assertIsInstance(title_record['byte_range']['start'], int)
        self.assertGreater(
            title_record['byte_range']['end'],
            title_record['byte_range']['start'],
        )
        self.assertEqual('한글', title_record['value'])

        create_record = [
            p for p in property_set['properties']
            if p['id'] == PIDSI_CREATE_DTM.id
        ][0]
        self.assertEqual(129225153735400000, create_record['value'])
        self.assertIn('datetime', create_record)

    def test_raw_text_explanation_preserves_bytes(self):
        info = self._summary_info([
            (1, VT_I4, i4(99999)),
            (PIDSI_TITLE.id, VT_LPSTR, lpstr(b'\xff\xfe\x00')),
        ])
        title_record = [
            p for p in info.explanation()['property_sets'][0]['properties']
            if p['id'] == PIDSI_TITLE.id
        ][0]
        self.assertIsNone(title_record['value'])
        self.assertEqual('fffe', title_record['raw_text']['raw_hex'])
        self.assertEqual(99999, title_record['raw_text']['codepage'])
        json.dumps(title_record)  # must be serializable

    def test_time_sources_keep_conflicting_values_separate(self):
        filetime_value = 129225153735400000
        info = self._summary_info([
            (PIDSI_CREATE_DTM.id, VT_FILETIME, filetime(filetime_value)),
            (PIDSI_CREATE_DTM.id, VT_FILETIME,
             filetime(filetime_value + 50000000)),
            (HWPPIDSI_DATE_STR.id, VT_LPWSTR, lpwstr('2010년 7월 2일')),
        ])
        sources = info.time_sources()
        create_sources = [s for s in sources if s['source'] ==
                          'create_filetime']
        self.assertEqual(2, len(create_sources))
        self.assertNotEqual(
            create_sources[0]['value'], create_sources[1]['value'],
            'the newer time must not mask the other occurrence',
        )
        date_sources = [s for s in sources if s['source'] ==
                        'hwp_date_string']
        self.assertEqual(1, len(date_sources))
        self.assertEqual('VT_LPWSTR', date_sources[0]['type'])

        formatter = HwpSummaryInfoTextFormatter()
        lines = list(formatter._value_lines('Created at',
                                            info.createdTimes()))
        self.assertEqual(2, len(lines))
        self.assertIn('Created at [0]:', lines[0])
        self.assertIn('Created at [1]:', lines[1])

    def test_formatter_tolerates_missing_and_raw_properties(self):
        info = self._summary_info([
            (1, VT_I4, i4(99999)),
            (HWPPIDSI_DATE_STR.id, VT_LPSTR, lpstr(b'\xff\xfe\x00')),
        ])
        formatter = HwpSummaryInfoTextFormatter()
        lines = list(formatter.formatTextLines(info))

        # missing identifiers render as empty values instead of raising
        self.assertIn('Title: ', lines)

        text = '\n'.join(lines)
        # the unknown-codepage date keeps its raw bytes and is still listed
        self.assertIn('raw-bytes', text)
        self.assertIn('fffe', text)
        self.assertIn('hwp_date_string (VT_LPSTR)', text)

