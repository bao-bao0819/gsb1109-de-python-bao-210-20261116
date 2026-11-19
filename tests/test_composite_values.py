#!/usr/bin/python
# -*- coding: utf-8 -*-
import struct
import unittest

from Registry import SettingsParse
from Registry import RegistryParse


def _name(text):
    encoded = text.encode("utf-16-le") + b"\x00\x00"
    return encoded, len(encoded) // 2 - 1


def item(item_type, text, payload, declared_len=None):
    """Serialize one composite item, padded to the 8-byte boundary."""
    name_bytes, name_chars = _name(text)
    body = struct.pack("<III", 0, item_type, name_chars) + name_bytes + payload
    if declared_len is None:
        declared_len = len(body)
    body = struct.pack("<III", declared_len, item_type, name_chars) + body[12:]
    padded = len(body)
    if padded % 8:
        body += b"\x00" * (8 - padded % 8)
    return body


def composite(items):
    """Serialize a nested composite payload (no trailing padding)."""
    return b"".join(items)


class TestCompositeUnknownValues(unittest.TestCase):
    def setUp(self):
        # Nested composite:
        #   Known = UInt32 0x11223344
        #   Mystery = unknown type 0x099 with a 6-byte payload
        # The outer value is a composite item with two children.
        self.inner_known = item(SettingsParse.RegUint32, "Known",
                                struct.pack("<I", 0x11223344))
        self.inner_unknown_payload = b"\xde\xad\xbe\xef\x01\x02"
        self.inner_unknown = item(0x099, "Mystery", self.inner_unknown_payload)
        self.inner = composite([self.inner_known, self.inner_unknown])
        self.outer_unknown_type = item(SettingsParse.RegCompositeValue, "Nested",
                                       self.inner)

    def test_known_fields_survive_unknown(self):
        parsed = SettingsParse.ParseAppDataCompositeStream(self.outer_unknown_type)
        self.assertEqual(["Nested"], list(parsed.keys()))
        nested = parsed["Nested"]
        self.assertEqual(2, nested.count)
        self.assertEqual(0x11223344, nested["Known"])
        self.assertIsInstance(nested["Mystery"], SettingsParse.UnknownCompositeValue)

    def test_unknown_type_and_raw_range(self):
        parsed = SettingsParse.ParseAppDataCompositeStream(self.outer_unknown_type)
        nested = parsed["Nested"]
        unknown = nested["Mystery"]
        self.assertEqual(0x099, unknown.item_type)
        self.assertEqual(self.inner_unknown_payload, unknown.data)
        self.assertEqual(len(self.inner_unknown_payload), unknown.size)

        data_start = nested.item_header_size
        expected_start = data_start + len(self.inner_known)
        expected_start += 12 + len("Mystery".encode("utf-16-le")) + 2
        self.assertEqual(expected_start, unknown.offset)
        self.assertEqual((expected_start, expected_start + len(self.inner_unknown_payload)),
                         unknown.raw_range)

    def test_outer_length_and_inner_count(self):
        parsed = SettingsParse.ParseAppDataCompositeStream(self.outer_unknown_type)
        nested = parsed["Nested"]
        # The outer item ends before its 8-byte alignment padding.
        self.assertLess(nested.item_byte_len, len(self.outer_unknown_type))
        self.assertEqual(len(self.outer_unknown_type), parsed.size)
        self.assertEqual(0, parsed.padding_size)
        # The nested stream is the declared payload after header + name.
        self.assertEqual(nested.item_byte_len - nested.item_header_size,
                         nested.size)
        self.assertEqual(2, nested.inner_count)
        self.assertFalse(nested.truncated)

    def test_outer_value_length_is_recorded(self):
        value = SettingsParse.ParseAppDataCompositeValue(
            SettingsParse.RegCompositeValue, self.inner, len(self.inner),
            outer_size=len(self.inner) + 64)
        self.assertEqual(len(self.inner) + 64, value.outer_size)
        self.assertEqual(2, value.inner_count)

    def test_malformed_length_does_not_cross_parent(self):
        # Claim a payload far larger than the available bytes.
        bad = item(0x099, "TooLong", b"\x01\x02", declared_len=12 + 16 + 0x1000)
        parsed = SettingsParse.ParseAppDataCompositeStream(bad)
        value = parsed["TooLong"]
        self.assertTrue(value.truncated)
        self.assertEqual(4, value.size)
        self.assertEqual(b"\x01\x02\x00\x00", value.data)
        self.assertTrue(parsed.truncated)

    def test_truncated_known_value_keeps_payload(self):
        bad = item(SettingsParse.RegUint32, "Short", b"\x01\x02")
        parsed = SettingsParse.ParseAppDataCompositeStream(bad)
        value = parsed["Short"]
        self.assertIsInstance(value, SettingsParse.UnknownCompositeValue)
        self.assertEqual(b"\x01\x02", value.data)
        self.assertIsNotNone(value.error)

    def test_bad_item_length_stops_without_losing_previous_items(self):
        first = item(SettingsParse.RegUint32, "First", struct.pack("<I", 7))
        # Name length too large for the declared item length.
        broken = struct.pack("<III", 16, 0x077, 100) + b"\x00" * 4
        parsed = SettingsParse.ParseAppDataCompositeStream(first + broken)
        self.assertEqual(7, parsed["First"])
        self.assertIsNotNone(parsed.trailing)
        self.assertTrue(parsed.truncated)

    def test_zero_tail_is_padding_not_item(self):
        parsed = SettingsParse.ParseAppDataCompositeStream(self.outer_unknown_type + b"\x00" * 8)
        self.assertEqual(8, parsed.padding_size)
        self.assertEqual(len(self.outer_unknown_type) + 8, parsed.size)

    def test_reparse_only_affected_slice(self):
        parsed = SettingsParse.ParseAppDataCompositeStream(self.outer_unknown_type)
        nested = parsed["Nested"]
        known_before = nested["Known"]
        unknown_before = nested["Mystery"]

        def parse_mystery(data, data_size, offset=0):
            return data

        replaced = SettingsParse.ReparseUnknownValues(
            parsed, {0x099: parse_mystery})

        self.assertEqual(1, replaced)
        self.assertIs(known_before, nested["Known"])
        self.assertEqual(self.inner_unknown_payload, nested["Mystery"])
        self.assertEqual(self.inner_unknown_payload, unknown_before.data)
        self.assertEqual((unknown_before.offset,
                          unknown_before.offset + len(self.inner_unknown_payload)),
                         unknown_before.raw_range)

    def test_register_type_then_reparse(self):
        parsed = SettingsParse.ParseAppDataCompositeStream(self.outer_unknown_type)
        nested = parsed["Nested"]

        SettingsParse.RegisterCompositeValueType(
            0x099, lambda data, size, offset=0: list(data))
        try:
            self.assertEqual(1, SettingsParse.ReparseUnknownValues(parsed))
        finally:
            SettingsParse.UnregisterCompositeValueType(0x099)
        self.assertEqual(list(self.inner_unknown_payload), nested["Mystery"])

    def test_reparse_to_nested_keeps_length_bookkeeping(self):
        # Unknown type that later becomes a nested composite stream.
        inner = composite([item(SettingsParse.RegUint32, "Leaf",
                                struct.pack("<I", 5))])
        encoded = item(0x0AB, "Later", inner)
        parsed = SettingsParse.ParseAppDataCompositeStream(encoded)
        before = parsed["Later"]

        SettingsParse.RegisterCompositeValueType(
            0x0AB, SettingsParse.ParseAppDataCompositeStream)
        try:
            self.assertEqual(1, SettingsParse.ReparseUnknownValues(parsed))
        finally:
            SettingsParse.UnregisterCompositeValueType(0x0AB)

        now = parsed["Later"]
        self.assertIsInstance(now, SettingsParse.CompositeValue)
        self.assertEqual(5, now["Leaf"])
        self.assertEqual(before.item_byte_len, now.item_byte_len)
        self.assertEqual(before.item_header_size, now.item_header_size)
        self.assertTrue(now.verified)

    def test_composite_value_is_a_dict(self):
        parsed = SettingsParse.ParseAppDataCompositeStream(self.outer_unknown_type)
        self.assertIsInstance(parsed, dict)
        self.assertEqual(1, parsed.count)
        self.assertEqual(parsed.size, parsed.length)
        self.assertEqual(parsed.size, parsed.byte_length)

    def test_no_item_is_dropped_around_unknown_type(self):
        parsed = SettingsParse.ParseAppDataCompositeStream(self.outer_unknown_type)
        nested = parsed["Nested"]
        self.assertEqual(["Known", "Mystery"], list(nested.keys()))
        self.assertEqual(
            [("Mystery", nested["Mystery"])], nested.unknown_values())

    def test_non_padding_tail_is_preserved_not_swallowed(self):
        stream = composite([self.inner_known]) + b"\x01\x02\x03"
        parsed = SettingsParse.ParseAppDataCompositeStream(stream)
        self.assertEqual(0x11223344, parsed["Known"])
        self.assertTrue(parsed.truncated)
        self.assertIsNotNone(parsed.trailing)
        self.assertEqual((parsed.size, parsed.size + 3), parsed.trailing.raw_range)
        self.assertEqual(b"\x01\x02\x03", parsed.trailing.data)

    def test_unknown_range_stays_inside_parent(self):
        parsed = SettingsParse.ParseAppDataCompositeStream(self.outer_unknown_type)
        nested = parsed["Nested"]
        parent_start, parent_end = nested.raw_range
        for value in nested.values():
            if isinstance(value, SettingsParse.UnknownCompositeValue):
                self.assertGreaterEqual(value.offset, parent_start)
                self.assertLessEqual(value.end, parent_end)


class _StubVKRecord(RegistryParse.VKRecord):
    """VKRecord with just the fields VKRecord.data() consults."""

    def __init__(self, data_type, payload):
        self._data_type = data_type
        self._payload = payload

    def data_type(self):
        return self._data_type

    def raw_data_length(self):
        return len(self._payload) + 8

    def raw_data(self, overrun=0):
        return self._payload + b"\x22" * 8

    def data_offset(self):
        return 0x1234

    def name(self):
        return "Val"


class TestVKRecordCompositeDispatch(unittest.TestCase):
    def test_known_settings_type_still_decodes(self):
        value = _StubVKRecord(0x105, struct.pack("<I", 42) + b"\x11" * 8).data()
        self.assertEqual(42, value)

    def test_unknown_settings_type_is_preserved(self):
        value = _StubVKRecord(0x1AA, b"abcdefgh").data()
        self.assertIsInstance(value, SettingsParse.UnknownCompositeValue)
        self.assertEqual(0xAA, value.item_type)
        self.assertEqual(b"abcdefgh", value.data)
        self.assertEqual((0x1234, 0x123C), value.raw_range)


if __name__ == "__main__":
    unittest.main(verbosity=2)
