#!/usr/bin/python
# -*- coding: utf-8 -*-
import hashlib
import os
import unittest

from Registry import Registry, RegistryParse

MAX_SEGMENT = RegistryParse.DBIndirectBlock.MAX_SEGMENT_SIZE


def fragment_payload(seed):
    return bytes(((index * 7 + seed) & 0xFF) for index in range(MAX_SEGMENT))


class TestCorruptBigDataRecovery(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(os.path.dirname(__file__),
                                 "reg_samples", "corrupt_big_data.hive")
        self.root = Registry.Registry(self.path).root()
        self.fragments = [fragment_payload(1), fragment_payload(2), fragment_payload(3)]
        self.full_data = b"".join(self.fragments)

    def _value(self, name):
        return self.root.value(name)

    def test_unrelated_complete_value_is_kept(self):
        recovered = self._value("Small").recover_raw_data()
        self.assertTrue(recovered.is_complete())
        self.assertEqual(recovered.status, RegistryParse.RecoveredData.COMPLETE)
        self.assertEqual(recovered.data, b"A" * 16)
        self.assertEqual(recovered.expected_length, 16)
        self.assertEqual(recovered.issues, [])
        self.assertEqual(recovered.fragment_locations, [])
        self.assertEqual(self._value("Small").raw_data(), b"A" * 16)

    def test_intact_big_data_recovered_completely(self):
        value = self._value("IntactBig")
        recovered = value.recover_raw_data()

        self.assertTrue(recovered.is_complete())
        self.assertEqual(len(recovered.data), MAX_SEGMENT * 3)
        self.assertEqual(recovered.data, self.full_data)
        self.assertEqual(
            hashlib.md5(recovered.data).hexdigest(),
            hashlib.md5(self.full_data).hexdigest())
        self.assertEqual(recovered.issues, [])
        self.assertEqual(value.raw_data(), self.full_data)

        offsets = [location.cell_offset for location in recovered.fragment_locations]
        self.assertEqual(offsets, [0x2020, 0x6020, 0xA038])
        for index, location in enumerate(recovered.fragment_locations):
            self.assertEqual(location.index, index)
            self.assertEqual(location.length, MAX_SEGMENT)
            self.assertEqual(location.value_start, index * MAX_SEGMENT)
            self.assertEqual(location.data_offset, location.cell_offset + 4)

        # fragments must not be physically adjacent: a filler cell precedes
        # the last fragment and a decoy cell follows it
        self.assertNotEqual(0x6020 + 0x3FE0, 0xA038)
        decoy_cell = RegistryParse.HBINCell(
            value._vkrecord._buf, 0xE018, value._vkrecord)
        self.assertTrue(decoy_cell.raw_data().startswith(b"DECOY!!"))

    def test_broken_last_reference_yields_definite_prefix_only(self):
        value = self._value("BrokenBig")
        recovered = value.recover_raw_data()

        self.assertFalse(recovered.is_complete())
        self.assertEqual(recovered.status, RegistryParse.RecoveredData.PARTIAL)
        self.assertEqual(recovered.expected_length, MAX_SEGMENT * 3 + 100)
        self.assertEqual(recovered.prefix_length, MAX_SEGMENT * 3)
        self.assertEqual(recovered.data, self.full_data)

        # the unknown tail must not be invented from the physically adjacent cell
        self.assertNotIn(b"DECOY!!", recovered.data)
        self.assertEqual(len(recovered.data), recovered.prefix_length)

        self.assertEqual(len(recovered.issues), 1)
        self.assertIn("invalid fragment reference", recovered.issues[0])
        self.assertIn("0x77770", recovered.issues[0])
        self.assertEqual(
            [location.index for location in recovered.fragment_locations],
            [0, 1, 2])

        # raw_data() refuses to fabricate the tail but keeps the prefix
        with self.assertRaises(RegistryParse.RegistryDataRecoveryException) as context:
            value.raw_data()
        exception = context.exception
        self.assertEqual(exception.recovered_data().data, self.full_data)
        self.assertEqual(exception.recovered_data().prefix_length,
                         MAX_SEGMENT * 3)
        self.assertEqual(exception.recovered_data().issues, recovered.issues)

    def test_duplicate_reference_and_length_contradiction_recorded(self):
        value = self._value("DupBig")
        recovered = value.recover_raw_data()

        self.assertFalse(recovered.is_complete())
        self.assertEqual(recovered.expected_length, MAX_SEGMENT * 3)
        self.assertEqual(recovered.prefix_length, MAX_SEGMENT)
        self.assertEqual(recovered.data, self.fragments[0])
        self.assertEqual(len(recovered.issues), 1)
        self.assertIn("duplicate fragment reference", recovered.issues[0])
        self.assertIn("index 1", recovered.issues[0])
        self.assertEqual(
            [location.index for location in recovered.fragment_locations], [0])

        with self.assertRaises(RegistryParse.RegistryDataRecoveryException):
            value.raw_data()

    def test_truncated_chain_records_length_contradiction(self):
        value = self._value("TruncatedBig")
        recovered = value.recover_raw_data()

        self.assertFalse(recovered.is_complete())
        self.assertEqual(recovered.expected_length, MAX_SEGMENT * 3)
        self.assertEqual(recovered.prefix_length, MAX_SEGMENT)
        self.assertEqual(recovered.data, self.fragments[0])
        self.assertEqual(len(recovered.issues), 1)
        self.assertIn("length contradiction", recovered.issues[0])
        self.assertEqual(
            [location.index for location in recovered.fragment_locations], [0])

        with self.assertRaises(RegistryParse.RegistryDataRecoveryException):
            value.raw_data()

    def test_fragments_are_only_read_through_references(self):
        # walking physical neighbors after the last fragment would reach filler
        # and decoy cells, neither of which may appear in recovered data
        recovered = self._value("BrokenBig").recover_raw_data()
        self.assertNotIn(b"NOT-A-FRAGMENT", recovered.data)
        self.assertNotIn(b"DECOY!!", recovered.data)

        intact = self._value("IntactBig").recover_raw_data()
        self.assertEqual(
            [location.reference for location in intact.fragment_locations],
            [0x1020, 0x5020, 0x9038])


if __name__ == "__main__":
    unittest.main(verbosity=2)
