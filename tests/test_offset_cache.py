#!/usr/bin/python
# -*- coding: utf-8 -*-
import io
import os
import struct
import datetime
import unittest

from Registry import Registry
from Registry import Caching
from Registry.Caching import OffsetCache, ChangeType, StaleOffsetCacheException


SAMPLE = os.path.join(os.path.dirname(__file__),
                      "reg_samples", "REORGANIZED_VALUES_TESTS")

EPOCH = datetime.datetime(1601, 1, 1)


def windows_timestamp(dt):
    ticks = (dt - EPOCH).total_seconds() * 10000000
    return struct.pack("<Q", int(ticks))


def tampered_reorganization(path, new_timestamp=datetime.datetime(2020, 1, 1)):
    """
    Return a BytesIO copy of the hive whose REGF reorganized timestamp was
    bumped, as Windows does when a hive is reorganized.
    """
    with open(path, "rb") as f:
        buf = bytearray(f.read())
    buf[0xA8:0xB0] = windows_timestamp(new_timestamp)
    return io.BytesIO(bytes(buf))


def all_values(registry):
    result = []
    root_path = registry.root().path()

    def relative_path(key):
        path = key.path()
        if path == root_path:
            return ""
        return path[len(root_path) + 1:]

    def walk(key):
        for value in key.values():
            result.append((relative_path(key), value))
        for subkey in key.subkeys():
            walk(subkey)

    walk(registry.root())
    return result


class TestOffsetCache(unittest.TestCase):
    def setUp(self):
        self.registry = Registry.Registry(SAMPLE)
        self.cache = OffsetCache.build(self.registry)
        self.logical = [(path, value.name()) for path, value in all_values(self.registry)]

    def test_cache_entries_use_logical_identity_and_live_offsets(self):
        self.assertIsNotNone(self.cache.source_state())
        self.assertEqual(len(self.cache._entries), len(self.logical))
        for path, name in self.logical:
            entry = self.cache._entries[(path, name)]
            value = self.registry.open(path).value(name)
            self.assertEqual(entry.offset, value._vkrecord.offset())
            self.assertEqual(entry.fingerprint, Caching.value_fingerprint(value))

    def test_source_matches_ignores_external_file_name_and_size(self):
        same_bytes = io.BytesIO(open(SAMPLE, "rb").read())
        other = Registry.Registry(same_bytes)
        self.assertTrue(self.cache.source_matches(other))
        self.assertFalse(self.cache.reorganization_detected(other))

    def test_serialization_round_trip(self):
        payload = io.BytesIO()
        self.cache.save(payload)
        payload.seek(0)
        restored = OffsetCache.load(payload)

        self.assertEqual(restored.source_state(), self.cache.source_state())
        self.assertEqual(set(restored._entries), set(self.cache._entries))
        self.assertEqual(restored.to_dict(), self.cache.to_dict())

    def test_unchanged_hive_books_nothing_twice(self):
        resolutions = list(self.cache.changes(self.registry))
        self.assertEqual(len(resolutions), len(self.logical))
        self.assertTrue(all(r.change == ChangeType.UNCHANGED for r in resolutions))
        self.assertEqual(self.cache.bookable(self.registry), [])

    def test_reorganization_is_detected_from_regf_markers(self):
        reorganized = Registry.Registry(tampered_reorganization(SAMPLE))
        self.assertTrue(self.cache.reorganization_detected(reorganized))
        self.assertFalse(self.cache.source_matches(reorganized))
        with self.assertRaises(StaleOffsetCacheException):
            self.cache.validate_source(reorganized)

    def test_validate_source_rejects_foreign_hive(self):
        data = self.cache.to_dict()
        data["source"]["hive_name"] = "C:\\different\\hive"
        foreign = OffsetCache.from_dict(data)
        with self.assertRaises(StaleOffsetCacheException):
            foreign.validate_source(self.registry)

    def test_stale_offset_pointing_at_foreign_cell_is_relocated(self):
        values = all_values(self.registry)
        victim_path, victim = values[0]
        foreign_path, foreign = values[1]

        entry_key = (victim_path, victim.name())
        stale_entry = self.cache._entries[entry_key]._replace(
            offset=foreign._vkrecord.offset())
        self.cache._entries[entry_key] = stale_entry

        # The stale offset is readable, but the cell carries another key's
        # value name, so it must not be accepted.
        self.assertFalse(
            self.cache.cached_offset_is_valid(self.registry, victim_path, victim.name()))

        resolution = self.cache.resolve(self.registry, victim_path, victim.name())
        self.assertTrue(resolution.offset_changed)
        self.assertEqual(resolution.current_offset, victim._vkrecord.offset())
        self.assertEqual(resolution.value.name(), victim.name())
        self.assertEqual(resolution.value.raw_data(), victim.raw_data())
        self.assertEqual(resolution.change, ChangeType.UNCHANGED)
        self.assertNotIn(resolution, self.cache.bookable(self.registry))

    def test_content_change_is_booked_even_when_offset_is_stable(self):
        path, name = self.logical[0]
        entry_key = (path, name)
        entry = self.cache._entries[entry_key]
        self.cache._entries[entry_key] = entry._replace(fingerprint="0" * 64)

        resolution = self.cache.resolve(self.registry, path, name)
        self.assertFalse(resolution.offset_changed)
        self.assertEqual(resolution.change, ChangeType.CHANGED)
        self.assertEqual(self.cache.bookable(self.registry)[0].key_path, path)

    def test_new_value_is_booked(self):
        empty = OffsetCache(source=self.cache.source_state())
        resolutions = list(empty.changes(self.registry))
        self.assertTrue(all(r.change == ChangeType.NEW for r in resolutions))
        self.assertEqual(len(empty.bookable(self.registry)), len(self.logical))

    def test_missing_value_is_reported_not_booked(self):
        path, name = self.logical[0]
        resolution = self.cache.resolve(self.registry, path, "does-not-exist")
        self.assertEqual(resolution.change, ChangeType.MISSING)
        self.assertIsNone(resolution.value)

        entry = self.cache._entries[(path, name)]
        ghost = entry._replace(key_path=path, value_name="vanished")
        self.cache._entries[(path, "vanished")] = ghost
        missing = [r for r in self.cache.changes(self.registry)
                   if r.change == ChangeType.MISSING]
        self.assertEqual([(r.key_path, r.value_name) for r in missing],
                         [(path, "vanished")])

    def test_relocate_rebinds_after_reorganization_without_double_booking(self):
        reorganized = Registry.Registry(tampered_reorganization(SAMPLE))
        self.assertTrue(self.cache.reorganization_detected(reorganized))
        self.cache.relocate(reorganized)

        self.assertTrue(self.cache.source_matches(reorganized))
        self.cache.validate_source(reorganized)
        self.assertEqual(self.cache.bookable(reorganized), [])
        for path, name in self.logical:
            self.assertEqual(
                self.cache.resolve(reorganized, path, name).change,
                ChangeType.UNCHANGED)

        # the rebuilt cache no longer flags the same hive as reorganized
        self.assertFalse(self.cache.reorganization_detected(reorganized))

    def test_invalid_cache_format_is_rejected(self):
        data = self.cache.to_dict()
        data["format_version"] = 999
        with self.assertRaises(StaleOffsetCacheException):
            OffsetCache.from_dict(data)

    def test_default_value_name_round_trips_to_empty_internal_name(self):
        self.assertEqual(
            OffsetCache._public_to_internal_name("(default)"), "")
        self.assertEqual(
            OffsetCache._public_to_internal_name("Named"), "Named")
        resolution = self.cache.resolve(
            self.registry, self.logical[0][0], "(default)")
        self.assertEqual(resolution.change, ChangeType.MISSING)


if __name__ == "__main__":
    unittest.main(verbosity=2)
