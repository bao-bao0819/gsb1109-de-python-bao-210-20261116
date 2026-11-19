#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import unittest

from Registry import Registry
from Registry import RegistryDiff


def make_key(path, values=None, offset=0x1000):
    key = RegistryDiff.KeySnapshot(path, path.split("\\")[-1], offset, None)
    for name, value_type, type_str, data, voffset in (values or []):
        vsnap = RegistryDiff.ValueSnapshot(name, value_type, type_str, data, voffset)
        key.values.setdefault(name.lower(), []).append(vsnap)
    return key


def make_snapshot(keys):
    snap = RegistryDiff.HiveSnapshot("test")
    for canonical, key in keys:
        snap.add_key(canonical, key)
    return snap


class TestReorganizedHiveDiff(unittest.TestCase):
    def setUp(self):
        samples = os.path.join(os.path.dirname(__file__), 'reg_samples')
        self.original = Registry.Registry(os.path.join(samples, 'UNICODE_TESTS'))
        self.reorganized = Registry.Registry(os.path.join(samples, 'REORGANIZED_VALUES_TESTS'))

    def test_reorganization_is_not_a_delete_recreate(self):
        diff = RegistryDiff.diff_registries(self.original, self.reorganized)
        self.assertFalse(diff.has_changes())
        self.assertEqual(0, len(diff.keys_removed))
        self.assertEqual(0, len(diff.keys_added))
        self.assertEqual(0, len(diff.values_removed))
        self.assertEqual(0, len(diff.values_added))
        self.assertEqual(0, len(diff.values_type_changed))
        self.assertEqual(0, len(diff.values_data_changed))
        # every key is paired logically, none left over
        self.assertEqual(3, len(diff.keys_matched))

    def test_self_diff_is_empty(self):
        diff = RegistryDiff.diff_registries(self.original, self.original)
        self.assertFalse(diff.has_changes())

    def test_matched_keys_keep_physical_provenance(self):
        diff = RegistryDiff.diff_registries(self.original, self.reorganized)
        for old_key, new_key in diff.keys_matched:
            self.assertTrue(old_key.offset > 0)
            self.assertTrue(new_key.offset > 0)


class TestLogicalPairing(unittest.TestCase):
    def test_key_add_remove_and_case_rename(self):
        old = make_snapshot([
            ("", make_key("ROOT")),
            ("root\\gone", make_key("ROOT\\Gone", offset=0x2000)),
            ("root\\stay", make_key("ROOT\\STAY", offset=0x3000)),
        ])
        new = make_snapshot([
            ("", make_key("ROOT")),
            ("root\\stay", make_key("ROOT\\stay", offset=0x5000)),
            ("root\\fresh", make_key("ROOT\\Fresh", offset=0x6000)),
        ])
        diff = RegistryDiff.diff_snapshots(old, new)
        self.assertEqual(["ROOT\\Gone"], [k.path for k in diff.keys_removed])
        self.assertEqual(["ROOT\\Fresh"], [k.path for k in diff.keys_added])
        # case-only rename is matched, not deleted and recreated
        matched = {o.path: n.path for (o, n) in diff.keys_matched}
        self.assertEqual("ROOT\\stay", matched["ROOT\\STAY"])
        self.assertEqual(0x3000, [o for (o, n) in diff.keys_matched
                                  if o.path == "ROOT\\STAY"][0].offset)
        self.assertEqual(0x5000, [n for (o, n) in diff.keys_matched
                                  if n.path == "ROOT\\stay"][0].offset)

    def test_duplicate_paths_are_conflicts_not_arbitrary_pairs(self):
        old = make_snapshot([
            ("", make_key("ROOT")),
            ("root\\dup", make_key("ROOT\\Dup", offset=0x2000)),
            ("root\\dup", make_key("ROOT\\dup", offset=0x2008)),
        ])
        new = make_snapshot([
            ("", make_key("ROOT")),
            ("root\\dup", make_key("ROOT\\DUP", offset=0x9000)),
        ])
        diff = RegistryDiff.diff_snapshots(old, new)
        self.assertEqual(1, len(diff.key_conflicts))
        conflict = diff.key_conflicts[0]
        self.assertEqual("root\\dup", conflict.canonical_path)
        self.assertEqual([0x2000, 0x2008], [k.offset for k in conflict.old_keys])
        self.assertEqual([0x9000], [k.offset for k in conflict.new_keys])
        # nothing paired, nothing silently added or removed
        self.assertEqual(1, len(diff.keys_matched))  # only the root
        self.assertEqual(0, len(diff.keys_removed))
        self.assertEqual(0, len(diff.keys_added))

    def test_value_type_and_data_changes(self):
        old = make_snapshot([
            ("", make_key("ROOT", values=[
                ("Typed", 0x1, "RegSZ", b"a\x00", 0x100),
                ("Data", 0x4, "RegDWord", b"\x01\x00\x00\x00", 0x200),
                ("Same", 0x4, "RegDWord", b"\x02\x00\x00\x00", 0x300),
                ("Gone", 0x1, "RegSZ", b"g\x00", 0x400),
            ])),
        ])
        new = make_snapshot([
            ("", make_key("ROOT", values=[
                ("Typed", 0x4, "RegDWord", b"\x01\x00\x00\x00", 0x500),
                ("Data", 0x4, "RegDWord", b"\x03\x00\x00\x00", 0x600),
                ("Same", 0x4, "RegDWord", b"\x02\x00\x00\x00", 0x700),
                ("Fresh", 0x1, "RegSZ", b"f\x00", 0x800),
            ], offset=0x1000)),
        ])
        diff = RegistryDiff.diff_snapshots(old, new)

        self.assertEqual(1, len(diff.values_type_changed))
        change = diff.values_type_changed[0]
        self.assertEqual("Typed", change.old_value.name)
        self.assertEqual("RegSZ", change.old_value.value_type_str)
        self.assertEqual("RegDWord", change.new_value.value_type_str)
        self.assertEqual((0x100, 0x500),
                         (change.old_value.offset, change.new_value.offset))

        self.assertEqual(1, len(diff.values_data_changed))
        change = diff.values_data_changed[0]
        self.assertEqual("Data", change.old_value.name)
        self.assertEqual((0x200, 0x600),
                         (change.old_value.offset, change.new_value.offset))

        self.assertEqual(["Gone"], [v.name for (p, v) in diff.values_removed])
        self.assertEqual(["Fresh"], [v.name for (p, v) in diff.values_added])

    def test_value_case_conflict_is_not_paired(self):
        old = make_snapshot([
            ("", make_key("ROOT", values=[
                ("Foo", 0x1, "RegSZ", b"a\x00", 0x100),
                ("FOO", 0x1, "RegSZ", b"b\x00", 0x200),
            ])),
        ])
        new = make_snapshot([
            ("", make_key("ROOT", values=[
                ("foo", 0x1, "RegSZ", b"a\x00", 0x300),
            ])),
        ])
        diff = RegistryDiff.diff_snapshots(old, new)
        self.assertEqual(1, len(diff.value_conflicts))
        conflict = diff.value_conflicts[0]
        self.assertEqual("foo", conflict.canonical_name)
        self.assertEqual([0x100, 0x200], [v.offset for v in conflict.old_values])
        self.assertEqual([0x300], [v.offset for v in conflict.new_values])
        self.assertEqual(0, len(diff.values_removed))
        self.assertEqual(0, len(diff.values_added))

    def test_default_value_name_matches_empty_name(self):
        old = make_snapshot([
            ("", make_key("ROOT", values=[("", 0x1, "RegSZ", b"a\x00", 0x100)])),
        ])
        new = make_snapshot([
            ("", make_key("ROOT", values=[("", 0x1, "RegSZ", b"b\x00", 0x200)])),
        ])
        diff = RegistryDiff.diff_snapshots(old, new)
        self.assertEqual(1, len(diff.values_data_changed))
        self.assertEqual("(default)",
                         diff.values_data_changed[0].old_value.display_name())

    def test_report_mentions_provenance_offsets(self):
        old = make_snapshot([("", make_key("ROOT", offset=0x111))])
        new = make_snapshot([("", make_key("ROOT", offset=0x222)),
                             ("root\\a", make_key("ROOT\\A", offset=0x333))])
        diff = RegistryDiff.diff_snapshots(old, new)
        report = RegistryDiff.format_report(diff)
        self.assertIn("0x333", report)
        self.assertIn("Keys added", report)


if __name__ == '__main__':
    unittest.main(verbosity=2)
