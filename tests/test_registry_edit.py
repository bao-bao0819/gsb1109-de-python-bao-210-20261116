#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import unittest

from Registry import Registry
from Registry import RegistryEdit
from Registry.Registry import DEFAULT_VALUE
from Registry.RegistryEdit import (
    HiveSnapshot, EditPlan, EditGroup, ValueIdentity, ValuePresence,
    RehearsalVerificationError,
)


def make_snapshot():
    return HiveSnapshot.from_mapping("Root", {
        "Root": [
            {"name": "", "type": "RegSZ", "value": "default-data"},
            {"name": "(Default)", "type": "RegSZ", "value": "literal-data"},
            {"name": "Keep", "type": "RegDWord", "value": 7},
            {"name": "Clear", "type": "RegBin", "value": b"abc"},
            {"name": "Empty", "type": "RegNone", "value": b""},
        ],
        "Root\\Sub": [
            {"name": "Rename", "type": "RegSZ", "value": "hello"},
            {"name": "Delete", "type": "RegQWord", "value": 9},
            {"name": "Retype", "type": "RegSZ", "value": "42"},
        ],
    })


class TestRegistryValueIdentity(unittest.TestCase):
    def setUp(self):
        self.registry = Registry.Registry(
            os.path.join(os.path.dirname(__file__), "reg_samples", "issue22.hive"))
        self.key = self.registry.root()

    def test_default_is_unnamed(self):
        snapshot = make_snapshot()
        value = snapshot.find_value("Root", DEFAULT_VALUE)
        self.assertTrue(value.is_default)
        self.assertEqual(value.name, "")
        self.assertEqual(value.display_name, "(default)")
        self.assertEqual(value.value, "default-data")

    def test_value_exact_distinguishes_default_and_literal(self):
        # This hive has no unnamed default and no literal "(Default)" value.
        with self.assertRaises(Registry.RegistryValueNotFoundException):
            self.key.value_exact("")
        with self.assertRaises(Registry.RegistryValueNotFoundException):
            self.key.value_exact("(Default)")
        with self.assertRaises(Registry.RegistryValueNotFoundException):
            self.key.value(DEFAULT_VALUE)

    def test_friendly_default_lookup_still_works(self):
        self.assertFalse(self.key.has_value(DEFAULT_VALUE))
        self.assertFalse(self.key.has_value("(default)"))
        self.assertTrue(self.key.has_value("Bias"))
        self.assertEqual(self.key.value("Bias").value(), self.key.value("bias").value())

    def test_missing_key_value_is_not_default(self):
        with self.assertRaises(Registry.RegistryValueNotFoundException):
            self.key.value("NoSuchValue")


class TestSnapshotIdentity(unittest.TestCase):
    def setUp(self):
        self.snapshot = make_snapshot()

    def test_default_and_literal_have_distinct_identity(self):
        default = self.snapshot.find_value("Root", DEFAULT_VALUE)
        literal = self.snapshot.find_value("Root", "(Default)")
        self.assertIsNotNone(default)
        self.assertIsNotNone(literal)
        self.assertTrue(default.is_default)
        self.assertFalse(literal.is_default)
        self.assertEqual(default.value, "default-data")
        self.assertEqual(literal.value, "literal-data")
        self.assertNotEqual(default.identity, literal.identity)

    def test_empty_existing_value_differs_from_missing(self):
        empty = self.snapshot.find_value("Root", "Empty")
        missing = self.snapshot.find_value("Root", "NotThere")
        self.assertIsNotNone(empty)
        self.assertTrue(empty.is_empty)
        self.assertIsNone(missing)
        self.assertEqual(self.snapshot.presence("Root", "Empty"), ValuePresence.NAMED)
        self.assertEqual(self.snapshot.presence("Root", "NotThere"), ValuePresence.MISSING)
        self.assertEqual(self.snapshot.presence("Root", DEFAULT_VALUE), ValuePresence.DEFAULT)

    def test_presence_reports_literal_default_as_named(self):
        self.assertEqual(self.snapshot.presence("Root", "(Default)"), ValuePresence.NAMED)
        # Logical identity is case-insensitive, so lowercase "(default)" also
        # names the literal value for exact lookups.
        self.assertEqual(self.snapshot.presence("Root", "(default)"), ValuePresence.NAMED)
        self.assertEqual(self.snapshot.presence_friendly("Root", "(default)"),
                         ValuePresence.DEFAULT)
        self.assertEqual(self.snapshot.resolve("Root", "(default)").value,
                         "default-data")
        literal = self.snapshot.find_value("Root", "(Default)")
        self.assertEqual(literal.value, "literal-data")
        self.assertFalse(literal.is_default)

    def test_snapshot_from_real_hive(self):
        registry = Registry.Registry(
            os.path.join(os.path.dirname(__file__), "reg_samples", "issue22.hive"))
        snapshot = HiveSnapshot.from_registry(registry)
        self.assertEqual(snapshot.revision, 1)
        bias = snapshot.find_value("TimeZoneInformation", "Bias")
        self.assertEqual(bias.value_type_name, "RegDWord")
        self.assertEqual(bias.raw_data, b"\xc4\xff\xff\xff")
        self.assertEqual(bias.origin.revision, 1)


class TestConversions(unittest.TestCase):
    def test_convert_default_to_named(self):
        plan = EditPlan(make_snapshot())
        plan.add_group(EditGroup("convert").convert_default("Root", "Promoted"))
        result = plan.apply()
        self.assertTrue(result.verification.passed)
        self.assertEqual([g.name for g in result.applied_groups], ["convert"])
        self.assertIsNone(result.snapshot.find_value("Root", DEFAULT_VALUE))
        promoted = result.snapshot.find_value("Root", "Promoted")
        self.assertIsNotNone(promoted)
        self.assertEqual(promoted.value, "default-data")
        self.assertEqual(promoted.origin.identity.name, "")
        self.assertIsNotNone(result.snapshot.find_value("Root", "(Default)"))

    def test_convert_default_to_literal_default_is_occupancy_conflict(self):
        plan = EditPlan(make_snapshot())
        plan.add_group(EditGroup("conflict").convert_default("Root", "(Default)"))
        result = plan.apply()
        self.assertEqual(result.applied_groups, [])
        self.assertEqual(len(result.rejected_groups), 1)
        self.assertTrue(result.verification.passed)

    def test_rename_to_occupied_target_is_rejected(self):
        plan = EditPlan(make_snapshot())
        plan.add_group(EditGroup("conflict").rename("Root", "Keep", "(Default)"))
        result = plan.apply()
        self.assertEqual(result.applied_groups, [])
        self.assertEqual(len(result.rejected_groups), 1)
        self.assertIn("occupied", result.rejected_groups[0].operations[0].reason.lower())

    def test_rename_missing_source_is_rejected_without_guessing(self):
        plan = EditPlan(make_snapshot())
        plan.add_group(EditGroup("missing").rename("Root", "Absent", "New"))
        result = plan.apply()
        self.assertEqual(result.applied_groups, [])
        self.assertEqual(len(result.rejected_groups), 1)
        self.assertIn("does not exist", result.rejected_groups[0].operations[0].reason)

    def test_rename_default_to_a_new_name(self):
        plan = EditPlan(make_snapshot())
        plan.add_group(EditGroup("rename").rename("Root", DEFAULT_VALUE, "Renamed"))
        result = plan.apply()
        renamed = result.snapshot.find_value("Root", "Renamed")
        self.assertEqual(renamed.value, "default-data")
        self.assertIsNone(result.snapshot.find_value("Root", DEFAULT_VALUE))

    def test_delete_value(self):
        plan = EditPlan(make_snapshot())
        plan.add_group(EditGroup("delete").delete("Root\\Sub", "Delete"))
        result = plan.apply()
        self.assertTrue(result.verification.passed)
        self.assertIsNone(result.snapshot.find_value("Root\\Sub", "Delete"))
        self.assertEqual([str(i) for i in result.deleted_identities],
                         ["Root\\Sub\\Delete"])
        change = result.deleted_values[0]
        self.assertTrue(change.before.exists)
        self.assertFalse(change.after.exists)
        self.assertEqual(change.before.raw_data,
                         b"\x09\x00\x00\x00\x00\x00\x00\x00")

    def test_delete_missing_value_is_rejected(self):
        plan = EditPlan(make_snapshot())
        plan.add_group(EditGroup("delete").delete("Root\\Sub", "NoSuchValue"))
        result = plan.apply()
        self.assertEqual(result.applied_groups, [])

    def test_change_type_preserves_or_replaces_data(self):
        plan = EditPlan(make_snapshot())
        plan.add_group(EditGroup("retype").change_type("Root\\Sub", "Retype", "RegDWord"))
        result = plan.apply()
        retyped = result.snapshot.find_value("Root\\Sub", "Retype")
        self.assertEqual(retyped.value_type, Registry.RegDWord)
        self.assertEqual(retyped.value, 42)
        self.assertEqual(retyped.raw_data, b"*\x00\x00\x00")

    def test_change_type_with_new_encoding(self):
        plan = EditPlan(make_snapshot())
        plan.add_group(EditGroup("retype").change_type(
            "Root\\Sub", "Retype", "RegDWord", 42))
        result = plan.apply()
        retyped = result.snapshot.find_value("Root\\Sub", "Retype")
        self.assertEqual(retyped.value, 42)
        self.assertEqual(retyped.raw_data, b"*\x00\x00\x00")

    def test_change_type_with_explicit_value(self):
        plan = EditPlan(make_snapshot())
        plan.add_group(EditGroup("retype").change_type(
            "Root", "Keep", "RegSZ", "new"))
        result = plan.apply()
        self.assertEqual(result.snapshot.find_value("Root", "Keep").value, "new")

    def test_clear_keeps_value_present_with_empty_data(self):
        plan = EditPlan(make_snapshot())
        plan.add_group(EditGroup("clear").clear("Root", "Clear"))
        result = plan.apply()
        cleared = result.snapshot.find_value("Root", "Clear")
        self.assertIsNotNone(cleared)
        self.assertEqual(cleared.raw_data, b"")
        self.assertTrue(cleared.is_empty)
        self.assertEqual(cleared.value_type, Registry.RegBin)

    def test_clear_missing_value_is_rejected(self):
        plan = EditPlan(make_snapshot())
        plan.add_group(EditGroup("clear").clear("Root", "Missing"))
        result = plan.apply()
        self.assertEqual(result.applied_groups, [])


class TestGroups(unittest.TestCase):
    def test_only_independent_groups_are_applied(self):
        plan = EditPlan(make_snapshot())
        plan.add_group(EditGroup("a").rename("Root\\Sub", "Rename", "X"))
        plan.add_group(EditGroup("b").rename("Root\\Sub", "Rename", "Y"))
        plan.add_group(EditGroup("ok").clear("Root", "Clear"))
        result = plan.apply()
        applied = [group.name for group in result.applied_groups]
        skipped = [group.name for group in result.skipped_groups]
        self.assertEqual(applied, ["ok"])
        self.assertEqual(sorted(skipped), ["a", "b"])
        self.assertTrue(result.verification.passed)
        unchanged = result.snapshot.find_value("Root\\Sub", "Rename")
        self.assertEqual(unchanged.value, "hello")

    def test_writer_and_reader_of_same_identity_both_skip(self):
        plan = EditPlan(make_snapshot())
        plan.add_group(EditGroup("writer").rename("Root\\Sub", "Rename", "X"))
        plan.add_group(EditGroup("reader").change_type(
            "Root\\Sub", "Rename", "RegQWord"))
        result = plan.apply()
        self.assertEqual(result.applied_groups, [])
        self.assertEqual(sorted(g.name for g in result.skipped_groups),
                         ["reader", "writer"])
        self.assertEqual(len(result.rehearsal.dependencies), 1)

    def test_unrelated_groups_both_apply(self):
        plan = EditPlan(make_snapshot())
        plan.add_group(EditGroup("one").clear("Root", "Clear"))
        plan.add_group(EditGroup("two").delete("Root\\Sub", "Delete"))
        result = plan.apply()
        self.assertEqual(sorted(g.name for g in result.applied_groups),
                         ["one", "two"])
        self.assertTrue(result.verification.passed)

    def test_one_invalid_group_does_not_block_valid_group(self):
        plan = EditPlan(make_snapshot())
        plan.add_group(EditGroup("bad").delete("Root\\Sub", "NoSuchValue"))
        plan.add_group(EditGroup("good").clear("Root", "Clear"))
        result = plan.apply()
        self.assertEqual([g.name for g in result.applied_groups], ["good"])
        self.assertEqual([g.name for g in result.rejected_groups], ["bad"])
        self.assertTrue(result.verification.passed)

    def test_group_with_dependent_operations_validates_against_working_state(self):
        plan = EditPlan(make_snapshot())
        group = (EditGroup("chain")
                 .convert_default("Root", "Temporary")
                 .rename("Root", "Temporary", "Final"))
        plan.add_group(group)
        result = plan.apply()
        self.assertEqual(result.applied_groups[0].name, "chain")
        self.assertEqual(result.snapshot.find_value("Root", "Final").value,
                         "default-data")
        self.assertIsNone(result.snapshot.find_value("Root", "Temporary"))

    def test_group_rolls_back_when_later_operation_is_invalid(self):
        plan = EditPlan(make_snapshot())
        group = (EditGroup("partial")
                 .clear("Root", "Clear")
                 .delete("Root\\Sub", "MissingValue"))
        plan.add_group(group)
        result = plan.apply()
        self.assertEqual(result.applied_groups, [])
        self.assertEqual(len(result.rejected_groups), 1)
        self.assertEqual(result.snapshot.find_value("Root", "Clear").raw_data, b"abc")

    def test_duplicate_group_names_are_rejected_conservatively(self):
        plan = EditPlan(make_snapshot())
        plan.add_group(EditGroup("dup").rename("Root", "Keep", "One"))
        plan.add_group(EditGroup("dup").rename("Root", "Keep", "Two"))
        result = plan.apply()
        self.assertEqual(result.snapshot.find_value("Root", "Keep").value, 7)
        rejected = [group for group in result.rehearsal.groups
                    if group.status.value == "rejected"]
        self.assertEqual(len(rejected), 2)
        self.assertTrue(all("Duplicate" in group.reason for group in rejected))


class TestProvenance(unittest.TestCase):
    def test_original_values_remain_available(self):
        snapshot = make_snapshot()
        plan = EditPlan(snapshot)
        plan.add_group(EditGroup("change").change_type(
            "Root", "Keep", "RegQWord", 99))
        result = plan.apply()
        self.assertEqual(result.snapshot.find_value("Root", "Keep").value, 99)
        original = result.original_snapshot.find_value("Root", "Keep")
        self.assertEqual(original.value, 7)
        self.assertEqual(original.value_type, Registry.RegDWord)

    def test_history_tracks_rename_origin(self):
        plan = EditPlan(make_snapshot())
        plan.add_group(EditGroup("rename").convert_default("Root", "Promoted"))
        result = plan.apply()
        identity = ValueIdentity("Root", "Promoted")
        history = result.history_for(identity)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].operation, "ConvertDefaultValue")
        self.assertEqual(history[0].before.identity.name, "")
        self.assertEqual(history[0].after.identity.name, "Promoted")
        self.assertEqual(history[0].before.value, "default-data")

    def test_origin_lookup_for_deleted_value(self):
        plan = EditPlan(make_snapshot())
        plan.add_group(EditGroup("delete").delete("Root\\Sub", "Delete"))
        result = plan.apply()
        origin = result.origin("Root\\Sub", "Delete")
        self.assertIsNotNone(origin)
        self.assertEqual(origin.raw_data, b"\x09\x00\x00\x00\x00\x00\x00\x00")

    def test_rehearsal_does_not_modify_snapshot(self):
        snapshot = make_snapshot()
        plan = EditPlan(snapshot)
        plan.add_group(EditGroup("clear").clear("Root", "Clear"))
        rehearsal = plan.rehearse()
        self.assertTrue(rehearsal.expected_snapshot.find_value("Root", "Clear").is_empty)
        self.assertEqual(snapshot.find_value("Root", "Clear").raw_data, b"abc")


class TestVerification(unittest.TestCase):
    def test_verification_failure_is_reported(self):
        plan = EditPlan(make_snapshot())
        plan.add_group(EditGroup("clear").clear("Root", "Clear"))

        def tamper(snapshot):
            values = snapshot.values_at("Root")
            original = snapshot.find_value("Root", "Clear")
            tampered = original.copy_with(value_type=Registry.RegDWord,
                                          raw_data=b"\x01\x00\x00\x00", value=1)
            replacement = []
            for value in values:
                if value.identity == tampered.identity:
                    replacement.append(tampered)
                else:
                    replacement.append(value)
            keys = [key for key in snapshot.keys()
                    if key.path != "Root"] + [
                RegistryEdit.KeyState("Root", replacement)]
            return HiveSnapshot("Root", keys, revision=snapshot.revision)

        result = plan.apply(collector=tamper)
        self.assertFalse(result.verification.passed)
        self.assertFalse(result.is_success)
        with self.assertRaises(RehearsalVerificationError):
            result.verification.raise_if_failed()

    def test_verification_detects_unexpected_value(self):
        plan = EditPlan(make_snapshot())
        plan.add_group(EditGroup("clear").clear("Root", "Clear"))

        def inject(snapshot):
            values = list(snapshot.values_at("Root"))
            values.append(RegistryEdit.ValueState(
                ValueIdentity("Root", "Injected"), Registry.RegSZ,
                b"x\x00", "x"))
            keys = [key for key in snapshot.keys()
                    if key.path != "Root"] + [
                RegistryEdit.KeyState("Root", values)]
            return HiveSnapshot("Root", keys, revision=snapshot.revision)

        result = plan.apply(collector=inject)
        self.assertFalse(result.verification.passed)
        self.assertTrue(any("unexpected" in mismatch.reason.lower()
                            for mismatch in result.verification.mismatches))


if __name__ == "__main__":
    unittest.main(verbosity=2)
