#!/usr/bin/python
# -*- coding: utf-8 -*-
import itertools
import unittest

from Registry import RegistryDiff
from Registry import RegistryName


def snapshot(kind, pairs):
    names = [pair[0] for pair in pairs]
    identities = [pair[1] for pair in pairs]
    return RegistryDiff.Snapshot.from_names(kind, names, identities=identities)


def intents(source, pairs):
    return [RegistryDiff.RenameIntent(source.get(identity), target)
            for target, identity in pairs]


class TestRegistryNameFormat(unittest.TestCase):
    def test_ascii_upcase_only_affects_ascii_letters(self):
        self.assertEqual(u"SOFTWARE\\MICROSOFT", RegistryName.ascii_upcase(
            u"Software\\Microsoft"))
        # Unicode lower/upper mappings must not apply to a compressed name.
        self.assertEqual(u"K", RegistryName.ascii_upcase(u"K"))
        self.assertEqual(u"ſ", RegistryName.ascii_upcase(u"ſ"))
        self.assertEqual(u"é", RegistryName.ascii_upcase(u"é"))

    def test_ascii_and_utf16_compare_differently(self):
        # KELVIN SIGN upcases to itself, so it never folds to K.  A plain
        # lower() would merge them; a source-format aware comparison must not.
        kelvin = RegistryName.lookup_key(u"K", "key", False)
        ascii_k = RegistryName.lookup_key(u"K", "key", True)
        self.assertNotEqual(kelvin, ascii_k)
        self.assertNotEqual(kelvin,
                            RegistryName.lookup_key(u"k", "key", False))

    def test_lower_is_not_the_comparison(self):
        # LONG S upcases to 'S' but lowercases to itself.  A lowercase-based
        # comparison would not merge it with 's'; the UTF-16 upcase does.
        long_s = RegistryName.lookup_key(u"ſ", "key", False)
        self.assertEqual(long_s, RegistryName.lookup_key(u"S", "key", False))
        self.assertNotEqual(long_s,
                            RegistryName.lookup_key(u"ſ", "key", True))

    def test_name_keeps_raw_bytes_and_format(self):
        name = RegistryName.Name(raw=b"A\x00B\x00", is_ascii=False,
                                 kind="key")
        self.assertEqual(u"AB", name.text)
        self.assertEqual(b"A\x00B\x00", name.raw)
        self.assertFalse(name.is_ascii)
        self.assertEqual(b"A\x00B\x00", name.raw)

    def test_default_value_is_one_key(self):
        default = RegistryName.Name.default_value()
        self.assertTrue(default.is_default())
        self.assertEqual(default.lookup_key(),
                         RegistryName.lookup_key(u"(default)", "value", True))
        self.assertEqual(default.lookup_key(),
                         RegistryName.lookup_key(u"", "value", False))
        self.assertNotEqual(default.lookup_key(),
                            RegistryName.lookup_key(u"x", "value", True))

    def test_query_matches_default_spellings(self):
        self.assertTrue(RegistryName.query_matches(
            u"", True, u"", "value", present=False))
        self.assertTrue(RegistryName.query_matches(
            u"", True, u"(default)", "value", present=False))
        self.assertFalse(RegistryName.query_matches(
            u"", True, u"other", "value", present=False))


class TestRegistryLookup(unittest.TestCase):
    def setUp(self):
        import os
        from Registry import Registry
        self.path = os.path.join(os.path.dirname(__file__), "reg_samples",
                                 "UNICODE_TESTS")
        self.registry = Registry.Registry(self.path)

    def test_key_name_bytes_are_exposed(self):
        root = self.registry.root()
        child = root.subkey("ASCII_KEY_NAME0")
        self.assertEqual(b"ASCII_KEY_NAME0", child.name_bytes())
        self.assertTrue(child.name_is_ascii())
        self.assertEqual(b"ASCII_KEY_NAME0", child.registry_name().raw)

    def test_value_name_bytes_are_exposed(self):
        key = self.registry.open("ASCII_KEY_NAME0")
        value = key.value("ASCII_VALUE_NAME0")
        self.assertEqual(b"ASCII_VALUE_NAME0", value.name_bytes())
        self.assertTrue(value.name_is_ascii())
        self.assertEqual(b"ASCII_VALUE_NAME0", value.registry_name().raw)

    def test_case_insensitive_lookup_still_works(self):
        key = self.registry.open("ASCII_KEY_NAME0")
        self.assertEqual(u"ASCII_VALUE_NAME0",
                         key.value("ascii_value_name0").name())


class TestFoldGroups(unittest.TestCase):
    def test_sibling_names_that_collapse_are_grouped(self):
        level = snapshot("key", [(u"a", "A"), (u"A", "B"), (u"unrelated", "C")])
        groups = level.fold_groups()
        self.assertEqual(1, len(groups))
        self.assertEqual(set(["A", "B"]), set(groups[0].identities))

    def test_different_queries_are_not_chained(self):
        # 'a' matches A, and 'c' matches C, but no single query matches all
        # three; a transitive closure would wrongly report one group.
        level = snapshot("key", [(u"a", "A"), (u"A", "B"), (u"c", "C")])
        groups = level.fold_groups()
        self.assertEqual(1, len(groups))
        self.assertNotIn("C", groups[0].identities)

    def test_resolve_reports_ambiguity(self):
        level = snapshot("key", [(u"a", "A"), (u"A", "B")])
        with self.assertRaises(RegistryDiff.AmbiguousQueryError):
            level.resolve("a")


class TestRenamePlan(unittest.TestCase):
    def test_cycle_uses_temporary_and_completes(self):
        level = snapshot("key", [(u"a", "A"), (u"b", "B")])
        plan = RegistryDiff.plan_renames(
            level, intents=level_intents(u"a", u"b"))
        self.assertEqual(0, plan.minimum_unbound_count())
        self.assertEqual({"A": u"b", "B": u"a"},
                         dict((identity, name.text) for identity, name
                              in plan.final_names().items()))
        self.assertTrue(any(step.temporary for step in plan.steps()))
        self.assertEqual(3, len(plan.steps()))

    def test_collision_unbinds_minimum_one(self):
        level = snapshot("key", [(u"a", "A"), (u"b", "B")])
        plan = RegistryDiff.plan_renames(level, intents=[
            RegistryDiff.RenameIntent(level.get("A"), u"c"),
            RegistryDiff.RenameIntent(level.get("B"), u"C"),
        ])
        self.assertEqual(1, plan.minimum_unbound_count())
        self.assertEqual(1, len(plan.conflict_groups()))
        self.assertTrue(
            "duplicate_target" in plan.conflict_groups()[0].to_dict()["reasons"])

    def test_unbound_entry_keeps_original_name_and_bytes(self):
        level = snapshot("key", [
            (RegistryName.Name(text=u"a", is_ascii=True), "A"),
            (RegistryName.Name(text=u"b", is_ascii=True), "B"),
        ])
        plan = RegistryDiff.plan_renames(level, intents=[
            RegistryDiff.RenameIntent(level.get("A"), u"c"),
            RegistryDiff.RenameIntent(level.get("B"), u"C"),
        ])
        accepted = set(intent.source_identity() for intent in plan.accepted)
        unbound = set(intent.source_identity() for intent in plan.unbound)
        self.assertEqual(set(["A", "B"]), accepted | unbound)
        for identity in unbound:
            self.assertEqual(level.get(identity).name.raw,
                             plan.final_names()[identity].raw)

    def test_non_participant_is_not_overwritten(self):
        level = snapshot("key", [(u"a", "A"), (u"keep", "K")])
        plan = RegistryDiff.plan_renames(level, intents=[
            RegistryDiff.RenameIntent(level.get("A"), u"keep"),
        ])
        self.assertEqual(1, plan.minimum_unbound_count())
        self.assertEqual(u"keep", plan.final_names()["K"].text)

    def test_destination_only_entry_is_a_blocker(self):
        level = snapshot("key", [(u"a", "A")])
        blocker = RegistryDiff.Entry("N", RegistryName.Name(text=u"new"))
        plan = RegistryDiff.plan_renames(
            level,
            intents=[RegistryDiff.RenameIntent(level.get("A"), u"new")],
            blockers=[blocker])
        self.assertEqual(1, plan.minimum_unbound_count())
        self.assertEqual(u"new", plan.final_names()["N"].text)

    def test_freeing_target_requires_chain_to_be_accepted(self):
        level = snapshot("key", [(u"a", "A"), (u"b", "B"), (u"c", "C")])
        plan = RegistryDiff.plan_renames(level, intents=[
            RegistryDiff.RenameIntent(level.get("A"), u"b"),
            RegistryDiff.RenameIntent(level.get("B"), u"c"),
        ])
        # b and c never become free, so neither rename can proceed.
        self.assertEqual(2, plan.minimum_unbound_count())

    def test_chain_to_free_name_is_accepted(self):
        level = snapshot("key", [(u"a", "A"), (u"b", "B"), (u"c", "C")])
        plan = RegistryDiff.plan_renames(level, intents=[
            RegistryDiff.RenameIntent(level.get("A"), u"b"),
            RegistryDiff.RenameIntent(level.get("B"), u"c"),
            RegistryDiff.RenameIntent(level.get("C"), u"z"),
        ])
        self.assertEqual(0, plan.minimum_unbound_count())
        self.assertEqual([u"z", u"c", u"b"],
                         [step.new_name.text for step in plan.steps()])

    def test_components_are_independently_applicable(self):
        level = snapshot("key", [(u"a", "A"), (u"b", "B"),
                                 (u"x", "X"), (u"y", "Y")])
        plan = RegistryDiff.plan_renames(level, intents=[
            RegistryDiff.RenameIntent(level.get("A"), u"b"),
            RegistryDiff.RenameIntent(level.get("B"), u"a"),
            RegistryDiff.RenameIntent(level.get("X"), u"z"),
        ])
        self.assertEqual(2, len(plan.components()))
        self.assertTrue(all(component.is_complete()
                            for component in plan.components()))

    def test_order_does_not_choose_the_winner(self):
        base = [(u"a", u"c", "A"), (u"b", u"C", "B"),
                (u"c", u"a", "C"), (u"x", u"y", "X")]
        results = set()
        for permutation in itertools.permutations(base):
            level = snapshot("key", [(old, identity)
                                     for old, _, identity in permutation])
            plan = RegistryDiff.plan_renames(level, intents=[
                RegistryDiff.RenameIntent(level.get(identity), new)
                for _, new, identity in permutation])
            results.add((
                tuple((intent.source_identity(), intent.new_name.text)
                      for intent in plan.accepted),
                tuple((intent.source_identity(), intent.new_name.text)
                      for intent in plan.unbound),
                tuple((step.entry_id, step.new_name.text, step.temporary)
                      for step in plan.steps()),
            ))
        self.assertEqual(1, len(results))

    def test_format_is_preserved_when_target_fits(self):
        level = snapshot("key", [
            (RegistryName.Name(text=u"old", is_ascii=True), "A")])
        plan = RegistryDiff.plan_renames(level, intents=[
            RegistryDiff.RenameIntent(level.get("A"), u"new")])
        self.assertTrue(plan.accepted[0].new_name.is_ascii)
        self.assertEqual(b"new", plan.accepted[0].new_name.raw)

    def test_unicode_target_uses_utf16(self):
        level = snapshot("key", [
            (RegistryName.Name(text=u"old", is_ascii=True), "A")])
        plan = RegistryDiff.plan_renames(level, intents=[
            RegistryDiff.RenameIntent(level.get("A"), u"中")])
        self.assertFalse(plan.accepted[0].new_name.is_ascii)
        self.assertEqual(u"中".encode("utf-16le"),
                         plan.accepted[0].new_name.raw)

    def test_mapping_matches_by_identity(self):
        source = snapshot("value", [(u"Alpha", "V")])
        destination = snapshot("value", [(u"Beta", "V")])
        mapping = RegistryDiff.build_rename_map(source, destination)
        self.assertEqual(1, len(mapping.intents))
        self.assertEqual(u"Alpha", mapping.intents[0].old_name.text)
        self.assertEqual(u"Beta", mapping.intents[0].new_name.text)

    def test_mapping_ignores_position(self):
        source = snapshot("value", [(u"one", "V1"), (u"two", "V2")])
        destination = snapshot("value", [(u"two", "V2"), (u"one", "V1")])
        mapping = RegistryDiff.build_rename_map(source, destination)
        self.assertEqual([], mapping.intents)

    def test_same_text_different_format_is_a_rename(self):
        source = snapshot("key", [
            (RegistryName.Name(text=u"abc", is_ascii=True), "A")])
        destination = snapshot("key", [
            (RegistryName.Name(text=u"abc", is_ascii=False), "A")])
        mapping = RegistryDiff.build_rename_map(source, destination)
        self.assertEqual(1, len(mapping.intents))
        self.assertEqual(b"abc", mapping.intents[0].old_name.raw)
        self.assertEqual(u"abc".encode("utf-16le"),
                         mapping.intents[0].new_name.raw)

    def test_large_duplicate_target_is_linear(self):
        pairs = [(u"n%d" % index, "I%d" % index) for index in range(40)]
        level = snapshot("key", pairs)
        plan = RegistryDiff.plan_renames(level, intents=[
            RegistryDiff.RenameIntent(level.get(identity), u"target")
            for _, identity in pairs])
        self.assertEqual(1, len(plan.accepted))
        self.assertEqual(39, plan.minimum_unbound_count())

    def test_long_rename_chain_scales(self):
        count = 60
        names = [u"n%03d" % index for index in range(count)]
        identities = ["I%03d" % index for index in range(count)]
        level = RegistryDiff.Snapshot.from_names("key", names,
                                                 identities=identities)
        intents = [
            RegistryDiff.RenameIntent(
                level.get(identities[index]),
                names[index + 1] if index + 1 < count else u"free")
            for index in range(count)]
        plan = RegistryDiff.plan_renames(level, intents=intents)
        self.assertEqual(count, len(plan.accepted))
        self.assertEqual(0, plan.minimum_unbound_count())


class TestSnapshotCapture(unittest.TestCase):
    def test_capture_keys_from_real_hive(self):
        import os
        from Registry import Registry
        path = os.path.join(os.path.dirname(__file__), "reg_samples",
                            "UNICODE_TESTS")
        root = Registry.Registry(path).root()
        level = RegistryDiff.Snapshot.from_key(root, kind="key")
        self.assertEqual(2, len(level))
        self.assertEqual(set([u"ASCII_KEY_NAME0",
                              u"UNICODE_JUMBLE_{H~░\xf4\xab}"]),
                         set(entry.name.text for entry in level))
        formats = dict((entry.name.text, entry.name.is_ascii)
                       for entry in level)
        self.assertTrue(formats[u"ASCII_KEY_NAME0"])
        self.assertFalse(formats[u"UNICODE_JUMBLE_{H~░\xf4\xab}"])

    def test_capture_values_from_real_hive(self):
        import os
        from Registry import Registry
        path = os.path.join(os.path.dirname(__file__), "reg_samples",
                            "issue22.hive")
        root = Registry.Registry(path).root()
        level = RegistryDiff.Snapshot.from_key(root, kind="value")
        names = [entry.name.text for entry in level]
        self.assertIn(u"TimeZoneKeyName", names)
        self.assertEqual(len(names), len(set(names)))

    def test_cross_snapshot_rename_from_captured_hive(self):
        import os
        from Registry import Registry
        path = os.path.join(os.path.dirname(__file__), "reg_samples",
                            "issue22.hive")
        root = Registry.Registry(path).root()
        source = RegistryDiff.Snapshot.from_key(root, kind="value")
        renamed = []
        for entry in source:
            renamed.append((entry.name.with_text(entry.name.text.upper()),
                            entry.identity))
        destination = RegistryDiff.Snapshot.from_names(
            "value", [item[0] for item in renamed],
            identities=[item[1] for item in renamed])
        plan = RegistryDiff.plan_from_mapping(
            RegistryDiff.build_rename_map(source, destination))
        self.assertEqual(len(source), len(plan.accepted))
        for intent in plan.accepted:
            self.assertEqual(intent.old_name.text.upper(),
                             intent.new_name.text)


def level_intents(first, second):
    level = snapshot("key", [(first, "A"), (second, "B")])
    return [RegistryDiff.RenameIntent(level.get("A"), second),
            RegistryDiff.RenameIntent(level.get("B"), first)]


if __name__ == "__main__":
    unittest.main(verbosity=2)
