#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import unittest

from Registry import Expand, Registry


SYSTEM_HIVE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "testing", "reg_samples", "new_log_1", "SYSTEM")

ENVIRONMENT_KEY = (
    "ControlSet001\\Control\\Session Manager\\Environment")


class TestTargetEnvironmentExpansion(unittest.TestCase):
    def test_raw_and_effective_are_separate(self):
        raw = "%USERPROFILE%\\Desktop"
        environment = {"USERPROFILE": "C:\\Users\\target"}

        result = Expand.expand_environment_strings(raw, environment)

        self.assertEqual(result.raw, raw)
        self.assertEqual(result.effective, "C:\\Users\\target\\Desktop")
        self.assertEqual(str(result), "C:\\Users\\target\\Desktop")
        self.assertEqual(result.raw_value, raw)
        self.assertEqual(result.status, Expand.STATUS_RESOLVED)

    def test_does_not_use_acquisition_host_environment(self):
        marker = object()
        previous = os.environ.get("USERPROFILE", marker)
        os.environ["USERPROFILE"] = "C:\\Users\\acquisition"
        try:
            result = Expand.expand_environment_strings("%USERPROFILE%")
        finally:
            if previous is marker:
                del os.environ["USERPROFILE"]
            else:
                os.environ["USERPROFILE"] = previous

        self.assertEqual(result.effective, "%USERPROFILE%")
        self.assertEqual(result.unresolved, ["USERPROFILE"])
        self.assertTrue(result.has_unresolved)

    def test_unbound_variables_remain_pending(self):
        result = Expand.expand_environment_strings(
            "%KNOWN%\\%UNBOUND%", {"KNOWN": "value"})

        self.assertEqual(result.effective, "value\\%UNBOUND%")
        self.assertEqual(result.raw, "%KNOWN%\\%UNBOUND%")
        self.assertEqual(result.pending, ["UNBOUND"])
        self.assertEqual(result.status, Expand.STATUS_UNRESOLVED)

    def test_nested_variable_references(self):
        environment = {
            "PROFILE_ROOT": "C:\\Users",
            "USERPROFILE": "%PROFILE_ROOT%\\target",
            "DESKTOP": "%USERPROFILE%\\Desktop",
        }

        result = Expand.expand_environment_strings("%DESKTOP%", environment)

        self.assertEqual(result.effective, "C:\\Users\\target\\Desktop")
        self.assertEqual(result.status, Expand.STATUS_RESOLVED)
        self.assertEqual([record.name for record in result.provenance],
                         ["DESKTOP"])
        self.assertEqual([record.name for record in result.trace],
                         ["DESKTOP", "USERPROFILE", "PROFILE_ROOT"])
        self.assertEqual(result.trace[-1].chain,
                         ("DESKTOP", "USERPROFILE", "PROFILE_ROOT"))

    def test_self_reference_cycle_terminates(self):
        result = Expand.expand_environment_strings(
            "%A%", {"A": "%A%"})

        self.assertEqual(result.effective, "%A%")
        self.assertEqual(result.cycles, ["A"])
        self.assertEqual(result.status, Expand.STATUS_CYCLE)

    def test_mutual_reference_cycle_terminates(self):
        result = Expand.expand_environment_strings(
            "%A%", {"A": "%B%", "B": "%A%"})

        self.assertEqual(result.effective, "%A%")
        self.assertEqual(result.cycles, ["A", "B"])
        self.assertEqual(result.status, Expand.STATUS_CYCLE)

    def test_depth_limit_terminates(self):
        environment = {
            "A": "%B%",
            "B": "%C%",
            "C": "%D%",
            "D": "%E%",
            "E": "end",
        }

        result = Expand.expand_environment_strings(
            "%A%", environment, max_depth=2)

        self.assertEqual(result.status, Expand.STATUS_DEPTH_LIMIT)
        self.assertEqual(result.depth_limited, ["D"])
        self.assertEqual(result.effective, "%D%")

    def test_equal_text_from_different_variables_stays_separate(self):
        result = Expand.expand_environment_strings(
            "%ONE%\\%TWO%", {"ONE": "same", "TWO": "same"})

        self.assertEqual(result.effective, "same\\same")
        self.assertEqual(len(result.provenance), 2)
        self.assertEqual([record.name for record in result.provenance],
                         ["ONE", "TWO"])
        self.assertEqual([record.source for record in result.provenance],
                         ["%ONE%", "%TWO%"])
        self.assertEqual(len(result.source_map()["same"]), 2)
        self.assertEqual(
            [record.name for record in result.sources_for("same")],
            ["ONE", "TWO"])

    def test_repeated_variable_occurrences_stay_separate(self):
        result = Expand.expand_environment_strings(
            "%ROOT%\\a\\%ROOT%\\b", {"ROOT": "C:\\root"})

        self.assertEqual(result.effective, "C:\\root\\a\\C:\\root\\b")
        self.assertEqual(len(result.provenance), 2)
        self.assertEqual([record.start for record in result.provenance],
                         [0, 10])

    def test_variable_names_are_case_insensitive(self):
        result = Expand.expand_environment_strings(
            "%SystemRoot%\\%SYSTEMROOT%", {"SYSTEMROOT": "C:\\Windows"})

        self.assertEqual(result.effective, "C:\\Windows\\C:\\Windows")
        self.assertEqual(len(result.provenance), 2)
        self.assertEqual([record.name for record in result.provenance],
                         ["SystemRoot", "SYSTEMROOT"])


class TestRegistryValueExpansion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = Registry.Registry(SYSTEM_HIVE)
        cls.key = cls.registry.open(ENVIRONMENT_KEY)

    def test_value_returns_stored_string(self):
        path = self.key.value("Path")

        self.assertEqual(
            path.value(),
            "%SystemRoot%\\system32;%SystemRoot%;"
            "%SystemRoot%\\System32\\Wbem;"
            "%SYSTEMROOT%\\System32\\WindowsPowerShell\\v1.0\\")
        self.assertEqual(path.raw_value(), path.value())

    def test_expand_value_uses_target_environment(self):
        path = self.key.value("Path")

        result = path.expand({"SystemRoot": "D:\\TARGET"})

        self.assertEqual(
            result.effective,
            "D:\\TARGET\\system32;D:\\TARGET;"
            "D:\\TARGET\\System32\\Wbem;"
            "D:\\TARGET\\System32\\WindowsPowerShell\\v1.0\\")
        self.assertIn("%SystemRoot%", result.raw)
        self.assertEqual(len(result.provenance), 4)

    def test_environment_read_from_key_keeps_raw_values(self):
        environment = self.key.environment()

        self.assertIn("windir", environment)
        self.assertIn("Path", environment)
        self.assertEqual(environment.get("WINDIR"), "%SystemRoot%")

    def test_expand_against_key_environment(self):
        environment = self.key.environment()

        comspec = self.key.value("ComSpec").expand(environment)

        self.assertEqual(comspec.raw, "%SystemRoot%\\system32\\cmd.exe")
        self.assertEqual(comspec.effective,
                         "%SystemRoot%\\system32\\cmd.exe")
        self.assertEqual(comspec.unresolved, ["SystemRoot"])

    def test_non_expand_sz_value_stays_literal(self):
        result = self.key.value("OS").expand({"OS": "replaced"})

        self.assertEqual(result.raw, "Windows_NT")
        self.assertEqual(result.effective, "Windows_NT")
        self.assertEqual(result.status, Expand.STATUS_NOT_EXPANDABLE)
        self.assertEqual(result.provenance, [])


    def test_binary_value_passes_through_unchanged(self):
        value = self.registry.open(
            "ControlSet001\\Control\\ProductOptions").value("ProductPolicy")

        result = value.expand({"A": "B"})

        self.assertIsInstance(result, bytes)
        self.assertEqual(result, value.raw_value())


    def test_key_expand_values_only_includes_expand_sz(self):
        results = self.key.expand_values({"SystemRoot": "C:\\Windows"})

        self.assertIn("Path", results)
        self.assertIn("ComSpec", results)
        self.assertNotIn("OS", results)
        for name, result in results.items():
            self.assertEqual(result.raw, self.key.value(name).raw_value())


class TestEnvironmentSources(unittest.TestCase):
    def test_mapping_pairs_and_name_value_strings(self):
        self.assertEqual(
            Expand.expand_environment_strings(
                "%A%/%B%", {"A": "1", "B": "2"}).effective,
            "1/2")
        self.assertEqual(
            Expand.expand_environment_strings(
                "%A%/%B%", [("A", "1"), ("B", "2")]).effective,
            "1/2")
        self.assertEqual(
            Expand.expand_environment_strings(
                "%A%/%B%", ["A=1", "B=2"]).effective,
            "1/2")
        self.assertEqual(
            Expand.expand_environment_strings(
                "%A%/%B%", "A=1", max_depth=32).effective,
            "1/%B%")

    def test_target_environment_is_case_insensitive(self):
        environment = Expand.TargetEnvironment({"SystemRoot": "C:\\Windows"})
        self.assertIn("systemroot", environment)
        self.assertEqual(environment.get("SYSTEMROOT"), "C:\\Windows")

    def test_environment_extended_after_expansion(self):
        result = Expand.expand_environment_strings(
            "%A%\\%B%", {"A": "one"})

        self.assertEqual(result.effective, "one\\%B%")
        self.assertEqual(result.pending, ["B"])

        completed = result.resolve({"B": "two"})
        self.assertEqual(completed.effective, "one\\two")
        self.assertEqual(completed.pending, [])
        self.assertEqual(result.effective, "one\\%B%")

    def test_none_entry_marks_variable_as_pending_not_bound(self):
        result = Expand.expand_environment_strings(
            "%USERPROFILE%\\%SystemRoot%",
            {"USERPROFILE": None, "SystemRoot": "C:\\Windows"})

        self.assertEqual(result.effective,
                         "%USERPROFILE%\\C:\\Windows")
        self.assertEqual(result.unresolved, ["USERPROFILE"])
        self.assertEqual(result.pending, ["USERPROFILE"])

    def test_serializable_explanation(self):
        result = Expand.expand_environment_strings(
            "%ONE%\\%TWO%", {"ONE": "same", "TWO": "same"})

        explanation = result.explain()
        self.assertEqual(explanation["raw"], "%ONE%\\%TWO%")
        self.assertEqual(explanation["effective"], "same\\same")
        self.assertEqual(len(explanation["provenance"]), 2)
        self.assertEqual(explanation["provenance"][0]["name"], "ONE")
        self.assertEqual(explanation["provenance"][1]["name"], "TWO")


if __name__ == "__main__":
    unittest.main(verbosity=2)
