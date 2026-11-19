#!/usr/bin/python
# -*- coding: utf-8 -*-
import json
import os
import unittest

from Registry import Registry, RegistryExport


FIXTURE = os.path.join(os.path.dirname(__file__), "reg_samples",
                       "FORENSIC_EXPORT_TESTS")


class TestConfigExport(unittest.TestCase):
    def setUp(self):
        self.report = RegistryExport.classify(FIXTURE)

    def test_active_values_match_high_level_api(self):
        registry = Registry.Registry(FIXTURE)
        expected = []
        for key in registry.root().subkeys():
            for value in key.values():
                expected.append((key.path(), value.name()))
        actual = [(item["path"], item["value_name"])
                  for item in self.report.active]
        self.assertEqual(sorted(expected), sorted(actual))

    def test_residual_is_not_active(self):
        residual = [item for item in self.report.residual
                    if item["value_name"] == "ASCII_VALUE_NAME0"]
        self.assertEqual(len(residual), 1)
        item = residual[0]
        self.assertEqual(item["value"], "C:\\OldInstall\\OldApp")
        self.assertEqual(item["location"]["cell_state"], "free")
        self.assertFalse(item["root_reachable"])
        self.assertNotIn(item, self.report.active)

    def test_same_path_is_flagged_not_merged(self):
        self.assertEqual(len(self.report.collisions), 1)
        collision = self.report.collisions[0]
        self.assertEqual(collision["path"],
                         "UNICODE_TESTS\\ASCII_KEY_NAME0")
        self.assertEqual(collision["value_name"], "ASCII_VALUE_NAME0")
        self.assertEqual(sorted(collision["states"]), ["active", "residual"])
        self.assertEqual(len(collision["active_identities"]), 1)
        self.assertEqual(len(collision["residual_identities"]), 1)

    def test_broken_parent_is_uncertain_not_residual(self):
        item = [item for item in self.report.uncertain
                if item["value_name"] == "OldBrokenValue"][0]
        self.assertEqual(item["state"], "uncertain")
        self.assertEqual(item["path_status"], "broken")
        self.assertFalse(item["root_reachable"])
        self.assertTrue(any(evidence["type"] == "parent_chain_broken"
                            for evidence in item["evidence"]))
        self.assertEqual([x for x in self.report.residual
                          if x["value_name"] == "OldBrokenValue"], [])

    def test_allocated_unreferenced_value_is_uncertain(self):
        item = [item for item in self.report.uncertain
                if item["value_name"] == "UnrefValue"][0]
        self.assertEqual(item["location"]["cell_state"], "allocated")
        self.assertEqual(item["path_status"], "unknown")
        self.assertFalse(item["root_reachable"])
        # It is explicitly not claimed to be deleted or residual.
        self.assertEqual([x for x in self.report.residual
                          if x["value_name"] == "UnrefValue"], [])

    def test_every_row_has_evidence_and_location(self):
        for item in (self.report.active + self.report.residual +
                     self.report.uncertain):
            self.assertTrue(item["evidence"])
            self.assertIsNotNone(item["location"]["cell_offset_hex"])
            self.assertIsNotNone(item["location"]["file_offset_hex"])
            self.assertIn("data", item["location"])
            types = [evidence["type"] for evidence in item["evidence"]]
            self.assertIn("record", types)
            self.assertIn("allocation", types)
            self.assertIn("root_reachability", types)

    def test_json_is_serializable(self):
        payload = json.loads(self.report.to_json())
        self.assertEqual(payload["schema"], RegistryExport.SCHEMA)
        self.assertEqual(payload["summary"]["collisions"], 1)
        self.assertEqual(len(payload["active"]), 22)

    def test_search_filters_without_changing_classification(self):
        report = RegistryExport.classify(FIXTURE, needle="OldInstall")
        self.assertEqual(len(report.active), 0)
        self.assertEqual(len(report.residual), 1)
        self.assertEqual(len(report.uncertain), 0)
        self.assertEqual(len(report.collisions), 1)

    def test_clean_hive_has_no_residual(self):
        path = os.path.join(os.path.dirname(__file__), "reg_samples",
                            "issue22.hive")
        report = RegistryExport.classify(path)
        self.assertEqual(len(report.active), 10)
        self.assertEqual(report.residual, [])
        self.assertEqual(report.uncertain, [])
        self.assertEqual(report.collisions, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
