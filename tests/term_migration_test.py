#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part of pymzml test cases

Tests for offline-pinned terminology import and the versioned migration
mapping that produces the unified term view.
"""

import sys
import os

sys.path.append(os.path.abspath("."))
import json
import tempfile
import unittest

import pymzml.run
from pymzml.obo import OboTranslator
from pymzml.term_migration import TermMigration, build_unified_view
from tests.test_file_paths import DATA_FOLDER


MIGRATION_PAYLOAD = {
    "version": "mapping-1.0.0",
    "terms": {
        # renamed between obo 3.25.0 and 4.1.x
        "MS:1000138": {
            "accession": "MS:1000138",
            "name": "normalized collision energy",
            "status": "renamed",
        },
        # deprecated without replacement
        "MS:1000048": {
            "accession": None,
            "name": None,
            "status": "deprecated",
        },
    },
}


class TestTermMigration(unittest.TestCase):
    def test_defaults(self):
        migration = TermMigration(terms={"MS:1000138": {"name": "new name"}})
        entry = migration.lookup("MS:1000138")
        self.assertEqual(entry["accession"], "MS:1000138")
        self.assertEqual(entry["name"], "new name")
        self.assertEqual(entry["status"], "renamed")
        self.assertIsNone(migration.version)
        self.assertIsNone(migration.lookup("MS:9999999"))

    def test_from_json_roundtrip(self):
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False
        ) as fh:
            json.dump(MIGRATION_PAYLOAD, fh)
            path = fh.name
        try:
            migration = TermMigration.load(path)
        finally:
            os.remove(path)
        self.assertEqual(migration.version, "mapping-1.0.0")
        self.assertEqual(len(migration), 2)
        entry = migration.lookup("MS:1000048")
        self.assertEqual(entry["status"], "deprecated")
        self.assertIsNone(entry["accession"])

    def test_load_normalization(self):
        self.assertIsNone(TermMigration.load(None))
        migration = TermMigration.load(MIGRATION_PAYLOAD)
        self.assertEqual(migration.version, "mapping-1.0.0")
        plain = TermMigration.load({"MS:1": {"name": "x"}})
        self.assertIsNone(plain.version)
        self.assertEqual(plain.lookup("MS:1")["name"], "x")
        same = TermMigration.load(migration)
        self.assertIs(same, migration)
        with self.assertRaises(TypeError):
            TermMigration.load(42)


class TestBuildUnifiedView(unittest.TestCase):
    def setUp(self):
        self.obo = OboTranslator.from_cache("3.25.0", allow_download=False)
        self.migration = TermMigration.load(MIGRATION_PAYLOAD)

    def test_renamed_deprecated_current_unknown(self):
        terms = [
            ("MS:1000138", "percent collision energy"),  # old label, in mapping
            ("MS:1000048", "gas"),  # deprecated, in mapping
            ("MS:1000016", "scan time"),  # unchanged, only in pinned obo
            ("MS:9999999", "mystery term"),  # unknown everywhere
        ]
        view = build_unified_view(
            terms, obo_translator=self.obo, migration=self.migration
        )
        self.assertEqual(
            [record["accession"] for record in view],
            sorted(record["accession"] for record in view),
        )
        by_acc = {record["accession"]: record for record in view}

        renamed = by_acc["MS:1000138"]
        self.assertEqual(renamed["status"], "renamed")
        self.assertEqual(renamed["name"], "percent collision energy")
        self.assertEqual(renamed["unified_name"], "normalized collision energy")
        self.assertEqual(renamed["unified_accession"], "MS:1000138")

        deprecated = by_acc["MS:1000048"]
        self.assertEqual(deprecated["status"], "deprecated")
        self.assertEqual(deprecated["name"], "gas")
        self.assertIsNone(deprecated["unified_accession"])
        self.assertIsNone(deprecated["unified_name"])

        current = by_acc["MS:1000016"]
        self.assertEqual(current["status"], "current")
        # name resolved against the pinned obo, old label retained
        self.assertEqual(current["name"], "scan time")
        self.assertEqual(current["unified_name"], "scan start time")

        unknown = by_acc["MS:9999999"]
        self.assertEqual(unknown["status"], "unknown")
        # original label is kept, never replaced by an online lookup
        self.assertEqual(unknown["unified_name"], "mystery term")
        self.assertEqual(unknown["unified_accession"], "MS:9999999")

        for record in view:
            self.assertEqual(record["obo_version"], "3.25.0")
            self.assertEqual(record["mapping_version"], "mapping-1.0.0")

    def test_reproducible_and_deduplicated(self):
        terms = [
            ("MS:1000016", "scan time"),
            ("MS:1000138", "percent collision energy"),
            ("MS:1000016", "scan time"),
        ]
        first = build_unified_view(
            terms, obo_translator=self.obo, migration=self.migration
        )
        second = build_unified_view(
            list(reversed(terms)),
            obo_translator=self.obo,
            migration=self.migration,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)

    def test_without_migration(self):
        view = build_unified_view(
            [("MS:1000016", "scan time")], obo_translator=self.obo
        )
        self.assertEqual(view[0]["status"], "current")
        self.assertIsNone(view[0]["mapping_version"])


class TestOfflineOboTranslator(unittest.TestCase):
    def test_missing_obo_raises_instead_of_downloading(self):
        translator = OboTranslator("9.99.99", allow_download=False)
        with self.assertRaises(FileNotFoundError):
            translator.parseOBO()

    def test_from_cache_distinguishes_download_flag(self):
        offline = OboTranslator.from_cache("3.25.0", allow_download=False)
        online = OboTranslator.from_cache("3.25.0", allow_download=True)
        self.assertIsNot(offline, online)
        self.assertIs(
            offline, OboTranslator.from_cache("3.25.0", allow_download=False)
        )
        self.assertFalse(offline.allow_download)

    def test_offline_translator_resolves_pinned_version(self):
        translator = OboTranslator.from_cache("3.25.0", allow_download=False)
        self.assertEqual(translator["MS:1000138"], "percent collision energy")


class TestOfflineReader(unittest.TestCase):
    def setUp(self):
        self.example = os.path.join(DATA_FOLDER, "example.mzML")
        self.no_obo_version = os.path.join(
            DATA_FOLDER, "example_no_obo_version.mzML"
        )

    def test_offline_pins_declared_obo_version(self):
        reader = pymzml.run.Reader(self.example, offline=True)
        self.assertEqual(reader.info["obo_version"], "3.25.0")
        self.assertFalse(reader.OT.allow_download)
        reader.close()

    def test_offline_missing_declared_version_raises(self):
        with self.assertRaises(ValueError):
            pymzml.run.Reader(self.no_obo_version, offline=True)

    def test_offline_unavailable_obo_version_raises(self):
        with self.assertRaises(FileNotFoundError):
            pymzml.run.Reader(self.example, obo_version="9.9.9", offline=True)

    def test_online_mode_still_falls_back(self):
        reader = pymzml.run.Reader(self.example, obo_version="9.9.9")
        self.assertEqual(reader.info["obo_version"], "4.1.79")
        reader.close()

    def test_unified_terms_view_and_reimport(self):
        def collect():
            reader = pymzml.run.Reader(
                self.example, offline=True, term_migration=MIGRATION_PAYLOAD
            )
            views = []
            for spectrum in reader:
                views.append(spectrum.unified_terms())
                if len(views) == 3:
                    break
            reader.close()
            return views

        first_pass = collect()
        second_pass = collect()
        # re-importing the same partition reproduces the identical view
        self.assertEqual(first_pass, second_pass)

        for view in first_pass:
            for record in view:
                # original accession and label are always retained
                self.assertTrue(record["accession"].startswith("MS:"))
                self.assertIsNotNone(record["name"])
                self.assertEqual(record["obo_version"], "3.25.0")
                self.assertEqual(record["mapping_version"], "mapping-1.0.0")
                self.assertIn(
                    record["status"],
                    ("renamed", "deprecated", "current", "unknown"),
                )

    def test_unified_terms_without_migration(self):
        reader = pymzml.run.Reader(self.example, offline=True)
        spectrum = next(reader)
        view = spectrum.unified_terms()
        self.assertTrue(len(view) > 0)
        for record in view:
            self.assertIsNone(record["mapping_version"])
        reader.close()


if __name__ == "__main__":
    unittest.main(verbosity=3)
