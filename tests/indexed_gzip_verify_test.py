"""
Offline consistency tests for indexed gzip mzML random access.

All fixtures are generated in a temporary directory; no network and no
pre-existing test data is required.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import indexed_gzip_fixtures as fixtures
from pymzml.utils.verify import (
    compare_indexed_gzip,
    verify_indexed_gzip,
)
from pymzml.utils.GSGR import GSGR


def redirect_header_key(path, key, to_offset):
    """Point a GSGW header index key at another block (header stays readable)."""
    fixtures.tamper_header_offsets(path, {key: to_offset})
    return path


def make_spectra(count=5, id_template="scan={0}"):
    spectra = []
    for position in range(1, count + 1):
        spectra.append(
            {
                "key": position,
                "xml": fixtures.spectrum_xml(
                    position - 1,
                    id_template.format(position),
                    [100.0 + position + offset for offset in range(4)],
                    [10.0 * position + offset for offset in range(4)],
                ),
            }
        )
    return spectra


def make_document(spectra, entries=None):
    if entries is None:
        entries = [
            ("scan={0}".format(spectrum["key"]), {"key": spectrum["key"]})
            for spectrum in spectra
        ]
    return fixtures.build_document(spectra, entries)


class IndexedGzipVerificationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = self.tmp.name

    def path(self, name):
        return os.path.join(self.directory, name)

    def write_clean_indexed(self, name="clean.idx.gz", spectra=None, count=5):
        spectra = spectra or make_spectra(count)
        document, offsets = make_document(spectra)
        target = self.path(name)
        fixtures.write_indexed_gzip(
            target, fixtures.spectrum_chunks(document, spectra)
        )
        return target, spectra, document, offsets

    # -- happy path ---------------------------------------------------------

    def test_clean_indexed_gzip_is_consistent(self):
        target, _spectra, _document, _offsets = self.write_clean_indexed()
        report = verify_indexed_gzip(target)
        self.assertTrue(report.ok, report.format())
        self.assertEqual(report.summary["document_items"], 5)
        self.assertEqual(report.summary["gzip_members"], 7)
        self.assertIn(1, report.summary["query_ids"])

    def test_clean_report_is_json_serialisable(self):
        target, _spectra, _document, _offsets = self.write_clean_indexed()
        report = verify_indexed_gzip(target)
        import json

        payload = json.dumps(report.to_dict())
        self.assertIn('"findings"', payload)

    # -- duplicate numeric fragments ----------------------------------------

    def test_duplicate_trailing_numbers_are_shadowed_not_hits(self):
        spectra = [
            {
                "key": 1,
                "xml": fixtures.spectrum_xml(
                    0, "sample=1 spectrum=1", [100.0, 101.0], [10.0, 11.0]
                ),
            },
            {
                "key": 2,
                "xml": fixtures.spectrum_xml(
                    1, "sample=2 spectrum=1", [200.0, 201.0], [20.0, 21.0]
                ),
            },
            {
                "key": 3,
                "xml": fixtures.spectrum_xml(
                    2, "sample=3 spectrum=3", [300.0, 301.0], [30.0, 31.0]
                ),
            },
        ]
        entries = [
            ("sample=1 spectrum=1", {"key": 1}),
            ("sample=2 spectrum=1", {"key": 2}),
            ("sample=3 spectrum=3", {"key": 3}),
        ]
        document, _ = make_document(spectra, entries)
        target = self.path("duplicates.idx.gz")
        fixtures.write_indexed_gzip(
            target, fixtures.spectrum_chunks(document, spectra)
        )

        report = verify_indexed_gzip(target)
        self.assertIn("identity_trailing_collision", report.codes())
        collision = next(
            finding
            for finding in report.findings
            if finding.code == "identity_trailing_collision"
        )
        self.assertEqual(collision.index, 1)
        self.assertEqual(
            set(collision.identity["ids"]),
            {"sample=1 spectrum=1", "sample=2 spectrum=1"},
        )
        self.assertEqual(
            sorted(collision.details["index_keys"]),
            ["sample=1 spectrum=1", "sample=2 spectrum=1"],
        )

    def test_duplicate_native_keys_are_errors_with_both_offsets(self):
        spectra = make_spectra(3)
        entries = [
            ("scan=1", {"key": 1}),
            ("scan=1", {"key": 2}),
            ("scan=3", {"key": 3}),
        ]
        document, offsets = make_document(spectra, entries)
        target = self.path("dup-key.idx.gz")
        fixtures.write_indexed_gzip(
            target, fixtures.spectrum_chunks(document, spectra)
        )
        report = verify_indexed_gzip(target)
        duplicate = next(
            finding
            for finding in report.findings
            if finding.code == "native_index_duplicate_key"
        )
        self.assertEqual(duplicate.index, 1)
        self.assertEqual(
            duplicate.details["offsets"], [offsets[1], offsets[2]]
        )
        self.assertIn("affected_range", duplicate.details)

    def test_native_index_key_pointing_at_another_element_is_error(self):
        """Readable index, readable file, wrong native target."""
        spectra = make_spectra(4)
        document, offsets = make_document(spectra)
        entries = [
            ("scan=1", {"key": 1}),
            ("scan=2", {"key": 2, "offset": offsets[4]}),
            ("scan=3", {"key": 3}),
            ("scan=4", {"key": 4}),
        ]
        document, _ = make_document(spectra, entries)
        target = self.path("native-wrong-target.idx.gz")
        fixtures.write_indexed_gzip(
            target, fixtures.spectrum_chunks(document, spectra)
        )

        report = verify_indexed_gzip(target)
        finding = next(
            finding
            for finding in report.findings
            if finding.code == "native_index_identity_mismatch"
        )
        self.assertEqual(finding.index, 2)
        self.assertEqual(finding.identity["id"], "scan=4")
        self.assertEqual(finding.details["id_ref"], "scan=2")
        self.assertEqual(finding.details["target_index"], 2)

    def test_native_index_offset_inside_element_is_located(self):
        spectra = make_spectra(3)
        document, offsets = make_document(spectra)
        entries = [
            ("scan=1", {"key": 1}),
            ("scan=2", {"key": 2, "offset": offsets[3] + 17}),
            ("scan=3", {"key": 3}),
        ]
        document, _ = make_document(spectra, entries)
        target = self.path("native-misaligned.idx.gz")
        fixtures.write_indexed_gzip(
            target, fixtures.spectrum_chunks(document, spectra)
        )

        report = verify_indexed_gzip(target)
        finding = next(
            finding
            for finding in report.findings
            if finding.code == "native_index_offset_misaligned"
        )
        self.assertEqual(finding.index, 2)
        self.assertEqual(finding.details["delta"], 17)
        self.assertEqual(finding.details["affected_range"], [2, 3])
        self.assertEqual(
            finding.boundary["nearest_element_start"], offsets[3]
        )
        self.assertEqual(finding.identity["id"], "scan=3")

    # -- compression block boundaries --------------------------------------

    def test_split_array_reports_index_and_both_members(self):
        spectra = make_spectra(5)
        document, offsets = make_document(spectra)
        target = self.path("split.idx.gz")
        fixtures.write_indexed_gzip(
            target, fixtures.split_array_chunks(document, spectra, 2)
        )
        report = verify_indexed_gzip(target)
        self.assertIn("element_spans_members", report.codes())
        self.assertIn("array_spans_members", report.codes())

        element = next(
            finding
            for finding in report.findings
            if finding.code == "element_spans_members"
        )
        self.assertEqual(element.index, 2)
        self.assertEqual(element.details["target_index"], 2)
        self.assertEqual(len(element.boundary["members"]), 2)
        self.assertEqual(element.boundary["start"], offsets[2])
        self.assertEqual(
            element.details["array_members"],
            {0: element.boundary["members"], 1: [element.boundary["members"][-1]]},
        )

        array = next(
            finding
            for finding in report.findings
            if finding.code == "array_spans_members"
        )
        self.assertEqual(array.arrays, ("m/z array",))
        self.assertEqual(array.details["members"], element.boundary["members"])

    def test_split_array_is_reported_by_random_access(self):
        spectra = make_spectra(5)
        document, _offsets = make_document(spectra)
        target = self.path("split-query.idx.gz")
        fixtures.write_indexed_gzip(
            target, fixtures.split_array_chunks(document, spectra, 2)
        )
        report = verify_indexed_gzip(target, query_ids=[2])
        raised = next(
            finding
            for finding in report.findings
            if finding.code == "random_access_raised"
        )
        self.assertEqual(raised.index, 2)
        self.assertIn("ParseError", raised.details["exception"])
        self.assertEqual(raised.details["target_index"], 2)

    # -- readable but wrong index ------------------------------------------

    def test_bad_offset_pointing_at_another_spectrum_is_reported(self):
        spectra = make_spectra(4)
        document, offsets = make_document(
            spectra,
            [
                ("scan=1", {"key": 1}),
                ("scan=2", {"key": 2}),
                ("scan=3", {"key": 3}),
                ("scan=4", {"key": 4}),
            ],
        )
        target = self.path("wrong-target.idx.gz")
        fixtures.write_indexed_gzip(
            target, fixtures.spectrum_chunks(document, spectra)
        )
        index = dict(GSGR(target).index)
        redirect_header_key(target, 3, index[2])

        report = verify_indexed_gzip(target, query_ids=[3])
        self.assertIn("header_index_wrong_element", report.codes())
        mismatch = next(
            finding
            for finding in report.findings
            if finding.code == "header_index_wrong_element"
        )
        self.assertEqual(mismatch.index, 3)
        self.assertEqual(mismatch.identity["id"], "scan=2")

        wrong = next(
            finding
            for finding in report.findings
            if finding.code == "random_access_wrong_element"
        )
        self.assertEqual(wrong.index, 3)
        self.assertEqual(wrong.identity["returned_id"], "scan=2")
        self.assertEqual(wrong.details["actual_target"]["key"], 2)
        for name, evidence in wrong.details["array_evidence"].items():
            self.assertTrue(evidence["lengths_equal"], name)
            self.assertFalse(evidence["values_equal"], name)

    def test_equal_length_arrays_are_not_a_match(self):
        spectra = make_spectra(3)
        document, offsets = make_document(spectra)
        # Make every array the same length (the fixture already is), then
        # shift only the values of scan=2 so lengths still agree.
        swapped = []
        for position, spectrum in enumerate(spectra, start=1):
            if position == 2:
                spectrum = {
                    "key": 2,
                    "xml": fixtures.spectrum_xml(
                        1,
                        "scan=2",
                        [900.0, 901.0, 902.0, 903.0],
                        [70.0, 71.0, 72.0, 73.0],
                    ),
                }
            swapped.append(spectrum)
        document, offsets = make_document(swapped)
        target = self.path("equal-length.idx.gz")
        fixtures.write_indexed_gzip(
            target, fixtures.spectrum_chunks(document, swapped)
        )
        index = dict(GSGR(target).index)
        redirect_header_key(target, 2, index[1])

        report = verify_indexed_gzip(target, query_ids=[2])
        mismatch = next(
            finding
            for finding in report.findings
            if finding.code == "random_access_wrong_element"
        )
        self.assertEqual(mismatch.index, 2)
        for evidence in mismatch.details["array_evidence"].values():
            self.assertTrue(evidence["lengths_equal"])
            self.assertFalse(evidence["values_equal"])
            self.assertNotEqual(
                evidence["expected_digest"], evidence["returned_digest"]
            )
        self.assertTrue(mismatch.arrays)

    # -- interleaved / replaced input --------------------------------------

    def test_interleaved_wrong_target_is_located(self):
        spectra = make_spectra(5)
        document, offsets = make_document(spectra)
        target = self.path("interleaved.idx.gz")
        fixtures.write_indexed_gzip(
            target, fixtures.spectrum_chunks(document, spectra)
        )
        index = dict(GSGR(target).index)
        redirect_header_key(target, 2, index[1])
        report = verify_indexed_gzip(target, query_ids=[2])
        finding = next(
            finding
            for finding in report.findings
            if finding.code == "interleaved_wrong_element"
        )
        self.assertEqual(finding.index, 2)
        self.assertEqual(finding.details["actual_target"]["key"], 1)
        self.assertEqual(finding.details["affected_range"], [2, 1])

    def test_replaced_input_reports_changed_actual_targets(self):
        reference_spectra = make_spectra(4)
        reference_document, _ = make_document(reference_spectra)
        reference = self.path("reference.idx.gz")
        fixtures.write_indexed_gzip(
            reference,
            fixtures.spectrum_chunks(reference_document, reference_spectra),
        )

        replacement_spectra = []
        for position, spectrum in enumerate(reference_spectra, start=1):
            replacement_spectra.append(
                {
                    "key": position,
                    "xml": fixtures.spectrum_xml(
                        position - 1,
                        "replacement-{0}".format(position),
                        [500.0 + position, 501.0 + position],
                        [900.0 + position, 901.0 + position],
                    ),
                }
            )
        replacement_document, _ = make_document(replacement_spectra)
        replacement = self.path("replacement.idx.gz")
        fixtures.write_indexed_gzip(
            replacement,
            fixtures.spectrum_chunks(replacement_document, replacement_spectra),
        )

        report = compare_indexed_gzip(reference, replacement)
        self.assertFalse(report.ok)
        changed = [
            finding
            for finding in report.findings
            if finding.code == "input_native_index_target_changed"
        ]
        self.assertEqual(len(changed), 4)
        self.assertEqual(
            {finding.index for finding in changed}, {1, 2, 3, 4}
        )
        first = changed[0]
        self.assertEqual(first.identity["reference_id"], "scan=1")
        self.assertEqual(first.identity["candidate_id"], "replacement-1")
        self.assertEqual(first.details["actual_target"]["id"], "replacement-1")
        self.assertEqual(report.summary["changed_keys"], [1, 2, 3, 4])
        self.assertEqual(report.summary["affected_range"], [1, 4])

    def test_reopened_input_still_reports_wrong_target(self):
        spectra = make_spectra(5)
        document, offsets = make_document(spectra)
        target = self.path("reopen.idx.gz")
        fixtures.write_indexed_gzip(
            target, fixtures.spectrum_chunks(document, spectra)
        )
        index = dict(GSGR(target).index)
        redirect_header_key(target, 2, index[1])
        report = verify_indexed_gzip(target, query_ids=[2])
        reopen = next(
            finding
            for finding in report.findings
            if finding.code == "reopen_wrong_element"
        )
        self.assertEqual(reopen.index, 2)
        self.assertEqual(reopen.details["returned_id"], "scan=1")
        self.assertEqual(reopen.details["target_index"], 2)

    def test_report_groups_findings_and_exposes_affected_range(self):
        spectra = make_spectra(4)
        document, offsets = make_document(spectra)
        target = self.path("grouped.idx.gz")
        fixtures.write_indexed_gzip(
            target, fixtures.spectrum_chunks(document, spectra)
        )
        index = dict(GSGR(target).index)
        redirect_header_key(target, 2, index[1])
        report = verify_indexed_gzip(target, query_ids=[2])
        explanation = report.explain()
        self.assertIn("index 2", explanation)
        self.assertIn("affected_range", explanation)
        selected = report.for_index(2)
        self.assertTrue(selected)
        self.assertTrue(
            any(finding.details.get("actual_target") for finding in selected)
        )

    def test_unindexed_gzip_skips_random_access_cleanly(self):
        spectra = make_spectra(3)
        document, _offsets = fixtures.build_document(spectra, None)
        target = self.path("plain.mzML.gz")
        import gzip

        with gzip.open(target, "wb") as handle:
            handle.write(document.encode("iso-8859-1"))

        report = verify_indexed_gzip(target)
        self.assertTrue(report.ok, report.format())
        self.assertIn("random_access_unsupported_input", report.codes())
        self.assertFalse(report.summary["indexed"])

    # -- container integrity ------------------------------------------------

    def test_truncated_member_is_located(self):
        spectra = make_spectra(4)
        document, _ = make_document(spectra)
        target = self.path("truncated.idx.gz")
        fixtures.write_indexed_gzip(
            target, fixtures.spectrum_chunks(document, spectra)
        )
        with open(target, "rb") as handle:
            data = handle.read()
        with open(target, "wb") as handle:
            handle.write(data[:-5])

        report = verify_indexed_gzip(target)
        self.assertIn("gzip_member_unterminated", report.codes())
        finding = next(
            finding
            for finding in report.findings
            if finding.code == "gzip_member_unterminated"
        )
        self.assertIsNotNone(finding.details.get("member"))
        self.assertEqual(finding.severity, "error")


if __name__ == "__main__":
    unittest.main(verbosity=2)
