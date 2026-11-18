import base64
import struct
import unittest
import xml.etree.ElementTree as ElementTree

import numpy as np

from pymzml.run import Reader
from pymzml.search3d import (
    CalibrationError,
    Hit3D,
    MobilityCalibration,
    MobilityCalibrationSegment,
    Peak3D,
    PeakIndex3D,
    PeakValidationError,
    Projection2D,
    Search3DError,
    SearchQuery,
    UncalibratedMobilityError,
    extract_peaks3d,
    extract_reader_peaks3d,
)
from pymzml.spec import Spectrum

NS = "http://psi.hupo.org/ms/mzml"


def b64(values):
    return base64.b64encode(struct.pack("<{0}d".format(len(values)), *values)).decode()


def array_xml(accession, name, values, unit=None):
    unit_attrs = ""
    if unit is not None:
        unit_attrs = ' unitCvRef="MS" unitAccession="{0}" unitName="{1}"'.format(
            *unit
        )
    return (
        '<binaryDataArray encodedLength="0">'
        '<cvParam cvRef="MS" accession="{0}" name="{1}"{2}/>'
        '<cvParam cvRef="MS" accession="MS:1000523" name="64-bit float"/>'
        '<cvParam cvRef="MS" accession="MS:1000576" name="no compression"/>'
        "<binary>{3}</binary></binaryDataArray>"
    ).format(accession, name, unit_attrs, b64(values))


def make_spectrum(
    mz, i, mobility, scan_time=1.5, unit=("UO:0000031", "minute"), mobility_unit=True
):
    xml = (
        '<spectrum xmlns="{ns}" index="0" id="scan=7" defaultArrayLength="{n}">'
        '<scanList count="1"><scan>'
        '<cvParam cvRef="MS" accession="MS:1000016" name="scan start time" '
        'value="{rt}" unitCvRef="UO" unitAccession="{ua}" unitName="{un}"/>'
        "</scan></scanList>"
        '<binaryDataArrayList count="3">{mz}{i}{mob}</binaryDataArrayList>'
        "</spectrum>"
    ).format(
        ns=NS,
        n=len(mz),
        rt=scan_time,
        ua=unit[0],
        un=unit[1],
        mz=array_xml(
            "MS:1000514", "m/z array", mz, ("MS:1000040", "m/z")
        ),
        i=array_xml(
            "MS:1000515",
            "intensity array",
            i,
            ("MS:1000131", "number of detector counts"),
        ),
        mob=array_xml(
            "MS:1003006",
            "mean inverse reduced ion mobility array",
            mobility,
            ("MS:1002814", "volt-second per square centimeter")
            if mobility_unit
            else None,
        ),
    )
    return Spectrum(ElementTree.fromstring(xml), obo_version="4.1.79")


def two_segment_table():
    return MobilityCalibration(
        segments=(
            MobilityCalibrationSegment(
                0.80,
                1.00,
                slope=1.0,
                intercept=0.0,
                error=0.002,
                label="low mobility",
            ),
            MobilityCalibrationSegment(
                1.00,
                1.20,
                slope=1.0,
                intercept=0.0,
                error=0.002,
                label="high mobility",
            ),
        ),
        name="test table",
        revision="v1",
    )


class JointPeakRuleTest(unittest.TestCase):
    def test_coordinates_of_different_peaks_are_never_combined(self):
        peaks = [
            Peak3D("rt-only", 5.00, 600.0, 0.95),
            Peak3D("mz-only", 6.00, 500.0, 0.95),
            Peak3D("mobility-only", 6.00, 600.0, 0.90),
        ]
        index = PeakIndex3D(peaks, two_segment_table())
        result = index.search(SearchQuery(5.0, 500.0, 0.9, 0.01, 0.01, 0.01))
        self.assertEqual(result.hits, ())
        self.assertEqual(result.hit_peak_ids, ())
        self.assertEqual(result.evaluated_peak_count, 3)

    def test_hit_is_single_peak_with_all_three_dimensions(self):
        index = PeakIndex3D([Peak3D("p1", 5.00, 500.0, 0.90)], two_segment_table())
        result = index.search(SearchQuery(5.0, 500.0, 0.90, 0.01, 0.01, 0.01))
        self.assertEqual(len(result.hits), 1)
        hit = result.hits[0]
        self.assertEqual(hit.peak_id, "p1")
        self.assertTrue(hit.same_peak)
        self.assertEqual(
            set(hit.dimensions),
            {"retention_time", "mz", "mobility"},
        )
        for evidence in hit.dimensions.values():
            self.assertEqual(evidence.status, "definite")
        self.assertEqual(hit.retention_time, 5.0)
        self.assertEqual(hit.mz, 500.0)
        self.assertAlmostEqual(hit.calibrated_mobility, 0.90)

    def test_duplicate_peak_ids_are_rejected(self):
        with self.assertRaises(Exception):
            PeakIndex3D(
                [Peak3D("p1", 5.0, 500.0, 0.9), Peak3D("p1", 5.0, 500.0, 0.9)],
                two_segment_table(),
            )

    def test_add_peaks_is_all_or_nothing_on_duplicate(self):
        index = PeakIndex3D([Peak3D("p1", 5.0, 500.0, 0.9)], two_segment_table())
        with self.assertRaises(Exception):
            index.add_peaks(
                [Peak3D("p2", 5.0, 500.0, 0.9), Peak3D("p1", 5.0, 500.0, 0.9)]
            )
        self.assertEqual(len(index), 1)

    def test_mz_ppm_tolerance(self):
        index = PeakIndex3D([Peak3D("p1", 5.0, 500.0, 0.9)], two_segment_table())
        result = index.search(
            SearchQuery(5.0, 500.0, 0.9, 0.01, mobility_tolerance=0.01, mz_ppm=20)
        )
        self.assertEqual(len(result.hits), 1)
        self.assertAlmostEqual(result.query.mz_tolerance, 500.0 * 20e-6)

    def test_dimension_aliases(self):
        index = PeakIndex3D([Peak3D("p", 5.0, 500.0, 0.9)], two_segment_table())
        hit = index.search(SearchQuery(5.0, 500.0, 0.9, 0.01, 0.01, 0.01)).hits[0]
        self.assertEqual(hit["rt"].status, "definite")
        self.assertEqual(hit["m/z"].status, "definite")
        self.assertEqual(hit["ion mobility"].status, "definite")


class HitClassificationTest(unittest.TestCase):
    def test_definite_and_possible_hits_on_one_peak(self):
        index = PeakIndex3D(
            [Peak3D("wide", 5.0, 500.0, 0.90, retention_time_uncertainty=0.05)],
            two_segment_table(),
        )
        result = index.search(SearchQuery(5.0, 500.0, 0.90, 0.01, 0.01, 0.05))
        hit = result.hits[0]
        self.assertEqual(hit.status, "possible")
        self.assertEqual(hit.dimensions["retention_time"].status, "possible")
        self.assertEqual(hit.dimensions["mz"].status, "definite")
        self.assertEqual(result.definite_hits, ())

    def test_calibration_error_can_make_a_hit_possible(self):
        calibration = MobilityCalibration(
            (
                MobilityCalibrationSegment(
                    0.8, 1.0, slope=1.0, error=0.003, label="seg"
                ),
            )
        )
        index = PeakIndex3D([Peak3D("p", 5.0, 500.0, 0.90)], calibration)
        tight = index.search(SearchQuery(5.0, 500.0, 0.900, 0.01, 0.01, 0.0001))
        self.assertEqual(tight.hits[0].status, "possible")
        loose = index.search(SearchQuery(5.0, 500.0, 0.900, 0.01, 0.01, 0.010))
        self.assertEqual(loose.hits[0].status, "definite")

    def test_miss_in_any_dimension_is_not_a_hit(self):
        index = PeakIndex3D([Peak3D("p", 5.0, 500.0, 0.90)], two_segment_table())
        for query in (
            SearchQuery(5.0, 500.0, 0.95, 0.01, 0.01, 0.001),
            SearchQuery(5.0, 501.0, 0.90, 0.01, 0.01, 0.001),
            SearchQuery(6.0, 500.0, 0.90, 0.01, 0.01, 0.001),
        ):
            self.assertEqual(index.search(query).hits, ())

    def test_uncalibrated_peak_is_reported_not_hit(self):
        table = MobilityCalibration(
            (MobilityCalibrationSegment(0.8, 1.0, slope=1.0, label="seg"),)
        )
        index = PeakIndex3D(
            [Peak3D("inside", 5.0, 500.0, 0.9), Peak3D("gap", 5.0, 500.0, 1.1)],
            table,
        )
        result = index.search(SearchQuery(5.0, 500.0, 0.9, 0.01, 0.01, 0.01))
        self.assertEqual(result.hit_peak_ids, ("inside",))
        self.assertEqual(result.uncalibrated_peak_ids, ("gap",))
        with self.assertRaises(UncalibratedMobilityError):
            table.calibrate(1.1)


class CalibrationEvidenceTest(unittest.TestCase):
    def test_piecewise_segments_keep_their_own_error_range(self):
        table = two_segment_table().with_segment(
            1, error_low=0.001, error_high=0.004
        )
        low = table.calibrate(0.90)
        high = table.calibrate(1.10)
        self.assertEqual(low.segment_index, 0)
        self.assertEqual(high.segment_index, 1)
        self.assertAlmostEqual(low.lower, 0.90 - 0.002)
        self.assertAlmostEqual(low.upper, 0.90 + 0.002)
        self.assertAlmostEqual(high.lower, 1.10 - 0.001)
        self.assertAlmostEqual(high.upper, 1.10 + 0.004)
        self.assertEqual(low.segment.label, "low mobility")
        self.assertEqual(high.segment.label, "high mobility")

    def test_hit_stores_calibration_basis(self):
        table = two_segment_table()
        index = PeakIndex3D([Peak3D("p", 5.0, 500.0, 1.10)], table)
        hit = index.search(
            SearchQuery(5.0, 500.0, 1.10, 0.01, 0.01, 0.01)
        ).hits[0]
        basis = hit.calibration_basis
        self.assertEqual(basis.segment_index, 1)
        self.assertEqual(basis.segment_label, "high mobility")
        self.assertEqual(basis.calibration_name, "test table")
        self.assertEqual(basis.calibration_revision, "v1")
        self.assertAlmostEqual(basis.measured_mobility, 1.10)
        self.assertEqual(basis.calibrated_interval, (1.10 - 0.002, 1.10 + 0.002))
        self.assertEqual(basis.slope, 1.0)
        self.assertEqual(basis.intercept, 0.0)
        self.assertEqual(hit.calibration_basis.to_dict()["segment_index"], 1)

    def test_calibration_gaps_and_boundaries(self):
        table = MobilityCalibration(
            (
                MobilityCalibrationSegment(0.0, 1.0, slope=2.0, label="first"),
                MobilityCalibrationSegment(1.2, 2.0, slope=3.0, label="second"),
            )
        )
        self.assertEqual(table.segment_index_for(1.0), None)
        self.assertEqual(table.segment_index_for(1.1), None)
        self.assertEqual(table.segment_index_for(1.2), 1)
        self.assertEqual(table.segment_index_for(2.0), 1)
        self.assertEqual(table.segment_index_for(2.1), None)

    def test_overlapping_segments_are_rejected(self):
        with self.assertRaises(CalibrationError):
            MobilityCalibration(
                (
                    MobilityCalibrationSegment(0.0, 1.0),
                    MobilityCalibrationSegment(0.9, 1.5),
                )
            )


class IncrementalRecalibrationTest(unittest.TestCase):
    def setUp(self):
        self.table = two_segment_table()
        self.peaks = [
            Peak3D("low", 5.0, 500.0, 0.90),
            Peak3D("high", 5.0, 500.0, 1.10),
            Peak3D("boundary", 5.0, 500.0, 0.99),
        ]
        self.index = PeakIndex3D(self.peaks, self.table)

    def test_only_affected_segment_is_recalibrated(self):
        query = SearchQuery(5.0, 500.0, 1.1, 0.01, 0.01, 0.01)
        first = self.index.search(query)
        self.assertEqual(first.recomputed_peak_count, 3)
        report = self.index.correct_calibration_segment(1, slope=1.2)
        self.assertEqual(report.changed_segment_indices, (1,))
        self.assertEqual(set(report.recomputed_peak_ids), {"high"})
        self.assertEqual(set(report.unaffected_peak_ids), {"low", "boundary"})
        second = self.index.search(query)
        self.assertEqual(second.recomputed_peak_count, 1)
        self.assertEqual(second.cached_peak_count, 2)
        basis = self.index.calibration_basis_for("high")
        self.assertEqual(basis.slope, 1.2)
        self.assertAlmostEqual(basis.calibrated_mobility, 1.10 * 1.2)
        self.assertEqual(self.index.calibration_basis_for("low").slope, 1.0)

    def test_corrected_segment_can_flip_a_hit(self):
        index = PeakIndex3D([Peak3D("p", 5.0, 500.0, 1.10)], self.table)
        target = SearchQuery(5.0, 500.0, 1.10, 0.01, 0.01, 0.02)
        self.assertEqual(index.search(target).hits[0].status, "definite")
        index.correct_calibration_segment(1, intercept=0.019)
        # The calibrated value is now 1.119 with a symmetric 0.002 error,
        # so the mobility interval overlaps but is not inside the window.
        target = SearchQuery(5.0, 500.0, 1.119, 0.01, 0.01, 0.0005)
        hit = index.search(target).hits[0]
        self.assertEqual(hit.status, "possible")
        self.assertAlmostEqual(hit.calibrated_mobility, 1.119)
        self.assertAlmostEqual(hit.calibration.calibrated_lower, 1.117)
        self.assertAlmostEqual(hit.calibration.calibrated_upper, 1.121)

    def test_uncorrected_peak_results_are_reused_unchanged(self):
        index = self.index
        q1 = SearchQuery(5, 500, 0.9, 0.01, 0.01, 0.01)
        first = index.search(q1)
        before = first.by_peak_id("low")
        index.correct_calibration_segment(1, slope=1.3)
        after = index.search(q1).by_peak_id("low")
        self.assertEqual(before, after)
        self.assertEqual(index.calibration_basis_for("low").segment_index, 0)

    def test_removing_a_segment_marks_peaks_uncalibrated(self):
        table = MobilityCalibration(
            (MobilityCalibrationSegment(0.8, 1.2, slope=1.0, label="only"),)
        )
        index = PeakIndex3D(
            [Peak3D("p", 5.0, 500.0, 0.9), Peak3D("q", 5.0, 500.0, 1.1)], table
        )
        report = index.update_calibration(
            MobilityCalibration(
                (MobilityCalibrationSegment(0.8, 1.0, slope=1.0, label="only"),)
            )
        )
        self.assertEqual(report.recomputed_peak_ids, ("q",))
        result = index.search(SearchQuery(5, 500, 1.1, 0.01, 0.01, 0.01))
        self.assertEqual(result.uncalibrated_peak_ids, ("q",))
        self.assertEqual(result.hits, ())

    def test_revision_and_report_are_tracked(self):
        report = self.index.correct_calibration_segment(0, error=0.003)
        self.assertEqual(report.previous_calibration_revision, 0)
        self.assertEqual(report.calibration_revision, 1)
        self.assertEqual(self.index.last_recalibration, report)
        self.assertEqual(self.index.calibration_revision, 1)
        self.assertEqual(report.changed_segment_labels, ("low mobility",))


class ProjectionTest(unittest.TestCase):
    def setUp(self):
        self.table = two_segment_table()
        self.index = PeakIndex3D(
            [
                Peak3D("low", 5.0, 500.0, 0.90),
                Peak3D("high", 5.0, 500.0, 1.10),
                Peak3D("same-rt-mz", 5.0, 500.0, 0.95),
            ],
            self.table,
        )
        self.projection = self.index.projection_2d()
        self.query = SearchQuery(5.0, 500.0, 0.90, 0.01, 0.01, 0.01)
        self.result = self.index.search(self.query, projection=self.projection)

    def test_projection_keeps_original_peaks(self):
        rows = self.projection.rows()
        self.assertEqual(len(rows), 3)
        self.assertEqual(
            {row.peak_id for row in rows}, {"low", "high", "same-rt-mz"}
        )
        by_id = {row.peak_id: row for row in rows}
        self.assertAlmostEqual(by_id["low"].calibrated_mobility, 0.90)
        self.assertAlmostEqual(by_id["high"].calibrated_mobility, 1.10)
        self.assertEqual(by_id["low"].segment_label, "low mobility")
        self.assertEqual(by_id["high"].segment_label, "high mobility")
        self.assertAlmostEqual(by_id["low"].calibrated_interval[1], 0.902)

    def test_projection_overlay_matches_joint_hits(self):
        self.assertEqual(self.projection.annotations, {"low": "definite"})
        by_id = {row.peak_id: row for row in self.projection.rows()}
        self.assertEqual(by_id["low"].hit_status, "definite")
        self.assertIsNone(by_id["high"].hit_status)

    def test_projection_updates_only_affected_rows(self):
        old_rows = {row.peak_id: row for row in self.projection.rows()}
        report = self.index.correct_calibration_segment(1, slope=1.2)
        self.assertEqual(report.projection_rows_recomputed, 1)
        self.assertTrue(self.projection.is_synchronized)
        new_rows = {row.peak_id: row for row in self.projection.rows()}
        self.assertEqual(new_rows["low"], old_rows["low"])
        self.assertEqual(new_rows["same-rt-mz"], old_rows["same-rt-mz"])
        self.assertAlmostEqual(new_rows["high"].calibrated_mobility, 1.10 * 1.2)
        self.assertAlmostEqual(new_rows["high"].calibrated_interval[0], 1.10 * 1.2 - 0.002)
        self.assertIsNone(new_rows["high"].hit_status)
        self.assertEqual(
            self.projection.last_recomputed_peak_ids, ("high",)
        )

    def test_projection_annotation_requires_current_revision(self):
        self.index.correct_calibration_segment(1, slope=1.2)
        # The view synchronizes on demand before checking the result revision.
        with self.assertRaises(Search3DError):
            self.projection.annotate(self.result)
        self.assertTrue(self.projection.is_synchronized)

    def test_projection_rebuilds_after_added_peak(self):
        self.index.add_peak(Peak3D("new", 6.0, 600.0, 1.05))
        self.assertTrue(self.projection.is_synchronized)
        self.assertIn("new", {row.peak_id for row in self.projection.rows()})
        self.assertEqual(
            self.projection.last_recomputed_peak_ids, ("new",)
        )

    def test_projection_matches_after_each_correction(self):
        for slope in (1.05, 1.1, 0.95):
            self.index.correct_calibration_segment(
                1, slope=slope, revision="slope-{0}".format(slope)
            )
            self.assertTrue(self.projection.is_synchronized)
            rows = {row.peak_id: row for row in self.projection.rows()}
            self.assertAlmostEqual(
                rows["high"].calibrated_mobility, 1.10 * slope
            )
            self.assertEqual(
                self.index.calibration_basis_for("high").calibration_revision,
                "slope-{0}".format(slope),
            )

    def test_projection_plot_columns_are_aligned(self):
        data = self.projection.to_plot_data()
        self.assertEqual(len(data["x"]), 3)
        self.assertEqual(len(data["y"]), 3)
        self.assertEqual(len(data["peak_id"]), 3)
        self.assertEqual(data["peak_id"], [row.peak_id for row in self.projection.rows()])
        statuses = data["hit_status"]
        self.assertEqual(
            statuses,
            ["definite" if value == "definite" else None for value in statuses],
        )
        self.assertEqual(
            self.projection.status_counts(),
            {"definite": 1, "possible": 0, "none": 2},
        )

    def test_calibration_basis_can_be_saved(self):
        import json
        import os
        import tempfile

        handle, path = tempfile.mkstemp(suffix=".json")
        os.close(handle)
        try:
            self.index.save_calibration_basis(path)
            with open(path, encoding="utf-8") as stream:
                payload = json.load(stream)
            self.assertEqual(payload["calibration_revision"], 0)
            self.assertEqual(len(payload["peaks"]), 3)
            high = [item for item in payload["peaks"] if item["peak_id"] == "high"][0]
            self.assertEqual(high["basis"]["segment_index"], 1)
            self.assertEqual(high["basis"]["segment_label"], "high mobility")
            self.assertEqual(high["basis"]["error_low"], 0.002)
        finally:
            os.unlink(path)

    def test_projection_drops_removed_peak(self):
        self.index.remove_peak("same-rt-mz")
        self.assertTrue(self.projection.is_synchronized)
        self.assertNotIn("same-rt-mz", {row.peak_id for row in self.projection.rows()})
        self.assertNotIn("same-rt-mz", self.projection.annotations)


class SpectrumExtractionTest(unittest.TestCase):
    def test_extract_peaks3d_uses_mobility_array_positions(self):
        spectrum = make_spectrum(
            mz=[500.0, 500.1],
            i=[100.0, 200.0],
            mobility=[0.90, 1.10],
            scan_time=1.5,
        )
        peaks = extract_peaks3d(spectrum)
        self.assertEqual(len(peaks), 2)
        self.assertEqual(peaks[0].peak_id, "7:0")
        self.assertEqual(peaks[0].retention_time, 1.5)
        self.assertEqual(peaks[0].mz, 500.0)
        self.assertEqual(peaks[0].intensity, 100.0)
        self.assertEqual(peaks[0].measured_mobility, 0.90)
        self.assertEqual(peaks[1].measured_mobility, 1.10)
        self.assertEqual(
            spectrum.get_all_arrays_in_spec()[-1],
            "mean inverse reduced ion mobility array",
        )
        self.assertEqual(spectrum.ion_mobility.tolist(), [0.90, 1.10])

    def test_extract_peaks3d_honours_retention_time_unit(self):
        spectrum = make_spectrum(
            mz=[500.0], i=[100.0], mobility=[0.90], scan_time=90.0,
            unit=("UO:0000010", "second"),
        )
        peaks = extract_peaks3d(spectrum)
        self.assertAlmostEqual(peaks[0].retention_time, 1.5)
        peaks = extract_peaks3d(
            spectrum, retention_time=90000.0, retention_time_unit="millisecond"
        )
        self.assertAlmostEqual(peaks[0].retention_time, 1.5)

    def test_extract_peaks3d_rejects_missing_or_misaligned_mobility(self):
        spectrum = make_spectrum(mz=[500.0], i=[100.0], mobility=[0.90])
        spectrum._peak_dict["raw"] = np.array([[500.0, 100.0], [501.0, 200.0]])
        with self.assertRaises(Search3DError):
            extract_peaks3d(spectrum)

    def test_extract_reader_peaks3d_uses_scan_time(self):
        spectrum = make_spectrum(
            mz=[500.0], i=[100.0], mobility=[0.90], scan_time=0.5
        )
        peaks = extract_reader_peaks3d([spectrum], max_spectra=1)
        self.assertEqual(len(peaks), 1)
        self.assertAlmostEqual(peaks[0].retention_time, 0.5)
        peaks = extract_reader_peaks3d(
            [spectrum],
            calibrate_retention_time=lambda spec: 2.0,
        )
        self.assertAlmostEqual(peaks[0].retention_time, 2.0)

    def test_extract_reader_peaks3d_skips_spectra_without_mobility(self):
        with_mobility = make_spectrum(
            mz=[500.0], i=[100.0], mobility=[0.90]
        )
        without_mobility = Spectrum(
            ElementTree.fromstring(
                '<spectrum xmlns="{0}" index="0" id="scan=8" '
                'defaultArrayLength="0"><binaryDataArrayList count="0"/>'
                "</spectrum>".format(NS)
            ),
            obo_version="4.1.79",
        )
        peaks = extract_reader_peaks3d([without_mobility, with_mobility])
        self.assertEqual([peak.peak_id for peak in peaks], ["7:0"])

    def test_mobility_array_without_unit_is_still_found(self):
        spectrum = make_spectrum(
            mz=[500.0], i=[100.0], mobility=[0.90], mobility_unit=False
        )
        self.assertEqual(spectrum.get_ion_mobility().tolist(), [0.90])
        peaks = extract_peaks3d(spectrum)
        self.assertEqual(peaks[0].measured_mobility, 0.90)

    def test_extract_peaks3d_to_index_search(self):
        spectrum = make_spectrum(
            mz=[500.0, 500.1], i=[100.0, 200.0], mobility=[0.90, 1.10]
        )
        peaks = extract_peaks3d(spectrum)
        index = PeakIndex3D(peaks, two_segment_table())
        result = index.search(
            SearchQuery(1.5, 500.0, 0.90, 0.001, 0.001, 0.001)
        )
        self.assertEqual(result.hit_peak_ids, ("7:0",))
        self.assertEqual(result.hits[0].same_peak, True)
        self.assertEqual(result.hits[0].intensity, 100.0)


class ReaderHelperTest(unittest.TestCase):
    def test_reader_peaks3d_and_index(self):
        spectrum = make_spectrum(mz=[500.0], i=[100.0], mobility=[0.90])
        reader = [spectrum]
        self.assertEqual(len(reader), 1)
        peaks = extract_reader_peaks3d(reader)
        self.assertEqual(peaks[0].peak_id, "7:0")

    def test_reader_build_3d_index_helper(self):
        from pymzml.run import Reader

        class Holder:
            def __init__(self, spectra):
                self.spectra = spectra

            def __iter__(self):
                return iter(self.spectra)

            peaks3d = Reader.peaks3d

            build_3d_index = Reader.build_3d_index

        holder = Holder([make_spectrum(mz=[500.0], i=[100.0], mobility=[0.90])])
        index = holder.build_3d_index(two_segment_table())
        self.assertEqual(index.calibration.name, "test table")
        self.assertEqual(len(index), 1)
        result = index.search(SearchQuery(1.5, 500.0, 0.9, 0.01, 0.01, 0.01))
        self.assertEqual(result.hit_peak_ids, ("7:0",))


class ValidationTest(unittest.TestCase):
    def test_peak_validation(self):
        with self.assertRaises(PeakValidationError):
            Peak3D("p", float("nan"), 500.0, 0.9)
        with self.assertRaises(PeakValidationError):
            Peak3D("p", 5.0, -500.0, 0.9)
        with self.assertRaises(PeakValidationError):
            Peak3D("p", 5.0, 500.0, 0.9, mz_uncertainty=-1)

    def test_query_validation(self):
        with self.assertRaises(PeakValidationError):
            SearchQuery(5.0, 500.0, 0.9, -1.0, 0.01, 0.01)

    def test_identity_calibration_is_used_without_a_table(self):
        index = PeakIndex3D([Peak3D("p", 5.0, 500.0, 0.9)])
        hit = index.search(SearchQuery(5, 500, 0.9, 0.01, 0.01, 0.001)).hits[0]
        self.assertEqual(hit.calibration.segment_label, "identity")


if __name__ == "__main__":
    unittest.main(verbosity=3)
