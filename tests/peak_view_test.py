#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for versioned calibrated peak views."""

import os
import sys
import unittest

sys.path.append(os.path.abspath("."))

import numpy as np

import pymzml.run as run
from pymzml.peak_view import (
    Calibration,
    CalibrationError,
    NonInvertibleError,
    PeakQueryIndex,
    PeakView,
    VersionConflictError,
    VersionedPeakViews,
)
import test_file_paths


PEAKS = np.array(
    [
        [100.0000, 1000.0],
        [100.2000, 50.0],
        [500.0000, 500.0],
        [500.0200, 200.0],
        [800.0000, 10.0],
    ]
)


class PeakViewTest(unittest.TestCase):
    def setUp(self):
        self.original = PeakView(PEAKS, version="original", measured_precision=5e-6)

    def test_original_and_revised_views_coexist(self):
        revised = PeakView(
            PEAKS,
            Calibration({"scale": 1.0, "offset": 0.003}, version="revised"),
            "revised",
            5e-6,
        )
        np.testing.assert_allclose(self.original.mz, PEAKS[:, 0])
        np.testing.assert_allclose(revised.mz, PEAKS[:, 0] + 0.003)
        # Original results are still available after adding the revision.
        self.assertEqual(
            self.original.has_peak(100.0), [(100.0, 1000.0)]
        )
        self.assertEqual(revised.has_peak(100.003)[0][0], 100.003)

    def test_peak_identity_is_stable_across_calibrations(self):
        revised = PeakView(
            PEAKS,
            Calibration({"offset": 0.01}),
            "revised",
            20e-6,
        )
        self.assertEqual(
            [identity.as_tuple() for identity in self.original.identities],
            [identity.as_tuple() for identity in revised.identities],
        )
        self.assertEqual(revised.original_mz[0], 100.0)
        self.assertEqual(revised.source_indices[0], 0)
        self.assertEqual(revised.original_index_for(0), 0)
        self.assertEqual(revised.view_index_for_original(0), 0)

    def test_revised_view_does_not_overwrite_original(self):
        first = self.original
        revised = PeakView(PEAKS, Calibration({"offset": 0.02}), "revised")
        self.assertIs(self.original, first)
        np.testing.assert_allclose(self.original.mz, PEAKS[:, 0])
        np.testing.assert_allclose(revised.mz, PEAKS[:, 0] + 0.02)
        np.testing.assert_allclose(revised.original_mz, PEAKS[:, 0])

    def test_error_window_uses_selected_version(self):
        revised = PeakView(
            PEAKS, Calibration({"offset": 0.003}), "revised", measured_precision=1e-6
        )
        # A 0.002 window finds the peak only in the revised view.
        self.assertEqual(self.original.find_peaks(100.003, error_window=0.002), [])
        matches = revised.find_peaks(100.003, error_window=0.002)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].original_mz, 100.0)

    def test_window_boundaries_are_inclusive(self):
        view = PeakView(PEAKS, version="original")
        self.assertEqual(len(view.find_peaks(100.2, error_window=0.0)), 1)
        self.assertEqual(len(view.find_peaks(100.2, error_window=0.2)), 2)

    def test_ppm_query_is_relative(self):
        view = PeakView([(100.0, 1.0)], version="original", measured_precision=5e-6)
        self.assertEqual(len(view.find_peaks(100.0004, ppm=5e-6)), 1)
        self.assertEqual(view.find_peaks(100.001, ppm=5e-6), [])

    def test_precision_change_invalidates_query_index(self):
        view = PeakView([(100.0, 1.0)], version="original", measured_precision=5e-6)
        self.assertEqual(view.find_peaks(100.002, ppm=5e-6), [])
        view.measured_precision = 50e-6
        self.assertEqual(len(view.find_peaks(100.002, ppm=50e-6)), 1)
        view.measured_precision = 5e-6
        self.assertEqual(view.find_peaks(100.002, ppm=5e-6), [])

    def test_interleaved_queries_do_not_cross_caches(self):
        revised = PeakView(
            PEAKS, Calibration({"offset": 0.003}), "revised", measured_precision=5e-6
        )
        for _ in range(3):
            self.assertEqual(self.original.find_peaks(100.003, ppm=5e-6), [])
            self.assertEqual(len(revised.find_peaks(100.003, ppm=5e-6)), 1)
            self.assertEqual(len(self.original.find_peaks(100.0, ppm=5e-6)), 1)
            self.assertEqual(revised.find_peaks(100.0, ppm=5e-6), [])

    def test_views_do_not_share_writable_arrays(self):
        revised = PeakView(PEAKS, Calibration({"offset": 0.003}), "revised")
        with self.assertRaises(ValueError):
            self.original.positions[0] = 1.0
        with self.assertRaises(ValueError):
            revised.mz[0] = 1.0
        self.assertEqual(self.original.mz[0], PEAKS[0, 0])

    def test_recalibrated_returns_new_view(self):
        revised = self.original.recalibrated(Calibration({"offset": 0.003}), "revised")
        np.testing.assert_allclose(self.original.mz, PEAKS[:, 0])
        np.testing.assert_allclose(revised.mz, PEAKS[:, 0] + 0.003)


class VersionedPeakViewsTest(unittest.TestCase):
    def test_original_is_always_available(self):
        views = VersionedPeakViews(peaks=PEAKS)
        self.assertEqual(views.versions, ("original",))
        original = views.original
        self.assertEqual(original.version, "original")
        self.assertEqual(views.versions, ("original",))
        self.assertEqual(len(views), 1)
        self.assertIn("original", views)

    def test_duplicate_version_requires_explicit_replace(self):
        views = VersionedPeakViews(peaks=PEAKS)
        views.add("revised", {"offset": 0.003})
        with self.assertRaises(VersionConflictError):
            views.add("revised", {"offset": 0.02})
        replacement = views.add("revised", {"offset": 0.02}, replace=True)
        self.assertEqual(replacement.mz[0], PEAKS[0, 0] + 0.02)

    def test_same_calibration_is_idempotent(self):
        views = VersionedPeakViews(peaks=PEAKS)
        first = views.add("revised", {"offset": 0.003})
        second = views.add("revised", {"offset": 0.003})
        self.assertIs(first, second)

    def test_activate_only_changes_default(self):
        views = VersionedPeakViews(peaks=PEAKS)
        original = views.original
        revised = views.add("revised", {"offset": 0.003})
        views.activate("revised")
        self.assertIs(views.active_view, revised)
        self.assertEqual(original.mz[0], PEAKS[0, 0])
        views.activate("original")
        self.assertIs(views.active_view, original)

    def test_view_lookup_errors_name_known_versions(self):
        views = VersionedPeakViews(peaks=PEAKS)
        views.add("revised", {"offset": 0.003})
        with self.assertRaises(KeyError) as context:
            views.view("unknown")
        self.assertIn("revised", str(context.exception))

    def test_added_version_uses_original_identities(self):
        views = VersionedPeakViews(peaks=PEAKS)
        original = views.original
        revised = views.add("revised", {"offset": 0.003})
        self.assertEqual(
            revised.source_indices.tolist(), original.source_indices.tolist()
        )
        self.assertEqual(revised.original_mz[1], PEAKS[1, 0])


class CalibrationTest(unittest.TestCase):
    def test_affine(self):
        calibration = Calibration({"scale": 1.000001, "offset": 0.002})
        self.assertAlmostEqual(calibration.apply(100.0), 100.0021)
        self.assertAlmostEqual(calibration.inverse(100.0021), 100.0)
        self.assertTrue(calibration.is_monotonic)
        self.assertTrue(calibration.is_injective)

    def test_piecewise_monotonic(self):
        calibration = Calibration([(0.0, 1.0), (10.0, 11.0), (20.0, 21.0)])
        self.assertTrue(calibration.is_monotonic)
        self.assertEqual(calibration.reverse_lookup(11.0).preimages, [10.0])
        self.assertTrue(calibration.reverse_lookup(11.0).unique)

    def test_non_monotonic_reverse_is_explained(self):
        calibration = Calibration([(0.0, 0.0), (1.0, 2.0), (2.0, 1.0)])
        self.assertFalse(calibration.is_monotonic)
        result = calibration.reverse_lookup(1.5)
        self.assertFalse(result.unique)
        self.assertEqual(len(result.preimages), 2)
        self.assertIn("non-monotonic", result.explain())
        self.assertEqual(len(calibration.non_monotonic_segments()), 1)

    def test_flat_segment_is_not_uniquely_invertible(self):
        calibration = Calibration([(0.0, 0.0), (1.0, 1.0), (2.0, 1.0), (3.0, 2.0)])
        result = calibration.reverse_lookup(1.0)
        self.assertFalse(result.unique)
        self.assertIn("constant", result.explain())
        with self.assertRaises(NonInvertibleError):
            calibration.inverse(1.0)
        self.assertTrue(calibration.can_reverse_uniquely(0.5))

    def test_unique_inverse_helpers(self):
        calibration = Calibration({"offset": 0.5})
        self.assertEqual(calibration.unique_inverse(100.5), 100.0)
        self.assertTrue(calibration.can_reverse_uniquely(100.5))

        curved = Calibration([(0.0, 0.0), (1.0, 2.0), (2.0, 1.0)])
        self.assertIsNone(curved.unique_inverse(1.5))
        self.assertFalse(curved.can_reverse_uniquely(1.5))
        self.assertEqual(curved.unique_inverse(1.9), 1.9 - 0.0 if False else None)

    def test_identity_calibration(self):
        calibration = Calibration()
        self.assertTrue(calibration.is_identity)
        self.assertEqual(calibration.reverse(12.5), [12.5])

    def test_view_level_reverse_reports_peaks(self):
        view = PeakView(PEAKS, Calibration({"offset": 0.003}), "revised")
        result = view.reverse_lookup(100.003, ppm=5e-6)
        self.assertTrue(result.unique)
        self.assertEqual(result.preimages[0].original_mz, 100.0)
        self.assertEqual(result.original.original_mz, 100.0)

    def test_view_level_reverse_ambiguous_window(self):
        view = PeakView(PEAKS, version="original", measured_precision=5e-6)
        result = view.reverse_lookup(100.1, error_window=0.2)
        self.assertFalse(result.unique)
        self.assertEqual(len(result.preimages), 2)
        self.assertIn("ambiguous", result.explain())


class PeakQueryIndexTest(unittest.TestCase):
    def test_index_caches_are_per_precision_and_position(self):
        a = PeakQueryIndex([100.0], 5e-6)
        b = PeakQueryIndex([100.0], 50e-6)
        self.assertNotEqual(a.internal_precision, b.internal_precision)
        self.assertEqual(a.has_peak(100.002), [])
        self.assertEqual(b.has_peak(100.002), [0])
        self.assertNotEqual(a.t_mz_set, b.t_mz_set)

    def test_query_window_is_sorted(self):
        index = PeakQueryIndex([100.2, 100.0, 100.1], 5e-6)
        self.assertEqual(index.query_window(100.1, 0.1), [1, 2, 0])


class ConstructionCompatibilityTest(unittest.TestCase):
    def test_segment_tuples_and_duplicate_shared_endpoints(self):
        for points in (
            [(0.0, 1.0, 0.0, 1.0), (1.0, 2.0, 1.0, 3.0)],
            [(0.0, 0.0, 1.0, 1.0), (1.0, 1.0, 2.0, 3.0)],
            [(0.0, 0.0), (1.0, 1.0), (1.0, 1.0), (2.0, 3.0)],
        ):
            calibration = Calibration.piecewise(points)
            self.assertEqual(calibration.apply(0.5), 0.5)

    def test_segment_dict_control_points(self):
        calibration = Calibration(
            [{"x0": 0.0, "x1": 1.0, "y0": 0.0, "y1": 2.0}]
        )
        self.assertEqual(calibration.apply(0.25), 0.5)

    def test_conflicting_duplicate_endpoint_rejected(self):
        with self.assertRaises(CalibrationError):
            Calibration([(0.0, 0.0), (1.0, 1.0), (1.0, 2.0)])

    def test_calibration_class_constructors(self):
        self.assertEqual(Calibration.shift(2.0).apply(1.0), 3.0)
        self.assertEqual(Calibration.affine(2.0, 1.0).apply(3.0), 7.0)
        self.assertEqual(
            Calibration.from_callable(lambda x: 2 * x, domain=(0, 10)).apply(2.0),
            4.0,
        )
        self.assertTrue(Calibration.identity().is_identity)

    def test_callable_peaks_property(self):
        view = PeakView(PEAKS, version="original")
        self.assertEqual(view.peaks.shape, (len(PEAKS), 2))
        np.testing.assert_allclose(view.peaks(), view.peaks)

    def test_get_with_default(self):
        views = VersionedPeakViews(peaks=PEAKS)
        self.assertIsNone(views.get("missing"))
        self.assertIs(views.get("missing", "fallback"), "fallback")


class ExtrapolatedReverseTest(unittest.TestCase):
    def test_reverse_extrapolates_like_forward_mapping(self):
        calibration = Calibration([(0.0, 0.0), (1.0, 1.0), (2.0, 3.0)])
        self.assertEqual(calibration.apply(2.5), 4.0)
        self.assertEqual(calibration.reverse_lookup(4.0).preimages, [2.5])
        self.assertTrue(calibration.reverse_lookup(4.0).unique)
        self.assertEqual(calibration.reverse_lookup(-0.5).preimages, [-0.5])


class SpectrumPeakViewTest(unittest.TestCase):
    def setUp(self):
        self.paths = test_file_paths.paths
        self.spec = run.Reader(self.paths[2])[6]
        # Deterministic small peak set with two close peaks.
        self.spec.set_peaks(PEAKS.tolist(), "centroided")
        self.spec.set_peaks(PEAKS.tolist(), "raw")

    def test_spectrum_versions(self):
        spec = self.spec
        original = spec.peak_views.original
        revised = spec.add_peak_view("revised", {"offset": 0.003})
        self.assertEqual(spec.peak_views.versions, ("original", "revised"))
        np.testing.assert_allclose(original.mz, PEAKS[:, 0])
        np.testing.assert_allclose(revised.mz, PEAKS[:, 0] + 0.003)
        self.assertEqual(spec.get_peak_view("revised").version, "revised")

    def test_spectrum_error_window_per_version(self):
        spec = self.spec
        spec.activate_peak_view("original")
        self.assertEqual(spec.find_peaks(100.003, error_window=0.002), [])
        spec.add_peak_view("revised", {"offset": 0.003})
        records = spec.find_peaks(100.003, error_window=0.002, version="revised")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].original_mz, 100.0)
        # The active view did not change as a side effect.
        self.assertEqual(spec.active_peak_view.version, "original")

    def test_set_peaks_invalidates_peak_views(self):
        spec = self.spec
        spec.add_peak_view("revised", {"offset": 0.003})
        spec.set_peaks([(1000.0, 1.0)], "centroided")
        view = spec.get_peak_view("original")
        self.assertEqual(len(view), 1)
        self.assertEqual(view.mz[0], 1000.0)

    def test_spectrum_precision_change_does_not_leak_cache(self):
        spec = self.spec
        spec.measured_precision = 5e-6
        narrow = spec.has_peak(100.001)
        narrow_set = set(spec.t_mz_set)
        spec.measured_precision = 50e-6
        wide = spec.has_peak(100.001)
        wide_set = set(spec.t_mz_set)
        self.assertEqual(narrow, [])
        self.assertGreaterEqual(len(wide), 1)
        self.assertNotEqual(narrow_set, wide_set)
        # Restoring the old precision rebuilds a matching index again.
        spec.measured_precision = 5e-6
        self.assertEqual(spec.has_peak(100.001), [])
        self.assertEqual(set(spec.t_mz_set), narrow_set)


class CompatibilityAndEdgeCaseTest(unittest.TestCase):
    def test_x_y_calibration_constructor(self):
        calibration = Calibration([0.0, 1.0, 2.0], [0.0, 2.0, 1.0])
        self.assertFalse(calibration.is_monotonic)
        self.assertEqual(calibration.non_monotonic_ranges, ((1.0, 2.0),))
        calibration = Calibration(x=[0.0, 1.0], y=[0.0, 1.5])
        self.assertAlmostEqual(calibration.apply(0.5), 0.75)

    def test_query_window_alias(self):
        view = PeakView([(100.0, 1.0)], version="original")
        self.assertEqual(len(view.find_peaks(100.05, window=0.1)), 1)
        self.assertEqual(len(view.find_peaks(100.05, mz_window=0.01)), 0)
        self.assertEqual(len(view.find_peaks(100.05, tolerance=0.1)), 1)

    def test_original_is_property_and_callable(self):
        views = VersionedPeakViews(peaks=PEAKS)
        self.assertIs(views.original, views.original())
        self.assertIs(views.original, views.base)
        self.assertIs(views.original, views.get_original())

    def test_original_peak_identity_remap(self):
        views = VersionedPeakViews(peaks=PEAKS)
        views.add("revised", {"offset": 0.003})
        self.assertEqual(views.original_peak_identity(2).source_index, 2)
        self.assertEqual(views.remap_original_index(2, "revised"), 2)
        with self.assertRaises(KeyError):
            views.remap_original_index(99, "revised")

    def test_constant_calibration_is_not_unique(self):
        calibration = Calibration([(0.0, 5.0)])
        self.assertFalse(calibration.reverse_lookup(5.0).unique)
        self.assertIn("constant", calibration.explain_reverse(5.0))
        self.assertEqual(calibration.apply(123.0), 5.0)

    def test_sequence_protocol(self):
        view = PeakView(PEAKS, version="original")
        self.assertEqual(list(view)[0], (100.0, 1000.0))
        self.assertEqual(view[0], (100.0, 1000.0))
        self.assertEqual(len(view[:2]), 2)

    def test_stale_index_reference_keeps_old_precision(self):
        view = PeakView([(100.0, 1.0)], version="original", measured_precision=5e-6)
        old_index = view.query_index
        view.measured_precision = 50e-6
        self.assertIsNot(view.query_index, old_index)
        self.assertEqual(old_index.measured_precision, 5e-6)
        self.assertEqual(old_index.has_peak(100.002), [])
        self.assertEqual(view.query_index.has_peak(100.002), [0])

    def test_activated_view_drives_versionless_lookup(self):
        spectrum = run.Reader(test_file_paths.paths[2])[6]
        spectrum.set_peaks(PEAKS.tolist(), "centroided")
        spectrum.add_peak_view("revised", {"offset": 0.003}, activate=True)
        self.assertEqual(spectrum.active_peak_view.version, "revised")
        self.assertEqual(len(spectrum.has_peak(100.003)), 1)
        spectrum.activate_peak_view("original")
        self.assertEqual(spectrum.has_peak(100.003), [])
        self.assertEqual(len(spectrum.has_peak(100.003, version="revised")), 1)

    def test_original_view_cannot_be_replaced_or_removed(self):
        views = VersionedPeakViews(peaks=PEAKS)
        views.original
        with self.assertRaises(VersionConflictError):
            views.add("original", {"offset": 0.01})
        with self.assertRaises(VersionConflictError):
            views.add("base", {"offset": 0.01}, replace=True)
        with self.assertRaises(VersionConflictError):
            views.remove("original")
        # A pure identity re-request is idempotent, not destructive.
        original = views.original
        self.assertIs(views.add("original", None), original)

    def test_view_calibration_is_read_only(self):
        view = PeakView(PEAKS, {"offset": 0.003}, "revised")
        with self.assertRaises(AttributeError):
            view.calibration = {"offset": 1.0}
        self.assertEqual(view.mz[0], PEAKS[0, 0] + 0.003)

    def test_reverse_result_is_mapping_like(self):
        view = PeakView(PEAKS, {"offset": 0.003}, "revised")
        result = view.reverse_lookup(100.003, ppm=5e-6)
        self.assertEqual(result["unique"], True)
        self.assertTrue(result == [result.preimages[0]])
        self.assertEqual(result.get("missing", 7), 7)
        self.assertEqual(result.as_dict()["unique"], True)

    def test_compare_versions_by_original_identity(self):
        views = VersionedPeakViews(peaks=PEAKS)
        views.add("revised", {"offset": 0.003})
        rows = views.compare("original", "revised")
        self.assertEqual(len(rows), len(PEAKS))
        self.assertEqual(rows[0]["source_index"], 0)
        self.assertAlmostEqual(rows[0]["mz_a"], 100.0)
        self.assertAlmostEqual(rows[0]["mz_b"], 100.003)
        self.assertEqual(rows[0]["original_mz"], 100.0)

    def test_monotonicity_is_reported_on_the_view(self):
        view = PeakView(
            PEAKS,
            Calibration([(0.0, 0.0), (1.0, 2.0), (2.0, 1.0)]),
            "folding",
        )
        self.assertFalse(view.is_monotonic)
        self.assertFalse(view.is_injective)
        self.assertEqual(len(view.non_monotonic_segments), 1)


if __name__ == "__main__":
    unittest.main()
