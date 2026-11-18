#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test cases for the explicit data representation labels and the integration
conventions of spectra and chromatograms.
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import unittest
import xml.etree.ElementTree as ElementTree

import numpy as np

import pymzml.run as run
from pymzml.chromatogram import Chromatogram
from pymzml.spec import Spectrum
from pymzml.msdata import (
    REPRESENTATION_CENTROID,
    REPRESENTATION_PROFILE,
    REPRESENTATION_REPROFILED,
    REPRESENTATION_UNKNOWN,
)
import test_file_paths


def trapezoid(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    dx = np.diff(x)
    return float((dx * (y[:-1] + y[1:]) / 2.0).sum())


class RepresentationTest(unittest.TestCase):
    def setUp(self):
        self.paths = test_file_paths.paths
        self.reader = run.Reader(self.paths[0])

    def test_imported_centroid_spectrum_is_labelled(self):
        spec = self.reader[6]
        self.assertEqual(spec.representation, REPRESENTATION_CENTROID)

    def test_profile_and_centroid_detection_from_cvparam(self):
        ns = "{http://psi.hupo.org/ms/mzml}"
        for accession, expected in (
            ("MS:1000128", REPRESENTATION_PROFILE),
            ("MS:1000127", REPRESENTATION_CENTROID),
        ):
            element = ElementTree.Element(ns + "spectrum")
            ElementTree.SubElement(
                element,
                ns + "cvParam",
                {"accession": accession, "name": "spectrum type"},
            )
            self.assertEqual(Spectrum(element).representation, expected)

    def test_spectrum_without_type_term_is_unknown(self):
        self.assertEqual(Spectrum().representation, REPRESENTATION_UNKNOWN)

    def test_set_peaks_updates_representation(self):
        spec = Spectrum()
        spec.set_peaks([(1.0, 10.0), (2.0, 20.0)], "centroided")
        self.assertEqual(spec.representation, REPRESENTATION_CENTROID)
        spec.set_peaks([(1.0, 10.0), (2.0, 20.0)], "reprofiled")
        self.assertEqual(spec.representation, REPRESENTATION_REPROFILED)

    def test_invalid_representation_rejected(self):
        with self.assertRaises(ValueError):
            Spectrum().representation = "densely sampled whatever"

    def test_chromatogram_is_a_profile_curve(self):
        reader = run.Reader(
            self.paths[3], skip_chromatogram=False
        )
        chromatogram = reader.get_chromatogram(0)
        self.assertEqual(chromatogram.representation, REPRESENTATION_PROFILE)


class CentroidIntegrationTest(unittest.TestCase):
    def test_centroid_integral_is_sum_of_peak_intensities(self):
        spec = Spectrum()
        peaks = [(100.0, 200.0), (200.0, 300.0)]
        spec.set_peaks(peaks, "centroided")
        self.assertEqual(spec.integrate(), 500.0)
        self.assertEqual(
            spec.integrate(), float(spec.peaks("centroided")[:, 1].sum())
        )

    def test_imported_centroid_integral_matches_intensity_sum(self):
        reader = run.Reader(test_file_paths.paths[0])
        spec = reader[6]
        self.assertAlmostEqual(spec.integrate(), float(spec.i.sum()))

    def test_integral_is_not_the_stored_tic_cvparam(self):
        reader = run.Reader(test_file_paths.paths[0])
        spec = reader[6]
        # the TIC cvParam is reported by the instrument software and is an
        # independent quantity from the pymzml integral convention
        self.assertIsInstance(spec.TIC, float)
        self.assertNotEqual(spec.integrate(), spec.TIC)

    def test_empty_centroid_spectrum_integrates_to_zero(self):
        spec = Spectrum()
        spec.set_peaks(np.empty((0, 2)), "centroided")
        self.assertEqual(spec.integrate(), 0.0)


class ProfileIntegrationTest(unittest.TestCase):
    def _linear_curve(self, x_values):
        x = np.asarray(x_values, dtype=np.float64)
        y = 3.0 * x + 1.0
        return np.column_stack((x, y))

    def test_profile_integral_uses_trapezoidal_rule(self):
        spec = Spectrum()
        peaks = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]
        spec.set_peaks(peaks, "raw")
        spec.representation = REPRESENTATION_PROFILE
        self.assertEqual(spec.integrate(), 1.0)
        self.assertEqual(spec.integrate("raw"), trapezoid([0, 1, 2], [0, 1, 0]))

    def test_same_curve_different_sampling_density_same_integral(self):
        # an exactly linear function is integrated without discretization
        # error, hence the integral must be identical at any sampling density
        coarse = Spectrum()
        coarse.set_peaks(self._linear_curve([0, 2, 4, 6, 8, 10]), "raw")
        fine = Spectrum()
        fine.set_peaks(
            self._linear_curve(np.linspace(0, 10, 1001)), "raw"
        )
        expected = 1.5 * 10.0 ** 2 + 10.0  # integral of 3x + 1 over [0, 10]
        self.assertAlmostEqual(coarse.integrate("raw"), expected)
        self.assertAlmostEqual(fine.integrate("raw"), expected)

        # the naive point-wise sum instead scales with the point count
        self.assertNotAlmostEqual(
            float(coarse.peaks("raw")[:, 1].sum()),
            float(fine.peaks("raw")[:, 1].sum()),
        )

    def test_denser_sampling_converges_for_smooth_curve(self):
        exact = 20.0 + (1.0 - np.cos(10.0))

        def spec_at(n_points):
            x = np.linspace(0, 10, n_points)
            spec = Spectrum()
            spec.set_peaks(np.column_stack((x, np.sin(x) + 2.0)), "raw")
            return spec.integrate("raw")

        coarse_error = abs(spec_at(11) - exact)
        fine_error = abs(spec_at(1001) - exact)
        self.assertLess(fine_error, coarse_error)

    def test_gap_is_not_crossed_when_threshold_given(self):
        peaks = [
            (0.0, 0.0),
            (1.0, 1.0),
            (2.0, 2.0),
            (8.0, 2.0),
            (9.0, 1.0),
            (10.0, 0.0),
        ]
        spec = Spectrum()
        spec.set_peaks(peaks, "raw")
        x = [p[0] for p in peaks]
        y = [p[1] for p in peaks]
        # default: adjacent samples are connected, including across the gap
        self.assertAlmostEqual(spec.integrate("raw"), trapezoid(x, y))
        self.assertAlmostEqual(spec.integrate("raw"), 16.0)
        # with a threshold smaller than the 6-unit gap, that trapezoid is
        # skipped rather than filled with interpolated area
        self.assertAlmostEqual(spec.integrate("raw", gap_threshold=5.0), 4.0)

    def test_single_point_curve_has_no_area(self):
        spec = Spectrum()
        spec.set_peaks([(1.0, 5.0)], "raw")
        self.assertEqual(spec.integrate("raw"), 0.0)

    def test_unsorted_coordinates_are_sorted_before_integration(self):
        spec = Spectrum()
        spec.set_peaks([(2.0, 0.0), (0.0, 0.0), (1.0, 1.0)], "raw")
        self.assertEqual(spec.integrate("raw"), 1.0)

    def test_reprofiled_curve_uses_curve_convention(self):
        # reprofiled peaks are stored as a {mz: intensity} mapping, the same
        # representation produced by _reprofile_Peaks
        spec = Spectrum()
        spec.set_peaks({0.0: 1.0, 1.0: 2.0, 2.0: 1.0}, "reprofiled")
        self.assertEqual(spec.representation, REPRESENTATION_REPROFILED)
        # trapezoidal area = 3.0, point-wise sum = 4.0
        self.assertEqual(spec.integrate(), 3.0)
        self.assertNotEqual(spec.integrate(), 1.0 + 2.0 + 1.0)


class ChromatogramIntegrationTest(unittest.TestCase):
    def test_imported_chromatogram_integral_is_trapezoidal_area(self):
        reader = run.Reader(test_file_paths.paths[3], skip_chromatogram=False)
        chromatogram = reader.get_chromatogram(0)
        profile = chromatogram.profile
        expected = trapezoid(profile[:, 0], profile[:, 1])
        self.assertAlmostEqual(chromatogram.integrate(), expected)

    def test_chromatogram_integral_independent_of_sampling_density(self):
        coarse = Chromatogram(None)
        fine = Chromatogram(None)
        coarse.profile = [
            (t, 2.0 * t) for t in (0.0, 2.0, 4.0, 6.0, 8.0, 10.0)
        ]
        fine.profile = [
            (t, 2.0 * t) for t in np.linspace(0.0, 10.0, 1001)
        ]
        self.assertAlmostEqual(coarse.integrate(), fine.integrate())
        self.assertAlmostEqual(coarse.integrate(), 100.0)

    def test_gap_threshold_skips_chromatogram_gap(self):
        chromatogram = Chromatogram(None)
        chromatogram.profile = [(0.0, 1.0), (1.0, 1.0), (5.0, 1.0), (6.0, 1.0)]
        self.assertAlmostEqual(chromatogram.integrate(), 6.0)
        self.assertAlmostEqual(
            chromatogram.integrate(gap_threshold=2.0), 2.0
        )


if __name__ == "__main__":
    unittest.main(verbosity=3)
