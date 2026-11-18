import os
import sys
import unittest

sys.path.append(os.path.abspath("."))

try:
    import numpy as np

    numpy_installed = True
except ImportError:
    numpy_installed = False

from pymzml.peak_merger import (
    AmbiguousPeakGroupError,
    build_peak_groups,
    merge_peaks,
    within_tolerance,
)
from pymzml.spec import Spectrum


class ToleranceTest(unittest.TestCase):
    def test_ppm_tolerance(self):
        # 5 ppm at m/z 100 allows an error of 0.0005
        self.assertTrue(within_tolerance(100.0, 100.0004, ppm=5.0))
        self.assertFalse(within_tolerance(100.0, 100.0006, ppm=5.0))

    def test_absolute_tolerance_combined_with_ppm(self):
        # ppm window is smaller than the absolute window, absolute wins
        self.assertTrue(
            within_tolerance(100.0, 100.0009, ppm=1.0, abs_tolerance=0.001)
        )
        self.assertFalse(
            within_tolerance(100.0, 100.0011, ppm=1.0, abs_tolerance=0.001)
        )

    def test_tolerance_is_symmetric(self):
        self.assertEqual(
            within_tolerance(100.0, 100.0004, ppm=5.0),
            within_tolerance(100.0004, 100.0, ppm=5.0),
        )


class PeakGroupTest(unittest.TestCase):
    def test_three_peak_chain_is_ambiguous(self):
        # adjacent pairs are within 5 ppm (0.0004 < 0.0005) but the first and
        # last peak differ by 0.0008, exceeding the allowed error
        peaks = [(100.0, 1.0), (100.0004, 2.0), (100.0008, 3.0)]
        with self.assertRaises(AmbiguousPeakGroupError):
            build_peak_groups(peaks, ppm=5.0)
        with self.assertRaises(AmbiguousPeakGroupError):
            merge_peaks(peaks, ppm=5.0)

    def test_ambiguous_error_exposes_offending_peaks_and_rules(self):
        peaks = [(100.0, 1.0), (100.0004, 2.0), (100.0008, 3.0)]
        try:
            build_peak_groups(peaks, ppm=5.0, abs_tolerance=0.0)
        except AmbiguousPeakGroupError as error:
            self.assertEqual(error.peaks, tuple(peaks))
            self.assertEqual(error.ppm, 5.0)
            self.assertEqual(error.abs_tolerance, 0.0)
        else:
            self.fail("AmbiguousPeakGroupError was not raised")

    def test_absolute_tolerance_can_resolve_chain(self):
        # with an absolute tolerance of 0.001 all pairs mutually match
        peaks = [(100.0, 1.0), (100.0004, 2.0), (100.0008, 3.0)]
        groups = build_peak_groups(peaks, ppm=0.0, abs_tolerance=0.001)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0].mzs), 3)
        self.assertEqual(groups[0].indices, (0, 1, 2))

    def test_groups_are_independent_of_input_order(self):
        peaks = [
            (300.0, 9.0),
            (100.0004, 3.0),
            (100.0, 1.0),
            (200.0005, 2.0),
            (200.0, 4.0),
        ]
        groups = build_peak_groups(peaks, ppm=5.0)
        grouped_mzs = [group.mzs for group in groups]
        self.assertEqual(
            grouped_mzs,
            [
                (100.0, 100.0004),
                (200.0, 200.0005),
                (300.0,),
            ],
        )

        shuffled = [peaks[index] for index in (2, 4, 0, 1, 3)]
        shuffled_groups = build_peak_groups(shuffled, ppm=5.0)
        self.assertEqual(
            [group.mzs for group in shuffled_groups],
            [group.mzs for group in groups],
        )

    def test_tied_mz_values_are_ordered_by_original_position(self):
        peaks = [(100.0, 2.0), (100.0, 1.0)]
        groups = build_peak_groups(peaks, ppm=5.0)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].indices, (0, 1))
        self.assertEqual(groups[0].intensities, (2.0, 1.0))

    def test_singleton_group_for_isolated_peak(self):
        groups = build_peak_groups([(100.0, 5.0)], ppm=5.0)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].indices, (0,))
        self.assertEqual(groups[0].as_peak(), (100.0, 5.0))

    def test_invalid_peak_raises_value_error(self):
        with self.assertRaises(ValueError):
            build_peak_groups([(100.0, 1.0), ("not a peak",)], ppm=5.0)


class MergePeaksTest(unittest.TestCase):
    def test_weighted_mz_summed_intensity_and_contributors(self):
        peaks = [(100.0, 1.0), (100.0004, 3.0)]
        merged = merge_peaks(peaks, ppm=5.0)
        self.assertEqual(len(merged), 1)
        merged_peak = merged[0]
        self.assertAlmostEqual(merged_peak.mz, 100.0003)
        self.assertEqual(merged_peak.intensity, 4.0)
        self.assertEqual(
            merged_peak.contributors,
            ((100.0, 1.0), (100.0004, 3.0)),
        )

    def test_unrelated_peaks_remain_separate(self):
        peaks = [(100.0, 1.0), (200.0, 2.0)]
        merged = merge_peaks(peaks, ppm=5.0)
        self.assertEqual(len(merged), 2)
        self.assertEqual(
            [peak.contributors for peak in merged],
            [((100.0, 1.0),), ((200.0, 2.0),)],
        )

    def test_zero_intensity_group_uses_unweighted_mean(self):
        merged = merge_peaks([(100.0, 0.0), (100.0004, 0.0)], ppm=5.0)
        self.assertAlmostEqual(merged[0].mz, 100.0002)
        self.assertEqual(merged[0].intensity, 0.0)

    @unittest.skipUnless(numpy_installed, "numpy is required")
    def test_accepts_numpy_arrays(self):
        peaks = np.array([(100.0, 1.0), (100.0004, 3.0)])
        merged = merge_peaks(peaks, ppm=5.0)
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(merged[0].mz, 100.0003)


class SpectrumMergeTest(unittest.TestCase):
    def _build_spectrum(self, peaks, measured_precision=5e-6):
        spectrum = Spectrum(measured_precision=measured_precision)
        spectrum.set_peaks(peaks, "centroided")
        return spectrum

    def test_spectrum_merge_uses_measured_precision_by_default(self):
        spectrum = self._build_spectrum(
            [(100.0, 1.0), (100.0004, 3.0), (300.0, 9.0)]
        )
        merged = spectrum.merge_neighboring_peaks()
        self.assertEqual(len(merged), 2)
        self.assertAlmostEqual(merged[0].mz, 100.0003)
        self.assertEqual(merged[0].intensity, 4.0)
        self.assertEqual(
            merged[0].contributors,
            ((100.0, 1.0), (100.0004, 3.0)),
        )

    def test_spectrum_peak_groups(self):
        spectrum = self._build_spectrum(
            [(100.0, 1.0), (100.0004, 3.0), (300.0, 9.0)]
        )
        groups = spectrum.build_peak_groups()
        self.assertEqual(
            [group.indices for group in groups], [(0, 1), (2,)]
        )

    def test_spectrum_merge_in_place(self):
        spectrum = self._build_spectrum(
            [(100.0, 1.0), (100.0004, 3.0), (300.0, 9.0)]
        )
        merged = spectrum.merge_neighboring_peaks(in_place=True)
        stored = spectrum.peaks("centroided")
        self.assertEqual(len(stored), len(merged))
        self.assertAlmostEqual(stored[0][0], merged[0].mz)
        self.assertAlmostEqual(stored[0][1], merged[0].intensity)

    def test_spectrum_reports_ambiguous_chain(self):
        spectrum = self._build_spectrum(
            [(100.0, 1.0), (100.0004, 2.0), (100.0008, 3.0)]
        )
        with self.assertRaises(AmbiguousPeakGroupError):
            spectrum.merge_neighboring_peaks()

    def test_spectrum_absolute_tolerance(self):
        spectrum = self._build_spectrum(
            [(100.0, 1.0), (100.0004, 2.0), (100.0008, 3.0)]
        )
        merged = spectrum.merge_neighboring_peaks(ppm=0.0, abs_tolerance=0.001)
        self.assertEqual(len(merged), 1)
        self.assertEqual(len(merged[0].contributors), 3)


if __name__ == "__main__":
    unittest.main(verbosity=3)
