#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for pymzml.blank: blank subtraction on differing, gapped m/z grids.
"""

import json
import math
import os
import random
import sys
import unittest

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import numpy as np

from pymzml.blank import (
    BlankSubtractionResult,
    BlankSubtractor,
    subtract_blank,
    subtract_blank_arrays,
)
from pymzml.spec import Spectrum


class BlankSubtractionTest(unittest.TestCase):
    def test_direct_match(self):
        result = subtract_blank([(100.0, 10.0)], [(100.0, 4.0)])
        self.assertEqual(result.points[0].status, "matched")
        self.assertAlmostEqual(result.points[0].value, 6.0)
        source = result.points[0].sources[0]
        self.assertEqual(source.index, 0)
        self.assertEqual(source.weight, 1.0)

    def test_linear_interpolation_and_both_sides(self):
        result = subtract_blank(
            [(150.0, 20.0)],
            [(100.0, 2.0), (200.0, 4.0)],
        )
        point = result.points[0]
        self.assertEqual(point.status, "interpolated")
        self.assertAlmostEqual(point.blank_intensity, 3.0)
        self.assertAlmostEqual(point.value, 17.0)
        self.assertEqual([source.mz for source in point.sources], [100.0, 200.0])
        self.assertAlmostEqual(point.sources[0].weight, 0.5)
        self.assertAlmostEqual(point.sources[1].weight, 0.5)
        self.assertEqual(point.left.mz, 100.0)
        self.assertEqual(point.right.mz, 200.0)

    def test_tolerance_uses_direct_match(self):
        result = subtract_blank(
            [(100.2, 10.0)],
            [(100.0, 4.0), (200.0, 4.0)],
            tolerance=0.5,
        )
        point = result.points[0]
        self.assertEqual(point.status, "matched")
        self.assertAlmostEqual(point.value, 6.0)
        self.assertEqual(len(point.sources), 1)

    def test_outside_blank_is_unknown_not_zero(self):
        result = subtract_blank(
            [(50.0, 10.0), (100.0, 10.0), (150.0, 10.0)],
            [(90.0, 2.0), (110.0, 4.0)],
        )
        self.assertEqual(len(result.known_points), 1)
        self.assertAlmostEqual(result.known_points[0].value, 7.0)
        statuses = {point.mz: point.status for point in result}
        self.assertEqual(statuses[50.0], "unknown")
        self.assertEqual(statuses[150.0], "unknown")
        for point in result.unknown_points:
            self.assertIsNone(point.value)
            self.assertIsNone(point.blank_intensity)

    def test_max_gap_reports_unknown_interval(self):
        result = subtract_blank(
            [(100.0, 1.0), (150.0, 1.0), (300.0, 1.0)],
            [(100.0, 1.0), (200.0, 2.0), (300.0, 3.0)],
            max_gap=10.0,
        )
        unknown = result.unknown_points
        self.assertEqual([point.mz for point in unknown], [150.0])
        self.assertEqual(unknown[0].unknown_reason, "gap")
        self.assertEqual(unknown[0].left.mz, 100.0)
        self.assertEqual(unknown[0].right.mz, 200.0)

    def test_unknown_interval_bounds(self):
        result = subtract_blank(
            [(50.0, 1.0), (100.0, 1.0), (200.0, 1.0), (260.0, 1.0)],
            [(100.0, 1.0), (200.0, 1.0)],
        )
        intervals = result.unknown_intervals
        self.assertEqual(len(intervals), 2)
        self.assertEqual(intervals[0].mz_start, 50.0)
        self.assertEqual(intervals[0].mz_end, 100.0)
        self.assertEqual(intervals[0].reason, "left_of_blank")
        self.assertEqual(intervals[1].mz_start, 200.0)
        self.assertEqual(intervals[1].mz_end, 260.0)
        self.assertEqual(intervals[1].reason, "right_of_blank")

    def test_negative_values_are_preserved(self):
        result = subtract_blank([(100.0, 1.0)], [(100.0, 5.0)])
        self.assertAlmostEqual(result.points[0].value, -4.0)
        self.assertAlmostEqual(result.peaks[0][1], -4.0)

    def test_clip_is_explicit(self):
        result = subtract_blank([(100.0, 1.0)], [(100.0, 5.0)], clip=True)
        self.assertEqual(result.points[0].value, 0.0)

    def test_scaling(self):
        result = subtract_blank(
            [(150.0, 20.0)],
            [(100.0, 2.0), (200.0, 4.0)],
            scale=0.5,
        )
        self.assertAlmostEqual(result.points[0].blank_intensity, 1.5)
        self.assertAlmostEqual(result.points[0].value, 18.5)

    def test_scale_callable(self):
        result = subtract_blank(
            [(100.0, 10.0), (200.0, 10.0)],
            [(100.0, 4.0), (200.0, 4.0)],
            scale=lambda mz: 0.5 if mz < 150 else 2.0,
        )
        values = [point.value for point in result]
        self.assertAlmostEqual(values[0], 8.0)
        self.assertAlmostEqual(values[1], 2.0)

    def test_invalid_scale_is_unknown(self):
        result = subtract_blank(
            [(100.0, 10.0)],
            [(100.0, 4.0)],
            scale=lambda mz: math.nan,
        )
        point = result.points[0]
        self.assertEqual(point.status, "unknown")
        self.assertEqual(point.unknown_reason, "invalid_scale")
        self.assertIsNone(point.value)

    def test_duplicate_blank_mz_is_summed(self):
        result = subtract_blank([(100.0, 10.0)], [(100.0, 1.0), (100.0, 3.0)])
        self.assertAlmostEqual(result.points[0].blank_intensity, 4.0)
        self.assertAlmostEqual(result.points[0].value, 6.0)

    def test_empty_blank_makes_everything_unknown(self):
        result = subtract_blank([(100.0, 10.0)], [])
        self.assertEqual(len(result), 1)
        self.assertEqual(result.points[0].unknown_reason, "no_blank")
        self.assertEqual(len(result.unknown_intervals), 1)

    def test_empty_sample(self):
        result = subtract_blank([], [(100.0, 1.0)])
        self.assertEqual(len(result), 0)
        self.assertEqual(result.peaks.shape, (0, 2))

    def test_nan_sample_point_is_unknown(self):
        result = subtract_blank([(math.nan, 5.0)], [(100.0, 1.0)])
        self.assertEqual(result.points[0].unknown_reason, "invalid_sample")

    def test_input_order_and_provenance_is_preserved(self):
        result = subtract_blank(
            [(300.0, 3.0), (100.0, 1.0), (200.0, 2.0)],
            [(100.0, 1.0), (200.0, 2.0), (300.0, 3.0)],
        )
        self.assertEqual([point.mz for point in result], [300.0, 100.0, 200.0])
        self.assertEqual([point.sample_index for point in result], [0, 1, 2])
        self.assertAlmostEqual(result.points[0].value, 0.0)

    def test_array_wrapper(self):
        result = subtract_blank_arrays(
            [100.0, 150.0],
            [10.0, 20.0],
            [100.0, 200.0],
            [2.0, 4.0],
        )
        self.assertTrue(np.allclose(result.peaks, [[100.0, 8.0], [150.0, 17.0]]))

    def test_spectrum_method(self):
        sample = Spectrum()
        sample.set_peaks([(100.0, 10.0), (150.0, 20.0)], "raw")
        blank = Spectrum()
        blank.set_peaks([(100.0, 2.0), (200.0, 4.0)], "raw")
        result = sample.subtract_blank(blank)
        self.assertTrue(np.allclose(result.peaks, [[100.0, 8.0], [150.0, 17.0]]))
        # the source peak arrays are not modified
        self.assertTrue(
            np.allclose(sample.peaks("raw"), [[100.0, 10.0], [150.0, 20.0]])
        )

    def test_spectrum_method_with_reprofiled_peaks(self):
        sample = Spectrum()
        sample.set_peaks([(100.0, 10.0)], "centroided")
        blank = Spectrum()
        blank.set_peaks([(100.0, 2.0)], "centroided")
        result = sample.subtract_blank(
            blank, peak_type="centroided", blank_peak_type="centroided"
        )
        self.assertAlmostEqual(result.points[0].value, 8.0)

    def test_result_json_roundtrip(self):
        result = subtract_blank(
            [(50.0, 1.0), (100.0, 10.0), (150.0, 20.0)],
            [(100.0, 2.0), (200.0, 4.0)],
            scale=0.5,
        )
        payload = json.loads(result.to_json())
        self.assertEqual(payload["options"]["scale"], 0.5)
        self.assertEqual(len(payload["points"]), 3)
        self.assertEqual(payload["points"][0]["status"], "unknown")
        self.assertEqual(payload["points"][1]["sources"][0]["mz"], 100.0)
        self.assertEqual(payload["unknown_intervals"][0]["reason"], "left_of_blank")

    def test_as_arrays_keeps_unknown(self):
        result = subtract_blank(
            [(50.0, 1.0), (100.0, 10.0)],
            [(100.0, 2.0)],
        )
        mz, sample, blank, value = result.as_arrays(include_unknown=True)
        self.assertEqual(len(mz), 2)
        self.assertTrue(math.isnan(value[0]))
        self.assertAlmostEqual(value[1], 8.0)

    def test_chunk_size_is_invariant(self):
        rng = random.Random(11)
        sample_mz = sorted(rng.uniform(50.0, 250.0) for _ in range(40))
        sample_i = [rng.uniform(-2.0, 50.0) for _ in sample_mz]
        blank_mz = sorted(set(rng.uniform(60.0, 240.0) for _ in range(30)))
        blank_i = [rng.uniform(0.0, 10.0) for _ in blank_mz]
        sample = list(zip(sample_mz, sample_i))
        blank = list(zip(blank_mz, blank_i))
        reference = subtract_blank(sample, blank, tolerance=0.01, max_gap=15.0)
        for chunk_size in (1, 2, 7, 40, 1000):
            chunked = subtract_blank(
                sample,
                blank,
                tolerance=0.01,
                max_gap=15.0,
                chunk_size=chunk_size,
            )
            self.assertTrue(
                np.allclose(reference.all_peaks, chunked.all_peaks, equal_nan=True)
            )
            self.assertEqual(
                [point.status for point in reference],
                [point.status for point in chunked],
            )
            self.assertEqual(reference.unknown_intervals, chunked.unknown_intervals)

    def test_iter_chunks_combine_matches_full_pass(self):
        sample = [(mz, float(mz)) for mz in range(100, 140)]
        blank = [(100.0, 1.0), (139.0, 2.0)]
        subtractor = BlankSubtractor(sample, blank)
        full = subtractor.subtract()
        combined = BlankSubtractionResult.combine(list(subtractor.iter_chunks(7)))
        self.assertTrue(np.allclose(full.all_peaks, combined.all_peaks, equal_nan=True))
        self.assertEqual(full.unknown_intervals, combined.unknown_intervals)

    def test_chunk_boundary_between_brackets(self):
        # The left bracket is in one chunk and the right bracket in the next;
        # a per-chunk grid would wrongly report the middle point as unknown.
        sample = [(100.0, 10.0), (150.0, 20.0), (200.0, 30.0)]
        blank = [(100.0, 1.0), (200.0, 3.0)]
        result = subtract_blank(sample, blank, chunk_size=2)
        self.assertAlmostEqual(result.points[1].blank_intensity, 2.0)
        self.assertAlmostEqual(result.points[1].value, 18.0)

    def test_chunk_boundary_inside_unknown_gap(self):
        # Both blank points bracket the whole sample, so no gap is reported and
        # splitting between the two brackets changes nothing.
        sample = [(50.0, 1.0), (60.0, 1.0), (70.0, 1.0), (80.0, 1.0)]
        blank = [(10.0, 1.0), (100.0, 1.0)]
        full = subtract_blank(sample, blank)
        result = subtract_blank(sample, blank, chunk_size=1)
        self.assertEqual(len(result.unknown_intervals), 0)
        self.assertEqual(full.unknown_intervals, result.unknown_intervals)

    def test_blank_value_exposes_alignment_and_scaling(self):
        subtractor = BlankSubtractor(
            [(150.0, 20.0)],
            [(100.0, 2.0), (200.0, 4.0)],
            scale=0.5,
        )
        estimate = subtractor.blank_value(150.0)
        self.assertEqual(estimate["status"], "interpolated")
        self.assertAlmostEqual(estimate["blank_intensity"], 1.5)
        self.assertEqual([source.mz for source in estimate["sources"]], [100.0, 200.0])

        outside = subtractor.blank_value(50.0)
        self.assertEqual(outside["status"], "unknown")
        self.assertIsNone(outside["blank_intensity"])
        self.assertEqual(outside["right"].mz, 100.0)

    def test_blank_value_callable_scale(self):
        subtractor = BlankSubtractor(
            [(100.0, 1.0)],
            [(100.0, 4.0)],
            scale=lambda mz: mz / 100.0,
        )
        self.assertAlmostEqual(subtractor.blank_value(100.0)["blank_intensity"], 4.0)

    def test_invalid_chunk_size(self):
        with self.assertRaises(ValueError):
            subtract_blank([(1.0, 1.0)], [(1.0, 1.0)], chunk_size=0)

    def test_invalid_input_shape(self):
        with self.assertRaises(ValueError):
            subtract_blank([1.0, 2.0], [(1.0, 1.0)])

        with self.assertRaises(TypeError):
            subtract_blank([("a", "b")], [(1.0, 1.0)])


if __name__ == "__main__":
    unittest.main(verbosity=3)
