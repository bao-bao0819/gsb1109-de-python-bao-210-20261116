import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import unittest

from pymzml.chromatogram import Chromatogram


def make_chromatogram(points):
    chrom = Chromatogram(None)
    chrom.profile = points
    return chrom


def roll(start, stop, step, intensity):
    return [(float(t), float(intensity)) for t in frange(start, stop, step)]


def frange(start, stop, step):
    t = start
    while t <= stop:
        yield t
        t += step


class ChromatogramIntegrationTest(unittest.TestCase):
    def setUp(self):
        # acquisition rolls 0-10, 20-30 and 40-50 min with constant
        # intensity 100; two rolls (10-20 and 30-40 min) are missing
        self.max_gap = 2.0
        self.available = (
            roll(0, 10, 1, 100) + roll(20, 30, 1, 100) + roll(40, 50, 1, 100)
        )

    def test_continuous_single_segment(self):
        chrom = make_chromatogram(
            [(0, 0), (1, 10), (2, 20), (3, 20), (4, 10), (5, 0)]
        )
        result = chrom.integrate(max_gap=2.0)
        self.assertEqual(len(result["segments"]), 1)
        self.assertAlmostEqual(result["area"], 60.0)
        self.assertEqual(result["gaps"], [])

    def test_gaps_are_not_bridged(self):
        chrom = make_chromatogram(self.available)
        result = chrom.integrate(max_gap=self.max_gap)
        # provable area only covers the three acquired rolls
        self.assertAlmostEqual(result["area"], 3000.0)
        self.assertEqual(len(result["segments"]), 3)
        self.assertEqual(
            [(gap["start"], gap["end"]) for gap in result["gaps"]],
            [(10.0, 20.0), (30.0, 40.0)],
        )
        # a straight line across the whole 0-50 min span would pretend 5000
        self.assertLess(result["area"], 5000.0)

    def test_single_point_segment_has_no_area(self):
        chrom = make_chromatogram([(0, 5), (1, 5), (10, 7)])
        result = chrom.integrate(max_gap=2.0)
        self.assertEqual(len(result["segments"]), 2)
        single = result["segments"][1]
        self.assertEqual(single["n_points"], 1)
        self.assertEqual(single["area"], 0.0)
        self.assertAlmostEqual(result["area"], 5.0)
        self.assertEqual(
            [(gap["start"], gap["end"]) for gap in result["gaps"]],
            [(1.0, 10.0)],
        )

    def test_recovered_roll_only_adds_supported_area(self):
        chrom = make_chromatogram(self.available)
        before = chrom.integrate(max_gap=self.max_gap)
        # recover the 10-20 min roll, 30-40 min is still missing
        recovered = sorted(self.available + roll(10, 20, 1, 100))
        chrom.profile = recovered
        after = chrom.integrate(max_gap=self.max_gap)
        self.assertAlmostEqual(after["area"] - before["area"], 1000.0)
        self.assertAlmostEqual(after["area"], 4000.0)
        # the still missing roll must not be levelled over
        self.assertEqual(
            [(gap["start"], gap["end"]) for gap in after["gaps"]],
            [(30.0, 40.0)],
        )

    def test_unsorted_input(self):
        shuffled = [self.available[i] for i in (5, 0, 30, 10, 22, 32, 1)]
        shuffled += [p for p in self.available if p not in shuffled]
        chrom = make_chromatogram(shuffled)
        result = chrom.integrate(max_gap=self.max_gap)
        self.assertAlmostEqual(result["area"], 3000.0)
        self.assertEqual(len(result["gaps"]), 2)

    def test_gap_boundary_is_allowed(self):
        chrom = make_chromatogram([(0, 1), (2, 1)])
        result = chrom.integrate(max_gap=2.0)
        self.assertEqual(len(result["segments"]), 1)
        self.assertAlmostEqual(result["area"], 2.0)
        self.assertEqual(result["gaps"], [])

    def test_empty_profile(self):
        chrom = Chromatogram(None)
        chrom._time = []
        chrom._i = []
        result = chrom.integrate(max_gap=self.max_gap)
        self.assertEqual(result["area"], 0.0)
        self.assertEqual(result["segments"], [])
        self.assertEqual(result["gaps"], [])


if __name__ == "__main__":
    unittest.main(verbosity=3)
