import unittest

from pygtfs.geometry import (haversine_meters, shape_length, project_to_shape,
                            feasible_stop_ranges, traveled_distances)


class FakeShapePoint(object):
    def __init__(self, lat, lon, sequence, dist=None):
        self.shape_pt_lat = lat
        self.shape_pt_lon = lon
        self.shape_pt_sequence = sequence
        self.shape_dist_traveled = dist


class FakeStop(object):
    def __init__(self, lat, lon):
        self.stop_lat = lat
        self.stop_lon = lon


class FakeStopTime(object):
    def __init__(self, sequence, stop_id, lat, lon, dist=None):
        self.stop_sequence = sequence
        self.stop_id = stop_id
        self.shape_dist_traveled = dist
        self.stop = FakeStop(lat, lon)


def square_loop(side=0.01):
    """A closed square: A B C D back to A, roughly 1.1 km per side."""
    corners = [(0.0, 0.0), (0.0, side), (side, side), (side, 0.0),
               (0.0, 0.0)]
    return [FakeShapePoint(lat, lon, i) for i, (lat, lon) in enumerate(corners)]


def figure_eight(side=0.01):
    """Two loops that touch at the crossing X=(0, 0), visited three times."""
    coords = [(0.0, 0.0), (0.0, side), (side, side), (side, 0.0),
              (0.0, 0.0), (-side, 0.0), (-side, -side), (0.0, -side),
              (0.0, 0.0)]
    return [FakeShapePoint(lat, lon, i) for i, (lat, lon) in enumerate(coords)]


def open_figure_eight(side=0.01):
    """Self-intersecting polyline; the crossing is visited twice but it does
    not close back on itself, so a backward jump is not a loop wrap."""
    coords = [(0.0, 0.0), (0.0, side), (side, side), (side, 0.0),
              (0.0, 0.0), (-side, 0.0), (-side, -side), (0.0, -side)]
    return [FakeShapePoint(lat, lon, i) for i, (lat, lon) in enumerate(coords)]


def lollipop(side=0.01):
    """An open polyline that reaches X, loops around and returns to X.

    X is visited twice and both occurrences lie after the stick start S.
    """
    coords = [(0.0, -side), (0.0, 0.0), (0.0, side), (side, side),
              (side, 0.0), (0.0, 0.0)]
    return [FakeShapePoint(lat, lon, i) for i, (lat, lon) in enumerate(coords)]


class GeometryTests(unittest.TestCase):

    def test_haversine_known_distance(self):
        # One degree of longitude at the equator, with the mean earth radius.
        from pygtfs.geometry import EARTH_RADIUS_M
        import math
        expected = EARTH_RADIUS_M * math.radians(1.0)
        self.assertAlmostEqual(haversine_meters(0.0, 0.0, 0.0, 1.0),
                               expected, delta=1.0)

    def test_shape_length_closed_square(self):
        points = square_loop()
        length = shape_length(points)
        self.assertAlmostEqual(length, 4.0 * haversine_meters(0.0, 0.0,
                                                              0.0, 0.01),
                               delta=1.0)

    def test_project_to_shape_keeps_every_self_intersection(self):
        points = figure_eight()
        hits = project_to_shape(0.0, 0.0, points, tolerance_m=25.0)
        distances = sorted(hit.distance for hit in hits)
        # The crossing exists at the start, between the loops and at the end.
        self.assertGreaterEqual(len(distances), 3)
        self.assertLessEqual(distances[0], 25.0)
        total = shape_length(points)
        self.assertGreaterEqual(distances[-1], total - 25.0)
        middle = [d for d in distances if 25.0 < d < total - 25.0]
        self.assertTrue(middle)

    def test_ranges_follow_stop_and_shape_order_on_a_straight_line(self):
        points = [FakeShapePoint(0.0, 0.0, 0),
                  FakeShapePoint(0.0, 0.01, 1),
                  FakeShapePoint(0.0, 0.02, 2)]
        visits = [FakeStopTime(1, 'S0', 0.0, 0.0),
                  FakeStopTime(2, 'M', 0.0, 0.005),
                  FakeStopTime(3, 'S2', 0.0, 0.02)]
        ranges = feasible_stop_ranges(visits, points, tolerance_m=25.0)
        self.assertEqual([r.stop_id for r in ranges], ['S0', 'M', 'S2'])
        for stop_range in ranges:
            self.assertTrue(stop_range.geometric)
            self.assertLessEqual(stop_range.lower, stop_range.upper)
        self.assertLess(ranges[0].upper, ranges[1].lower)
        self.assertLess(ranges[1].upper, ranges[2].lower)
        total = shape_length(points)
        # The middle stop sits a quarter of the way along the line.
        self.assertAlmostEqual(ranges[1].lower, total / 4.0, delta=1.0)

    def test_far_stop_keeps_envelope_instead_of_nearest_snap(self):
        points = [FakeShapePoint(0.0, 0.0, 0),
                  FakeShapePoint(0.0, 0.02, 1)]
        visits = [FakeStopTime(1, 'S0', 0.0, 0.0),
                  FakeStopTime(2, 'LOST', 1.0, 1.0),
                  FakeStopTime(3, 'S1', 0.0, 0.02)]
        ranges = feasible_stop_ranges(visits, points, tolerance_m=25.0)
        self.assertFalse(ranges[1].geometric)
        # The lost visit keeps the whole feasible envelope of its neighbors.
        self.assertLessEqual(ranges[1].lower, ranges[0].upper + 1e-6)
        self.assertGreaterEqual(ranges[1].upper, ranges[2].lower - 1e-6)

    def test_circular_route_pins_shared_terminal_to_start_and_end(self):
        points = square_loop()
        total = shape_length(points)
        visits = [
            FakeStopTime(1, 'A', 0.0, 0.0),
            FakeStopTime(2, 'M1', 0.0, 0.005),
            FakeStopTime(3, 'M2', 0.005, 0.01),
            FakeStopTime(4, 'M3', 0.01, 0.005),
            FakeStopTime(5, 'M4', 0.005, 0.0),
            FakeStopTime(6, 'A', 0.0, 0.0),
        ]
        ranges = feasible_stop_ranges(visits, points, tolerance_m=25.0)
        self.assertLessEqual(ranges[0].upper, total * 0.25)
        self.assertGreaterEqual(ranges[-1].lower, total * 0.75)
        for earlier, later in zip(ranges, ranges[1:]):
            self.assertLessEqual(earlier.lower, later.upper)

    def test_self_intersecting_shape_retains_both_visits_of_crossing(self):
        points = figure_eight()
        total = shape_length(points)
        loop_total = total / 2.0
        visits = [
            FakeStopTime(1, 'X', 0.0, 0.0),
            FakeStopTime(2, 'P2', 0.01, 0.01),
            FakeStopTime(3, 'X', 0.0, 0.0),
            FakeStopTime(4, 'Q2', -0.01, -0.01),
            FakeStopTime(5, 'X', 0.0, 0.0),
        ]
        ranges = feasible_stop_ranges(visits, points, tolerance_m=25.0)
        # First crossing visit stays on the first portion, last one at the end.
        self.assertLessEqual(ranges[0].upper, 25.0)
        self.assertGreaterEqual(ranges[-1].lower, total - 25.0)
        # The middle crossing visit is pinned around the loop junction.
        self.assertGreater(ranges[2].lower, 25.0)
        self.assertLess(ranges[2].upper, total - 25.0)
        self.assertAlmostEqual((ranges[2].lower + ranges[2].upper) / 2.0,
                               loop_total, delta=200.0)

    def test_traveled_distances_from_geometry(self):
        points = [FakeShapePoint(0.0, 0.0, 0),
                  FakeShapePoint(0.0, 0.01, 1),
                  FakeShapePoint(0.0, 0.02, 2)]
        visits = [FakeStopTime(1, 'S0', 0.0, 0.0),
                  FakeStopTime(2, 'M', 0.0, 0.005),
                  FakeStopTime(3, 'S2', 0.0, 0.02)]
        gaps = traveled_distances(visits, points, tolerance_m=25.0)
        half = haversine_meters(0.0, 0.0, 0.0, 0.005)
        self.assertEqual(len(gaps), 2)
        self.assertAlmostEqual(gaps[0], half, delta=1.0)
        self.assertAlmostEqual(gaps[1], 3.0 * half, delta=1.0)

    def test_ambiguous_geometry_does_not_invent_distance(self):
        points = lollipop()
        visits = [
            FakeStopTime(1, 'S', 0.0, -0.01),
            FakeStopTime(2, 'X', 0.0, 0.0),
        ]
        gaps = traveled_distances(visits, points, tolerance_m=25.0)
        # The crossing X exists twice, both after S; neither order nor
        # geometry picks one, so no seemingly precise gap is produced.
        self.assertIsNone(gaps[0])

    def test_mileage_regression_wraps_closed_loop_with_unique_continuation(self):
        points = square_loop()
        cumulative = [0.0]
        for previous, current in zip(points, points[1:]):
            cumulative.append(cumulative[-1] +
                              haversine_meters(previous.shape_pt_lat,
                                               previous.shape_pt_lon,
                                               current.shape_pt_lat,
                                               current.shape_pt_lon))
        anchored = [FakeShapePoint(p.shape_pt_lat, p.shape_pt_lon,
                                   p.shape_pt_sequence, cumulative[i])
                    for i, p in enumerate(points)]
        total = cumulative[-1]
        half_side = haversine_meters(0.0, 0.0, 0.0, 0.005)
        visits = [
            FakeStopTime(1, 'A', 0.0, 0.0, 0.0),
            FakeStopTime(2, 'M1', 0.0, 0.005, half_side),
            FakeStopTime(3, 'A', 0.0, 0.0, total),
            FakeStopTime(4, 'M1', 0.0, 0.005, half_side),
        ]
        gaps = traveled_distances(visits, anchored, tolerance_m=25.0)
        self.assertEqual(len(gaps), 3)
        self.assertAlmostEqual(gaps[0], half_side, delta=1.0)
        self.assertIsNotNone(gaps[1])
        self.assertAlmostEqual(gaps[2], half_side, delta=1.0)
        # The regressed gap covers the remainder of the first lap.
        self.assertAlmostEqual(gaps[1], total - half_side, delta=1.0)

    def test_mileage_regression_with_ambiguous_continuation_returns_none(self):
        points = open_figure_eight()
        cumulative = [0.0]
        for previous, current in zip(points, points[1:]):
            cumulative.append(cumulative[-1] +
                              haversine_meters(previous.shape_pt_lat,
                                               previous.shape_pt_lon,
                                               current.shape_pt_lat,
                                               current.shape_pt_lon))
        anchored = [FakeShapePoint(p.shape_pt_lat, p.shape_pt_lon,
                                   p.shape_pt_sequence, cumulative[i])
                    for i, p in enumerate(points)]
        # Anchor resets to zero at a crossing that exists twice on an open
        # polyline, so the backward jump cannot be a unique loop wrap.
        visits = [
            FakeStopTime(1, 'P2', 0.01, 0.01, cumulative[2]),
            FakeStopTime(2, 'X', 0.0, 0.0, 0.0),
        ]
        gaps = traveled_distances(visits, anchored, tolerance_m=25.0)
        self.assertEqual(len(gaps), 1)
        self.assertIsNone(gaps[0])

    def test_monotonic_anchors_are_reported_directly(self):
        points = square_loop()
        visits = [
            FakeStopTime(1, 'A', 0.0, 0.0, 0.0),
            FakeStopTime(2, 'M1', 0.0, 0.005, 500.0),
            FakeStopTime(3, 'M2', 0.005, 0.01, 1200.0),
        ]
        gaps = traveled_distances(visits, points, tolerance_m=100.0,
                                  anchor_tolerance_m=600.0)
        self.assertEqual(gaps, [500.0, 700.0])


if __name__ == '__main__':
    unittest.main()
