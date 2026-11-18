# -*- coding: utf-8 -*-

import os.path
import unittest

from pygtfs import Schedule, overwrite_feed
from pygtfs.direction import (
    SAME, OPPOSITE, UNKNOWN, UNASSIGNED,
    TripPattern, DirectionGrouper, DirectionConflictError,
    anchors_from_trips, group_route_directions,
    compare_stop_sequences, compare_shapes,
)


def pattern(stops, shape=None):
    return TripPattern(stops=list(stops), shape=shape)


class StopSequenceComparisonTest(unittest.TestCase):
    def test_identical_sequence_is_same(self):
        verdict, _ = compare_stop_sequences(
            ["s1", "s2", "s3"], ["s1", "s2", "s3"])
        self.assertEqual(SAME, verdict)

    def test_reversed_sequence_is_opposite(self):
        verdict, _ = compare_stop_sequences(
            ["s1", "s2", "s3", "s4"], ["s4", "s3", "s2", "s1"])
        self.assertEqual(OPPOSITE, verdict)

    def test_circular_trips_with_same_terminals_are_distinguished(self):
        # Both trips start and end at s1, so terminal-based grouping would
        # collapse them; the ordered sequence still tells the directions
        # apart.
        clockwise = ["s1", "s2", "s3", "s4", "s5", "s1"]
        counterclockwise = ["s1", "s5", "s4", "s3", "s2", "s1"]
        verdict, _ = compare_stop_sequences(clockwise, counterclockwise)
        self.assertEqual(OPPOSITE, verdict)

    def test_weakly_overlapping_sequences_stay_unknown(self):
        # A detour branch only shares the two terminals with the trunk;
        # neither orientation reaches the match threshold.
        verdict, _ = compare_stop_sequences(
            ["s1", "s2", "s3", "s4", "s5"],
            ["s1", "x1", "x2", "x3", "s5"])
        self.assertEqual(UNKNOWN, verdict)

    def test_short_turn_matches_trunk_direction(self):
        verdict, _ = compare_stop_sequences(
            ["s1", "s2", "s3", "s4", "s5"],
            ["s1", "s2", "s3", "s4"])
        self.assertEqual(SAME, verdict)

    def test_empty_sequences_yield_no_evidence(self):
        self.assertIsNone(compare_stop_sequences([], ["s1"]))


class ShapeComparisonTest(unittest.TestCase):
    def setUp(self):
        self.forward = [
            (0.0, 0.0), (0.0, 0.001), (0.0, 0.002),
            (0.0, 0.003), (0.0, 0.004),
        ]
        self.backward = list(reversed(self.forward))

    def test_coincident_shapes_are_same(self):
        verdict, _ = compare_shapes(self.forward, list(self.forward))
        self.assertEqual(SAME, verdict)

    def test_reversed_shapes_are_opposite(self):
        verdict, _ = compare_shapes(self.forward, self.backward)
        self.assertEqual(OPPOSITE, verdict)

    def test_diverging_shapes_stay_unknown(self):
        other = [
            (0.1, 0.0), (0.1, 0.001), (0.1, 0.002),
            (0.1, 0.003), (0.1, 0.004),
        ]
        verdict, _ = compare_shapes(self.forward, other)
        self.assertEqual(UNKNOWN, verdict)

    def test_missing_shapes_yield_no_evidence(self):
        self.assertIsNone(compare_shapes(None, self.forward))
        self.assertIsNone(compare_shapes([], self.forward))

    def test_disjoint_stop_ids_same_geometry_group_together(self):
        # The two trips share no stop id; geometry is the only evidence.
        patterns = {
            "T1": pattern(["a1", "a2", "a3", "a4", "a5"], self.forward),
            "T2": pattern(["b1", "b2", "b3", "b4", "b5"], self.forward),
        }
        result = DirectionGrouper(patterns).group()
        self.assertEqual(1, len(result.groups))
        self.assertEqual(("T1", "T2"), result.groups[0].trip_ids)

    def test_contradictory_stop_and_shape_evidence_stays_unknown(self):
        # Stops claim opposite, geometry claims same direction: the pair
        # must not be force-labelled.
        patterns = {
            "T1": pattern(["s1", "s2", "s3", "s4", "s5"], self.forward),
            "T2": pattern(["s5", "s4", "s3", "s2", "s1"], self.forward),
        }
        result = DirectionGrouper(patterns).group()
        verdict = result.pair_verdicts[("T1", "T2")].verdict
        self.assertEqual(UNKNOWN, verdict)
        self.assertEqual(2, len(result.groups))


class DirectionGrouperTest(unittest.TestCase):
    def setUp(self):
        # Trunk corridor s1 -> s5, with a short turn, the reverse trip,
        # a circular pair and a detour branch.
        self.patterns = {
            "T1": pattern(["s1", "s2", "s3", "s4", "s5"]),
            "T2": pattern(["s1", "s2", "s3", "s4", "s5"]),
            "T3": pattern(["s1", "s2", "s3", "s4"]),
            "T4": pattern(["s5", "s4", "s3", "s2", "s1"]),
            "C1": pattern(["s1", "s2", "s3", "s4", "s5", "s1"]),
            "C2": pattern(["s1", "s5", "s4", "s3", "s2", "s1"]),
            "B1": pattern(["s1", "x1", "x2", "x3", "s5"]),
        }
        self.grouper = DirectionGrouper(self.patterns)

    def test_pair_verdicts_cover_every_pair(self):
        result = self.grouper.group()
        trip_ids = sorted(self.patterns)
        self.assertEqual(21, len(result.pair_verdicts))
        for index, trip_a in enumerate(trip_ids):
            for trip_b in trip_ids[index + 1:]:
                self.assertIn((trip_a, trip_b), result.pair_verdicts)

    def test_same_direction_trips_share_one_group(self):
        result = self.grouper.group()
        trunk = result.group_of("T1")
        # C1 shares the s1..s5 corridor in the same direction and joins
        # provably; the short turn T3 joins for the same reason.
        self.assertEqual(("C1", "T1", "T2", "T3"), trunk.trip_ids)
        self.assertEqual(SAME,
                         result.pair_verdicts[("T1", "T2")].verdict)
        self.assertEqual(SAME,
                         result.pair_verdicts[("T1", "T3")].verdict)

    def test_reverse_trip_is_opposite_not_merged(self):
        result = self.grouper.group()
        self.assertEqual(OPPOSITE,
                         result.pair_verdicts[("T1", "T4")].verdict)
        self.assertNotEqual(result.group_of("T1").group_id,
                            result.group_of("T4").group_id)

    def test_circular_trips_split_despite_same_terminals(self):
        # Both circular trips start and end at s1; only the ordered stop
        # sequence can tell the two travel directions apart.
        circular = {
            "C1": pattern(["s1", "s2", "s3", "s4", "s5", "s1"]),
            "C2": pattern(["s1", "s5", "s4", "s3", "s2", "s1"]),
        }
        result = DirectionGrouper(circular).group()
        self.assertEqual(OPPOSITE,
                         result.pair_verdicts[("C1", "C2")].verdict)
        self.assertEqual(2, len(result.groups))
        self.assertNotEqual(result.group_of("C1").group_id,
                            result.group_of("C2").group_id)

    def test_detour_branch_is_not_force_merged_into_trunk(self):
        result = self.grouper.group()
        self.assertEqual(UNKNOWN,
                         result.pair_verdicts[("B1", "T1")].verdict)
        branch = result.group_of("B1")
        self.assertEqual(("B1",), branch.trip_ids)

    def test_without_anchors_everything_is_unassigned(self):
        result = self.grouper.group()
        self.assertTrue(all(direction is UNASSIGNED
                            for direction in result.assignments.values()))
        self.assertEqual(
            sorted(self.patterns), result.unassigned_trip_ids)

    def test_anchor_labels_group_and_propagates_across_opposite_edges(self):
        result = self.grouper.group(anchors={"T1": 0})
        self.assertEqual(0, result.assignments["T1"])
        self.assertEqual(0, result.assignments["T2"])
        self.assertEqual(0, result.assignments["T3"])
        # T4 is provably opposite the anchored trunk.
        self.assertEqual(1, result.assignments["T4"])
        # The branch has no provable relation: it must stay unassigned.
        self.assertEqual(UNASSIGNED, result.assignments["B1"])
        self.assertIn("anchored by T1=0",
                      result.group_of("T1").reason)
        self.assertIn("opposite group",
                      result.group_of("T4").reason)

    def test_anchor_on_branch_leaves_trunk_to_opposite_evidence(self):
        # Anchoring the branch must not drag the trunk into its group.
        result = self.grouper.group(anchors={"B1": 1})
        self.assertEqual(1, result.assignments["B1"])
        self.assertEqual(UNASSIGNED, result.assignments["T1"])
        self.assertEqual(UNASSIGNED, result.assignments["T4"])

    def test_conflicting_anchors_inside_one_group_raise(self):
        with self.assertRaises(DirectionConflictError):
            self.grouper.group(anchors={"T1": 0, "T2": 1})

    def test_conflicting_anchors_across_opposite_edge_raise(self):
        with self.assertRaises(DirectionConflictError):
            self.grouper.group(anchors={"T1": 0, "T4": 0})

    def test_invalid_anchor_inputs_are_rejected(self):
        with self.assertRaises(KeyError):
            self.grouper.group(anchors={"NOPE": 0})
        with self.assertRaises(ValueError):
            self.grouper.group(anchors={"T1": 2})

    def test_adding_an_anchor_explains_which_trips_change_assignment(self):
        before = self.grouper.group()
        after = self.grouper.group(anchors={"T1": 0})

        changes = after.explain_changes(before)
        changed = dict((change.trip_id, change) for change in changes)

        # The whole same-direction group gets direction 0 and every
        # provably opposite group gets 1; the floating branch is not
        # reported because nothing can prove its direction.
        self.assertEqual(
            set(changed),
            {"T1", "T2", "T3", "T4", "C1", "C2"})
        self.assertEqual((UNASSIGNED, 0),
                         (changed["T2"].old_direction_id,
                          changed["T2"].new_direction_id))
        self.assertEqual((UNASSIGNED, 1),
                         (changed["T4"].old_direction_id,
                          changed["T4"].new_direction_id))
        self.assertNotIn("B1", changed)
        self.assertIn("anchored by T1=0", changed["T1"].reason)

        # A second anchor on the previously floating branch only moves the
        # branch itself.
        after_second = self.grouper.group(
            anchors={"T1": 0, "B1": 0})
        second_changes = after_second.explain_changes(after)
        self.assertEqual(
            ["B1"], [change.trip_id for change in second_changes])
        self.assertEqual(
            (UNASSIGNED, 0),
            (second_changes[0].old_direction_id,
             second_changes[0].new_direction_id))

    def test_explain_changes_accepts_plain_assignment_dict(self):
        result = self.grouper.group(anchors={"T1": 0})
        changes = result.explain_changes({"T1": 1})
        change = next(change for change in changes
                      if change.trip_id == "T1")
        self.assertEqual((1, 0),
                         (change.old_direction_id,
                          change.new_direction_id))

    def test_no_evidence_trips_remain_separate_and_unassigned(self):
        patterns = {
            "E1": pattern([]),
            "E2": pattern([]),
        }
        result = DirectionGrouper(patterns).group()
        self.assertEqual(2, len(result.groups))
        self.assertEqual(UNKNOWN,
                         result.pair_verdicts[("E1", "E2")].verdict)
        self.assertEqual(
            {"E1": UNASSIGNED, "E2": UNASSIGNED}, result.assignments)


class OrmIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.schedule = Schedule(":memory:")
        data_location = os.path.join(os.path.dirname(__file__),
                                     "data", "sample_feed")
        overwrite_feed(self.schedule, data_location)
        self.routes = dict(
            (route.route_id, route) for route in self.schedule.routes)

    def test_sample_feed_city_route_splits_into_two_directions(self):
        result = group_route_directions(self.routes["CITY"])
        # CITY1 and CITY2 traverse the same stops in reverse order.
        self.assertEqual(OPPOSITE,
                         result.pair_verdicts[("CITY1", "CITY2")].verdict)
        self.assertEqual(2, len(result.groups))
        self.assertEqual(
            [UNASSIGNED, UNASSIGNED],
            [group.direction_id for group in result.groups])

    def test_anchors_from_trips_label_existing_direction_ids(self):
        route = self.routes["CITY"]
        anchors = anchors_from_trips(route.trips)
        self.assertEqual({"CITY1": 0, "CITY2": 1}, anchors)

        result = group_route_directions(route, anchors=anchors)
        self.assertEqual(0, result.assignments["CITY1"])
        self.assertEqual(1, result.assignments["CITY2"])

    def test_sample_feed_ab_route_opposite_terminals(self):
        result = group_route_directions(self.routes["AB"])
        self.assertEqual(OPPOSITE,
                         result.pair_verdicts[("AB1", "AB2")].verdict)
        anchored = group_route_directions(
            self.routes["AB"], anchors={"AB1": 0})
        self.assertEqual(0, anchored.assignments["AB1"])
        self.assertEqual(1, anchored.assignments["AB2"])

    def test_single_trip_route_forms_its_own_group(self):
        result = group_route_directions(self.routes["STBA"])
        self.assertEqual(1, len(result.groups))
        self.assertEqual(("STBA",), result.groups[0].trip_ids)
        self.assertEqual(UNASSIGNED, result.groups[0].direction_id)

    def test_grouping_is_per_route_and_ignores_route_short_names(self):
        # AAMV (short name 50) and CITY (short name 40) are grouped in
        # separate calls; nothing ever merges trips across routes or keys
        # on the route short name.
        aamv = group_route_directions(self.routes["AAMV"])
        city = group_route_directions(self.routes["CITY"])
        aamv_trips = set(aamv.assignments)
        city_trips = set(city.assignments)
        self.assertEqual(
            {"AAMV1", "AAMV2", "AAMV3", "AAMV4"}, aamv_trips)
        self.assertEqual({"CITY1", "CITY2"}, city_trips)
        self.assertFalse(aamv_trips & city_trips)


class DirectShapeLookupTest(unittest.TestCase):
    def test_shape_id_fallback_without_loader_mapping(self):
        from pygtfs.gtfs_entities import (Feed, Route, Trip, Stop, StopTime,
                                          ShapePoint, Service)

        schedule = Schedule(":memory:")
        feed = Feed(feed_name="manual")
        schedule.session.add(feed)
        schedule.session.flush()
        schedule.session.add(Service(
            feed_id=feed.feed_id, service_id="S1",
            monday="1", tuesday="1", wednesday="1", thursday="1",
            friday="1", saturday="1", sunday="1"))
        route = Route(feed_id=feed.feed_id, route_id="R1",
                      route_short_name="1", route_type=3)
        schedule.session.add(route)
        for stop_id in ("a1", "a2", "a3", "b1", "b2", "b3"):
            schedule.session.add(Stop(feed_id=feed.feed_id,
                                      stop_id=stop_id, stop_name=stop_id,
                                      stop_lat=0.0, stop_lon=0.0))
        schedule.session.flush()

        forward = [(0.0, 0.0), (0.0, 0.002), (0.0, 0.004)]
        shapes = (
            ("shF", forward),
            ("shR", list(reversed(forward))),
        )
        for shape_id, shape in shapes:
            for index, (lat, lon) in enumerate(shape):
                schedule.session.add(ShapePoint(
                    feed_id=feed.feed_id, shape_id=shape_id,
                    shape_pt_lat=lat, shape_pt_lon=lon,
                    shape_pt_sequence=index + 1))

        specs = (
            ("T1", "shF", ("a1", "a2", "a3")),
            ("T2", "shF", ("b1", "b2", "b3")),
            ("T3", "shR", ("b3", "b2", "b1")),
        )
        for trip_id, shape_id, stops in specs:
            schedule.session.add(Trip(feed_id=feed.feed_id, route_id="R1",
                                      service_id="S1", trip_id=trip_id,
                                      shape_id=shape_id))
            for sequence, stop_id in enumerate(stops, start=1):
                schedule.session.add(StopTime(
                    feed_id=feed.feed_id, trip_id=trip_id,
                    stop_id=stop_id, stop_sequence=sequence))
        schedule.session.commit()

        # The loader's _trip_shapes mapping was never populated.
        trip = schedule.trips_by_id("T1")[0]
        self.assertEqual([], list(trip.shape_points))

        result = group_route_directions(route)
        self.assertEqual(SAME,
                         result.pair_verdicts[("T1", "T2")].verdict)
        self.assertEqual(OPPOSITE,
                         result.pair_verdicts[("T1", "T3")].verdict)


if __name__ == '__main__':
    unittest.main()
