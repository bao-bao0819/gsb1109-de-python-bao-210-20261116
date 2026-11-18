# -*- coding: utf-8 -*-

import datetime
import os.path
import unittest
from decimal import Decimal

from pygtfs import Schedule, FarePlanner, JourneyLeg, plan_fares
from pygtfs.gtfs_entities import Feed, Fare, FareRule


def _make_schedule():
    schedule = Schedule(":memory:")
    feed = Feed(feed_name="fare_test")
    schedule.session.add(feed)
    schedule.session.flush()
    return schedule, feed.feed_id


def _add_fare(schedule, feed_id, fare_id, price, currency="USD",
              transfers=0, transfer_duration=None):
    schedule.session.add(Fare(
        feed_id=feed_id, fare_id=fare_id, price=Decimal(str(price)),
        currency_type=currency, payment_method=1, transfers=transfers,
        transfer_duration=transfer_duration))


def _add_rule(schedule, feed_id, fare_id, route_id=None, origin_id=None,
              destination_id=None, contains_id=None):
    schedule.session.add(FareRule(
        feed_id=feed_id, fare_id=fare_id, route_id=route_id,
        origin_id=origin_id, destination_id=destination_id,
        contains_id=contains_id))


def _leg(trip_id, route_id, stop_a, stop_b, zone_a, zone_b,
         departure_hour, arrival_hour):
    return JourneyLeg(
        trip_id=trip_id, route_id=route_id,
        from_stop_id=stop_a, to_stop_id=stop_b,
        from_zone=zone_a, to_zone=zone_b,
        departure=datetime.timedelta(hours=departure_hour),
        arrival=datetime.timedelta(hours=arrival_hour))


class TestFarePlanner(unittest.TestCase):
    def setUp(self):
        self.schedule, self.feed_id = _make_schedule()

    def _plan(self, legs):
        self.schedule.session.commit()
        return FarePlanner(self.schedule).plan(legs)

    def test_zone_and_route_restrictions(self):
        _add_fare(self.schedule, self.feed_id, "z", "2.00")
        _add_rule(self.schedule, self.feed_id, "z",
                  origin_id="Z1", destination_id="Z2")
        _add_fare(self.schedule, self.feed_id, "r", "5.00")
        _add_rule(self.schedule, self.feed_id, "r", route_id="R1")

        solutions = self._plan([
            _leg("T1", "R1", "S1", "S2", "Z1", "Z2", 8, 9)])

        self.assertEqual(1, len(solutions))
        self.assertEqual(("z",), solutions[0].segment_coverage)
        self.assertEqual({"USD": Decimal("2.00")}, solutions[0].costs)

    def test_contains_rule_and_unrestricted_rule(self):
        _add_fare(self.schedule, self.feed_id, "h", "1.00")
        _add_rule(self.schedule, self.feed_id, "h", contains_id="Z3")
        _add_fare(self.schedule, self.feed_id, "any", "9.00")
        _add_rule(self.schedule, self.feed_id, "any")

        solutions = self._plan([
            _leg("T1", "R1", "S1", "S2", "Z1", "Z2", 8, 9),
            _leg("T2", "R2", "S2", "S3", "Z3", "Z4", 9, 10)])

        self.assertEqual(1, len(solutions))
        self.assertEqual(("any", "h"), solutions[0].segment_coverage)
        self.assertEqual({"USD": Decimal("10.00")}, solutions[0].costs)

    def test_fare_without_rules_applies_everywhere(self):
        _add_fare(self.schedule, self.feed_id, "generic", "2.00",
                  transfers=1)

        solutions = self._plan([
            _leg("T1", "R9", "S9", "S8", "Z9", "Z8", 8, 9),
            _leg("T2", "R8", "S8", "S7", "Z8", "Z7", 9, 10)])

        self.assertEqual(1, len(solutions))
        self.assertEqual(("generic", "generic"),
                         solutions[0].segment_coverage)
        self.assertEqual(1, len(solutions[0].tickets))
        self.assertEqual({"USD": Decimal("2.00")}, solutions[0].costs)

    def test_no_legal_ticket_returns_empty(self):
        _add_fare(self.schedule, self.feed_id, "z", "2.00")
        _add_rule(self.schedule, self.feed_id, "z",
                  origin_id="Z1", destination_id="Z2")

        solutions = self._plan([
            _leg("T1", "R9", "S9", "S8", "Z9", "Z8", 8, 9)])

        self.assertEqual([], solutions)

    def test_transfer_limit_blocks_multi_leg_ticket(self):
        _add_fare(self.schedule, self.feed_id, "single", "4", transfers=0)
        _add_rule(self.schedule, self.feed_id, "single")
        _add_fare(self.schedule, self.feed_id, "pass", "10", transfers=1)
        _add_rule(self.schedule, self.feed_id, "pass")

        legs = [
            _leg("T1", "R1", "S1", "S2", "Z1", "Z2", 8, 8.5),
            _leg("T2", "R2", "S2", "S3", "Z2", "Z3", 9, 9.5),
            _leg("T3", "R3", "S3", "S4", "Z3", "Z4", 10, 10.5)]
        solutions = self._plan(legs)

        self.assertEqual(1, len(solutions))
        best = solutions[0]
        self.assertEqual(("single", "single", "single"),
                         best.segment_coverage)
        self.assertEqual({"USD": Decimal("12")}, best.costs)
        for use in best.tickets:
            self.assertLessEqual(use.transfers_used, 1)

    def test_unlimited_transfers(self):
        _add_fare(self.schedule, self.feed_id, "single", "4", transfers=0)
        _add_rule(self.schedule, self.feed_id, "single")
        _add_fare(self.schedule, self.feed_id, "day", "2", transfers=None)
        _add_rule(self.schedule, self.feed_id, "day")

        legs = [
            _leg("T1", "R1", "S1", "S2", "Z1", "Z2", 8, 8.5),
            _leg("T2", "R2", "S2", "S3", "Z2", "Z3", 9, 9.5),
            _leg("T3", "R3", "S3", "S4", "Z3", "Z4", 10, 10.5)]
        solutions = self._plan(legs)

        self.assertEqual(1, len(solutions))
        self.assertEqual(("day", "day", "day"),
                         solutions[0].segment_coverage)
        self.assertEqual({"USD": Decimal("2")}, solutions[0].costs)
        self.assertEqual(1, len(solutions[0].tickets))
        self.assertEqual(2, solutions[0].tickets[0].transfers_used)

    def test_missing_times_skip_duration_check(self):
        _add_fare(self.schedule, self.feed_id, "timed", "10",
                  transfers=1, transfer_duration=60)
        _add_rule(self.schedule, self.feed_id, "timed")

        legs = [
            JourneyLeg("T1", "R1", "S1", "S2", "Z1", "Z2", None, None),
            JourneyLeg("T2", "R2", "S2", "S3", "Z2", "Z3", None, None)]
        solutions = self._plan(legs)

        self.assertEqual(1, len(solutions))
        self.assertEqual(1, len(solutions[0].tickets))
        self.assertEqual(("timed", "timed"),
                         solutions[0].segment_coverage)

    def test_same_fare_bought_twice_charges_twice(self):
        _add_fare(self.schedule, self.feed_id, "single", "3", transfers=0)
        _add_rule(self.schedule, self.feed_id, "single")
        _add_fare(self.schedule, self.feed_id, "pass", "7", transfers=1)
        _add_rule(self.schedule, self.feed_id, "pass")

        legs = [
            _leg("T1", "R1", "S1", "S2", "Z1", "Z2", 8, 8.5),
            _leg("T2", "R2", "S2", "S3", "Z2", "Z3", 9, 9.5)]
        solutions = self._plan(legs)

        self.assertEqual(1, len(solutions))
        solution = solutions[0]
        self.assertEqual(("single", "single"), solution.segment_coverage)
        self.assertEqual(2, len(solution.tickets))
        self.assertEqual([(0,), (1,)],
                         [use.covered for use in solution.tickets])
        self.assertEqual({"USD": Decimal("6")}, solution.costs)

    def test_transfer_duration_expired(self):
        _add_fare(self.schedule, self.feed_id, "timed", "10",
                  transfers=1, transfer_duration=3600)
        _add_rule(self.schedule, self.feed_id, "timed")
        _add_fare(self.schedule, self.feed_id, "single", "6", transfers=0)
        _add_rule(self.schedule, self.feed_id, "single")

        legs = [
            _leg("T1", "R1", "S1", "S2", "Z1", "Z2", 8, 8.5),
            _leg("T2", "R2", "S2", "S3", "Z2", "Z3", 9, 9.5)]
        solutions = self._plan(legs)

        self.assertEqual(1, len(solutions))
        self.assertEqual(("single", "single"),
                         solutions[0].segment_coverage)
        self.assertEqual({"USD": Decimal("12")}, solutions[0].costs)

    def test_transfer_duration_within_window_charged_once(self):
        _add_fare(self.schedule, self.feed_id, "timed", "10",
                  transfers=1, transfer_duration=3600)
        _add_rule(self.schedule, self.feed_id, "timed")
        _add_fare(self.schedule, self.feed_id, "single", "6", transfers=0)
        _add_rule(self.schedule, self.feed_id, "single")

        legs = [
            _leg("T1", "R1", "S1", "S2", "Z1", "Z2", 8, 8.5),
            _leg("T2", "R2", "S2", "S3", "Z2", "Z3", 8.5833, 9)]
        solutions = self._plan(legs)

        self.assertEqual(1, len(solutions))
        solution = solutions[0]
        self.assertEqual(("timed", "timed"), solution.segment_coverage)
        self.assertEqual(1, len(solution.tickets))
        self.assertEqual({"USD": Decimal("10")}, solution.costs)
        self.assertEqual(
            (datetime.timedelta(hours=8), datetime.timedelta(hours=9)),
            (solution.tickets[0].first_departure,
             solution.tickets[0].last_arrival))

    def test_per_leg_cheapest_is_not_journey_cheapest(self):
        _add_fare(self.schedule, self.feed_id, "single", "1.00",
                  transfers=0)
        _add_rule(self.schedule, self.feed_id, "single")
        _add_fare(self.schedule, self.feed_id, "pass", "1.50",
                  transfers=1)
        _add_rule(self.schedule, self.feed_id, "pass")

        legs = [
            _leg("T1", "R1", "S1", "S2", "Z1", "Z2", 8, 8.5),
            _leg("T2", "R2", "S2", "S3", "Z2", "Z3", 9, 9.5)]
        solutions = self._plan(legs)

        self.assertEqual(1, len(solutions))
        solution = solutions[0]
        self.assertEqual(("pass", "pass"), solution.segment_coverage)
        self.assertEqual(1, len(solution.tickets))
        self.assertEqual({"USD": Decimal("1.50")}, solution.costs)

    def test_dominated_solutions_are_pruned(self):
        _add_fare(self.schedule, self.feed_id, "single", "1", transfers=0)
        _add_rule(self.schedule, self.feed_id, "single")
        _add_fare(self.schedule, self.feed_id, "pass", "3", transfers=1)
        _add_rule(self.schedule, self.feed_id, "pass")

        legs = [
            _leg("T1", "R1", "S1", "S2", "Z1", "Z2", 8, 8.5),
            _leg("T2", "R2", "S2", "S3", "Z2", "Z3", 9, 9.5)]
        solutions = self._plan(legs)

        self.assertEqual(1, len(solutions))
        self.assertEqual({"USD": Decimal("2")}, solutions[0].costs)

    def test_multiple_currencies_kept_separate(self):
        _add_fare(self.schedule, self.feed_id, "us", "3",
                  currency="USD", transfers=0)
        _add_rule(self.schedule, self.feed_id, "us")
        _add_fare(self.schedule, self.feed_id, "eu", "2",
                  currency="EUR", transfers=0)
        _add_rule(self.schedule, self.feed_id, "eu")

        legs = [
            _leg("T1", "R1", "S1", "S2", "Z1", "Z2", 8, 8.5),
            _leg("T2", "R2", "S2", "S3", "Z2", "Z3", 9, 9.5)]
        solutions = self._plan(legs)

        cost_vectors = {tuple(sorted(solution.costs.items()))
                        for solution in solutions}
        self.assertEqual(
            {(("EUR", Decimal("2")), ("USD", Decimal("3"))),
             (("EUR", Decimal("4")),),
             (("USD", Decimal("6")),)},
            cost_vectors)
        mixed = next(solution for solution in solutions
                     if set(solution.costs) == {"EUR", "USD"})
        self.assertEqual(
            {"EUR": Decimal("2"), "USD": Decimal("3")}, mixed.costs)
        self.assertEqual(2, len(mixed.tickets))
        self.assertNotEqual(
            mixed.tickets[0].fare_id, mixed.tickets[1].fare_id)

    def test_invalid_legs_raise(self):
        _add_fare(self.schedule, self.feed_id, "single", "1")
        _add_rule(self.schedule, self.feed_id, "single")
        self.schedule.session.commit()
        planner = FarePlanner(self.schedule)

        with self.assertRaises(ValueError):
            planner.plan(())
        with self.assertRaises(ValueError):
            planner.plan([_leg("T1", "R1", "S1", "S1", "Z1", "Z1", 8, 9)])
        with self.assertRaises(ValueError):
            planner.plan([_leg("T1", "R1", "S1", "S2", "Z1", "Z2", 9, 8)])
        with self.assertRaises(ValueError):
            planner.plan([
                _leg("T1", "R1", "S1", "S2", "Z1", "Z2", 8, 9),
                _leg("T2", "R2", "S2", "S3", "Z2", "Z3", 8.5, 9.5)])

    def test_plan_fares_convenience_function(self):
        _add_fare(self.schedule, self.feed_id, "single", "1")
        _add_rule(self.schedule, self.feed_id, "single")
        self.schedule.session.commit()

        solutions = plan_fares(self.schedule, [
            _leg("T1", "R1", "S1", "S2", "Z1", "Z2", 8, 9)])

        self.assertEqual(1, len(solutions))
        self.assertEqual(("single",), solutions[0].segment_coverage)

    def test_multiple_feeds_require_explicit_feed_id(self):
        self.schedule.session.add(Feed(feed_name="second"))
        self.schedule.session.commit()

        with self.assertRaises(ValueError):
            FarePlanner(self.schedule)


class TestFarePlannerSampleFeed(unittest.TestCase):
    def setUp(self):
        self.schedule = Schedule(":memory:")
        data_location = os.path.join(os.path.dirname(__file__),
                                     "data", "sample_feed")
        from pygtfs import overwrite_feed
        overwrite_feed(self.schedule, data_location)

    def test_two_route_legs_need_two_zero_transfer_tickets(self):
        legs = [
            JourneyLeg("AB1", "AB", "BEATTY_AIRPORT", "BULLFROG",
                       None, None,
                       datetime.timedelta(hours=8),
                       datetime.timedelta(hours=8, minutes=10)),
            JourneyLeg("BFC1", "BFC", "BULLFROG", "FUR_CREEK_RES",
                       None, None,
                       datetime.timedelta(hours=8, minutes=20),
                       datetime.timedelta(hours=9, minutes=20))]

        solutions = FarePlanner(self.schedule).plan(legs)

        self.assertEqual(1, len(solutions))
        solution = solutions[0]
        self.assertEqual(("p", "p"), solution.segment_coverage)
        self.assertEqual(2, len(solution.tickets))
        self.assertEqual(
            {"USD": Decimal("2.50")}, solution.costs)

    def test_different_routes_pick_matching_fares(self):
        legs = [
            JourneyLeg("AB2", "AB", "BULLFROG", "BEATTY_AIRPORT",
                       None, None,
                       datetime.timedelta(hours=12, minutes=5),
                       datetime.timedelta(hours=12, minutes=15)),
            JourneyLeg("AAMV3", "AAMV", "BEATTY_AIRPORT", "AMV",
                       None, None,
                       datetime.timedelta(hours=13),
                       datetime.timedelta(hours=14))]

        solutions = FarePlanner(self.schedule).plan(legs)

        self.assertEqual(1, len(solutions))
        self.assertEqual(("p", "a"), solutions[0].segment_coverage)
        self.assertEqual(
            {"USD": Decimal("6.50")}, solutions[0].costs)
        self.assertEqual(
            [("p", (0,)), ("a", (1,))],
            [(use.fare_id, use.covered) for use in solutions[0].tickets])


if __name__ == '__main__':
    unittest.main()
