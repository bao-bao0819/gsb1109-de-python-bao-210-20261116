# -*- coding: utf-8 -*-

import unittest

from pygtfs import Schedule
from pygtfs.gtfs_entities import Stop, Transfer
from pygtfs.transfer import (SCOPE_STOP, SCOPE_ROUTE, SCOPE_TRIP,
                             STOP_SOURCE_STOP, STOP_SOURCE_PARENT,
                             TransferCandidate, transfer_candidates,
                             preferred_transfer, walking_time_seconds,
                             same_named_stops, rule_scope)

FEED = 1
OTHER_FEED = 2


def make_stop(stop_id, name, parent=None, location_type=None, feed_id=FEED):
    return Stop(feed_id=feed_id, stop_id=stop_id, stop_name=name,
                stop_lat=0.0, stop_lon=0.0, parent_station=parent,
                location_type=location_type)


def make_transfer(from_stop, to_stop, transfer_type=None,
                  min_transfer_time=None, from_route_id='', to_route_id='',
                  from_trip_id='', to_trip_id='', feed_id=FEED):
    return Transfer(feed_id=feed_id, from_stop_id=from_stop, to_stop_id=to_stop,
                    from_route_id=from_route_id, to_route_id=to_route_id,
                    from_trip_id=from_trip_id, to_trip_id=to_trip_id,
                    transfer_type=transfer_type,
                    min_transfer_time=min_transfer_time)


class TransferRuleTest(unittest.TestCase):
    def setUp(self):
        self.schedule = Schedule(":memory:")
        self.session = self.schedule.session
        self.session.add_all([
            make_stop('STN_A', 'Central', location_type=1),
            make_stop('STN_A_P1', 'Central', parent='STN_A', location_type=3),
            make_stop('STN_A_P2', 'Central', parent='STN_A', location_type=3),
            make_stop('STN_B', 'Mall', location_type=1),
            make_stop('STN_B_P1', 'Mall', parent='STN_B', location_type=3),
            make_stop('STN_C', 'Central'),
            make_stop('OTHER_X', 'Central', feed_id=OTHER_FEED),
            make_stop('OTHER_Y', 'Mall', feed_id=OTHER_FEED),
        ])
        self.session.add_all([
            # Generic / route / trip rules sharing one pair of stations.
            make_transfer('STN_A', 'STN_B', 0, 120),
            make_transfer('STN_A', 'STN_B', 2, 180,
                          from_route_id='R1', to_route_id='R2'),
            make_transfer('STN_A', 'STN_B', 2, 60,
                          from_trip_id='T1A', to_trip_id='T2B'),
            make_transfer('STN_A', 'STN_B', 2, 90,
                          from_route_id='RX', to_route_id='RY'),
            # Trip-specific prohibition on the same pair.
            make_transfer('STN_A', 'STN_B', 3,
                          from_trip_id='FX', to_trip_id='FY'),
            # Allowed generic rule directly on a platform; a route-specific
            # prohibition inherited from the parent station competes with it.
            make_transfer('STN_A_P1', 'STN_B', 0, 200),
            make_transfer('STN_A', 'STN_B', 3,
                          from_route_id='RF', to_route_id='RG'),
            # Direct platform rule competing with the parent-station rule.
            make_transfer('STN_A_P1', 'STN_B_P1', 0, 30),
            # Documented transfer without any walking time evidence.
            make_transfer('STN_A_P2', 'STN_B', 0, None),
            # Explicit zero-minute transfer.
            make_transfer('STN_A_P2', 'STN_B_P1', 0, 0),
            # A rule in another feed must never leak across feeds.
            make_transfer('STN_A', 'STN_B', 0, 999, feed_id=OTHER_FEED),
        ])
        self.session.commit()

    def stop(self, stop_id):
        return self.session.query(Stop).filter(
            Stop.feed_id == FEED, Stop.stop_id == stop_id).one()

    def test_every_candidate_keeps_its_source_row(self):
        candidates = transfer_candidates(
            self.schedule, self.stop('STN_A'), self.stop('STN_B'),
            from_route_id='R1', to_route_id='R2')
        self.assertTrue(candidates)
        for candidate in candidates:
            self.assertIsInstance(candidate, TransferCandidate)
            self.assertIsInstance(candidate.source, Transfer)
            self.assertEqual(candidate.source.from_stop_id, 'STN_A')
            self.assertEqual(candidate.source.to_stop_id, 'STN_B')

    def test_generic_rule_matches_without_route_or_trip_context(self):
        candidate = preferred_transfer(
            self.schedule, self.stop('STN_A'), self.stop('STN_B'))
        self.assertEqual(candidate.scope, SCOPE_STOP)
        self.assertEqual(candidate.min_transfer_time, 120)

    def test_route_rule_beats_generic_rule(self):
        candidate = preferred_transfer(
            self.schedule, self.stop('STN_A'), self.stop('STN_B'),
            from_route_id='R1', to_route_id='R2')
        self.assertEqual(candidate.scope, SCOPE_ROUTE)
        self.assertEqual(candidate.min_transfer_time, 180)

    def test_route_rule_requires_matching_route_context(self):
        candidate = preferred_transfer(
            self.schedule, self.stop('STN_A'), self.stop('STN_B'),
            from_route_id='R9', to_route_id='R9')
        self.assertEqual(candidate.scope, SCOPE_STOP)
        self.assertEqual(candidate.min_transfer_time, 120)

    def test_trip_rule_beats_route_and_generic_rules(self):
        candidate = preferred_transfer(
            self.schedule, self.stop('STN_A'), self.stop('STN_B'),
            from_route_id='R1', to_route_id='R2',
            from_trip_id='T1A', to_trip_id='T2B')
        self.assertEqual(candidate.scope, SCOPE_TRIP)
        self.assertEqual(candidate.min_transfer_time, 60)

    def test_trip_specific_rule_ignored_without_trip_context(self):
        candidates = transfer_candidates(
            self.schedule, self.stop('STN_A'), self.stop('STN_B'))
        scopes = {candidate.scope for candidate in candidates}
        self.assertNotIn(SCOPE_TRIP, scopes)

    def test_forbidden_rule_wins_over_allowed_rule(self):
        # The parent station's route-specific prohibition overrides the
        # generic rule stored directly on the platform.
        candidate = preferred_transfer(
            self.schedule, self.stop('STN_A_P1'), self.stop('STN_B'),
            from_route_id='RF', to_route_id='RG')
        self.assertTrue(candidate.is_forbidden)
        self.assertEqual(candidate.transfer_type, 3)
        self.assertIsNone(
            walking_time_seconds(
                self.schedule, self.stop('STN_A_P1'), self.stop('STN_B'),
                from_route_id='RF', to_route_id='RG'))

    def test_prohibition_does_not_apply_without_its_route_context(self):
        candidate = preferred_transfer(
            self.schedule, self.stop('STN_A_P1'), self.stop('STN_B'))
        self.assertFalse(candidate.is_forbidden)
        self.assertEqual(candidate.min_transfer_time, 200)

    def test_trip_specific_prohibition_needs_matching_trip(self):
        allowed = preferred_transfer(
            self.schedule, self.stop('STN_A'), self.stop('STN_B'))
        self.assertFalse(allowed.is_forbidden)
        forbidden = preferred_transfer(
            self.schedule, self.stop('STN_A'), self.stop('STN_B'),
            from_trip_id='FX', to_trip_id='FY')
        self.assertTrue(forbidden.is_forbidden)
        self.assertEqual(forbidden.scope, SCOPE_TRIP)

    def test_parent_station_rule_applies_to_platforms(self):
        candidates = transfer_candidates(
            self.schedule, self.stop('STN_A_P1'), self.stop('STN_B_P1'))
        sources = {(candidate.from_stop_source, candidate.to_stop_source)
                   for candidate in candidates}
        self.assertIn((STOP_SOURCE_PARENT, STOP_SOURCE_PARENT), sources)
        self.assertIn((STOP_SOURCE_STOP, STOP_SOURCE_STOP), sources)

    def test_direct_platform_rule_beats_parent_station_rule(self):
        candidate = preferred_transfer(
            self.schedule, self.stop('STN_A_P1'), self.stop('STN_B_P1'))
        self.assertEqual(candidate.from_stop_source, STOP_SOURCE_STOP)
        self.assertEqual(candidate.to_stop_source, STOP_SOURCE_STOP)
        self.assertEqual(candidate.min_transfer_time, 30)

    def test_missing_walking_time_is_unknown_not_zero(self):
        candidate = preferred_transfer(
            self.schedule, self.stop('STN_A_P2'), self.stop('STN_B'))
        self.assertIsNone(candidate.min_transfer_time)
        self.assertIsNone(
            walking_time_seconds(
                self.schedule, self.stop('STN_A_P2'), self.stop('STN_B')))

    def test_explicit_zero_walking_time_is_preserved(self):
        self.assertEqual(
            walking_time_seconds(
                self.schedule, self.stop('STN_A_P2'),
                self.stop('STN_B_P1')),
            0)

    def test_same_name_without_rule_is_not_a_transfer(self):
        candidates = transfer_candidates(
            self.schedule, self.stop('STN_A'), self.stop('STN_C'))
        self.assertEqual(candidates, [])
        self.assertIsNone(
            preferred_transfer(
                self.schedule, self.stop('STN_A'), self.stop('STN_C')))
        self.assertIsNone(
            walking_time_seconds(
                self.schedule, self.stop('STN_A'), self.stop('STN_C')))

    def test_same_name_platforms_without_rule_have_no_candidate(self):
        self.assertEqual(
            transfer_candidates(
                self.schedule, self.stop('STN_A_P1'),
                self.stop('STN_A_P2')),
            [])

    def test_same_named_stops_helper_does_not_imply_transfer(self):
        namesakes = same_named_stops(self.schedule, self.stop('STN_A'))
        self.assertEqual(
            sorted(stop.stop_id for stop in namesakes),
            ['STN_A_P1', 'STN_A_P2', 'STN_C'])
        self.assertEqual(
            transfer_candidates(
                self.schedule, self.stop('STN_A'), self.stop('STN_C')),
            [])

    def test_rules_do_not_leak_across_feeds(self):
        candidate = preferred_transfer(
            self.schedule, self.stop('STN_A'), self.stop('STN_B'))
        self.assertEqual(candidate.min_transfer_time, 120)

    def test_rule_scope_classification(self):
        generic = self.session.query(Transfer).filter(
            Transfer.from_route_id == '', Transfer.from_trip_id == '').first()
        route_rule = self.session.query(Transfer).filter(
            Transfer.from_route_id == 'R1').one()
        trip_rule = self.session.query(Transfer).filter(
            Transfer.from_trip_id == 'T1A').one()
        self.assertEqual(rule_scope(generic), SCOPE_STOP)
        self.assertEqual(rule_scope(route_rule), SCOPE_ROUTE)
        self.assertEqual(rule_scope(trip_rule), SCOPE_TRIP)


if __name__ == '__main__':
    unittest.main()
