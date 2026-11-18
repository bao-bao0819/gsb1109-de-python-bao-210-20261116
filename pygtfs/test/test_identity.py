# -*- coding: utf-8 -*-

import os
import shutil
import tempfile
import unittest

from pygtfs import (Schedule, append_feed, add_alias, add_aliases,
                    add_aliases_from_file, get_identity,
                    identities_for_key, record_lineage, lineage_chain,
                    register_all_identities)
from pygtfs.exceptions import PygtfsException
from pygtfs.gtfs_entities import EntityIdentity, EntityAlias, FeedLineage

_DATA = os.path.join(os.path.dirname(__file__), "data", "multifeed")
ALPHA_FEED = os.path.join(_DATA, "feed_alpha")
BETA_FEED = os.path.join(_DATA, "feed_beta")


class MultiFeedTestBase(unittest.TestCase):
    def setUp(self):
        self.schedule = Schedule(":memory:")
        append_feed(self.schedule, ALPHA_FEED)
        append_feed(self.schedule, BETA_FEED)
        self.alpha_feed_id = self.schedule.feeds[0].feed_id
        self.beta_feed_id = self.schedule.feeds[1].feed_id

    def route_identity(self, feed_id):
        return get_identity(self.schedule.session, feed_id, "route", "1")


class TestSourceScopedIdentity(MultiFeedTestBase):
    def test_two_feeds_are_kept_distinct(self):
        self.assertEqual(2, len(self.schedule.feeds))
        self.assertNotEqual(self.alpha_feed_id, self.beta_feed_id)

    def test_colliding_route_id_gets_distinct_identities(self):
        # Both agencies publish route_id "1". Their identities must not merge.
        alpha_route = self.route_identity(self.alpha_feed_id)
        beta_route = self.route_identity(self.beta_feed_id)
        self.assertIsNotNone(alpha_route)
        self.assertIsNotNone(beta_route)
        self.assertNotEqual(alpha_route.identity_id, beta_route.identity_id)
        self.assertNotEqual(alpha_route.identity_key, beta_route.identity_key)

    def test_colliding_route_id_returns_two_feed_scoped_rows(self):
        routes = self.schedule.routes_by_id("1")
        self.assertEqual(2, len(routes))
        agencies = sorted(route.agency_id for route in routes)
        self.assertEqual(["ALPHA", "BETA"], agencies)
        feed_ids = sorted(route.feed_id for route in routes)
        self.assertEqual(sorted([self.alpha_feed_id, self.beta_feed_id]),
                         feed_ids)

    def test_route_detail_rows_do_not_cross_feeds(self):
        # The detail (trips) hanging off each route stays inside its own feed,
        # which is the original "mingled details" bug being guarded against.
        routes = self.schedule.routes_by_id("1")
        trips_by_agency = {route.agency_id:
                           sorted(trip.trip_id for trip in route.trips)
                           for route in routes}
        self.assertEqual({"ALPHA": ["T1"], "BETA": ["T9"]},
                         trips_by_agency)

    def test_shared_stop_does_not_imply_shared_identity(self):
        # The two feeds reuse stop id SHARED_STOP at identical coordinates,
        # but a shared stop must not be treated as a shared entity by default.
        alpha_stop = get_identity(self.schedule.session,
                                  self.alpha_feed_id, "stop", "SHARED_STOP")
        beta_stop = get_identity(self.schedule.session,
                                 self.beta_feed_id, "stop", "SHARED_STOP")
        self.assertNotEqual(alpha_stop.identity_key, beta_stop.identity_key)
        stops = self.schedule.stops_by_id("SHARED_STOP")
        self.assertEqual(2, len(stops))

    def test_identity_registered_for_every_identified_entity(self):
        for entity_type in ("agency", "stop", "route", "trip", "service"):
            rows = (self.schedule.entity_identities_query
                    .filter(EntityIdentity.entity_type == entity_type).all())
            self.assertTrue(rows, "no identities registered for " + entity_type)
            for row in rows:
                self.assertEqual(row.identity_id, row.identity_key)

    def test_identity_natural_key_is_feed_scoped(self):
        alpha = self.route_identity(self.alpha_feed_id)
        beta = self.route_identity(self.beta_feed_id)
        self.assertEqual(("route", "1"),
                         (alpha.entity_type, alpha.source_id))
        self.assertEqual(("route", "1"),
                         (beta.entity_type, beta.source_id))
        self.assertEqual((alpha.feed_id, alpha.entity_type, alpha.source_id),
                         (self.alpha_feed_id, "route", "1"))


class TestExplicitAlias(MultiFeedTestBase):
    def test_alias_is_required_to_share_an_identity_key(self):
        alpha_route = self.route_identity(self.alpha_feed_id)
        beta_route = self.route_identity(self.beta_feed_id)
        self.assertNotEqual(alpha_route.identity_key, beta_route.identity_key)

        shared_key = add_alias(self.schedule.session, "route",
                               self.alpha_feed_id, "1",
                               self.beta_feed_id, "1")

        group = identities_for_key(self.schedule.session, "route", shared_key)
        self.assertEqual(2, len(group))
        self.assertEqual(shared_key, group[0].identity_key)
        self.assertEqual(shared_key, group[1].identity_key)
        self.assertEqual(1, self.schedule.entity_aliases_query.count())

    def test_alias_does_not_merge_underlying_feed_rows(self):
        add_alias(self.schedule.session, "route",
                  self.alpha_feed_id, "1", self.beta_feed_id, "1")
        # Even after aliasing, each feed keeps its own route and detail rows.
        self.assertEqual(2, len(self.schedule.routes_by_id("1")))
        routes = self.schedule.routes_by_id("1")
        trip_ids = sorted(trip.trip_id for route in routes
                          for trip in route.trips)
        self.assertEqual(["T1", "T9"], trip_ids)

    def test_alias_is_idempotent_in_either_direction(self):
        add_alias(self.schedule.session, "route",
                  self.alpha_feed_id, "1", self.beta_feed_id, "1")
        key = add_alias(self.schedule.session, "route",
                        self.beta_feed_id, "1", self.alpha_feed_id, "1")
        self.assertEqual(1, self.schedule.entity_aliases_query.count())
        self.assertEqual(2, len(identities_for_key(
            self.schedule.session, "route", key)))

    def test_alias_unions_groups_transitively(self):
        # Group the alpha shared stop with the beta shared stop, then pull the
        # alpha-only stop into the same group through a second alias.
        add_alias(self.schedule.session, "stop",
                  self.alpha_feed_id, "SHARED_STOP",
                  self.beta_feed_id, "SHARED_STOP")
        key = add_alias(self.schedule.session, "stop",
                        self.beta_feed_id, "SHARED_STOP",
                        self.alpha_feed_id, "ALPHA_STOP")
        group = identities_for_key(self.schedule.session, "stop", key)
        self.assertEqual(3, len(group))
        self.assertTrue(all(row.identity_key == key for row in group))

    def test_alias_requires_registered_identities(self):
        with self.assertRaises(PygtfsException):
            add_alias(self.schedule.session, "route",
                      self.alpha_feed_id, "1",
                      self.beta_feed_id, "DOES_NOT_EXIST")

    def test_alias_rejects_unknown_entity_type(self):
        with self.assertRaises(PygtfsException):
            add_alias(self.schedule.session, "spaceship",
                      self.alpha_feed_id, "1", self.beta_feed_id, "1")

    def test_add_aliases_from_file(self):
        csv_text = (
            "# explicit cross-agency mapping\n"
            "entity_type,feed_id_a,source_id_a,feed_id_b,source_id_b\n"
            "route,{a},1,{b},1\n"
            "stop,{a},SHARED_STOP,{b},SHARED_STOP\n"
        ).format(a=self.alpha_feed_id, b=self.beta_feed_id)
        with tempfile.NamedTemporaryFile("w", suffix=".csv",
                                         delete=False) as handle:
            handle.write(csv_text)
            alias_file = handle.name
        try:
            keys = add_aliases_from_file(self.schedule.session, alias_file)
        finally:
            os.unlink(alias_file)
        self.assertEqual(2, len(keys))
        self.assertEqual(2, self.schedule.entity_aliases_query.count())

    def test_add_aliases_helper(self):
        keys = add_aliases(self.schedule.session, [
            ("route", self.alpha_feed_id, "1", self.beta_feed_id, "1"),
        ])
        self.assertEqual(1, len(keys))


class TestFeedLineage(unittest.TestCase):
    def setUp(self):
        self.schedule = Schedule(":memory:")
        append_feed(self.schedule, ALPHA_FEED)
        self.first_feed_id = self.schedule.feeds[0].feed_id
        # A second publication of the same upstream feed, kept separately.
        self.publication_dir = tempfile.mkdtemp()
        self.second_publication = os.path.join(self.publication_dir,
                                               "feed_alpha_v2")
        shutil.copytree(ALPHA_FEED, self.second_publication)
        append_feed(self.schedule, self.second_publication,
                    predecessor_feed_id=self.first_feed_id,
                    lineage_note="second publication")
        self.second_feed_id = self.schedule.feeds[1].feed_id

    def tearDown(self):
        shutil.rmtree(self.publication_dir, ignore_errors=True)

    def test_lineage_link_is_recorded_separately(self):
        links = self.schedule.feed_lineages
        self.assertEqual(1, len(links))
        link = links[0]
        self.assertEqual(self.second_feed_id, link.feed_id)
        self.assertEqual(self.first_feed_id, link.predecessor_feed_id)
        self.assertEqual("second publication", link.note)
        self.assertEqual([self.second_feed_id, self.first_feed_id],
                         lineage_chain(self.schedule.session,
                                       self.second_feed_id))
        self.assertEqual([self.first_feed_id],
                         lineage_chain(self.schedule.session,
                                       self.first_feed_id))

    def test_lineage_does_not_merge_or_delete_old_publication(self):
        # Both publications remain, each with its own feed-scoped route row.
        self.assertEqual(2, len(self.schedule.routes_by_id("1")))
        self.assertEqual(2, len(self.schedule.feeds))
        first_route = self.route_identity(self.first_feed_id)
        second_route = self.route_identity(self.second_feed_id)
        self.assertNotEqual(first_route.identity_key, second_route.identity_key)

    def route_identity(self, feed_id):
        return get_identity(self.schedule.session, feed_id, "route", "1")

    def test_old_publication_detail_is_still_reviewable(self):
        old_trips = (self.schedule.trips_query
                     .filter_by(feed_id=self.first_feed_id).all())
        self.assertEqual(["T1"], [trip.trip_id for trip in old_trips])
        old_stop_times = (self.schedule.stop_times_query
                          .filter_by(feed_id=self.first_feed_id).all())
        self.assertEqual(2, len(old_stop_times))

    def test_recording_same_lineage_link_is_idempotent(self):
        record_lineage(self.schedule.session, self.second_feed_id,
                       self.first_feed_id)
        self.assertEqual(1, len(self.schedule.feed_lineages))

    def test_a_publication_has_at_most_one_predecessor(self):
        # A valid, distinct third feed cannot replace the existing predecessor.
        append_feed(self.schedule, BETA_FEED)
        third_feed_id = self.schedule.feeds[2].feed_id
        self.assertEqual(1, len(self.schedule.feed_lineages))

        with self.assertRaises(PygtfsException):
            record_lineage(self.schedule.session, self.second_feed_id,
                           third_feed_id)
        # The original link is left untouched.
        link = self.schedule.feed_lineages[0]
        self.assertEqual(self.first_feed_id, link.predecessor_feed_id)

    def test_record_lineage_rejects_unknown_or_self_referential_feeds(self):
        with self.assertRaises(PygtfsException):
            record_lineage(self.schedule.session, self.second_feed_id,
                           self.second_feed_id)
        with self.assertRaises(PygtfsException):
            record_lineage(self.schedule.session, self.second_feed_id, 999)


class TestIdentityCascade(unittest.TestCase):
    def test_dropping_feed_removes_its_identities_and_alias_links(self):
        schedule = Schedule(":memory:")
        append_feed(schedule, ALPHA_FEED)
        append_feed(schedule, BETA_FEED)
        alpha_id = schedule.feeds[0].feed_id
        beta_id = schedule.feeds[1].feed_id
        add_alias(schedule.session, "route", alpha_id, "1", beta_id, "1")
        self.assertEqual(1, schedule.entity_aliases_query.count())

        schedule.drop_feed(alpha_id)

        remaining_identities = (schedule.entity_identities_query
                                .filter(EntityIdentity.feed_id == alpha_id)
                                .count())
        self.assertEqual(0, remaining_identities)
        # The alias referenced the dropped feed, so it is cascade-deleted too.
        self.assertEqual(0, schedule.entity_aliases_query.count())
        surviving = schedule.entity_identities_query.all()
        self.assertTrue(all(row.feed_id == beta_id for row in surviving))


class TestIdentityBackfill(MultiFeedTestBase):
    def test_backfill_registers_missing_identities_for_all_feeds(self):
        # Simulate a pre-identity database by clearing the auto-registered rows.
        self.schedule.session.query(EntityIdentity).delete()
        self.schedule.session.commit()
        self.assertEqual(0, self.schedule.entity_identities_query.count())

        created = register_all_identities(self.schedule.session)
        self.schedule.session.commit()

        self.assertGreater(created, 0)
        # Both feeds' colliding route id are present as distinct identities.
        alpha = self.route_identity(self.alpha_feed_id)
        beta = self.route_identity(self.beta_feed_id)
        self.assertIsNotNone(alpha)
        self.assertIsNotNone(beta)
        self.assertNotEqual(alpha.identity_key, beta.identity_key)

    def test_backfill_is_idempotent(self):
        first = register_all_identities(self.schedule.session)
        second = register_all_identities(self.schedule.session)
        self.assertEqual(0, first)
        self.assertEqual(0, second)


class TestLineageCascade(MultiFeedTestBase):
    def test_dropping_predecessor_removes_lineage_without_orphans(self):
        record_lineage(self.schedule.session, self.beta_feed_id,
                       self.alpha_feed_id, note="continuation")
        self.assertEqual(1, self.schedule.feed_lineages_query.count())

        self.schedule.drop_feed(self.alpha_feed_id)

        # The lineage row referenced the dropped predecessor and is removed;
        # the surviving feed keeps its own identities with no orphans.
        self.assertEqual(0, self.schedule.feed_lineages_query.count())
        surviving = self.schedule.entity_identities_query.all()
        self.assertTrue(surviving)
        self.assertTrue(all(row.feed_id == self.beta_feed_id
                            for row in surviving))


if __name__ == "__main__":
    unittest.main()
