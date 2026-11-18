# -*- coding: utf-8 -*-

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from sqlalchemy import text

from pygtfs import Schedule, append_feed, diff_feed, update_feed
from pygtfs.diff import FeedDiff, FeedSnapshot
from pygtfs.gtfs_entities import (Frequency, ShapePoint, StopTime, Transfer,
                                  _stop_translations)


SAMPLE = Path(os.path.dirname(__file__)) / "data" / "sample_feed"


def _rewrite(path, lines):
    Path(path).write_text("\n".join(lines) + "\n")


class _RevisionMixin(object):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.old = Path(self.temp_dir) / "old_feed"
        self.new = Path(self.temp_dir) / "new_feed"
        shutil.copytree(str(SAMPLE), str(self.old))
        shutil.copytree(str(SAMPLE), str(self.new))

        # The revision removes the Furnace Creek stop, cancels AAMV4 and
        # leaves the old AAMV2 -> AAMV4 transfer row in transfers.txt.
        _rewrite(self.new / "stops.txt", [
            line for line in self.old.joinpath("stops.txt").read_text().splitlines()
            if not line.startswith("FUR_CREEK_RES,")
        ])
        _rewrite(self.new / "stop_times.txt", [
            line for line in self.old.joinpath("stop_times.txt").read_text().splitlines()
            if "FUR_CREEK_RES" not in line and
            line.split(",")[0] != "AAMV4"
        ])
        _rewrite(self.new / "trips.txt", [
            line for line in self.old.joinpath("trips.txt").read_text().splitlines()
            if line.split(",")[2] != "AAMV4"
        ])
        # Route BFC also changes: its stop_times lost the removed stop.
        _rewrite(self.new / "fare_rules.txt", [
            line for line in self.old.joinpath("fare_rules.txt").read_text().splitlines()
            if not line.startswith("p,BFC,")
        ])

        # A translation and a transfer unrelated to the revision must survive.
        for feed in (self.old, self.new):
            _rewrite(feed / "translations.txt", [
                line for line in feed.joinpath("translations.txt").read_text().splitlines()
            ] + ["Nye County Airport (Demo),EN,Nye County Airport"])
            _rewrite(feed / "transfers.txt", [
                line for line in feed.joinpath("transfers.txt").read_text().splitlines()
            ] + ["2,STAGECOACH,NANAA,,,,"])

        trips = self.new.joinpath("trips.txt").read_text().splitlines()
        _rewrite(self.new / "trips.txt", [
            line.replace("AB1,to Bullfrog,0,1,", "AB1,to Bullfrog,0,1,S1")
                .replace("AB2,to Airport,1,2,", "AB2,to Airport,1,2,S1")
            for line in trips
        ])
        _rewrite(self.new / "shapes.txt", [
            "shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence,shape_dist_traveled",
            "S1,36.86,-116.78,1,0",
            "S1,36.88,-116.81,2,1000",
            "S1,36.91,-116.75,3,2000",
        ])

        self.schedule = Schedule(":memory:")
        append_feed(self.schedule, str(self.old))

    def tearDown(self):
        self.schedule.session.close()
        shutil.rmtree(self.temp_dir)

    def _incremental(self):
        return update_feed(self.schedule, str(self.new),
                           old_feed_filename=str(self.old))

    def _fresh_schedule(self):
        fresh = Schedule(":memory:")
        append_feed(fresh, str(self.new))
        return fresh

    def _snapshot(self, schedule):
        return FeedSnapshot.from_schedule(schedule,
                                          schedule.feeds[0].feed_id)


class TestFeedDiff(_RevisionMixin, unittest.TestCase):
    def test_diff_lists_removed_stop_and_rebuilds(self):
        changes = diff_feed(self.schedule, str(self.new),
                            old_feed_filename=str(self.old))
        report = changes.report()

        self.assertEqual(report["removed"]["stops"], ["FUR_CREEK_RES"])
        self.assertEqual(report["removed"]["trips"], ["AAMV4"])
        self.assertIn("stop_times", report["rebuild"])
        self.assertIn("transfers", report["rebuild"])
        self.assertIn("_trip_shapes", report["rebuild"])

    def test_diff_identifies_shape_change_and_its_trips(self):
        changes = diff_feed(self.schedule, str(self.new),
                            old_feed_filename=str(self.old))
        report = changes.report()

        self.assertEqual(report["added"]["shapes"], ["S1|1", "S1|2", "S1|3"])
        self.assertIn("AB1", report["affected"]["trips"])
        self.assertIn("AB2", report["affected"]["trips"])
        trip_dependency = [dep for dep in report["dependencies"]
                           if dep["entity"] == "trip" and dep["key"] == "AB1"]
        self.assertEqual(len(trip_dependency), 1)
        self.assertIn("_trip_shapes", trip_dependency[0]["rebuild"])
        # AB1/AB2 need link rows regenerated; AAMV4's old link rows are
        # deleted with the cancelled trip.
        self.assertEqual(report["derived"]["_trip_shapes"],
                         ["AAMV4", "AB1", "AB2"])

    def test_diff_is_empty_for_identical_input(self):
        changes = diff_feed(self.schedule, str(self.old))
        self.assertFalse(changes.has_changes())
        self.assertEqual(changes.report()["dependencies"], [])


class TestIncrementalUpdate(_RevisionMixin, unittest.TestCase):
    def test_only_relations_change_for_unchanged_entities(self):
        before = self.schedule.stops_by_id("BEATTY_AIRPORT")[0]
        before_pk = before.stop_id
        before_name = before.stop_name

        self._incremental()

        after = self.schedule.stops_by_id("BEATTY_AIRPORT")[0]
        self.assertEqual(after.stop_id, before_pk)
        self.assertEqual(after.stop_name, before_name)
        # The deleted stop and the cancelled trip are gone.
        self.assertEqual(self.schedule.stops_by_id("FUR_CREEK_RES"), [])
        self.assertEqual(self.schedule.trips_by_id("AAMV4"), [])

    def test_incremental_matches_full_import(self):
        self._incremental()
        fresh = self._fresh_schedule()
        try:
            diff = FeedDiff(self._snapshot(self.schedule),
                            self._snapshot(fresh))
            self.assertFalse(diff.has_changes(), diff.summary())
        finally:
            fresh.session.close()

    def test_cancelled_trip_transfer_entries_are_gone(self):
        self._incremental()
        # No transfer may still reference the cancelled AAMV4 trip.
        stale = (self.schedule.session.query(Transfer)
                 .filter((Transfer.from_trip_id == "AAMV4") |
                         (Transfer.to_trip_id == "AAMV4"))
                 .all())
        self.assertEqual(stale, [])
        # The transfers that only referenced the removed stop are gone too.
        self.assertFalse([t for t in self.schedule.transfers
                          if t.from_stop_id == "FUR_CREEK_RES" or
                          t.to_stop_id == "FUR_CREEK_RES"])
        # A transfer unrelated to the revision is still present.
        self.assertTrue([t for t in self.schedule.transfers
                         if t.from_stop_id == "STAGECOACH" and
                         t.to_stop_id == "NANAA"])
        # So are translations of stops that still exist.
        self.assertEqual(
            len(self.schedule.stops_by_id("BEATTY_AIRPORT")[0].translations), 1)

    def test_shape_relations_are_rebuilt(self):
        self._incremental()
        rows = self.schedule.session.execute(text(
            "SELECT shape_pt_sequence FROM _trip_shapes "
            "WHERE trip_id = 'AB1' ORDER BY shape_pt_sequence")).fetchall()
        self.assertEqual([row[0] for row in rows], [1, 2, 3])

        shape_rows = (self.schedule.session.query(ShapePoint)
                      .filter(ShapePoint.shape_id == "S1")
                      .order_by(ShapePoint.shape_pt_sequence).all())
        self.assertEqual([p.shape_pt_lat for p in shape_rows],
                         [36.86, 36.88, 36.91])

    def test_update_does_not_duplicate_feed(self):
        self._incremental()
        self.assertEqual(len(self.schedule.feeds), 1)

    def test_update_is_idempotent(self):
        self._incremental()
        changes = update_feed(self.schedule, str(self.new),
                              old_feed_filename=str(self.old))
        self.assertFalse(changes.has_changes())

    def test_stop_times_and_frequencies_for_cancelled_trip_are_gone(self):
        self._incremental()
        self.assertEqual(
            self.schedule.session.query(StopTime)
            .filter(StopTime.trip_id == "AAMV4").count(), 0)
        self.assertEqual(
            self.schedule.session.query(Frequency)
            .filter(Frequency.trip_id == "AAMV4").count(), 0)

    def test_valid_link_rows_survive(self):
        before = self.schedule.session.execute(
            _stop_translations.select().where(
                _stop_translations.c.stop_id == "BEATTY_AIRPORT")).fetchall()
        self._incremental()
        after = self.schedule.session.execute(
            _stop_translations.select().where(
                _stop_translations.c.stop_id == "BEATTY_AIRPORT")).fetchall()
        # The stop name is unchanged, so its translation link is untouched.
        self.assertEqual(before, after)

    def test_changed_non_key_column_updates_in_place(self):
        _rewrite(self.new / "stops.txt", [
            line.replace("Nye County Airport (Demo)", "Nye County Airport")
            for line in self.new.joinpath("stops.txt").read_text().splitlines()
        ])
        _rewrite(self.new / "translations.txt", [
            line.replace("Nye County Airport (Demo)", "Nye County Airport")
            for line in self.new.joinpath("translations.txt").read_text().splitlines()
        ])
        before = self.schedule.stops_by_id("BEATTY_AIRPORT")[0]

        changes = self._incremental()

        after = self.schedule.stops_by_id("BEATTY_AIRPORT")[0]
        self.assertIs(before, after)
        self.assertEqual(after.stop_name, "Nye County Airport")
        modified = {entry["key"] for entry in
                    changes.report()["modified"]["stops"]}
        self.assertEqual(modified, {"BEATTY_AIRPORT"})
        # Translation links are name based, so the renamed stop is relinked.
        self.assertEqual(len(after.translations), 1)
        self.assertEqual(after.translations[0].trans_id, "Nye County Airport")

    def test_unchanged_trip_keeps_identity_when_shape_changes(self):
        before = self.schedule.trips_by_id("AB1")[0]
        self._incremental()
        after = self.schedule.trips_by_id("AB1")[0]
        self.assertIs(before, after)
        self.assertEqual(after.shape_id, "S1")


if __name__ == "__main__":
    unittest.main()
