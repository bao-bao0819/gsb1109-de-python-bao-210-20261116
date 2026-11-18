# -*- coding: utf-8 -*-

import datetime
import json
import os.path
import unittest

from pygtfs import Schedule, overwrite_feed, append_feed
from pygtfs.block_chains import (
    build_block_chain_reports,
    block_chains_as_dicts,
    report_to_dict,
)


def _data_dir(name):
    return os.path.join(os.path.dirname(__file__), "data", name)


class BlockChainTestBase(unittest.TestCase):
    def setUp(self):
        self.schedule = Schedule(":memory:")
        overwrite_feed(self.schedule, _data_dir("block_chain_feed"))
        self.reports = {
            report.block_id: report
            for report in build_block_chain_reports(self.schedule)
        }

    def report(self, block_id):
        return self.reports[block_id]

    def edge(self, report, from_trip_id, to_trip_id):
        matches = [
            edge
            for edge in report.edges
            if edge.from_trip_id == from_trip_id
            and edge.to_trip_id == to_trip_id
        ]
        self.assertEqual(len(matches), 1)
        return matches[0]

    def chain(self, report, trip_ids):
        matches = [
            chain for chain in report.chains if chain.trip_ids == trip_ids
        ]
        self.assertEqual(len(matches), 1)
        return matches[0]


class TestServiceDays(BlockChainTestBase):
    def test_calendar_weekdays_and_removal_exception(self):
        # S_MON covers every January 2024 day except the type-2 exception.
        info = self.report("B1").trips["T_A1"]
        self.assertEqual(len(info.operating_dates), 30)
        self.assertIn(datetime.date(2024, 1, 1), info.operating_dates)
        self.assertNotIn(datetime.date(2024, 1, 8), info.operating_dates)

    def test_added_exception_day_activates_service(self):
        # S_ADHOC has no weekday flags at all; only the type-1 exception runs.
        info = self.report("B6").trips["T_H1"]
        self.assertEqual(info.operating_dates, [datetime.date(2024, 1, 20)])

    def test_weekday_only_service(self):
        info = self.report("B2").trips["T_Y2"]
        wednesdays = [
            datetime.date(2024, 1, 3),
            datetime.date(2024, 1, 10),
            datetime.date(2024, 1, 17),
            datetime.date(2024, 1, 24),
            datetime.date(2024, 1, 31),
        ]
        self.assertEqual(info.operating_dates, wednesdays)


class TestCleanChain(BlockChainTestBase):
    def test_three_trip_chain_with_timetable_evidence(self):
        report = self.report("B1")
        self.assertEqual(
            [(edge.from_trip_id, edge.to_trip_id) for edge in report.edges],
            [("T_A1", "T_A2"), ("T_A2", "T_A3")],
        )
        chain = self.chain(report, ["T_A1", "T_A2", "T_A3"])
        self.assertEqual(len(chain.operating_dates), 30)
        self.assertFalse(chain.ambiguous)
        self.assertEqual(len(chain.links), 2)

        link = chain.links[0]
        self.assertEqual(link.connection_stop_id, "STN_B")
        self.assertEqual(link.arrival_time, datetime.timedelta(hours=9))
        self.assertEqual(link.departure_time, datetime.timedelta(hours=9, minutes=30))
        self.assertEqual(link.dwell, datetime.timedelta(minutes=30))
        self.assertEqual(len(link.valid_dates), 30)
        self.assertEqual(report.conflicts, [])

    def test_raw_stop_time_evidence_is_kept(self):
        info = self.report("B1").trips["T_A1"]
        self.assertEqual(info.first_stop.stop_id, "STN_A")
        self.assertEqual(info.first_stop.stop_sequence, 1)
        self.assertEqual(info.last_stop.stop_id, "STN_B")
        self.assertEqual(len(info.stop_times), 2)
        self.assertEqual(
            info.stop_times[1].arrival_time, datetime.timedelta(hours=9)
        )

    def test_times_past_midnight_are_compared_as_durations(self):
        report = self.report("B7")
        edge = self.edge(report, "T_I1", "T_I2")
        self.assertEqual(edge.connection_stop_id, "STN_B")
        self.assertEqual(edge.arrival_time, datetime.timedelta(hours=25, minutes=30))
        self.assertEqual(edge.departure_time, datetime.timedelta(hours=25, minutes=45))
        self.assertEqual(edge.dwell, datetime.timedelta(minutes=15))
        self.chain(report, ["T_I1", "T_I2"])
        self.assertEqual(report.conflicts, [])


class TestCandidateSuccessors(BlockChainTestBase):
    def test_all_feasible_successors_are_retained(self):
        # T_X can be followed by both T_Y1 and T_Y2; no nearest-only picking.
        report = self.report("B2")
        successors = {
            edge.to_trip_id: edge for edge in report.edges
            if edge.from_trip_id == "T_X"
        }
        self.assertEqual(set(successors), {"T_Y1", "T_Y2"})
        self.assertTrue(all(edge.candidate for edge in successors.values()))
        self.assertEqual(
            sorted(chain.trip_ids for chain in report.chains),
            sorted([["T_X", "T_Y1"], ["T_X", "T_Y2"]]),
        )

    def test_candidate_edges_keep_their_own_valid_dates(self):
        report = self.report("B2")
        y1_edge = self.edge(report, "T_X", "T_Y1")
        y2_edge = self.edge(report, "T_X", "T_Y2")
        self.assertEqual(len(y1_edge.valid_dates), 30)
        self.assertEqual(len(y2_edge.valid_dates), 5)

    def test_forking_conflict_on_shared_service_day(self):
        report = self.report("B2")
        forks = [
            conflict
            for conflict in report.conflicts
            if conflict.conflict_type == "forking_succession"
        ]
        self.assertEqual(len(forks), 5)
        for conflict in forks:
            self.assertEqual(
                conflict.trip_ids, ["T_X", "T_Y1", "T_Y2"]
            )
            self.assertEqual(conflict.service_date.weekday(), 2)

    def test_simultaneous_trips_with_same_window_conflict(self):
        report = self.report("B2")
        overlaps = [
            conflict
            for conflict in report.conflicts
            if conflict.conflict_type == "simultaneous_operation"
        ]
        self.assertEqual(len(overlaps), 5)
        self.assertEqual(
            sorted(overlaps[0].trip_ids), ["T_Y1", "T_Y2"]
        )


class TestSimultaneousOperation(BlockChainTestBase):
    def test_overlapping_trips_leave_conflict_and_no_illegal_edge(self):
        report = self.report("B3")
        # T_P (08:00-09:00) and T_Q (08:30-09:30) genuinely overlap on Fridays.
        overlaps = [
            conflict
            for conflict in report.conflicts
            if conflict.conflict_type == "simultaneous_operation"
        ]
        self.assertEqual(len(overlaps), 4)
        self.assertEqual(sorted(overlaps[0].trip_ids), ["T_P", "T_Q"])
        for conflict in overlaps:
            self.assertEqual(conflict.service_date.weekday(), 4)

        # T_Q departs STN_B before T_P arrives there: timing rule rejects it.
        self.assertFalse(
            any(
                edge.from_trip_id == "T_P" and edge.to_trip_id == "T_Q"
                for edge in report.edges
            )
        )
        self.assertTrue(
            any("before the arrival" in note for note in report.notes)
        )
        # The feasible part of the block still chains normally.
        self.edge(report, "T_Q", "T_R")
        self.chain(report, ["T_Q", "T_R"])
        self.chain(report, ["T_P"])


class TestNonOperatingBridge(BlockChainTestBase):
    def test_ghost_trip_is_excluded_and_cannot_bridge(self):
        report = self.report("B4")
        excluded = {item.trip_id: item for item in report.excluded_trips}
        self.assertIn("T_G2", excluded)
        self.assertEqual(excluded["T_G2"].reason, "no_service_days")

        # G2 appears nowhere in edges or chains.
        self.assertEqual(
            [edge for edge in report.edges if "T_G2" in
             (edge.from_trip_id, edge.to_trip_id)],
            [],
        )
        for chain in report.chains:
            self.assertNotIn("T_G2", chain.trip_ids)

        # G1 ends at STN_B while G3 starts at STN_C: no edge without the ghost.
        self.assertEqual(report.edges, [])
        self.assertEqual(
            sorted(chain.trip_ids for chain in report.chains),
            sorted([["T_G1"], ["T_G3"]]),
        )
        self.assertEqual(report.conflicts, [])


class TestForkAndMerge(BlockChainTestBase):
    def test_fork_merge_enumerates_both_alternatives(self):
        report = self.report("B5")
        self.assertEqual(
            sorted(
                (edge.from_trip_id, edge.to_trip_id) for edge in report.edges
            ),
            sorted([
                ("T_N1", "T_N2"),
                ("T_N1", "T_N3"),
                ("T_N2", "T_N4"),
                ("T_N3", "T_N4"),
            ]),
        )
        chains = sorted(chain.trip_ids for chain in report.chains)
        self.assertEqual(
            chains,
            sorted([["T_N1", "T_N2", "T_N4"], ["T_N1", "T_N3", "T_N4"]]),
        )
        self.assertTrue(all(chain.ambiguous for chain in report.chains))

        forks = [
            conflict for conflict in report.conflicts
            if conflict.conflict_type == "forking_succession"
        ]
        merges = [
            conflict for conflict in report.conflicts
            if conflict.conflict_type == "merging_succession"
        ]
        self.assertEqual(len(forks), 4)
        self.assertEqual(len(merges), 4)
        self.assertEqual(forks[0].trip_ids, ["T_N1", "T_N2", "T_N3"])
        self.assertEqual(merges[0].trip_ids, ["T_N2", "T_N3", "T_N4"])
        for conflict in forks + merges:
            self.assertEqual(conflict.service_date.weekday(), 5)


class TestAllGhostBlock(BlockChainTestBase):
    def test_block_with_no_operating_trip_is_still_reported(self):
        report = self.report("B8")
        self.assertEqual(report.trips, {})
        self.assertEqual(report.edges, [])
        self.assertEqual(report.chains, [])
        self.assertEqual(
            sorted(item.trip_id for item in report.excluded_trips),
            ["T_Z1", "T_Z2"],
        )
        self.assertTrue(
            all(item.reason == "no_service_days" for item in report.excluded_trips)
        )


class TestDisjointServiceDays(BlockChainTestBase):
    def test_no_shared_day_means_no_succession(self):
        report = self.report("B6")
        self.assertEqual(report.edges, [])
        self.assertEqual(
            sorted(chain.trip_ids for chain in report.chains),
            sorted([["T_H1"], ["T_H2"]]),
        )
        self.assertTrue(
            any("share no service day" in note for note in report.notes)
        )
        self.assertEqual(report.conflicts, [])


class TestFeedIsolation(unittest.TestCase):
    def test_same_block_id_in_two_feeds_never_joins(self):
        schedule = Schedule(":memory:")
        data_location = _data_dir("block_chain_feed")
        append_feed(schedule, data_location)
        append_feed(schedule, data_location)
        reports = build_block_chain_reports(schedule)
        feed_ids = {report.feed_id for report in reports}
        self.assertEqual(len(feed_ids), 2)
        # Eight blocks per feed, nothing merged across the feed boundary.
        self.assertEqual(len(reports), 16)
        for report in reports:
            self.assertTrue(
                all(info.feed_id == report.feed_id for info in report.trips.values())
            )
            for edge in report.edges:
                self.assertIn(edge.from_trip_id, report.trips)
                self.assertIn(edge.to_trip_id, report.trips)

    def test_filters_by_feed_and_block(self):
        schedule = Schedule(":memory:")
        overwrite_feed(schedule, _data_dir("block_chain_feed"))
        feed_id = schedule.feeds[0].feed_id
        reports = build_block_chain_reports(
            schedule, feed_id=feed_id, block_ids=["B1"]
        )
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0].block_id, "B1")


class TestSerialization(BlockChainTestBase):
    def test_report_to_dict_is_json_serializable(self):
        payload = report_to_dict(self.report("B1"))
        encoded = json.dumps(payload)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["block_id"], "B1")
        self.assertEqual(
            decoded["trips"]["T_A1"]["first_stop"]["stop_id"], "STN_A"
        )
        # Times are exported as seconds since service-day start.
        self.assertEqual(
            decoded["edges"][0]["arrival_time"], 9 * 3600
        )
        self.assertEqual(
            decoded["edges"][0]["dwell"], 30 * 60
        )

    def test_block_chains_as_dicts_covers_every_block(self):
        payloads = block_chains_as_dicts(self.schedule)
        self.assertEqual(
            sorted(payload["block_id"] for payload in payloads),
            ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"],
        )
        self.assertTrue(all(json.dumps(p) for p in payloads))


class TestSampleFeedIntegration(unittest.TestCase):
    def test_sample_feed_blocks(self):
        schedule = Schedule(":memory:")
        overwrite_feed(schedule, _data_dir("sample_feed"))
        reports = {
            report.block_id: report
            for report in build_block_chain_reports(schedule)
        }
        self.assertEqual(set(reports), {"1", "2"})

        block_one = reports["1"]
        edge = block_one.edges[0]
        self.assertEqual(
            (edge.from_trip_id, edge.to_trip_id), ("AB1", "BFC1")
        )
        self.assertEqual(edge.connection_stop_id, "BULLFROG")
        self.assertEqual(edge.dwell, datetime.timedelta(minutes=10))
        self.assertEqual(
            [chain.trip_ids for chain in block_one.chains],
            [["AB1", "BFC1"]],
        )
        self.assertEqual(block_one.conflicts, [])
        # FULLW runs daily 2007-01-01..2010-12-31 minus the 2007-06-04 removal.
        self.assertEqual(len(edge.valid_dates), 1460)
        self.assertNotIn(datetime.date(2007, 6, 4), edge.valid_dates)

        block_two = reports["2"]
        self.assertEqual(
            [(edge.from_trip_id, edge.to_trip_id) for edge in block_two.edges],
            [("BFC2", "AB2")],
        )
        self.assertEqual(
            block_two.edges[0].connection_stop_id, "BULLFROG"
        )


if __name__ == "__main__":
    unittest.main()
