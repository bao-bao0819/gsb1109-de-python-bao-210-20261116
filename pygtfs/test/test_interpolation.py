# -*- coding: utf-8 -*-

import datetime
import os.path
import tempfile
import unittest

from pygtfs import Schedule, append_feed
from pygtfs.gtfs_entities import StopTime
from pygtfs.interpolation import (
    estimate_stop_times, interpolate_trip, interpolate_schedule,
    TimeEstimate, EXACT, PROVIDED, INTERPOLATED, UNKNOWN,
)

TD = datetime.timedelta


def _row(sequence, stop_id, arrival=None, departure=None, timepoint=None,
         distance=None, trip_id="T"):
    return StopTime(
        feed_id=None,
        trip_id=trip_id,
        stop_id=stop_id,
        stop_sequence=sequence,
        arrival_time=arrival,
        departure_time=departure,
        timepoint=timepoint,
        shape_dist_traveled=distance,
    )


class EstimateStopTimesTest(unittest.TestCase):

    def test_linearly_interpolates_between_anchors(self):
        rows = [
            _row(1, "S1", "8:00:00", "8:00:00", timepoint=1, distance=0.0),
            _row(2, "S2", distance=2.5),
            _row(3, "S3", distance=5.0),
            _row(4, "S4", distance=7.5),
            _row(5, "S5", "8:10:00", "8:11:00", timepoint=1, distance=10.0),
        ]

        estimates = estimate_stop_times(rows)

        self.assertEqual(
            [e.source for e in estimates],
            [EXACT, INTERPOLATED, INTERPOLATED, INTERPOLATED, EXACT])
        self.assertEqual(estimates[1].arrival_time,
                         TD(seconds=8 * 3600 + 150))
        self.assertEqual(estimates[2].arrival_time,
                         TD(seconds=8 * 3600 + 300))
        self.assertEqual(estimates[3].arrival_time,
                         TD(seconds=8 * 3600 + 450))
        # No invented dwell: arrival equals departure at interpolated stops.
        for estimate in estimates[1:4]:
            self.assertEqual(estimate.arrival_time, estimate.departure_time)

    def test_bounds_respect_anchor_departure_and_arrival(self):
        rows = [
            _row(1, "S1", "8:00:00", "8:00:30", timepoint=1, distance=0.0),
            _row(2, "S2", distance=5.0),
            _row(3, "S3", "8:10:00", "8:11:00", timepoint=1, distance=10.0),
        ]

        estimate = estimate_stop_times(rows)[1]
        eight_am = 8 * 3600

        self.assertEqual(
            estimate.arrival_bounds,
            (TD(seconds=eight_am + 30), TD(seconds=eight_am + 600)))
        self.assertEqual(
            estimate.departure_bounds,
            (TD(seconds=eight_am + 30), TD(seconds=eight_am + 660)))
        # The point sits inside both ranges.
        self.assertGreaterEqual(estimate.arrival_time,
                                estimate.arrival_bounds[0])
        self.assertLessEqual(estimate.arrival_time,
                             estimate.arrival_bounds[1])

    def test_repeated_distance_does_not_divide_by_zero(self):
        rows = [
            _row(1, "S1", "8:00:00", "8:00:00", timepoint=1, distance=0.0),
            _row(2, "S2", distance=5.0),
            _row(3, "S3", distance=5.0),
            _row(4, "S4", "8:10:00", "8:10:00", timepoint=1, distance=10.0),
        ]

        estimates = estimate_stop_times(rows)

        self.assertEqual(estimates[1].source, INTERPOLATED)
        self.assertEqual(estimates[2].source, UNKNOWN)
        self.assertIsNone(estimates[2].arrival_time)
        self.assertIsNone(estimates[2].departure_time)
        # The range is still derivable from the anchors.
        self.assertEqual(estimates[2].arrival_bounds,
                         (TD(seconds=8 * 3600), TD(seconds=8 * 3600 + 600)))

    def test_missing_distances_keep_derivable_ranges(self):
        rows = [
            _row(1, "S1", "8:00:00", "8:00:00", timepoint=1, distance=0.0),
            _row(2, "S2"),
            _row(3, "S3", "8:10:00", "8:10:00", timepoint=1, distance=10.0),
        ]

        estimate = estimate_stop_times(rows)[1]

        self.assertEqual(estimate.source, UNKNOWN)
        self.assertIsNone(estimate.arrival_time)
        self.assertEqual(estimate.arrival_bounds,
                         (TD(seconds=8 * 3600), TD(seconds=8 * 3600 + 600)))

    def test_stops_outside_anchors_stay_unknown_without_bounds(self):
        rows = [
            _row(1, "S1"),
            _row(2, "S2", "8:00:00", "8:00:00", timepoint=1, distance=0.0),
            _row(3, "S3", distance=5.0),
            _row(4, "S4", "8:10:00", "8:10:00", timepoint=1, distance=10.0),
            _row(5, "S5"),
        ]

        estimates = estimate_stop_times(rows)

        self.assertEqual(estimates[0].source, UNKNOWN)
        self.assertIsNone(estimates[0].arrival_bounds)
        self.assertEqual(estimates[2].source, INTERPOLATED)
        self.assertEqual(estimates[4].source, UNKNOWN)
        self.assertIsNone(estimates[4].arrival_bounds)

    def test_timepoint_zero_is_not_an_anchor(self):
        rows = [
            _row(1, "S1", "8:00:00", "8:00:00", timepoint=1, distance=0.0),
            _row(2, "S2", "8:03:00", "8:03:30", timepoint=0, distance=3.0),
            _row(3, "S3", distance=6.0),
            _row(4, "S4", "8:10:00", "8:10:00", timepoint=1, distance=10.0),
        ]

        estimates = estimate_stop_times(rows)

        # The approximate row is passed through, but S3 still interpolates
        # against the two confirmed timepoints rather than S2.
        self.assertEqual(estimates[1].source, PROVIDED)
        self.assertEqual(estimates[1].arrival_time, TD(seconds=8 * 3600 + 180))
        self.assertEqual(estimates[2].source, INTERPOLATED)
        self.assertEqual(estimates[2].arrival_time, TD(seconds=8 * 3600 + 360))

    def test_blank_timepoint_with_times_is_anchor(self):
        rows = [
            _row(1, "S1", "8:00:00", "8:00:00", distance=0.0),
            _row(2, "S2", distance=5.0),
            _row(3, "S3", "8:10:00", "8:10:00", distance=10.0),
        ]

        estimates = estimate_stop_times(rows)

        self.assertEqual(estimates[0].source, EXACT)
        self.assertEqual(estimates[2].source, EXACT)
        self.assertEqual(estimates[1].source, INTERPOLATED)

    def test_anchor_with_only_departure_fills_bounds(self):
        rows = [
            _row(1, "S1", departure="8:00:00", timepoint=1, distance=0.0),
            _row(2, "S2", distance=5.0),
            _row(3, "S3", arrival="8:10:00", timepoint=1, distance=10.0),
        ]

        estimate = estimate_stop_times(rows)[1]

        self.assertEqual(estimate.source, INTERPOLATED)
        self.assertEqual(estimate.arrival_bounds,
                         (TD(seconds=8 * 3600), TD(seconds=8 * 3600 + 600)))

    def test_out_of_order_distances_never_run_clock_backwards(self):
        rows = [
            _row(1, "S1", "8:00:00", "8:00:00", timepoint=1, distance=0.0),
            _row(2, "S2", distance=8.0),
            _row(3, "S3", distance=2.0),
            _row(4, "S4", "8:10:00", "8:10:00", timepoint=1, distance=10.0),
        ]

        estimates = estimate_stop_times(rows)

        self.assertEqual(estimates[1].arrival_time, TD(seconds=8 * 3600 + 480))
        # A naive interpolation here would yield 8:02, earlier than S2.
        self.assertEqual(estimates[2].source, UNKNOWN)
        self.assertIsNone(estimates[2].arrival_time)
        self.assertIsNotNone(estimates[2].arrival_bounds)

        emitted = [e.arrival_time for e in estimates if e.arrival_time]
        self.assertEqual(emitted, sorted(emitted))

    def test_input_is_sorted_and_not_modified(self):
        rows = [
            _row(3, "S3", "8:10:00", "8:10:00", timepoint=1, distance=10.0),
            _row(1, "S1", "8:00:00", "8:00:00", timepoint=1, distance=0.0),
            _row(2, "S2", distance=5.0),
        ]

        estimates = estimate_stop_times(rows)

        self.assertEqual([e.stop_sequence for e in estimates], [1, 2, 3])
        self.assertTrue(all(st.arrival_time is None or
                            isinstance(st.arrival_time, datetime.timedelta)
                            for st in rows))

    def test_returns_time_estimate_namedtuples(self):
        rows = [
            _row(1, "S1", "8:00:00", "8:00:00", timepoint=1, distance=0.0),
        ]

        estimate = estimate_stop_times(rows)[0]

        self.assertIsInstance(estimate, TimeEstimate)
        self.assertEqual(estimate.trip_id, "T")


class InterpolateScheduleTest(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        feed_dir = self.temp_dir.name
        self._write(feed_dir, "agency.txt", [
            "agency_id,agency_name,agency_url,agency_timezone",
            "DTA,Demo,http://example.com,UTC",
        ])
        self._write(feed_dir, "stops.txt", [
            "stop_id,stop_name,stop_lat,stop_lon",
            "S1,One,36.0,-117.0",
            "S2,Two,36.1,-117.1",
            "S3,Three,36.2,-117.2",
        ])
        self._write(feed_dir, "routes.txt", [
            "route_id,agency_id,route_short_name,route_long_name,route_type",
            "R1,DTA,R1,,3",
        ])
        self._write(feed_dir, "calendar.txt", [
            "service_id,monday,tuesday,wednesday,thursday,friday,saturday,"
            "sunday,start_date,end_date",
            "FULLW,1,1,1,1,1,1,1,20230101,20231231",
        ])
        self._write(feed_dir, "trips.txt", [
            "route_id,service_id,trip_id",
            "R1,FULLW,T1",
        ])
        self._write(feed_dir, "stop_times.txt", [
            "trip_id,arrival_time,departure_time,stop_id,stop_sequence,"
            "timepoint,shape_dist_traveled",
            "T1,8:00:00,8:00:00,S1,1,1,0",
            "T1,,,S2,2,,5",
            "T1,8:10:00,8:11:00,S3,3,1,10",
        ])

        self.schedule = Schedule(":memory:")
        append_feed(self.schedule, feed_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def _write(feed_dir, filename, lines):
        with open(os.path.join(feed_dir, filename), "w",
                  encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

    def test_interpolate_schedule_end_to_end(self):
        results = interpolate_schedule(self.schedule)

        self.assertEqual(len(results), 1)
        estimates = results[(1, "T1")]
        self.assertEqual([e.stop_sequence for e in estimates], [1, 2, 3])
        self.assertEqual([e.source for e in estimates],
                         [EXACT, INTERPOLATED, EXACT])
        self.assertEqual(estimates[1].arrival_time,
                         TD(seconds=8 * 3600 + 300))

    def test_interpolate_trip_orm_instance(self):
        trip = self.schedule.trips_by_id("T1")[0]
        estimates = interpolate_trip(trip)

        self.assertEqual([e.stop_id for e in estimates], ["S1", "S2", "S3"])
        self.assertEqual(estimates[1].source, INTERPOLATED)
        # The database rows themselves remain untouched.
        middle = next(st for st in trip.stop_times if st.stop_sequence == 2)
        self.assertIsNone(middle.arrival_time)
        self.assertIsNone(middle.departure_time)


if __name__ == '__main__':
    unittest.main()
