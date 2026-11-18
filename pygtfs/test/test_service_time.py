# -*- coding: utf-8 -*-

import csv
import datetime
import io
import unittest

import pytz

from pygtfs import Schedule
from pygtfs.gtfs_entities import (Agency, Feed, Route, Service, Stop, StopTime,
                                  Trip)
from pygtfs.service_time import (format_gtfs_time, service_day_start,
                                 service_time)

NEW_YORK = pytz.timezone("America/New_York")


def _add_feed(schedule):
    session = schedule.session
    feed = Feed(feed_name="service-day-test")
    session.add(feed)
    session.flush()
    feed_id = feed.feed_id

    session.add(Agency(
        feed_id=feed_id, agency_id="A", agency_name="Test Agency",
        agency_url="http://example.com", agency_timezone="America/New_York"))
    session.add(Stop(feed_id=feed_id, stop_id="S1", stop_name="First",
                     stop_lat=40.0, stop_lon=-73.0))
    session.add(Stop(feed_id=feed_id, stop_id="S2", stop_name="Second",
                     stop_lat=40.1, stop_lon=-73.1))
    session.add(Route(
        feed_id=feed_id, route_id="R", agency_id="A",
        route_short_name="R", route_long_name="Night Route", route_type=3))
    session.add(Service(
        feed_id=feed_id, service_id="DAILY",
        monday="1", tuesday="1", wednesday="1", thursday="1",
        friday="1", saturday="1", sunday="1",
        start_date="20240101", end_date="20241231"))
    session.add(Trip(
        feed_id=feed_id, route_id="R", service_id="DAILY", trip_id="NIGHT"))
    session.add(StopTime(
        feed_id=feed_id, trip_id="NIGHT", stop_id="S1", stop_sequence=1,
        arrival_time="24:00:00", departure_time="24:00:00"))
    session.add(StopTime(
        feed_id=feed_id, trip_id="NIGHT", stop_id="S2", stop_sequence=2,
        arrival_time="25:10:00", departure_time="25:10:00"))
    session.commit()
    return feed_id


class TestServiceDayArithmetic(unittest.TestCase):
    def test_normal_day_starts_at_local_midnight(self):
        start = service_day_start(datetime.date(2024, 3, 9), NEW_YORK)
        self.assertEqual(start, NEW_YORK.localize(
            datetime.datetime(2024, 3, 9, 0, 0)))

    def test_spring_forward_starts_before_calendar_midnight(self):
        start = service_day_start(datetime.date(2024, 3, 10), NEW_YORK)
        self.assertEqual(start, NEW_YORK.localize(
            datetime.datetime(2024, 3, 9, 23, 0)))
        self.assertEqual(start.date(), datetime.date(2024, 3, 9))

    def test_fall_back_starts_at_one_am_daylight_time(self):
        # Noon on the fall-back day is EST; twelve hours earlier is 01:00
        # EDT, because the repeated hour makes the local day 25 hours long.
        start = service_day_start(datetime.date(2024, 11, 3), NEW_YORK)
        self.assertEqual(start.isoformat(), "2024-11-03T01:00:00-04:00")

    def test_gtfs_time_is_not_wrapped_to_24_hours(self):
        offset = datetime.timedelta(hours=25, minutes=10)
        self.assertEqual(format_gtfs_time(offset), "25:10:00")

        normal = service_time(datetime.date(2024, 3, 9), offset, NEW_YORK)
        self.assertEqual(normal.date(), datetime.date(2024, 3, 10))
        self.assertEqual(normal.isoformat(), "2024-03-10T01:10:00-05:00")

    def test_spring_forward_keeps_elapsed_time(self):
        offset = datetime.timedelta(hours=25, minutes=10)
        actual = service_time(datetime.date(2024, 3, 10), offset, NEW_YORK)
        # 25h10 elapsed from 2024-03-09 23:00 EST: the skipped hour means the
        # local wall clock reads 01:10 EDT on the next date, not 02:10.
        self.assertEqual(actual.isoformat(), "2024-03-11T01:10:00-04:00")

    def test_fall_back_keeps_elapsed_time(self):
        offset = datetime.timedelta(hours=25, minutes=10)
        actual = service_time(datetime.date(2024, 11, 3), offset, NEW_YORK)
        # 25h10 elapsed from 2024-11-03 01:00 EDT crosses the repeated hour,
        # so the local wall clock is 01:10 EST on the next date.
        self.assertEqual(actual.isoformat(), "2024-11-04T01:10:00-05:00")


class TestScheduleServiceDayQueries(unittest.TestCase):
    def setUp(self):
        self.schedule = Schedule(":memory:")
        self.feed_id = _add_feed(self.schedule)

    def test_trip_query_preserves_service_date_and_original_clock(self):
        records = self.schedule.stop_times_for_trip(
            "NIGHT", datetime.date(2024, 3, 9))
        self.assertEqual([record.stop_id for record in records], ["S1", "S2"])
        arrival = records[-1]
        self.assertEqual(arrival.service_date, datetime.date(2024, 3, 9))
        self.assertEqual(arrival.arrival_gtfs_time, "25:10:00")
        self.assertEqual(arrival.arrival_datetime.isoformat(),
                         "2024-03-10T01:10:00-05:00")
        self.assertEqual(arrival.trip_id, "NIGHT")

    def test_after_midnight_window_finds_same_operating_run(self):
        start = NEW_YORK.localize(datetime.datetime(2024, 3, 10, 0, 30))
        end = NEW_YORK.localize(datetime.datetime(2024, 3, 10, 2, 0))
        records = self.schedule.actual_stop_times(start=start, end=end)
        self.assertEqual([record.trip_id for record in records], ["NIGHT"])
        self.assertEqual(records[0].service_date, datetime.date(2024, 3, 9))
        self.assertEqual(records[0].arrival_gtfs_time, "25:10:00")

    def test_naive_window_is_read_in_agency_timezone(self):
        start = datetime.datetime(2024, 3, 10, 0, 30)
        end = datetime.datetime(2024, 3, 10, 2, 0)
        records = self.schedule.actual_stop_times(start=start, end=end)
        self.assertEqual([record.trip_id for record in records], ["NIGHT"])

    def test_service_exception_removes_the_service_day(self):
        from pygtfs.gtfs_entities import ServiceException

        self.schedule.session.add(ServiceException(
            feed_id=self.feed_id, service_id="DAILY", date="20240310",
            exception_type=2))
        self.schedule.session.commit()
        records = self.schedule.stop_times_for_trip("NIGHT", "20240310")
        self.assertEqual(records, [])

    def test_export_persists_queryable_rows(self):
        from pygtfs.gtfs_entities import ActualStopTime

        written = self.schedule.export_actual_stop_times(
            trip="NIGHT", service_date="20240309")
        self.assertEqual(written, 2)
        # Re-exporting the same scope replaces rather than duplicates.
        self.schedule.export_actual_stop_times(
            trip="NIGHT", service_date="20240309")
        rows = self.schedule.exported_stop_times
        self.assertEqual(len(rows), 2)

        arrival = [row for row in rows if row.stop_id == "S2"][0]
        self.assertEqual(arrival.service_date, datetime.date(2024, 3, 9))
        self.assertEqual(arrival.arrival_gtfs_time, "25:10:00")
        self.assertEqual(arrival.arrival_time, datetime.timedelta(
            hours=25, minutes=10))
        self.assertEqual(arrival.arrival_local, "2024-03-10T01:10:00-05:00")

        start_utc = datetime.datetime(2024, 3, 10, 6, 0)
        end_utc = datetime.datetime(2024, 3, 10, 7, 0)
        queried = (self.schedule.session.query(ActualStopTime)
                   .filter(ActualStopTime.arrival_utc >= start_utc)
                   .filter(ActualStopTime.arrival_utc < end_utc)
                   .all())
        self.assertEqual([row.stop_id for row in queried], ["S2"])
        self.assertEqual(queried[0].service_date, datetime.date(2024, 3, 9))

    def test_csv_export_keeps_both_representations(self):
        target = io.StringIO()
        written = self.schedule.write_actual_stop_times(
            target, trip="NIGHT", service_date="20240309")
        self.assertEqual(written, 2)
        rows = list(csv.DictReader(io.StringIO(target.getvalue())))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["service_date"], "2024-03-09")
        self.assertEqual(rows[1]["arrival_gtfs_time"], "25:10:00")
        self.assertEqual(rows[1]["arrival_time"], "25:10:00")
        self.assertEqual(rows[1]["arrival_datetime"],
                         "2024-03-10T01:10:00-05:00")

    def test_stop_time_helper_uses_agency_timezone(self):
        stop_time = [st for st in self.schedule.stop_times
                     if st.stop_id == "S2"][0]
        arrival, departure = stop_time.actual_times("20240309")
        self.assertEqual(arrival.isoformat(), "2024-03-10T01:10:00-05:00")
        self.assertEqual(arrival, departure)


if __name__ == "__main__":
    unittest.main()
