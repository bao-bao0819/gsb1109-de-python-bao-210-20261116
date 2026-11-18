# -*- coding: utf-8 -*-

import datetime
import os.path
import shutil
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from pygtfs import overwrite_feed
from pygtfs import Schedule
from pygtfs.exceptions import PygtfsException
from pygtfs.feed import Feed, derive_feed_name

from sqlalchemy.orm import Query

SAMPLE_FEED = os.path.join(os.path.dirname(__file__), "data", "sample_feed")


def copy_sample_feed(test_case):
    """Copy the sample feed into a temp directory and clean up afterwards."""
    temp_dir = tempfile.mkdtemp()
    test_case.addCleanup(shutil.rmtree, temp_dir)
    return shutil.copytree(
        SAMPLE_FEED, os.path.join(temp_dir, "sample_feed"))


def remove_table(feed_dir, filename):
    os.remove(os.path.join(feed_dir, filename))


def write_table(feed_dir, filename, content):
    with open(os.path.join(feed_dir, filename), "w") as table:
        table.write(content)


class TestSchedule(unittest.TestCase):
    def setUp(self):
        self.schedule = Schedule(":memory:")
        data_location = os.path.join(os.path.dirname(__file__),
                                     "data", "sample_feed")
        overwrite_feed(self.schedule, data_location)

    def test_routes(self):
        self.assertEqual(self.schedule.routes[0].route_id, "AB")

    def test_services(self):
        ser = [service.service_id for service in self.schedule.services]
        self.assertEqual(ser, ["FULLW", "WE"])

    def test_stops(self):
        self.assertEqual([st.stop_id for st in self.schedule.stops],
                         ['FUR_CREEK_RES', 'BEATTY_AIRPORT', 'BULLFROG',
                          'STAGECOACH', 'NADAV', 'NANAA', 'DADAN', 'EMSI',
                          'AMV'])

    def test_route_trips(self):
        self.assertEqual([tr.trip_id for tr in self.schedule.routes[0].trips],
                         ['AB1', 'AB2'])

    def test_trip_stop_times(self):
        self.assertEqual([(st.arrival_time if st.arrival_time else None,
                           st.departure_time if st.departure_time else None) for st in self.schedule.routes[0].trips[0].stop_times],
                        [(datetime.timedelta(0, 28800), datetime.timedelta(0, 28800)),
                         (datetime.timedelta(0, 29400), datetime.timedelta(0, 29700))])

    def test_service_trips(self):
        self.assertEqual([tr.trip_id for tr in self.schedule.services[1].trips],
                         ['AAMV1', 'AAMV2', 'AAMV3', 'AAMV4'])

    def test_stop_stop_times(self):
        self.assertEqual([(st.arrival_time, st.departure_time) for st in self.schedule.stops[0].stop_times],
                         [(datetime.timedelta(0, 33600), datetime.timedelta(0, 33600)),
                          (datetime.timedelta(0, 39600), datetime.timedelta(0, 39600))])

    def test_stop_translations(self):
        self.assertEqual(sorted([(tr.lang, tr.translation)
                                 for tr in self.schedule.stops[0].translations]),
                         [('EN', 'Furnace Creek Resort (Demo)'), ('HE', 'אתר הנופש ערוץ הכבשן (דמו)')])

    def test_agencies(self):
        self.assertEqual([ag.agency_id for ag in self.schedule.agencies],
                         ['DTA'])

    def test_agency_routes(self):
        self.assertEqual([rt.route_id for rt in self.schedule.agencies[0].routes],
                         ['AAMV', 'AB', 'BFC', 'CITY', 'EXT', 'STBA'])

    def test_transfers(self):
        self.assertEqual(3, len(self.schedule.stops[0].transfers_from))
        self.assertEqual(0, len(self.schedule.stops[0].transfers_to))
        self.assertEqual(0, len(self.schedule.stops[1].transfers_from))
        self.assertEqual(3, len(self.schedule.stops[1].transfers_to))
        self.assertEqual([
            (0, 'BEATTY_AIRPORT', '', ''),
            (3, 'BEATTY_AIRPORT', '', 'AAMV4'),
            (1, 'BEATTY_AIRPORT', 'STBA', ''),
        ], [(tr.transfer_type, tr.to_stop_id, tr.to_route_id, tr.to_trip_id) for tr in self.schedule.stops[1].transfers_to])

    def test_transfers_route_to_route(self):
        self.assertEqual(1, len(self.schedule.agencies[0].routes[0].transfers_from))
        self.assertEqual(0, len(self.schedule.agencies[0].routes[0].transfers_to))
        self.assertEqual(0, len(self.schedule.agencies[0].routes[5].transfers_from))
        self.assertEqual(1, len(self.schedule.agencies[0].routes[5].transfers_to))
        self.assertEqual([1], [tr.transfer_type for tr in self.schedule.agencies[0].routes[0].transfers_from])
        self.assertEqual([], [tr.transfer_type for tr in self.schedule.agencies[0].routes[0].transfers_to])
        self.assertEqual([], [tr.transfer_type for tr in self.schedule.agencies[0].routes[5].transfers_from])
        self.assertEqual([1], [tr.transfer_type for tr in self.schedule.agencies[0].routes[5].transfers_to])

    def test_transfers_trip_to_trip(self):
        self.assertEqual(1, len(self.schedule.trips[8].transfers_from))
        self.assertEqual(0, len(self.schedule.trips[8].transfers_to))
        self.assertEqual(0, len(self.schedule.trips[10].transfers_from))
        self.assertEqual(1, len(self.schedule.trips[10].transfers_to))
        self.assertEqual([3], [tr.transfer_type for tr in self.schedule.trips[8].transfers_from])
        self.assertEqual([], [tr.transfer_type for tr in self.schedule.trips[8].transfers_to])
        self.assertEqual([], [tr.transfer_type for tr in self.schedule.trips[10].transfers_from])
        self.assertEqual([3], [tr.transfer_type for tr in self.schedule.trips[10].transfers_to])

    def test_trips_bikes_allowed(self):
        for t in self.schedule.trips:
            self.assertIsNone(t.bikes_allowed)
        t = self.schedule.trips[0]
        t.bikes_allowed = 0
        t.bikes_allowed = 1
        t.bikes_allowed = 2
        with self.assertRaises(ValueError):
            t.bikes_allowed = 3
        with self.assertRaises(ValueError):
            t.bikes_allowed = -1
        self.assertEqual(t.bikes_allowed, 2)

    def test_query_methods(self):
        """
            Test that the object query getters return
            :py:class:`sqlalchemy.orm.Query` objects
        """
        # Ideally assertIsInstance would be used, but this is not Python 2.6
        # compatile
        self.assertEqual(isinstance(self.schedule.agencies_query, Query),
                         True)

class TestIgnoreFiles(unittest.TestCase):
    def setUp(self):
        self.schedule = Schedule(":memory:")
        data_location = os.path.join(os.path.dirname(__file__),
                                     "data", "sample_feed")
        overwrite_feed(self.schedule, data_location, ignore_files="routes.txt")

    def test_routes(self):
        self.assertFalse(self.schedule.routes)

    def test_services(self):
        ser = [service.service_id for service in self.schedule.services]
        self.assertEqual(ser, ["FULLW", "WE"])


class TestPathInputs(unittest.TestCase):
    def setUp(self):
        self.data_location = Path(os.path.dirname(__file__)) / "data" / "sample_feed"

    def test_derive_feed_name_accepts_path(self):
        self.assertEqual(derive_feed_name(self.data_location), "sample_feed")
        self.assertEqual(derive_feed_name(Path("/tmp/caltrain-ca-us.zip")),
                         "caltrain-ca-us.zip")

    def test_feed_accepts_path_for_directory_inputs(self):
        agency_rows = Feed(self.data_location).reader("agency.txt")
        self.assertEqual(next(agency_rows)[0], "agency_id")

    def test_feed_accepts_path_for_zip_file_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = Path(temp_dir) / "sample_feed.zip"
            with ZipFile(zip_path, "w") as zip_file:
                for feed_file in self.data_location.iterdir():
                    zip_file.write(feed_file, arcname=feed_file.name)

            feed = Feed(zip_path)
            agency_rows = feed.reader("agency.txt")

            self.assertEqual(feed.feed_name, "sample_feed.zip")
            self.assertEqual(next(agency_rows)[0], "agency_id")

    def test_feed_handles_trailing_slash_in_directory_path(self):
        path_with_trailing_slash = "{0}/".format(self.data_location)
        feed = Feed(path_with_trailing_slash)
        # the feed still opens as a directory-backed feed:
        agency_rows = feed.reader("agency.txt")

        self.assertEqual(feed.feed_name, "sample_feed")
        self.assertEqual(next(agency_rows)[0], "agency_id")


class TestServiceIntegrity(unittest.TestCase):
    """A missing optional calendar table must not reject a feed, but a trip
    must never keep a dangling service relationship."""

    def _load(self, feed_dir):
        schedule = Schedule(":memory:")
        overwrite_feed(schedule, feed_dir)
        return schedule

    def test_calendar_dates_only_feed_loads(self):
        # calendar.txt is optional when calendar_dates.txt is present.
        feed_dir = copy_sample_feed(self)
        remove_table(feed_dir, "calendar.txt")
        write_table(feed_dir, "calendar_dates.txt",
                    "service_id,date,exception_type\n"
                    "FULLW,20070604,2\n"
                    "WE,20070604,1\n")

        schedule = self._load(feed_dir)

        self.assertEqual(sorted(s.service_id for s in schedule.services),
                         ["FULLW", "WE"])
        self.assertTrue(all(trip.service is not None
                            for trip in schedule.trips))

    def test_dates_only_partial_service_coverage(self):
        # No calendar.txt and calendar_dates.txt omits a service that trips
        # still reference: keep the trips, back the gap with a placeholder.
        feed_dir = copy_sample_feed(self)
        remove_table(feed_dir, "calendar.txt")
        write_table(feed_dir, "calendar_dates.txt",
                    "service_id,date,exception_type\n"
                    "FULLW,20070604,2\n")

        schedule = self._load(feed_dir)

        self.assertEqual(sorted(s.service_id for s in schedule.services),
                         ["FULLW", "WE"])
        self.assertEqual([trip for trip in schedule.trips
                          if trip.service is None], [])
        we_service = schedule.services_by_id("WE")[0]
        self.assertFalse(any((we_service.monday, we_service.tuesday,
                              we_service.wednesday, we_service.thursday,
                              we_service.friday, we_service.saturday,
                              we_service.sunday)))

    def test_trip_referencing_undeclared_service(self):
        feed_dir = copy_sample_feed(self)
        trips_path = os.path.join(feed_dir, "trips.txt")
        with open(trips_path) as trips:
            content = trips.read()
        with open(trips_path, "w") as trips:
            trips.write(content.rstrip("\n") +
                        "\nAB,GHOST,GHOSTTRIP,to Nowhere,0,,\n")

        schedule = self._load(feed_dir)

        ghost_trip = next(trip for trip in schedule.trips
                          if trip.trip_id == "GHOSTTRIP")
        self.assertIsNotNone(ghost_trip.service)
        self.assertEqual(ghost_trip.service.service_id, "GHOST")
        self.assertIn("GHOST",
                      [service.service_id for service in schedule.services])

    def test_missing_both_calendar_tables_rejected(self):
        feed_dir = copy_sample_feed(self)
        remove_table(feed_dir, "calendar.txt")
        remove_table(feed_dir, "calendar_dates.txt")

        with self.assertRaises(PygtfsException):
            self._load(feed_dir)

    def test_missing_required_table_rejected(self):
        feed_dir = copy_sample_feed(self)
        remove_table(feed_dir, "routes.txt")

        with self.assertRaises(IOError):
            self._load(feed_dir)

    def test_other_optional_table_missing_does_not_reject_feed(self):
        feed_dir = copy_sample_feed(self)
        remove_table(feed_dir, "shapes.txt")

        schedule = self._load(feed_dir)
        self.assertEqual(len(schedule.trips), 11)
        self.assertEqual(schedule.shapes, [])


if __name__ == '__main__':
    unittest.main()
