# -*- coding: utf-8 -*-

import datetime
import os.path
import shutil
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from pygtfs import append_feed
from pygtfs import overwrite_feed
from pygtfs import Schedule
from pygtfs.feed import Feed, derive_feed_name

from sqlalchemy.orm import Query


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


class TestFeedVersions(unittest.TestCase):
    def setUp(self):
        self.schedule = Schedule(":memory:")
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        # keep the original directory name, so that every import is seen as
        # another revision of the same feed
        self.feed_dir = os.path.join(self.tmpdir.name, "sample_feed")
        shutil.copytree(os.path.join(os.path.dirname(__file__),
                                     "data", "sample_feed"),
                        self.feed_dir)

    def publish(self, version, start_date, end_date):
        with open(os.path.join(self.feed_dir, "feed_info.txt"), "w") as fh:
            fh.write("feed_publisher_name,feed_publisher_url,feed_lang,"
                     "feed_start_date,feed_end_date,feed_version\n")
            fh.write("the feeder,transit.example.com,en,{0},{1},{2}\n"
                     .format(start_date, end_date, version))

    def test_version_and_validity_are_stored(self):
        self.publish("2026-w45", "20261102", "20261129")
        append_feed(self.schedule, self.feed_dir,
                    append_date=datetime.date(2026, 11, 2))
        feed = self.schedule.feeds[0]
        self.assertEqual(feed.feed_version, "2026-w45")
        self.assertEqual(feed.feed_start_date, datetime.date(2026, 11, 2))
        self.assertEqual(feed.feed_end_date, datetime.date(2026, 11, 29))
        self.assertEqual(feed.feed_append_date, datetime.date(2026, 11, 2))

    def test_reimporting_same_revision_is_idempotent(self):
        self.publish("2026-w45", "20261102", "20261129")
        append_feed(self.schedule, self.feed_dir)
        trip_count = len(self.schedule.trips)
        append_feed(self.schedule, self.feed_dir)
        self.assertEqual(len(self.schedule.feeds), 1)
        self.assertEqual(len(self.schedule.trips), trip_count)

    def test_overwrite_same_revision_replaces_without_duplicates(self):
        self.publish("2026-w45", "20261102", "20261129")
        overwrite_feed(self.schedule, self.feed_dir)
        trip_count = len(self.schedule.trips)
        overwrite_feed(self.schedule, self.feed_dir)
        self.assertEqual(len(self.schedule.feeds), 1)
        self.assertEqual(len(self.schedule.trips), trip_count)

    def test_late_correction_does_not_erase_previous_version(self):
        self.publish("2026-w45", "20261102", "20261129")
        overwrite_feed(self.schedule, self.feed_dir,
                       append_date=datetime.date(2026, 11, 2))
        trip_count = len(self.schedule.trips)
        self.publish("2026-w46", "20261109", "20261206")
        overwrite_feed(self.schedule, self.feed_dir,
                       append_date=datetime.date(2026, 11, 9))
        self.assertEqual(len(self.schedule.feeds), 2)
        self.assertEqual(len(self.schedule.trips), 2 * trip_count)
        self.assertEqual(sorted(f.feed_version for f in self.schedule.feeds),
                         ["2026-w45", "2026-w46"])

    def test_feed_as_of(self):
        self.publish("2026-w45", "20261102", "20261129")
        append_feed(self.schedule, self.feed_dir,
                    append_date=datetime.date(2026, 11, 2))
        self.publish("2026-w46", "20261109", "20261206")
        append_feed(self.schedule, self.feed_dir,
                    append_date=datetime.date(2026, 11, 9))
        self.assertIsNone(self.schedule.feed_as_of(datetime.date(2026, 11, 1)))
        self.assertEqual(self.schedule.feed_as_of(datetime.date(2026, 11, 2)).feed_version,
                         "2026-w45")
        self.assertEqual(self.schedule.feed_as_of(datetime.date(2026, 11, 8)).feed_version,
                         "2026-w45")
        self.assertEqual(self.schedule.feed_as_of(datetime.date(2026, 11, 9)).feed_version,
                         "2026-w46")
        self.assertEqual(self.schedule.feed_as_of("2026-11-19").feed_version,
                         "2026-w46")
        self.assertEqual(self.schedule.feed_as_of("2026-11-19", feed_name="sample_feed").feed_version,
                         "2026-w46")
        self.assertIsNone(self.schedule.feed_as_of("2026-11-19", feed_name="other_feed"))

    def test_feed_as_of_gives_access_to_the_data_published_then(self):
        self.publish("2026-w45", "20261102", "20261129")
        append_feed(self.schedule, self.feed_dir,
                    append_date=datetime.date(2026, 11, 2))
        self.publish("2026-w46", "20261109", "20261206")
        append_feed(self.schedule, self.feed_dir,
                    append_date=datetime.date(2026, 11, 9))
        feed = self.schedule.feed_as_of(datetime.date(2026, 11, 5))
        route_ab = [rt for rt in feed.routes if rt.route_id == 'AB'][0]
        self.assertEqual([tr.trip_id for tr in route_ab.trips],
                         ['AB1', 'AB2'])


if __name__ == '__main__':
    unittest.main()
