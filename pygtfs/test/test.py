# -*- coding: utf-8 -*-

import datetime
import os.path
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

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


class TestLoopLine(unittest.TestCase):
    """A loop line visits the same stop_id twice in one trip.

    Each stop visit is a distinct StopTime row keyed by stop_sequence, and
    they all point at the same Stop entity rather than being deduplicated by
    stop_id.
    """

    def setUp(self):
        self.schedule = Schedule(":memory:")
        data_location = os.path.join(os.path.dirname(__file__),
                                     "data", "loop_feed")
        overwrite_feed(self.schedule, data_location)

    def test_repeated_stop_visits_are_both_exported(self):
        from pygtfs.gtfs_entities import StopTime
        rows = (self.schedule.session.query(StopTime)
                .filter_by(trip_id='LOOP', stop_id='STAGECOACH')
                .order_by(StopTime.stop_sequence).all())
        self.assertEqual([st.stop_sequence for st in rows], [1, 4])
        self.assertEqual(
            [(st.arrival_time, st.departure_time) for st in rows],
            [(datetime.timedelta(hours=6), datetime.timedelta(hours=6)),
             (datetime.timedelta(seconds=23400),
              datetime.timedelta(seconds=23520))])

    def test_trip_stop_times_follow_stop_sequence(self):
        from pygtfs.gtfs_entities import Trip
        trip = self.schedule.session.query(Trip).filter_by(
            trip_id='LOOP').one()
        self.assertEqual(
            [(st.stop_sequence, st.stop_id) for st in trip.stop_times],
            [(1, 'STAGECOACH'), (2, 'NANAA'), (3, 'NADAV'),
             (4, 'STAGECOACH'), (5, 'DADAN')])

    def test_visits_share_a_single_stop_entity(self):
        from pygtfs.gtfs_entities import Stop, Trip
        stop = self.schedule.session.query(Stop).filter_by(
            stop_id='STAGECOACH').one()
        trip = self.schedule.session.query(Trip).filter_by(
            trip_id='LOOP').one()
        visits = [st for st in trip.stop_times if st.stop_id == 'STAGECOACH']
        self.assertEqual(len(visits), 2)
        self.assertTrue(all(st.stop is stop for st in visits))
        self.assertEqual(
            sorted(st.stop_sequence for st in stop.stop_times), [1, 4])

    def test_boarding_second_visit_does_not_revisit_first(self):
        from pygtfs.gtfs_entities import Trip
        trip = self.schedule.session.query(Trip).filter_by(
            trip_id='LOOP').one()
        ordered = trip.stop_times
        second_visit = next(st for st in ordered
                            if st.stop_id == 'STAGECOACH'
                            and st.stop_sequence == 4)
        onward = ordered[ordered.index(second_visit):]
        self.assertEqual(
            [(st.stop_sequence, st.stop_id) for st in onward],
            [(4, 'STAGECOACH'), (5, 'DADAN')])
        sequences = [st.stop_sequence for st in onward]
        self.assertEqual(sequences, sorted(sequences))
        self.assertEqual(sequences.count(1), 0)


class TestDuplicateStopTimeConflict(unittest.TestCase):
    """Two rows sharing the same (trip, stop, sequence) primary key.

    The first occurrence is retained and the conflicting row is skipped with a
    warning instead of aborting the whole feed.
    """

    def setUp(self):
        self.schedule = Schedule(":memory:")
        self.data_location = os.path.join(os.path.dirname(__file__),
                                          "data", "loop_conflict_feed")

    def test_original_row_is_kept_and_duplicate_skipped(self):
        import logging
        from pygtfs.gtfs_entities import StopTime
        with self.assertLogs('pygtfs.loader', level=logging.WARNING) as logs:
            overwrite_feed(self.schedule, self.data_location)
        rows = (self.schedule.session.query(StopTime)
                .filter_by(trip_id='LOOP')
                .order_by(StopTime.stop_sequence).all())
        self.assertEqual(len(rows), 5)
        conflicting = [st for st in rows if st.stop_sequence == 4]
        self.assertEqual(len(conflicting), 1)
        self.assertEqual(conflicting[0].stop_headsign,
                         'first occurrence')
        self.assertTrue(any('duplicate' in msg.lower()
                            for msg in logs.output))


if __name__ == '__main__':
    unittest.main()
