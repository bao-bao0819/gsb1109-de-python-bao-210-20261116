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
from pygtfs.gtfs_entities import (MATCH_BASIS_RECORD_ID,
                                  MATCH_BASIS_FIELD_VALUE)

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

class TestTranslationViews(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        feed_dir = Path(self.temp_dir.name) / "translation_feed"
        feed_dir.mkdir()
        self._write(feed_dir / "agency.txt", [
            "agency_id,agency_name,agency_url,agency_timezone",
            "A1,Central,http://example.com,America/Los_Angeles",
        ])
        self._write(feed_dir / "stops.txt", [
            "stop_id,stop_name,stop_lat,stop_lon",
            "S1,Central,36.0,-117.0",
            "S2,Central,36.1,-117.1",
        ])
        self._write(feed_dir / "routes.txt", [
            "route_id,agency_id,route_short_name,route_long_name,route_type",
            "R1,A1,50,Central Loop,3",
        ])
        self._write(feed_dir / "calendar.txt", [
            "service_id,monday,tuesday,wednesday,thursday,friday,saturday,"
            "sunday,start_date,end_date",
            "SVC,1,1,1,1,1,1,1,20070101,20101231",
        ])
        self._write(feed_dir / "trips.txt", [
            "route_id,service_id,trip_id",
            "R1,SVC,T1",
        ])
        self._write(feed_dir / "stop_times.txt", [
            "trip_id,arrival_time,departure_time,stop_id,stop_sequence",
            "T1,8:00:00,8:00:00,S1,1",
        ])
        self._write(feed_dir / "translations.txt", [
            "trans_id,lang,translation,table_name,field_name,record_id",
            # record-id scope: only S1 must ever receive this translation
            "Central Station,HE,תחנה מרכזית,stops,stop_name,S1",
            # route record-id scope on a different table/field
            "50,HE,חמישים,routes,route_short_name,R1",
            # field-value scope: literal "Central" fans out to every hit
            "Central,HE,מרכז,,,",
            "Central,EN,Central,,,",
            "Central,FR,Centre,,,",
        ])
        self.schedule = Schedule(":memory:")
        overwrite_feed(self.schedule, str(feed_dir))
        self.feed_id = self.schedule.feeds[0].feed_id

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def _write(path, lines):
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _stop(self, stop_id):
        return self.schedule.stops_by_id(stop_id)[0]

    def test_record_id_translation_is_scoped_to_its_record(self):
        s1 = self._stop("S1")
        s2 = self._stop("S2")
        record_hits = [(t.trans_id, t.lang) for t in s1.translations
                       if t.trans_id == "Central Station"]
        self.assertEqual(record_hits, [("Central Station", "HE")])
        # the same stop_name on S2 must not inherit S1's record-id hit
        self.assertNotIn(("Central Station", "HE"),
                         [(t.trans_id, t.lang) for t in s2.translations])
        agency = self.schedule.agencies_by_id("A1")[0]
        self.assertNotIn(("Central Station", "HE"),
                         [(t.trans_id, t.lang) for t in agency.translations])

    def test_field_value_translation_fans_out_to_every_hit(self):
        expected = {("Central", "HE"), ("Central", "EN"), ("Central", "FR")}
        for record in (self._stop("S1"), self._stop("S2"),
                       self.schedule.agencies_by_id("A1")[0]):
            self.assertEqual(
                expected,
                {(t.trans_id, t.lang) for t in record.translations
                 if t.trans_id == "Central"})

    def test_route_record_id_translation(self):
        route = self.schedule.routes_by_id("R1")[0]
        self.assertIn(("50", "HE"),
                      [(t.trans_id, t.lang) for t in route.translations])
        binding = (self.schedule.translation_targets_query
                   .filter_by(target_table="routes",
                              target_record_id="R1").one())
        self.assertEqual(MATCH_BASIS_RECORD_ID, binding.match_basis)
        self.assertIsNone(binding.field_value)

    def test_binding_keeps_the_actual_match_basis(self):
        bindings = self.schedule.translation_targets
        record_binding = next(b for b in bindings
                              if b.match_basis == MATCH_BASIS_RECORD_ID
                              and b.target_record_id == "S1")
        self.assertEqual(record_binding.target_table, "stops")
        self.assertEqual(record_binding.target_field, "stop_name")
        self.assertEqual(record_binding.trans_id, "Central Station")
        self.assertIsNone(record_binding.field_value)
        value_bindings = [b for b in bindings
                          if b.match_basis == MATCH_BASIS_FIELD_VALUE]
        self.assertTrue(value_bindings)
        for binding in value_bindings:
            self.assertEqual(binding.field_value, "Central")
        scopes = {(b.target_table, b.target_field) for b in value_bindings}
        self.assertEqual(scopes, {("stops", "stop_name"),
                                  ("agency", "agency_name")})

    def test_view_lists_candidates_when_target_not_unique(self):
        rows = self.schedule.translation_view(
            lang="HE", table_name="stops", record_id="S1")
        # record-id and field-value translations both apply: both stay visible
        self.assertEqual(
            {MATCH_BASIS_RECORD_ID, MATCH_BASIS_FIELD_VALUE},
            {row.match_basis for row in rows})
        self.assertEqual(
            {"Central Station", "Central"},
            {row.trans_id for row in rows})

    def test_view_scope_filters_do_not_leak_across_records(self):
        rows = self.schedule.translation_view(
            lang="HE", table_name="stops", record_id="S2")
        self.assertTrue(rows)
        self.assertTrue(all(row.match_basis == MATCH_BASIS_FIELD_VALUE
                            for row in rows))
        self.assertTrue(all(row.trans_id == "Central" for row in rows))

    def test_view_filters_by_language_and_table(self):
        fr_rows = self.schedule.translation_view(lang="FR")
        self.assertTrue(fr_rows)
        self.assertTrue(all(row.lang == "FR" for row in fr_rows))
        agency_rows = self.schedule.translation_view(table_name="agency")
        self.assertTrue(agency_rows)
        self.assertTrue(all(row.target_table == "agency" for row in agency_rows))
        scoped = self.schedule.translation_view(
            lang="EN", table_name="agency", field_name="agency_name",
            record_id="A1", feed_id=self.feed_id)
        self.assertEqual(len(scoped), 1)
        self.assertEqual(scoped[0].translation, "Central")
        self.assertEqual(scoped[0].match_basis, MATCH_BASIS_FIELD_VALUE)


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


if __name__ == '__main__':
    unittest.main()
