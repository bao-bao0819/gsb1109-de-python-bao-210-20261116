# -*- coding: utf-8 -*-

import datetime
import unittest

import pygtfs
from pygtfs import Schedule
from pygtfs.exceptions import FrequencyConflictError, InvalidFrequencyError
from pygtfs.frequencies import (DepartureInstance, FrequencyConflict,
                                ServiceWindow, detect_conflicts,
                                expand_frequencies, format_time,
                                frequency_instances, parse_time,
                                service_windows)


def td(hours=0, minutes=0, seconds=0):
    return datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)


class FakeFrequency(object):
    feed_id = 1

    def __init__(self, trip_id, start_time, end_time, headway_secs,
                 exact_times=None, trip=None):
        self.trip_id = trip_id
        self.start_time = parse_time(start_time)
        self.end_time = parse_time(end_time)
        self.headway_secs = headway_secs
        self.exact_times = exact_times
        self.trip = trip


class TestTimeHelpers(unittest.TestCase):
    def test_parse_and_format_extended_hours(self):
        self.assertEqual(parse_time('25:10:00'), td(hours=25, minutes=10))
        self.assertEqual(format_time(td(hours=25, minutes=10)), '25:10:00')

    def test_parse_accepts_timedelta_and_none(self):
        value = td(minutes=5)
        self.assertIs(parse_time(value), value)
        self.assertIsNone(parse_time(None))


class TestExactFrequencies(unittest.TestCase):
    def test_exact_row_expands_to_departures_before_end(self):
        row = FakeFrequency('T1', '06:00:00', '07:00:00', 600, exact_times=1)
        expansion = expand_frequencies(row)

        self.assertEqual(
            [format_time(item.departure_time) for item in expansion.departures],
            ['06:00:00', '06:10:00', '06:20:00', '06:30:00', '06:40:00',
             '06:50:00'])
        self.assertEqual(expansion.windows, [])
        self.assertEqual(expansion.conflicts, [])

    def test_end_time_is_exclusive_by_default(self):
        row = FakeFrequency('T1', '06:00:00', '06:20:00', 600, exact_times=1)
        self.assertEqual(
            [format_time(item.departure_time)
             for item in expand_frequencies(row).departures],
            ['06:00:00', '06:10:00'])

    def test_end_time_can_be_included_explicitly(self):
        row = FakeFrequency('T1', '06:00:00', '06:20:00', 600, exact_times=1)
        self.assertEqual(
            [format_time(item.departure_time)
             for item in expand_frequencies(row, end_inclusive=True).departures],
            ['06:00:00', '06:10:00', '06:20:00'])

    def test_partial_final_headway_is_not_rounded_up(self):
        row = FakeFrequency('T1', '06:00:00', '06:25:00', 600, exact_times=1)
        self.assertEqual(
            [format_time(item.departure_time)
             for item in expand_frequencies(row).departures],
            ['06:00:00', '06:10:00', '06:20:00'])

    def test_invalid_headway_is_rejected(self):
        row = FakeFrequency('T1', '06:00:00', '07:00:00', 0, exact_times=1)
        with self.assertRaises(InvalidFrequencyError):
            expand_frequencies(row)

    def test_invalid_period_is_rejected(self):
        row = FakeFrequency('T1', '07:00:00', '06:00:00', 600, exact_times=1)
        with self.assertRaises(InvalidFrequencyError):
            expand_frequencies(row)


class TestNonExactFrequencies(unittest.TestCase):
    def test_non_exact_row_yields_window_not_departures(self):
        row = FakeFrequency('T1', '06:00:00', '07:00:00', 600, exact_times=0)
        expansion = expand_frequencies(row)

        self.assertEqual(expansion.departures, [])
        self.assertEqual(len(expansion.windows), 1)
        window = expansion.windows[0]
        self.assertIsInstance(window, ServiceWindow)
        self.assertEqual(window.start_time, td(hours=6))
        self.assertEqual(window.end_time, td(hours=7))
        self.assertEqual(window.waiting_range, (td(), td(minutes=10)))
        self.assertEqual(window.min_wait, td())
        self.assertEqual(window.max_wait, td(minutes=10))
        self.assertFalse(window.is_exact)

    def test_omitted_exact_times_is_frequency_based(self):
        row = FakeFrequency('T1', '06:00:00', '07:00:00', 600, exact_times=None)
        expansion = expand_frequencies(row)
        self.assertEqual(expansion.departures, [])
        self.assertEqual(len(expansion.windows), 1)

    def test_frequency_instances_helper_skips_non_exact_rows(self):
        exact = FakeFrequency('T1', '06:00:00', '07:00:00', 600,
                              exact_times=1)
        approximate = FakeFrequency('T2', '06:00:00', '07:00:00', 600,
                                    exact_times=0)
        self.assertEqual(len(frequency_instances([exact, approximate])), 6)
        self.assertEqual(len(service_windows([exact, approximate])), 1)

    def test_empty_window_is_allowed(self):
        row = FakeFrequency('T1', '06:00:00', '06:00:00', 600, exact_times=0)
        expansion = expand_frequencies(row)
        self.assertEqual(expansion.departures, [])
        self.assertEqual(len(expansion.windows), 1)

    def test_hours_beyond_24_are_preserved(self):
        row = FakeFrequency('T1', '25:00:00', '27:00:00', 1800, exact_times=1)
        self.assertEqual(
            [format_time(item.departure_time)
             for item in expand_frequencies(row).departures],
            ['25:00:00', '25:30:00', '26:00:00', '26:30:00'])


class TestConflicts(unittest.TestCase):
    def test_adjacent_periods_do_not_conflict(self):
        rows = [
            FakeFrequency('T1', '06:00:00', '08:00:00', 600, exact_times=1),
            FakeFrequency('T1', '08:00:00', '09:00:00', 600, exact_times=1),
        ]
        self.assertEqual(detect_conflicts(rows), [])
        expansion = expand_frequencies(rows)
        self.assertFalse(expansion.has_conflicts)
        self.assertEqual(len(expansion.departures), 12 + 6)

    def test_overlap_is_reported(self):
        rows = [
            FakeFrequency('T1', '06:00:00', '08:00:00', 600, exact_times=1),
            FakeFrequency('T1', '07:30:00', '09:00:00', 600, exact_times=1),
        ]
        expansion = expand_frequencies(rows)

        self.assertTrue(expansion.has_conflicts)
        self.assertEqual(len(expansion.conflicts), 1)
        conflict = expansion.conflicts[0]
        self.assertIsInstance(conflict, FrequencyConflict)
        self.assertEqual(conflict.trip_id, 'T1')
        self.assertEqual(conflict.overlap_start, td(hours=7, minutes=30))
        self.assertEqual(conflict.overlap_end, td(hours=8))
        self.assertEqual(conflict.overlap, td(minutes=30))

    def test_overlap_can_be_raised(self):
        rows = [
            FakeFrequency('T1', '06:00:00', '08:00:00', 600, exact_times=1),
            FakeFrequency('T1', '07:30:00', '09:00:00', 600, exact_times=1),
        ]
        with self.assertRaises(FrequencyConflictError):
            expand_frequencies(rows, raise_on_conflict=True)

    def test_containing_period_is_reported(self):
        rows = [
            FakeFrequency('T1', '06:00:00', '10:00:00', 600, exact_times=0),
            FakeFrequency('T1', '07:00:00', '08:00:00', 600, exact_times=0),
        ]
        conflicts = detect_conflicts(rows)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].overlap, td(hours=1))

    def test_different_trips_do_not_conflict(self):
        rows = [
            FakeFrequency('T1', '06:00:00', '08:00:00', 600, exact_times=1),
            FakeFrequency('T2', '07:00:00', '09:00:00', 600, exact_times=1),
        ]
        self.assertEqual(detect_conflicts(rows), [])

    def test_same_trip_in_different_feeds_does_not_conflict(self):
        first = FakeFrequency('T1', '06:00:00', '08:00:00', 600,
                              exact_times=1)
        second = FakeFrequency('T1', '07:00:00', '09:00:00', 600,
                               exact_times=1)
        second.feed_id = 2
        self.assertEqual(detect_conflicts([first, second]), [])

    def test_empty_period_does_not_conflict(self):
        rows = [
            FakeFrequency('T1', '06:00:00', '08:00:00', 600, exact_times=1),
            FakeFrequency('T1', '07:00:00', '07:00:00', 600, exact_times=0),
        ]
        self.assertEqual(detect_conflicts(rows), [])


class TestScheduleIntegration(unittest.TestCase):
    def setUp(self):
        import os.path
        self.schedule = Schedule(':memory:')
        data_location = os.path.join(os.path.dirname(__file__), 'data',
                                     'sample_feed')
        pygtfs.overwrite_feed(self.schedule, data_location)

    def tearDown(self):
        self.schedule.session.close()

    def test_sample_feed_rows_are_all_frequency_based(self):
        expansion = self.schedule.expand_frequencies()
        self.assertEqual(expansion.departures, [])
        self.assertEqual(len(expansion.windows), 11)
        self.assertEqual(expansion.conflicts, [])

    def test_exact_times_value_is_persisted_and_expanded(self):
        from pygtfs.gtfs_entities import Frequency
        row = Frequency(feed_id=1, trip_id='AB1', start_time='06:00:00',
                        end_time='06:30:00', headway_secs=600,
                        exact_times=1)
        self.schedule.session.add(row)
        self.schedule.session.commit()

        self.assertTrue(row.is_exact)
        self.assertEqual(
            [format_time(item.departure_time) for item in row.departures()],
            ['06:00:00', '06:10:00', '06:20:00'])
        self.assertIsNone(row.service_window())

        expansion = self.schedule.expand_frequencies(trip_id='AB1')
        self.assertEqual(len(expansion.departures), 3)
        self.assertEqual(expansion.windows, [])

    def test_convenience_methods(self):
        self.assertEqual(self.schedule.frequency_instances(), [])
        self.assertEqual(len(self.schedule.service_windows()), 11)
        self.assertEqual(self.schedule.frequency_conflicts(), [])

    def test_trip_convenience_methods(self):
        trip = self.schedule.trips_by_id('STBA')[0]
        self.assertEqual(trip.frequency_instances(), [])
        self.assertEqual(len(trip.service_windows()), 1)
        self.assertEqual(trip.frequency_conflicts(), [])


if __name__ == '__main__':
    unittest.main()
