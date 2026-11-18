# -*- coding: utf-8 -*-

import datetime
import os.path
import unittest

from pygtfs import overwrite_feed
from pygtfs import Schedule
from pygtfs import service_dates, ServiceDateFact
from pygtfs.exceptions import PygtfsValidationError
from pygtfs.gtfs_entities import Service, ServiceException
from pygtfs.service_dates import CALENDAR_SOURCE


def weekday_service():
    # 2007-01-01 is a Monday, so this covers one full week, Mon 1st - Sun 7th.
    return Service(service_id="WK", monday="1", tuesday="1", wednesday="1",
                   thursday="1", friday="1", saturday="0", sunday="0",
                   start_date="20070101", end_date="20070107")


def exception(date, exception_type, service_id="WK"):
    return ServiceException(service_id=service_id, date=date,
                            exception_type=exception_type)


class TestServiceDates(unittest.TestCase):
    def test_weekday_pattern_and_date_range(self):
        facts = service_dates(weekday_service(), [])
        self.assertEqual([fact.date for fact in facts],
                         [datetime.date(2007, 1, day) for day in range(1, 8)])
        self.assertEqual([fact.runs for fact in facts],
                         [True, True, True, True, True, False, False])

    def test_facts_keep_their_source(self):
        facts = service_dates(weekday_service(), [])
        self.assertEqual([fact.source for fact in facts],
                         [CALENDAR_SOURCE] * 7)

    def test_added_exception_makes_a_day_off_run(self):
        # the regular service rests on Sunday, but works one extra Sunday
        added = exception("20070107", 1)
        facts = service_dates(weekday_service(), [added])
        sunday = facts[-1]
        self.assertEqual(sunday.date, datetime.date(2007, 1, 7))
        self.assertTrue(sunday.runs)
        self.assertIs(sunday.source, added)

    def test_removed_exception_cancels_a_regular_day(self):
        removed = exception("20070103", 2)
        facts = service_dates(weekday_service(), [removed])
        wednesday = facts[2]
        self.assertEqual(wednesday.date, datetime.date(2007, 1, 3))
        self.assertFalse(wednesday.runs)
        self.assertIs(wednesday.source, removed)
        # the surrounding days still follow the regular pattern
        self.assertTrue(facts[1].runs)
        self.assertTrue(facts[3].runs)

    def test_exception_outside_the_date_range_is_included(self):
        added = exception("20070109", 1)
        facts = service_dates(weekday_service(), [added])
        self.assertEqual(facts[-1],
                         ServiceDateFact(datetime.date(2007, 1, 9), True, added))

    def test_exceptions_of_other_services_are_ignored(self):
        facts = service_dates(weekday_service(),
                              [exception("20070103", 2, service_id="OTHER")])
        self.assertTrue(facts[2].runs)

    def test_conflicting_exceptions_raise_instead_of_last_row_winning(self):
        service = weekday_service()
        with self.assertRaises(PygtfsValidationError):
            service_dates(service, [exception("20070103", 1),
                                    exception("20070103", 2)])
        # the input order must not matter either
        with self.assertRaises(PygtfsValidationError):
            service_dates(service, [exception("20070103", 2),
                                    exception("20070103", 1)])

    def test_identical_duplicate_exceptions_collapse(self):
        facts = service_dates(weekday_service(), [exception("20070103", 2),
                                                  exception("20070103", 2)])
        self.assertEqual(len(facts), 7)
        self.assertFalse(facts[2].runs)

    def test_revising_an_exception_only_changes_its_own_dates(self):
        service = weekday_service()
        cancelled_wednesday = exception("20070103", 2)
        cancelled_friday = exception("20070105", 2)
        before = service_dates(service, [cancelled_wednesday, cancelled_friday])

        # the Friday cancellation is revised into an extra service day
        revised_friday = exception("20070105", 1)
        after = service_dates(service, [cancelled_wednesday, revised_friday])

        friday = datetime.date(2007, 1, 5)
        changed = [fact_before.date for fact_before, fact_after
                   in zip(before, after) if fact_before != fact_after]
        self.assertEqual(changed, [friday])

    def test_service_method_uses_related_exceptions_by_default(self):
        service = weekday_service()
        removed = exception("20070103", 2)
        service.exceptions = [removed]
        facts = service.service_dates()
        self.assertFalse(facts[2].runs)
        self.assertIs(facts[2].source, removed)


class TestServiceDatesOnSampleFeed(unittest.TestCase):
    def setUp(self):
        self.schedule = Schedule(":memory:")
        data_location = os.path.join(os.path.dirname(__file__),
                                     "data", "sample_feed")
        overwrite_feed(self.schedule, data_location)

    def test_sample_feed_removal_keeps_source(self):
        service = self.schedule.services_by_id("FULLW")[0]
        facts = service.service_dates()
        by_date = {fact.date: fact for fact in facts}
        # 2007-06-04 is a Monday, regularly served, but cancelled in
        # calendar_dates
        cancelled = by_date[datetime.date(2007, 6, 4)]
        self.assertFalse(cancelled.runs)
        self.assertIsInstance(cancelled.source, ServiceException)
        self.assertEqual(cancelled.source.exception_type, 2)
        # the next day is back to the regular pattern
        regular = by_date[datetime.date(2007, 6, 5)]
        self.assertTrue(regular.runs)
        self.assertEqual(regular.source, CALENDAR_SOURCE)

    def test_service_exceptions_are_related_to_the_service(self):
        service = self.schedule.services_by_id("FULLW")[0]
        self.assertEqual([exc.date for exc in service.exceptions],
                         [datetime.date(2007, 6, 4)])


if __name__ == '__main__':
    unittest.main()
