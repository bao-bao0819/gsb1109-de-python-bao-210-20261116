"""Derive the concrete dates on which a service runs.

The regular weekly pattern from ``calendar`` (a date range plus one flag
per weekday) is combined with the exceptions from ``calendar_dates``
(``exception_type`` 1 adds a day, 2 removes a day). Every resulting fact
keeps track of its source: the regular calendar pattern, or the specific
exception that overrides it.

The computation is a pure function of the service and its exceptions:
each date is decided independently, so changing one exception only ever
changes the facts for the dates that exception touches.
"""

import datetime
from collections import namedtuple

from .exceptions import PygtfsValidationError

EXCEPTION_ADDED = 1
EXCEPTION_REMOVED = 2

#: Source marker used for facts that follow from the regular weekly pattern.
CALENDAR_SOURCE = "calendar"

_WEEKDAY_ATTRIBUTES = ("monday", "tuesday", "wednesday", "thursday",
                       "friday", "saturday", "sunday")


class ServiceDateFact(namedtuple("ServiceDateFact", ["date", "runs", "source"])):
    """Whether a service runs on one date, and what that is based on.

    ``source`` is :data:`CALENDAR_SOURCE` when the fact follows from the
    regular weekly pattern of the service, or the
    :py:class:`pygtfs.gtfs_entities.ServiceException` that overrides the
    regular pattern for this date.
    """
    __slots__ = ()

    def __repr__(self):
        return "ServiceDateFact(date={0!r}, runs={1!r}, source={2!r})".format(
            self.date, self.runs, self.source)


def _exceptions_by_date(service, exceptions):
    """Index the exceptions of a service by date, rejecting conflicts.

    Exceptions for other services are ignored. Two exceptions for the
    same date with the same ``exception_type`` collapse into one, so the
    result does not depend on the input order. Two exceptions for the
    same date with different types are a contradiction; silently keeping
    the last row would hide a data error, so a
    :py:class:`pygtfs.exceptions.PygtfsValidationError` is raised instead.
    """
    by_date = {}
    for exception in exceptions:
        if exception.service_id != service.service_id:
            continue
        previous = by_date.get(exception.date)
        if previous is not None:
            if previous.exception_type != exception.exception_type:
                raise PygtfsValidationError(
                    "Conflicting exceptions for service {0} on {1}: "
                    "exception_type {2} and {3}".format(
                        service.service_id, exception.date,
                        previous.exception_type, exception.exception_type))
            continue
        by_date[exception.date] = exception
    return by_date


def service_dates(service, exceptions=None):
    """Compute the facts about every date a service is scheduled for.

    :param service: a :py:class:`pygtfs.gtfs_entities.Service` with a date
        range and one flag per weekday.
    :param exceptions: the candidate
        :py:class:`pygtfs.gtfs_entities.ServiceException` rows. Exceptions
        belonging to other services are ignored. When omitted, the
        exceptions related to the service are used.
    :return: a list of :py:class:`ServiceDateFact`, one per date in the
        service date range plus one per exception date outside that range,
        ordered by date. Dates on which the service does not run are
        included with ``runs`` set to ``False``.
    """
    if exceptions is None:
        exceptions = service.exceptions
    exceptions_by_date = _exceptions_by_date(service, exceptions)

    facts = []
    if service.start_date is not None and service.end_date is not None:
        day = service.start_date
        while day <= service.end_date:
            exception = exceptions_by_date.pop(day, None)
            if exception is None:
                runs = getattr(service, _WEEKDAY_ATTRIBUTES[day.weekday()])
                facts.append(ServiceDateFact(day, runs, CALENDAR_SOURCE))
            else:
                facts.append(ServiceDateFact(
                    day, exception.exception_type == EXCEPTION_ADDED,
                    exception))
            day += datetime.timedelta(days=1)
    # exceptions dated outside the regular date range
    for day in sorted(exceptions_by_date):
        exception = exceptions_by_date[day]
        facts.append(ServiceDateFact(
            day, exception.exception_type == EXCEPTION_ADDED, exception))
    facts.sort(key=lambda fact: fact.date)
    return facts
