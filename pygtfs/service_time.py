"""Helpers for GTFS service days and timezone-aware stop times.

GTFS ``stop_times.txt`` values are not wall-clock times.  They are offsets
from the start of a *service day*, which the specification defines as noon
minus twelve hours in the agency's timezone.  On daylight-saving transition
days that starting point is not necessarily local midnight, and it can even
fall on the previous or following calendar date.  The helpers in this module
keep the original offset intact and resolve it to a timezone-aware datetime
without ever wrapping it modulo 24 hours.
"""

import datetime
from collections import namedtuple

import pytz

from .exceptions import PygtfsTimezoneError

_SERVICE_DAY_START = datetime.time(12, 0)
_ONE_DAY = datetime.timedelta(days=1)

ResolvedStopTime = namedtuple(
    "ResolvedStopTime",
    [
        "feed_id",
        "trip_id",
        "service_id",
        "service_date",
        "stop_sequence",
        "stop_id",
        "arrival_time",
        "departure_time",
        "arrival_gtfs_time",
        "departure_gtfs_time",
        "arrival_datetime",
        "departure_datetime",
        "timezone",
    ],
)
ResolvedStopTime.__doc__ = """A resolved stop time.

``arrival_time`` and ``departure_time`` keep the original GTFS offsets,
including hours greater than twenty-four.  ``arrival_gtfs_time`` and
``departure_gtfs_time`` are the same values in ``HH:MM:SS`` form.
``service_date`` is the operating day the trip belongs to; the timezone-aware
``arrival_datetime`` and ``departure_datetime`` may fall on the next calendar
date without changing that service date.
"""


def coerce_service_date(value):
    """Return a :class:`datetime.date` from a date, datetime, or string.

    Strings may be ISO dates (``2024-01-01``) or GTFS dates (``20240101``).
    """

    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    if isinstance(value, str):
        for fmt in ("%Y%m%d", "%Y-%m-%d"):
            try:
                return datetime.datetime.strptime(value, fmt).date()
            except ValueError:
                pass
        raise ValueError("Unrecognized service date: {0!r}".format(value))
    raise TypeError("service date must be a date or string, got {0!r}".format(value))


def get_timezone(value):
    """Return a usable timezone for a name or timezone object.

    Strings are interpreted as IANA timezone names.  A ``pytz`` timezone is
    returned as-is; other ``tzinfo`` objects (for example
    :class:`zoneinfo.ZoneInfo`) are accepted as well.
    """

    if isinstance(value, str):
        try:
            return pytz.timezone(value)
        except pytz.UnknownTimeZoneError:
            raise PygtfsTimezoneError(
                "Unknown agency timezone: {0!r}".format(value))
    if isinstance(value, datetime.tzinfo):
        return value
    raise PygtfsTimezoneError(
        "Expected an IANA timezone name or tzinfo object, got {0!r}".format(value))


def timezone_name(value):
    """Return the canonical name of a timezone value."""
    if isinstance(value, str):
        tz = get_timezone(value)
        return getattr(tz, "zone", None) or str(tz)
    return getattr(value, "key", None) or getattr(value, "zone", None) or str(value)


def _localize(naive, tz):
    if hasattr(tz, "localize"):
        try:
            return tz.localize(naive, is_dst=None)
        except (pytz.AmbiguousTimeError, pytz.NonExistentTimeError):
            # Noon is normally unambiguous.  For the rare zone whose offset
            # changes at noon, prefer standard time rather than failing.
            return tz.localize(naive, is_dst=False)
    return naive.replace(tzinfo=tz)


def service_day_start(service_date, timezone):
    """Return the start of a GTFS service day as an aware datetime.

    The value is noon on *service_date* in ``timezone`` minus twelve hours,
    measured as elapsed time.  On a spring-forward day this may be 23:00 on
    the previous calendar date; on a fall-back day it may be 01:00 local time.
    """

    service_date = coerce_service_date(service_date)
    tz = get_timezone(timezone)
    noon = _localize(datetime.datetime.combine(service_date, _SERVICE_DAY_START), tz)
    start_utc = noon.astimezone(pytz.utc) - datetime.timedelta(hours=12)
    return start_utc.astimezone(tz)


def service_time(service_date, offset, timezone):
    """Resolve a GTFS time offset into an aware datetime.

    ``offset`` is the original :class:`datetime.timedelta` from
    ``stop_times.txt``; it is added as elapsed time and is never reduced
    modulo twenty-four hours.
    """

    if offset is None:
        return None
    if not isinstance(offset, datetime.timedelta):
        raise TypeError("GTFS time offset must be a timedelta, got {0!r}".format(offset))
    tz = get_timezone(timezone)
    start_utc = service_day_start(service_date, tz).astimezone(pytz.utc)
    return (start_utc + offset).astimezone(tz)


def service_dates(start_date, end_date):
    """Yield every date in an inclusive service-date range."""
    start_date = coerce_service_date(start_date)
    end_date = coerce_service_date(end_date)
    if end_date < start_date:
        raise ValueError("end_date must not be before start_date")
    current = start_date
    while current <= end_date:
        yield current
        current += _ONE_DAY


def format_gtfs_time(offset):
    """Format a timedelta using GTFS's hours-from-service-day notation.

    This deliberately keeps hours greater than twenty-four: a value of
    25:10:00 remains ``"25:10:00"`` instead of becoming ``"01:10:00"`` or
    being wrapped to the next calendar day.
    """

    if offset is None:
        return None
    if not isinstance(offset, datetime.timedelta):
        raise TypeError("GTFS time offset must be a timedelta, got {0!r}".format(offset))
    total_seconds = int(offset.total_seconds())
    sign = "-" if total_seconds < 0 else ""
    total_seconds = abs(total_seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{0}{1:02d}:{2:02d}:{3:02d}".format(sign, hours, minutes, seconds)
