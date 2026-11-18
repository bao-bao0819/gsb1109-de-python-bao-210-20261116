"""Helpers for interpreting GTFS ``frequencies.txt`` rows.

A frequency row describes service between ``start_time`` and ``end_time``
with a fixed ``headway_secs``.  The two kinds of service are deliberately
kept apart:

``exact_times == 1``
    The row is schedule based.  The departures at the first stop are known
    exactly and are returned as :class:`DepartureInstance` objects.

``exact_times == 0`` (also the GTFS default when the field is empty)
    The row is frequency based.  The feed does not promise individual
    departure times, so this module only returns a :class:`ServiceWindow`
    describing the period and the range of possible waiting times.  It does
    not invent departures for these rows.

Frequency periods use a half-open interval, ``[start_time, end_time)``.  This
matches the GTFS meaning of ``end_time`` (the time at which the service
changes or ceases) and means two periods that merely touch at a boundary do
not overlap.  Pass ``end_inclusive=True`` to :func:`expand_frequencies` to
include the end instant for callers that need inclusive boundaries.

Rows for the same trip are checked for overlaps.  Overlaps are reported as
:class:`FrequencyConflict` objects; they can also be raised as an exception
by passing ``raise_on_conflict=True``.
"""

import datetime
from collections import defaultdict
from dataclasses import dataclass, field

from sqlalchemy.orm import object_session

from .exceptions import (FrequencyConflictError, FrequencyError,
                         InvalidFrequencyError, NonExactFrequencyError)

_WEEKDAYS = ('monday', 'tuesday', 'wednesday', 'thursday', 'friday',
             'saturday', 'sunday')


def parse_time(value):
    """Return *value* as a :class:`datetime.timedelta`.

    GTFS times are normally represented by pygtfs as timedeltas already.
    Strings (``"HH:MM:SS"``, including hours greater than 24), times and
    numeric second counts are accepted as a convenience.
    """
    if value is None:
        return None
    if isinstance(value, datetime.timedelta):
        return value
    if isinstance(value, datetime.datetime):
        return datetime.timedelta(hours=value.hour, minutes=value.minute,
                                  seconds=value.second,
                                  microseconds=value.microsecond)
    if isinstance(value, datetime.time):
        return datetime.timedelta(hours=value.hour, minutes=value.minute,
                                  seconds=value.second,
                                  microseconds=value.microsecond)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        sign = -1 if text.startswith('-') else 1
        if text[0] in '+-':
            text = text[1:]
        parts = text.split(':')
        if len(parts) not in (2, 3):
            raise ValueError("Invalid GTFS time: {0!r}".format(value))
        try:
            numbers = [int(part) for part in parts]
        except ValueError:
            raise ValueError("Invalid GTFS time: {0!r}".format(value))
        hours, minutes = numbers[0], numbers[1]
        seconds = numbers[2] if len(numbers) == 3 else 0
        return sign * datetime.timedelta(hours=hours, minutes=minutes,
                                         seconds=seconds)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return datetime.timedelta(seconds=value)
    raise TypeError("Cannot interpret {0!r} as a GTFS time".format(value))


def format_time(value):
    """Format a timedelta as ``HH:MM:SS``, preserving hours >= 24."""
    if value is None:
        return None
    value = parse_time(value)
    total = int(value.total_seconds())
    sign = '-' if total < 0 else ''
    total = abs(total)
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{0}{1:02d}:{2:02d}:{3:02d}'.format(sign, hours, minutes, seconds)


def _as_date(value):
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    if isinstance(value, str):
        text = value.strip()
        for fmt in ('%Y%m%d', '%Y-%m-%d'):
            try:
                return datetime.datetime.strptime(text, fmt).date()
            except ValueError:
                pass
        raise ValueError("Invalid service date: {0!r}".format(value))
    raise TypeError("Cannot interpret {0!r} as a date".format(value))


def _exact_value(frequency):
    value = getattr(frequency, 'exact_times', None)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return 0
        try:
            value = int(value)
        except ValueError:
            return 0
    if isinstance(value, float):
        value = int(value)
    return 1 if value in (1, True) else 0


def _seconds(value, field_name='headway_secs'):
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise InvalidFrequencyError(
            "{0} must be a positive integer, was {1!r}".format(field_name,
                                                               value))
    if result <= 0:
        raise InvalidFrequencyError(
            "{0} must be a positive integer, was {1!r}".format(field_name,
                                                               value))
    return result


def _check_range(frequency):
    start = parse_time(getattr(frequency, 'start_time', None))
    end = parse_time(getattr(frequency, 'end_time', None))
    if start is None or end is None:
        raise InvalidFrequencyError(
            "Frequency for trip {0!r} has no start_time or end_time".format(
                getattr(frequency, 'trip_id', None)))
    if end < start:
        raise InvalidFrequencyError(
            "Frequency for trip {0!r} ends before it starts ({1} - {2})".format(
                getattr(frequency, 'trip_id', None), format_time(start),
                format_time(end)))
    return start, end


@dataclass(frozen=True)
class StopTimeInstance:
    """A stop time shifted to a concrete departure of a frequency row."""

    trip_id: str
    stop_id: str
    stop_sequence: int
    arrival_time: datetime.timedelta
    departure_time: datetime.timedelta
    source: object = field(default=None, repr=False, compare=False)

    @property
    def stop(self):
        return getattr(self.source, 'stop', None)

    def to_dict(self, format_times=True):
        converter = format_time if format_times else (lambda value: value)
        return {
            'trip_id': self.trip_id,
            'stop_id': self.stop_id,
            'stop_sequence': self.stop_sequence,
            'arrival_time': converter(self.arrival_time),
            'departure_time': converter(self.departure_time),
        }


@dataclass(frozen=True)
class DepartureInstance:
    """A departure known exactly because ``exact_times`` is 1.

    ``departure_time`` is the departure from the first stop of the trip.
    The originating frequency row is available as ``source`` for callers
    that need the original timetable details.  :attr:`stop_times` shifts the
    trip's stop times so that the first stop departs at this instance's
    ``departure_time``; it is empty when stop times are not available.
    """

    trip_id: str
    departure_time: datetime.timedelta
    start_time: datetime.timedelta = None
    end_time: datetime.timedelta = None
    headway_secs: int = None
    feed_id: int = None
    source: object = field(default=None, repr=False, compare=False)

    @property
    def time(self):
        """Alias for ``departure_time``."""
        return self.departure_time

    @property
    def departure(self):
        return self.departure_time

    @property
    def scheduled_time(self):
        return self.departure_time

    @property
    def arrival_time(self):
        """The first stop's departure is the best available arrival value."""
        return self.departure_time

    @property
    def exact_times(self):
        return 1

    @property
    def is_exact(self):
        return True

    @property
    def is_deterministic(self):
        return True

    @property
    def frequency(self):
        """Alias for ``source``."""
        return self.source

    @property
    def trip(self):
        return getattr(self.source, 'trip', None)

    @property
    def stop_times(self):
        """The trip's stop times shifted to this departure.

        The offset is relative to the first stop's scheduled departure in
        ``stop_times.txt``.  This is a derived view: the feed itself only
        promised the first-stop departure for an ``exact_times=1`` row.
        """
        trip = self.trip
        if trip is None:
            return []
        rows = sorted(getattr(trip, 'stop_times', ()) or (),
                      key=lambda row: row.stop_sequence)
        if not rows:
            return []
        first = rows[0]
        first_time = first.departure_time
        if first_time is None:
            first_time = first.arrival_time
        if first_time is None:
            first_time = datetime.timedelta(0)
        offset = self.departure_time - first_time
        instances = []
        for row in rows:
            arrival = (row.arrival_time + offset
                       if row.arrival_time is not None else None)
            departure = (row.departure_time + offset
                         if row.departure_time is not None else None)
            instances.append(StopTimeInstance(
                trip_id=self.trip_id,
                stop_id=row.stop_id,
                stop_sequence=row.stop_sequence,
                arrival_time=arrival,
                departure_time=departure,
                source=row))
        return instances

    def to_dict(self, format_times=True):
        converter = format_time if format_times else (lambda value: value)
        return {
            'trip_id': self.trip_id,
            'departure_time': converter(self.departure_time),
            'start_time': converter(self.start_time),
            'end_time': converter(self.end_time),
            'headway_secs': self.headway_secs,
            'exact_times': 1,
            'feed_id': self.feed_id,
        }

    as_dict = to_dict

    def __str__(self):
        return '<DepartureInstance {0} at {1}>'.format(
            self.trip_id, format_time(self.departure_time))


@dataclass(frozen=True)
class ServiceWindow:
    """A frequency-based period; individual departures are not promised.

    ``waiting_range`` is the range of possible waits for a passenger arriving
    during the window: zero up to one headway.  No exact departure times are
    implied by this object.
    """

    trip_id: str
    start_time: datetime.timedelta
    end_time: datetime.timedelta
    headway_secs: int
    feed_id: int = None
    source: object = field(default=None, repr=False, compare=False)

    @property
    def frequency(self):
        return self.source

    @property
    def start(self):
        return self.start_time

    @property
    def end(self):
        return self.end_time

    @property
    def headway(self):
        return self.headway_secs

    @property
    def wait_min(self):
        return self.min_wait

    @property
    def wait_max(self):
        return self.max_wait

    @property
    def exact_times(self):
        return 0

    @property
    def is_exact(self):
        return False

    @property
    def is_frequency_based(self):
        return True

    @property
    def min_wait(self):
        return datetime.timedelta(0)

    @property
    def max_wait(self):
        return datetime.timedelta(seconds=self.headway_secs)

    @property
    def waiting_range(self):
        return (self.min_wait, self.max_wait)

    @property
    def duration(self):
        return self.end_time - self.start_time

    @property
    def earliest_departure(self):
        """The window starts when service starts, not a guaranteed departure."""
        return self.start_time

    @property
    def latest_departure(self):
        """The window end is a boundary, not a guaranteed departure."""
        return self.end_time

    def to_dict(self, format_times=True):
        converter = format_time if format_times else (lambda value: value)
        return {
            'trip_id': self.trip_id,
            'start_time': converter(self.start_time),
            'end_time': converter(self.end_time),
            'headway_secs': self.headway_secs,
            'min_wait_secs': int(self.min_wait.total_seconds()),
            'max_wait_secs': int(self.max_wait.total_seconds()),
            'waiting_range': [converter(self.min_wait),
                              converter(self.max_wait)],
            'exact_times': 0,
            'feed_id': self.feed_id,
        }

    as_dict = to_dict

    def __str__(self):
        return '<ServiceWindow {0} {1}-{2} every {3}s>'.format(
            self.trip_id, format_time(self.start_time),
            format_time(self.end_time), self.headway_secs)


@dataclass(frozen=True)
class FrequencyConflict:
    """Two frequency periods for the same trip overlap."""

    trip_id: str
    earlier: object
    later: object
    feed_id: int = None

    @property
    def first(self):
        return self.earlier

    @property
    def second(self):
        return self.later

    @property
    def frequency_a(self):
        return self.earlier

    @property
    def frequency_b(self):
        return self.later

    @property
    def start_time(self):
        return max(parse_time(self.earlier.start_time),
                   parse_time(self.later.start_time))

    @property
    def end_time(self):
        return min(parse_time(self.earlier.end_time),
                   parse_time(self.later.end_time))

    @property
    def overlap_start(self):
        return self.start_time

    @property
    def overlap_end(self):
        return self.end_time

    @property
    def overlap(self):
        return self.end_time - self.start_time

    @property
    def duration(self):
        return self.overlap

    def to_dict(self, format_times=True):
        converter = format_time if format_times else (lambda value: value)
        return {
            'trip_id': self.trip_id,
            'feed_id': self.feed_id,
            'overlap_start': converter(self.overlap_start),
            'overlap_end': converter(self.overlap_end),
            'overlap_secs': int(self.overlap.total_seconds()),
            'first': _frequency_to_dict(self.earlier, converter),
            'second': _frequency_to_dict(self.later, converter),
        }

    as_dict = to_dict

    def __str__(self):
        return ('<FrequencyConflict {0}: {1}-{2} overlaps {3}-{4}>'
                .format(self.trip_id,
                        format_time(parse_time(self.earlier.start_time)),
                        format_time(parse_time(self.earlier.end_time)),
                        format_time(parse_time(self.later.start_time)),
                        format_time(parse_time(self.later.end_time))))


def _frequency_to_dict(frequency, converter):
    return {
        'trip_id': getattr(frequency, 'trip_id', None),
        'start_time': converter(parse_time(getattr(frequency, 'start_time',
                                                  None))),
        'end_time': converter(parse_time(getattr(frequency, 'end_time', None))),
        'headway_secs': getattr(frequency, 'headway_secs', None),
        'exact_times': _exact_value(frequency),
    }


class FrequencyExpansion:
    """The result of expanding a collection of frequency rows.

    ``departures`` contains only instances backed by ``exact_times=1``.
    ``windows`` contains frequency-based service and deliberately has no
    departure list.  ``conflicts`` reports overlapping periods.
    """

    def __init__(self, departures=None, windows=None, conflicts=None,
                 frequencies=None):
        self.departures = list(departures or ())
        self.windows = list(windows or ())
        self.conflicts = list(conflicts or ())
        self.frequencies = list(frequencies or ())

    @property
    def instances(self):
        return self.departures

    @property
    def departure_instances(self):
        return self.departures

    @property
    def exact_departures(self):
        return self.departures

    @property
    def schedule_based_departures(self):
        return self.departures

    @property
    def service_windows(self):
        return self.windows

    @property
    def waiting_windows(self):
        return self.windows

    @property
    def frequency_windows(self):
        return self.windows

    @property
    def has_conflicts(self):
        return bool(self.conflicts)

    def raise_for_conflicts(self):
        if self.conflicts:
            raise FrequencyConflictError(self.conflicts)

    def to_dict(self, format_times=True):
        converter = format_time if format_times else (lambda value: value)
        return {
            'departures': [item.to_dict(format_times) for item in self.departures],
            'service_windows': [item.to_dict(format_times)
                                for item in self.windows],
            'conflicts': [item.to_dict(format_times) for item in self.conflicts],
        }

    as_dict = to_dict

    def export(self, format_times=True):
        return self.to_dict(format_times=format_times)

    def __iter__(self):
        for item in self.departures:
            yield item
        for item in self.windows:
            yield item

    def __len__(self):
        return len(self.departures) + len(self.windows)

    def __bool__(self):
        return bool(self.departures or self.windows or self.conflicts)

    def __getitem__(self, key):
        if isinstance(key, str):
            return getattr(self, key)
        return list(self)[key]

    def __repr__(self):
        return ('<FrequencyExpansion: {0} departure(s), {1} window(s), '
                '{2} conflict(s)>'.format(len(self.departures),
                                          len(self.windows),
                                          len(self.conflicts)))


def _looks_like_frequency(value):
    return (not isinstance(value, (str, bytes, list, tuple, set, dict))
            and hasattr(value, 'trip_id')
            and hasattr(value, 'start_time')
            and hasattr(value, 'end_time')
            and hasattr(value, 'headway_secs'))


def _rows_from_source(source, trip_id=None, feed_id=None):
    """Normalise a Schedule, Trip, Frequency or iterable into frequency rows."""
    if source is None:
        raise TypeError("A Schedule, Trip, Frequency or iterable is required")

    # Import lazily to keep pygtfs.gtfs_entities' import graph simple.
    from .gtfs_entities import Frequency
    from .schedule import Schedule

    if isinstance(source, str):
        trip_id = source if trip_id is None else trip_id
        rows = []
    elif isinstance(source, Schedule):
        rows = list(source.frequencies)
    elif isinstance(source, Frequency) or _looks_like_frequency(source):
        rows = [source]
    elif hasattr(source, 'frequencies') and not isinstance(source, (list, tuple)):
        # A Trip (or any object exposing its frequency rows).
        rows = list(source.frequencies)
    else:
        try:
            rows = list(source)
        except TypeError:
            raise TypeError(
                "Expected a Schedule, Trip, Frequency or iterable, got {0!r}"
                .format(type(source).__name__))

    trip_ids = _as_filter_set(trip_id)
    feed_ids = _as_filter_set(feed_id)
    if trip_ids is not None:
        rows = [row for row in rows if row.trip_id in trip_ids]
    if feed_ids is not None:
        rows = [row for row in rows if getattr(row, 'feed_id', None) in feed_ids]
    return rows


def _as_filter_set(value):
    if value is None:
        return None
    if isinstance(value, str):
        return {value}
    try:
        return set(value)
    except TypeError:
        return {value}


def _grouped(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(getattr(row, 'feed_id', None), row.trip_id)].append(row)
    for key in sorted(grouped, key=lambda item: (item[0] is None, item[0],
                                                  item[1])):
        grouped[key].sort(key=lambda row: (parse_time(row.start_time),
                                           parse_time(row.end_time),
                                           _exact_value(row)))
    return grouped


def _service_is_active(row, service_date):
    """Return whether the row's trip is active on service_date.

    If no service information is available the row is kept; the caller asked
    for a date but the feed did not supply enough information to exclude it.
    """
    if service_date is None:
        return True
    service_date = _as_date(service_date)
    trip = getattr(row, 'trip', None)
    if trip is None:
        return True
    service = getattr(trip, 'service', None)

    active = False
    service_id = getattr(trip, 'service_id', None)
    if service_id is None:
        return True
    if service is not None:
        day_name = _WEEKDAYS[service_date.weekday()]
        in_range = True
        start = getattr(service, 'start_date', None)
        end = getattr(service, 'end_date', None)
        if start is not None and service_date < start:
            in_range = False
        if end is not None and service_date > end:
            in_range = False
        active = bool(getattr(service, day_name, False)) and in_range

    session = object_session(row)
    if session is None:
        return active
    from .gtfs_entities import ServiceException
    exceptions = (session.query(ServiceException)
                  .filter(ServiceException.feed_id == row.feed_id)
                  .filter(ServiceException.service_id == service_id)
                  .filter(ServiceException.date == service_date)
                  .all())
    for exception in exceptions:
        if exception.exception_type == 1:
            active = True
        elif exception.exception_type == 2:
            active = False
    return active


def _departures_for(row, end_inclusive):
    start, end = _check_range(row)
    headway = _seconds(getattr(row, 'headway_secs', None))
    departures = []
    current = start
    step = datetime.timedelta(seconds=headway)
    while current < end or (end_inclusive and current == end):
        departures.append(DepartureInstance(
            trip_id=row.trip_id,
            departure_time=current,
            start_time=start,
            end_time=end,
            headway_secs=headway,
            feed_id=getattr(row, 'feed_id', None),
            source=row))
        current += step
    return departures


def _window_for(row):
    start, end = _check_range(row)
    headway = _seconds(getattr(row, 'headway_secs', None))
    return ServiceWindow(
        trip_id=row.trip_id,
        start_time=start,
        end_time=end,
        headway_secs=headway,
        feed_id=getattr(row, 'feed_id', None),
        source=row)


def detect_conflicts(source, trip_id=None, feed_id=None):
    """Return overlapping frequency rows for the same trip.

    Periods are half-open.  A row ending exactly when the next one starts is
    adjacent, not overlapping, and is not reported.
    """
    rows = _rows_from_source(source, trip_id=trip_id, feed_id=feed_id)
    conflicts = []
    for (row_feed_id, row_trip_id), group in _grouped(rows).items():
        for index, earlier in enumerate(group):
            earlier_start, earlier_end = _check_range(earlier)
            for later in group[index + 1:]:
                later_start, later_end = _check_range(later)
                if later_start >= earlier_end:
                    # Sorted by start time; no later row can overlap this one.
                    break
                if later_end == later_start:
                    # An empty period promises no service and cannot overlap.
                    continue
                if later_end > earlier_start:
                    conflicts.append(FrequencyConflict(
                        trip_id=row_trip_id,
                        earlier=earlier,
                        later=later,
                        feed_id=row_feed_id))
    return conflicts


def expand_frequencies(source, trip_id=None, feed_id=None, service_date=None,
                       end_inclusive=False, raise_on_conflict=False,
                       strict=False):
    """Expand frequency rows without inventing non-exact departures.

    :param source: A :class:`~pygtfs.Schedule`, a trip, a frequency row or an
        iterable of frequency rows.
    :param trip_id: Optional trip id (or iterable of ids) to select.
    :param feed_id: Optional feed id (or iterable of ids) to select.
    :param service_date: Optionally restrict to trips active on this date.
    :param end_inclusive: If true, include a departure exactly at ``end_time``.
        By default the period is ``[start_time, end_time)``.
    :param raise_on_conflict: Raise :class:`FrequencyConflictError` if any
        periods overlap.
    :param strict: Alias for ``raise_on_conflict``.
    :returns: :class:`FrequencyExpansion`.
    """
    rows = _rows_from_source(source, trip_id=trip_id, feed_id=feed_id)
    if service_date is not None:
        rows = [row for row in rows if _service_is_active(row, service_date)]

    conflicts = detect_conflicts(rows)
    if (raise_on_conflict or strict) and conflicts:
        raise FrequencyConflictError(conflicts)

    departures = []
    windows = []
    for row in rows:
        exact = _exact_value(row)
        if exact:
            departures.extend(_departures_for(row, end_inclusive))
        else:
            windows.append(_window_for(row))
    return FrequencyExpansion(departures=departures, windows=windows,
                              conflicts=conflicts, frequencies=rows)


def frequency_instances(source, **kwargs):
    """Return only exactly-known departures (``exact_times=1``)."""
    return expand_frequencies(source, **kwargs).departures


frequency_departures = frequency_instances


def departure_instances(source, **kwargs):
    return frequency_instances(source, **kwargs)


def deterministic_departures(source, **kwargs):
    return frequency_instances(source, **kwargs)


def service_windows(source, **kwargs):
    """Return only frequency-based service windows (``exact_times=0``)."""
    return expand_frequencies(source, **kwargs).windows


def waiting_windows(source, **kwargs):
    return service_windows(source, **kwargs)


def find_conflicts(source, trip_id=None, feed_id=None):
    return detect_conflicts(source, trip_id=trip_id, feed_id=feed_id)


# Descriptive aliases for callers that prefer an "export" spelling.
export_frequencies = expand_frequencies
resolve_frequencies = expand_frequencies
frequency_expansion = expand_frequencies
detect_frequency_conflicts = detect_conflicts
frequency_conflicts = detect_conflicts


__all__ = [
    'DepartureInstance',
    'ServiceWindow',
    'FrequencyConflict',
    'FrequencyExpansion',
    'expand_frequencies',
    'frequency_instances',
    'frequency_departures',
    'departure_instances',
    'deterministic_departures',
    'service_windows',
    'waiting_windows',
    'detect_conflicts',
    'find_conflicts',
    'frequency_conflicts',
    'detect_frequency_conflicts',
    'export_frequencies',
    'resolve_frequencies',
    'frequency_expansion',
    'parse_time',
    'format_time',
    'FrequencyError',
    'FrequencyConflictError',
    'NonExactFrequencyError',
    'InvalidFrequencyError',
]
