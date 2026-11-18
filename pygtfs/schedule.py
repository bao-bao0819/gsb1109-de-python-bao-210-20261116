import csv
import datetime

import pytz
import sqlalchemy
import sqlalchemy.orm

from . import service_time
from .exceptions import PygtfsTimezoneError
from .gtfs_entities import (ActualStopTime, Feed, Base, gtfs_all)


class Schedule:
    """Represents the full database.

    The schedule is the most important object in pygtfs. It represents the
    entire dataset. Most of the properties come straight from the gtfs
    reference. Two of them were renamed: calendar is called `services`, and
    calendar_dates `service_exceptions`. One addition is the `feeds` table,
    which is here to support more than one feed in a database.

    Each of the properties is a list created upon access by sqlalchemy. Then,
    each element of the list as attributes following the gtfs reference. In
    addition, if they are related to another table, this can also be accessed
    by attribute.

    GTFS times are offsets from the start of a service day, which the GTFS
    specification defines as noon minus twelve hours in the agency timezone.
    The ``stop_times_for_*`` methods resolve those offsets to timezone-aware
    datetimes while preserving the original service date and the original
    ``HH:MM:SS`` value (which may exceed twenty-four hours).

    :param db_conection: Either a sqlalchemy database url or a filename to be used with sqlite.

    """

    def __init__(self, db_connection):
        self.db_connection = db_connection
        self.db_filename = None
        if '://' not in db_connection:
            self.db_connection = 'sqlite:///%s' % self.db_connection
        if self.db_connection.startswith('sqlite'):
            self.db_filename = self.db_connection
        self.engine = sqlalchemy.create_engine(self.db_connection)
        Session = sqlalchemy.orm.sessionmaker(bind=self.engine)
        self.session = Session()
        Base.metadata.create_all(self.engine)

    def drop_feed(self, feed_id):
        """ Delete a feed from a database by feed id"""
        # the following does not cascade unfortunatly.
        # self.session.query(Feed).filter(Feed.feed_id == feed_id).delete()
        feed = self.session.query(Feed).get(feed_id)
        self.session.delete(feed)
        self.session.commit()

    def service_dates_for_trip(self, trip, start_date=None, end_date=None):
        """Return the service dates on which a trip operates.

        ``trip`` may be a :class:`pygtfs.gtfs_entities.Trip` object or a trip
        id.  ``calendar.txt`` weekday rules and ``calendar_dates.txt``
        additions and removals are both honored.  If no bounds are supplied,
        the service's own date range is used, extended by any added exception
        dates.
        """

        trips = self._resolve_trips(trip)
        if not trips:
            raise ValueError("Unknown trip: {0!r}".format(trip))
        requested_start = (service_time.coerce_service_date(start_date)
                           if start_date is not None else None)
        requested_end = (service_time.coerce_service_date(end_date)
                         if end_date is not None else None)
        if (requested_start is not None and requested_end is not None
                and requested_end < requested_start):
            raise ValueError("end_date must not be before start_date")

        dates = set()
        for one_trip in trips:
            service = self._service_for_trip(one_trip)
            exceptions = self._exceptions_for_trip(one_trip)
            added = {day for day, exception_type in exceptions.items()
                     if exception_type == 1}
            removed = {day for day, exception_type in exceptions.items()
                       if exception_type == 2}
            if service is not None and service.start_date is not None:
                start = service.start_date
                end = service.end_date or service.start_date
            else:
                start = min(added) if added else None
                end = max(added) if added else None
            if start is None:
                continue
            if added:
                start = min(start, min(added))
                end = max(end, max(added))
            if requested_start is not None:
                start = max(start, requested_start)
            if requested_end is not None:
                end = min(end, requested_end)
            if end < start:
                continue
            for day in service_time.service_dates(start, end):
                if day in removed:
                    continue
                if day in added or self._service_runs_on(service, day):
                    dates.add(day)
        return sorted(dates)

    def stop_times_for_trip(self, trip, service_date, timezone=None):
        """Return resolved stop times for one trip on one service day.

        ``trip`` may be a :class:`pygtfs.gtfs_entities.Trip` or a trip id.
        Each result keeps ``service_date`` and the original GTFS offsets, and
        also carries timezone-aware ``arrival_datetime`` and
        ``departure_datetime`` values.  A time such as ``25:10:00`` therefore
        resolves to the following calendar date while remaining attached to
        the same service day and the same trip.
        """

        service_date = service_time.coerce_service_date(service_date)
        trips = self._resolve_trips(trip)
        if not trips:
            raise ValueError("Unknown trip: {0!r}".format(trip))
        records = []
        for one_trip in trips:
            if service_date not in self.service_dates_for_trip(
                    one_trip, service_date, service_date):
                continue
            tz = self._timezone_for_trip(one_trip, timezone)
            stop_times = sorted(one_trip.stop_times,
                                key=lambda stop_time: stop_time.stop_sequence)
            for stop_time in stop_times:
                records.append(self._actual_stop_time(
                    one_trip, stop_time, service_date, tz))
        return records

    def stop_times_for_trip_id(self, trip_id, service_date, feed_id=None,
                               timezone=None):
        """Like :meth:`stop_times_for_trip`, taking an id instead of an object."""
        trips = self._resolve_trips(trip_id, feed_id=feed_id)
        if not trips:
            raise ValueError("Unknown trip: {0!r}".format(trip_id))
        records = []
        for trip in trips:
            records.extend(self.stop_times_for_trip(trip, service_date,
                                                    timezone=timezone))
        return records

    def actual_stop_times(self, start=None, end=None, trip=None, stop_id=None,
                          service_date=None, feed_id=None, timezone=None):
        """Return resolved stop times that fall in an absolute time window.

        ``start`` and ``end`` are timezone-aware datetimes (naive values are
        interpreted in the resolved agency timezone).  The query is performed
        against the actual instants, not against the textual GTFS clock, so a
        trip whose ``25:10:00`` arrival falls after midnight is still returned
        with its original service date and trip id.

        ``trip`` may be a trip object or trip id; ``stop_id`` restricts the
        results to one stop.  Either a time window or ``service_date`` (or
        both) may be supplied.
        """

        if start is None and end is None and service_date is None:
            raise ValueError("Provide start/end or service_date")
        if start is not None and end is not None and end < start:
            raise ValueError("end must not be before start")

        trips = self._resolve_trips(trip, feed_id=feed_id) if trip is not None else None
        if trips is None:
            from .gtfs_entities import Trip

            trips_query = self.session.query(Trip)
            if feed_id is not None:
                trips_query = trips_query.filter(Trip.feed_id == feed_id)
            trips = trips_query.all()

        records = []
        dates_cache = {}
        start_local_cache = {}
        end_local_cache = {}
        for one_trip in trips:
            tz = self._timezone_for_trip(one_trip, timezone)
            stop_times = one_trip.stop_times
            if stop_id is not None:
                stop_times = [st for st in stop_times if st.stop_id == stop_id]
            if not stop_times:
                continue
            max_offset = max(
                (offset for stop_time in stop_times
                 for offset in (stop_time.arrival_time, stop_time.departure_time)
                 if offset is not None),
                default=datetime.timedelta(0))
            if service_date is not None:
                requested_date = service_time.coerce_service_date(service_date)
                first_date = last_date = requested_date
            else:
                first_date = last_date = None
                if start is not None:
                    start_local = self._as_local(start, tz)
                    start_local_cache[tz] = start_local
                    first_date = start_local.date() - datetime.timedelta(
                        days=max_offset.days + 1)
                if end is not None:
                    end_local = self._as_local(end, tz)
                    end_local_cache[tz] = end_local
                    last_date = end_local.date() + datetime.timedelta(
                        days=max_offset.days + 1)
            cache_key = (one_trip.feed_id, one_trip.trip_id,
                         first_date, last_date)
            dates = dates_cache.get(cache_key)
            if dates is None:
                dates = self.service_dates_for_trip(one_trip, first_date,
                                                    last_date)
                dates_cache[cache_key] = dates
            for day in dates:
                for stop_time in sorted(stop_times,
                                        key=lambda st: st.stop_sequence):
                    record = self._actual_stop_time(one_trip, stop_time, day, tz)
                    event_time = record.arrival_datetime or record.departure_datetime
                    if event_time is None:
                        continue
                    if start is not None:
                        start_local = start_local_cache.get(tz)
                        if start_local is None:
                            start_local = self._as_local(start, tz)
                            start_local_cache[tz] = start_local
                        if event_time < start_local:
                            continue
                    if end is not None:
                        end_local = end_local_cache.get(tz)
                        if end_local is None:
                            end_local = self._as_local(end, tz)
                            end_local_cache[tz] = end_local
                        if event_time >= end_local:
                            continue
                    records.append(record)
        records.sort(key=lambda record: (
            record.arrival_datetime or record.departure_datetime,
            record.feed_id, record.trip_id, record.stop_sequence))
        return records

    def export_actual_stop_times(self, start=None, end=None, trip=None,
                                 stop_id=None, service_date=None, feed_id=None,
                                 timezone=None, replace=True):
        """Persist resolved stop times into the ``actual_stop_times`` table.

        The arguments are the same as :meth:`actual_stop_times`.  By default
        existing rows in the selected scope are replaced, so calling this
        repeatedly does not create duplicates.  The return value is the number
        of rows written.
        """

        from .gtfs_entities import ActualStopTime

        records = self.actual_stop_times(
            start=start, end=end, trip=trip, stop_id=stop_id,
            service_date=service_date, feed_id=feed_id, timezone=timezone)
        if replace:
            scopes = {(record.feed_id, record.trip_id, record.service_date)
                      for record in records}
            for scope_feed_id, scope_trip_id, scope_date in scopes:
                (self.session.query(ActualStopTime)
                 .filter(ActualStopTime.feed_id == scope_feed_id)
                 .filter(ActualStopTime.trip_id == scope_trip_id)
                 .filter(ActualStopTime.service_date == scope_date)
                 .delete(synchronize_session=False))
        for record in records:
            self.session.add(ActualStopTime(
                feed_id=record.feed_id,
                trip_id=record.trip_id,
                service_id=record.service_id,
                service_date=record.service_date,
                stop_sequence=record.stop_sequence,
                stop_id=record.stop_id,
                arrival_time=record.arrival_time,
                departure_time=record.departure_time,
                arrival_gtfs_time=record.arrival_gtfs_time,
                departure_gtfs_time=record.departure_gtfs_time,
                arrival_utc=self._to_utc_naive(record.arrival_datetime),
                departure_utc=self._to_utc_naive(record.departure_datetime),
                arrival_local=(record.arrival_datetime.isoformat()
                               if record.arrival_datetime is not None else None),
                departure_local=(record.departure_datetime.isoformat()
                                 if record.departure_datetime is not None else None),
                timezone=record.timezone,
            ))
        self.session.commit()
        return len(records)

    @staticmethod
    def _to_utc_naive(value):
        if value is None:
            return None
        return value.astimezone(datetime.timezone.utc).replace(tzinfo=None)

    def write_actual_stop_times(self, target, **kwargs):
        """Write resolved stop times as CSV and return the number of rows.

        Keyword arguments are passed to :meth:`actual_stop_times` (for
        example ``trip=...``, ``service_date=...``, ``start=...``, ``end=...``,
        or ``stop_id=...``).  ``target`` may be a path or a file-like object.
        The CSV includes both the original ``HH:MM:SS`` GTFS values (which can
        exceed twenty-four hours) and the resolved local datetimes.
        """

        records = self.actual_stop_times(**kwargs)
        fieldnames = list(service_time.ResolvedStopTime._fields)
        close = False
        if isinstance(target, (str, bytes)) or hasattr(target, "__fspath__"):
            handle = open(target, "w", newline="", encoding="utf-8")
            close = True
        else:
            handle = target
        try:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                row = record._asdict()
                row["arrival_time"] = record.arrival_gtfs_time
                row["departure_time"] = record.departure_gtfs_time
                for key in ("arrival_datetime", "departure_datetime"):
                    value = row[key]
                    row[key] = value.isoformat() if value is not None else None
                writer.writerow(row)
        finally:
            if close:
                handle.close()
        return len(records)

    def _resolve_trips(self, trip, feed_id=None):
        from .gtfs_entities import Trip

        query = self.session.query(Trip)
        if isinstance(trip, Trip):
            trips = [trip]
        elif isinstance(trip, str):
            query = query.filter(Trip.trip_id == trip)
            if feed_id is not None:
                query = query.filter(Trip.feed_id == feed_id)
            trips = query.all()
        else:
            raise TypeError("trip must be a Trip object or trip id")
        if feed_id is not None and isinstance(trip, Trip):
            trips = [one for one in trips if one.feed_id == feed_id]
        return trips

    def _service_for_trip(self, trip):
        from .gtfs_entities import Service

        return (self.session.query(Service)
                .filter(Service.feed_id == trip.feed_id)
                .filter(Service.service_id == trip.service_id)
                .first())

    def _exceptions_for_trip(self, trip):
        from .gtfs_entities import ServiceException

        rows = (self.session.query(ServiceException)
                .filter(ServiceException.feed_id == trip.feed_id)
                .filter(ServiceException.service_id == trip.service_id)
                .all())
        return {row.date: row.exception_type for row in rows}

    @staticmethod
    def _service_runs_on(service, day):
        if service is None:
            return False
        weekdays = ('monday', 'tuesday', 'wednesday', 'thursday', 'friday',
                    'saturday', 'sunday')
        if not getattr(service, weekdays[day.weekday()]):
            return False
        start = service.start_date
        end = service.end_date or service.start_date
        if start is not None and day < start:
            return False
        if end is not None and day > end:
            return False
        return True

    def _timezone_for_trip(self, trip, timezone=None):
        if timezone is not None:
            return service_time.get_timezone(timezone)
        route = trip.route
        if route is not None and route.agency is not None:
            name = route.agency.agency_timezone
            if name:
                return service_time.get_timezone(name)
        from .gtfs_entities import Agency

        agency = (self.session.query(Agency)
                  .filter(Agency.feed_id == trip.feed_id)
                  .first())
        if agency is not None and agency.agency_timezone:
            return service_time.get_timezone(agency.agency_timezone)
        raise PygtfsTimezoneError(
            "No agency timezone found for trip {0!r}".format(trip.trip_id))

    @staticmethod
    def _as_local(value, tz):
        if isinstance(value, str):
            value = datetime.datetime.fromisoformat(value)
        if not isinstance(value, datetime.datetime):
            raise TypeError("expected a datetime, got {0!r}".format(value))
        if value.tzinfo is None:
            if hasattr(tz, "localize"):
                try:
                    return tz.localize(value, is_dst=None)
                except pytz.AmbiguousTimeError:
                    # During the repeated fall-back hour, treat a naive
                    # timestamp as standard time.
                    return tz.localize(value, is_dst=False)
                except pytz.NonExistentTimeError:
                    return tz.localize(value, is_dst=False)
            return value.replace(tzinfo=tz)
        return value.astimezone(tz)

    @staticmethod
    def _actual_stop_time(trip, stop_time, service_date, tz):
        arrival = service_time.service_time(service_date, stop_time.arrival_time, tz)
        departure = service_time.service_time(service_date, stop_time.departure_time, tz)
        return service_time.ResolvedStopTime(
            feed_id=trip.feed_id,
            trip_id=trip.trip_id,
            service_id=trip.service_id,
            service_date=service_date,
            stop_sequence=stop_time.stop_sequence,
            stop_id=stop_time.stop_id,
            arrival_time=stop_time.arrival_time,
            departure_time=stop_time.departure_time,
            arrival_gtfs_time=service_time.format_gtfs_time(stop_time.arrival_time),
            departure_gtfs_time=service_time.format_gtfs_time(stop_time.departure_time),
            arrival_datetime=arrival,
            departure_datetime=departure,
            timezone=service_time.timezone_name(tz),
        )


def _meta_query_all(entity, docstring=None):
    def _query_all(instance_self):
        """ A list generated on access """
        return instance_self.session.query(entity).all()

    if docstring is not None:
        _query_all.__doc__ = docstring
    return property(_query_all)


def _meta_query_by_id(entity, docstring=None):
    def _query_by_id(self, id):
        """ A function that returns a list of entries with matching ids """
        return self.session.query(entity).filter(entity.id == id).all()
    if docstring is not None:
        _query_by_id.__doc__ = docstring
    return _query_by_id


def _meta_query_raw(entity, docstring=None):
    def _query_raw(instance_self):
        """
            A raw sqlalchemy query object that the user can then manipulate
            manually
        """
        return instance_self.session.query(entity)

    if docstring is not None:
        _query_raw.__doc__ = docstring
    return property(_query_raw)


for entity in (gtfs_all + [Feed, ActualStopTime]):
    entity_doc = "A list of :py:class:`pygtfs.gtfs_entities.{0}` objects".format(entity.__name__)
    entity_raw_doc = ("A :py:class:`sqlalchemy.orm.Query` object to fetch "
                      ":py:class:`pygtfs.gtfs_entities.{0}` objects"
                      .format(entity.__name__))
    entity_by_id_doc = "A list of :py:class:`pygtfs.gtfs_entities.{0}` objects with matching id".format(entity.__name__)
    setattr(Schedule, entity._plural_name_, _meta_query_all(entity, entity_doc))
    setattr(Schedule, entity._plural_name_ + "_query",
            _meta_query_raw(entity, entity_raw_doc))
    if hasattr(entity, 'id'):
        setattr(Schedule, entity._plural_name_ + "_by_id", _meta_query_by_id(entity, entity_by_id_doc))

