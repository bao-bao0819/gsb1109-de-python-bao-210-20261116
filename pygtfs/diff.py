"""Diff two GTFS feeds and describe the incremental rebuild they require.

The diff is expressed in terms of the entities exposed by
:mod:`pygtfs.gtfs_entities`.  Rows are compared using their natural GTFS keys,
not their database surrogate values, so an unchanged row keeps its database
identity when an update is applied.
"""

import contextlib
import datetime
import io
from decimal import Decimal

from .gtfs_entities import Feed, gtfs_all


TABLE_ORDER = [entity._plural_name_ for entity in gtfs_all]
CLASS_BY_TABLE = {entity._plural_name_: entity for entity in gtfs_all}

# Columns which are implementation details and never part of a GTFS row.
_INTERNAL_COLUMNS = {
    'feed_id',
    'fare_rule_internal_key',
}

# The natural key of each GTFS table.  For tables whose SQLAlchemy primary key
# contains a surrogate column (fare_rules) the whole GTFS row is the key.
KEY_FIELDS = {
    'agencies': ('agency_id',),
    'stops': ('stop_id',),
    'transfers': ('from_stop_id', 'to_stop_id', 'from_route_id',
                  'to_route_id', 'from_trip_id', 'to_trip_id'),
    'routes': ('route_id',),
    'fares': ('fare_id',),
    'fare_rules': ('fare_id', 'route_id', 'origin_id', 'destination_id',
                   'contains_id'),
    'shapes': ('shape_id', 'shape_pt_sequence'),
    'services': ('service_id',),
    'service_exceptions': ('service_id', 'date'),
    'trips': ('trip_id',),
    'frequencies': ('trip_id', 'start_time', 'end_time'),
    'stop_times': ('trip_id', 'stop_id', 'stop_sequence'),
    'feed_infos': ('feed_publisher_name', 'feed_publisher_url'),
    'translations': ('trans_id', 'lang'),
}

# Order in which rows are removed.  Children and relationship-bearing rows come
# before the entities they point at.
DELETE_ORDER = (
    'translations',
    'feed_infos',
    'stop_times',
    'frequencies',
    'transfers',
    'fare_rules',
    'trips',
    'service_exceptions',
    'services',
    'shapes',
    'fares',
    'routes',
    'stops',
    'agencies',
)

_SINGULAR = {
    'agencies': 'agency',
    'stops': 'stop',
    'transfers': 'transfer',
    'routes': 'route',
    'fares': 'fare',
    'fare_rules': 'fare_rule',
    'shapes': 'shape',
    'services': 'service',
    'service_exceptions': 'service_exception',
    'trips': 'trip',
    'frequencies': 'frequency',
    'stop_times': 'stop_time',
    'feed_infos': 'feed_info',
    'translations': 'translation',
}


def _columns(plural):
    cls = CLASS_BY_TABLE[plural]
    return [column.name for column in cls.__table__.columns
            if column.name not in _INTERNAL_COLUMNS]


def _natural_key(plural, row):
    return tuple(row.get(field) for field in KEY_FIELDS[plural])


def _key_label(plural, key):
    if len(key) == 1:
        return _text_value(key[0])
    return '|'.join('' if value is None else _text_value(value)
                    for value in key)


def _row_values_differ(plural, old_row, new_row):
    for column in _columns(plural):
        if old_row.get(column) != new_row.get(column):
            return True
    return False


def _changed_columns(plural, old_row, new_row):
    return [column for column in _columns(plural)
            if old_row.get(column) != new_row.get(column)]


def _is_reference(value):
    return value not in (None, '')


def _entity_ids(snapshot, plural):
    """First component of each natural key in ``snapshot`` (e.g. stop_ids)."""

    return {key[0][0] for key in snapshot.rows[plural]}


def _text_value(value):
    """Render a stored Python value the way GTFS text expresses it."""

    if value is None:
        return None
    if isinstance(value, datetime.timedelta):
        total = int(value.total_seconds())
        hours, remainder = divmod(total, 3600)
        minutes, seconds = divmod(remainder, 60)
        return '{0}:{1:02d}:{2:02d}'.format(hours, minutes, seconds)
    if isinstance(value, datetime.date):
        return value.strftime('%Y%m%d')
    if isinstance(value, bool):
        return '1' if value else '0'
    if isinstance(value, Decimal):
        return str(value)
    return str(value)


def _text_row(plural, row):
    return {column: _text_value(row.get(column))
            for column in _columns(plural)}


class FeedSnapshot(object):
    """A feed's GTFS rows, indexed by natural key.

    ``rows`` maps a table plural name to ``(natural_key, occurrence) -> row``.
    The occurrence component exists only to keep otherwise identical rows
    (notably duplicate fare rules) distinct.
    """

    def __init__(self, rows=None):
        self.rows = {plural: {} for plural in TABLE_ORDER}
        if rows:
            for plural, table_rows in rows.items():
                for row in table_rows:
                    self.add_row(plural, row)

    def add_row(self, plural, row):
        row = _text_row(plural, row)
        key = _natural_key(plural, row)
        occurrence = 0
        table = self.rows[plural]
        while (key, occurrence) in table:
            occurrence += 1
        table[(key, occurrence)] = row

    def table_rows(self, plural):
        return list(self.rows[plural].values())

    def row(self, plural, key, occurrence=0):
        return self.rows[plural].get((key, occurrence))

    @classmethod
    def from_schedule(cls, schedule, feed_id):
        snapshot = cls()
        for plural in TABLE_ORDER:
            entity = CLASS_BY_TABLE[plural]
            query = schedule.session.query(entity).filter(
                entity.feed_id == feed_id)
            for instance in query:
                row = {}
                for column in entity.__table__.columns:
                    if column.name in _INTERNAL_COLUMNS:
                        continue
                    row[column.name] = getattr(instance, column.name)
                snapshot.add_row(plural, row)
        return snapshot

    @classmethod
    def from_filename(cls, feed_filename, strip_fields=True, chunk_size=5000,
                      ignore_files=()):
        # Imported lazily: loader imports the diff helpers.
        from .schedule import Schedule
        from .loader import append_feed

        schedule = Schedule(':memory:')
        try:
            # A feed load is intentionally quiet here: callers asked for a
            # diff, not for loader progress output.
            with contextlib.redirect_stdout(io.StringIO()):
                append_feed(schedule, feed_filename,
                            strip_fields=strip_fields,
                            chunk_size=chunk_size,
                            ignore_files=ignore_files)
            return cls.from_schedule(schedule, schedule.feeds[0].feed_id)
        finally:
            schedule.session.close()


class FeedDiff(object):
    """The row-level difference between two :class:`FeedSnapshot` objects."""

    def __init__(self, old, new):
        self.old = old
        self.new = new
        self.added = {plural: [] for plural in TABLE_ORDER}
        self.removed = {plural: [] for plural in TABLE_ORDER}
        self.changed = {plural: [] for plural in TABLE_ORDER}
        self.unchanged = {plural: 0 for plural in TABLE_ORDER}
        self.changed_columns = {plural: {} for plural in TABLE_ORDER}
        self._compare()
        self._compute_impact()

    def _compare(self):
        for plural in TABLE_ORDER:
            old_rows = self.old.rows[plural]
            new_rows = self.new.rows[plural]

            for key, old_row in old_rows.items():
                new_row = new_rows.get(key)
                if new_row is None:
                    self.removed[plural].append(old_row)
                elif _row_values_differ(plural, old_row, new_row):
                    self.changed[plural].append((old_row, new_row))
                    natural_key = key[0]
                    self.changed_columns[plural][natural_key] = (
                        _changed_columns(plural, old_row, new_row))
                else:
                    self.unchanged[plural] += 1

            for key, new_row in new_rows.items():
                if key not in old_rows:
                    self.added[plural].append(new_row)

    def _all_affected_rows(self, plural):
        rows = list(self.added[plural])
        rows.extend(self.removed[plural])
        rows.extend(new_row for _, new_row in self.changed[plural])
        return rows

    def _compute_impact(self):
        # ``changed_keys`` names rows whose own payload differs.
        # ``affected`` additionally names entities whose derived link rows may
        # be stale.  Dependents never propagate back up: a transfer edit must
        # not mark its trip as changed, and a renamed stop does not rewrite
        # its stop_times (those reference stop_id, not stop_name).
        changed = {plural: set() for plural in TABLE_ORDER}
        affected = {plural: set() for plural in TABLE_ORDER}

        def touch(plural, key):
            if key is not None:
                affected[plural].add(key)

        for plural in TABLE_ORDER:
            for row in self._all_affected_rows(plural):
                key = _natural_key(plural, row)
                changed[plural].add(key)
                touch(plural, key)

        # Changing a shape point or a trip's shape_id regenerates
        # _trip_shapes for every trip on the shape, even when the trip row
        # itself is untouched.
        changed_shape_ids = {row.get('shape_id')
                             for row in self._all_affected_rows('shapes')}
        for old, new in self.changed['trips']:
            if old.get('shape_id') != new.get('shape_id'):
                if _is_reference(old.get('shape_id')):
                    changed_shape_ids.add(old.get('shape_id'))
                if _is_reference(new.get('shape_id')):
                    changed_shape_ids.add(new.get('shape_id'))
        for trip_row in (self.new.table_rows('trips') +
                         self.old.table_rows('trips')):
            if trip_row.get('shape_id') in changed_shape_ids:
                touch('trips', (trip_row.get('trip_id'),))

        # _stop_translations is name based.  A stop rename or a translation
        # edit regenerates the links for the matching stop names.
        changed_names = {row.get('stop_name')
                         for row in self._all_affected_rows('stops')}
        changed_names.update(row.get('trans_id')
                             for row in self._all_affected_rows(
                                 'translations'))
        for stop_row in (self.new.table_rows('stops') +
                         self.old.table_rows('stops')):
            if stop_row.get('stop_name') in changed_names:
                touch('stops', (stop_row.get('stop_id'),))

        for plural in affected:
            affected[plural] = {key for key in affected[plural]
                                if key != (None,)}
            changed[plural] = {key for key in changed[plural]
                               if key != (None,)}

        self.changed_keys = changed
        self.affected = affected
        self.rebuild = self._rebuild_tables()
        self.dependencies = self._dependencies()

    def _rebuild_tables(self):
        rebuild = []
        if self.changed_keys['stops'] or self.changed_keys['stop_times']:
            rebuild.append('stop_times')
        if any(self.changed_keys[plural] for plural in
               ('stops', 'routes', 'trips', 'transfers')):
            rebuild.append('transfers')
        if any(self.changed_keys[plural] for plural in
               ('trips', 'frequencies')):
            rebuild.append('frequencies')
        if self.changed_keys['stops'] or self.changed_keys['translations']:
            rebuild.append('_stop_translations')
        if self.changed_keys['shapes'] or self.changed_keys['trips']:
            rebuild.append('_trip_shapes')
        return rebuild

    def _dependencies(self):
        dependencies = []
        dependency_map = {
            'stops': ('stop_times', 'transfers', '_stop_translations'),
            'routes': ('trips', 'fare_rules', 'transfers'),
            'trips': ('stop_times', 'frequencies', 'transfers',
                      '_trip_shapes'),
            'shapes': ('_trip_shapes', 'trips'),
            'translations': ('_stop_translations',),
            'services': ('trips',),
            'fares': ('fare_rules',),
        }
        for plural in ('stops', 'routes', 'trips', 'shapes', 'translations',
                       'services', 'fares'):
            # Only entities that changed themselves have derived results to
            # rebuild; dependents found solely through a reference are
            # reported in ``affected`` but do not need a row rewrite.
            for key in sorted(self.changed_keys[plural],
                              key=lambda item: str(item)):
                dependencies.append({
                    'entity': _SINGULAR[plural],
                    'key': _key_label(plural, key),
                    'rebuild': list(dependency_map[plural]),
                })
        return dependencies

    def has_changes(self):
        return any(self.added[plural] or self.removed[plural] or
                   self.changed[plural] for plural in TABLE_ORDER)

    def counts(self):
        return {
            plural: {
                'added': len(self.added[plural]),
                'removed': len(self.removed[plural]),
                'modified': len(self.changed[plural]),
                'unchanged': self.unchanged[plural],
            }
            for plural in TABLE_ORDER
        }

    def _keys_for(self, plural, rows=None):
        source = self._all_affected_rows(plural) if rows is None else rows
        keys = [_natural_key(plural, row) for row in source]
        return sorted({key for key in keys if key != (None,)},
                      key=lambda item: str(item))

    def transfer_rewrites(self):
        """Transfer rows which must be deleted and reinserted.

        Returns ``(delete_keys, insert_keys)`` using transfer natural keys.
        A row is inserted when it is new, and deleted when it is gone or when
        it points at a stop/route/trip absent from the revision.  A transfer
        that remains valid is intentionally left untouched; a payload-only
        edit (for example ``transfer_type``) is updated in place.
        """

        old_rows = self.old.rows['transfers']
        new_rows = self.new.rows['transfers']

        insert_keys = {key for key in new_rows if key not in old_rows}

        # Only rows that disappear or whose referenced entity disappears.
        new_stops = _entity_ids(self.new, 'stops')
        new_routes = _entity_ids(self.new, 'routes')
        new_trips = _entity_ids(self.new, 'trips')
        delete_keys = set()
        for key, row in old_rows.items():
            if key not in new_rows:
                delete_keys.add(key)
                continue
            if (row['from_stop_id'] not in new_stops or
                    row['to_stop_id'] not in new_stops):
                delete_keys.add(key)
                continue
            if any(_is_reference(row.get(field)) and row[field] not in new_routes
                   for field in ('from_route_id', 'to_route_id')):
                delete_keys.add(key)
                continue
            if any(_is_reference(row.get(field)) and row[field] not in new_trips
                   for field in ('from_trip_id', 'to_trip_id')):
                delete_keys.add(key)

        return delete_keys, insert_keys

    def link_rewrites(self):
        """Trips/stops whose many-to-many link rows must be replaced.

        Returns a dict with ``trip_shapes`` and ``stop_translations``, each
        mapping to ``(delete_ids, insert_ids)``.  Deleting includes removed
        entities so their old link rows cannot linger; inserting only covers
        entities present in the revision.
        """

        old_trips = _entity_ids(self.old, 'trips')
        new_trips = _entity_ids(self.new, 'trips')
        trip_ids = {key[0] for key in self.changed_keys['trips']}
        trip_ids.update(key[0] for key in self.affected['trips'])

        changed_shape_ids = {row.get('shape_id')
                             for row in self._all_affected_rows('shapes')}
        for old, new in self.changed['trips']:
            if old.get('shape_id') != new.get('shape_id'):
                for shape_id in (old.get('shape_id'), new.get('shape_id')):
                    if _is_reference(shape_id):
                        changed_shape_ids.add(shape_id)
        for trip_row in (self.new.table_rows('trips') +
                         self.old.table_rows('trips')):
            if trip_row.get('shape_id') in changed_shape_ids:
                trip_ids.add(trip_row.get('trip_id'))

        changed_names = {row.get('stop_name')
                         for row in self._all_affected_rows('stops')}
        changed_names.update(row.get('trans_id')
                             for row in self._all_affected_rows(
                                 'translations'))
        stop_ids = {key[0] for key in self.changed_keys['stops']}
        stop_ids.update(key[0] for key in self.affected['stops'])
        stop_ids.update(
            stop_row.get('stop_id')
            for stop_row in (self.new.table_rows('stops') +
                             self.old.table_rows('stops'))
            if stop_row.get('stop_name') in changed_names)

        return {
            'trip_shapes': (trip_ids & old_trips, trip_ids & new_trips),
            'stop_translations': (stop_ids & _entity_ids(self.old, 'stops'),
                                  stop_ids & _entity_ids(self.new, 'stops')),
        }

    def _derived_keys(self):
        """Exact keys of derived rows which must be regenerated.

        ``stop_times`` and ``frequencies`` list inserted, removed or modified
        rows; ``transfers`` lists the relation rows deleted or inserted;
        ``_trip_shapes`` and ``_stop_translations`` name the entities whose
        link rows are removed or regenerated.
        """

        stop_times = {
            _natural_key('stop_times', row)
            for row in self._all_affected_rows('stop_times')
        }
        frequencies = {
            _natural_key('frequencies', row)
            for row in self._all_affected_rows('frequencies')
        }

        link_rewrites = self.link_rewrites()
        delete_transfers, insert_transfers = self.transfer_rewrites()
        delete_trips, insert_trips = link_rewrites['trip_shapes']
        delete_stops, insert_stops = link_rewrites['stop_translations']

        return {
            'stop_times': [
                _key_label('stop_times', key)
                for key in sorted(stop_times, key=lambda item: str(item))
            ],
            'frequencies': [
                _key_label('frequencies', key)
                for key in sorted(frequencies, key=lambda item: str(item))
            ],
            'transfers': [
                _key_label('transfers', key[0])
                for key in sorted(delete_transfers | insert_transfers,
                                  key=lambda item: str(item))
            ],
            '_trip_shapes': sorted(delete_trips | insert_trips),
            '_stop_translations': sorted(delete_stops | insert_stops),
        }

    def removed_keys(self, plural):
        return {_natural_key(plural, row) for row in self.removed[plural]}

    def report(self):
        modified = {}
        for plural in TABLE_ORDER:
            entries = []
            for old_row, new_row in self.changed[plural]:
                key = _natural_key(plural, old_row)
                entries.append({
                    'key': _key_label(plural, key),
                    'fields': self.changed_columns[plural][key],
                })
            if entries:
                modified[plural] = entries

        return {
            'changed': self.has_changes(),
            'counts': self.counts(),
            'added': {plural: [_key_label(plural, key)
                               for key in self._keys_for(plural,
                                                         self.added[plural])]
                      for plural in TABLE_ORDER if self.added[plural]},
            'removed': {plural: [_key_label(plural, key)
                                 for key in self._keys_for(
                                     plural, self.removed[plural])]
                        for plural in TABLE_ORDER if self.removed[plural]},
            'modified': modified,
            'affected': {plural: [_key_label(plural, key)
                                  for key in sorted(self.affected[plural],
                                                    key=lambda item: str(item))]
                         for plural in TABLE_ORDER
                         if self.affected[plural]},
            'rebuild': list(self.rebuild),
            'derived': self._derived_keys(),
            'dependencies': self.dependencies,
        }

    def summary(self):
        lines = []
        for plural in TABLE_ORDER:
            counts = self.counts()[plural]
            if counts['added'] or counts['removed'] or counts['modified']:
                lines.append(
                    '{0}: +{1} -{2} ~{3}'.format(
                        plural, counts['added'], counts['removed'],
                        counts['modified']))
        if self.rebuild:
            lines.append('rebuild: {0}'.format(', '.join(self.rebuild)))
        for dependency in self.dependencies:
            lines.append('  {entity} {key} -> {rebuild}'.format(
                entity=dependency['entity'],
                key=dependency['key'],
                rebuild=', '.join(dependency['rebuild'])))
        if not lines:
            return 'no changes'
        return '\n'.join(lines)


def prune_invalid_references(schedule, feed_id):
    """Delete rows whose GTFS references do not resolve within the feed.

    SQLite does not enforce every GTFS reference (transfers and stop_times
    have no database constraint on all of their columns).  A complete import
    must nevertheless agree with an incremental update, so references to
    stops, routes, trips and fares which are not present are discarded here
    as well.
    """

    from sqlalchemy import and_, or_, select

    from .gtfs_entities import (Fare, FareRule, Frequency, Route, Stop,
                                StopTime, Transfer, Trip)

    session = schedule.session

    def ids(entity, column):
        return select(getattr(entity, column)).where(entity.feed_id == feed_id)

    stop_ids = ids(Stop, 'stop_id')
    route_ids = ids(Route, 'route_id')
    trip_ids = ids(Trip, 'trip_id')
    fare_ids = ids(Fare, 'fare_id')

    (session.query(StopTime)
     .filter(StopTime.feed_id == feed_id)
     .filter(or_(StopTime.trip_id.notin_(trip_ids),
                 StopTime.stop_id.notin_(stop_ids)))
     .delete(synchronize_session=False))

    (session.query(Frequency)
     .filter(Frequency.feed_id == feed_id)
     .filter(Frequency.trip_id.notin_(trip_ids))
     .delete(synchronize_session=False))

    (session.query(FareRule)
     .filter(FareRule.feed_id == feed_id)
     .filter(or_(FareRule.fare_id.notin_(fare_ids),
                 and_(FareRule.route_id.isnot(None),
                      FareRule.route_id != '',
                      FareRule.route_id.notin_(route_ids))))
     .delete(synchronize_session=False))

    (session.query(Transfer)
     .filter(Transfer.feed_id == feed_id)
     .filter(or_(Transfer.from_stop_id.notin_(stop_ids),
                 Transfer.to_stop_id.notin_(stop_ids),
                 and_(Transfer.from_route_id != '',
                      Transfer.from_route_id.notin_(route_ids)),
                 and_(Transfer.to_route_id != '',
                      Transfer.to_route_id.notin_(route_ids)),
                 and_(Transfer.from_trip_id != '',
                      Transfer.from_trip_id.notin_(trip_ids)),
                 and_(Transfer.to_trip_id != '',
                      Transfer.to_trip_id.notin_(trip_ids))))
     .delete(synchronize_session=False))
    session.commit()


def sanitize_snapshot(snapshot):
    """Drop derived/link rows whose referenced entity is absent.

    GTFS refers to stops, routes and trips from several tables without
    declaring database foreign keys for all of them (notably transfers).  A
    revision may cancel a trip while leaving such a row behind; the sanitized
    snapshot is the set of rows a consistent import can actually keep, and it
    is what the incremental path applies.
    """

    sanitized = FeedSnapshot({
        plural: snapshot.table_rows(plural) for plural in TABLE_ORDER
    })

    stop_ids = {row['stop_id'] for row in sanitized.table_rows('stops')}
    route_ids = {row['route_id'] for row in sanitized.table_rows('routes')}
    trip_ids = {row['trip_id'] for row in sanitized.table_rows('trips')}
    fare_ids = {row['fare_id'] for row in sanitized.table_rows('fares')}

    def valid(table, row):
        if table == 'transfers':
            if row.get('from_stop_id') not in stop_ids:
                return False
            if row.get('to_stop_id') not in stop_ids:
                return False
            for field in ('from_route_id', 'to_route_id'):
                if _is_reference(row.get(field)) and row[field] not in route_ids:
                    return False
            for field in ('from_trip_id', 'to_trip_id'):
                if _is_reference(row.get(field)) and row[field] not in trip_ids:
                    return False
        elif table == 'stop_times':
            if row.get('trip_id') not in trip_ids:
                return False
            if row.get('stop_id') not in stop_ids:
                return False
        elif table == 'frequencies':
            if row.get('trip_id') not in trip_ids:
                return False
        elif table == 'fare_rules':
            if row.get('fare_id') not in fare_ids:
                return False
            if _is_reference(row.get('route_id')) and row['route_id'] not in route_ids:
                return False
        return True

    for table in ('transfers', 'stop_times', 'frequencies', 'fare_rules'):
        rows = sanitized.rows[table]
        sanitized.rows[table] = {
            key: row for key, row in rows.items() if valid(table, row)
        }
    return sanitized


def resolve_feed(schedule, feed_filename, old_feed_filename=None,
                 feed_id=None):
    """Find the feed row an update should modify, if one exists."""

    if feed_id is not None:
        feed_entry = schedule.session.query(Feed).get(feed_id)
        if feed_entry is None:
            raise ValueError('unknown feed_id: {0}'.format(feed_id))
        return feed_entry

    from .feed import derive_feed_name

    names = {derive_feed_name(feed_filename)}
    if old_feed_filename is not None:
        names.add(derive_feed_name(old_feed_filename))

    matches = (schedule.session.query(Feed)
               .filter(Feed.feed_name.in_(names))
               .order_by(Feed.feed_id.desc())
               .all())
    if matches:
        return matches[0]

    all_feeds = schedule.session.query(Feed).all()
    if len(all_feeds) == 1:
        # A revision often arrives under a new directory/zip name.  A database
        # containing a single feed is unambiguous even then.
        return all_feeds[0]
    return None


def diff_feeds(old_feed_filename, new_feed_filename, strip_fields=True,
               chunk_size=5000, ignore_files=()):
    """Compare two feed versions without touching a database."""

    old = sanitize_snapshot(FeedSnapshot.from_filename(
        old_feed_filename, strip_fields=strip_fields, chunk_size=chunk_size,
        ignore_files=ignore_files))
    new = sanitize_snapshot(FeedSnapshot.from_filename(
        new_feed_filename, strip_fields=strip_fields, chunk_size=chunk_size,
        ignore_files=ignore_files))
    return FeedDiff(old, new)


def diff_feed(schedule, feed_filename, old_feed_filename=None, feed_id=None,
              strip_fields=True, chunk_size=5000, ignore_files=()):
    """Compare a feed version with an existing feed in ``schedule``.

    If ``old_feed_filename`` is supplied it is used as the baseline; otherwise
    the stored rows of the matching feed are used.  The comparison uses the
    same sanitized revision the updater applies, so rows pointing at entities
    absent from the revision are reported as removals rather than surviving
    as dangling relations.
    """

    new = sanitize_snapshot(FeedSnapshot.from_filename(
        feed_filename, strip_fields=strip_fields, chunk_size=chunk_size,
        ignore_files=ignore_files))
    if old_feed_filename is not None:
        old = sanitize_snapshot(FeedSnapshot.from_filename(
            old_feed_filename, strip_fields=strip_fields,
            chunk_size=chunk_size, ignore_files=ignore_files))
    else:
        feed_entry = resolve_feed(schedule, feed_filename, feed_id=feed_id)
        if feed_entry is None:
            old = FeedSnapshot()
        else:
            old = sanitize_snapshot(
                FeedSnapshot.from_schedule(schedule, feed_entry.feed_id))
    return FeedDiff(old, new)
