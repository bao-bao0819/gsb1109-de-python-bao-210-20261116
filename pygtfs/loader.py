from datetime import date
import logging
import sys

from sqlalchemy import and_, or_
from sqlalchemy.sql.expression import select

from .gtfs_entities import (Feed, Service, ServiceException, gtfs_required,
                            Translation, Stop, Trip, ShapePoint, Transfer,
                            _stop_translations, _trip_shapes, gtfs_calendar,
                            gtfs_all)
from . import feed

logger = logging.getLogger(__name__)


def list_feeds(schedule):
    for (i, a_feed) in enumerate(schedule.feeds):
        print("{0}. id {1.feed_id}, name {1.feed_name}, "
              "loaded on {1.feed_append_date}".format(i, a_feed))


def delete_feed(schedule, feed_filename, interactive=False):

    feed_name = feed.derive_feed_name(feed_filename)
    feeds_with_name = schedule.session.query(Feed).filter(Feed.feed_name == feed_name).all()
    delete_all = not interactive
    for matching_feed in feeds_with_name:
        if not delete_all:
            print("Found feed ({0.feed_id}) named {0.feed_name} loaded on {0.feed_append_date}".format(matching_feed))
            logger.info("Found feed ({0.feed_id}) named {0.feed_name} loaded on {0.feed_append_date}".format(matching_feed))
            ans = ""
            while ans not in ("K", "O", "A"):
                ans = input("(K)eep / (O)verwrite / overwrite (A)ll ? ").upper()
            if ans == "K":
                continue
            if ans == "A":
                delete_all = True
        # you get here if ans is A or O, and if delete_all
        schedule.drop_feed(matching_feed.feed_id)


def overwrite_feed(schedule, feed_filename, *args, **kwargs):
    interactive = kwargs.pop('interactive', False)
    delete_feed(schedule, feed_filename, interactive=interactive)
    append_feed(schedule, feed_filename, *args, **kwargs)


def update_feed(schedule, feed_filename, old_feed_filename=None, feed_id=None,
                strip_fields=True, chunk_size=5000, ignore_files=()):
    """Apply a revised feed to an existing feed in ``schedule``.

    Only rows which actually differ are touched: removed rows are deleted,
    modified rows are updated in place (keeping their database identity),
    unchanged rows are left alone, and new rows are inserted.  The derived
    transfer, ``_trip_shapes`` and ``_stop_translations`` rows affected by the
    revision are rebuilt from the new input.

    A transfer which points at a trip that the revision cancelled is dropped;
    it cannot survive as an entry for a trip that is no longer present.
    """

    from . import diff as feed_diff

    feed_entry = feed_diff.resolve_feed(
        schedule, feed_filename, old_feed_filename=old_feed_filename,
        feed_id=feed_id)
    if feed_entry is None:
        raise ValueError('no existing feed matches {0!r}'.format(feed_filename))

    old_snapshot = feed_diff.FeedSnapshot.from_schedule(
        schedule, feed_entry.feed_id)
    new_snapshot = feed_diff.sanitize_snapshot(
        feed_diff.FeedSnapshot.from_filename(
            feed_filename, strip_fields=strip_fields, chunk_size=chunk_size,
            ignore_files=ignore_files))

    changes = feed_diff.FeedDiff(old_snapshot, new_snapshot)
    _apply_feed_diff(schedule, feed_entry, changes, new_snapshot)
    return changes


def diff_feed(schedule, feed_filename, old_feed_filename=None, feed_id=None,
              strip_fields=True, chunk_size=5000, ignore_files=()):
    """Return the incremental rebuild required by a revised feed."""

    from . import diff as feed_diff

    return feed_diff.diff_feed(
        schedule, feed_filename, old_feed_filename=old_feed_filename,
        feed_id=feed_id, strip_fields=strip_fields, chunk_size=chunk_size,
        ignore_files=ignore_files)


def _set_row_values(instance, plural, row):
    from . import diff as feed_diff
    for column in feed_diff._columns(plural):
        setattr(instance, column, row.get(column))


def _delete_removed_rows(session, entity, feed_id, plural, rows):
    from . import diff as feed_diff

    if not rows:
        return
    key_columns = [column for column in entity.__table__.columns
                   if column.primary_key and column.name != 'feed_id']
    conditions = []
    for row in rows:
        key = feed_diff._natural_key(plural, row)
        if len(key_columns) != len(key):
            continue
        conditions.append(and_(*[
            getattr(entity, column.name) == value
            for column, value in zip(key_columns, key)
        ]))
    if conditions:
        (session.query(entity)
         .filter(entity.feed_id == feed_id)
         .filter(or_(*conditions))
         .delete(synchronize_session=False))
        return

    # Tables keyed by a surrogate column (fare_rules) are matched on their
    # full GTFS payload instead.
    for row in rows:
        query = session.query(entity).filter(entity.feed_id == feed_id)
        for column in feed_diff._columns(plural):
            query = query.filter(getattr(entity, column) == row.get(column))
        instance = query.first()
        if instance is not None:
            session.delete(instance)


def _apply_feed_diff(schedule, feed_entry, changes, new_snapshot):
    from . import diff as feed_diff
    from .gtfs_entities import gtfs_all

    CLASS_BY_TABLE = {entity._plural_name_: entity for entity in gtfs_all}

    feed_id = feed_entry.feed_id
    session = schedule.session

    link_rewrites = changes.link_rewrites()
    delete_trips, insert_trips = link_rewrites['trip_shapes']
    delete_stops, insert_stops = link_rewrites['stop_translations']
    delete_transfers, insert_transfers = changes.transfer_rewrites()

    # 1. Remove derived rows first.  This keeps the order valid for databases
    #    which enforce the link tables' foreign keys, and prevents a removed
    #    stop/trip/shape from leaving link rows behind.
    if delete_trips:
        session.execute(
            _trip_shapes.delete().where(and_(
                _trip_shapes.c.trip_feed_id == feed_id,
                _trip_shapes.c.trip_id.in_(delete_trips))))
    if delete_stops:
        session.execute(
            _stop_translations.delete().where(and_(
                _stop_translations.c.stop_feed_id == feed_id,
                _stop_translations.c.stop_id.in_(delete_stops))))
    if delete_transfers:
        _delete_transfer_keys(session, feed_id, delete_transfers)
    session.flush()

    # 2. Remove source rows which no longer exist in the revision.  This
    #    happens before inserts so unique keys can be reused.
    for plural in feed_diff.DELETE_ORDER:
        entity = CLASS_BY_TABLE[plural]
        _delete_removed_rows(session, entity, feed_id, plural,
                             changes.removed[plural])
        session.flush()

    # 3. Update modified rows in place.
    for plural in feed_diff.TABLE_ORDER:
        entity = CLASS_BY_TABLE[plural]
        key_columns = [column for column in entity.__table__.columns
                       if column.primary_key and column.name != 'feed_id']
        for old_row, new_row in changes.changed[plural]:
            key = feed_diff._natural_key(plural, old_row)
            if len(key_columns) != len(key):
                continue
            instance = (session.query(entity)
                        .filter(entity.feed_id == feed_id)
                        .filter(and_(*[getattr(entity, column.name) == value
                                       for column, value in zip(key_columns, key)]))
                        .first())
            if instance is None:
                continue
            _set_row_values(instance, plural, new_row)
        session.flush()

    # 4. Insert added source rows.  Transfers are handled with the derived
    #    relations below.
    for plural in feed_diff.TABLE_ORDER:
        if plural == 'transfers':
            continue
        entity = CLASS_BY_TABLE[plural]
        for row in changes.added[plural]:
            instance = entity(feed_id=feed_id, **dict(row))
            if plural == 'fare_rules':
                instance.fare_rule_internal_key = None
            session.add(instance)
        session.flush()

    # 5. Insert transfers defined by the revision.  A transfer that referenced
    #    a cancelled trip was deleted above and is absent here, so it cannot
    #    survive as a stale entry.
    for key in sorted(insert_transfers, key=lambda item: str(item)):
        row = new_snapshot.rows['transfers'][key]
        session.add(Transfer(feed_id=feed_id, **dict(row)))
    session.flush()

    # 6. Regenerate the affected many-to-many link rows from the revision.
    if insert_trips:
        _rebuild_trip_shapes(session, feed_id, insert_trips)
    if insert_stops:
        _rebuild_stop_translations(session, feed_id, insert_stops)

    session.flush()
    session.commit()


def _delete_transfer_keys(session, feed_id, keys):
    from .gtfs_entities import Transfer

    conditions = []
    for key in keys:
        natural_key = key[0]
        conditions.append(and_(*[
            getattr(Transfer, column) == value
            for column, value in zip(
                ('from_stop_id', 'to_stop_id', 'from_route_id',
                 'to_route_id', 'from_trip_id', 'to_trip_id'),
                natural_key)
        ]))
    if conditions:
        (session.query(Transfer)
         .filter(Transfer.feed_id == feed_id)
         .filter(or_(*conditions))
         .delete(synchronize_session=False))


def _rebuild_stop_translations(session, feed_id, stop_ids=None):
    q = (session.query(
            Stop.feed_id.label('stop_feed_id'),
            Translation.feed_id.label('translation_feed_id'),
            Stop.stop_id.label('stop_id'),
            Translation.trans_id.label('trans_id'),
            Translation.lang.label('lang'))
        .filter(Stop.feed_id == feed_id)
        .filter(Translation.feed_id == feed_id)
        .filter(Stop.stop_name == Translation.trans_id))
    if stop_ids is not None:
        q = q.filter(Stop.stop_id.in_(stop_ids))
    session.execute(
        _stop_translations.insert().from_select(
            ['stop_feed_id', 'translation_feed_id', 'stop_id', 'trans_id',
             'lang'], q))


def _rebuild_trip_shapes(session, feed_id, trip_ids=None):
    q = (session.query(
            Trip.feed_id.label('trip_feed_id'),
            ShapePoint.feed_id.label('shape_feed_id'),
            Trip.trip_id.label('trip_id'),
            ShapePoint.shape_id.label('shape_id'),
            ShapePoint.shape_pt_sequence.label('shape_pt_sequence'))
        .filter(Trip.feed_id == feed_id)
        .filter(ShapePoint.feed_id == feed_id)
        .filter(ShapePoint.shape_id == Trip.shape_id))
    if trip_ids is not None:
        q = q.filter(Trip.trip_id.in_(trip_ids))
    session.execute(
        _trip_shapes.insert().from_select(
            ['trip_feed_id', 'shape_feed_id', 'trip_id', 'shape_id',
             'shape_pt_sequence'], q))


def append_feed(schedule, feed_filename, strip_fields=True,
                chunk_size=5000, agency_id_override=None, ignore_files=()):

    fd = feed.Feed(feed_filename, strip_fields)

    gtfs_tables = {}
    for gtfs_class in gtfs_all:
        gtfs_filename = gtfs_class.__tablename__ + '.txt'

        if gtfs_filename in ignore_files:
            logger.info("ignoring file %s as requested" % gtfs_filename)
            continue

        print('Loading GTFS data for %s:' % gtfs_class)
        logger.info('Loading GTFS data for %s:' % gtfs_class)

        try:
            # We ignore the feed supplied feed id, because we create our own
            # later.
            gtfs_tables[gtfs_class] = fd.read_table(gtfs_filename,
                                                    set(c.name for c in gtfs_class.__table__.columns) - {'feed_id'})
        except (KeyError, IOError):
            if gtfs_class in gtfs_required:
                raise IOError('Error: could not find %s' % gtfs_filename)

    if len(set(gtfs_tables) & gtfs_calendar) == 0:
        raise PygtfsException('Must have Calendar.txt or Calendar_dates.txt')


    # create new feed
    feed_entry = Feed(feed_name=fd.feed_name, feed_append_date=date.today())
    schedule.session.add(feed_entry)
    schedule.session.flush()
    feed_id = feed_entry.feed_id
    for gtfs_class in gtfs_all:
        if gtfs_class not in gtfs_tables:
            continue
        gtfs_table = gtfs_tables[gtfs_class]


        for i, record in enumerate(gtfs_table):
            if not record:
                # Empty row.
                continue

            try:
                instance = gtfs_class(feed_id=feed_id, **record._asdict())
            except:
                print("Failure while writing {0}".format(record))
                logger.error("Failure while writing {0}".format(record))
                raise
            schedule.session.add(instance)

            if isinstance(instance, ServiceException):
                service = schedule.session.execute(
                    select(Service).where(Service.service_id == instance.service_id).where(Service.feed_id == feed_id)
                ).one_or_none()
                if service is None:
                    # ServiceException was added for a day that has no associated regular service
                    # Create a dummy service object for this service exception
                    dummy = Service(
                        feed_id=feed_id,
                        service_id=instance.service_id,
                        monday='0',
                        tuesday='0',
                        wednesday='0',
                        thursday='0',
                        friday='0',
                        saturday='0',
                        sunday='0',
                        start_date=instance.date.strftime('%Y%m%d'),
                        end_date=instance.date.strftime('%Y%m%d'))
                    schedule.session.add(dummy)

            if i % chunk_size == 0 and i > 0:
                schedule.session.flush()
                sys.stdout.write('.')
                sys.stdout.flush()
        print('%d record%s read for %s.' % ((i+1), '' if i == 0 else 's',
                                            gtfs_class))
        logger.info('%d record%s read for %s.' % ((i+1), '' if i == 0 else 's',
                                            gtfs_class))
    schedule.session.flush()
    schedule.session.commit()

    # A full import must match the incremental path: rows which point at a
    # stop, route, trip or fare absent from this same feed do not represent a
    # usable relation and are dropped.
    from . import diff as feed_diff
    feed_diff.prune_invalid_references(schedule, feed_id)

    # load many to many relationships
    if Translation in gtfs_tables:
        print('Mapping translations to stops')
        logger.info('Mapping translations to stops')
        q = (schedule.session.query(
                Stop.feed_id.label('stop_feed_id'),
                Translation.feed_id.label('translation_feed_id'),
                Stop.stop_id.label('stop_id'),
                Translation.trans_id.label('trans_id'),
                Translation.lang.label('lang'))
            .filter(Stop.feed_id==feed_id)
            .filter(Translation.feed_id==feed_id)
            .filter(Stop.stop_name==Translation.trans_id)
            )
        upd = _stop_translations.insert().from_select(
                ['stop_feed_id', 'translation_feed_id', 'stop_id', 'trans_id', 'lang'], q)
        schedule.session.execute(upd)
    if ShapePoint in gtfs_tables:
        print('Mapping shapes to trips')
        logger.info('Mapping shapes to trips')
        q = (schedule.session.query(
                Trip.feed_id.label('trip_feed_id'),
                ShapePoint.feed_id.label('shape_feed_id'),
                Trip.trip_id.label('trip_id'),
                ShapePoint.shape_id.label('shape_id'),
                ShapePoint.shape_pt_sequence.label('shape_pt_sequence'))
            .filter(Trip.feed_id==feed_id)
            .filter(ShapePoint.feed_id==feed_id)
            .filter(ShapePoint.shape_id==Trip.shape_id)
            )
        upd = _trip_shapes.insert().from_select(
                ['trip_feed_id', 'shape_feed_id', 'trip_id', 'shape_id', 'shape_pt_sequence'], q)
        schedule.session.execute(upd)
    schedule.session.commit()

    print('Complete.')
    logger.info('Complete.')
    return schedule
