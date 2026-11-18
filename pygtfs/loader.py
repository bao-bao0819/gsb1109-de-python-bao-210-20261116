from datetime import date
import logging
import sys

from sqlalchemy import and_
from sqlalchemy.sql.expression import select, join

from . import feed, gtfs_entities
from .gtfs_entities import (Feed, Service, ServiceException, gtfs_required,
                            Translation, TranslationTarget, Trip,
                            ShapePoint, _trip_shapes, gtfs_calendar, gtfs_all,
                            TRANSLATION_TARGETS, MATCH_BASIS_RECORD_ID,
                            MATCH_BASIS_FIELD_VALUE)

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
    # load many to many relationships
    if Translation in gtfs_tables:
        _map_translations(schedule, feed_id)
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


def _map_translations(schedule, feed_id):
    """Resolve translations.txt rows onto concrete feed records.

    A row carrying ``table_name``/``field_name``/``record_id`` is bound to
    that exact record with basis ``record_id`` and is never expanded to
    other records that share the same field value. Every other row is
    matched against the translatable field values of the supported tables
    with basis ``field_value``; each hit (table, field and record) becomes a
    separate candidate binding. All bindings keep the basis that actually
    fired, plus the matched field value.
    """
    print('Mapping translations to feed records')
    logger.info('Mapping translations to feed records')
    session = schedule.session
    entity_by_table = {target[1]: target for target in TRANSLATION_TARGETS}
    translations = (session.query(Translation)
                    .filter(Translation.feed_id == feed_id).all())
    bindings = []
    for trans in translations:
        if (trans.table_name is not None and trans.field_name is not None
                and trans.record_id is not None):
            bindings.extend(_record_id_bindings(session, feed_id, trans,
                                                entity_by_table))
        elif trans.trans_id is not None:
            bindings.extend(_field_value_bindings(session, feed_id, trans))
    if bindings:
        session.execute(TranslationTarget.__table__.insert(), bindings)
    session.commit()


def _record_id_bindings(session, feed_id, trans, entity_by_table):
    target = entity_by_table.get(trans.table_name)
    if target is None:
        logger.warning('Ignoring translation %s: unsupported table %s',
                       trans.trans_id, trans.table_name)
        return []
    entity_name, table_name, id_field, fields = target
    if trans.field_name not in fields:
        logger.warning('Ignoring translation %s: field %s is not translatable '
                       'on %s', trans.trans_id, trans.field_name, table_name)
        return []
    entity = getattr(gtfs_entities, entity_name)
    exists = (session.query(entity)
              .filter(entity.feed_id == feed_id)
              .filter(getattr(entity, id_field) == trans.record_id)
              .first())
    if exists is None:
        logger.warning('Ignoring translation %s: no %s record with id %s',
                       trans.trans_id, table_name, trans.record_id)
        return []
    return [{
        'feed_id': feed_id,
        'translation_feed_id': feed_id,
        'target_table': table_name,
        'target_field': trans.field_name,
        'target_record_id': trans.record_id,
        'trans_id': trans.trans_id,
        'lang': trans.lang,
        'match_basis': MATCH_BASIS_RECORD_ID,
        'field_value': None,
    }]


def _field_value_bindings(session, feed_id, trans):
    bindings = []
    for entity_name, table_name, id_field, fields in TRANSLATION_TARGETS:
        entity = getattr(gtfs_entities, entity_name)
        for field_name in fields:
            records = (session.query(getattr(entity, id_field))
                       .filter(entity.feed_id == feed_id)
                       .filter(getattr(entity, field_name) == trans.trans_id)
                       .all())
            for (record_id,) in records:
                bindings.append({
                    'feed_id': feed_id,
                    'translation_feed_id': feed_id,
                    'target_table': table_name,
                    'target_field': field_name,
                    'target_record_id': record_id,
                    'trans_id': trans.trans_id,
                    'lang': trans.lang,
                    'match_basis': MATCH_BASIS_FIELD_VALUE,
                    'field_value': trans.trans_id,
                })
    if not bindings:
        logger.warning('Translation %s (%s) matched no field value',
                       trans.trans_id, trans.lang)
    return bindings
