from datetime import date
import logging
import sys

from sqlalchemy import and_
from sqlalchemy.sql.expression import select, join

from .gtfs_entities import (Feed, Service, ServiceException, gtfs_required,
                            Translation, Stop, Trip, ShapePoint, _stop_translations,
                            _trip_shapes, gtfs_calendar, gtfs_all)
from .exceptions import PygtfsException
from . import feed

logger = logging.getLogger(__name__)


def _get_or_create_placeholder_service(session, feed_id, service_id,
                                       reference_date=None, reason=None):
    """Return the Service for service_id, creating a placeholder if missing.

    A GTFS feed may omit calendar.txt as long as it ships calendar_dates.txt,
    and either calendar file may leave a service_id that other tables still
    reference undeclared. Rather than dropping the referencing rows or leaving
    a dangling foreign key, we keep the independently verifiable entity (the
    trip or service exception) and synthesize an inactive, single-day Service
    placeholder so the relationship stays intact.
    """
    service = session.execute(
        select(Service).where(Service.service_id == service_id).where(
            Service.feed_id == feed_id)
    ).one_or_none()
    if service is not None:
        return service, False

    if reference_date is None:
        reference_date = date.today().strftime('%Y%m%d')
    placeholder = Service(
        feed_id=feed_id,
        service_id=service_id,
        monday='0',
        tuesday='0',
        wednesday='0',
        thursday='0',
        friday='0',
        saturday='0',
        sunday='0',
        start_date=reference_date,
        end_date=reference_date)
    session.add(placeholder)
    if reason == "calendar_dates.txt":
        message = (
            "service_id %r is used by calendar_dates.txt but has no regular "
            "service in calendar.txt; inserting a placeholder service"
            % service_id)
    else:
        message = (
            "service_id %r%s is defined in neither calendar.txt nor "
            "calendar_dates.txt; inserting a placeholder service"
            % (service_id, " referenced by %s" % reason if reason else ""))
    logger.warning(message)
    return placeholder, True


def _reconcile_trip_services(session, feed_id):
    """Ensure every trip resolves to a persisted Service.

    A missing calendar.txt is legal whenever calendar_dates.txt is present,
    but a trip whose service_id appears in neither calendar file would
    otherwise keep a null service relationship. Such trips are independently
    verifiable entities, so instead of dropping them we back the referenced
    service with a placeholder and surface the data gap via a warning.
    """
    if not session.query(Trip.feed_id).filter(Trip.feed_id == feed_id).first():
        return

    declared_service_ids = set(
        session.execute(
            select(Service.service_id).where(Service.feed_id == feed_id)
        ).scalars()
    )
    referenced_service_ids = set(
        session.execute(
            select(Trip.service_id)
            .where(Trip.feed_id == feed_id)
            .where(Trip.service_id.isnot(None))
            .distinct()
        ).scalars()
    )

    for service_id in sorted(referenced_service_ids - declared_service_ids):
        _get_or_create_placeholder_service(
            session, feed_id, service_id, reason="trips.txt")


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
                # ServiceException was added for a day that has no associated
                # regular service. Create a placeholder service so the
                # exception keeps a valid relationship instead of dangling.
                _get_or_create_placeholder_service(
                    schedule.session, feed_id, instance.service_id,
                    reference_date=instance.date.strftime('%Y%m%d'),
                    reason="calendar_dates.txt")

            if i % chunk_size == 0 and i > 0:
                schedule.session.flush()
                sys.stdout.write('.')
                sys.stdout.flush()
        print('%d record%s read for %s.' % ((i+1), '' if i == 0 else 's',
                                            gtfs_class))
        logger.info('%d record%s read for %s.' % ((i+1), '' if i == 0 else 's',
                                            gtfs_class))
    schedule.session.flush()
    _reconcile_trip_services(schedule.session, feed_id)
    schedule.session.commit()
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
