"""Actual stop-time derivation for GTFS feeds.

This module turns the offsets in ``stop_times.txt`` into timezone-aware
datetimes and can persist or export them.  It is available both as a library
API and through the ``gtfs-actual-stop-times`` command.

The service day is interpreted as specified by GTFS: noon minus twelve hours
in the agency timezone.  On daylight-saving transition days this starting
point may not be local midnight, and offsets such as ``25:10:00`` are kept as
elapsed time from that starting point.  They are never reduced modulo 24
hours, so an after-midnight stop remains on its original operating trip and
service date.
"""

import argparse
import sys

from .schedule import Schedule


def _service_dates(schedule, feed_ids, start_date, end_date):
    dates = set()
    for feed_id in feed_ids:
        for trip in schedule.trips:
            if trip.feed_id != feed_id:
                continue
            dates.update(schedule.service_dates_for_trip(
                trip, start_date, end_date))
    return sorted(dates)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="gtfs-actual-stop-times",
        description="Resolve GTFS stop_times to actual local datetimes.")
    parser.add_argument("database",
                        help="SQLite file or SQLAlchemy database URL")
    parser.add_argument("--feed", type=int, action="append", default=None,
                        help="feed id to export (repeatable; default: all)")
    parser.add_argument("--service-date", default=None,
                        help="single service date, YYYYMMDD or YYYY-MM-DD")
    parser.add_argument("--start-date", default=None,
                        help="first service date to export")
    parser.add_argument("--end-date", default=None,
                        help="last service date to export")
    parser.add_argument("--start", default=None,
                        help="start of an absolute time window (ISO 8601)")
    parser.add_argument("--end", default=None,
                        help="end of an absolute time window (ISO 8601)")
    parser.add_argument("--trip", default=None, help="trip id to export")
    parser.add_argument("--stop", default=None, help="stop id to export")
    parser.add_argument("--timezone", default=None,
                        help="override the agency timezone")
    parser.add_argument("--csv", default=None,
                        help="write CSV to this path instead of a table")
    parser.add_argument("--no-persist", action="store_true",
                        help="do not write the actual_stop_times table")
    args = parser.parse_args(argv)

    schedule = Schedule(args.database)

    if args.feed:
        feed_ids = args.feed
    else:
        feed_ids = [feed.feed_id for feed in schedule.feeds]
    missing = set(feed_ids) - {feed.feed_id for feed in schedule.feeds}
    if missing:
        parser.error("unknown feed id(s): {0}".format(
            ", ".join(str(feed_id) for feed_id in sorted(missing))))

    filters = dict(trip=args.trip, stop_id=args.stop, timezone=args.timezone,
                   service_date=args.service_date, start=args.start,
                   end=args.end)
    if args.start_date or args.end_date:
        if args.service_date is not None:
            parser.error("--service-date cannot be combined with date bounds")
        first = args.start_date or args.end_date
        last = args.end_date or args.start_date
        if not args.no_persist and not args.csv:
            total = 0
            for day in _service_dates(schedule, feed_ids, first, last):
                for feed_id in feed_ids:
                    total += schedule.export_actual_stop_times(
                        service_date=day, feed_id=feed_id, **filters)
            print("Exported {0} actual stop time(s).".format(total))
            return 0
        total = 0
        for day in _service_dates(schedule, feed_ids, first, last):
            for feed_id in feed_ids:
                day_filters = dict(filters, service_date=day)
                if args.csv:
                    path = "{0}.{1}{2}".format(
                        args.csv, day.strftime("%Y%m%d"),
                        ".{0}".format(feed_id) if args.feed else "")
                    total += schedule.write_actual_stop_times(
                        path, feed_id=feed_id, **day_filters)
                else:
                    total += len(schedule.actual_stop_times(
                        feed_id=feed_id, **day_filters))
        print("Exported {0} actual stop time(s).".format(total))
        return 0

    if args.csv:
        written = schedule.write_actual_stop_times(
            args.csv, **filters)
    elif args.no_persist:
        written = len(schedule.actual_stop_times(**filters))
    else:
        written = 0
        for feed_id in feed_ids:
            written += schedule.export_actual_stop_times(
                feed_id=feed_id, **filters)
    print("Exported {0} actual stop time(s).".format(written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
