""" gtfs2db - convert a gtfs feed to a pygtfs database

Usage:
  gtfs2db append <feed_file> <database> [--chunk-size <integer>] [--ignore_files FILES]
  gtfs2db overwrite <feed_file> <database> [-i, --interactive] [--chunk-size <integer>] [--ignore_files FILES]
  gtfs2db update <feed_file> <database> [--old-feed-file <file>] [--feed-id <id>] [--chunk-size <integer>] [--ignore_files FILES]
  gtfs2db diff <feed_file> <database> [--old-feed-file <file>] [--feed-id <id>] [--ignore_files FILES]
  gtfs2db delete <feed_file> <database> [-i, --interactive]
  gtfs2db list <database>
  gtfs2db (-h | --help)
  gtfs2db --version

Options:
  -h --help             Show this help screen.
  --version             Show version.
  -i --interactive      Ask before deleting or overwriting existing feeds.
  --chunk-size <int>    How often to flush database. If memory consumption is high,
                        lower this number. [default: 10000]
  --ignore_files FILES  Ignore these files in the read, e.g. "routes.txt,stop_times.txt".
  --old-feed-file <file> The previous feed version to use as the diff baseline.
  --feed-id <id>        Feed id to update when several feeds exist.
  <feed_file>           The gtfs file on which to operate. Can be either a folder
                        containing .txt files, or a .zip file.
  <database>            The database. Can be either a file, which is interpreted
                        as an sqlite database stored in this file, or a sqlalchemy
                        database connection.

Commands:
  append            appends the gtfs feed to the database
  overwrite         delete any existing feeds which had the same original
                    filename as the new file, and then append the new file.
  update            apply only the rows which changed in a revised feed
  diff              show what an update would rebuild, without changing anything
  delete            delete from the database any feeds with the name supplied.
  list              list existing feeds in the database.

Description:
  This is a tool to manage a database containing several gtfs feeds. The
  database is in a pygtfs 0.1.0 format, and can be stored as any database
  supported by sqlalchemy (the default being sqlite).
  The database file can later be used to create a `pygtfs.Schedule` instance.
"""


import json

from docopt import docopt

from . import (__version__, append_feed, delete_feed, diff_feed,
               overwrite_feed, update_feed, list_feeds)
from .schedule import Schedule


def main():
    """Script to load GTFS data into a database."""

    args = docopt(__doc__, version=__version__)
    schedule = Schedule(args['<database>'])

    ignore_files = ()
    if ignore_files_str := args['--ignore_files']:
        ignore_files = ignore_files_str.split(',')

    feed_id = int(args['--feed-id']) if args['--feed-id'] else None

    if args['append']:
        append_feed(schedule, args['<feed_file>'],
                    chunk_size=int(args['--chunk-size']),
                    ignore_files=ignore_files)
    elif args['delete']:
        delete_feed(schedule, args['<feed_file>'],
                    interactive=args['--interactive'])
    elif args['overwrite']:
        overwrite_feed(schedule, args['<feed_file>'],
                       interactive=args['--interactive'],
                       chunk_size=int(args['--chunk-size']),
                       ignore_files=ignore_files)
    elif args['update']:
        update_feed(schedule, args['<feed_file>'],
                    old_feed_filename=args['--old-feed-file'],
                    feed_id=feed_id,
                    chunk_size=int(args['--chunk-size']),
                    ignore_files=ignore_files)
    elif args['diff']:
        changes = diff_feed(schedule, args['<feed_file>'],
                            old_feed_filename=args['--old-feed-file'],
                            feed_id=feed_id,
                            ignore_files=ignore_files)
        print(json.dumps(changes.report(), indent=2, default=str))
        return changes
    elif args['list']:
        list_feeds(schedule)

if __name__ == '__main__':
    main()
