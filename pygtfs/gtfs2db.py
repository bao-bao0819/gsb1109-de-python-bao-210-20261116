""" gtfs2db - convert a gtfs feed to a pygtfs database

Usage:
  gtfs2db append <feed_file> <database> [--chunk-size <integer>] [--ignore_files FILES] [--predecessor-feed-id <id>] [--lineage-note <text>] [--aliases <file>]
  gtfs2db overwrite <feed_file> <database> [-i, --interactive] [--chunk-size <integer>] [--ignore_files FILES] [--predecessor-feed-id <id>] [--lineage-note <text>] [--aliases <file>]
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
  --predecessor-feed-id <id>  feed_id of the earlier publication this one continues.
                              Records feed lineage without merging or deleting it.
  --lineage-note <text>      Free-text note stored alongside the lineage link.
  --aliases <file>     CSV of explicit cross-feed entity aliases, with header
                       "entity_type,feed_id_a,source_id_a,feed_id_b,source_id_b".
                       Applied after the feed is imported; feeds must exist first.
  <feed_file>           The gtfs file on which to operate. Can be either a folder
                        containing .txt files, or a .zip file.
  <database>            The database. Can be either a file, which is interpreted
                        as an sqlite database stored in this file, or a sqlalchemy
                        database connection.

Commands:
  append            appends the gtfs feed to the database
  overwrite         delete any existing feeds which had the same original
                    filename as the new file, and then append the new file.
  delete            delete from the database any feeds with the name supplied.
  list              list existing feeds in the database.

Description:
  This is a tool to manage a database containing several gtfs feeds. The
  database is in a pygtfs 0.1.0 format, and can be stored as any database
  supported by sqlalchemy (the default being sqlite).
  The database file can later be used to create a `pygtfs.Schedule` instance.

  Every imported entity receives a source-scoped identity keyed by
  (feed_id, entity_type, source_id), so feeds from different agencies never
  merge just because they reuse an id. Cross-feed equivalence is declared
  explicitly through an aliases file, and feed publication history is tracked
  separately through predecessor links.
"""


from docopt import docopt

from . import (__version__, append_feed, delete_feed, overwrite_feed,
               list_feeds, add_aliases_from_file)
from .schedule import Schedule


def main():
    """Script to load GTFS data into a database."""

    args = docopt(__doc__, version=__version__)
    schedule = Schedule(args['<database>'])

    ignore_files = ()
    if ignore_files_str := args['--ignore_files']:
        ignore_files = ignore_files_str.split(',')

    predecessor_feed_id = args['--predecessor-feed-id']
    if predecessor_feed_id is not None:
        predecessor_feed_id = int(predecessor_feed_id)

    if args['append']:
        append_feed(schedule, args['<feed_file>'],
                    chunk_size=int(args['--chunk-size']),
                    ignore_files=ignore_files,
                    predecessor_feed_id=predecessor_feed_id,
                    lineage_note=args['--lineage-note'])
    elif args['delete']:
        delete_feed(schedule, args['<feed_file>'],
                    interactive=args['--interactive'])
    elif args['overwrite']:
        overwrite_feed(schedule, args['<feed_file>'],
                       interactive=args['--interactive'],
                       chunk_size=int(args['--chunk-size']),
                       ignore_files=ignore_files,
                       predecessor_feed_id=predecessor_feed_id,
                       lineage_note=args['--lineage-note'])
    elif args['list']:
        list_feeds(schedule)

    if args['--aliases']:
        add_aliases_from_file(schedule.session, args['--aliases'])
        schedule.session.commit()

if __name__ == '__main__':
    main()
