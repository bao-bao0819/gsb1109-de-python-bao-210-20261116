The gtfs2db script
===================

This is a script to manage the database. Here is its help message::

     gtfs2db - convert a gtfs feed to a pygtfs database

    Usage:
      gtfs2db append <feed_file> <database> [--chunk-size <integer>]
      gtfs2db overwrite <feed_file> <database> [-i, --interactive] [--chunk-size <integer>]
      gtfs2db update <feed_file> <database> [--old-feed-file <file>] [--feed-id <id>]
      gtfs2db diff <feed_file> <database> [--old-feed-file <file>] [--feed-id <id>]
      gtfs2db delete <feed_file> <database> [-i, --interactive]
      gtfs2db list <database>
      gtfs2db (-h | --help)
      gtfs2db --version

    Options:
      -h --help          Show this help screen.
      --version          Show version.
      -i --interactive   Ask before deleting or overwriting existing feeds.
      --chunk-size <int> How often to flush database. If memory consumption is high,
			 lower this number. [default: 10000]
      --old-feed-file <file> Previous feed version to use as the diff baseline.
      --feed-id <id>     Feed id to update when several feeds exist.
      <feed_file>        The gtfs file on which to operate. Can be either a folder
			 containing .txt files, or a .zip file.
      <database>         The database. Can be either a file, which is interpreted
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

Incremental revisions
---------------------

``gtfs2db update`` compares a revised feed with the currently stored feed and
touches only what the revision requires: removed rows are deleted, modified
rows are updated in place (keeping their database identity), unchanged rows
are left alone, and new rows are inserted.  ``gtfs2db diff`` prints the same
comparison as JSON without changing the database.

The report separates:

- ``added`` / ``removed`` / ``modified``: source GTFS rows which differ.
- ``affected``: entities whose derived results depend on a change.
- ``derived``: the exact derived rows an update regenerates -- stop times and
  frequencies, transfer relation rows, and the ``_trip_shapes`` and
  ``_stop_translations`` link rows.
- ``rebuild``: the derived tables involved.
- ``dependencies``: for each changed entity, the derived results which depend
  on it.

Derived relations are rebuilt from the new input, not copied from the old
database.  In particular, a transfer whose ``from_trip_id`` or ``to_trip_id``
names a trip cancelled by the revision is dropped, so a stale transfer entry
cannot survive.  After the update the database matches a fresh full import of
the revision. 
