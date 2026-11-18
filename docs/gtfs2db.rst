The gtfs2db script
===================

This is a script to manage the database. Here is its help message::

     gtfs2db - convert a gtfs feed to a pygtfs database

    Usage:
      gtfs2db append <feed_file> <database> [--chunk-size <integer>]
      gtfs2db overwrite <feed_file> <database> [-i, --interactive] [--chunk-size <integer>]
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
      <feed_file>        The gtfs file on which to operate. Can be either a folder
			 containing .txt files, or a .zip file.
      <database>         The database. Can be either a file, which is interpreted
			 as an sqlite database stored in this file, or a sqlalchemy
			 database connection.

    Commands:
      append            appends the gtfs feed to the database. If the same
			revision (same filename and feed version) was already
			imported, it is skipped instead of duplicated.
      overwrite         delete any existing feed which had the same original
			filename and the same feed version as the new file, and
			then append the new file. Revisions published at other
			times are kept, so the data as published at any point
			in time can still be queried.
      delete            delete from the database any feeds with the name supplied.
      list              list existing feeds in the database. 

    Description:
      This is a tool to manage a database containing several gtfs feeds. The
      database is in a pygtfs 0.1.0 format, and can be stored as any database
      supported by sqlalchemy (the default being sqlite).
      The database file can later be used to create a `pygtfs.Schedule` instance. 
