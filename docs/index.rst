.. pygtfs documentation master file, created by
   sphinx-quickstart on Wed Jan 22 11:52:42 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pygtfs - a database backed python gtfs interface!
=================================================

Get it
------
The source is available on github: https://github.com/jarondl/pygtfs

Basic usage
-----------

To include pygtfs functionality in your application, use `import pygtfs`. 

The first thing you need to to is to create a new schedule object::

    sched = pygtfs.Schedule(":memory:")

This will create an in memory sqlite database. Instead you can supply a
filename to be used for sqlite (such as ('gtfs.sqlite'), or a sqlalchemy
database connection.

Then you can load gtfs feeds into the databas, by using append::

    pygtfs.append_feed(sched, "sample-gtfs-feed.zip")

Where the gtfs feed can be either a `.zip` file, or a folder full of `.txt` files.
You can add as many feeds as you want into a single database, without fear
of conflicts (but you can two stop names for one place, one from each feed for example).
Another option to load feeds is to use the 'gtfs2db' script as explained later.

The Schedule object represents a collection of objects that correspond to the
contents of a GTFS feed. You can get the list of agencies, stops, routes, etc.
with fairly straightforwardly named attributes, see :py:mod:`pygtfs.schedule` 
for more details.

    >>> sched.agencies
    [<Agency BART: Bay Area Rapid Transit>, <Agency AirBART: AirBART>]
    >>> sched.routes
    [<Route AirBART: >, <Route 01: >, <Route 03: >, <Route 05: >, <Route 07: >, <Route 11: >]

For GTFS entities that are identified by a dataset-unique identifier, there is
also a function to get them by id: 

    >>> sched.agencies_by_id('AirBART')
    [<Agency AirBART: AirBART>]
    >>> sched.stops_by_id('SFIA')
    [<Stop SFIA: San Francisco Int'l Airport>]

The GTFS entity objects have attributes that correspond in name to the field
definitions in the [GTFS reference](https://developers.google.com/transit/gtfs/reference). 

    >>> sched.stops_by_id('SFIA')[0].stop_name
    u"San Francisco Int'l Airport"
    >>> sched.routes[1].route_long_name
    u'Pittsburg/Bay Point - SFIA/Millbrae'

GTFS entities which cross-reference each other can also be obtained straightforwardly with attributes (again, see "Reference" below for full details):

    >>> sched.trips_by_id('01SFO10').service  # the service associated with trip 01SFO10
    <Service WKDY (MTWThFSSu)>

gtfs2db
-------

`setup.py install` will also install a command-line script `gtfs2db` that takes a GTFS zip file or directory as an argument and will load the data into a database usable with pygtfs. Run `gtfs2db --help` for more. 

Detailed refernce
-----------------
The best place to start is :py:mod:`pygtfs.schedule`

Service days and actual stop times
----------------------------------
GTFS times in ``stop_times.txt`` are offsets from the start of a service day,
not wall-clock times.  The service day begins at noon minus twelve hours in
the agency timezone, which keeps its identity across midnight and across
daylight-saving transitions.

::

    >>> sched.stop_times_for_trip("NIGHT", "20240309")[-1]
    ResolvedStopTime(trip_id='NIGHT', service_date=datetime.date(2024, 3, 9),
                     stop_id='S2', arrival_gtfs_time='25:10:00',
                     arrival_datetime=datetime.datetime(2024, 3, 10, 1, 10,
                                                        tzinfo=...))

The original ``25:10:00`` value and the service date are both retained, while
``arrival_datetime`` is the actual timezone-aware instant.  To query by actual
time, use ``actual_stop_times(start=..., end=...)``.  To persist results for
SQL access, use ``export_actual_stop_times(...)``: rows are stored in the
``actual_stop_times`` table with ``service_date``, the original
``arrival_gtfs_time``/``departure_gtfs_time`` strings, UTC instants, and local
ISO 8601 timestamps.  ``write_actual_stop_times(path, ...)`` writes the same
data as CSV.

The ``gtfs-actual-stop-times`` command wraps the export::

    gtfs-actual-stop-times gtfs.sqlite --service-date 20240309 \
        --trip NIGHT --csv night.csv


Contents:
---------

.. toctree::
   :maxdepth: 2

   schedule
   loader
   gtfs_entities
   gtfs2db
   internal



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

