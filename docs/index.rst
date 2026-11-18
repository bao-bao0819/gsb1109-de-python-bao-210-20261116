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

Multi-feed identity, aliases, and lineage
-----------------------------------------

A GTFS id is only unique inside one feed, so two agencies can both publish,
for example, ``route_id = "1"``. On import, every entity receives a
source-scoped identity keyed by ``(feed_id, entity_type, source_id)``, so such
collisions never mix detail rows across feeds::

    >>> from pygtfs import get_identity
    >>> alpha = get_identity(sched.session, feed_id=1, entity_type="route", source_id="1")
    >>> beta  = get_identity(sched.session, feed_id=2, entity_type="route", source_id="1")
    >>> alpha.identity_key == beta.identity_key
    False

Equivalence across feeds is only established through an explicit alias; a
shared stop or an identical id never causes an automatic merge::

    >>> from pygtfs import add_alias
    >>> shared_key = add_alias(sched.session, "route", 1, "1", 2, "1")

Different publications of the same feed are linked, not merged. Passing a
predecessor records lineage while keeping the older publication reviewable::

    >>> append_feed(sched, "feed_v2.zip", predecessor_feed_id=1)
    >>> from pygtfs import lineage_chain
    >>> lineage_chain(sched.session, feed_id=2)
    [2, 1]

See :py:mod:`pygtfs.identity` for the full API.

gtfs2db
-------

`setup.py install` will also install a command-line script `gtfs2db` that takes a GTFS zip file or directory as an argument and will load the data into a database usable with pygtfs. Run `gtfs2db --help` for more. 

Detailed refernce
-----------------
The best place to start is :py:mod:`pygtfs.schedule`


Contents:
---------

.. toctree::
   :maxdepth: 2

   schedule
   loader
   identity
   gtfs_entities
   gtfs2db
   internal



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

