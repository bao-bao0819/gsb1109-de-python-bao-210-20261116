pygtfs
========

[![Travis](https://img.shields.io/travis/jarondl/pygtfs/master.svg?style=flat-square)](https://travis-ci.org/jarondl/pygtfs)
[![PyPI](https://img.shields.io/pypi/v/pygtfs.svg)](https://pypi.python.org/pypi/pygtfs)


Overview
--------

pygtfs is a library that models information stored in Google's
[General Transit Feed Specification (GTFS)](https://developers.google.com/transit/)
format. GTFS is a format designed to specify information about a transit
system, such as a city's subways or a private company's bus services. pygtfs
stores information in an SQLite database using SQLAlchemy to facilitate the
storage of Python objects in a relational database. 

pygtfs is a fork of @eoghanmurray's fork of a @andrewblim's gtfs-sql which is
a fork of @bmander's gtfs. See the git logs for more fun history.

License: MIT, included in `license.txt`.


Dependencies
------------

- [SQLAlchemy](http://www.sqlalchemy.org/) 0.7.8. Used for all mapping of GTFS
  objects to the relational DB. You'll need to be familiar with it to read the
  code; the [documentation](http://docs.sqlalchemy.org/) is pretty solid. 
- [pytz](http://pytz.sourceforge.net/) 2012d. A few GTFS fields are expected
  to be in a [tz time zone format](http://en.wikipedia.org/wiki/List_of_tz_database_time_zones). 
- [docopt](http://docopt.org/). Pythonic command line arguments parser, that
  will make you smile

Installation
------------

Get [setuptools](http://pypi.python.org/pypi/setuptools) if you don't have it,
clone the repo, and use `python setup.py install`.

Documentation
-------------
Hosted on https://pygtfs.readthedocs.org/

Service days and actual stop times
----------------------------------
GTFS times are offsets from the start of a service day (noon minus twelve
hours in the agency timezone), so ``25:10:00`` belongs to the previous
operating day even though its actual clock time falls after midnight.

```python
import pygtfs

sched = pygtfs.Schedule("gtfs.sqlite")
records = sched.stop_times_for_trip("NIGHT", "20240309")
records[-1].service_date                 # datetime.date(2024, 3, 9)
records[-1].arrival_gtfs_time            # "25:10:00"
records[-1].arrival_datetime             # 2024-03-10 01:10-05:00

# Query by actual instant; the operating trip is preserved.
sched.actual_stop_times(start="2024-03-10T00:30:00-05:00",
                        end="2024-03-10T02:00:00-05:00")

# Persist for SQL access, or write CSV with both representations.
sched.export_actual_stop_times(trip="NIGHT", service_date="20240309")
sched.write_actual_stop_times("night.csv", trip="NIGHT",
                              service_date="20240309")
```

The ``gtfs-actual-stop-times`` command exposes the same export:
``gtfs-actual-stop-times gtfs.sqlite --service-date 20240309 --trip NIGHT``.

Daylight-saving transitions follow the same rule: the service-day start is
computed from local noon, so spring-forward days start an hour earlier and
fall-back days an hour later than usual.  Times are never reduced modulo 24
hours.

TODO
-----

- Improve testing; add some unit testing framework and test with a variety of GTFS data feeds. 
- Add more docs

Why fork?
--------------
- natively support several gtfs feeds per database
- less SLOC, more DRY
- add python3 support
- renamed to a more generic name
- will continue to maintain
