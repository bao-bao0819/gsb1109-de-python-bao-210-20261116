Frequency based service
=======================

``frequencies.txt`` rows come in two flavours.  pygtfs keeps them separate
instead of pretending that every headway is a timetable.

Exact service (``exact_times=1``)
---------------------------------

The times are scheduled, so the row can be exported as concrete departures::

    >>> expansion = sched.expand_frequencies(trip_id='CITY1')
    >>> expansion.departures
    [<DepartureInstance CITY1 at 08:00:00>, ...]

Each :class:`pygtfs.frequencies.DepartureInstance` also exposes
``stop_times``, the trip's template stop times shifted to that departure.

Frequency based service (``exact_times=0`` or empty)
----------------------------------------------------

No individual departure is promised.  pygtfs returns a
:class:`pygtfs.frequencies.ServiceWindow` instead::

    >>> window = expansion.windows[0]
    >>> window.waiting_range
    (datetime.timedelta(0), datetime.timedelta(seconds=600))

``min_wait`` is zero and ``max_wait`` is ``headway_secs``.  The window does
not contain a departure list, because expanding the headway would turn a
service guarantee into invented exact times.

Boundaries and conflicts
------------------------

Periods are half-open, ``[start_time, end_time)``.  A departure is generated
only while it is strictly before ``end_time``; a period ending exactly when
the next one starts is adjacent, not overlapping.  Use
``end_inclusive=True`` if a caller needs the closing instant as well.

Overlaps for the same trip (within the same feed) are reported as
:class:`pygtfs.frequencies.FrequencyConflict` objects::

    >>> expansion = sched.expand_frequencies()
    >>> expansion.conflicts
    [<FrequencyConflict CITY1: 07:00:00-09:00:00 overlaps 08:00:00-10:00:00>]

Pass ``raise_on_conflict=True`` (or ``strict=True``) to raise
:class:`pygtfs.exceptions.FrequencyConflictError` instead.  Overlapping rows
are never merged or expanded into a single promise.

Filtering
---------

``expand_frequencies`` accepts ``trip_id``, ``feed_id`` and ``service_date``
filters.  ``Schedule``, ``Trip`` and ``Frequency`` all provide
``expand_frequencies``; ``Schedule`` and ``Trip`` additionally provide
``frequency_instances``, ``service_windows`` and ``frequency_conflicts``.
