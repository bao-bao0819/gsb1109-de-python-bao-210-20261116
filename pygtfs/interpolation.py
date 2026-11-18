"""Interpolation of missing stop times from timepoints and shape distances.

GTFS feeds may leave ``arrival_time`` / ``departure_time`` empty for the
intermediate stops of a trip while still providing ``timepoint`` flags and
``shape_dist_traveled`` values. This module derives what the evidence
supports and refuses to invent the rest:

* A stop is an *anchor* when ``timepoint == 1`` (or, for feeds that leave
  timepoint blank, when times are present). Rows with ``timepoint == 0``
  are approximate by definition: their times, if any, are passed through
  but are never used as anchors for other stops.
* An unknown stop bracketed by two anchors gets inclusive time ranges,
  using the previous anchor's departure and the next anchor's arrival or
  departure.
* A single point estimate is produced only when the stop and both anchors
  carry shape distances that are strictly ordered. The estimate lies on
  the timeline running from the previous anchor's departure to the next
  anchor's arrival, so dwell time is consumed at the anchors and every
  emitted time is non-decreasing.
* Equal distances cannot define a speed: a stop sharing its distance with
  the previous row keeps its range and stays unknown instead of dividing
  by zero into an infinite speed.
* Stops without two surrounding anchors, or without usable distances, stay
  unknown.

The functions are read-only; nothing is written back to the schedule.
"""

import datetime
from collections import namedtuple
from bisect import bisect_right

from .gtfs_entities import StopTime

TimeEstimate = namedtuple('TimeEstimate', [
    'trip_id',
    'stop_sequence',
    'stop_id',
    'arrival_time',
    'departure_time',
    'source',
    'arrival_bounds',
    'departure_bounds',
])
TimeEstimate.__doc__ = """Estimated times for one stop of a trip.

``source`` is ``'exact'`` for feed-provided anchor times, ``'provided'``
for non-anchor (``timepoint == 0``) times carried by the feed,
``'interpolated'`` for a derived point estimate, or ``'unknown'`` when the
evidence is insufficient. Bounds, when present, are inclusive
``(lower, upper)`` timedelta tuples.
"""

EXACT = 'exact'
PROVIDED = 'provided'
INTERPOLATED = 'interpolated'
UNKNOWN = 'unknown'


def _seconds(timedelta_value):
    if timedelta_value is None:
        return None
    return timedelta_value.total_seconds()


def _timedelta(seconds):
    return datetime.timedelta(seconds=int(round(seconds)))


def _is_anchor(stop_time):
    """Return True for a row whose times may be treated as confirmed."""
    if stop_time.timepoint == 0:
        return False
    if stop_time.timepoint == 1:
        return stop_time.arrival_time is not None or \
            stop_time.departure_time is not None
    # A blank timepoint is exact only when the feed supplies times.
    return stop_time.arrival_time is not None and \
        stop_time.departure_time is not None


def _anchor_event_times(stop_time):
    """Return inclusive (arrival_seconds, departure_seconds) for an anchor."""
    arrival = _seconds(stop_time.arrival_time)
    departure = _seconds(stop_time.departure_time)
    if arrival is None:
        arrival = departure
    if departure is None:
        departure = arrival
    return arrival, departure


def _valid_bounds(lower, upper):
    if lower is None or upper is None or lower > upper:
        return None
    return lower, upper


def _bounds_as_timedeltas(bounds):
    if bounds is None:
        return None
    lower, upper = bounds
    return _timedelta(lower), _timedelta(upper)


def _has_provided_time(stop_time):
    return stop_time.arrival_time is not None or \
        stop_time.departure_time is not None


def estimate_stop_times(stop_times):
    """Estimate missing times for one list of :class:`StopTime` rows.

    :param stop_times: rows of a single trip, in any order; they are sorted
        by ``stop_sequence`` internally.
    :returns: a list of :class:`TimeEstimate` in stop order. The input rows
        are never modified.
    """
    ordered = sorted(stop_times, key=lambda st: st.stop_sequence)
    anchor_positions = [index for index, st in enumerate(ordered)
                        if _is_anchor(st)]

    estimates = []
    segment_start_time = None
    for index, stop_time in enumerate(ordered):
        if _is_anchor(stop_time):
            estimates.append(TimeEstimate(
                trip_id=stop_time.trip_id,
                stop_sequence=stop_time.stop_sequence,
                stop_id=stop_time.stop_id,
                arrival_time=stop_time.arrival_time,
                departure_time=stop_time.departure_time,
                source=EXACT,
                arrival_bounds=None,
                departure_bounds=None,
            ))
            # Each anchor restarts the travel timeline for the next segment.
            segment_start_time = _anchor_departure(stop_time)
            continue

        previous, following = _surrounding_anchors(
            ordered, anchor_positions, index)

        if previous is None or following is None:
            estimates.append(_passthrough_or_unknown(stop_time))
            continue

        _, prev_departure = _anchor_event_times(previous)
        next_arrival, next_departure = _anchor_event_times(following)
        arrival_bounds = _bounds_as_timedeltas(
            _valid_bounds(prev_departure, next_arrival))
        departure_bounds = _bounds_as_timedeltas(
            _valid_bounds(prev_departure, next_departure))

        # Approximate feed-provided times are kept, never replaced by a
        # guess and never treated as anchors.
        if _has_provided_time(stop_time):
            estimates.append(_passthrough_or_unknown(
                stop_time,
                arrival_bounds=arrival_bounds,
                departure_bounds=departure_bounds,
            ))
            continue

        prior_distance = ordered[index - 1].shape_dist_traveled \
            if index > 0 else None
        point = _interpolate(stop_time, previous, following,
                             prev_departure, next_arrival, prior_distance)
        if point is not None and segment_start_time is not None and \
                point < segment_start_time:
            # Distances out of order would force the clock backwards; the
            # evidence is not strong enough for a point estimate.
            point = None
        if point is None:
            estimates.append(_passthrough_or_unknown(
                stop_time,
                arrival_bounds=arrival_bounds,
                departure_bounds=departure_bounds,
            ))
            continue

        estimates.append(TimeEstimate(
            trip_id=stop_time.trip_id,
            stop_sequence=stop_time.stop_sequence,
            stop_id=stop_time.stop_id,
            arrival_time=point,
            departure_time=point,
            source=INTERPOLATED,
            arrival_bounds=arrival_bounds,
            departure_bounds=departure_bounds,
        ))
        segment_start_time = point

    return estimates


def _anchor_departure(stop_time):
    _, departure = _anchor_event_times(stop_time)
    if departure is None:
        return None
    return _timedelta(departure)


def _surrounding_anchors(ordered, anchor_positions, index):
    """Return the nearest anchor rows before and after ``index``."""
    insertion = bisect_right(anchor_positions, index)
    previous = (ordered[anchor_positions[insertion - 1]]
                if insertion > 0 else None)
    following = (ordered[anchor_positions[insertion]]
                 if insertion < len(anchor_positions) else None)
    return previous, following


def _interpolate(stop_time, previous, following,
                 prev_departure, next_arrival, prior_distance):
    """Linearly interpolate on the anchor departure -> arrival timeline."""
    distance = stop_time.shape_dist_traveled
    prev_distance = previous.shape_dist_traveled
    next_distance = following.shape_dist_traveled
    if distance is None or prev_distance is None or next_distance is None:
        return None
    # A distance repeated on the previous row carries no speed information:
    # interpolating would imply an infinite speed over zero distance.
    if prior_distance is not None and prior_distance == distance:
        return None
    # Strict ordering relative to both anchors.
    if not (prev_distance < distance < next_distance):
        return None
    if prev_departure is None or next_arrival is None:
        return None
    travel_seconds = next_arrival - prev_departure
    if travel_seconds < 0:
        return None
    fraction = (float(distance) - float(prev_distance)) / \
        (float(next_distance) - float(prev_distance))
    return _timedelta(prev_departure + fraction * travel_seconds)


def _passthrough_or_unknown(stop_time, arrival_bounds=None,
                            departure_bounds=None):
    """Keep approximate feed-provided times; otherwise admit ignorance."""
    source = PROVIDED if _has_provided_time(stop_time) else UNKNOWN
    return TimeEstimate(
        trip_id=stop_time.trip_id,
        stop_sequence=stop_time.stop_sequence,
        stop_id=stop_time.stop_id,
        arrival_time=stop_time.arrival_time,
        departure_time=stop_time.departure_time,
        source=source,
        arrival_bounds=arrival_bounds,
        departure_bounds=departure_bounds,
    )


def interpolate_trip(trip):
    """Return :class:`TimeEstimate` objects for a Trip ORM instance."""
    return estimate_stop_times(list(trip.stop_times))


def interpolate_schedule(schedule):
    """Export estimates for every trip in a schedule.

    :returns: a dict keyed by ``(feed_id, trip_id)`` whose values are lists
        of :class:`TimeEstimate` in stop order.
    """
    rows = schedule.session.query(StopTime).order_by(
        StopTime.feed_id, StopTime.trip_id,
        StopTime.stop_sequence).all()
    trips = {}
    for stop_time in rows:
        key = (stop_time.feed_id, stop_time.trip_id)
        trips.setdefault(key, []).append(stop_time)
    return {key: estimate_stop_times(trip_rows)
            for key, trip_rows in trips.items()}
