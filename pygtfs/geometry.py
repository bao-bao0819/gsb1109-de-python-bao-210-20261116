"""Match ordered stops to a route shape.

A circular route can start and end at the same stop, and its shape can pass
through the same intersection more than once. One geometric point can then
correspond to several positions on the shape, so snapping a stop to the
single nearest shape point is not safe.

The helpers in this module keep every projection that is compatible with the
stop order (``stop_sequence``), the shape order (``shape_pt_sequence``) and
the available ``shape_dist_traveled`` anchors, and report a feasible *range*
of shape positions for each stop instead of a hard snapped point.

Mileage regressions, where an anchor is smaller than the previous one, are
only resolved when the shape offers a unique forward continuation (for
example wrapping around a closed loop or continuing after a segment distance
reset). When the continuation is not unique, ``None`` is returned rather
than inventing a seemingly precise traveled distance.

The functions duck-type on the pygtfs ORM objects, but plain objects exposing
the same attributes (``stop_id``, ``stop.stop_lat``, ``shape_pt_lon`` ...)
work just as well.
"""

import math
from collections import namedtuple


EARTH_RADIUS_M = 6371008.8
_EPS_M = 1e-6


class ShapeProjection(namedtuple('ShapeProjection',
                                 ['segment', 'fraction', 'distance',
                                  'offset_m'])):
    """A candidate position of a stop on the shape polyline.

    ``segment`` is the index of the shape segment between shape points
    ``segment`` and ``segment + 1``, ``fraction`` is the position on that
    segment in ``[0, 1]``, ``distance`` is the arc length in meters from the
    start of the shape and ``offset_m`` is the perpendicular distance in
    meters between the stop and the shape.
    """


class StopRange(namedtuple('StopRange',
                           ['stop_sequence', 'stop_id', 'lower', 'upper',
                            'anchored', 'geometric'])):
    """Feasible position of one stop visit on the shape.

    ``lower`` and ``upper`` are arc lengths in meters from the start of the
    shape. ``anchored`` tells whether a ``shape_dist_traveled`` anchor was
    available for the visit and ``geometric`` whether at least one shape
    segment lies within the projection tolerance.
    """


def haversine_meters(lat1, lon1, lat2, lon2):
    """Great-circle distance between two coordinates, in meters."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (math.sin(d_phi / 2.0) ** 2
         + math.cos(phi1) * math.cos(phi2)
         * math.sin(d_lambda / 2.0) ** 2)
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def shape_length(shape_points):
    """Total arc length of an ordered shape polyline, in meters."""
    points = sorted(shape_points, key=lambda p: p.shape_pt_sequence)
    if len(points) < 2:
        return 0.0
    total = 0.0
    for previous, current in zip(points, points[1:]):
        total += haversine_meters(previous.shape_pt_lat,
                                  previous.shape_pt_lon,
                                  current.shape_pt_lat,
                                  current.shape_pt_lon)
    return total


def _cumulative_distances(points):
    cumulative = [0.0]
    for previous, current in zip(points, points[1:]):
        cumulative.append(cumulative[-1] +
                          haversine_meters(previous.shape_pt_lat,
                                           previous.shape_pt_lon,
                                           current.shape_pt_lat,
                                           current.shape_pt_lon))
    return cumulative


def _local_xy(lat, lon, lat0):
    """Equirectangular projection around ``lat0``, in meters."""
    x = EARTH_RADIUS_M * math.radians(lon) * math.cos(math.radians(lat0))
    y = EARTH_RADIUS_M * math.radians(lat)
    return x, y


def _piecewise(x, xs, ys):
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            x0, x1 = xs[i - 1], xs[i]
            if x1 == x0:
                return ys[i]
            return ys[i - 1] + (x - x0) * (ys[i] - ys[i - 1]) / (x1 - x0)
    return ys[-1]


def _calibrations(points, cumulative):
    """Build conversions between arc length (meters) and reported distance.

    Shape points carrying ``shape_dist_traveled`` anchor the reported
    distance unit to the geometric arc length. With fewer than two anchors
    the reported unit is assumed to be meters.
    """
    arc = []
    reported = []
    for i, point in enumerate(points):
        if point.shape_dist_traveled is not None:
            arc.append(cumulative[i])
            reported.append(float(point.shape_dist_traveled))
    if len(arc) >= 2 and arc[-1] > arc[0]:
        def arc_to_reported(distance):
            return _piecewise(distance, arc, reported)

        def reported_to_arc(distance):
            return _piecewise(distance, reported, arc)
    else:
        def arc_to_reported(distance):
            return distance

        def reported_to_arc(distance):
            return distance
    return arc_to_reported, reported_to_arc


def _visit_coordinates(visit):
    stop = getattr(visit, 'stop', None)
    if stop is not None:
        return stop.stop_lat, stop.stop_lon
    return visit.stop_lat, visit.stop_lon


def _visit_anchor(visit):
    if visit.shape_dist_traveled is None:
        return None
    return float(visit.shape_dist_traveled)


def _dedupe(projections):
    distinct = []
    for projection in sorted(projections, key=lambda p: p.distance):
        if not distinct or abs(projection.distance - distinct[-1].distance) > _EPS_M:
            distinct.append(projection)
    return distinct


def project_to_shape(lat, lon, shape_points, tolerance_m=25.0,
                     cumulative=None):
    """All projections of a coordinate onto the shape segments.

    Every segment within ``tolerance_m`` is returned. A self-intersecting
    shape therefore yields several projections for the same coordinate. The
    nearest segment is never picked on its own.
    """
    points = sorted(shape_points, key=lambda p: p.shape_pt_sequence)
    if not points:
        raise ValueError("shape_points must not be empty")
    if cumulative is None:
        cumulative = _cumulative_distances(points)
    lat0 = sum(point.shape_pt_lat for point in points) / len(points)
    point_x, point_y = _local_xy(lat, lon, lat0)
    projections = []
    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]
        start_x, start_y = _local_xy(start.shape_pt_lat,
                                     start.shape_pt_lon, lat0)
        end_x, end_y = _local_xy(end.shape_pt_lat, end.shape_pt_lon, lat0)
        v_x = end_x - start_x
        v_y = end_y - start_y
        denom = v_x * v_x + v_y * v_y
        if denom == 0.0:
            fraction = 0.0
        else:
            fraction = ((point_x - start_x) * v_x
                        + (point_y - start_y) * v_y) / denom
            fraction = max(0.0, min(1.0, fraction))
        projected_x = start_x + fraction * v_x
        projected_y = start_y + fraction * v_y
        offset = math.hypot(point_x - projected_x, point_y - projected_y)
        if offset <= tolerance_m:
            segment_length = cumulative[i + 1] - cumulative[i]
            distance = cumulative[i] + fraction * segment_length
            projections.append(ShapeProjection(i, fraction, distance, offset))
    return projections


def feasible_stop_ranges(stop_times, shape_points, tolerance_m=25.0,
                         anchor_tolerance_m=None):
    """Locate ordered stop visits along the shape.

    Returns one :class:`StopRange` per visit, in ``stop_sequence`` order. The
    ranges honor the stop order, the shape order and any usable
    ``shape_dist_traveled`` anchors. A circular route whose first and last
    visit use the same stop is pinned to the start and end of the shape.
    Stops with no segment inside the tolerance keep the envelope allowed by
    their neighbors instead of being snapped to the nearest segment.
    """
    if anchor_tolerance_m is None:
        anchor_tolerance_m = tolerance_m
    visits = sorted(stop_times, key=lambda visit: visit.stop_sequence)
    points = sorted(shape_points, key=lambda point: point.shape_pt_sequence)
    if not points:
        raise ValueError("shape_points must not be empty")
    if not visits:
        return []

    cumulative = _cumulative_distances(points)
    total_length = cumulative[-1]
    arc_to_reported, reported_to_arc = _calibrations(points, cumulative)
    circular = (len(visits) > 1
                and visits[0].stop_id == visits[-1].stop_id)

    candidate_sets = []
    anchored_flags = []
    geometric_flags = []
    for visit in visits:
        lat, lon = _visit_coordinates(visit)
        projections = project_to_shape(lat, lon, points, tolerance_m,
                                       cumulative)
        geometric_flags.append(bool(projections))
        anchor = _visit_anchor(visit)
        anchored_flags.append(anchor is not None)
        if anchor is not None:
            projections = [
                projection for projection in projections
                if abs(arc_to_reported(projection.distance) - anchor)
                <= anchor_tolerance_m
            ]
        candidate_sets.append(_dedupe(projections))

    if circular:
        # The first visit leaves from the start of the loop and the last
        # visit returns to its end, even though both share a coordinate.
        if candidate_sets[0]:
            candidate_sets[0] = [min(candidate_sets[0],
                                     key=lambda p: p.distance)]
        if candidate_sets[-1]:
            candidate_sets[-1] = [max(candidate_sets[-1],
                                      key=lambda p: p.distance)]

    lower = [0.0] * len(visits)
    upper = [total_length] * len(visits)

    previous = 0.0
    for i, candidates in enumerate(candidate_sets):
        forward = [p.distance for p in candidates
                   if p.distance + _EPS_M >= previous]
        if forward:
            previous = min(forward)
        lower[i] = previous

    following = total_length
    for i in range(len(visits) - 1, -1, -1):
        backward = [p.distance for p in candidate_sets[i]
                    if p.distance <= following + _EPS_M]
        if backward:
            following = max(backward)
        upper[i] = following

    return [
        StopRange(visit.stop_sequence, visit.stop_id, lower[i], upper[i],
                  anchored_flags[i], geometric_flags[i])
        for i, visit in enumerate(visits)
    ]


def _anchor_consistent_continuation(visit, previous_arc, anchor, points,
                                    cumulative, closed_loop, tolerance_m,
                                    anchor_tolerance_m, arc_to_reported):
    """Return the unique projection consistent with a regressed anchor.

    A closed loop can wrap back to its start: at the shared start/end point
    both an arc-zero and an arc-total projection exist, and only one of them
    matches the small anchor of the new lap. ``None`` means no unique
    continuation exists and no precise distance should be reported.
    """
    lat, lon = _visit_coordinates(visit)
    projections = _dedupe(project_to_shape(lat, lon, points, tolerance_m,
                                           cumulative))
    matches = [
        projection for projection in projections
        if abs(arc_to_reported(projection.distance) - anchor)
        <= anchor_tolerance_m
    ]
    forward = [projection for projection in matches
               if projection.distance + _EPS_M >= previous_arc]
    behind = [projection for projection in matches
              if projection.distance + _EPS_M < previous_arc]
    if len(forward) == 1 and not behind:
        return forward[0], False
    if closed_loop and len(behind) == 1 and not forward:
        return behind[0], True
    return None, False


def traveled_distances(stop_times, shape_points, tolerance_m=25.0,
                       anchor_tolerance_m=None):
    """Traveled distance between consecutive stop visits.

    Returns one value per gap (one fewer than the number of visits), in the
    reported distance unit when anchors are available and in meters
    otherwise. A gap is ``None`` when a mileage regression cannot be matched
    to a unique forward continuation.
    """
    if anchor_tolerance_m is None:
        anchor_tolerance_m = tolerance_m
    visits = sorted(stop_times, key=lambda visit: visit.stop_sequence)
    points = sorted(shape_points, key=lambda point: point.shape_pt_sequence)
    if not points:
        raise ValueError("shape_points must not be empty")
    if len(visits) < 2:
        return []

    cumulative = _cumulative_distances(points)
    total_length = cumulative[-1]
    arc_to_reported, reported_to_arc = _calibrations(points, cumulative)
    closed_loop = haversine_meters(points[0].shape_pt_lat,
                                   points[0].shape_pt_lon,
                                   points[-1].shape_pt_lat,
                                   points[-1].shape_pt_lon) <= tolerance_m
    ranges = feasible_stop_ranges(visits, points, tolerance_m,
                                  anchor_tolerance_m)
    distances = []
    for i in range(len(visits) - 1):
        previous_visit = visits[i]
        visit = visits[i + 1]
        previous_anchor = _visit_anchor(previous_visit)
        anchor = _visit_anchor(visit)

        if previous_anchor is None or anchor is None:
            previous_range = ranges[i]
            current_range = ranges[i + 1]
            if (previous_range.upper - previous_range.lower > _EPS_M
                    or current_range.upper - current_range.lower > _EPS_M):
                distances.append(None)
                continue
            if current_range.lower + _EPS_M < previous_range.upper:
                distances.append(None)
                continue
            distances.append(arc_to_reported(current_range.lower)
                             - arc_to_reported(previous_range.upper))
            continue

        if anchor + _EPS_M >= previous_anchor:
            distances.append(anchor - previous_anchor)
            continue

        # Mileage regression: only resolve it with a unique continuation.
        previous_arc = reported_to_arc(previous_anchor)
        projection, wraps = _anchor_consistent_continuation(
            visit, previous_arc, anchor, points, cumulative, closed_loop,
            tolerance_m, anchor_tolerance_m, arc_to_reported)
        if projection is None:
            distances.append(None)
            continue
        if wraps:
            reported_total = arc_to_reported(total_length)
            distances.append((reported_total - previous_anchor) + anchor)
        else:
            distances.append(arc_to_reported(projection.distance)
                             - previous_anchor)
    return distances


def locate_trip(trip, tolerance_m=25.0, anchor_tolerance_m=None):
    """Convenience wrapper around :func:`feasible_stop_ranges` for a trip."""
    visits = sorted(trip.stop_times,
                    key=lambda visit: visit.stop_sequence)
    points = sorted(trip.shape_points,
                    key=lambda point: point.shape_pt_sequence)
    return feasible_stop_ranges(visits, points, tolerance_m,
                                anchor_tolerance_m)
