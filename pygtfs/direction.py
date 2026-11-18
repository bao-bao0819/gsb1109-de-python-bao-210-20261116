"""Direction grouping for trips that lack ``direction_id``.

Legacy feeds often leave ``trips.direction_id`` empty while running several
patterns under the same route.  The first and last stop may even be identical
(circular lines, short turns, detour branches), so grouping by the terminals
or by the route short name silently merges patterns that do not run in the
same direction.

The tools in this module group the trips of a route using only two pieces of
evidence:

* the ordered stop sequence (``stop_times.txt``), compared through the
  longest common subsequence in both directions;
* the geometric shape (``shapes.txt``), resampled to a fixed number of
  points and compared both head-to-head and head-to-tail.

Every pairwise comparison returns one of three verdicts:

* :data:`SAME` -- the two trips are *provably* running in the same
  direction;
* :data:`OPPOSITE` -- the two trips are *provably* running in opposite
  directions;
* :data:`UNKNOWN` -- the evidence is missing, weak or contradictory, so the
  pair is left undecided.

Only :data:`SAME` pairs are merged.  Direction anchors (a trip whose
direction is known, labelled ``0`` or ``1``) label the group they belong to
and the label propagates across :data:`OPPOSITE` edges.  Groups that cannot
be reached from an anchor stay unassigned instead of being force-merged,
which keeps detour branches separate from the trunk line.
"""

import math
from collections import namedtuple


SAME = "same"
OPPOSITE = "opposite"
UNKNOWN = "unknown"

#: Direction id used for groups that no anchor can reach.
UNASSIGNED = None

_EARTH_RADIUS_M = 6371009.0
_DEFAULT_STOP_SAME_THRESHOLD = 0.8
_DEFAULT_STOP_MARGIN = 0.15
_DEFAULT_SHAPE_TOLERANCE = 0.08
_SHAPE_SAMPLE_POINTS = 48


class DirectionConflictError(ValueError):
    """Raised when anchors imply contradictory direction ids.

    This happens either because two anchors with different direction ids sit
    in a group whose trips are provably running in the same direction, or
    because propagating labels across opposite edges reaches a group with an
    incompatible label.
    """


#: A single trip's evidence: an ordered list of stop ids and, optionally,
#: an ordered list of ``(lat, lon)`` shape points.
TripPattern = namedtuple("TripPattern", ["stops", "shape"])


class PairVerdict(namedtuple("PairVerdict",
                             ["trip_a", "trip_b", "verdict", "reasons"])):
    """The verdict for one pair of trips.

    ``reasons`` contains human readable summaries of the individual pieces of
    evidence (stop sequence and/or shape) that lead to the verdict.
    """


class DirectionGroup(namedtuple("DirectionGroup",
                                ["group_id", "trip_ids",
                                 "direction_id", "reason"])):
    """A group of trips provably running in the same direction.

    ``direction_id`` is :data:`UNASSIGNED` when no anchor can reach the
    group.  ``reason`` explains how the direction id was derived.
    """


class AssignmentChange(namedtuple("AssignmentChange",
                                  ["trip_id", "old_direction_id",
                                   "new_direction_id", "reason"])):
    """A trip whose direction assignment changed between two groupings."""


class DirectionResult(object):
    """The outcome of :meth:`DirectionGrouper.group`.

    Attributes:
        groups: tuple of :class:`DirectionGroup`, sorted by group id.
        pair_verdicts: dict keyed by ``(trip_a, trip_b)`` tuples (trip ids
            sorted lexicographically) holding :class:`PairVerdict` objects.
        assignments: dict mapping every trip id to ``0``, ``1`` or
            :data:`UNASSIGNED`.
    """

    def __init__(self, groups, pair_verdicts, assignments):
        self.groups = tuple(groups)
        self.pair_verdicts = dict(pair_verdicts)
        self.assignments = dict(assignments)

    @property
    def unassigned_trip_ids(self):
        """Trip ids that no anchor could label."""
        return sorted(trip_id for trip_id, direction_id
                      in self.assignments.items()
                      if direction_id is UNASSIGNED)

    def group_of(self, trip_id):
        """Return the :class:`DirectionGroup` containing ``trip_id``."""
        for group in self.groups:
            if trip_id in group.trip_ids:
                return group
        raise KeyError(trip_id)

    def explain_changes(self, previous):
        """Compare these assignments with an earlier grouping.

        ``previous`` may be another :class:`DirectionResult` or a plain
        ``trip_id -> direction_id`` dict (missing entries default to
        :data:`UNASSIGNED`).  The returned list of
        :class:`AssignmentChange` objects covers every trip whose direction
        id changed, including trips that went from unassigned to labelled
        after an anchor was added.
        """
        if isinstance(previous, DirectionResult):
            old_assignments = previous.assignments
        else:
            old_assignments = dict(previous)

        changes = []
        for trip_id in sorted(self.assignments):
            old = old_assignments.get(trip_id, UNASSIGNED)
            new = self.assignments[trip_id]
            if old != new:
                reason = self.group_of(trip_id).reason
                changes.append(AssignmentChange(trip_id, old, new, reason))
        return changes


def _haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points, in meters."""
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2.0) ** 2
         + math.cos(lat1_r) * math.cos(lat2_r)
         * math.sin(d_lon / 2.0) ** 2)
    return 2.0 * _EARTH_RADIUS_M * math.asin(math.sqrt(a))


def _polyline_length_m(points):
    return sum(_haversine_m(points[i][0], points[i][1],
                            points[i + 1][0], points[i + 1][1])
               for i in range(len(points) - 1))


def _resample_polyline(points, n_points):
    """Resample a polyline to ``n_points`` equally spaced points."""
    if len(points) < 2:
        return list(points)

    segments = []
    cumulative = [0.0]
    for i in range(len(points) - 1):
        length = _haversine_m(points[i][0], points[i][1],
                              points[i + 1][0], points[i + 1][1])
        segments.append(length)
        cumulative.append(cumulative[-1] + length)
    total = cumulative[-1]
    if total <= 0.0:
        return [points[0]] * n_points

    resampled = []
    segment_index = 0
    for step in range(n_points):
        target = total * step / float(n_points - 1)
        while (segment_index < len(segments) - 1
               and cumulative[segment_index + 1] < target):
            segment_index += 1
        segment_length = segments[segment_index]
        if segment_length <= 0.0:
            ratio = 0.0
        else:
            ratio = ((target - cumulative[segment_index])
                     / segment_length)
        start = points[segment_index]
        end = points[segment_index + 1]
        resampled.append((
            start[0] + (end[0] - start[0]) * ratio,
            start[1] + (end[1] - start[1]) * ratio,
        ))
    return resampled


def _lcs_length(seq_a, seq_b):
    """Length of the longest common subsequence of two sequences."""
    if not seq_a or not seq_b:
        return 0
    previous = [0] * (len(seq_b) + 1)
    for item_a in seq_a:
        current = [0] * (len(seq_b) + 1)
        for j, item_b in enumerate(seq_b, start=1):
            if item_a == item_b:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])
        previous = current
    return previous[-1]


def compare_stop_sequences(stops_a, stops_b,
                           same_threshold=_DEFAULT_STOP_SAME_THRESHOLD,
                           margin=_DEFAULT_STOP_MARGIN):
    """Compare two ordered stop sequences.

    Returns a :class:`PairVerdict`-like ``(verdict, reasons)`` tuple built
    from the forward and reverse LCS ratios, or ``None`` when either
    sequence is empty.  The reverse ratio measures how much of the longer
    sequence matches when one trip is travelled backwards, which is what
    distinguishes the two directions even when both terminals are
    identical.
    """
    if not stops_a or not stops_b:
        return None

    longest = float(max(len(stops_a), len(stops_b)))
    forward = _lcs_length(stops_a, stops_b) / longest
    reverse = _lcs_length(stops_a, list(reversed(stops_b))) / longest

    reason = ("stops: forward overlap {0:.2f}, reverse overlap {1:.2f}"
              .format(forward, reverse))
    if forward >= same_threshold and forward - reverse >= margin:
        return SAME, [reason]
    if reverse >= same_threshold and reverse - forward >= margin:
        return OPPOSITE, [reason]
    return UNKNOWN, [reason]


def compare_shapes(shape_a, shape_b,
                   tolerance=_DEFAULT_SHAPE_TOLERANCE,
                   n_points=_SHAPE_SAMPLE_POINTS):
    """Compare two shape polylines.

    Both polylines are resampled to the same number of equally spaced
    points.  The mean point distance is taken head-to-head (same direction)
    and head-to-tail (opposite directions), normalized by the mean polyline
    length.  Returns ``None`` when either shape is missing or degenerate.
    """
    if not shape_a or not shape_b or len(shape_a) < 2 or len(shape_b) < 2:
        return None

    sampled_a = _resample_polyline(shape_a, n_points)
    sampled_b = _resample_polyline(shape_b, n_points)
    length_scale = (
        _polyline_length_m(shape_a) + _polyline_length_m(shape_b)
    ) / 2.0
    if length_scale <= 0.0:
        return None

    forward_distance = sum(
        _haversine_m(sampled_a[i][0], sampled_a[i][1],
                     sampled_b[i][0], sampled_b[i][1])
        for i in range(n_points)
    ) / n_points / length_scale
    reverse_distance = sum(
        _haversine_m(sampled_a[i][0], sampled_a[i][1],
                     sampled_b[n_points - 1 - i][0],
                     sampled_b[n_points - 1 - i][1])
        for i in range(n_points)
    ) / n_points / length_scale

    reason = ("shape: forward distance {0:.3f}, reverse distance {1:.3f} "
              "(fraction of route length)"
              .format(forward_distance, reverse_distance))
    # The favoured orientation must be clearly better, not merely equal.
    if (forward_distance <= tolerance
            and forward_distance <= reverse_distance * 0.75):
        return SAME, [reason]
    if (reverse_distance <= tolerance
            and reverse_distance <= forward_distance * 0.75):
        return OPPOSITE, [reason]
    return UNKNOWN, [reason]


def _combine_evidence(evidence):
    """Merge per-source verdicts into one provable verdict."""
    decisive = {verdict for verdict, _ in evidence
                if verdict != UNKNOWN}
    reasons = []
    for _, source_reasons in evidence:
        reasons.extend(source_reasons)
    if decisive == {SAME}:
        return SAME, reasons
    if decisive == {OPPOSITE}:
        return OPPOSITE, reasons
    # No evidence, weak evidence or contradictory sources: stay undecided.
    return UNKNOWN, reasons


class DirectionGrouper(object):
    """Groups trips of one route into provable direction groups.

    Args:
        patterns: mapping of ``trip_id`` to :class:`TripPattern`.
        stop_same_threshold: minimum LCS ratio for a stop-sequence match.
        stop_margin: minimum difference between the winning and the losing
            orientation, preventing near-symmetric patterns from being
            labelled.
        shape_tolerance: maximum mean point distance, as a fraction of the
            mean polyline length, for a shape match.
    """

    def __init__(self, patterns,
                 stop_same_threshold=_DEFAULT_STOP_SAME_THRESHOLD,
                 stop_margin=_DEFAULT_STOP_MARGIN,
                 shape_tolerance=_DEFAULT_SHAPE_TOLERANCE):
        self._patterns = dict(patterns)
        self._stop_same_threshold = stop_same_threshold
        self._stop_margin = stop_margin
        self._shape_tolerance = shape_tolerance

    def _compare(self, trip_a, trip_b):
        pattern_a = self._patterns[trip_a]
        pattern_b = self._patterns[trip_b]
        evidence = []

        stop_evidence = compare_stop_sequences(
            pattern_a.stops, pattern_b.stops,
            same_threshold=self._stop_same_threshold,
            margin=self._stop_margin)
        if stop_evidence is not None:
            evidence.append(stop_evidence)

        shape_evidence = compare_shapes(
            pattern_a.shape, pattern_b.shape,
            tolerance=self._shape_tolerance)
        if shape_evidence is not None:
            evidence.append(shape_evidence)

        verdict, reasons = _combine_evidence(evidence)
        return PairVerdict(trip_a, trip_b, verdict, tuple(reasons))

    def group(self, anchors=None):
        """Group the trips and propagate direction anchors.

        Args:
            anchors: mapping of ``trip_id`` to ``0`` or ``1``.  Trips in the
                same provable group must agree, otherwise
                :class:`DirectionConflictError` is raised.

        Returns:
            DirectionResult
        """
        anchors = dict(anchors or {})
        trip_ids = sorted(self._patterns)

        for trip_id, direction_id in anchors.items():
            if trip_id not in self._patterns:
                raise KeyError("anchor trip {0!r} is not part of the "
                               "grouping".format(trip_id))
            if direction_id not in (0, 1):
                raise ValueError(
                    "anchor direction must be 0 or 1, got {0!r} for trip "
                    "{1!r}".format(direction_id, trip_id))

        pair_verdicts = {}
        for index, trip_a in enumerate(trip_ids):
            for trip_b in trip_ids[index + 1:]:
                pair_verdicts[(trip_a, trip_b)] = self._compare(
                    trip_a, trip_b)

        parent = dict((trip_id, trip_id) for trip_id in trip_ids)

        def find(trip_id):
            root = trip_id
            while parent[root] != root:
                root = parent[root]
            while parent[trip_id] != root:
                parent[trip_id], trip_id = root, parent[trip_id]
            return root

        def union(trip_a, trip_b):
            root_a = find(trip_a)
            root_b = find(trip_b)
            if root_a != root_b:
                parent[root_b] = root_a

        # Only provably same-direction trips may end up in one group.
        for (trip_a, trip_b), verdict in pair_verdicts.items():
            if verdict.verdict == SAME:
                union(trip_a, trip_b)

        roots = dict((trip_id, find(trip_id)) for trip_id in trip_ids)
        root_to_group_id = {}
        groups = []
        group_members = {}
        for trip_id in trip_ids:
            root = roots[trip_id]
            if root not in root_to_group_id:
                group_id = len(groups)
                root_to_group_id[root] = group_id
                group_members[group_id] = []
                groups.append(group_id)
            group_members[root_to_group_id[root]].append(trip_id)

        # Opposite edges connect, but never merge, groups.
        opposite_edges = set()
        for (trip_a, trip_b), verdict in pair_verdicts.items():
            if verdict.verdict == OPPOSITE:
                group_a = root_to_group_id[roots[trip_a]]
                group_b = root_to_group_id[roots[trip_b]]
                if group_a != group_b:
                    opposite_edges.add(
                        (min(group_a, group_b), max(group_a, group_b)))

        labels = dict((group_id, UNASSIGNED) for group_id in groups)
        label_reason = {}

        # Seed labels from the anchors.
        group_anchors = {}
        for trip_id, direction_id in anchors.items():
            group_id = root_to_group_id[roots[trip_id]]
            group_anchors.setdefault(group_id, []).append(
                (trip_id, direction_id))
        for group_id, seeded in group_anchors.items():
            direction_ids = set(direction_id for _, direction_id in seeded)
            if len(direction_ids) > 1:
                raise DirectionConflictError(
                    "anchors {0} disagree inside group {1} whose trips are "
                    "provably running in the same direction".format(
                        seeded, group_members[group_id]))
            direction_id = direction_ids.pop()
            labels[group_id] = direction_id
            anchor_text = ", ".join(
                "{0}={1}".format(trip_id, direction_id)
                for trip_id, _ in seeded)
            label_reason[group_id] = "anchored by {0}".format(anchor_text)

        # Propagate labels across the opposite edges.
        neighbors = dict((group_id, []) for group_id in groups)
        for group_a, group_b in opposite_edges:
            neighbors[group_a].append(group_b)
            neighbors[group_b].append(group_a)

        pending = [group_id for group_id in groups
                   if labels[group_id] is not UNASSIGNED]
        while pending:
            current = pending.pop()
            for neighbor in neighbors[current]:
                expected = 1 - labels[current]
                if labels[neighbor] is UNASSIGNED:
                    labels[neighbor] = expected
                    label_reason[neighbor] = (
                        "opposite group {0} ({1})".format(
                            current, label_reason[current]))
                    pending.append(neighbor)
                elif labels[neighbor] != expected:
                    raise DirectionConflictError(
                        "group {0} cannot be both direction {1} and {2}; "
                        "anchors and opposite edges disagree".format(
                            group_members[neighbor], labels[neighbor],
                            expected))

        direction_groups = []
        assignments = {}
        for group_id in groups:
            members = group_members[group_id]
            direction_id = labels[group_id]
            if direction_id is UNASSIGNED:
                reason = "no anchor reachable; left unassigned"
            else:
                reason = label_reason[group_id]
            direction_groups.append(DirectionGroup(
                group_id=group_id,
                trip_ids=tuple(members),
                direction_id=direction_id,
                reason=reason))
            for trip_id in members:
                assignments[trip_id] = direction_id

        return DirectionResult(direction_groups, pair_verdicts, assignments)


def anchors_from_trips(trips):
    """Build an anchor mapping from trips that already carry direction ids.

    This is the typical starting point when a new feed provides
    ``direction_id`` and an old feed does not: anchor the old trips with the
    labelled ones and inspect :meth:`DirectionResult.explain_changes` to see
    which trips moved.
    """
    anchors = {}
    for trip in trips:
        if trip.direction_id is not None:
            anchors[trip.trip_id] = trip.direction_id
    return anchors


def group_route_directions(route, anchors=None, **kwargs):
    """Group the trips of an ORM :class:`~pygtfs.gtfs_entities.Route`.

    Stop sequences come from each trip's ``stop_times`` and shapes from its
    ``shape_points``; trips without either simply contribute no evidence of
    that kind.  When the ``shape_points`` relationship is empty although the
    trip carries a ``shape_id`` (for example when shape points were inserted
    directly without the loader's secondary-table mapping), they are looked
    up by ``shape_id`` within the same feed instead.

    Args:
        route: a :class:`pygtfs.gtfs_entities.Route` instance.
        anchors: optional ``trip_id -> 0/1`` mapping.  Build one from
            labelled trips with :func:`anchors_from_trips`.
        **kwargs: forwarded to :class:`DirectionGrouper`.

    Returns:
        DirectionResult
    """
    from sqlalchemy.orm import object_session

    from .gtfs_entities import ShapePoint

    patterns = {}
    for trip in route.trips:
        stops = [stop_time.stop_id for stop_time
                 in sorted(trip.stop_times,
                           key=lambda stop_time: stop_time.stop_sequence)]
        shape_points = list(trip.shape_points)
        if not shape_points and trip.shape_id:
            session = object_session(trip)
            if session is not None:
                shape_points = (
                    session.query(ShapePoint)
                    .filter(ShapePoint.feed_id == trip.feed_id)
                    .filter(ShapePoint.shape_id == trip.shape_id)
                    .all()
                )
        shape = [(point.shape_pt_lat, point.shape_pt_lon) for point
                 in sorted(shape_points,
                           key=lambda point: point.shape_pt_sequence)]
        patterns[trip.trip_id] = TripPattern(stops=stops, shape=shape)

    grouper = DirectionGrouper(patterns, **kwargs)
    return grouper.group(anchors)
