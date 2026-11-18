"""Group LAS points into emission pulses.

A LAS point does not directly reference the pulse that produced it. What a
point does carry is a GPS time, a flight line (``point_source_id``), a scanner
channel (``scanner_channel`` in point formats 6 and later), a return number,
and a declaration of how many returns the pulse was expected to have
(``number_of_returns``).

Those fields are enough to define which points *may* belong to the same pulse,
but they do not always define one unique grouping. GPS times are quantised and
may be rounded differently by different producers, two channels can record the
same timestamp, and a file may contain only a filtered subset of the original
points. This module therefore distinguishes between:

* groups that are proven: they are the same in every minimum-pulse grouping
  compatible with the data;
* ambiguous clusters: several minimum-pulse groupings remain possible, so only
  the relations shared by all of them are reported (often no full group at
  all);
* points whose return declaration cannot be interpreted.

Input order is never used to decide a grouping. Point occurrence identity is
preserved through :attr:`PulseGroup.point_indices` (positions in the input) and
:attr:`PulseGroup.original_indices` (caller supplied identities).

The main entry point is :func:`group_pulses`.
"""

import bisect
import math
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Tuple

import numpy as np

# Exact search is exponential; beyond this many points in one time component
# the result is reported as unresolvable instead of risking a pathological run.
_MAX_EXACT_COMPONENT_SIZE = 64

__all__ = [
    "AmbiguousCluster",
    "InvalidPoint",
    "InvalidReason",
    "MissingReason",
    "MissingReturn",
    "PulseGroup",
    "PulseGrouping",
    "build_pulse_groups",
    "group_into_pulses",
    "group_pulses",
]


class MissingReason(Enum):
    """Why a declared return is absent from a pulse group."""

    KNOWN_FILTERED = "known_filtered"
    """An excluded point proves that the return existed and was filtered out."""

    UNKNOWN = "unknown"
    """No excluded point is available to explain the absence."""


class InvalidReason(Enum):
    """Why a point could not take part in pulse grouping."""

    INVALID_RETURN_NUMBER = "invalid_return_number"
    INVALID_NUMBER_OF_RETURNS = "invalid_number_of_returns"
    RETURN_EXCEEDS_DECLARATION = "return_exceeds_declaration"
    NON_FINITE_TIME = "non_finite_time"


@dataclass(frozen=True)
class MissingReturn:
    """A declared return number that has no point in a pulse group.

    Attributes
    ----------
    return_number:
        The missing return number, starting at 1.
    reason:
        ``KNOWN_FILTERED`` when an excluded point with matching identity fields
        proves that the return existed and was removed, otherwise ``UNKNOWN``.
    filtered_indices:
        Identities of the excluded points that explain a known filtering, in
        the identity space supplied to :func:`group_pulses`.
    """

    return_number: int
    reason: MissingReason
    filtered_indices: Tuple[Any, ...] = ()


@dataclass(frozen=True)
class InvalidPoint:
    """A point that cannot be assigned a return number."""

    point_index: int
    original_index: Any
    reason: InvalidReason


@dataclass(frozen=True)
class PulseGroup:
    """A proven pulse group.

    The point order is deterministic but does not reflect how the points were
    stored in the file: members are ordered by GPS time, return number, then
    input position.
    """

    pulse_id: int
    flight_line: int
    channel: int
    number_of_returns: int
    point_indices: Tuple[int, ...]
    original_indices: Tuple[Any, ...]
    return_numbers: Tuple[int, ...]
    gps_times: Tuple[float, ...]
    gps_time: float
    time_span: float
    missing: Tuple[MissingReturn, ...] = ()

    @property
    def point_count(self) -> int:
        return len(self.point_indices)

    @property
    def present_returns(self) -> Tuple[int, ...]:
        return self.return_numbers

    @property
    def missing_returns(self) -> Tuple[int, ...]:
        return tuple(item.return_number for item in self.missing)

    @property
    def is_complete(self) -> bool:
        return not self.missing

    @property
    def has_unknown_missing(self) -> bool:
        return any(item.reason is MissingReason.UNKNOWN for item in self.missing)

    @property
    def expected_returns(self) -> Tuple[int, ...]:
        return tuple(range(1, self.number_of_returns + 1))

    def member_for_return(self, return_number: int) -> Optional[int]:
        """Return the input position of the point claiming ``return_number``.

        ``None`` is returned when that return is missing.
        """
        for point_index, number in zip(self.point_indices, self.return_numbers):
            if number == return_number:
                return point_index
        return None


@dataclass(frozen=True)
class AmbiguousCluster:
    """Points whose pulse assignment cannot be uniquely established.

    Every listed point belongs to exactly one pulse, but at least two
    minimum-pulse groupings are compatible with the data. ``candidate_pulses``
    contains the pulses seen in the enumerated groupings; it is not a proposed
    solution. ``common_groups`` contains complete pulse membership sets shared
    by *all* enumerated groupings (usually empty). ``common_pairs`` contains
    individual point pairs that are together in every enumerated grouping,
    which is the finest relation the data supports.

    When ``exhaustive`` is false, the search had to stop before all minimum
    groupings were enumerated (or the component was too large to enumerate).
    In that case no group or pair is claimed as common, because the result
    cannot be proven.
    """

    flight_line: int
    channel: int
    number_of_returns: int
    point_indices: Tuple[int, ...]
    original_indices: Tuple[Any, ...]
    return_numbers: Tuple[int, ...]
    gps_times: Tuple[float, ...]
    minimum_pulse_count: int
    exhaustive: bool
    grouping_count: int
    candidate_pulses: Tuple[Tuple[int, ...], ...] = ()
    common_groups: Tuple[Tuple[int, ...], ...] = ()
    common_pairs: Tuple[Tuple[int, int], ...] = ()

    @property
    def point_count(self) -> int:
        return len(self.point_indices)


@dataclass(frozen=True)
class PulseGrouping:
    """Result of :func:`group_pulses`."""

    groups: Tuple[PulseGroup, ...]
    ambiguous_clusters: Tuple[AmbiguousCluster, ...]
    invalid_points: Tuple[InvalidPoint, ...]
    time_precision: float
    source_point_count: int

    @property
    def ambiguous(self) -> bool:
        return bool(self.ambiguous_clusters)

    @property
    def proven_pulse_count(self) -> int:
        return len(self.groups)

    @property
    def minimum_pulse_count(self) -> int:
        """Minimum number of pulses over all minimum groupings.

        Invalid points are not counted, and an unresolved cluster that could
        not be enumerated is counted at its proven lower bound.
        """
        return len(self.groups) + sum(
            cluster.minimum_pulse_count for cluster in self.ambiguous_clusters
        )

    @property
    def complete(self) -> bool:
        """True when all points are assigned to proven, complete pulses."""
        return not self.ambiguous_clusters and not self.invalid_points and all(
            group.is_complete for group in self.groups
        )

    def __iter__(self):
        return iter(self.groups)

    def __len__(self) -> int:
        return len(self.groups)

    def group_for_point(self, point_index: int) -> Optional[PulseGroup]:
        """Return the proven group containing an input position, if any."""
        for group in self.groups:
            if point_index in group.point_indices:
                return group
        return None

    def pulse_ids(self) -> np.ndarray:
        """Return a per-input-point array of proven pulse ids.

        ``-1`` marks a point in an ambiguous cluster and ``-2`` an invalid
        point. The array length is ``source_point_count``; occurrence
        positions are preserved.
        """
        ids = np.full(self.source_point_count, -1, dtype=np.int64)
        for point in self.invalid_points:
            ids[point.point_index] = -2
        for group in self.groups:
            ids[list(group.point_indices)] = group.pulse_id
        return ids


@dataclass
class _Point:
    index: int
    identity: Any
    flight_line: int
    channel: int
    return_number: int
    number_of_returns: int
    gps_time: float
    valid: bool = True
    invalid_reason: Optional[InvalidReason] = None


def _first_set(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _dimension_from_source(source, name):
    try:
        value = source[name]
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        try:
            value = getattr(source, name)
        except (AttributeError, ValueError):
            raise KeyError(name) from None
    return value


def _extract_arrays(source, overrides):
    names = (
        "gps_time",
        "return_number",
        "number_of_returns",
        "point_source_id",
        "scanner_channel",
    )
    arrays = {}
    for name in names:
        override = overrides.get(name)
        if override is not None:
            arrays[name] = np.asarray(override)
            continue
        try:
            arrays[name] = np.asarray(_dimension_from_source(source, name))
        except KeyError:
            if name in ("point_source_id", "scanner_channel"):
                arrays[name] = None
            else:
                raise ValueError(
                    f"Pulse grouping needs the '{name}' dimension; "
                    "pass it explicitly if the source does not provide it"
                ) from None
    lengths = {len(value) for value in arrays.values() if value is not None}
    if len(lengths) != 1:
        raise ValueError(
            "All pulse grouping dimensions must have the same length, "
            f"got lengths {sorted(lengths)}"
        )
    return arrays


def _as_point_list(
    source,
    *,
    overrides,
    identity,
):
    arrays = _extract_arrays(source, overrides)
    count = len(arrays["gps_time"])
    if identity is None:
        identities = list(range(count))
    else:
        identities = np.asarray(identity).tolist()
        if len(identities) != count:
            raise ValueError("identity must have one value per point")

    if arrays["point_source_id"] is None:
        flight_lines = [0] * count
    else:
        flight_lines = np.asarray(arrays["point_source_id"]).tolist()
    if arrays["scanner_channel"] is None:
        channels = [0] * count
    else:
        channels = np.asarray(arrays["scanner_channel"]).tolist()

    gps_times = np.asarray(arrays["gps_time"], dtype=np.float64)
    return_numbers = np.asarray(arrays["return_number"]).tolist()
    totals = np.asarray(arrays["number_of_returns"]).tolist()

    points = []
    for index in range(count):
        time_value = float(gps_times[index])
        return_number = int(return_numbers[index])
        total = int(totals[index])
        point = _Point(
            index=index,
            identity=identities[index],
            flight_line=int(flight_lines[index]),
            channel=int(channels[index]),
            return_number=return_number,
            number_of_returns=total,
            gps_time=time_value,
        )
        if not math.isfinite(time_value):
            point.valid = False
            point.invalid_reason = InvalidReason.NON_FINITE_TIME
        elif return_number < 1:
            point.valid = False
            point.invalid_reason = InvalidReason.INVALID_RETURN_NUMBER
        elif total < 1:
            point.valid = False
            point.invalid_reason = InvalidReason.INVALID_NUMBER_OF_RETURNS
        elif return_number > total:
            point.valid = False
            point.invalid_reason = InvalidReason.RETURN_EXCEEDS_DECLARATION
        points.append(point)
    return points


def _time_components(points, precision):
    """Split points into components that cannot share a pulse across them."""
    ordered = sorted(points, key=lambda point: (point.gps_time, point.index))
    components = []
    current = []
    current_max = None
    for point in ordered:
        if current and point.gps_time - current_max > precision:
            components.append(current)
            current = []
        current.append(point)
        current_max = point.gps_time
    if current:
        components.append(current)
    return components


def _enumerate_min_partitions(points, precision, max_solutions, max_steps):
    """Enumerate all minimum-cardinality valid partitions of ``points``.

    Returns ``(solutions, exhaustive)``. A solution is a frozenset of
    frozensets of input point indices. The minimum number of groups is the
    maximum per-return multiplicity, because two points with the same return
    number can never share a pulse; extra groups are added only when the time
    precision makes that lower bound infeasible.
    """
    return_counts = Counter(point.return_number for point in points)
    order = sorted(
        points,
        key=lambda point: (
            -return_counts[point.return_number],
            point.gps_time,
            point.return_number,
            point.index,
        ),
    )
    lower_bound = max(return_counts.values(), default=0)
    upper_bound = len(points)
    steps = 0
    exhausted = True

    def search_for_target(target):
        nonlocal steps, exhausted
        solutions = set()
        groups = []

        def search(position):
            nonlocal steps, exhausted
            steps += 1
            if steps > max_steps or len(solutions) > max_solutions:
                exhausted = False
                return
            if position == len(order):
                if len(groups) == target:
                    solutions.add(
                        frozenset(frozenset(group["indices"]) for group in groups)
                    )
                return
            if len(groups) + (len(order) - position) < target:
                return
            point = order[position]
            for group in groups:
                if point.return_number in group["returns"]:
                    continue
                new_min = min(group["time_min"], point.gps_time)
                new_max = max(group["time_max"], point.gps_time)
                if new_max - new_min > precision:
                    continue
                group["indices"].append(point.index)
                group["returns"].add(point.return_number)
                old_min, old_max = group["time_min"], group["time_max"]
                group["time_min"], group["time_max"] = new_min, new_max
                search(position + 1)
                group["time_min"], group["time_max"] = old_min, old_max
                group["returns"].remove(point.return_number)
                group["indices"].pop()
                if not exhausted:
                    return
            if len(groups) < target:
                groups.append(
                    {
                        "indices": [point.index],
                        "returns": {point.return_number},
                        "time_min": point.gps_time,
                        "time_max": point.gps_time,
                    }
                )
                search(position + 1)
                groups.pop()
            if not exhausted:
                return

        search(0)
        return solutions

    for target in range(lower_bound, upper_bound + 1):
        solutions = search_for_target(target)
        if not exhausted:
            return set(), False
        if solutions:
            return solutions, True
    return set(), True


def _min_partitions(points, precision, max_solutions, max_steps):
    """Return minimum partitions plus a fast-path flag.

    The return value is ``(solutions, exhaustive)``. ``solutions`` is a set of
    frozensets of frozensets of input indices.
    """
    if not points:
        return {frozenset()}, True

    counts = Counter(point.return_number for point in points)
    lower_bound = max(counts.values())

    if lower_bound == len(points):
        return {frozenset(frozenset((point.index,)) for point in points)}, True

    times = [point.gps_time for point in points]
    if (
        lower_bound == 1
        and len(counts) == len(points)
        and max(times) - min(times) <= precision
    ):
        return {frozenset((frozenset(point.index for point in points),))}, True

    if len(points) > _MAX_EXACT_COMPONENT_SIZE:
        return set(), False

    solutions, exhaustive = _enumerate_min_partitions(
        points,
        precision,
        max_solutions=max_solutions,
        max_steps=max_steps,
    )
    return solutions, exhaustive


def _common_relations(solutions):
    """Return groups and pairs present in every solution."""
    if not solutions:
        return set(), set()
    group_sets = [set(solution) for solution in solutions]
    common_groups = set.intersection(*group_sets) if group_sets else set()

    pair_sets = []
    pair_count = 0
    for solution in solutions:
        pairs = set()
        for group in solution:
            members = sorted(group)
            for left in range(len(members)):
                for right in range(left + 1, len(members)):
                    pairs.add((members[left], members[right]))
        pair_sets.append(pairs)
        pair_count += len(pairs)
        if pair_count > 200_000:
            return common_groups, set()
    common_pairs = set.intersection(*pair_sets) if pair_sets else set()
    return common_groups, common_pairs


def _excluded_points(source, identity):
    if source is None:
        return []
    points = _as_point_list(
        source,
        overrides={},
        identity=identity,
    )
    return [point for point in points if point.valid]


def _excluded_index(excluded):
    index = {}
    for excluded_index, point in enumerate(excluded):
        key = (point.flight_line, point.channel, point.number_of_returns,
               point.return_number)
        index.setdefault(key, []).append((point.gps_time, excluded_index))
    for entries in index.values():
        entries.sort()
    return index


def _times_in_range(entries, low, high):
    """Indices whose entries have ``low <= time <= high``."""
    left = bisect.bisect_left(entries, (low, -1))
    right = bisect.bisect_right(entries, (high, float("inf")))
    return [excluded_index for _, excluded_index in entries[left:right]]


def _candidate_slots(group, excluded_index, precision):
    """Return, for each missing return, the excluded points compatible with it."""
    present = set(group.return_numbers)
    total = group.number_of_returns
    time_min = min(group.gps_times)
    time_max = max(group.gps_times)
    result = {}
    for return_number in range(1, total + 1):
        if return_number in present:
            continue
        key = (group.flight_line, group.channel, total, return_number)
        result[return_number] = _times_in_range(
            excluded_index.get(key, ()),
            time_min - precision,
            time_max + precision,
        )
    return result


def _apply_missing_evidence(groups, excluded, precision):
    """Attach missing-return reasons to groups.

    An excluded point is only accepted as proof of a known filtering when it
    is compatible with exactly one missing slot. If several groups could each
    have owned it, its absence remains unexplained instead of being claimed by
    both groups.
    """
    index = _excluded_index(excluded)
    slots = []
    for group_id, group in enumerate(groups):
        for return_number, candidates in _candidate_slots(
            group, index, precision
        ).items():
            slots.append((group_id, return_number, candidates))

    slot_by_candidate = {}
    for slot_id, (_, _, candidates) in enumerate(slots):
        for excluded_index in candidates:
            slot_by_candidate.setdefault(excluded_index, set()).add(slot_id)

    missing_by_group = {}
    for slot_id, (group_id, return_number, candidates) in enumerate(slots):
        unique = [
            excluded[index].identity
            for index in candidates
            if slot_by_candidate[index] == {slot_id}
        ]
        missing_by_group.setdefault(group_id, []).append(
            MissingReturn(
                return_number=return_number,
                reason=(
                    MissingReason.KNOWN_FILTERED
                    if unique
                    else MissingReason.UNKNOWN
                ),
                filtered_indices=tuple(unique),
            )
        )

    return tuple(
        PulseGroup(
            pulse_id=group.pulse_id,
            flight_line=group.flight_line,
            channel=group.channel,
            number_of_returns=group.number_of_returns,
            point_indices=group.point_indices,
            original_indices=group.original_indices,
            return_numbers=group.return_numbers,
            gps_times=group.gps_times,
            gps_time=group.gps_time,
            time_span=group.time_span,
            missing=tuple(missing_by_group.get(group_id, ())),
        )
        for group_id, group in enumerate(groups)
    )


def _pulse_group(group_id, group_indices, point_by_index):
    group_points = [point_by_index[index] for index in group_indices]
    group_points.sort(
        key=lambda point: (point.gps_time, point.return_number, point.index)
    )
    times = [point.gps_time for point in group_points]
    return PulseGroup(
        pulse_id=group_id,
        flight_line=group_points[0].flight_line,
        channel=group_points[0].channel,
        number_of_returns=group_points[0].number_of_returns,
        point_indices=tuple(point.index for point in group_points),
        original_indices=tuple(point.identity for point in group_points),
        return_numbers=tuple(point.return_number for point in group_points),
        gps_times=tuple(times),
        gps_time=sum(times) / len(times),
        time_span=max(times) - min(times),
        missing=(),
    )


def group_pulses(
    source,
    time_precision: Optional[float] = None,
    *,
    tolerance: Optional[float] = None,
    time_tolerance: Optional[float] = None,
    time_epsilon: Optional[float] = None,
    flight_line=None,
    point_source_id=None,
    channel=None,
    scanner_channel=None,
    return_number=None,
    return_num=None,
    number_of_returns=None,
    num_returns=None,
    gps_time=None,
    time=None,
    identity=None,
    point_indices=None,
    original_indices=None,
    excluded=None,
    excluded_points=None,
    filtered_out=None,
    excluded_identity=None,
    excluded_point_indices=None,
    max_solutions: int = 256,
    max_steps: int = 200_000,
) -> PulseGrouping:
    """Build provable pulse groups from LAS points.

    Parameters
    ----------
    source:
        A :class:`laspy.LasData`, a point record, or a mapping/object exposing
        ``gps_time``, ``return_number``, ``number_of_returns`` and optionally
        ``point_source_id`` and ``scanner_channel``.
    time_precision:
        Maximum absolute GPS-time difference allowed between two points of the
        same pulse. Use ``0`` for exact equality. ``tolerance`` and
        ``time_tolerance`` are accepted as aliases.
    flight_line, channel, return_number, number_of_returns, gps_time:
        Explicit arrays overriding the dimensions read from ``source``.
        ``point_source_id``, ``scanner_channel``, ``return_num``, ``num_returns``
        and ``time`` are accepted aliases.
    identity:
        One original-occurrence identity per input point. Defaults to the input
        position. ``point_indices`` and ``original_indices`` are aliases.
    excluded:
        Points that are known to have been filtered out before ``source`` was
        produced. Missing returns matched against this set are reported as
        :attr:`MissingReason.KNOWN_FILTERED`; all other missing returns are
        :attr:`MissingReason.UNKNOWN`. ``excluded_points`` and
        ``filtered_out`` are aliases. The excluded points must have the same
        coordinate fields (flight line, channel, totals, GPS time) as the
        source.
    excluded_identity:
        Optional identities for the excluded points. When omitted, positions
        in ``excluded`` are used. ``excluded_point_indices`` is an alias.
    max_solutions, max_steps:
        Bounds for the exact-cover enumeration. When a component exceeds them
        it is reported as an ambiguous cluster with ``exhaustive=False`` and no
        unproven group is claimed.

    Returns
    -------
    PulseGrouping

    Notes
    -----
    Two points can share a pulse only when their flight line, scanner channel
    and declared number of returns are equal, their GPS times differ by at
    most ``time_precision``, and their return numbers differ. A valid grouping
    assigns every valid point to exactly one pulse. Among the groupings using
    the fewest pulses, only memberships shared by all of them are reported as
    proven groups. A file's point order is never used as evidence.

    Identical GPS times on different channels, or with duplicate return
    numbers, never become one pulse merely because the times are equal.
    """
    precision = _first_set(
        time_precision, tolerance, time_tolerance, time_epsilon
    )
    if precision is None:
        precision = 0.0
    precision = float(precision)
    if not math.isfinite(precision) or precision < 0:
        raise ValueError("time_precision must be a finite, non-negative number")
    if max_solutions < 1 or max_steps < 1:
        raise ValueError("max_solutions and max_steps must be positive")

    overrides = {
        "gps_time": _first_set(gps_time, time),
        "return_number": _first_set(return_number, return_num),
        "number_of_returns": _first_set(number_of_returns, num_returns),
        "point_source_id": _first_set(flight_line, point_source_id),
        "scanner_channel": _first_set(channel, scanner_channel),
    }
    identity = _first_set(identity, point_indices, original_indices)
    excluded_source = _first_set(excluded, excluded_points, filtered_out)

    points = _as_point_list(
        source,
        overrides=overrides,
        identity=identity,
    )
    invalid = tuple(
        InvalidPoint(
            point_index=point.index,
            original_index=point.identity,
            reason=point.invalid_reason,
        )
        for point in points
        if not point.valid
    )
    valid_points = [point for point in points if point.valid]
    excluded_list = _excluded_points(
        excluded_source,
        _first_set(excluded_identity, excluded_point_indices),
    )

    groups = []
    ambiguous = []
    group_id = 0

    by_declaration = {}
    for point in valid_points:
        key = (point.flight_line, point.channel, point.number_of_returns)
        by_declaration.setdefault(key, []).append(point)

    for key in sorted(by_declaration):
        declaration_points = by_declaration[key]
        for component in _time_components(declaration_points, precision):
            point_by_index = {point.index: point for point in component}
            solutions, exhaustive = _min_partitions(
                component,
                precision,
                max_solutions=max_solutions,
                max_steps=max_steps,
            )
            if exhaustive and solutions:
                common_groups, _ = _common_relations(solutions)
                for group_indices in sorted(
                    common_groups,
                    key=lambda group: sorted(group),
                ):
                    ordered = tuple(sorted(group_indices))
                    groups.append(_pulse_group(group_id, ordered, point_by_index))
                    group_id += 1
                covered = set().union(*common_groups) if common_groups else set()
                residual = [
                    point for point in component if point.index not in covered
                ]
                if not residual:
                    continue
                residual_solutions = {
                    frozenset(
                        group for group in solution if not group <= covered
                    )
                    for solution in solutions
                }
                residual_groups, residual_pairs = _common_relations(
                    residual_solutions
                )
                target = len(next(iter(solutions))) - len(common_groups)
                ambiguous.append(
                    _ambiguous_cluster(
                        residual,
                        target,
                        exhaustive=True,
                        grouping_count=len(residual_solutions),
                        candidate_pulses=_candidate_pulses(residual_solutions),
                        common_groups=residual_groups,
                        common_pairs=residual_pairs,
                    )
                )
            else:
                lower_bound = max(
                    Counter(point.return_number for point in component).values()
                )
                ambiguous.append(
                    _ambiguous_cluster(
                        component,
                        lower_bound,
                        exhaustive=exhaustive,
                        grouping_count=len(solutions),
                        candidate_pulses=_candidate_pulses(solutions),
                        common_groups=set(),
                        common_pairs=set(),
                    )
                )

    groups.sort(
        key=lambda group: (
            group.flight_line,
            group.channel,
            group.gps_time,
            group.return_numbers,
            group.point_indices,
        )
    )
    groups = tuple(
        PulseGroup(
            pulse_id=index,
            flight_line=group.flight_line,
            channel=group.channel,
            number_of_returns=group.number_of_returns,
            point_indices=group.point_indices,
            original_indices=group.original_indices,
            return_numbers=group.return_numbers,
            gps_times=group.gps_times,
            gps_time=group.gps_time,
            time_span=group.time_span,
            missing=group.missing,
        )
        for index, group in enumerate(groups)
    )
    groups = _apply_missing_evidence(groups, excluded_list, precision)
    ambiguous.sort(
        key=lambda cluster: (
            cluster.flight_line,
            cluster.channel,
            cluster.number_of_returns,
            cluster.gps_times,
            cluster.point_indices,
        )
    )

    return PulseGrouping(
        groups=groups,
        ambiguous_clusters=tuple(ambiguous),
        invalid_points=invalid,
        time_precision=precision,
        source_point_count=len(points),
    )


def _candidate_pulses(solutions):
    if not solutions:
        return ()
    candidates = set()
    for solution in solutions:
        candidates.update(solution)
    return tuple(
        tuple(sorted(group))
        for group in sorted(candidates, key=lambda group: sorted(group))
    )


def _ambiguous_cluster(
    component,
    minimum_pulse_count,
    *,
    exhaustive,
    grouping_count,
    candidate_pulses,
    common_groups,
    common_pairs,
):
    component = sorted(component, key=lambda point: point.index)
    return AmbiguousCluster(
        flight_line=component[0].flight_line,
        channel=component[0].channel,
        number_of_returns=component[0].number_of_returns,
        point_indices=tuple(point.index for point in component),
        original_indices=tuple(point.identity for point in component),
        return_numbers=tuple(point.return_number for point in component),
        gps_times=tuple(point.gps_time for point in component),
        minimum_pulse_count=minimum_pulse_count,
        exhaustive=exhaustive,
        grouping_count=grouping_count,
        candidate_pulses=candidate_pulses,
        common_groups=tuple(
            tuple(sorted(group))
            for group in sorted(common_groups, key=lambda group: sorted(group))
        ),
        common_pairs=tuple(sorted(common_pairs)),
    )


# Friendly alternative names used by different callers.
build_pulse_groups = group_pulses
group_into_pulses = group_pulses
