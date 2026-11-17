# -*- coding: utf-8 -*-
"""Aggregation of COO sparse-coordinate contributions.

The same coordinate may be delivered more than once.  Each delivery declares
how it expects to be combined:

* ``ACCUMULATE``: contributions at the coordinate are summed;
* ``FORBID_DUPLICATE``: the source claims exclusive authority, so any other
  delivery at that coordinate is a conflict.

Contributions carry their source identity and a delivery identifier.  A retry
reuses the same delivery identifier (possibly in a different fragment) and is
counted exactly once.  Every unique contribution is retained as evidence,
including deliveries whose accumulated sum is exactly zero and deliveries that
participate in an unresolved conflict.

Three states must stay distinguishable after a dense round-trip:

* a coordinate that was never delivered (``ABSENT``),
* a coordinate whose value is missing (``MISSING``),
* a coordinate whose accumulated value is genuinely zero (``ZERO``),
* a coordinate with an unresolved policy/data conflict (``CONFLICT``).
"""

from enum import Enum
import math


ACCUMULATE = 'accumulate'
FORBID_DUPLICATE = 'forbid_duplicate'
_POLICIES = (ACCUMULATE, FORBID_DUPLICATE)


class LineState(Enum):
    """Lifecycle state of a synthesized logical line."""

    ABSENT = 'absent'
    MISSING = 'missing'
    VALUE = 'value'
    ZERO = 'zero'
    CONFLICT = 'conflict'


ABSENT = LineState.ABSENT
MISSING = LineState.MISSING
ZERO = LineState.ZERO
CONFLICT = LineState.CONFLICT


class COOAggregationError(Exception):
    """Raised when an aggregation object is used incorrectly."""


class Contribution(object):
    """A single delivery of a value at a coordinate.

    Parameters
    ----------
    coord:
        Coordinate tuple, e.g. ``(row, column)``.
    source:
        Identity of the source that produced the contribution.
    value:
        Numeric value, or ``None`` when the source reports the value as
        missing.
    policy:
        One of :data:`ACCUMULATE` or :data:`FORBID_DUPLICATE`.
    delivery:
        Stable identifier of this delivery.  A retry keeps the same
        identifier, even when it arrives through another fragment.
    fragment:
        Optional fragment/routing identifier; retries through different
        fragments are still counted once.
    """

    __slots__ = ('coord', 'source', 'value', 'policy', 'delivery', 'fragment')

    def __init__(self, coord, source, value, policy, delivery, fragment=None):
        if policy not in _POLICIES:
            raise COOAggregationError('unknown policy %r' % (policy,))
        if value is not None and not isinstance(value, (int, float)):
            raise COOAggregationError(
                'contribution value must be numeric or None, got %r'
                % (type(value).__name__,))
        coord = tuple(coord)
        if not coord:
            raise COOAggregationError('coordinate must not be empty')
        self.coord = coord
        self.source = source
        self.value = value
        self.policy = policy
        self.delivery = delivery
        self.fragment = fragment

    @property
    def dedup_key(self):
        """Identity used to collapse retries (fragment is intentionally absent)."""
        return self.coord, self.source, self.delivery

    @property
    def _payload(self):
        return self.value, self.policy

    def __eq__(self, other):
        if not isinstance(other, Contribution):
            return NotImplemented
        return (self.coord == other.coord
                and self.source == other.source
                and self.value == other.value
                and self.policy == other.policy
                and self.delivery == other.delivery
                and self.fragment == other.fragment)

    def __hash__(self):
        return hash((self.coord, self.source, self.delivery,
                     self.value, self.policy, self.fragment))

    def __repr__(self):
        return ('Contribution(coord=%r, source=%r, value=%r, policy=%r, '
                'delivery=%r, fragment=%r)'
                % (self.coord, self.source, self.value, self.policy,
                   self.delivery, self.fragment))


class LogicalLine(object):
    """The synthesized result for one coordinate.

    Attributes
    ----------
    coord:
        Coordinate tuple.
    state:
        A :class:`LineState` member.
    value:
        Aggregated numeric value, or ``None`` for missing/conflict lines.
    contributions:
        Tuple of every unique :class:`Contribution` at this coordinate,
        retained as evidence (including zero sums and conflicts).
    conflicts:
        Tuple of human-readable conflict reasons; empty when
        ``state is not LineState.CONFLICT``.
    redeliveries:
        Number of retry deliveries that were collapsed away.
    """

    __slots__ = ('coord', 'state', 'value', 'contributions',
                 'conflicts', 'redeliveries')

    def __init__(self, coord, state, value, contributions,
                 conflicts=(), redeliveries=0):
        self.coord = tuple(coord)
        self.state = state
        self.value = value
        self.contributions = tuple(contributions)
        self.conflicts = tuple(conflicts)
        self.redeliveries = redeliveries

    @property
    def is_conflict(self):
        return self.state is LineState.CONFLICT

    def __eq__(self, other):
        if not isinstance(other, LogicalLine):
            return NotImplemented
        return (self.coord == other.coord
                and self.state == other.state
                and self.value == other.value
                and self.contributions == other.contributions
                and self.conflicts == other.conflicts
                and self.redeliveries == other.redeliveries)

    def __repr__(self):
        return ('LogicalLine(coord=%r, state=%s, value=%r, '
                'contributions=%d, conflicts=%r)'
                % (self.coord, self.state.name, self.value,
                   len(self.contributions), self.conflicts))


def _deterministic_sum(values):
    """Sum values independently of input ordering.

    Integer inputs keep an exact integer result; floating point inputs are
    accumulated with :func:`math.fsum` over a sorted sequence so that any
    permutation of the same inputs yields the same result.
    """
    if all(type(value) is int for value in values):
        return sum(sorted(values))
    return math.fsum(sorted(float(value) for value in values))


class COOAggregator(object):
    """Streaming, order-independent aggregator for COO contributions."""

    def __init__(self):
        # dedup_key -> list of distinct Contributions (usually one entry;
        # more than one means a delivery id was reused with different
        # payloads, which is an unresolved redelivery conflict).
        self._deliveries = {}
        # dedup_key -> number of collapsed retry deliveries
        self._redeliveries = {}
        # dedup_key -> list of conflict reason strings
        self._payload_conflicts = {}

    def add(self, contribution):
        """Add one :class:`Contribution`, collapsing retries by delivery id."""
        if not isinstance(contribution, Contribution):
            raise COOAggregationError(
                'expected a Contribution, got %r'
                % (type(contribution).__name__,))

        key = contribution.dedup_key
        bucket = self._deliveries.get(key)
        if bucket is None:
            self._deliveries[key] = [contribution]
            self._redeliveries[key] = 0
            return self

        for existing in bucket:
            if existing == contribution:
                # Exact retry (including an identical fragment change):
                # counted once.
                self._redeliveries[key] += 1
                return self

        if any(existing._payload == contribution._payload
               for existing in bucket):
            # Same value/policy redelivered via another fragment: still a
            # retry of the same contribution, counted once.
            self._redeliveries[key] += 1
            return self

        # Same delivery identity arriving with different content: unresolved
        # conflict.  Both payloads are retained as evidence.
        bucket.append(contribution)
        self._redeliveries[key] += 1
        self._payload_conflicts.setdefault(key, []).append(
            'delivery %r from source %r was redelivered with conflicting '
            'payloads %r and %r'
            % (contribution.delivery, contribution.source,
               bucket[0].value, contribution.value))
        return self

    def add_many(self, contributions):
        for contribution in contributions:
            self.add(contribution)
        return self

    def absorb(self, other):
        """Merge another aggregator into this one.

        Retries shared by both aggregators are still counted once.
        """
        if not isinstance(other, COOAggregator):
            raise COOAggregationError(
                'can only absorb another COOAggregator, got %r'
                % (type(other).__name__,))
        for key, bucket in other._deliveries.items():
            for contribution in bucket:
                self.add(contribution)
            self._redeliveries[key] = (
                self._redeliveries.get(key, 0)
                + other._redeliveries.get(key, 0))
        for key, reasons in other._payload_conflicts.items():
            self._payload_conflicts.setdefault(key, []).extend(reasons)
        return self

    def _all_contributions(self):
        for bucket in self._deliveries.values():
            for contribution in bucket:
                yield contribution

    def lines(self):
        """Return the synthesized :class:`LogicalLine` objects, sorted by coord."""
        grouped = {}
        redelivery_by_coord = {}
        for key, bucket in self._deliveries.items():
            coord = key[0]
            grouped.setdefault(coord, []).extend(bucket)
            redelivery_by_coord[coord] = (
                redelivery_by_coord.get(coord, 0)
                + self._redeliveries.get(key, 0))

        lines = []
        for coord in sorted(grouped):
            contributions = sorted(
                grouped[coord],
                key=lambda item: (repr(item.source), repr(item.delivery),
                                  repr(item.fragment)))
            conflicts = [
                reason
                for key, reasons in self._payload_conflicts.items()
                if key[0] == coord
                for reason in reasons]

            by_source = {}
            for contribution in contributions:
                by_source.setdefault(contribution.source, []).append(
                    contribution)

            for source, items in by_source.items():
                if any(item.policy == FORBID_DUPLICATE for item in items) \
                        and len(items) > 1:
                    conflicts.append(
                        'source %r delivered coordinate %r %d times under '
                        'policy %r'
                        % (source, coord, len(items), FORBID_DUPLICATE))

            if len(by_source) > 1 and any(
                    item.policy == FORBID_DUPLICATE
                    for item in contributions):
                policies = sorted({item.policy for item in contributions})
                conflicts.append(
                    'coordinate %r is claimed by %d sources with '
                    'incompatible policies %r'
                    % (coord, len(by_source), policies))

            if conflicts:
                state = LineState.CONFLICT
                value = None
            elif any(item.value is None for item in contributions):
                state = LineState.MISSING
                value = None
            else:
                value = _deterministic_sum(
                    [item.value for item in contributions])
                state = LineState.ZERO if value == 0 else LineState.VALUE

            lines.append(LogicalLine(
                coord=coord,
                state=state,
                value=value,
                contributions=contributions,
                conflicts=conflicts,
                redeliveries=redelivery_by_coord.get(coord, 0)))
        return lines

    def get(self, coord):
        """Return the :class:`LogicalLine` at ``coord`` or ``None``."""
        coord = tuple(coord)
        for line in self.lines():
            if line.coord == coord:
                return line
        return None


def merge_contributions(contributions):
    """Synthesize logical lines from an iterable of contributions."""
    aggregator = COOAggregator()
    aggregator.add_many(contributions)
    return aggregator.lines()


def _empty_grid(shape):
    if len(shape) == 1:
        return [LineState.ABSENT] * shape[0]
    return [_empty_grid(shape[1:]) for _ in range(shape[0])]


def _check_shape(shape):
    shape = tuple(shape)
    if not shape or any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0
            for dim in shape):
        raise COOAggregationError(
            'shape must be a non-empty tuple of positive ints, got %r'
            % (shape,))
    return shape


def to_dense(lines, shape):
    """Lay logical lines out on a dense nested list.

    Cells are filled with the aggregated number (a genuine zero stays
    ``0``), :data:`MISSING`, :data:`CONFLICT`, or :data:`ABSENT` for
    coordinates that no contribution ever touched.
    """
    shape = _check_shape(shape)
    grid = _empty_grid(shape)
    for line in lines:
        if len(line.coord) != len(shape):
            raise COOAggregationError(
                'coordinate %r has %d dimensions, shape has %d'
                % (line.coord, len(line.coord), len(shape)))
        for index, dim in zip(line.coord, shape):
            if not 0 <= index < dim:
                raise COOAggregationError(
                    'coordinate %r is outside shape %r' % (line.coord, shape))

        cell = grid
        for index in line.coord[:-1]:
            cell = cell[index]
        if line.state is LineState.CONFLICT:
            cell[line.coord[-1]] = LineState.CONFLICT
        elif line.state is LineState.MISSING:
            cell[line.coord[-1]] = LineState.MISSING
        elif line.state is LineState.ABSENT:
            continue
        else:
            cell[line.coord[-1]] = line.value
    return grid


def from_dense(grid):
    """Inverse of :func:`to_dense`.

    ``ABSENT`` cells produce no logical line; numeric zeros rebuild
    :data:`ZERO` lines, while :data:`MISSING` and :data:`CONFLICT` keep
    their states.  Dense storage carries no provenance, so rebuilt lines
    have empty ``contributions`` tuples.
    """
    lines = []

    def walk(node, prefix):
        if isinstance(node, list):
            for index, child in enumerate(node):
                walk(child, prefix + (index,))
            return
        coord = prefix
        if node is LineState.ABSENT:
            return
        if node is LineState.MISSING:
            lines.append(LogicalLine(coord, LineState.MISSING, None, ()))
        elif node is LineState.CONFLICT:
            lines.append(LogicalLine(coord, LineState.CONFLICT, None, ()))
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            state = LineState.ZERO if node == 0 else LineState.VALUE
            lines.append(LogicalLine(coord, state, node, ()))
        else:
            raise COOAggregationError(
                'unexpected dense cell %r at %r' % (node, coord))

    walk(grid, ())
    return lines
