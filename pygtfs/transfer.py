"""Transfer rules between stops.

A row of ``transfers.txt`` joins two stops and may additionally be narrowed
to a pair of routes or a pair of trips.  A feed can publish several rows for
the same physical transfer, for example a generic stop-to-stop rule and a
trip-specific rule, or a rule stored on a parent station in addition to one
stored on an individual platform.

This module turns those rows into :class:`TransferCandidate` objects.  Every
candidate keeps the :class:`~pygtfs.gtfs_entities.Transfer` row it came from,
so callers always know the source and scope of the evidence.

Precedence convention used by :func:`preferred_transfer`, strongest first:

1. An explicit prohibition (``transfer_type == 3``) always wins over any
   permitted rule for the same pair of stops.
2. Trip-specific rules beat route-specific rules, which beat generic
   stop-to-stop rules.
3. A rule stored directly on the queried platform beats a rule stored on its
   parent station.

Stops that share a ``stop_name`` are *not* assumed to be the same stop: a
candidate is only produced when an actual transfer row backs it, either on
the queried stop itself or on its parent station.  When no walking time is
documented, :attr:`TransferCandidate.min_transfer_time` is ``None``.  Missing
evidence means "unknown" and must never be treated as zero minutes; an
explicit ``0`` is preserved as ``0``.
"""

from .gtfs_entities import Stop, Transfer

SCOPE_STOP = 'stop'
SCOPE_ROUTE = 'route'
SCOPE_TRIP = 'trip'

STOP_SOURCE_STOP = 'stop'
STOP_SOURCE_PARENT = 'parent_station'

#: GTFS transfer_type 3: transfers are not allowed between the two points.
TRANSFER_FORBIDDEN = 3

_SCOPE_RANK = {SCOPE_TRIP: 0, SCOPE_ROUTE: 1, SCOPE_STOP: 2}


def _is_blank(value):
    return value is None or value == ''


def rule_scope(transfer):
    """Return the specificity scope of a transfer row.

    A row that pins one or both trips is trip-specific, a row that pins one
    or both routes is route-specific, and any other row is a generic
    stop-to-stop rule.  Blank route/trip columns count as unpinned.
    """
    if not _is_blank(transfer.from_trip_id) or \
            not _is_blank(transfer.to_trip_id):
        return SCOPE_TRIP
    if not _is_blank(transfer.from_route_id) or \
            not _is_blank(transfer.to_route_id):
        return SCOPE_ROUTE
    return SCOPE_STOP


def _side_applies(rule_from, rule_to, query_from, query_to):
    """Check a pinned from/to id pair against the queried ids."""
    if not _is_blank(rule_from) and (query_from is None or rule_from != query_from):
        return False
    if not _is_blank(rule_to) and (query_to is None or rule_to != query_to):
        return False
    return True


def _stop_id_chain(stop):
    ids = [stop.stop_id]
    if not _is_blank(stop.parent_station):
        ids.append(stop.parent_station)
    return ids


def _stop_source(rule_stop_id, queried_stop):
    if rule_stop_id == queried_stop.stop_id:
        return STOP_SOURCE_STOP
    return STOP_SOURCE_PARENT


class TransferCandidate(object):
    """A transfer rule that applies to a queried pair of stops.

    :ivar source: the :class:`~pygtfs.gtfs_entities.Transfer` row this
        candidate was built from.
    :ivar scope: one of :data:`SCOPE_STOP`, :data:`SCOPE_ROUTE`,
        :data:`SCOPE_TRIP`.
    :ivar from_stop_source: :data:`STOP_SOURCE_STOP` when the rule names the
        queried stop directly, :data:`STOP_SOURCE_PARENT` when it only names
        the stop's parent station.
    :ivar to_stop_source: same as ``from_stop_source`` for the destination.
    """

    def __init__(self, source, scope, from_stop_source, to_stop_source):
        self.source = source
        self.scope = scope
        self.from_stop_source = from_stop_source
        self.to_stop_source = to_stop_source

    @property
    def transfer_type(self):
        """GTFS transfer type, or ``None`` when the feed left it blank."""
        return self.source.transfer_type

    @property
    def min_transfer_time(self):
        """Documented walking time in seconds, or ``None`` if unknown."""
        return self.source.min_transfer_time

    @property
    def walking_time_seconds(self):
        """Alias for :attr:`min_transfer_time`; ``None`` means unknown."""
        return self.source.min_transfer_time

    @property
    def is_forbidden(self):
        """``True`` when the rule explicitly forbids this transfer."""
        return self.source.transfer_type == TRANSFER_FORBIDDEN

    def __repr__(self):
        return '<TransferCandidate {0}->{1} scope={2} type={3} time={4}>'.format(
            self.source.from_stop_id, self.source.to_stop_id, self.scope,
            self.transfer_type, self.min_transfer_time)


def _candidate_sort_key(candidate):
    parent_count = (
        (candidate.from_stop_source == STOP_SOURCE_PARENT) +
        (candidate.to_stop_source == STOP_SOURCE_PARENT))
    source = candidate.source
    return (
        _SCOPE_RANK[candidate.scope],
        parent_count,
        candidate.min_transfer_time is None,
        candidate.min_transfer_time or 0,
        candidate.transfer_type if candidate.transfer_type is not None else 99,
        source.from_stop_id, source.to_stop_id,
        source.from_route_id, source.to_route_id,
        source.from_trip_id, source.to_trip_id,
    )


def transfer_candidates(schedule, from_stop, to_stop,
                        from_route_id=None, to_route_id=None,
                        from_trip_id=None, to_trip_id=None):
    """Return all transfer rules applicable to a queried pair of stops.

    ``from_stop`` and ``to_stop`` are :class:`~pygtfs.gtfs_entities.Stop`
    objects.  A rule matches when it names the stop itself *or* the stop's
    parent station, and when every route/trip id it pins matches the
    corresponding queried id.  Generic stop-to-stop rules always match.

    The returned list is ordered by the precedence convention, so the first
    element is the preferred candidate unless a prohibition overrides it
    (see :func:`preferred_transfer`).  An empty list means the feed contains
    no evidence for this pair at all.
    """
    rows = (schedule.session.query(Transfer)
            .filter(Transfer.feed_id == from_stop.feed_id,
                    Transfer.from_stop_id.in_(_stop_id_chain(from_stop)),
                    Transfer.to_stop_id.in_(_stop_id_chain(to_stop)))
            .all())

    candidates = []
    for row in rows:
        scope = rule_scope(row)
        if scope == SCOPE_ROUTE and not _side_applies(
                row.from_route_id, row.to_route_id,
                from_route_id, to_route_id):
            continue
        if scope == SCOPE_TRIP and not _side_applies(
                row.from_trip_id, row.to_trip_id,
                from_trip_id, to_trip_id):
            continue
        candidates.append(TransferCandidate(
            row, scope,
            _stop_source(row.from_stop_id, from_stop),
            _stop_source(row.to_stop_id, to_stop)))

    candidates.sort(key=_candidate_sort_key)
    return candidates


def preferred_transfer(schedule, from_stop, to_stop,
                       from_route_id=None, to_route_id=None,
                       from_trip_id=None, to_trip_id=None):
    """Return the preferred transfer candidate, or ``None`` without evidence.

    A forbidden rule (``transfer_type == 3``) short-circuits the selection:
    if any applicable rule forbids the transfer, the most specific forbidden
    candidate is returned.  Otherwise the most specific allowed candidate
    wins (trip over route over stop, direct platform over parent station).

    ``None`` means the feed documents no rule for this pair; it does not mean
    a zero-minute transfer.
    """
    candidates = transfer_candidates(
        schedule, from_stop, to_stop,
        from_route_id=from_route_id, to_route_id=to_route_id,
        from_trip_id=from_trip_id, to_trip_id=to_trip_id)
    if not candidates:
        return None
    for candidate in candidates:
        if candidate.is_forbidden:
            return candidate
    return candidates[0]


def walking_time_seconds(schedule, from_stop, to_stop,
                         from_route_id=None, to_route_id=None,
                         from_trip_id=None, to_trip_id=None):
    """Return the preferred walking time in seconds, or ``None`` if unknown.

    ``None`` is returned both when no rule documents the pair and when the
    preferred rule forbids the transfer.  It is never turned into zero; an
    explicit ``min_transfer_time`` of ``0`` is returned as ``0``.
    """
    candidate = preferred_transfer(
        schedule, from_stop, to_stop,
        from_route_id=from_route_id, to_route_id=to_route_id,
        from_trip_id=from_trip_id, to_trip_id=to_trip_id)
    if candidate is None or candidate.is_forbidden:
        return None
    return candidate.min_transfer_time


def same_named_stops(schedule, stop):
    """Return other stops in the feed sharing ``stop.stop_name``.

    Matching names do not imply that transfers are allowed or instant:
    transfer rules still have to come from ``transfers.txt``.
    """
    return (schedule.session.query(Stop)
            .filter(Stop.feed_id == stop.feed_id,
                    Stop.stop_name == stop.stop_name,
                    Stop.stop_id != stop.stop_id)
            .all())
