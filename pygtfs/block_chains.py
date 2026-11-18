"""Vehicle block (``block_id``) succession chains.

A GTFS ``block_id`` only groups trips that are *planned* to be operated by the
same physical vehicle.  It is not, by itself, a vehicle trajectory: the trips
of a block may not run on the same days, may overlap in time, and the raw
ordering of ``trips.txt`` says nothing about which trip follows which.

This module reconstructs the succession chains that can actually be driven.
A trip ``A`` may be followed by a trip ``B`` only when *all* of the following
hold:

* ``A`` and ``B`` belong to the same feed and share the same ``block_id``
  (identical ids in different feeds are never joined);
* both trips actually run on at least one service day, computed from
  ``calendar.txt`` and ``calendar_dates.txt`` -- a trip that never runs cannot
  bridge two other trips;
* the last drop-off stop of ``A`` (largest ``stop_sequence``) is the same
  stop as the first pickup stop of ``B`` (smallest ``stop_sequence``);
* ``B``'s first departure is not earlier than ``A``'s last arrival, compared
  as service-day-relative times (GTFS times may go past 24:00, hence the
  values are compared as :class:`datetime.timedelta`).

Every trip satisfying those rules is retained as a candidate successor:
there is no "nearest departure" heuristic and no collapsing to a single
chain.  Time-window overlaps and forking/merging successions on a common
service day are reported as conflicts, and every result keeps the raw
``stop_times`` / calendar evidence it was derived from.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from .gtfs_entities import (
    Service,
    ServiceException,
    StopTime,
    Trip,
)

# Weekday flags on Service, in Monday -> Sunday order.
_WEEKDAY_ATTRS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)


@dataclass
class StopTimeEvidence:
    """One raw ``stop_times.txt`` row of a trip."""

    stop_sequence: int
    stop_id: str
    arrival_time: Optional[timedelta]
    departure_time: Optional[timedelta]


@dataclass
class TripOperatingInfo:
    """Everything known about one trip inside a block."""

    feed_id: int
    block_id: str
    trip_id: str
    route_id: str
    service_id: str
    operating_dates: List[date]
    stop_times: List[StopTimeEvidence]
    first_stop: Optional[StopTimeEvidence] = None
    last_stop: Optional[StopTimeEvidence] = None
    start_time: Optional[timedelta] = None
    end_time: Optional[timedelta] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def operates(self) -> bool:
        return bool(self.operating_dates)


@dataclass
class BlockEdge:
    """A feasible succession between two trips."""

    from_trip_id: str
    to_trip_id: str
    connection_stop_id: str
    arrival_time: timedelta
    departure_time: timedelta
    dwell: timedelta
    valid_dates: List[date]
    candidate: bool = False


@dataclass
class ChainLink:
    """One succession inside a chain, with its raw timetable evidence."""

    from_trip_id: str
    to_trip_id: str
    connection_stop_id: str
    arrival_time: timedelta
    departure_time: timedelta
    dwell: timedelta
    valid_dates: List[date]


@dataclass
class BlockChain:
    """A maximal feasible chain (one vehicle, one concrete alternative)."""

    trip_ids: List[str]
    operating_dates: List[date]
    links: List[ChainLink]
    ambiguous: bool = False


@dataclass
class BlockConflict:
    """A situation that prevents a block from being a clean trajectory."""

    conflict_type: str
    trip_ids: List[str]
    detail: str
    service_date: Optional[date] = None


@dataclass
class ExcludedTrip:
    """A trip that cannot take part in any succession, with the reason."""

    trip_id: str
    service_id: str
    reason: str


@dataclass
class BlockChainReport:
    """Result for one ``(feed_id, block_id)`` group."""

    feed_id: int
    block_id: str
    trips: Dict[str, TripOperatingInfo]
    edges: List[BlockEdge]
    chains: List[BlockChain]
    conflicts: List[BlockConflict]
    excluded_trips: List[ExcludedTrip]
    notes: List[str]


def _service_date_map(session) -> Dict[Tuple[int, str], List[date]]:
    """Compute the active service dates of every service.

    Regular weekday rules give the base set inside ``start_date`` /
    ``end_date``; ``calendar_dates`` rows add (exception type 1) or remove
    (exception type 2) individual days.  Services that only exist through
    ``calendar_dates`` (the loader creates an all-zero dummy calendar row for
    them) simply get the added exception days.
    """
    exceptions: Dict[Tuple[int, str], Dict[date, int]] = {}
    for exc in session.query(ServiceException).all():
        exceptions.setdefault(
            (exc.feed_id, exc.service_id), {}
        )[exc.date] = exc.exception_type

    result: Dict[Tuple[int, str], List[date]] = {}
    for service in session.query(Service).all():
        key = (service.feed_id, service.service_id)
        days = set()
        if service.start_date is not None and service.end_date is not None:
            current = service.start_date
            while current <= service.end_date:
                if getattr(service, _WEEKDAY_ATTRS[current.weekday()]):
                    days.add(current)
                current += timedelta(days=1)
        for day, exception_type in exceptions.get(key, {}).items():
            if exception_type == 1:
                days.add(day)
            else:
                days.discard(day)
        result[key] = sorted(days)
    return result


def _stop_time_evidence(stop_time: StopTime) -> StopTimeEvidence:
    return StopTimeEvidence(
        stop_sequence=stop_time.stop_sequence,
        stop_id=stop_time.stop_id,
        arrival_time=stop_time.arrival_time,
        departure_time=stop_time.departure_time,
    )


def _load_trip_info(
    trip: Trip, service_dates: Dict[Tuple[int, str], List[date]]
) -> TripOperatingInfo:
    stop_times = sorted(trip.stop_times, key=lambda st: st.stop_sequence)
    evidence = [_stop_time_evidence(st) for st in stop_times]

    info = TripOperatingInfo(
        feed_id=trip.feed_id,
        block_id=trip.block_id,
        trip_id=trip.trip_id,
        route_id=trip.route_id,
        service_id=trip.service_id,
        operating_dates=list(
            service_dates.get((trip.feed_id, trip.service_id), [])
        ),
        stop_times=evidence,
    )

    if not evidence:
        info.warnings.append("trip has no stop_times rows")
        return info

    info.first_stop = evidence[0]
    info.last_stop = evidence[-1]
    info.start_time = (
        info.first_stop.departure_time
        if info.first_stop.departure_time is not None
        else info.first_stop.arrival_time
    )
    info.end_time = (
        info.last_stop.arrival_time
        if info.last_stop.arrival_time is not None
        else info.last_stop.departure_time
    )
    if info.start_time is None:
        info.warnings.append("first stop has neither departure nor arrival time")
    if info.end_time is None:
        info.warnings.append("last stop has neither arrival nor departure time")
    return info


def _intersection(date_lists: List[List[date]]) -> List[date]:
    if not date_lists:
        return []
    common = set(date_lists[0])
    for dates in date_lists[1:]:
        common &= set(dates)
    return sorted(common)


def _build_edges(
    trips: Dict[str, TripOperatingInfo], notes: List[str]
) -> List[BlockEdge]:
    """Apply the succession rule, keeping every feasible successor."""
    edges: List[BlockEdge] = []
    usable = [
        info
        for info in trips.values()
        if info.operates
        and info.first_stop is not None
        and info.last_stop is not None
        and info.start_time is not None
        and info.end_time is not None
    ]

    for predecessor in usable:
        for successor in usable:
            if predecessor.trip_id == successor.trip_id:
                continue
            if predecessor.last_stop.stop_id != successor.first_stop.stop_id:
                continue
            if successor.start_time < predecessor.end_time:
                notes.append(
                    "trip {0} could follow {1} at stop {2} but departs "
                    "{3} before the arrival {4}; succession rejected".format(
                        successor.trip_id,
                        predecessor.trip_id,
                        successor.first_stop.stop_id,
                        successor.start_time,
                        predecessor.end_time,
                    )
                )
                continue
            valid_dates = _intersection(
                [predecessor.operating_dates, successor.operating_dates]
            )
            if not valid_dates:
                notes.append(
                    "trip {0} and trip {1} share no service day; "
                    "succession rejected".format(
                        predecessor.trip_id, successor.trip_id
                    )
                )
                continue
            edges.append(
                BlockEdge(
                    from_trip_id=predecessor.trip_id,
                    to_trip_id=successor.trip_id,
                    connection_stop_id=predecessor.last_stop.stop_id,
                    arrival_time=predecessor.end_time,
                    departure_time=successor.start_time,
                    dwell=successor.start_time - predecessor.end_time,
                    valid_dates=valid_dates,
                )
            )

    out_degree: Dict[str, int] = {}
    for edge in edges:
        out_degree[edge.from_trip_id] = out_degree.get(edge.from_trip_id, 0) + 1
    for edge in edges:
        edge.candidate = out_degree[edge.from_trip_id] > 1
    return edges


def _detect_conflicts(
    trips: Dict[str, TripOperatingInfo], edges: List[BlockEdge]
) -> List[BlockConflict]:
    conflicts: List[BlockConflict] = []
    operating = [info for info in trips.values() if info.operates]

    # 1. Two trips of one block actually being driven at the same time.
    #    Touching windows (arrival == departure at the connection stop) are
    #    the valid succession and therefore do not count as overlap.
    all_dates = sorted(
        {day for info in operating for day in info.operating_dates}
    )
    for day in all_dates:
        running = [
            info
            for info in operating
            if day in info.operating_dates
            and info.start_time is not None
            and info.end_time is not None
        ]
        for i, left in enumerate(running):
            for right in running[i + 1:]:
                if (
                    left.start_time < right.end_time
                    and right.start_time < left.end_time
                ):
                    conflicts.append(
                        BlockConflict(
                            conflict_type="simultaneous_operation",
                            service_date=day,
                            trip_ids=[left.trip_id, right.trip_id],
                            detail=(
                                "trips overlap on {0}: {1} {2}-{3}, "
                                "{4} {5}-{6}".format(
                                    day.isoformat(),
                                    left.trip_id,
                                    left.start_time,
                                    left.end_time,
                                    right.trip_id,
                                    right.start_time,
                                    right.end_time,
                                )
                            ),
                        )
                    )

    # 2. One trip has several feasible successors / predecessors that all run
    #    on the same day: the vehicle cannot take (or come from) more than one.
    outgoing: Dict[str, List[BlockEdge]] = {}
    incoming: Dict[str, List[BlockEdge]] = {}
    for edge in edges:
        outgoing.setdefault(edge.from_trip_id, []).append(edge)
        incoming.setdefault(edge.to_trip_id, []).append(edge)

    for trip_id, alternatives in outgoing.items():
        if len(alternatives) < 2:
            continue
        for day in all_dates:
            active = [
                edge.to_trip_id
                for edge in alternatives
                if day in edge.valid_dates
            ]
            if len(active) > 1:
                conflicts.append(
                    BlockConflict(
                        conflict_type="forking_succession",
                        service_date=day,
                        trip_ids=[trip_id] + active,
                        detail=(
                            "trip {0} has {1} feasible successors running on "
                            "{2}: {3}".format(
                                trip_id,
                                len(active),
                                day.isoformat(),
                                ", ".join(active),
                            )
                        ),
                    )
                )

    for trip_id, alternatives in incoming.items():
        if len(alternatives) < 2:
            continue
        for day in all_dates:
            active = [
                edge.from_trip_id
                for edge in alternatives
                if day in edge.valid_dates
            ]
            if len(active) > 1:
                conflicts.append(
                    BlockConflict(
                        conflict_type="merging_succession",
                        service_date=day,
                        trip_ids=active + [trip_id],
                        detail=(
                            "trip {0} has {1} feasible predecessors running on "
                            "{2}: {3}".format(
                                trip_id,
                                len(active),
                                day.isoformat(),
                                ", ".join(active),
                            )
                        ),
                    )
                )
    return conflicts


def _enumerate_chains(
    trips: Dict[str, TripOperatingInfo], edges: List[BlockEdge]
) -> List[BlockChain]:
    """Enumerate every maximal path; alternatives are never collapsed."""
    outgoing: Dict[str, List[BlockEdge]] = {}
    incoming: Dict[str, List[BlockEdge]] = {}
    for edge in edges:
        outgoing.setdefault(edge.from_trip_id, []).append(edge)
        incoming.setdefault(edge.to_trip_id, []).append(edge)

    ambiguous_nodes = {
        trip_id
        for trip_id, alternatives in outgoing.items()
        if len(alternatives) > 1
    }
    ambiguous_nodes.update(
        trip_id
        for trip_id, alternatives in incoming.items()
        if len(alternatives) > 1
    )

    chains: List[BlockChain] = []
    operating_ids = [
        info.trip_id
        for info in trips.values()
        if info.operates
        and info.first_stop is not None
        and info.last_stop is not None
        and info.start_time is not None
        and info.end_time is not None
    ]
    roots = [trip_id for trip_id in operating_ids if trip_id not in incoming]

    def walk(current: str, path: List[str], links: List[ChainLink],
             seen: set) -> None:
        if current in seen:
            # Defensive: timing rules make cycles impossible on clean data.
            chains.append(
                BlockChain(
                    trip_ids=list(path),
                    operating_dates=_intersection(
                        [trips[t].operating_dates for t in path]
                    ),
                    links=list(links),
                    ambiguous=True,
                )
            )
            return
        seen = seen | {current}
        successors = outgoing.get(current, [])
        if not successors:
            chains.append(
                BlockChain(
                    trip_ids=list(path),
                    operating_dates=_intersection(
                        [trips[t].operating_dates for t in path]
                    ),
                    links=list(links),
                    ambiguous=any(t in ambiguous_nodes for t in path),
                )
            )
            return
        for edge in successors:
            links.append(
                ChainLink(
                    from_trip_id=edge.from_trip_id,
                    to_trip_id=edge.to_trip_id,
                    connection_stop_id=edge.connection_stop_id,
                    arrival_time=edge.arrival_time,
                    departure_time=edge.departure_time,
                    dwell=edge.dwell,
                    valid_dates=list(edge.valid_dates),
                )
            )
            walk(edge.to_trip_id, path + [edge.to_trip_id], links, seen)
            links.pop()

    for root in roots:
        walk(root, [root], [], set())
    return chains


def build_block_chain_reports(
    schedule,
    feed_id: Optional[int] = None,
    block_ids: Optional[List[str]] = None,
) -> List[BlockChainReport]:
    """Build one :class:`BlockChainReport` per ``(feed_id, block_id)``.

    :param schedule: a loaded :class:`pygtfs.Schedule`.
    :param feed_id: restrict to one feed; defaults to every loaded feed.
    :param block_ids: restrict to these block ids; defaults to every block.
    """
    service_dates = _service_date_map(schedule.session)

    trip_query = schedule.session.query(Trip).filter(Trip.block_id.isnot(None))
    if feed_id is not None:
        trip_query = trip_query.filter(Trip.feed_id == feed_id)
    if block_ids is not None:
        trip_query = trip_query.filter(Trip.block_id.in_(block_ids))

    grouped: Dict[Tuple[int, str], Dict[str, TripOperatingInfo]] = {}
    excluded: Dict[Tuple[int, str], List[ExcludedTrip]] = {}

    for trip in trip_query.all():
        key = (trip.feed_id, trip.block_id)
        info = _load_trip_info(trip, service_dates)

        if not info.operating_dates:
            reason = (
                "service_undefined"
                if (trip.feed_id, trip.service_id) not in service_dates
                else "no_service_days"
            )
            excluded.setdefault(key, []).append(
                ExcludedTrip(
                    trip_id=trip.trip_id,
                    service_id=trip.service_id,
                    reason=reason,
                )
            )
            continue
        if not info.stop_times or info.start_time is None or info.end_time is None:
            excluded.setdefault(key, []).append(
                ExcludedTrip(
                    trip_id=trip.trip_id,
                    service_id=trip.service_id,
                    reason="missing_stop_time_data",
                )
            )
            continue
        grouped.setdefault(key, {})[trip.trip_id] = info

    reports: List[BlockChainReport] = []
    all_keys = sorted(
        set(grouped) | set(excluded),
        key=lambda item: (item[0], str(item[1])),
    )
    for key in all_keys:
        current_feed_id, block_id = key
        trips = grouped.get(key, {})
        notes: List[str] = []
        edges = _build_edges(trips, notes)
        conflicts = _detect_conflicts(trips, edges)
        chains = _enumerate_chains(trips, edges)
        reports.append(
            BlockChainReport(
                feed_id=current_feed_id,
                block_id=block_id,
                trips=trips,
                edges=edges,
                chains=chains,
                conflicts=conflicts,
                excluded_trips=excluded.get(key, []),
                notes=notes,
            )
        )
    return reports


def block_chains_as_dicts(
    schedule,
    feed_id: Optional[int] = None,
    block_ids: Optional[List[str]] = None,
) -> List[dict]:
    """Build every report and serialize it to JSON-ready plain dictionaries."""
    return [
        report_to_dict(report)
        for report in build_block_chain_reports(
            schedule, feed_id=feed_id, block_ids=block_ids
        )
    ]


def report_to_dict(report: BlockChainReport) -> dict:
    """Serialize a report to plain built-ins (for JSON/CSV export)."""

    def timedelta_value(value: Optional[timedelta]) -> Optional[int]:
        return None if value is None else int(value.total_seconds())

    return {
        "feed_id": report.feed_id,
        "block_id": report.block_id,
        "trips": {
            trip_id: {
                "route_id": info.route_id,
                "service_id": info.service_id,
                "operating_dates": [d.isoformat() for d in info.operating_dates],
                "start_time": timedelta_value(info.start_time),
                "end_time": timedelta_value(info.end_time),
                "first_stop": None
                if info.first_stop is None
                else {
                    "stop_sequence": info.first_stop.stop_sequence,
                    "stop_id": info.first_stop.stop_id,
                    "arrival_time": timedelta_value(info.first_stop.arrival_time),
                    "departure_time": timedelta_value(
                        info.first_stop.departure_time
                    ),
                },
                "last_stop": None
                if info.last_stop is None
                else {
                    "stop_sequence": info.last_stop.stop_sequence,
                    "stop_id": info.last_stop.stop_id,
                    "arrival_time": timedelta_value(info.last_stop.arrival_time),
                    "departure_time": timedelta_value(
                        info.last_stop.departure_time
                    ),
                },
                "stop_times": [
                    {
                        "stop_sequence": st.stop_sequence,
                        "stop_id": st.stop_id,
                        "arrival_time": timedelta_value(st.arrival_time),
                        "departure_time": timedelta_value(st.departure_time),
                    }
                    for st in info.stop_times
                ],
                "warnings": list(info.warnings),
            }
            for trip_id, info in sorted(report.trips.items())
        },
        "edges": [
            {
                "from_trip_id": edge.from_trip_id,
                "to_trip_id": edge.to_trip_id,
                "connection_stop_id": edge.connection_stop_id,
                "arrival_time": timedelta_value(edge.arrival_time),
                "departure_time": timedelta_value(edge.departure_time),
                "dwell": timedelta_value(edge.dwell),
                "valid_dates": [d.isoformat() for d in edge.valid_dates],
                "candidate": edge.candidate,
            }
            for edge in report.edges
        ],
        "chains": [
            {
                "trip_ids": list(chain.trip_ids),
                "operating_dates": [d.isoformat() for d in chain.operating_dates],
                "ambiguous": chain.ambiguous,
                "links": [
                    {
                        "from_trip_id": link.from_trip_id,
                        "to_trip_id": link.to_trip_id,
                        "connection_stop_id": link.connection_stop_id,
                        "arrival_time": timedelta_value(link.arrival_time),
                        "departure_time": timedelta_value(link.departure_time),
                        "dwell": timedelta_value(link.dwell),
                        "valid_dates": [d.isoformat() for d in link.valid_dates],
                    }
                    for link in chain.links
                ],
            }
            for chain in report.chains
        ],
        "conflicts": [
            {
                "conflict_type": conflict.conflict_type,
                "service_date": None
                if conflict.service_date is None
                else conflict.service_date.isoformat(),
                "trip_ids": list(conflict.trip_ids),
                "detail": conflict.detail,
            }
            for conflict in report.conflicts
        ],
        "excluded_trips": [
            {
                "trip_id": item.trip_id,
                "service_id": item.service_id,
                "reason": item.reason,
            }
            for item in report.excluded_trips
        ],
        "notes": list(report.notes),
    }
