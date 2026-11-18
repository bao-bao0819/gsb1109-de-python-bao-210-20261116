"""Fare combination planning for timed journeys.

Given a journey split into timed legs, find the legal combinations of
:class:`~pygtfs.gtfs_entities.Fare` tickets that cover every leg while
respecting:

* the zone / route restrictions expressed in
  :class:`~pygtfs.gtfs_entities.FareRule` rows,
* the per-ticket transfer limit (``Fare.transfers``),
* the per-ticket validity window (``Fare.transfer_duration``).

The same ticket may cover several consecutive legs, in which case its price
is charged only once. Choosing the cheapest ticket for every leg independently
does not generally yield the cheapest journey, so the planner enumerates all
legal contiguous groupings and keeps the Pareto-optimal cost vectors. Costs in
different currencies are never added together: without a conversion rule they
are reported separately and solutions priced in different currencies are all
returned.
"""

from collections import namedtuple
from decimal import Decimal

from .gtfs_entities import Fare, FareRule


JourneyLeg = namedtuple(
    'JourneyLeg',
    ['trip_id', 'route_id', 'from_stop_id', 'to_stop_id',
     'from_zone', 'to_zone', 'departure', 'arrival'])
JourneyLeg.__doc__ = """A single leg of a journey, travelled on one trip.

``departure`` and ``arrival`` are :class:`datetime.timedelta` values measured
from noon minus 12h at the start of the service day, mirroring GTFS time
fields; either may be ``None`` when the time is unknown.
"""

Ticket = namedtuple(
    'Ticket',
    ['fare_id', 'price', 'currency', 'transfers', 'transfer_duration'])
Ticket.__doc__ = """The price and transfer constraints of one fare.

``transfers`` of ``None`` means unlimited transfers, matching an empty
``transfers`` field in ``fare_attributes.txt``. ``transfer_duration`` is
expressed in seconds; ``None`` means no time limit.
"""

TicketUse = namedtuple(
    'TicketUse',
    ['fare_id', 'covered', 'transfers_used',
     'first_departure', 'last_arrival'])
TicketUse.__doc__ = """One ticket applied to a contiguous block of legs.

``covered`` is the tuple of covered leg indices, ``transfers_used`` the number
of trip changes inside the block.
"""

FareSolution = namedtuple(
    'FareSolution', ['tickets', 'costs', 'segment_coverage'])
FareSolution.__doc__ = """A legal way to ticket the whole journey.

``tickets`` lists the :class:`TicketUse` objects, ``costs`` maps each currency
to the amount due in that currency (one charge per distinct ticket), and
``segment_coverage`` gives, for every leg, the ``fare_id`` of the ticket
covering it.
"""


class FarePlanner(object):
    """Plans legal ticket combinations against fares stored in a schedule."""

    def __init__(self, schedule, feed_id=None):
        self.schedule = schedule
        self.feed_id = self._resolve_feed_id(feed_id)

    def _resolve_feed_id(self, feed_id):
        if feed_id is not None:
            return feed_id
        feed_ids = [a_feed.feed_id for a_feed in self.schedule.feeds]
        if not feed_ids:
            raise ValueError("the schedule contains no feed")
        if len(feed_ids) > 1:
            raise ValueError(
                "the schedule contains several feeds; pass feed_id explicitly")
        return feed_ids[0]

    def plan(self, legs):
        """Return the Pareto-optimal legal ticket combinations.

        :param legs: an ordered sequence of :class:`JourneyLeg` objects. Legs
            must be chronological, and every ticket covers a contiguous block
            of legs.
        :returns: a list of :class:`FareSolution` objects sorted by cost. A
            solution dominates another when it is no more expensive in every
            currency and strictly cheaper in at least one; only
            non-dominated solutions are returned. The list is empty when no
            ticket can legally cover one of the legs.
        """
        legs = tuple(legs)
        self._validate_legs(legs)
        leg_count = len(legs)

        tickets = self._load_tickets()
        eligible = self._eligibility(tickets, legs)

        # Every block starts at the smallest uncovered leg and extends
        # contiguously, so a state always covers the prefix 0..k-1. States
        # with different prefixes have different futures and must only be
        # compared against states sharing the same prefix.
        initial = _State(costs={}, uses=(), covered=0)
        frontier = {0: [initial]}

        for prefix in range(leg_count):
            for state in frontier.get(prefix, ()):
                for ticket in tickets:
                    if prefix not in eligible[ticket.fare_id]:
                        continue
                    for indices in self._candidate_blocks(
                            ticket, legs, prefix, eligible):
                        use = self._validate_use(ticket, legs, indices)
                        if use is None:
                            continue
                        new_costs = dict(state.costs)
                        currency = ticket.currency
                        new_costs[currency] = (
                            new_costs.get(currency, Decimal(0))
                            + ticket.price)
                        new_state = _State(
                            costs=new_costs,
                            uses=state.uses + (use,),
                            covered=prefix + len(indices))
                        bucket = frontier.setdefault(new_state.covered, [])
                        frontier[new_state.covered] = self._keep_non_dominated(
                            bucket, new_state)

        complete = frontier.get(leg_count, [])
        solutions = [self._build_solution(state, leg_count)
                     for state in complete]
        solutions.sort(key=_solution_sort_key)
        return solutions

    def _load_tickets(self):
        rows = (self.schedule.session.query(Fare)
                .filter(Fare.feed_id == self.feed_id).all())
        return [Ticket(fare.fare_id, fare.price, fare.currency_type,
                       fare.transfers, fare.transfer_duration)
                for fare in rows]

    def _rules_for_fare(self, fare_id):
        return (self.schedule.session.query(FareRule)
                .filter(FareRule.feed_id == self.feed_id)
                .filter(FareRule.fare_id == fare_id).all())

    def _eligibility(self, tickets, legs):
        result = {}
        for ticket in tickets:
            fare_rules = self._rules_for_fare(ticket.fare_id)
            if not fare_rules:
                # GTFS: a fare without any fare rule applies to all services.
                result[ticket.fare_id] = frozenset(range(len(legs)))
                continue
            result[ticket.fare_id] = frozenset(
                index for index, leg in enumerate(legs)
                if any(self._rule_matches(rule, leg) for rule in fare_rules))
        return result

    @staticmethod
    def _rule_matches(rule, leg):
        if rule.route_id is not None and rule.route_id != leg.route_id:
            return False
        if rule.origin_id is not None and rule.origin_id != leg.from_zone:
            return False
        if (rule.destination_id is not None
                and rule.destination_id != leg.to_zone):
            return False
        if (rule.contains_id is not None
                and rule.contains_id not in (leg.from_zone, leg.to_zone)):
            return False
        return True

    @staticmethod
    def _candidate_blocks(ticket, legs, first, eligible):
        indices = []
        for index in range(first, len(legs)):
            if index not in eligible[ticket.fare_id]:
                break
            indices.append(index)
            yield tuple(indices)

    @staticmethod
    def _validate_legs(legs):
        if not legs:
            raise ValueError("at least one journey leg is required")
        for index, leg in enumerate(legs):
            if leg.from_stop_id == leg.to_stop_id:
                raise ValueError(
                    "leg {0} has identical origin and destination".format(index))
            if (leg.departure is not None and leg.arrival is not None
                    and leg.arrival < leg.departure):
                raise ValueError(
                    "leg {0} arrives before it departs".format(index))
        for index in range(1, len(legs)):
            previous = legs[index - 1]
            current = legs[index]
            if (previous.arrival is not None
                    and current.departure is not None
                    and current.departure < previous.arrival):
                raise ValueError(
                    "leg {0} departs before the previous leg arrives".format(
                        index))

    @staticmethod
    def _validate_use(ticket, legs, indices):
        transfers_used = 0
        for position in range(1, len(indices)):
            previous_leg = legs[indices[position - 1]]
            current_leg = legs[indices[position]]
            if previous_leg.trip_id != current_leg.trip_id:
                transfers_used += 1
        if ticket.transfers is not None and transfers_used > ticket.transfers:
            return None

        first_departure = legs[indices[0]].departure
        last_arrival = legs[indices[-1]].arrival
        if (transfers_used > 0 and ticket.transfer_duration is not None
                and first_departure is not None
                and last_arrival is not None):
            elapsed = (last_arrival - first_departure).total_seconds()
            if elapsed > ticket.transfer_duration:
                return None

        return TicketUse(
            fare_id=ticket.fare_id,
            covered=indices,
            transfers_used=transfers_used,
            first_departure=first_departure,
            last_arrival=last_arrival)

    @staticmethod
    def _keep_non_dominated(states, new_state):
        survivors = []
        for state in states:
            if _dominates(state.costs, new_state.costs):
                return states
            if not _dominates(new_state.costs, state.costs):
                survivors.append(state)
        survivors.append(new_state)
        return survivors

    @staticmethod
    def _build_solution(state, leg_count):
        segment_coverage = [None] * leg_count
        for use in state.uses:
            for index in use.covered:
                segment_coverage[index] = use.fare_id
        return FareSolution(
            tickets=state.uses,
            costs=dict(state.costs),
            segment_coverage=tuple(segment_coverage))


_State = namedtuple('_State', ['costs', 'uses', 'covered'])


def _dominates(costs_a, costs_b):
    currencies = set(costs_a) | set(costs_b)
    strictly_cheaper = False
    for currency in currencies:
        cost_a = costs_a.get(currency, Decimal(0))
        cost_b = costs_b.get(currency, Decimal(0))
        if cost_a > cost_b:
            return False
        if cost_a < cost_b:
            strictly_cheaper = True
    return strictly_cheaper


def _solution_sort_key(solution):
    currencies = tuple(sorted(solution.costs))
    return (currencies,
            tuple(str(solution.costs[currency]) for currency in currencies),
            solution.segment_coverage)


def plan_fares(schedule, legs, feed_id=None):
    """Convenience wrapper for :meth:`FarePlanner.plan`."""
    return FarePlanner(schedule, feed_id=feed_id).plan(legs)
