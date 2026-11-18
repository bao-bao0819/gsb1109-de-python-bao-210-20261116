# -*- coding: utf-8 -*-
"""Pairing of breakend (BND) records into structural variant breakpoint tables

Two breakend records describe the same structural variant when each one points
back at the other's anchor.  A breakend points at another breakend through the
remote tag of its breakend ALT allele, e.g. ``G]17:198982]`` or ``[2:321682[G``.
The mate relationship is only accepted when the *full* reciprocal signatures
match:

* the chromosome and 1-based position of both anchors,
* the orientation of each breakend (the side on which the connecting sequence
  is written), and
* the mate orientation, i.e. the bracket style (``[`` versus ``]``).

Consequently, two breakends that merely sit next to each other in the file or
have adjacent coordinates are never paired automatically, and the order of the
input records never affects the result.

Every :py:class:`~vcfpy.record.Record` contributes one
:py:class:`BreakendEndpoint` per breakend ALT allele.  Endpoints without a
unique reciprocal partner are reported separately as missing or ambiguous
mates, they are never forced into a pair.
"""

import dataclasses
from typing import Any, Iterable

from vcfpy.record import BreakEnd, Record

#: Status code for a breakpoint with exactly one reciprocal partner
PAIRED = "paired"
#: Status code for a breakend that no other breakend points back to
MISSING_MATE = "missing_mate"
#: Status code for a breakend that has more than one candidate partner
AMBIGUOUS_MATE = "ambiguous_mate"


@dataclasses.dataclass(frozen=True)
class BreakendEndpoint:
    """One breakend ALT allele of a BND :py:class:`~vcfpy.record.Record`

    The endpoint keeps the original VCF spelling of both the record position
    and the breakend allele (``raw_alt``) together with the parsed
    orientations, so nothing of the original notation is lost when building
    the pairing table.

    :param record: the VCF record the allele belongs to
    :param alt_index: zero-based index of the allele within ``record.ALT``
    :param raw_alt: the breakend allele as written in the VCF ALT column,
        obtained from :py:meth:`BreakEnd.serialize`
    :param chrom: chromosome of this anchor, ``record.CHROM``
    :param pos: 1-based position of this anchor, ``record.POS``
    :param orientation: orientation of this breakend, ``"+"`` or ``"-"``
    :param mate_orientation: orientation of the mate as encoded by the
        brackets, ``"+"`` for ``[`` and ``"-"`` for ``"]"``
    :param mate_chrom: chromosome the breakend points at, or ``None`` for a
        :py:class:`~vcfpy.record.SingleBreakEnd`
    :param mate_pos: 1-based position the breakend points at, or ``None``
    """

    record: Record
    alt_index: int
    raw_alt: str
    chrom: str
    pos: int
    orientation: str
    mate_orientation: str | None
    mate_chrom: str | None
    mate_pos: int | None

    @property
    def ids(self) -> list[str]:
        """Return the ID column entries of the underlying record"""
        return list(self.record.ID)

    @property
    def local_key(self) -> tuple[str, int]:
        """Identity of this anchor (chromosome, 1-based position)"""
        return (self.chrom, self.pos)

    @property
    def mate_key(self) -> tuple[str, int] | None:
        """Anchor identity this breakend points at, if any"""
        if self.mate_chrom is None or self.mate_pos is None:
            return None
        return (self.mate_chrom, self.mate_pos)


@dataclasses.dataclass(frozen=True)
class BreakendPair:
    """Two breakend endpoints that point at each other

    ``first`` and ``second`` are ordered deterministically by their anchor
    coordinates so that shuffled input yields identical pairs.  Both original
    ALT spellings and orientations are retained on the endpoints themselves.
    """

    first: BreakendEndpoint
    second: BreakendEndpoint


@dataclasses.dataclass(frozen=True)
class BreakendPairing:
    """Result of pairing a collection of BND records

    :param pairs: breakends with exactly one reciprocal partner
    :param missing_mates: breakends that no other breakend points back to
    :param ambiguous_mates: breakends with more than one candidate partner,
        together with all candidate endpoints
    """

    pairs: list[BreakendPair]
    missing_mates: list[BreakendEndpoint]
    ambiguous_mates: list[tuple[BreakendEndpoint, list[BreakendEndpoint]]]

    def as_table(self) -> list[dict[str, Any]]:
        """Flatten the pairing into one ordered mapping per breakpoint

        Each row keeps the original VCF spelling and direction of the
        breakend under the ``breakend`` key.  Paired rows carry the partner
        under ``mate``, missing-mate rows have ``mate`` set to ``None``, and
        ambiguous rows list every candidate partner in ``mate_candidates``.
        """
        rows: list[dict[str, Any]] = []
        for pair in self.pairs:
            rows.append(_row(pair.first, PAIRED, [pair.second]))
        for endpoint in self.missing_mates:
            rows.append(_row(endpoint, MISSING_MATE, []))
        for endpoint, candidates in self.ambiguous_mates:
            rows.append(_row(endpoint, AMBIGUOUS_MATE, candidates))
        return rows


def iter_breakend_endpoints(records: Iterable[Record]) -> list[BreakendEndpoint]:
    """Extract one :py:class:`BreakendEndpoint` per breakend ALT allele

    Non-BND records and non-breakend ALT alleles are ignored.
    :py:class:`~vcfpy.record.SingleBreakEnd` alleles (telomere-like breakends
    without a mate) are included with ``mate_chrom`` and ``mate_pos`` set to
    ``None`` so they surface as missing mates.
    """
    endpoints: list[BreakendEndpoint] = []
    for record in records:
        if not isinstance(record, Record):
            continue
        for alt_index, alt in enumerate(record.ALT):
            if not isinstance(alt, BreakEnd):
                continue
            endpoints.append(
                BreakendEndpoint(
                    record=record,
                    alt_index=alt_index,
                    raw_alt=alt.serialize(),
                    chrom=record.CHROM,
                    pos=record.POS,
                    orientation=alt.orientation,
                    mate_orientation=alt.mate_orientation,
                    mate_chrom=alt.mate_chrom,
                    mate_pos=alt.mate_pos,
                )
            )
    return endpoints


def pair_breakends(records: Iterable[Record]) -> BreakendPairing:
    """Pair BND records into a structural variant breakpoint table

    Two endpoints are only paired when they reciprocally point at each
    other's anchor *and* the orientations agree in both directions: an
    endpoint at anchor ``A`` pointing at ``B`` requires a candidate at ``B``
    whose anchor is ``A`` and whose orientations are the mirror of the first
    endpoint's.  Input order is irrelevant; endpoints that are merely
    adjacent in coordinate or file position are never paired.

    Endpoints with zero candidates go to ``missing_mates``, endpoints with
    more than one distinct candidate go to ``ambiguous_mates``.  Every input
    breakend endpoint ends up in exactly one of the three result buckets.
    """
    endpoints = iter_breakend_endpoints(records)

    by_anchor: dict[tuple[str, int], list[BreakendEndpoint]] = {}
    for endpoint in endpoints:
        by_anchor.setdefault(endpoint.local_key, []).append(endpoint)

    candidates: dict[int, list[BreakendEndpoint]] = {}
    for index, endpoint in enumerate(endpoints):
        candidates[index] = _find_candidates(endpoint, by_anchor)

    index_of = {id(endpoint): index for index, endpoint in enumerate(endpoints)}

    pairs: list[BreakendPair] = []
    paired_endpoint_ids: set[int] = set()
    for index, endpoint in enumerate(endpoints):
        if index in paired_endpoint_ids:
            continue
        mates = candidates[index]
        # A pair requires exactly one candidate on both sides and the
        # candidate's unique candidate must be this endpoint itself.
        if len(mates) != 1:
            continue
        mate = mates[0]
        mate_index = index_of[id(mate)]
        mate_candidates = candidates[mate_index]
        if len(mate_candidates) != 1 or mate_candidates[0] is not endpoint:
            continue
        ordered = tuple(sorted((endpoint, mate), key=_endpoint_sort_key))
        pair = BreakendPair(ordered[0], ordered[1])
        if pair not in pairs:
            pairs.append(pair)
        paired_endpoint_ids.add(index)
        paired_endpoint_ids.add(mate_index)

    missing_mates: list[BreakendEndpoint] = []
    ambiguous_mates: list[tuple[BreakendEndpoint, list[BreakendEndpoint]]] = []
    for index, endpoint in enumerate(endpoints):
        if index in paired_endpoint_ids:
            continue
        mates = candidates[index]
        if not mates:
            missing_mates.append(endpoint)
        else:
            # Either several candidates, or a single candidate that does not
            # uniquely point back: the partnership is ambiguous.
            ambiguous_mates.append((endpoint, sorted(mates, key=_endpoint_sort_key)))

    pairs.sort(key=lambda pair: (_endpoint_sort_key(pair.first), _endpoint_sort_key(pair.second)))
    missing_mates.sort(key=_endpoint_sort_key)
    ambiguous_mates.sort(key=lambda item: _endpoint_sort_key(item[0]))
    return BreakendPairing(pairs=pairs, missing_mates=missing_mates, ambiguous_mates=ambiguous_mates)


def _flip(orientation: str | None) -> str | None:
    """Return the opposite strand orientation"""
    if orientation == "+":
        return "-"
    if orientation == "-":
        return "+"
    return None


def _find_candidates(
    endpoint: BreakendEndpoint, by_anchor: dict[tuple[str, int], list[BreakendEndpoint]]
) -> list[BreakendEndpoint]:
    """Return all endpoints sitting at the mate anchor with flipped signs

    A reciprocal mate is viewed from the opposite side of the junction, so
    both orientations are the mirror of this endpoint's: the mate's own
    orientation flips this endpoint's orientation and its mate orientation
    (the bracket style) flips this endpoint's mate orientation, e.g.
    ``G]17:198982]`` (``-``/``-``) pairs with ``[2:321681[A`` (``+``/``+``).
    Mere coordinate adjacency never satisfies this test.
    """
    mate_key = endpoint.mate_key
    if mate_key is None:
        return []
    result: list[BreakendEndpoint] = []
    for candidate in by_anchor.get(mate_key, []):
        if candidate is endpoint:
            continue
        if (
            candidate.mate_key == endpoint.local_key
            and candidate.orientation == _flip(endpoint.orientation)
            and candidate.mate_orientation == _flip(endpoint.mate_orientation)
        ):
            result.append(candidate)
    return result


def _endpoint_sort_key(endpoint: BreakendEndpoint) -> tuple[Any, ...]:
    return (
        endpoint.chrom,
        endpoint.pos,
        endpoint.orientation,
        endpoint.mate_orientation or "",
        endpoint.mate_chrom or "",
        endpoint.mate_pos if endpoint.mate_pos is not None else -1,
        endpoint.raw_alt,
        endpoint.alt_index,
        id(endpoint.record),
    )


def _endpoint_dict(endpoint: BreakendEndpoint) -> dict[str, Any]:
    """Return the original spelling and direction of one endpoint"""
    return {
        "id": list(endpoint.ids),
        "chrom": endpoint.chrom,
        "pos": endpoint.pos,
        "orientation": endpoint.orientation,
        "mate_orientation": endpoint.mate_orientation,
        "raw_alt": endpoint.raw_alt,
        "mate_chrom": endpoint.mate_chrom,
        "mate_pos": endpoint.mate_pos,
        "alt_index": endpoint.alt_index,
    }


def _row(endpoint: BreakendEndpoint, status: str, mates: list[BreakendEndpoint]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "status": status,
        "breakend": _endpoint_dict(endpoint),
        "mate": None,
        "mate_candidates": [],
    }
    if len(mates) == 1:
        row["mate"] = _endpoint_dict(mates[0])
    elif len(mates) > 1:
        row["mate_candidates"] = [_endpoint_dict(mate) for mate in mates]
    return row
