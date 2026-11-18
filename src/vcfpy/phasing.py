# -*- coding: utf-8 -*-
"""Grouping of phased genotype calls into traceable phase blocks.

A phase set (the VCF ``FORMAT/PS`` field) is only meaningful *within one
sample and one chromosome*.  The same integer PS value can legitimately be
reused on both sides of a shard boundary and on another chromosome, so PS
values alone cannot serve as global block identifiers.  This module builds
sample-level phase groups scoped by ``(sample, chromosome)`` and explains
where two phase sets were joined through explicit
:py:class:`PhaseBreakpoint` junctions.

Only fully phased calls (genotype separated by ``|``) carrying a ``PS``
value participate in the grouping.  Unphased calls, mixed-phased calls
(genotype containing both ``|`` and ``/``) and phased calls without a
``PS`` value never force a connection: they are excluded from the blocks
and sever the adjacency between surrounding phased calls of the same
sharded batch.  Consequently, blocks never span more than one chromosome.

Phase sets are only joined through transitions observed *within one
sharded batch* handed to :py:meth:`PhaseGrouper.add_records`.  Calls added
in separate batches (e.g. different shards) are never auto-joined across
the batch boundary, even when they are adjacent in insertion order; add
the bridging shard to make the grouping converge.
"""

import dataclasses
from typing import Iterable, Iterator

from vcfpy.record import Call, Record, UnparsedCall

__author__ = "Manuel Holtgrewe <manuel.holtgrewe@bihealth.de>"


class _DisjointSet:
    """Minimal union-find structure with path compression and union by rank."""

    def __init__(self) -> None:
        #: parent pointer for each item
        self.parent: dict[object, object] = {}
        #: rank used for union by rank
        self.rank: dict[object, int] = {}

    def add(self, item: object) -> None:
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0

    def find(self, item: object) -> object:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            nxt = self.parent[item]
            self.parent[item] = root
            item = nxt
        return root

    def union(self, left: object, right: object) -> object:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return left_root
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return left_root

    def __iter__(self) -> Iterator[object]:
        return iter(self.parent)


@dataclasses.dataclass
class PhaseBreakpoint:
    """Explicit explanation of a junction between two phase sets.

    Records why two phased calls carrying different ``PS`` values were
    joined into one :py:class:`PhaseBlock`.  A breakpoint is recorded when
    two consecutive participating calls of the same sample and chromosome
    within one sharded batch refer to different phase sets, e.g. when the
    middle shard of a chromosome bridges the phase sets of the surrounding
    shards.
    """

    #: name of the sample in which the junction was observed
    sample: str
    #: chromosome on which the junction was observed
    chrom: str
    #: 1-based POS of the record introducing the new phase set
    position: int
    #: PS value of the phase set on the left (upstream) side
    left_ps: int
    #: PS value of the phase set on the right (downstream) side
    right_ps: int
    #: record carrying ``left_ps``
    left_record: Record
    #: record carrying ``right_ps``
    right_record: Record


@dataclasses.dataclass
class PhaseBlock:
    """A traceable group of phased calls for one sample and chromosome.

    A block corresponds to one connected component of phased calls linked
    through shared phase sets (``PS``) or explicit breakpoints.  Blocks are
    always confined to a single sample and a single chromosome; equal PS
    values on different chromosomes yield different blocks.
    """

    #: name of the sample the block belongs to
    sample: str
    #: chromosome the block belongs to
    chrom: str
    #: stable, traceable identifier ``(sample, chrom, representative_ps)``
    key: tuple[str, str, int]
    #: all PS values merged into this block, in ascending order
    phase_sets: list[int]
    #: 1-based POS of the first member record
    start: int
    #: 1-based POS of the last member record
    end: int
    #: phased calls grouped into this block, in coordinate order
    calls: list[Call]
    #: explicit junctions between phase sets within this block
    breakpoints: list[PhaseBreakpoint]

    @property
    def representative_ps(self) -> int:
        """PS value used as the anchor of :py:attr:`key`."""
        return self.key[2]

    @property
    def id(self) -> str:
        """Human-readable identifier built from sample, chromosome and anchor PS."""
        return "{}@{}:PS{}".format(self.sample, self.chrom, self.representative_ps)

    def has_phase_set(self, ps: int) -> bool:
        """Return whether the block contains the given phase set ``ps``."""
        return ps in self.phase_sets

    def records(self) -> list[Record]:
        """Return the unique member records in coordinate order."""
        seen: set[int] = set()
        result: list[Record] = []
        for call in self.calls:
            site = call.site
            if site is not None and id(site) not in seen:
                seen.add(id(site))
                result.append(site)
        return result


class PhaseGrouper:
    """Incrementally group phased calls into sample-level phase blocks.

    Records are added shard by shard with :py:meth:`add_records` (or one at
    a time with :py:meth:`add_record`).  Calls sharing a phase set
    ``(sample, chromosome, PS)`` always belong to the same block.  Two
    different phase sets are joined only when a transition between them is
    observed inside one sharded batch; the junction is kept as an explicit
    :py:class:`PhaseBreakpoint`.  Adding the middle shard of a chromosome
    this way makes the grouping converge, while separate batches and
    different chromosomes are never force-joined.

    Records inside a batch are expected to be in coordinate order, as
    produced by VCF readers and sharded processing.
    """

    def __init__(self) -> None:
        #: union-find over phase set nodes ``(sample, chrom, ps)``
        self._dsu = _DisjointSet()
        #: phased calls per phase set node, in insertion order
        self._calls: dict[tuple[str, str, int], list[Call]] = {}
        #: minimum 1-based POS seen per phase set node
        self._min_pos: dict[tuple[str, str, int], int] = {}
        #: maximum 1-based POS seen per phase set node
        self._max_pos: dict[tuple[str, str, int], int] = {}
        #: all explicit junctions observed while adding records
        self._breakpoints: list[PhaseBreakpoint] = []
        #: lookup table from phase set node to the latest block build
        self._block_by_node: dict[tuple[str, str, int], PhaseBlock] = {}
        #: whether the cached block lookup has to be rebuilt
        self._stale = False

    def add_records(self, records: Iterable[Record]) -> "PhaseGrouper":
        """Add one contiguous shard of ``records`` and return ``self``.

        Consecutive participating calls of the same sample and chromosome
        within this batch may be joined through breakpoints.  No connection
        is created across batch boundaries, so feed each shard in its own
        call and add a bridging shard to converge separate phase sets.
        """
        last_in_batch: dict[tuple[str, str], tuple[str, str, int]] = {}
        for record in records:
            self._consume_record(record, last_in_batch)
        return self

    def add_record(self, record: Record) -> "PhaseGrouper":
        """Add a single :py:class:`Record` without linking across batches.

        A single record cannot introduce a phase-set transition on its own;
        use :py:meth:`add_records` for contiguous shards that bridge phase
        sets.
        """
        self._consume_record(record, {})
        return self

    def _consume_record(
        self, record: Record, last_in_batch: dict[tuple[str, str], tuple[str, str, int]]
    ) -> None:
        for call in record.calls:
            if isinstance(call, UnparsedCall):
                last_in_batch.pop((call.sample, record.CHROM), None)
                continue
            node = self._phase_set_node(call, record)
            if node is None:
                # unphased, mixed-phased or phased without PS: do not link
                last_in_batch.pop((call.sample, record.CHROM), None)
                continue
            sample, chrom, ps = node
            self._stale = True
            self._dsu.add(node)
            self._calls.setdefault(node, []).append(call)
            self._min_pos[node] = min(self._min_pos.get(node, record.POS), record.POS)
            self._max_pos[node] = max(self._max_pos.get(node, record.POS), record.POS)
            scope = (sample, chrom)
            previous = last_in_batch.get(scope)
            if previous is not None and previous != node:
                self._dsu.union(previous, node)
                self._breakpoints.append(
                    PhaseBreakpoint(
                        sample=sample,
                        chrom=chrom,
                        position=record.POS,
                        left_ps=previous[2],
                        right_ps=ps,
                        left_record=self._calls[previous][-1].site,  # type: ignore[arg-type]
                        right_record=record,
                    )
                )
            last_in_batch[scope] = node

    def _phase_set_node(self, call: Call, record: Record) -> tuple[str, str, int] | None:
        """Return the ``(sample, chrom, ps)`` node for a call or ``None``.

        Unphased and mixed-phased calls as well as calls without a usable
        ``PS`` value do not participate and never force a connection.
        """
        gt = call.data.get("GT")
        if not gt or "|" not in gt or "/" in gt:
            return None
        ps = call.data.get("PS")
        if ps is None:
            return None
        return (call.sample, record.CHROM, int(ps))

    def groups(self) -> list[PhaseBlock]:
        """Return the phase blocks, rebuilt from the current state.

        Blocks are ordered by sample, chromosome, start position and
        representative PS for deterministic iteration.
        """
        members_by_root: dict[object, list[tuple[str, str, int]]] = {}
        for node in self._calls:
            root = self._dsu.find(node)
            members_by_root.setdefault(root, []).append(node)
        blocks: list[PhaseBlock] = []
        block_by_node: dict[tuple[str, str, int], PhaseBlock] = {}
        for members in members_by_root.values():
            anchor = min(members, key=lambda node: (self._min_pos[node], node[2]))
            sample, chrom, representative_ps = anchor
            calls: list[Call] = []
            for node in members:
                calls.extend(self._calls[node])
            member_set = set(members)
            breakpoints = [
                breakpoint
                for breakpoint in self._breakpoints
                if (breakpoint.sample, breakpoint.chrom, breakpoint.left_ps) in member_set
            ]
            calls.sort(key=lambda member_call: member_call.site.POS if member_call.site is not None else 0)
            breakpoints.sort(key=lambda breakpoint: breakpoint.position)
            block = PhaseBlock(
                sample=sample,
                chrom=chrom,
                key=(sample, chrom, representative_ps),
                phase_sets=sorted(node[2] for node in members),
                start=min(self._min_pos[node] for node in members),
                end=max(self._max_pos[node] for node in members),
                calls=calls,
                breakpoints=breakpoints,
            )
            blocks.append(block)
            for node in members:
                block_by_node[node] = block
        blocks.sort(key=lambda block: (block.sample, block.chrom, block.start, block.representative_ps))
        self._block_by_node = block_by_node
        self._stale = False
        return blocks

    def group_for_phase_set(self, sample: str, chrom: str, ps: int) -> PhaseBlock | None:
        """Return the block containing ``(sample, chrom, ps)`` or ``None``."""
        if self._stale:
            self.groups()
        return self._block_by_node.get((sample, chrom, int(ps)))

    def group_for_call(self, call: Call) -> PhaseBlock | None:
        """Return the block of a phased call or ``None`` if it does not participate."""
        site = call.site
        if site is None:
            return None
        node = self._phase_set_node(call, site)
        if node is None:
            return None
        return self.group_for_phase_set(node[0], node[1], node[2])

    def __iter__(self) -> Iterator[PhaseBlock]:
        return iter(self.groups())

    def __len__(self) -> int:
        return len(self.groups())


def group_phase_blocks(records: Iterable[Record]) -> list[PhaseBlock]:
    """Convenience wrapper grouping all ``records`` as one shard.

    Equivalent to ``PhaseGrouper().add_records(records).groups()``.
    """
    return PhaseGrouper().add_records(records).groups()
