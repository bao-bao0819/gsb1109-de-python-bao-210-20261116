# -*- coding: utf-8 -*-
"""Provable haplotype fragment reconstruction from normalized small variants.

The reconstruction only joins alleles that are demonstrably carried by the
same chromosome copy:

* within one phase block (records sharing a ``PS`` FORMAT value; a phased
  record without ``PS`` only proves phasing of its own alleles), genotype
  slots with the same index are joined, and
* across phase block boundaries, slots are only joined through user-provided
  :py:class:`CrossBlockAnchor` objects.

Unphased calls (``/``), phase block boundaries without anchors, no-call
alleles (``.``) and alleles without a concrete sequence (symbolic alleles,
breakends) are never joined into a definite sequence.  They are kept as
stand-alone fragments or :py:class:`UnknownRegion` objects, so unknown
contributions of polyploid and partially missing calls are preserved.

Every emitted base keeps a :py:class:`Provenance` entry pointing back to the
reference range and the VCF allele it came from.  Insertions and deletions
shift output coordinates, but the provenance keeps both coordinate systems,
so every fragment can be traced back without fabricating a definite sequence.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence

from vcfpy.exceptions import VCFPyException
from vcfpy.record import Call, Record, Substitution

#: Allele index used in provenance for no-call / non-sequence alleles
UNKNOWN_ALLELE = -1

#: Genotype slot used when the chromosome copy cannot be determined
UNKNOWN_SLOT = -1

#: Provenance operation: reference bases emitted unchanged
OP_MATCH = "="
#: Provenance operation: substitution with equal-length REF/ALT
OP_SUBST = "X"
#: Provenance operation: insertion (ALT longer than REF)
OP_INS = "I"
#: Provenance operation: deletion (ALT shorter than REF)
OP_DEL = "D"


class HaplotypeConflictError(VCFPyException):
    """Raised when overlapping variants cannot act on the same chromosome copy"""


@dataclass(frozen=True)
class Provenance:
    """Trace one output interval of a fragment back to a VCF allele.

    All coordinates are half-open.  ``ref_start``/``ref_end`` are 0-based
    coordinates on the reference (the variant's anchor interval),
    ``out_start``/``out_end`` are positions inside the fragment's
    ``sequence``.  Deletions consume reference bases beyond the shared
    anchor without emitting them; insertions emit bases beyond the anchor
    without consuming extra reference bases.  The ``op`` field
    disambiguates both cases.
    """

    #: The VCF record the bases come from, ``None`` for plain reference filler
    record: Optional[Record]
    #: Allele index as used in GT (``0`` is REF, ``>=1`` ALT, ``-1`` unknown)
    allele_index: int
    #: 0-based inclusive start on the reference
    ref_start: int
    #: 0-based exclusive end on the reference
    ref_end: int
    #: Inclusive start inside the fragment sequence
    out_start: int
    #: Exclusive end inside the fragment sequence
    out_end: int
    #: One of ``OP_MATCH``, ``OP_SUBST``, ``OP_INS``, ``OP_DEL``
    op: str

    def covers_out_position(self, position: int) -> bool:
        """Return whether ``position`` (fragment-local) is covered."""
        return self.out_start <= position < self.out_end

    def covers_ref_position(self, position: int) -> bool:
        """Return whether ``position`` (0-based reference) is covered."""
        return self.ref_start <= position < self.ref_end


@dataclass(frozen=True)
class CrossBlockAnchor:
    """Assert that two genotype slots at two records are one chromosome copy.

    The two records must be phased in the reconstructed call.  Anchors are the
    only mechanism allowed to connect slots from different phase blocks.
    """

    #: Index of the first record into the reconstructed record sequence
    record_a: int
    #: Genotype slot (0-based) at the first record
    slot_a: int
    #: Index of the second record into the reconstructed record sequence
    record_b: int
    #: Genotype slot (0-based) at the second record
    slot_b: int


@dataclass(frozen=True)
class HaplotypeConflict:
    """Two variants that cannot jointly act on the same chromosome copy."""

    #: Indices of the conflicting records into the reconstructed sequence
    record_indices: tuple[int, int]
    #: 1-based VCF positions of the two records
    positions: tuple[int, int]
    #: 0-based overlapping reference ranges on the two records
    ref_ranges: tuple[tuple[int, int], tuple[int, int]]
    #: Human-readable explanation
    reason: str


@dataclass(frozen=True)
class UnknownRegion:
    """A reference span whose contribution to a chromosome copy is unknown.

    This covers no-call alleles (``.``) and alleles without a concrete
    sequence such as symbolic or breakend alleles.
    """

    #: 0-based inclusive reference start
    ref_start: int
    #: 0-based exclusive reference end
    ref_end: int
    #: Genotype slot, :py:data:`UNKNOWN_SLOT` if the copy itself is unknown
    slot: int
    #: Index of the responsible record, ``None`` for filler gaps
    record_index: Optional[int]
    #: Why the contribution is unknown
    reason: str


@dataclass(frozen=True)
class PhaseBlock:
    """A phase set: records sharing one ``PS`` value (or a singleton without)."""

    #: Block identifier, stable within one reconstruction
    block_id: int
    #: Record indices belonging to the block
    record_indices: tuple[int, ...]
    #: Ploidy of the call within the block
    ploidy: int


@dataclass
class HaplotypeFragment:
    """A continuous sequence provably carried by one chromosome copy."""

    #: Chromosome name
    chrom: str
    #: Genotype slot within the call, :py:data:`UNKNOWN_SLOT` if unphased
    ploidy_slot: int
    #: Phase block identifier, ``None`` for unphased fragments
    phase_block_id: Optional[int]
    #: Identifier of the physical chromosome copy; fragments with the same id
    #: are provably the same copy (same block or joined by anchors); ``None``
    #: when the copy cannot be determined
    copy_id: Optional[int]
    #: Whether the fragment is based on phased evidence
    phased: bool
    #: 0-based inclusive reference start of the fragment
    ref_start: int
    #: 0-based exclusive reference end of the fragment
    ref_end: int
    #: Reconstructed sequence (never contains fabricated filler bases)
    sequence: str
    #: Base-level trace back to reference ranges and VCF alleles
    provenance: list[Provenance] = field(default_factory=list)

    def locate(self, out_position: int) -> Optional[Provenance]:
        """Return the provenance entry covering fragment-local ``out_position``."""
        for entry in self.provenance:
            if entry.covers_out_position(out_position):
                return entry
        return None


@dataclass
class SampleHaplotypes:
    """All haplotype evidence reconstructed for one sample."""

    #: Sample name
    sample: str
    #: Ploidy of the call (maximal genotype length seen)
    ploidy: int
    #: Reconstructed fragments, ordered by copy and reference position
    fragments: list[HaplotypeFragment] = field(default_factory=list)
    #: Spans whose contribution cannot be stated as a definite sequence
    unknown_regions: list[UnknownRegion] = field(default_factory=list)
    #: Overlaps that cannot jointly act on one chromosome copy
    conflicts: list[HaplotypeConflict] = field(default_factory=list)
    #: Phase blocks observed in the call
    phase_blocks: list[PhaseBlock] = field(default_factory=list)


@dataclass(frozen=True)
class _Observation:
    """One allele observation of one slot at one record."""

    record_index: int
    slot: int
    block_id: Optional[int]
    allele_index: int
    ref_start: int
    ref_end: int
    bases: Optional[str]
    definite: bool
    reason: str = ""


class _UnionFind:
    """Tiny union-find for (phase block, slot) chromosome-copy keys."""

    def __init__(self) -> None:
        self._parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(self, key: tuple[int, int]) -> tuple[int, int]:
        self._parent.setdefault(key, key)
        root = key
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[key] != root:
            self._parent[key], key = root, self._parent[key]
        return root

    def union(self, left: tuple[int, int], right: tuple[int, int]) -> None:
        self._parent[self.find(left)] = self.find(right)


def _allele_bases(record: Record, allele_index: int) -> tuple[Optional[str], bool, str]:
    """Return (bases, definite, reason) for one GT allele index."""
    if allele_index < 0:
        return None, False, "no-call allele"
    if allele_index == 0:
        return record.REF, True, ""
    alt = record.ALT[allele_index - 1]
    if isinstance(alt, Substitution):
        return alt.value, True, ""
    return None, False, "non-substitution allele {}".format(alt.serialize())


def _classify_op(bases: str, ref: str) -> str:
    if bases == ref:
        return OP_MATCH
    if len(bases) == len(ref):
        return OP_SUBST
    if len(bases) > len(ref):
        return OP_INS
    return OP_DEL


def _assign_phase_blocks(
    records: Sequence[Record], sample: str
) -> tuple[dict[int, int], list[PhaseBlock], int]:
    """Assign phase block ids to phased records of one call.

    Phase blocks follow the VCF ``PS`` format field: records sharing a ``PS``
    value are one block.  A phased record (``|``) without ``PS`` only proves
    phasing of its own alleles and gets a singleton block, so vertical bars
    alone never join a whole chromosome into two copies.

    Returns a mapping ``record_index -> block_id``, the block list, and the
    ploidy (maximal genotype length).
    """
    block_of_record: dict[int, int] = {}
    ps_to_block: dict[int, int] = {}
    block_members: dict[int, list[int]] = {}
    block_ploidy: dict[int, int] = {}
    next_block_id = 0
    ploidy = 0
    for index, record in enumerate(records):
        call = record.call_for_sample.get(sample)
        if call is None or not isinstance(call, Call) or call.gt_alleles is None:
            continue
        ploidy = max(ploidy, call.ploidy or 0)
        if not call.is_phased:
            continue
        ps_value = call.data.get("PS")
        if ps_value is None:
            block_id = next_block_id
            next_block_id += 1
        else:
            if isinstance(ps_value, (list, tuple)):
                ps_value = ps_value[0]
            try:
                ps_key = int(ps_value)
            except (TypeError, ValueError):
                # Unparseable / missing PS: the record only proves its own
                # within-record phasing and gets a singleton block.
                block_id = next_block_id
                next_block_id += 1
            else:
                if ps_key not in ps_to_block:
                    ps_to_block[ps_key] = next_block_id
                    next_block_id += 1
                block_id = ps_to_block[ps_key]
        block_of_record[index] = block_id
        block_members.setdefault(block_id, []).append(index)
        block_ploidy[block_id] = max(block_ploidy.get(block_id, 0), call.ploidy or 0)
    blocks = [
        PhaseBlock(block_id=block_id, record_indices=tuple(members), ploidy=block_ploidy[block_id])
        for block_id, members in sorted(block_members.items())
    ]
    return block_of_record, blocks, ploidy


def _collect_observations(
    records: Sequence[Record],
    sample: str,
    block_of_record: dict[int, int],
) -> list[_Observation]:
    observations: list[_Observation] = []
    for index, record in enumerate(records):
        call = record.call_for_sample.get(sample)
        if call is None or not isinstance(call, Call) or call.gt_alleles is None:
            continue
        block_id = block_of_record.get(index)
        for slot, allele_index in enumerate(call.gt_alleles):
            effective_allele = UNKNOWN_ALLELE if allele_index is None else allele_index
            bases, definite, reason = _allele_bases(record, effective_allele)
            observations.append(
                _Observation(
                    record_index=index,
                    slot=slot,
                    block_id=block_id,
                    allele_index=effective_allele,
                    ref_start=record.begin,
                    ref_end=record.begin + len(record.REF),
                    bases=bases,
                    definite=definite,
                    reason=reason,
                )
            )
    return observations


def _build_copy_groups(
    observations: Sequence[_Observation],
    anchors: Sequence[CrossBlockAnchor],
    block_of_record: dict[int, int],
    records: Sequence[Record],
    sample: str,
) -> _UnionFind:
    union_find = _UnionFind()
    for observation in observations:
        if observation.block_id is not None:
            union_find.find((observation.block_id, observation.slot))
    record_by_index = {observation.record_index for observation in observations}
    for anchor in anchors:
        if anchor.record_a not in record_by_index or anchor.record_b not in record_by_index:
            raise ValueError("CrossBlockAnchor references unknown record index")
        block_a = block_of_record.get(anchor.record_a)
        block_b = block_of_record.get(anchor.record_b)
        if block_a is None or block_b is None:
            raise ValueError("CrossBlockAnchor requires phased records on both sides")
        ploidy_a = len(records[anchor.record_a].call_for_sample[sample].gt_alleles or [])
        ploidy_b = len(records[anchor.record_b].call_for_sample[sample].gt_alleles or [])
        if not 0 <= anchor.slot_a < ploidy_a or not 0 <= anchor.slot_b < ploidy_b:
            raise ValueError("CrossBlockAnchor references a genotype slot outside the ploidy")
        union_find.union((block_a, anchor.slot_a), (block_b, anchor.slot_b))
    return union_find


def _detect_conflicts(
    observations: Sequence[_Observation],
    records: Sequence[Record],
    union_find: _UnionFind,
) -> tuple[list[HaplotypeConflict], set[tuple[int, int]]]:
    """Find definite observations on one copy whose REF ranges overlap.

    Phased observations are grouped by the union-find root so that slots
    joined through cross-block anchors are treated as one chromosome copy.
    """
    conflicts: list[HaplotypeConflict] = []
    invalidated: set[tuple[int, int]] = set()
    by_copy: dict[tuple[int, int], list[_Observation]] = {}
    for observation in observations:
        if not observation.definite or observation.block_id is None:
            continue
        root = union_find.find((observation.block_id, observation.slot))
        by_copy.setdefault(root, []).append(observation)
    for group in by_copy.values():
        ordered = sorted(group, key=lambda item: (item.ref_start, item.ref_end))
        for left, right in zip(ordered, ordered[1:]):
            if left.ref_end > right.ref_start:
                conflicts.append(
                    HaplotypeConflict(
                        record_indices=(left.record_index, right.record_index),
                        positions=(records[left.record_index].POS, records[right.record_index].POS),
                        ref_ranges=(
                            (left.ref_start, left.ref_end),
                            (right.ref_start, right.ref_end),
                        ),
                        reason=(
                            "overlapping REF ranges cannot jointly act on one chromosome copy "
                            "without a joint allele specification"
                        ),
                    )
                )
                invalidated.add((left.record_index, left.slot))
                invalidated.add((right.record_index, right.slot))
    return conflicts, invalidated


def _emit_observation(
    observation: _Observation,
    records: Sequence[Record],
    out_cursor: int,
) -> tuple[str, Provenance, int]:
    record = records[observation.record_index]
    bases = observation.bases if observation.bases is not None else ""
    op = _classify_op(bases, record.REF)
    provenance = Provenance(
        record=record,
        allele_index=observation.allele_index,
        ref_start=observation.ref_start,
        ref_end=observation.ref_end,
        out_start=out_cursor,
        out_end=out_cursor + len(bases),
        op=op,
    )
    return bases, provenance, out_cursor + len(bases)


def _build_fragment(
    chrom: str,
    slot: int,
    block_id: Optional[int],
    copy_id: Optional[int],
    phased: bool,
    ordered: Sequence[_Observation],
    records: Sequence[Record],
    reference: Optional[str],
    blocked_ranges: Sequence[tuple[int, int, str]] = (),
) -> tuple[list[HaplotypeFragment], list[UnknownRegion]]:
    """Build fragments from one copy's observations sorted by reference position.

    ``blocked_ranges`` lists reference spans (with a reason) across which no
    reference filler may connect observations, e.g. intervening unphased or
    conflicting variants.  Such spans split fragments and are reported as
    :py:class:`UnknownRegion` instead of being silently filled.
    """
    fragments: list[HaplotypeFragment] = []
    unknown_regions: list[UnknownRegion] = []

    sequence_parts: list[str] = []
    provenance: list[Provenance] = []
    out_cursor = 0
    fragment_start: Optional[int] = None
    cursor: Optional[int] = None

    def flush() -> None:
        nonlocal sequence_parts, provenance, out_cursor, fragment_start, cursor
        if fragment_start is not None:
            fragments.append(
                HaplotypeFragment(
                    chrom=chrom,
                    ploidy_slot=slot,
                    phase_block_id=block_id,
                    copy_id=copy_id,
                    phased=phased,
                    ref_start=fragment_start,
                    ref_end=cursor if cursor is not None else fragment_start,
                    sequence="".join(sequence_parts),
                    provenance=list(provenance),
                )
            )
        sequence_parts = []
        provenance = []
        out_cursor = 0
        fragment_start = None
        cursor = None

    for observation in ordered:
        if cursor is not None and observation.ref_start > cursor:
            blockers = [
                (block_start, block_end, reason)
                for block_start, block_end, reason in blocked_ranges
                if block_start < observation.ref_start and block_end > cursor
            ]
            if blockers:
                # An intervening variant of unknown contribution splits the
                # fragment; the blocked span stays an UnknownRegion.
                gap_start, gap_end = cursor, observation.ref_start
                flush()
                for block_start, block_end, reason in blockers:
                    unknown_regions.append(
                        UnknownRegion(
                            ref_start=max(block_start, gap_start),
                            ref_end=min(block_end, gap_end),
                            slot=slot,
                            record_index=None,
                            reason=reason,
                        )
                    )
            elif reference is None:
                # No reference sequence available: do not fabricate filler.
                flush()
            else:
                filler = reference[cursor:observation.ref_start]
                sequence_parts.append(filler)
                provenance.append(
                    Provenance(
                        record=None,
                        allele_index=0,
                        ref_start=cursor,
                        ref_end=observation.ref_start,
                        out_start=out_cursor,
                        out_end=out_cursor + len(filler),
                        op=OP_MATCH,
                    )
                )
                out_cursor += len(filler)
        if fragment_start is None:
            fragment_start = observation.ref_start
        bases, entry, out_cursor = _emit_observation(observation, records, out_cursor)
        sequence_parts.append(bases)
        provenance.append(entry)
        cursor = observation.ref_end
    flush()
    return fragments, unknown_regions


def reconstruct_haplotypes(
    records: Sequence[Record],
    sample: str,
    *,
    cross_block_anchors: Sequence[CrossBlockAnchor] = (),
    reference: Optional[str] = None,
    raise_on_conflict: bool = False,
) -> SampleHaplotypes:
    """Reconstruct provable haplotype fragments for one sample.

    :param records: VCF records in VCF order, normalized small variants
        (REF/ALT sharing at least one anchor base).  All records must be on
        the same chromosome.
    :param sample: name of the sample to reconstruct.
    :param cross_block_anchors: optional assertions connecting genotype slots
        across phase block boundaries.  Nothing is joined across blocks
        without an anchor.
    :param reference: full reference sequence for the chromosome.  When
        ``None``, fragments are split at gaps between non-abutting variants
        instead of emitting filler bases that cannot be proven.
    :param raise_on_conflict: when ``True``, raise
        :py:class:`HaplotypeConflictError` instead of collecting conflicts.
    :returns: :py:class:`SampleHaplotypes` with fragments, provenance, unknown
        regions and conflicts.
    """
    records = list(records)
    chroms = {record.CHROM for record in records}
    if len(chroms) > 1:
        raise ValueError("All records must be on the same chromosome, got {}".format(sorted(chroms)))
    known_samples = {call.sample for record in records for call in record.calls if isinstance(call, Call)}
    if records and sample not in known_samples:
        raise ValueError("Sample {!r} not present in any record".format(sample))

    block_of_record, phase_blocks, ploidy = _assign_phase_blocks(records, sample)
    observations = _collect_observations(records, sample, block_of_record)
    union_find = _build_copy_groups(observations, cross_block_anchors, block_of_record, records, sample)

    conflicts, invalidated = _detect_conflicts(observations, records, union_find)
    if raise_on_conflict and conflicts:
        first = conflicts[0]
        raise HaplotypeConflictError(
            "conflicting variants at records {} (POS {}) and {} (POS {}): {}".format(
                first.record_indices[0],
                first.positions[0],
                first.record_indices[1],
                first.positions[1],
                first.reason,
            )
        )

    root_to_copy_id: dict[tuple[int, int], int] = {}
    fragments: list[HaplotypeFragment] = []
    unknown_regions: list[UnknownRegion] = []
    chrom = records[0].CHROM if records else ""

    # Reference spans that must not be bridged with reference filler for a
    # given chromosome copy: intervening unphased records and records whose
    # contribution was invalidated by a conflict.
    all_roots = {
        union_find.find((observation.block_id, observation.slot))
        for observation in observations
        if observation.block_id is not None
    }
    blockers_by_root: dict[tuple[int, int], list[tuple[int, int, str]]] = {}
    for index, record in enumerate(records):
        call = record.call_for_sample.get(sample)
        if call is None or not isinstance(call, Call) or call.gt_alleles is None:
            continue
        ref_range = (record.begin, record.begin + len(record.REF))
        if index not in block_of_record:
            for root in all_roots:
                blockers_by_root.setdefault(root, []).append(
                    (*ref_range, "intervening unphased record at POS {}".format(record.POS))
                )
        for obs_slot, _allele in enumerate(call.gt_alleles):
            if (index, obs_slot) in invalidated:
                block_id = block_of_record[index]
                root = union_find.find((block_id, obs_slot))
                blockers_by_root.setdefault(root, []).append(
                    (*ref_range, "conflicting overlapping variant at POS {}".format(record.POS))
                )

    # No-call / non-sequence phased observations block reference filler for
    # their own chromosome copy.
    for observation in observations:
        if observation.block_id is not None and not observation.definite:
            root = union_find.find((observation.block_id, observation.slot))
            blockers_by_root.setdefault(root, []).append(
                (observation.ref_start, observation.ref_end, observation.reason)
            )

    # Phased observations: grouped into chromosome copies, including the
    # unknown contributions that split their fragments.
    phased_groups: dict[tuple[int, int], list[_Observation]] = {}
    invalidated_by_root: dict[tuple[int, int], list[_Observation]] = {}
    indefinite_by_root: dict[tuple[int, int], list[_Observation]] = {}
    for observation in observations:
        if observation.block_id is None:
            continue
        root = union_find.find((observation.block_id, observation.slot))
        if (observation.record_index, observation.slot) in invalidated:
            invalidated_by_root.setdefault(root, []).append(observation)
        elif not observation.definite:
            indefinite_by_root.setdefault(root, []).append(observation)
        else:
            phased_groups.setdefault(root, []).append(observation)
    all_copy_roots = set(phased_groups) | set(invalidated_by_root) | set(indefinite_by_root)
    for root in sorted(all_copy_roots):
        copy_id = root_to_copy_id.setdefault(root, len(root_to_copy_id))
        group = phased_groups.get(root, [])
        group_blocks = {observation.block_id for observation in group if observation.block_id is not None}
        group_slots = {observation.slot for observation in group}
        # A fragment spanning anchor-joined blocks/slots has no single block id
        # or slot; copy_id still identifies the physical chromosome copy.
        block_id = next(iter(group_blocks)) if len(group_blocks) == 1 else None
        slot = next(iter(group_slots)) if len(group_slots) == 1 else UNKNOWN_SLOT
        for observation in invalidated_by_root.get(root, []):
            unknown_regions.append(
                UnknownRegion(
                    ref_start=observation.ref_start,
                    ref_end=observation.ref_end,
                    slot=observation.slot,
                    record_index=observation.record_index,
                    reason="excluded due to overlapping variant conflict",
                )
            )
        for observation in indefinite_by_root.get(root, []):
            unknown_regions.append(
                UnknownRegion(
                    ref_start=observation.ref_start,
                    ref_end=observation.ref_end,
                    slot=observation.slot,
                    record_index=observation.record_index,
                    reason=observation.reason,
                )
            )
        if not group:
            continue
        ordered = sorted(group, key=lambda item: (item.ref_start, item.ref_end))
        group_fragments, group_unknowns = _build_fragment(
            chrom=chrom,
            slot=slot,
            block_id=block_id,
            copy_id=copy_id,
            phased=True,
            ordered=ordered,
            records=records,
            reference=reference,
            blocked_ranges=blockers_by_root.get(root, []),
        )
        fragments.extend(group_fragments)
        unknown_regions.extend(group_unknowns)

    # Unphased observations: each record is evidence on its own.  Homozygous
    # calls yield one fragment because every copy carries the same allele;
    # heterozygous calls yield one fragment per distinct allele without
    # claiming which chromosome copy carries it.
    unphased_by_record: dict[int, list[_Observation]] = {}
    for observation in observations:
        if observation.block_id is None:
            unphased_by_record.setdefault(observation.record_index, []).append(observation)
    for record_index in sorted(unphased_by_record):
        group = unphased_by_record[record_index]
        emitted_alleles: set[int] = set()
        emitted_unknowns: set[tuple[int, int, str]] = set()
        for observation in group:
            if not observation.definite:
                unknown_key = (observation.ref_start, observation.ref_end, observation.reason)
                if unknown_key in emitted_unknowns:
                    continue
                emitted_unknowns.add(unknown_key)
                unknown_regions.append(
                    UnknownRegion(
                        ref_start=observation.ref_start,
                        ref_end=observation.ref_end,
                        slot=UNKNOWN_SLOT,
                        record_index=observation.record_index,
                        reason=observation.reason,
                    )
                )
                continue
            if observation.allele_index in emitted_alleles:
                continue
            emitted_alleles.add(observation.allele_index)
            record_fragments, _ = _build_fragment(
                chrom=chrom,
                slot=UNKNOWN_SLOT,
                block_id=None,
                copy_id=None,
                phased=False,
                ordered=[observation],
                records=records,
                reference=reference,
            )
            fragments.extend(record_fragments)

    fragments.sort(key=lambda item: (item.copy_id is None, item.copy_id or 0, item.ref_start))
    unknown_regions.sort(key=lambda item: (item.ref_start, item.ref_end))
    return SampleHaplotypes(
        sample=sample,
        ploidy=ploidy,
        fragments=fragments,
        unknown_regions=unknown_regions,
        conflicts=conflicts,
        phase_blocks=phase_blocks,
    )
