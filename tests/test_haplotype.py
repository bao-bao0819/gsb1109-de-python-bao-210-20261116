# -*- coding: utf-8 -*-
"""Tests for provable haplotype fragment reconstruction."""

import pytest

import vcfpy
from vcfpy.haplotype import (
    OP_DEL,
    OP_INS,
    OP_MATCH,
    OP_SUBST,
    UNKNOWN_SLOT,
)


def make_record(pos, ref, alts, gt, sample="S", chrom="chr1", ps=None):
    """Build a Record with one phased/unphased genotype call.

    ``ps`` sets the VCF ``PS`` FORMAT field that defines the phase set;
    phased records without ``ps`` only prove within-record phasing.
    """
    format_keys = ["GT"]
    data = {"GT": gt}
    if ps is not None:
        format_keys.append("PS")
        data["PS"] = ps
    return vcfpy.Record(
        CHROM=chrom,
        POS=pos,
        ID=[],
        REF=ref,
        ALT=[vcfpy.Substitution(var_type, value) for var_type, value in alts],
        QUAL=None,
        FILTER=[],
        INFO={},
        FORMAT=format_keys,
        calls=[vcfpy.Call(sample, data)],
    )


def fragment_sequences(result):
    return [(fragment.ref_start, fragment.ref_end, fragment.sequence) for fragment in result.fragments]


def test_phased_diploid_separates_copies_with_provenance():
    """Two phased SNVs produce one fragment per chromosome copy."""
    reference = "ACGTAC"
    records = [
        make_record(2, "C", [("SNV", "G")], "1|0", ps=100),
        make_record(4, "T", [("SNV", "A")], "0|1", ps=100),
    ]
    result = vcfpy.reconstruct_haplotypes(records, "S", reference=reference)

    assert result.ploidy == 2
    assert len(result.phase_blocks) == 1
    assert result.phase_blocks[0].record_indices == (0, 1)
    assert result.conflicts == []
    assert result.unknown_regions == []

    copy_zero = next(f for f in result.fragments if f.copy_id == 0)
    copy_one = next(f for f in result.fragments if f.copy_id == 1)
    assert copy_zero.phased and copy_one.phased
    assert copy_zero.sequence == "GGT"
    assert copy_one.sequence == "CGA"

    # Every emitted base traces back to a record/allele or reference filler.
    first = copy_zero.provenance[0]
    assert first.op == OP_SUBST
    assert first.record is records[0]
    assert first.allele_index == 1
    assert (first.ref_start, first.ref_end) == (1, 2)
    assert (first.out_start, first.out_end) == (0, 1)
    filler = copy_zero.provenance[1]
    assert filler.op == OP_MATCH and filler.record is None
    assert copy_zero.locate(0) is first
    assert copy_zero.locate(99) is None


def test_phase_set_groups_records_with_shared_ps_value():
    """Records sharing a PS value are provably one phase block."""
    records = [
        make_record(2, "C", [("SNV", "G")], "1|0", ps=100),
        make_record(5, "A", [("SNV", "T")], "1|0", ps=100),
    ]
    result = vcfpy.reconstruct_haplotypes(records, "S", reference="ACGTACGT")

    assert len(result.phase_blocks) == 1
    assert result.phase_blocks[0].record_indices == (0, 1)
    assert len(result.fragments) == 2
    assert {fragment.sequence for fragment in result.fragments} == {"GGTT", "CGTA"}


def test_fully_missing_phased_record_emits_unknown_only():
    """A completely no-called phased record leaves unknown slots, no sequence."""
    records = [
        make_record(2, "C", [("SNV", "G")], "1|0", ps=100),
        make_record(5, "A", [("SNV", "T")], ".|.", ps=100),
    ]
    result = vcfpy.reconstruct_haplotypes(records, "S", reference="ACGTACGT")

    assert len(result.unknown_regions) == 2
    assert all(region.reason == "no-call allele" for region in result.unknown_regions)
    # Only the definite first record produces fragments, split away from the gap.
    assert {fragment.sequence for fragment in result.fragments} == {"G", "C"}


def test_phase_blocks_are_not_joined_without_anchor():
    """Vertical bars alone never join separate phase blocks into two copies."""
    records = [
        make_record(2, "C", [("SNV", "G")], "1|0"),
        make_record(5, "A", [("SNV", "T")], "1|0"),
    ]
    result = vcfpy.reconstruct_haplotypes(records, "S", reference="ACGTACGT")

    assert len(result.phase_blocks) == 2
    copy_ids = {fragment.copy_id for fragment in result.fragments}
    assert len(copy_ids) == 4
    assert all(fragment.phased for fragment in result.fragments)
    # Each block contributes one fragment per slot, none spans both records.
    assert all(len(fragment.provenance) == 1 for fragment in result.fragments)


def test_cross_block_anchor_joins_only_anchored_slots():
    """An anchor joins the named slots; the other slots stay separate."""
    records = [
        make_record(2, "C", [("SNV", "G")], "1|0"),
        make_record(5, "A", [("SNV", "T")], "1|0"),
    ]
    anchor = vcfpy.CrossBlockAnchor(record_a=0, slot_a=0, record_b=1, slot_b=0)
    result = vcfpy.reconstruct_haplotypes(
        records, "S", reference="ACGTACGT", cross_block_anchors=[anchor]
    )

    assert len({fragment.copy_id for fragment in result.fragments}) == 3
    record_ids = {id(record) for record in records}
    joined = [
        fragment
        for fragment in result.fragments
        if {id(entry.record) for entry in fragment.provenance if entry.record is not None} == record_ids
    ]
    assert len(joined) == 1
    assert joined[0].sequence == "GGTT"
    # The anchor-joined fragment spans two blocks: no single block id, but the
    # slot stays meaningful because both anchored slots are slot 0.
    assert joined[0].phase_block_id is None
    assert joined[0].ploidy_slot == 0
    assert joined[0].copy_id is not None


def test_cross_block_anchor_across_different_slots_marks_slot_unknown():
    """An anchor joining different slot indices yields no definite slot label."""
    records = [
        make_record(2, "C", [("SNV", "G")], "1|0"),
        make_record(5, "A", [("SNV", "T")], "0|1"),
    ]
    anchor = vcfpy.CrossBlockAnchor(record_a=0, slot_a=0, record_b=1, slot_b=1)
    result = vcfpy.reconstruct_haplotypes(
        records, "S", reference="ACGTACGT", cross_block_anchors=[anchor]
    )
    record_ids = {id(record) for record in records}
    joined = [
        fragment
        for fragment in result.fragments
        if {id(entry.record) for entry in fragment.provenance if entry.record is not None} == record_ids
    ]
    assert len(joined) == 1
    assert joined[0].ploidy_slot == vcfpy.UNKNOWN_SLOT
    assert joined[0].phase_block_id is None


def test_unphased_record_blocks_anchor_bridge():
    """An intervening unphased variant cannot be bridged with reference filler."""
    records = [
        make_record(2, "C", [("SNV", "G")], "1|0"),
        make_record(3, "G", [("SNV", "N")], "0/1"),
        make_record(5, "A", [("SNV", "T")], "1|0"),
    ]
    anchor = vcfpy.CrossBlockAnchor(record_a=0, slot_a=0, record_b=2, slot_b=0)
    result = vcfpy.reconstruct_haplotypes(
        records, "S", reference="ACGTACGT", cross_block_anchors=[anchor]
    )

    phased_fragments = [f for f in result.fragments if f.phased]
    # The anchored copy stays split: the unphased record sits in between.
    assert all(f.ref_end <= 2 or f.ref_start >= 4 for f in phased_fragments)
    reasons = " ".join(region.reason for region in result.unknown_regions)
    assert "unphased" in reasons
    # The unphased heterozygous call yields one fragment per distinct allele.
    unphased = [f for f in result.fragments if not f.phased]
    assert {f.sequence for f in unphased} == {"G", "N"}
    assert all(f.copy_id is None and f.ploidy_slot == UNKNOWN_SLOT for f in unphased)


def test_polyploid_keeps_every_copy():
    """A triploid phased call yields one fragment per genotype slot."""
    records = [
        make_record(2, "C", [("SNV", "G"), ("SNV", "T")], "0|1|2", ps=100),
        make_record(5, "A", [("SNV", "T"), ("SNV", "C")], "2|1|0", ps=100),
    ]
    result = vcfpy.reconstruct_haplotypes(records, "S", reference="ACGTACGT")

    assert result.ploidy == 3
    assert len(result.fragments) == 3
    sequences = {fragment.sequence for fragment in result.fragments}
    assert sequences == {"CGTC", "GGTT", "TGTA"}


def test_partial_missing_call_preserves_unknown_contribution():
    """No-call alleles become UnknownRegions, definite slots still reconstruct."""
    records = [
        make_record(2, "C", [("SNV", "G")], ".|1"),
        make_record(5, "A", [("SNV", "T")], "1|."),
    ]
    result = vcfpy.reconstruct_haplotypes(records, "S", reference="ACGTACGT")

    assert len(result.unknown_regions) == 2
    assert all(region.slot in (0, 1) for region in result.unknown_regions)
    # Slot 1 at record 0 and slot 0 at record 1 are definite, each standalone.
    definite = result.fragments
    assert {fragment.sequence for fragment in definite} == {"G", "T"}
    assert all(fragment.phased for fragment in definite)


def test_symbolic_allele_is_unknown_not_fabricated():
    """Non-substitution alleles cannot yield a definite sequence."""
    record = vcfpy.Record(
        CHROM="chr1",
        POS=3,
        ID=[],
        REF="A",
        ALT=[vcfpy.SymbolicAllele("DUP")],
        QUAL=None,
        FILTER=[],
        INFO={},
        FORMAT=["GT"],
        calls=[vcfpy.Call("S", {"GT": "1|0"})],
    )
    result = vcfpy.reconstruct_haplotypes([record], "S", reference="ACGTACGT")

    assert len(result.unknown_regions) == 1
    assert "non-substitution" in result.unknown_regions[0].reason
    # Only the definite REF allele on slot 1 produces a fragment.
    assert [fragment.sequence for fragment in result.fragments] == ["A"]


def test_overlapping_variants_on_one_copy_report_conflict():
    """Overlapping REF ranges on the same copy are conflicts, not merged."""
    records = [
        make_record(2, "CG", [("MNV", "GA")], "1|.", ps=100),
        make_record(3, "GT", [("MNV", "AT")], "1|.", ps=100),
    ]
    result = vcfpy.reconstruct_haplotypes(records, "S", reference="ACGTACGT")

    assert len(result.conflicts) == 1
    conflict = result.conflicts[0]
    assert conflict.record_indices == (0, 1)
    assert conflict.positions == (2, 3)
    # The conflicting slot emits no fragment; its span is reported unknown.
    assert result.fragments == []
    assert any("conflict" in region.reason for region in result.unknown_regions)
    assert any(region.reason == "no-call allele" for region in result.unknown_regions)

    with pytest.raises(vcfpy.HaplotypeConflictError):
        vcfpy.reconstruct_haplotypes(records, "S", raise_on_conflict=True)


def test_adjacent_variants_do_not_conflict():
    """Abutting REF ranges (half-open) can jointly act on one copy."""
    records = [
        make_record(2, "CG", [("MNV", "GA")], "1|0", ps=100),
        make_record(4, "TA", [("MNV", "AT")], "1|0", ps=100),
    ]
    result = vcfpy.reconstruct_haplotypes(records, "S", reference="ACGTACGT")

    assert result.conflicts == []
    copy_zero = next(f for f in result.fragments if f.copy_id == 0)
    assert copy_zero.sequence == "GAAT"


def test_insertion_and_deletion_keep_reversible_coordinates():
    """Indels shift output positions but provenance keeps both coordinate systems."""
    reference = "ACGTACGT"
    records = [
        # INS: anchor A at 0-based [2, 3), emitted "ACG"
        make_record(3, "G", [("INS", "GCG")], "1|0", ps=100),
        # DEL: REF "TAC" at [3, 6) -> "T"
        make_record(4, "TAC", [("DEL", "T")], "1|0", ps=100),
    ]
    result = vcfpy.reconstruct_haplotypes(records, "S", reference=reference)
    copy_zero = next(f for f in result.fragments if f.copy_id == 0)

    insertion = next(p for p in copy_zero.provenance if p.op == OP_INS)
    assert insertion.record is records[0]
    assert (insertion.ref_start, insertion.ref_end) == (2, 3)
    assert (insertion.out_start, insertion.out_end) == (0, 3)
    deletion = next(p for p in copy_zero.provenance if p.op == OP_DEL)
    assert deletion.record is records[1]
    assert (deletion.ref_start, deletion.ref_end) == (3, 6)
    # The anchor base is emitted; the deleted bases consume reference only,
    # and the full REF interval still resolves through provenance.
    assert (deletion.out_start, deletion.out_end) == (3, 4)
    assert deletion.covers_ref_position(5)
    assert copy_zero.sequence == "GCGT"
    # Every output base resolves through provenance, none is fabricated.
    for position in range(len(copy_zero.sequence)):
        assert copy_zero.locate(position) is not None


def test_without_reference_gaps_split_fragments():
    """No reference means gaps cannot be filled; fragments stay separate."""
    records = [
        make_record(2, "C", [("SNV", "G")], "1|0"),
        make_record(5, "A", [("SNV", "T")], "1|0"),
    ]
    result = vcfpy.reconstruct_haplotypes(records, "S")

    # No PS: each phased record is its own block, so four standalone fragments.
    assert sorted(fragment_sequences(result)) == sorted(
        [
            (1, 2, "G"),
            (4, 5, "T"),
            (1, 2, "C"),
            (4, 5, "A"),
        ]
    )
    assert all(len(fragment.provenance) == 1 for fragment in result.fragments)


def test_unphased_fully_missing_call_is_single_unknown_region():
    """A ./. call contributes one unknown span, not one per no-call allele."""
    record = make_record(2, "C", [("SNV", "G")], "./.")
    result = vcfpy.reconstruct_haplotypes([record], "S")

    assert result.fragments == []
    assert len(result.unknown_regions) == 1
    assert result.unknown_regions[0].reason == "no-call allele"


def test_unphased_heterozygous_call_emits_standalone_alleles():
    """Unphased HET evidence never claims a chromosome copy."""
    record = make_record(2, "C", [("SNV", "G")], "0/1")
    result = vcfpy.reconstruct_haplotypes([record], "S")

    assert result.phase_blocks == []
    assert {fragment.sequence for fragment in result.fragments} == {"C", "G"}
    assert all(not fragment.phased for fragment in result.fragments)
    assert all(fragment.copy_id is None for fragment in result.fragments)


def test_anchor_validation():
    """Anchors must reference phased records and valid slots."""
    records = [
        make_record(2, "C", [("SNV", "G")], "1|0"),
        make_record(3, "G", [("SNV", "N")], "0/1"),
    ]
    with pytest.raises(ValueError, match="phased"):
        vcfpy.reconstruct_haplotypes(
            records,
            "S",
            cross_block_anchors=[vcfpy.CrossBlockAnchor(0, 0, 1, 0)],
        )
    with pytest.raises(ValueError, match="slot"):
        vcfpy.reconstruct_haplotypes(
            [make_record(2, "C", [("SNV", "G")], "1|0"), make_record(5, "A", [("SNV", "T")], "1|0")],
            "S",
            cross_block_anchors=[vcfpy.CrossBlockAnchor(0, 0, 1, 2)],
        )
    with pytest.raises(ValueError, match="unknown record"):
        vcfpy.reconstruct_haplotypes(
            records,
            "S",
            cross_block_anchors=[vcfpy.CrossBlockAnchor(0, 0, 9, 0)],
        )


def test_input_validation():
    """Reject mixed chromosomes and unknown samples."""
    records = [
        make_record(2, "C", [("SNV", "G")], "1|0", chrom="chr1"),
        make_record(2, "C", [("SNV", "G")], "1|0", chrom="chr2"),
    ]
    with pytest.raises(ValueError, match="same chromosome"):
        vcfpy.reconstruct_haplotypes(records, "S")
    with pytest.raises(ValueError, match="not present"):
        vcfpy.reconstruct_haplotypes([make_record(2, "C", [("SNV", "G")], "1|0")], "nobody")


def test_empty_input():
    """An empty record list yields an empty, well-formed result."""
    result = vcfpy.reconstruct_haplotypes([], "S")
    assert result.sample == "S"
    assert result.fragments == []
    assert result.unknown_regions == []
    assert result.conflicts == []
    assert result.phase_blocks == []


def test_substitution_operation_classification():
    """Equal-length replacement is OP_SUBST, reference allele OP_MATCH."""
    records = [
        make_record(2, "C", [("SNV", "G")], "1|0"),
    ]
    result = vcfpy.reconstruct_haplotypes(records, "S", reference="ACG")
    ops = {fragment.copy_id: fragment.provenance[0].op for fragment in result.fragments}
    assert ops == {0: OP_SUBST, 1: OP_MATCH}
