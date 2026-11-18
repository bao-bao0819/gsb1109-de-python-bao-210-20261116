# -*- coding: utf-8 -*-
"""Tests for sample-level phase block grouping (see :mod:`vcfpy.phasing`)."""

import io

import vcfpy

__author__ = "Manuel Holtgrewe <manuel.holtgrewe@bihealth.de>"


def make_record(chrom, pos, sample, gt, ps=None):
    """Build a one-sample record with the given GT/PS."""
    keys = ["GT"]
    data = {"GT": gt}
    if ps is not None:
        keys.append("PS")
        data["PS"] = ps
    calls = [vcfpy.Call(sample, data)]
    alt = vcfpy.Substitution(vcfpy.SNV, "T")
    return vcfpy.Record(chrom, pos, [], "A", [alt], None, [], {}, keys, calls)


def make_call(sample, gt, ps=None):
    data = {"GT": gt}
    if ps is not None:
        data["PS"] = ps
    return vcfpy.Call(sample, data)


# ---------------------------------------------------------------------------
# equal PS values must not act as global identifiers
# ---------------------------------------------------------------------------


def test_equal_ps_on_different_chromosomes_never_merged():
    records = [
        make_record("1", 100, "S", "0|1", 100),
        make_record("2", 100, "S", "0|1", 100),
    ]
    blocks = vcfpy.group_phase_blocks(records)
    assert len(blocks) == 2
    assert {block.chrom for block in blocks} == {"1", "2"}
    for block in blocks:
        assert block.sample == "S"
        assert block.phase_sets == [100]
        assert block.breakpoints == []


def test_equal_ps_scoped_by_sample():
    call1 = make_call("S1", "0|1", 100)
    call2 = make_call("S2", "0|1", 100)
    alt = vcfpy.Substitution(vcfpy.SNV, "T")
    record = vcfpy.Record("1", 100, [], "A", [alt], None, [], {}, ["GT", "PS"], [call1, call2])
    blocks = vcfpy.group_phase_blocks([record])
    assert len(blocks) == 2
    assert {block.key for block in blocks} == {("S1", "1", 100), ("S2", "1", 100)}


def test_same_phase_set_across_shard_boundary_stays_one_block():
    grouper = vcfpy.PhaseGrouper()
    grouper.add_records([make_record("1", 100, "S", "0|1", 100)])
    grouper.add_records([make_record("1", 200, "S", "1|0", 100)])
    blocks = grouper.groups()
    assert len(blocks) == 1
    assert blocks[0].phase_sets == [100]
    assert (blocks[0].start, blocks[0].end) == (100, 200)


# ---------------------------------------------------------------------------
# traceable grouping with explicit breakpoints
# ---------------------------------------------------------------------------


def test_middle_shard_converges_phase_sets_with_breakpoint():
    grouper = vcfpy.PhaseGrouper()
    # outer shards, handed in as separate batches
    grouper.add_records([make_record("1", 100, "S", "0|1", 100)])
    grouper.add_records([make_record("1", 400, "S", "1|0", 200)])
    assert len(grouper.groups()) == 2

    # bridging middle shard joins PS 100 -> PS 200 inside one batch
    grouper.add_records(
        [
            make_record("1", 200, "S", "0|1", 100),
            make_record("1", 300, "S", "1|0", 200),
        ]
    )
    blocks = grouper.groups()
    assert len(blocks) == 1
    block = blocks[0]
    assert block.key == ("S", "1", 100)
    assert block.id == "S@1:PS100"
    assert block.phase_sets == [100, 200]
    assert (block.start, block.end) == (100, 400)
    assert len(block.breakpoints) == 1
    breakpoint = block.breakpoints[0]
    assert (breakpoint.sample, breakpoint.chrom) == ("S", "1")
    assert breakpoint.position == 300
    assert (breakpoint.left_ps, breakpoint.right_ps) == (100, 200)
    assert breakpoint.left_record.POS == 200
    assert breakpoint.right_record.POS == 300


def test_separate_shards_without_bridge_are_not_force_joined():
    grouper = vcfpy.PhaseGrouper()
    grouper.add_records([make_record("1", 100, "S", "0|1", 100)])
    grouper.add_records([make_record("1", 200, "S", "1|0", 200)])
    blocks = grouper.groups()
    assert len(blocks) == 2
    assert all(block.breakpoints == [] for block in blocks)


def test_block_lookups_and_member_records_are_traceable():
    first = make_record("1", 100, "S", "0|1", 100)
    second = make_record("1", 200, "S", "1|0", 100)
    grouper = vcfpy.PhaseGrouper().add_records([first, second])
    block = grouper.group_for_phase_set("S", "1", 100)
    assert block is not None
    assert grouper.group_for_phase_set("S", "1", 999) is None
    assert grouper.group_for_call(first.calls[0]) is block
    assert block.has_phase_set(100)
    assert block.records() == [first, second]
    assert [call.site for call in block.calls] == [first, second]


def test_groups_are_deterministically_ordered():
    grouper = vcfpy.PhaseGrouper()
    # separate batches so the equal-PS, disjoint regions stay distinct blocks
    grouper.add_records([make_record("1", 300, "S2", "0|1", 300)])
    grouper.add_records([make_record("1", 100, "S2", "0|1", 100)])
    grouper.add_records([make_record("1", 100, "S1", "0|1", 100)])
    blocks = grouper.groups()
    assert [(block.sample, block.start) for block in blocks] == [
        ("S1", 100),
        ("S2", 100),
        ("S2", 300),
    ]


# ---------------------------------------------------------------------------
# unphased, mixed-phased and PS-less calls must never force a connection
# ---------------------------------------------------------------------------


def test_unphased_mixed_and_psless_calls_do_not_link():
    records = [
        make_record("1", 100, "S", "0|1", 100),
        make_record("1", 200, "S", "0/1"),
        make_record("1", 300, "S", "1|0", 200),
        make_record("1", 400, "S", "0/1|2", 200),
        make_record("1", 500, "S", "0|1"),
        make_record("1", 600, "S", "1|0", 300),
    ]
    blocks = vcfpy.group_phase_blocks(records)
    assert sorted(block.phase_sets for block in blocks) == [[100], [200], [300]]
    assert all(block.breakpoints == [] for block in blocks)
    grouper = vcfpy.PhaseGrouper().add_records(records)
    for record in records:
        if record.POS in (200, 400, 500):
            assert grouper.group_for_call(record.calls[0]) is None
        else:
            assert grouper.group_for_call(record.calls[0]) is not None


def test_unparsed_calls_are_skipped_and_sever_links():
    stream = io.StringIO(
        "##fileformat=VCFv4.3\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        '##FORMAT=<ID=PS,Number=1,Type=Integer,Description="Phase set">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\n"
        "1\t100\t.\tA\tT\t.\t.\t.\tGT:PS\t0|1:100\t0|1:100\n"
        "1\t200\t.\tA\tT\t.\t.\t.\tGT:PS\t1|0:100\t1|0:200\n"
    )
    reader = vcfpy.Reader.from_stream(stream, parsed_samples=["S1"])
    records = list(reader)
    assert isinstance(records[0].call_for_sample["S2"], vcfpy.UnparsedCall)
    blocks = vcfpy.PhaseGrouper().add_records(records).groups()
    # S2 is unparsed: only S1 participates and no S2 call leaks into a block
    assert len(blocks) == 1
    block = blocks[0]
    assert block.sample == "S1"
    assert block.phase_sets == [100]
    assert {call.sample for call in block.calls} == {"S1"}


# ---------------------------------------------------------------------------
# end-to-end parsing and container behaviour
# ---------------------------------------------------------------------------


def test_grouping_from_parsed_vcf():
    stream = io.StringIO(
        "##fileformat=VCFv4.3\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        '##FORMAT=<ID=PS,Number=1,Type=Integer,Description="Phase set">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS\n"
        "1\t100\t.\tA\tT\t.\t.\t.\tGT:PS\t0|1:100\n"
        "1\t200\t.\tA\tT\t.\t.\t.\tGT:PS\t1|0:100\n"
        "1\t300\t.\tA\tT\t.\t.\t.\tGT:PS\t0|1:200\n"
        "2\t100\t.\tA\tT\t.\t.\t.\tGT:PS\t0|1:100\n"
    )
    reader = vcfpy.Reader.from_stream(stream)
    grouper = vcfpy.PhaseGrouper().add_records(reader)
    blocks = grouper.groups()
    # chr1: PS 100 -> PS 200 joined within the batch; chr2 stays independent
    assert len(blocks) == 2
    chr1 = next(block for block in blocks if block.chrom == "1")
    chr2 = next(block for block in blocks if block.chrom == "2")
    assert chr1.phase_sets == [100, 200]
    assert len(chr1.breakpoints) == 1
    assert chr2.phase_sets == [100]
    assert len(grouper) == 2
    assert list(grouper) == blocks


def test_empty_input():
    grouper = vcfpy.PhaseGrouper()
    assert grouper.groups() == []
    assert len(grouper) == 0
    assert list(grouper) == []
    assert grouper.group_for_phase_set("S", "1", 1) is None
