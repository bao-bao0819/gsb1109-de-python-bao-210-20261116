# -*- coding: utf-8 -*-
"""Tests for pairing of BND (breakend) records into structural variant tables"""

import dataclasses
import io

import pytest

import vcfpy
from vcfpy import parser, record
from vcfpy.breakend import (
    AMBIGUOUS_MATE,
    MISSING_MATE,
    PAIRED,
    BreakendEndpoint,
    pair_breakends,
)


def make_record(chrom, pos, alts, ids=None):
    """Build a BND record with the given BreakEnd ALT alleles"""
    return vcfpy.Record(
        CHROM=chrom,
        POS=pos,
        ID=list(ids or []),
        REF="A",
        ALT=list(alts),
        QUAL=None,
        FILTER=["PASS"],
        INFO={"SVTYPE": "BND"},
    )


# A reciprocal mate pair as per VCF 4.3:
#   b1 = A[chr2:200[   at chr1:100  (orientation '-', mate orientation '+')
#   b2 = ]chr1:100]A   at chr2:200  (orientation '+', mate orientation '-')
def reciprocal_pair(ids1=("b1",), ids2=("b2",)):
    b1 = record.BreakEnd("chr2", 200, record.REVERSE, record.FORWARD, "A", True)
    b2 = record.BreakEnd("chr1", 100, record.FORWARD, record.REVERSE, "A", True)
    r1 = make_record("chr1", 100, [b1], ids1)
    r2 = make_record("chr2", 200, [b2], ids2)
    return r1, r2


def test_endpoint_extraction_keeps_raw_spelling_and_directions():
    b1 = record.BreakEnd("chr2", 200, record.REVERSE, record.FORWARD, "A", True)
    single = record.SingleBreakEnd(record.FORWARD, "T")
    substitution = vcfpy.Substitution(vcfpy.SNV, "T")
    rec = make_record("chr1", 100, [b1, single, substitution], ["r1"])
    # A non-BND record must be ignored completely
    other = vcfpy.Record(
        CHROM="chr3",
        POS=5,
        ID=[],
        REF="G",
        ALT=[vcfpy.Substitution(vcfpy.SNV, "C")],
        QUAL=None,
        FILTER=[],
        INFO={},
    )

    endpoints = vcfpy.iter_breakend_endpoints([rec, other])

    assert len(endpoints) == 2
    breakend, single_endpoint = endpoints
    assert isinstance(breakend, BreakendEndpoint)
    assert breakend.raw_alt == "A[chr2:200["
    assert (breakend.chrom, breakend.pos) == ("chr1", 100)
    assert breakend.orientation == "-"
    assert breakend.mate_orientation == "+"
    assert (breakend.mate_chrom, breakend.mate_pos) == ("chr2", 200)
    assert breakend.ids == ["r1"]
    assert breakend.alt_index == 0
    # single breakends have no mate anchor and end up as missing mates
    assert single_endpoint.mate_key is None
    assert single_endpoint.raw_alt == ".T"


def test_reciprocal_records_are_paired():
    r1, r2 = reciprocal_pair()

    result = pair_breakends([r1, r2])

    assert len(result.pairs) == 1
    assert result.missing_mates == []
    assert result.ambiguous_mates == []
    pair = result.pairs[0]
    # deterministic ordering by anchor coordinates
    assert (pair.first.chrom, pair.first.pos) == ("chr1", 100)
    assert (pair.second.chrom, pair.second.pos) == ("chr2", 200)
    # original ALT spelling and both directions are preserved
    assert pair.first.raw_alt == "A[chr2:200["
    assert pair.second.raw_alt == "]chr1:100]A"
    assert pair.first.orientation == "-"
    assert pair.first.mate_orientation == "+"
    assert pair.second.orientation == "+"
    assert pair.second.mate_orientation == "-"


def test_shuffled_input_gives_identical_result():
    r1, r2 = reciprocal_pair()
    extra1, extra2 = (
        make_record("chr7", 10, [record.BreakEnd("chr8", 20, record.FORWARD, record.REVERSE, "C", True)], ["e1"]),
        make_record("chr8", 20, [record.BreakEnd("chr7", 10, record.REVERSE, record.FORWARD, "C", True)], ["e2"]),
    )

    ordered = pair_breakends([r1, r2, extra1, extra2])
    shuffled = pair_breakends([extra2, r2, extra1, r1])

    assert ordered.as_table() == shuffled.as_table()
    assert [p.first.ids for p in ordered.pairs] == [["b1"], ["e1"]]


def test_adjacent_breakends_are_not_auto_paired():
    # Both sit at neighbouring coordinates on chr1 but point elsewhere;
    # neither points at the other's anchor.
    left = make_record(
        "chr1", 100, [record.BreakEnd("chr9", 9, record.REVERSE, record.FORWARD, "A", True)], ["left"]
    )
    neighbour = make_record(
        "chr1", 101, [record.BreakEnd("chr9", 8, record.REVERSE, record.FORWARD, "A", True)], ["neighbour"]
    )

    result = pair_breakends([left, neighbour])

    assert result.pairs == []
    assert sorted(endpoint.ids[0] for endpoint in result.missing_mates) == ["left", "neighbour"]
    assert result.ambiguous_mates == []


def test_missing_mate_is_reported_separately():
    r1, r2 = reciprocal_pair()
    lone = make_record("chr5", 500, [record.BreakEnd("chr6", 600, record.FORWARD, record.FORWARD, "G", True)], ["lone"])

    result = pair_breakends([r1, r2, lone])

    assert len(result.pairs) == 1
    assert len(result.missing_mates) == 1
    endpoint = result.missing_mates[0]
    assert endpoint.ids == ["lone"]
    assert endpoint.raw_alt == "[chr6:600[G"
    assert endpoint.mate_key == ("chr6", 600)


def test_wrong_bracket_style_prevents_pairing():
    # b1 needs a mate with orientation '+' and mate orientation '-';
    # the candidate uses mate orientation '+' (the other bracket).
    b1 = record.BreakEnd("chr2", 200, record.REVERSE, record.FORWARD, "A", True)
    wrong_bracket = record.BreakEnd("chr1", 100, record.FORWARD, record.FORWARD, "A", True)
    r1 = make_record("chr1", 100, [b1], ["b1"])
    r2 = make_record("chr2", 200, [wrong_bracket], ["b2"])

    result = pair_breakends([r1, r2])

    assert result.pairs == []
    assert {endpoint.ids[0] for endpoint in result.missing_mates} == {"b1", "b2"}


def test_wrong_direction_prevents_pairing():
    # Candidate sits at the right anchor but carries the wrong local
    # orientation (trailing sequence instead of leading).
    b1 = record.BreakEnd("chr2", 200, record.REVERSE, record.FORWARD, "A", True)
    wrong_direction = record.BreakEnd("chr1", 100, record.REVERSE, record.REVERSE, "A", True)
    r1 = make_record("chr1", 100, [b1], ["b1"])
    r2 = make_record("chr2", 200, [wrong_direction], ["b2"])

    result = pair_breakends([r1, r2])

    assert result.pairs == []
    assert len(result.missing_mates) == 2


def test_wrong_coordinate_prevents_pairing():
    b1 = record.BreakEnd("chr2", 200, record.REVERSE, record.FORWARD, "A", True)
    # points back at chr1:101 instead of chr1:100
    off_by_one = record.BreakEnd("chr1", 101, record.FORWARD, record.REVERSE, "A", True)
    r1 = make_record("chr1", 100, [b1], ["b1"])
    r2 = make_record("chr2", 200, [off_by_one], ["b2"])

    result = pair_breakends([r1, r2])

    assert result.pairs == []
    assert len(result.missing_mates) == 2


def test_ambiguous_multiple_mates_are_reported_with_all_candidates():
    b1 = record.BreakEnd("chr2", 200, record.REVERSE, record.FORWARD, "A", True)
    # Two distinct records at chr2:200, both valid reciprocal signatures
    b2a = record.BreakEnd("chr1", 100, record.FORWARD, record.REVERSE, "A", True)
    b2b = record.BreakEnd("chr1", 100, record.FORWARD, record.REVERSE, "C", True)
    r1 = make_record("chr1", 100, [b1], ["b1"])
    r2a = make_record("chr2", 200, [b2a], ["b2a"])
    r2b = make_record("chr2", 200, [b2b], ["b2b"])

    result = pair_breakends([r1, r2a, r2b])

    assert result.pairs == []
    assert result.missing_mates == []
    assert len(result.ambiguous_mates) == 3
    by_id = {endpoint.ids[0]: candidates for endpoint, candidates in result.ambiguous_mates}
    # b1 sees both candidates at its mate anchor
    assert [candidate.ids[0] for candidate in by_id["b1"]] == ["b2a", "b2b"]
    # each duplicate mate uniquely points back at b1 but the partnership is
    # still ambiguous because b1 itself has two candidates
    assert [candidate.ids[0] for candidate in by_id["b2a"]] == ["b1"]
    assert [candidate.ids[0] for candidate in by_id["b2b"]] == ["b1"]


def test_ambiguity_does_not_silently_force_a_pair():
    b1 = record.BreakEnd("chr2", 200, record.REVERSE, record.FORWARD, "A", True)
    b2a = record.BreakEnd("chr1", 100, record.FORWARD, record.REVERSE, "A", True)
    b2b = record.BreakEnd("chr1", 100, record.FORWARD, record.REVERSE, "C", True)
    records = [
        make_record("chr1", 100, [b1], ["b1"]),
        make_record("chr2", 200, [b2a], ["b2a"]),
        make_record("chr2", 200, [b2b], ["b2b"]),
    ]
    # every breakend endpoint must land in exactly one bucket
    result = pair_breakends(records)
    accounted = (
        len(result.pairs) * 2
        + len(result.missing_mates)
        + len(result.ambiguous_mates)
    )
    assert accounted == 3


def test_multiple_alt_alleles_in_one_record_are_handled_independently():
    b1 = record.BreakEnd("chr2", 200, record.REVERSE, record.FORWARD, "A", True)
    b_other = record.BreakEnd("chr8", 800, record.FORWARD, record.REVERSE, "T", True)
    multi = make_record("chr1", 100, [b1, b_other], ["multi"])
    mate = make_record("chr2", 200, [record.BreakEnd("chr1", 100, record.FORWARD, record.REVERSE, "A", True)], ["mate"])

    result = pair_breakends([multi, mate])

    assert len(result.pairs) == 1
    pair = result.pairs[0]
    assert pair.first.alt_index == 0
    assert pair.first.raw_alt == "A[chr2:200["
    assert pair.second.ids == ["mate"]
    # the second ALT allele has no reciprocal partner
    assert len(result.missing_mates) == 1
    assert result.missing_mates[0].raw_alt == "]chr8:800]T"
    assert result.missing_mates[0].alt_index == 1


def test_same_coordinate_distinct_breakends_stay_distinct():
    # Two breakends at chr1:100 pointing at two different mate anchors;
    # only one has a reciprocal partner.
    paired_alt = record.BreakEnd("chr2", 200, record.REVERSE, record.FORWARD, "A", True)
    unpaired_alt = record.BreakEnd("chr3", 300, record.REVERSE, record.FORWARD, "A", True)
    rec = make_record("chr1", 100, [paired_alt, unpaired_alt], ["r"])
    mate = make_record("chr2", 200, [record.BreakEnd("chr1", 100, record.FORWARD, record.REVERSE, "A", True)], ["m"])

    result = pair_breakends([rec, mate])

    assert len(result.pairs) == 1
    assert len(result.missing_mates) == 1
    assert result.missing_mates[0].mate_key == ("chr3", 300)


def test_table_rows_hold_original_spelling_and_directions():
    r1, r2 = reciprocal_pair()
    lone = make_record(
        "chr5", 500, [record.BreakEnd("chr6", 600, record.FORWARD, record.FORWARD, "G", True)], ["lone"]
    )

    table = pair_breakends([r2, lone, r1]).as_table()

    statuses = [row["status"] for row in table]
    assert statuses == [PAIRED, MISSING_MATE]
    paired_row = table[0]
    assert paired_row["breakend"] == {
        "id": ["b1"],
        "chrom": "chr1",
        "pos": 100,
        "orientation": "-",
        "mate_orientation": "+",
        "raw_alt": "A[chr2:200[",
        "mate_chrom": "chr2",
        "mate_pos": 200,
        "alt_index": 0,
    }
    assert paired_row["mate"]["raw_alt"] == "]chr1:100]A"
    assert paired_row["mate"]["orientation"] == "+"
    assert paired_row["mate"]["mate_orientation"] == "-"
    missing_row = table[1]
    assert missing_row["status"] == MISSING_MATE
    assert missing_row["mate"] is None
    assert missing_row["breakend"]["raw_alt"] == "[chr6:600[G"


def test_ambiguous_table_row_lists_candidates():
    b1 = record.BreakEnd("chr2", 200, record.REVERSE, record.FORWARD, "A", True)
    b2a = record.BreakEnd("chr1", 100, record.FORWARD, record.REVERSE, "A", True)
    b2b = record.BreakEnd("chr1", 100, record.FORWARD, record.REVERSE, "C", True)
    records = [
        make_record("chr1", 100, [b1], ["b1"]),
        make_record("chr2", 200, [b2a], ["b2a"]),
        make_record("chr2", 200, [b2b], ["b2b"]),
    ]

    table = pair_breakends(records).as_table()

    assert all(row["status"] == AMBIGUOUS_MATE for row in table)
    b1_row = next(row for row in table if row["breakend"]["id"] == ["b1"])
    assert b1_row["mate"] is None
    assert [candidate["id"] for candidate in b1_row["mate_candidates"]] == [["b2a"], ["b2b"]]
    assert b1_row["mate_candidates"][0]["raw_alt"] == "]chr1:100]A"


def test_pairing_does_not_mutate_input_records():
    r1, r2 = reciprocal_pair()
    snapshot = (str(r1), str(r2))

    pair_breakends([r1, r2])

    assert (str(r1), str(r2)) == snapshot


def test_empty_input():
    result = pair_breakends([])
    assert result.pairs == []
    assert result.missing_mates == []
    assert result.ambiguous_mates == []
    assert result.as_table() == []


def test_frozen_endpoints_and_pair():
    r1, r2 = reciprocal_pair()
    result = pair_breakends([r1, r2])
    endpoint = result.pairs[0].first
    with pytest.raises(dataclasses.FrozenInstanceError):
        endpoint.pos = 42
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.pairs[0].second = endpoint


HEADER = """
##fileformat=VCFv4.3
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="SV type">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tNA00001
""".lstrip()


def parse_records(body):
    p = parser.Parser(io.StringIO(HEADER + body), "<builtin>")
    p.parse_header()
    records = []
    while True:
        rec = p.parse_next_record()
        if rec is None:
            return records
        records.append(rec)


def test_end_to_end_through_vcf_parser():
    # Records are deliberately shuffled and mix every edge case:
    # one reciprocal pair, one duplicated-mate event, one orphan breakend,
    # two merely adjacent breakends, and a non-BND record.
    body = (
        "17\t198982\tp2\tC\t]2:321681]C\t6\tPASS\tSVTYPE=BND\tGT\t0/1\n"
        "6\t200\ta2a\tT\t]5:100]T\t6\tPASS\tSVTYPE=BND\tGT\t0/1\n"
        "1\t101\tn2\tA\tA[20:2[\t6\tPASS\tSVTYPE=BND\tGT\t0/1\n"
        "9\t50\tm1\tC\tC[10:60[\t6\tPASS\tSVTYPE=BND\tGT\t0/1\n"
        "6\t200\ta2b\tG\t]5:100]G\t6\tPASS\tSVTYPE=BND\tGT\t0/1\n"
        "5\t100\ta1\tA\tA[6:200[\t6\tPASS\tSVTYPE=BND\tGT\t0/1\n"
        "2\t321681\tp1\tG\tG[17:198982[\t6\tPASS\tSVTYPE=BND\tGT\t0/1\n"
        "1\t100\tn1\tA\tA[20:1[\t6\tPASS\tSVTYPE=BND\tGT\t0/1\n"
        "3\t7\t.\tG\tT\t6\tPASS\t.\tGT\t0/1\n"
    )

    result = pair_breakends(parse_records(body))

    assert len(result.pairs) == 1
    pair = result.pairs[0]
    # endpoints are ordered by the deterministic sort key, not file order
    assert (pair.first.chrom, pair.first.pos) == ("17", 198982)
    assert pair.first.raw_alt == "]2:321681]C"
    assert (pair.second.chrom, pair.second.pos) == ("2", 321681)
    assert pair.second.raw_alt == "G[17:198982["

    assert sorted(endpoint.ids[0] for endpoint in result.missing_mates) == ["m1", "n1", "n2"]
    ambiguous_ids = {endpoint.ids[0] for endpoint, _ in result.ambiguous_mates}
    assert ambiguous_ids == {"a1", "a2a", "a2b"}

    table = result.as_table()
    assert [row["status"] for row in table].count(PAIRED) == 1
    assert [row["status"] for row in table].count(MISSING_MATE) == 3
    assert [row["status"] for row in table].count(AMBIGUOUS_MATE) == 3
    # the plain SNV record contributes no breakpoint row
    assert len(table) == 7
