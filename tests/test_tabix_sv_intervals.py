# -*- coding: utf-8 -*-
"""Tests for SV-aware tabix span parsing and definite/possible fetch hits."""

import gzip
import io
import os
import struct

import pytest

from vcfpy import Reader
from vcfpy.bgzf import BgzfWriter, make_virtual_offset
from vcfpy.tabix import TabixFile, classify_record_overlap, parse_record_span

HEADER = """##fileformat=VCFv4.2
##INFO=<ID=END,Number=1,Type=Integer,Description="end">
##INFO=<ID=CIPOS,Number=2,Type=Integer,Description="ci pos">
##INFO=<ID=CIEND,Number=2,Type=Integer,Description="ci end">
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="type">
##ALT=<ID=DEL,Description="deletion">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
"""

# Spanning DEL with confidence intervals, an insertion point and an SNV.
RECORD_DEL = "1\t1000\t.\tN\t<DEL>\t.\tPASS\tSVTYPE=DEL;END=2000;CIPOS=-100,50;CIEND=-50,100\n"
RECORD_INS = "1\t3000\t.\tA\tATG\t.\tPASS\t.\n"
RECORD_SNV = "1\t5000\t.\tG\tA\t.\tPASS\t.\n"


def write_indexed_vcf(tmp_path, records):
    """Write a tiny one-block BGZF VCF plus a matching minimal TBI index."""
    vcf_path = tmp_path / "sv.vcf.gz"
    tbi_path = tmp_path / "sv.vcf.gz.tbi"

    writer = BgzfWriter(str(vcf_path))
    data_offset = 0
    for line in HEADER.splitlines(keepends=True):
        data_offset += len(line)
    writer.write(HEADER)
    for line in records:
        writer.write(line)
    writer.close()

    # The data occupies a single BGZF block followed by the 28-byte EOF block.
    data_block_end = os.path.getsize(vcf_path) - 28
    data_start = make_virtual_offset(0, data_offset)
    chunk_end = make_virtual_offset(data_block_end, 0)

    names = b"1\x00"
    buffer = io.BytesIO()
    buffer.write(b"TBI\x01")
    for value in (1, 2, 1, 2, 0, ord("#"), 0, len(names)):
        buffer.write(struct.pack("<i", value))
    buffer.write(names)
    # One reference: a single level-5 bin with one chunk and one linear entry.
    buffer.write(struct.pack("<i", 1))
    buffer.write(struct.pack("<I", 4681))
    buffer.write(struct.pack("<i", 1))
    buffer.write(struct.pack("<Q", data_start))
    buffer.write(struct.pack("<Q", chunk_end))
    buffer.write(struct.pack("<i", 1))
    buffer.write(struct.pack("<Q", data_start))
    with gzip.open(tbi_path, "wb") as handle:
        handle.write(buffer.getvalue())

    return str(vcf_path), str(tbi_path)


@pytest.fixture
def indexed_sv_vcf(tmp_path):
    return write_indexed_vcf(tmp_path, [RECORD_DEL, RECORD_INS, RECORD_SNV])


# Span parsing ----------------------------------------------------------------


def test_span_symbolic_del_uses_end_and_confidence_intervals():
    begin, end, ci_begin, ci_end, is_breakend = parse_record_span(RECORD_DEL)
    assert (begin, end) == (1000, 2000)  # END, not the single-base REF length
    assert (ci_begin, ci_end) == (900, 2100)
    assert is_breakend is False


def test_span_symbolic_insertion_collapses_to_point():
    line = "3\t500\t.\tC\t<INS:ME:L1>\t.\tPASS\tSVTYPE=INS;END=500;CIPOS=-16,22\n"
    begin, end, ci_begin, ci_end, _ = parse_record_span(line)
    assert (begin, end) == (500, 500)  # insertion point, END ignored as a span
    assert (ci_begin, ci_end) == (484, 522)


def test_span_unknown_length_sv_does_not_use_ref_length():
    line = "1\t1000\t.\tN\t<CNV>\t.\tPASS\tSVTYPE=CNV\n"
    begin, end, ci_begin, ci_end, _ = parse_record_span(line)
    assert (begin, end) == (1000, 1000)
    assert ci_begin is None and ci_end is None


def test_span_unknown_length_imprecise_sv_uses_cipos_on_both_sides():
    # No END: the finite tabix interval falls back to the CIPOS-bounded anchor
    # region instead of pretending to know an unbounded extent.
    line = "1\t1000\t.\tN\t<CNV>\t.\tPASS\tSVTYPE=CNV;IMPRECISE;CIPOS=-50,50\n"
    begin, end, ci_begin, ci_end, _ = parse_record_span(line)
    assert (begin, end) == (1000, 1000)
    assert (ci_begin, ci_end) == (950, 1050)


def test_span_breakend_is_point_without_span():
    line = "1\t2000\t.\tA\t[2:5000[A\t.\tPASS\t.\n"
    begin, end, ci_begin, ci_end, is_breakend = parse_record_span(line)
    assert (begin, end) == (2000, 2000)
    assert is_breakend is True
    assert ci_begin is None and ci_end is None


def test_span_ordinary_variant_uses_ref_length():
    begin, end, _, _, _ = parse_record_span(RECORD_SNV)
    assert (begin, end) == (5000, 5000)
    line = "20\t100\t.\tGTC\tG,GTCT\t.\tPASS\t.\n"
    begin, end, _, _, _ = parse_record_span(line)
    assert (begin, end) == (100, 102)  # mixed alleles: not treated as insertion


def test_span_explicit_insertion_is_point():
    begin, end, _, _, _ = parse_record_span(RECORD_INS)
    assert (begin, end) == (3000, 3000)


def test_span_explicit_allele_ignores_end_and_uses_ref():
    line = (
        "1\t2827693\t.\t"
        "CCGTGGATGCGGGGACCCGCATCCCCTCTCCCTTCACAGCTGAGTGACCCACATCCCCTCTCCCCTCGCA\tC\t.\tPASS\t"
        "SVTYPE=DEL;END=2827680;SVLEN=-66\n"
    )
    begin, end, ci_begin, ci_end, is_breakend = parse_record_span(line)
    assert begin == 2827693
    assert end == begin + len(line.split("\t")[3]) - 1
    assert end > begin  # not inverted by the smaller INFO/END
    assert ci_begin is None and ci_end is None
    assert is_breakend is False


# Overlap classification ------------------------------------------------------


def test_classify_separates_definite_possible_and_no_hit():
    begin, end, ci_begin, ci_end, _ = parse_record_span(RECORD_DEL)
    assert classify_record_overlap(begin, end, ci_begin, ci_end, 1500, 1500) == "definite"
    assert classify_record_overlap(begin, end, ci_begin, ci_end, 900, 950) == "possible"
    assert classify_record_overlap(begin, end, ci_begin, ci_end, 2050, 2099) == "possible"
    assert classify_record_overlap(begin, end, ci_begin, ci_end, 899, 899) == "no_hit"
    assert classify_record_overlap(begin, end, ci_begin, ci_end, 2101, 2101) == "no_hit"


def test_classify_without_confidence_interval_never_possible():
    begin, end, ci_begin, ci_end, _ = parse_record_span(RECORD_SNV)
    assert ci_begin is None and ci_end is None
    assert classify_record_overlap(begin, end, ci_begin, ci_end, 5001, 6000) == "no_hit"


# End-to-end tabix fetch ------------------------------------------------------


def test_tabix_fetch_marks_definite_and_possible_hits(indexed_sv_vcf):
    vcf_path, tbi_path = indexed_sv_vcf
    with TabixFile(filename=vcf_path, index=tbi_path) as tabix_file:
        def positions(query_start, query_end):
            iterator = tabix_file.fetch(reference="1", start=query_start, end=query_end)
            return [(line.split("\t")[1], iterator.last_match_class) for line in iterator]

        assert positions(900, 950) == [("1000", "possible")]
        assert positions(1000, 1000) == [("1000", "definite")]
        assert positions(1500, 1500) == [("1000", "definite")]
        assert positions(2050, 2099) == [("1000", "possible")]
        assert positions(2101, 2101) == []
        assert positions(3000, 3000) == [("3000", "definite")]
        assert positions(2999, 2999) == []


def test_tabix_fetch_whole_reference_is_definite(indexed_sv_vcf):
    vcf_path, tbi_path = indexed_sv_vcf
    with TabixFile(filename=vcf_path, index=tbi_path) as tabix_file:
        iterator = tabix_file.fetch(reference="1")
        hits = [(line.split("\t")[1], iterator.last_match_class) for line in iterator]
    assert hits == [("1000", "definite"), ("3000", "definite"), ("5000", "definite")]


def test_reader_fetch_classified(indexed_sv_vcf):
    vcf_path, tbi_path = indexed_sv_vcf
    with Reader.from_path(vcf_path, tabix_path=tbi_path) as reader:
        possible = list(reader.fetch_classified("1", 900, 950))
        assert len(possible) == 1
        record, match_class = possible[0]
        assert record.POS == 1000
        assert match_class == "possible"
        assert reader.last_match_class == "possible"

        definite = list(reader.fetch_classified("1", 1500, 1500))
        assert [(r.POS, cls) for r, cls in definite] == [(1000, "definite")]

        assert list(reader.fetch_classified("1", 2101, 2101)) == []
