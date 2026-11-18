# -*- coding: utf-8 -*-
"""Tests for vcfpy.gvcf per-sample determinable intervals"""

import io
import textwrap

import pytest

import vcfpy


@pytest.fixture
def gvcf_text() -> str:
    """Small gVCF exercising blocks, variants, no-calls, and duplicates"""
    return textwrap.dedent(
        """
        ##fileformat=VCFv4.3
        ##contig=<ID=1,length=1000>
        ##ALT=<ID=NON_REF,Description="Any possible alternative allele not already represented">
        ##INFO=<ID=END,Number=1,Type=Integer,Description="End of reference confidence block">
        ##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tNA1\tNA2
        1\t100\t.\tA\t<NON_REF>\t.\t.\tEND=199\tGT\t0/0\t0/0
        1\t150\t.\tG\tT,<NON_REF>\t.\t.\t.\tGT\t0/1\t0/0
        1\t300\t.\tC\t<NON_REF>\t.\t.\tEND=399\tGT\t0/0\t./.
        1\t300\t.\tC\t<NON_REF>\t.\t.\tEND=399\tGT\t0/0\t./.
        1\t500\t.\tG\t<NON_REF>\t.\t.\tEND=599\tGT\t0/0\t0/0
        1\t550\t.\tT\tA\t.\t.\t.\tGT\t./.\t./.
        1\t700\t.\tG\tT,<NON_REF>\t.\t.\t.\tGT\t0/0/1\t0/.
        1\t800\t.\tA\t<NON_REF>\t.\t.\tEND=809\tGT\t0/1\t0/0
        """
    ).lstrip()


@pytest.fixture
def gvcf_records(gvcf_text: str) -> list[vcfpy.Record]:
    with vcfpy.Reader.from_stream(io.StringIO(gvcf_text)) as reader:
        return list(reader)


def _make_record(chrom, pos, ref, alts, info=None, gt=None, sample="S"):
    alt_records = []
    for alt in alts:
        if alt.startswith("<"):
            alt_records.append(vcfpy.SymbolicAllele(alt[1:-1]))
        else:
            alt_records.append(vcfpy.Substitution("SNV" if len(alt) == 1 else "INDEL", alt))
    call = vcfpy.Call(sample, {"GT": gt} if gt is not None else {})
    return vcfpy.Record(
        chrom, pos, [], ref, alt_records, None, [], info or {}, ["GT"], [call]
    )


def test_is_non_ref_allele():
    assert vcfpy.is_non_ref_allele(vcfpy.SymbolicAllele("NON_REF"))
    assert vcfpy.is_non_ref_allele(vcfpy.SymbolicAllele("*"))
    assert vcfpy.is_non_ref_allele(vcfpy.Substitution("SNV", "*"))
    assert not vcfpy.is_non_ref_allele(vcfpy.Substitution("SNV", "T"))
    assert not vcfpy.is_non_ref_allele(vcfpy.SymbolicAllele("DEL"))


def test_is_reference_block():
    block = _make_record("1", 100, "A", ["<NON_REF>"], {"END": 199}, "0/0")
    assert vcfpy.is_reference_block(block)
    variant = _make_record("1", 100, "A", ["T", "<NON_REF>"], None, "0/1")
    assert not vcfpy.is_reference_block(variant)


def test_record_span_uses_end():
    block = _make_record("1", 100, "A", ["<NON_REF>"], {"END": 199}, "0/0")
    assert vcfpy.record_span(block) == (99, 199)


def test_record_span_without_end_covers_ref():
    record = _make_record("1", 100, "ATG", ["A"], None, "0/1")
    assert vcfpy.record_span(record) == (99, 102)


def test_classify_call_hom_ref():
    record = _make_record("1", 100, "A", ["<NON_REF>"], {"END": 199}, "0/0")
    cls = vcfpy.classify_call(record, record.calls[0])
    assert cls.state == vcfpy.STATE_REFERENCE
    assert cls.contribution == vcfpy.AlleleContribution(2, 2, 0, 0)


def test_classify_call_non_ref_is_not_a_variant():
    record = _make_record("1", 100, "A", ["<NON_REF>"], {"END": 199}, "0/1")
    cls = vcfpy.classify_call(record, record.calls[0])
    assert cls.state == vcfpy.STATE_NO_CALL
    assert cls.contribution.alt_alleles == 0


def test_classify_call_polyploid_keeps_contribution():
    record = _make_record("1", 100, "A", ["T", "<NON_REF>"], None, "0/0/1")
    cls = vcfpy.classify_call(record, record.calls[0])
    assert cls.state == vcfpy.STATE_VARIANT
    assert cls.contribution == vcfpy.AlleleContribution(3, 2, 1, 0)


def test_classify_call_partial_missing():
    record = _make_record("1", 100, "A", ["T"], None, "0/.")
    cls = vcfpy.classify_call(record, record.calls[0])
    assert cls.state == vcfpy.STATE_NO_CALL
    assert cls.contribution == vcfpy.AlleleContribution(2, 1, 0, 1)


def test_classify_call_variant_with_partial_missing():
    record = _make_record("1", 100, "A", ["T"], None, "1/.")
    cls = vcfpy.classify_call(record, record.calls[0])
    assert cls.state == vcfpy.STATE_VARIANT
    assert cls.contribution == vcfpy.AlleleContribution(2, 0, 1, 1)


def test_classify_call_fully_missing():
    record = _make_record("1", 100, "A", ["T"], None, "./.")
    cls = vcfpy.classify_call(record, record.calls[0])
    assert cls.state == vcfpy.STATE_NO_CALL
    assert cls.contribution == vcfpy.AlleleContribution(2, 0, 0, 2)


def test_classify_call_without_gt():
    record = _make_record("1", 100, "A", ["T"], None, None)
    cls = vcfpy.classify_call(record, record.calls[0])
    assert cls.state == vcfpy.STATE_NO_CALL
    assert cls.contribution.ploidy is None


def test_block_is_split_by_variant(gvcf_records):
    intervals = vcfpy.sample_intervals(gvcf_records, "NA1")
    first = [iv for iv in intervals if iv.start < 200]
    assert [(iv.start, iv.end, iv.state) for iv in first] == [
        (99, 149, vcfpy.STATE_REFERENCE),
        (149, 150, vcfpy.STATE_VARIANT),
        (150, 199, vcfpy.STATE_REFERENCE),
    ]


def test_hom_ref_at_variant_site_is_reference(gvcf_records):
    intervals = vcfpy.sample_intervals(gvcf_records, "NA2")
    first = [iv for iv in intervals if iv.start < 200]
    assert [(iv.start, iv.end, iv.state) for iv in first] == [
        (99, 199, vcfpy.STATE_REFERENCE)
    ]


def test_no_call_record_inside_block_not_filled_as_reference(gvcf_records):
    intervals = vcfpy.sample_intervals(gvcf_records, "NA1")
    middle = [iv for iv in intervals if 499 <= iv.start < 600]
    assert [(iv.start, iv.end, iv.state) for iv in middle] == [
        (499, 549, vcfpy.STATE_REFERENCE),
        (549, 550, vcfpy.STATE_NO_CALL),
        (550, 599, vcfpy.STATE_REFERENCE),
    ]


def test_no_call_block_genotype(gvcf_records):
    intervals = vcfpy.sample_intervals(gvcf_records, "NA2")
    block = [iv for iv in intervals if 299 <= iv.start < 400]
    assert [(iv.start, iv.end, iv.state) for iv in block] == [
        (299, 399, vcfpy.STATE_NO_CALL)
    ]
    assert block[0].contribution == vcfpy.AlleleContribution(2, 0, 0, 2)


def test_shard_edge_duplicate_block_counted_once(gvcf_records):
    intervals = vcfpy.sample_intervals(gvcf_records, "NA1")
    block = [iv for iv in intervals if 299 <= iv.start < 400]
    assert len(block) == 1
    assert block[0].end - block[0].start == 100


def test_polyploid_interval_keeps_contribution(gvcf_records):
    intervals = vcfpy.sample_intervals(gvcf_records, "NA1")
    (variant,) = [iv for iv in intervals if iv.state == vcfpy.STATE_VARIANT and iv.start == 699]
    assert (variant.start, variant.end) == (699, 700)
    assert variant.contribution == vcfpy.AlleleContribution(3, 2, 1, 0)


def test_partially_missing_genotype_is_no_call(gvcf_records):
    intervals = vcfpy.sample_intervals(gvcf_records, "NA2")
    (nocall,) = [iv for iv in intervals if iv.start == 699]
    assert nocall.state == vcfpy.STATE_NO_CALL
    assert nocall.contribution == vcfpy.AlleleContribution(2, 1, 0, 1)


def test_non_ref_genotype_is_no_call_not_variant(gvcf_records):
    intervals = vcfpy.sample_intervals(gvcf_records, "NA1")
    (nocall,) = [iv for iv in intervals if iv.start == 799]
    assert nocall.state == vcfpy.STATE_NO_CALL
    assert nocall.contribution.alt_alleles == 0


def test_gaps_stay_implicit_without_region(gvcf_records):
    intervals = vcfpy.sample_intervals(gvcf_records, "NA1")
    assert all(iv.state != vcfpy.STATE_UNCOVERED for iv in intervals)
    # neighbouring blocks must not be merged across the gap at [199, 299)
    starts = [(iv.start, iv.end) for iv in intervals]
    assert (150, 199) in starts
    assert (299, 399) in starts
    assert (150, 399) not in starts


def test_region_reports_uncovered_gaps(gvcf_records):
    intervals = vcfpy.sample_intervals(gvcf_records, "NA1", region=("1", 99, 810))
    uncovered = [iv for iv in intervals if iv.state == vcfpy.STATE_UNCOVERED]
    assert [(iv.start, iv.end) for iv in uncovered] == [
        (199, 299),
        (399, 499),
        (599, 699),
        (700, 799),
        (809, 810),
    ]
    for iv in uncovered:
        assert iv.contribution == vcfpy.AlleleContribution(None, 0, 0, 0)


def test_region_clips_intervals(gvcf_records):
    intervals = vcfpy.sample_intervals(gvcf_records, "NA1", region=("1", 100, 350))
    assert intervals[0].start == 100
    assert intervals[-1].end == 350
    assert all(100 <= iv.start < iv.end <= 350 for iv in intervals)


def test_region_without_any_record_is_fully_uncovered(gvcf_records):
    intervals = vcfpy.sample_intervals(gvcf_records, "NA1", region=("2", 0, 100))
    assert intervals == [
        vcfpy.SampleInterval("2", 0, 100, vcfpy.STATE_UNCOVERED, vcfpy.AlleleContribution(None, 0, 0, 0))
    ]


def test_all_sample_intervals(gvcf_records):
    result = vcfpy.all_sample_intervals(gvcf_records)
    assert set(result) == {"NA1", "NA2"}
    assert result["NA1"] == vcfpy.sample_intervals(gvcf_records, "NA1")
    assert result["NA2"] == vcfpy.sample_intervals(gvcf_records, "NA2")


def test_all_sample_intervals_explicit_samples(gvcf_records):
    result = vcfpy.all_sample_intervals(gvcf_records, samples=["NA2"])
    assert set(result) == {"NA2"}


def test_unknown_sample_yields_no_intervals(gvcf_records):
    assert vcfpy.sample_intervals(gvcf_records, "UNKNOWN") == []
