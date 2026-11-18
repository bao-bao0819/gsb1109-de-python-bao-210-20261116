# -*- coding: utf-8 -*-
"""Tests for splitting multi-allelic records into bi-allelic ones."""

import io

import pytest

import vcfpy


def make_header():
    """Return a header with all INFO/FORMAT fields used in the tests."""
    header = vcfpy.Header()
    header.add_line(vcfpy.HeaderLine("fileformat", "VCFv4.3"))
    header.add_info_line({"ID": "AF", "Number": "A", "Type": "Float", "Description": "Allele frequency"})
    header.add_info_line({"ID": "AD", "Number": "R", "Type": "Integer", "Description": "Read depth"})
    header.add_info_line({"ID": "DP", "Number": 1, "Type": "Integer", "Description": "Total depth"})
    header.add_info_line({"ID": "DB", "Number": 0, "Type": "Flag", "Description": "dbSNP membership"})
    header.add_info_line({"ID": "GLF", "Number": "G", "Type": "Float", "Description": "Genotype likelihoods"})
    header.add_info_line({"ID": "IDX", "Number": 1, "Type": "Integer", "Description": "Original ALT index"})
    header.add_format_line({"ID": "GT", "Number": 1, "Type": "String", "Description": "Genotype"})
    header.add_format_line({"ID": "AD", "Number": "R", "Type": "Integer", "Description": "Read depth"})
    header.add_format_line({"ID": "PL", "Number": "G", "Type": "Integer", "Description": "Phred likelihoods"})
    header.add_format_line({"ID": "DP", "Number": 1, "Type": "Integer", "Description": "Depth"})
    header.add_format_line({"ID": "OA", "Number": ".", "Type": "Integer", "Description": "Other alleles"})
    header.samples = vcfpy.SamplesInfos(["s1", "s2", "s3", "s4"])
    return header


def make_record():
    """Return a tri-allelic record with four samples of different ploidy."""
    return vcfpy.Record(
        CHROM="chr1",
        POS=100,
        ID=["rs1"],
        REF="A",
        ALT=[vcfpy.Substitution("SNV", "C"), vcfpy.Substitution("SNV", "G")],
        QUAL=50.0,
        FILTER=["PASS"],
        INFO={"AF": [0.1, 0.2], "AD": [30, 10, 5], "DP": 100, "DB": True},
        FORMAT=["GT", "AD", "PL", "DP"],
        calls=[
            vcfpy.Call("s1", {"GT": "0/1", "AD": [20, 10, 5], "PL": [0, 10, 20, 30, 40, 50], "DP": 30}),
            vcfpy.Call("s2", {"GT": "1/2", "AD": [15, 7, 8], "PL": [50, 40, 30, 20, 10, 0], "DP": 30}),
            vcfpy.Call("s3", {"GT": "./2", "AD": [12, None, 6], "PL": [1, 2, 3, 4, 5, 6], "DP": 18}),
            vcfpy.Call("s4", {"GT": "2", "AD": [9, 4, 5], "PL": [5, 6, 7], "DP": 9}),
        ],
    )


def test_split_into_one_record_per_alt():
    header = make_header()
    records = list(vcfpy.split_multiallelic(make_record(), header))
    assert len(records) == 2
    assert [r.ALT for r in records] == [
        [vcfpy.Substitution("SNV", "C")],
        [vcfpy.Substitution("SNV", "G")],
    ]
    for r in records:
        assert (r.CHROM, r.POS, r.ID, r.REF, r.QUAL, r.FILTER) == (
            "chr1",
            100,
            ["rs1"],
            "A",
            50.0,
            ["PASS"],
        )


def test_split_info_fields():
    header = make_header()
    rec1, rec2 = vcfpy.split_multiallelic(make_record(), header)
    # Number=A is reduced to the target allele
    assert rec1.INFO["AF"] == [0.1]
    assert rec2.INFO["AF"] == [0.2]
    # Number=R keeps reference and target allele entries
    assert rec1.INFO["AD"] == [30, 10]
    assert rec2.INFO["AD"] == [30, 5]
    # Number=1 and Flag are copied unchanged
    assert rec1.INFO["DP"] == 100
    assert rec2.INFO["DP"] == 100
    assert rec1.INFO["DB"] is True
    assert rec2.INFO["DB"] is True


def test_split_gt_remapping():
    header = make_header()
    rec1, rec2 = vcfpy.split_multiallelic(make_record(), header)
    # first bi-allelic row (ALT=C, original allele 1)
    assert rec1.call_for_sample["s1"].data["GT"] == "0/1"
    assert rec1.call_for_sample["s2"].data["GT"] == "1/."  # allele 2 not representable
    assert rec1.call_for_sample["s3"].data["GT"] == "./."  # partial missing kept, allele 2 dropped
    assert rec1.call_for_sample["s4"].data["GT"] == "."  # haploid, allele 2 dropped
    # second bi-allelic row (ALT=G, original allele 2)
    assert rec2.call_for_sample["s1"].data["GT"] == "0/."
    assert rec2.call_for_sample["s2"].data["GT"] == "./1"
    assert rec2.call_for_sample["s3"].data["GT"] == "./1"
    assert rec2.call_for_sample["s4"].data["GT"] == "1"


def test_split_other_alleles_listed_separately():
    """Other alleles must not be rewritten to reference but listed in OA."""
    header = make_header()
    rec1, rec2 = vcfpy.split_multiallelic(make_record(), header)
    # OA is appended to FORMAT because some calls have other alleles
    assert rec1.FORMAT == ["GT", "AD", "PL", "DP", "OA"]
    assert rec2.FORMAT == ["GT", "AD", "PL", "DP", "OA"]
    # original allele indices are preserved in OA
    assert rec1.call_for_sample["s1"].data["OA"] == []
    assert rec1.call_for_sample["s2"].data["OA"] == [2]
    assert rec1.call_for_sample["s3"].data["OA"] == [2]
    assert rec1.call_for_sample["s4"].data["OA"] == [2]
    assert rec2.call_for_sample["s1"].data["OA"] == [1]
    assert rec2.call_for_sample["s2"].data["OA"] == [1]
    assert rec2.call_for_sample["s3"].data["OA"] == []
    assert rec2.call_for_sample["s4"].data["OA"] == []


def test_split_format_number_r_and_scalar():
    header = make_header()
    rec1, rec2 = vcfpy.split_multiallelic(make_record(), header)
    # Number=R keeps reference and target allele entries
    assert rec1.call_for_sample["s1"].data["AD"] == [20, 10]
    assert rec2.call_for_sample["s1"].data["AD"] == [20, 5]
    # partially missing AD values stay missing
    assert rec1.call_for_sample["s3"].data["AD"] == [12, None]
    assert rec2.call_for_sample["s3"].data["AD"] == [12, 6]
    # Number=1 is copied unchanged
    assert rec1.call_for_sample["s3"].data["DP"] == 18
    assert rec2.call_for_sample["s4"].data["DP"] == 9


def test_split_format_number_g_diploid():
    header = make_header()
    rec1, rec2 = vcfpy.split_multiallelic(make_record(), header)
    # diploid PL: pick 0/0, 0/i, i/i from 0/0, 0/1, 1/1, 0/2, 1/2, 2/2
    assert rec1.call_for_sample["s1"].data["PL"] == [0, 10, 20]
    assert rec2.call_for_sample["s1"].data["PL"] == [0, 30, 50]
    assert rec1.call_for_sample["s2"].data["PL"] == [50, 40, 30]
    assert rec2.call_for_sample["s2"].data["PL"] == [50, 20, 0]


def test_split_format_number_g_haploid():
    header = make_header()
    rec1, rec2 = vcfpy.split_multiallelic(make_record(), header)
    # haploid PL follows the sample's actual ploidy
    assert rec1.call_for_sample["s4"].data["PL"] == [5, 6]
    assert rec2.call_for_sample["s4"].data["PL"] == [5, 7]


def test_split_format_number_g_triploid():
    header = make_header()
    header.samples = vcfpy.SamplesInfos(["t1"])
    record = vcfpy.Record(
        CHROM="chr1",
        POS=100,
        ID=[],
        REF="A",
        ALT=[vcfpy.Substitution("SNV", "C"), vcfpy.Substitution("SNV", "G")],
        QUAL=None,
        FILTER=[],
        INFO={},
        FORMAT=["GT", "PL"],
        calls=[vcfpy.Call("t1", {"GT": "0/1/2", "PL": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]})],
    )
    rec1, rec2 = vcfpy.split_multiallelic(record, header)
    # triploid: 10 genotypes, keep those over {0, i}
    assert rec1.call_for_sample["t1"].data["PL"] == [0, 1, 2, 3]
    assert rec2.call_for_sample["t1"].data["PL"] == [0, 4, 7, 9]
    assert rec1.call_for_sample["t1"].data["GT"] == "0/1/."
    assert rec1.call_for_sample["t1"].data["OA"] == [2]
    assert rec2.call_for_sample["t1"].data["GT"] == "0/./1"
    assert rec2.call_for_sample["t1"].data["OA"] == [1]


def test_split_info_number_g():
    header = make_header()
    record = make_record()
    record.INFO["GLF"] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    rec1, rec2 = vcfpy.split_multiallelic(record, header)
    # INFO Number=G: ploidy inferred from the number of values (diploid here)
    assert rec1.INFO["GLF"] == [0.0, 0.1, 0.2]
    assert rec2.INFO["GLF"] == [0.0, 0.3, 0.5]


def test_split_partially_missing_number_g():
    header = make_header()
    record = make_record()
    # short PL list and missing entries must not crash and stay missing
    record.call_for_sample["s1"].data["PL"] = [0, None, 20]
    (rec1,) = list(vcfpy.split_multiallelic(record, header))[:1]
    assert rec1.call_for_sample["s1"].data["PL"] == [0, None, 20]


def test_split_phased_gt():
    header = make_header()
    record = make_record()
    record.call_for_sample["s2"].set_genotype("1|2")
    (rec1,) = list(vcfpy.split_multiallelic(record, header))[:1]
    assert rec1.call_for_sample["s2"].data["GT"] == "1|."
    assert rec1.call_for_sample["s2"].data["OA"] == [2]


def test_split_biallelic_record():
    header = make_header()
    record = vcfpy.Record(
        CHROM="chr1",
        POS=100,
        ID=[],
        REF="A",
        ALT=[vcfpy.Substitution("SNV", "C")],
        QUAL=None,
        FILTER=[],
        INFO={"AF": [0.1]},
        FORMAT=["GT", "AD"],
        calls=[vcfpy.Call("s1", {"GT": "0/1", "AD": [10, 5]})],
    )
    (result,) = list(vcfpy.split_multiallelic(record, header))
    assert result.ALT == [vcfpy.Substitution("SNV", "C")]
    assert result.INFO["AF"] == [0.1]
    assert result.call_for_sample["s1"].data["GT"] == "0/1"
    assert result.call_for_sample["s1"].data["AD"] == [10, 5]
    # no other alleles, so no OA key is added
    assert result.FORMAT == ["GT", "AD"]


def test_split_allele_index_info():
    header = make_header()
    records = list(vcfpy.split_multiallelic(make_record(), header, allele_index_info="IDX"))
    assert [r.INFO["IDX"] for r in records] == [1, 2]


def test_split_other_alleles_key_disabled():
    header = make_header()
    (rec1,) = list(vcfpy.split_multiallelic(make_record(), header, other_alleles_key=None))[:1]
    assert "OA" not in rec1.FORMAT
    # other alleles still become missing, never reference
    assert rec1.call_for_sample["s2"].data["GT"] == "1/."


def test_split_unparsed_call_raises():
    header = make_header()
    record = vcfpy.Record(
        CHROM="chr1",
        POS=100,
        ID=[],
        REF="A",
        ALT=[vcfpy.Substitution("SNV", "C"), vcfpy.Substitution("SNV", "G")],
        QUAL=None,
        FILTER=[],
        INFO={},
        FORMAT=["GT"],
        calls=[vcfpy.UnparsedCall("s1", "0/1")],
    )
    with pytest.raises(ValueError, match="unparsed calls"):
        list(vcfpy.split_multiallelic(record, header))


def test_split_record_without_calls():
    header = make_header()
    record = vcfpy.Record(
        CHROM="chr1",
        POS=100,
        ID=[],
        REF="A",
        ALT=[vcfpy.Substitution("SNV", "C"), vcfpy.Substitution("SNV", "G")],
        QUAL=None,
        FILTER=[],
        INFO={"AF": [0.1, 0.2]},
    )
    records = list(vcfpy.split_multiallelic(record, header))
    assert len(records) == 2
    assert records[0].INFO["AF"] == [0.1]
    assert records[1].INFO["AF"] == [0.2]
    assert records[0].FORMAT == []
    assert records[0].calls == []


def test_split_results_are_independent():
    """Split records must not share mutable state with the input record."""
    header = make_header()
    record = make_record()
    rec1, _ = vcfpy.split_multiallelic(record, header)
    rec1.INFO["AD"][0] = 999
    rec1.call_for_sample["s1"].data["AD"][0] = 999
    assert record.INFO["AD"][0] == 30
    assert record.call_for_sample["s1"].data["AD"][0] == 20


def test_split_writer_roundtrip():
    """Split records can be written out and read back in."""
    header = make_header()
    out = io.StringIO()
    writer = vcfpy.Writer.from_stream(out, header)
    for record in vcfpy.split_multiallelic(make_record(), header):
        writer.write_record(record)
    reader = vcfpy.Reader.from_stream(io.StringIO(out.getvalue()))
    records = list(reader)
    assert len(records) == 2
    assert records[0].ALT == [vcfpy.Substitution("SNV", "C")]
    assert records[1].ALT == [vcfpy.Substitution("SNV", "G")]
    assert records[0].call_for_sample["s2"].data["GT"] == "1/."
    assert records[0].call_for_sample["s2"].data["OA"] == [2]
    assert records[1].call_for_sample["s2"].data["GT"] == "./1"
    assert records[1].call_for_sample["s2"].data["OA"] == [1]
    assert records[0].call_for_sample["s1"].data["PL"] == [0, 10, 20]
    assert records[1].call_for_sample["s1"].data["PL"] == [0, 30, 50]
