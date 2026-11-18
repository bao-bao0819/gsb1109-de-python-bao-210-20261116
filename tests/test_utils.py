# -*- coding: utf-8 -*-
"""Tests for ``vcfpy.utils.recompute_ac_an``"""

import io
import textwrap

import vcfpy
from vcfpy import utils


def _read_single_record(vcf_text: str, **kwargs) -> vcfpy.Record:
    """Parse the given VCF text and return the first record"""
    with vcfpy.Reader.from_stream(io.StringIO(textwrap.dedent(vcf_text).lstrip()), **kwargs) as reader:
        for record in reader:
            return record
    raise AssertionError("no record in VCF")  # pragma: no cover


def _read_record_and_header(vcf_text: str, **kwargs) -> tuple[vcfpy.Record, vcfpy.Header]:
    """Parse the given VCF text and return the first record and the header"""
    with vcfpy.Reader.from_stream(io.StringIO(textwrap.dedent(vcf_text).lstrip()), **kwargs) as reader:
        header = reader.header.copy()
        for record in reader:
            return record, header
    raise AssertionError("no record in VCF")  # pragma: no cover


#: VCF with three diploid samples and cohort-wide AC/AN/AF annotations
VCF_BIALLELIC = """
##fileformat=VCFv4.3
##contig=<ID=20>
##INFO=<ID=AC,Number=A,Type=Integer,Description="Allele count">
##INFO=<ID=AN,Number=1,Type=Integer,Description="Total number of alleles">
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele frequency from external annotation">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tNA00001\tNA00002\tNA00003
20\t14370\t.\tG\tA\t29\t.\tAC=3;AN=6;AF=0.5\tGT\t0/1\t1/1\t0/0
"""

#: VCF with multiallelic site
VCF_MULTIALLELIC = """
##fileformat=VCFv4.3
##contig=<ID=20>
##INFO=<ID=AC,Number=A,Type=Integer,Description="Allele count">
##INFO=<ID=AN,Number=1,Type=Integer,Description="Total number of alleles">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tNA00001\tNA00002\tNA00003
20\t1110696\t.\tA\tG,T\t67\t.\tAC=3,3;AN=6\tGT\t1/2\t2/1\t2/2
"""


def test_recompute_all_samples_biallelic():
    record = _read_single_record(VCF_BIALLELIC)
    assert utils.recompute_ac_an(record)
    assert record.INFO["AC"] == [3]
    assert record.INFO["AN"] == 6


def test_recompute_subset_of_samples():
    record = _read_single_record(VCF_BIALLELIC)
    assert utils.recompute_ac_an(record, samples=["NA00002"])
    assert record.INFO["AC"] == [2]
    assert record.INFO["AN"] == 2


def test_recompute_multiallelic():
    record = _read_single_record(VCF_MULTIALLELIC)
    assert utils.recompute_ac_an(record, samples=["NA00001", "NA00002"])
    assert record.INFO["AC"] == [2, 2]
    assert record.INFO["AN"] == 4


def test_recompute_uses_actual_ploidy():
    record = _read_single_record(
        """
        ##fileformat=VCFv4.3
        ##contig=<ID=20>
        ##INFO=<ID=AC,Number=A,Type=Integer,Description="Allele count">
        ##INFO=<ID=AN,Number=1,Type=Integer,Description="Total number of alleles">
        ##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\tS3
        20\t14370\t.\tG\tA\t29\t.\tAC=9;AN=9\tGT\t1\t0/1/1\t0/0
        """
    )
    assert utils.recompute_ac_an(record)
    # haploid, triploid, and diploid calls contribute their actual ploidy
    assert record.INFO["AC"] == [3]
    assert record.INFO["AN"] == 6


def test_recompute_partially_missing_genotypes():
    record = _read_single_record(
        """
        ##fileformat=VCFv4.3
        ##contig=<ID=20>
        ##INFO=<ID=AC,Number=A,Type=Integer,Description="Allele count">
        ##INFO=<ID=AN,Number=1,Type=Integer,Description="Total number of alleles">
        ##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\tS3\tS4
        20\t14370\t.\tG\tA\t29\t.\tAC=9;AN=9\tGT\t0/.\t./1\t./.\t1/1
        """
    )
    assert utils.recompute_ac_an(record)
    # partially missing calls still contribute their known alleles
    assert record.INFO["AC"] == [3]
    assert record.INFO["AN"] == 4


def test_recompute_keeps_values_without_gt():
    record = _read_single_record(
        """
        ##fileformat=VCFv4.3
        ##contig=<ID=20>
        ##INFO=<ID=AC,Number=A,Type=Integer,Description="Allele count">
        ##INFO=<ID=AN,Number=1,Type=Integer,Description="Total number of alleles">
        ##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read depth">
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2
        20\t14370\t.\tG\tA\t29\t.\tAC=3;AN=4\tDP\t10\t11
        """
    )
    assert not utils.recompute_ac_an(record)
    # denominator cannot be determined, previous values are kept
    assert record.INFO["AC"] == [3]
    assert record.INFO["AN"] == 4


def test_recompute_keeps_values_without_calls():
    record = _read_single_record(
        """
        ##fileformat=VCFv4.3
        ##contig=<ID=20>
        ##INFO=<ID=AC,Number=A,Type=Integer,Description="Allele count">
        ##INFO=<ID=AN,Number=1,Type=Integer,Description="Total number of alleles">
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
        20\t14370\t.\tG\tA\t29\t.\tAC=3;AN=4
        """
    )
    assert not utils.recompute_ac_an(record)
    assert record.INFO["AC"] == [3]
    assert record.INFO["AN"] == 4


def test_recompute_keeps_values_for_unparsed_calls():
    record = _read_single_record(VCF_BIALLELIC, parsed_samples=["NA00001"])
    assert not utils.recompute_ac_an(record)
    assert record.INFO["AC"] == [3]
    assert record.INFO["AN"] == 6
    # recomputing for the parsed sample only works
    assert utils.recompute_ac_an(record, samples=["NA00001"])
    assert record.INFO["AC"] == [1]
    assert record.INFO["AN"] == 2


def test_recompute_keeps_values_for_unknown_sample():
    record = _read_single_record(VCF_BIALLELIC)
    assert not utils.recompute_ac_an(record, samples=["NA00002", "nosuch"])
    assert record.INFO["AC"] == [3]
    assert record.INFO["AN"] == 6


def test_recompute_does_not_touch_frequency_annotations():
    record = _read_single_record(VCF_BIALLELIC)
    assert utils.recompute_ac_an(record, samples=["NA00003"])
    assert record.INFO["AC"] == [0]
    assert record.INFO["AN"] == 2
    # external annotation frequencies are not recomputed
    assert record.INFO["AF"] == [0.5]


def test_recompute_respects_header_declaration_number_1():
    record, header = _read_record_and_header(
        """
        ##fileformat=VCFv4.3
        ##contig=<ID=20>
        ##INFO=<ID=AC,Number=1,Type=Integer,Description="Allele count">
        ##INFO=<ID=AN,Number=1,Type=Integer,Description="Total number of alleles">
        ##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2
        20\t14370\t.\tG\tA\t29\t.\tAC=3;AN=4\tGT\t0/1\t1/1
        """
    )
    assert utils.recompute_ac_an(record, header=header)
    assert record.INFO["AC"] == 3
    assert record.INFO["AN"] == 4


def test_recompute_keeps_incompatible_declaration():
    record, header = _read_record_and_header(
        """
        ##fileformat=VCFv4.3
        ##contig=<ID=20>
        ##INFO=<ID=AC,Number=1,Type=String,Description="Not a count field">
        ##INFO=<ID=AN,Number=2,Type=Integer,Description="Not the allele number">
        ##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2
        20\t14370\t.\tG\tA\t29\t.\tAC=somevalue;AN=4,5\tGT\t0/1\t1/1
        """
    )
    assert utils.recompute_ac_an(record, header=header)
    # declarations are incompatible with allele counts, values are kept
    assert record.INFO["AC"] == "somevalue"
    assert record.INFO["AN"] == [4, 5]


def test_recompute_without_header_uses_reserved_definitions():
    record = _read_single_record(VCF_BIALLELIC)
    assert utils.recompute_ac_an(record, header=None, samples=["NA00002"])
    assert record.INFO["AC"] == [2]
    assert record.INFO["AN"] == 2


def test_recompute_is_idempotent():
    record = _read_single_record(VCF_BIALLELIC)
    assert utils.recompute_ac_an(record, samples=["NA00001", "NA00002"])
    first = (list(record.INFO["AC"]), record.INFO["AN"])
    assert utils.recompute_ac_an(record, samples=["NA00001", "NA00002"])
    assert (list(record.INFO["AC"]), record.INFO["AN"]) == first


def test_recompute_after_repeated_subsetting():
    record = _read_single_record(VCF_MULTIALLELIC)
    # first subsetting: keep NA00001 and NA00002
    assert utils.recompute_ac_an(record, samples=["NA00001", "NA00002"])
    assert record.INFO["AC"] == [2, 2]
    assert record.INFO["AN"] == 4
    # second subsetting: keep NA00001 only, denominator is recomputed from
    # the genotypes and does not shrink iteratively
    assert utils.recompute_ac_an(record, samples=["NA00001"])
    assert record.INFO["AC"] == [1, 1]
    assert record.INFO["AN"] == 2


def test_recompute_missing_info_keys_are_not_added():
    record = _read_single_record(
        """
        ##fileformat=VCFv4.3
        ##contig=<ID=20>
        ##INFO=<ID=AC,Number=A,Type=Integer,Description="Allele count">
        ##INFO=<ID=AN,Number=1,Type=Integer,Description="Total number of alleles">
        ##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1
        20\t14370\t.\tG\tA\t29\t.\t.\tGT\t0/1
        """
    )
    assert utils.recompute_ac_an(record)
    assert "AC" not in record.INFO
    assert "AN" not in record.INFO


def test_recompute_with_header_argument():
    with vcfpy.Reader.from_stream(
        io.StringIO(textwrap.dedent(VCF_BIALLELIC).lstrip())
    ) as reader:
        record = next(iter(reader))
        assert utils.recompute_ac_an(record, header=reader.header, samples=["NA00002"])
    assert record.INFO["AC"] == [2]
    assert record.INFO["AN"] == 2


def test_public_api_export():
    assert vcfpy.recompute_ac_an is utils.recompute_ac_an
