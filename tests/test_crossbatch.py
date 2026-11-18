# -*- coding: utf-8 -*-
"""Comparable cross-batch metric partitions"""

import io
import textwrap
import warnings

import pytest

import vcfpy
from vcfpy.crossbatch import (
    FieldConvention,
    IncompatibleConvention,
    Observation,
    ObsStatus,
    Partition,
    Stats,
    VariantKey,
    collect_reader,
)

__author__ = "Manuel Holtgrewe <manuel.holtgrewe@bihealth.de>"

#: old batch: DPX is Integer/1, AF is Float/A
OLD_VCF = textwrap.dedent(
    """
    ##fileformat=VCFv4.3
    ##contig=<ID=1>
    ##INFO=<ID=DPX,Number=1,Type=Integer,Description="depth">
    ##INFO=<ID=AF,Number=A,Type=Float,Description="frequency">
    #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
    1\t100\trs1\tA\tC\t.\t.\tDPX=10;AF=0.1
    1\t200\trs2\tG\tT\t.\t.\tDPX=.
    1\t300\t.\tC\tG\t.\t.\tAF=0.3
    1\t400\trs4\tT\tA\t.\t.\tDPX=abc;AF=0.4,0.5
    """
).lstrip()

#: second batch with the same conventions as the old one
MID_VCF = textwrap.dedent(
    """
    ##fileformat=VCFv4.3
    ##contig=<ID=1>
    ##INFO=<ID=DPX,Number=1,Type=Integer,Description="depth">
    ##INFO=<ID=AF,Number=A,Type=Float,Description="frequency">
    #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
    1\t500\trs5\tA\tG\t.\t.\tDPX=20;AF=0.9
    1\t600\t.\tT\tC\t.\t.\tDPX=5
    """
).lstrip()

#: new batch: DPX changed type to Float, AF changed cardinality to 1
NEW_VCF = textwrap.dedent(
    """
    ##fileformat=VCFv4.3
    ##contig=<ID=1>
    ##INFO=<ID=DPX,Number=1,Type=Float,Description="depth">
    ##INFO=<ID=AF,Number=1,Type=Float,Description="frequency">
    #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
    1\t700\trs7\tG\tA\t.\t.\tDPX=7.5;AF=0.15
    1\t800\t.\tC\tT\t.\t.\tDPX=2.5;AF=.
    """
).lstrip()

DPX_INT = FieldConvention("DPX", "Integer", "1")
DPX_FLOAT = FieldConvention("DPX", "Float", "1")
AF_A = FieldConvention("AF", "Float", "A")
AF_1 = FieldConvention("AF", "Float", "1")


def collect(text, batch, fields=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reader = vcfpy.Reader.from_stream(io.StringIO(text))
        return collect_reader(reader, batch, fields=fields)


def make_partition(values, convention=DPX_INT):
    partition = Partition(convention)
    variant = VariantKey("1", 1, "A", ("C",), ())
    for value in values:
        partition.add(
            Observation(convention, "b", variant, None, ObsStatus.VALID, float(value))
        )
    return partition


# missing values ------------------------------------------------------------


def test_missing_never_counted_as_zero():
    metrics = collect(OLD_VCF, "old")
    partition = metrics.partitions[DPX_INT]
    stats = partition.stats()
    assert stats.count == 1
    assert stats.total == 10.0
    assert stats.mean == 10.0
    assert stats.minimum == 10.0
    assert stats.maximum == 10.0
    # the '.' and the absent entry are kept as missing, not as zeros
    assert partition.missing_count == 2
    assert stats.mean != 10.0 / 3


def test_missing_observation_cannot_carry_value():
    variant = VariantKey("1", 1, "A", ("C",), ())
    with pytest.raises(ValueError):
        Observation(DPX_INT, "b", variant, None, ObsStatus.MISSING, 0.0)
    with pytest.raises(ValueError):
        Observation(DPX_INT, "b", variant, None, ObsStatus.FAILED, 0.0)
    with pytest.raises(ValueError):
        Observation(DPX_INT, "b", variant, None, ObsStatus.VALID, None)


# conversion failures and variant identity -----------------------------------


def test_conversion_failure_preserved_with_identity():
    metrics = collect(OLD_VCF, "old")
    partition = metrics.partitions[DPX_INT]
    assert partition.failed_count == 1
    (failed,) = [o for o in partition.observations if o.status is ObsStatus.FAILED]
    assert failed.raw == "abc"
    assert failed.value is None
    assert failed.variant == VariantKey("1", 400, "T", ("A",), ("rs4",))
    assert failed.batch == "old"


def test_valid_observation_keeps_whole_variant_identity():
    metrics = collect(OLD_VCF, "old")
    partition = metrics.partitions[DPX_INT]
    (valid,) = [o for o in partition.observations if o.status is ObsStatus.VALID]
    assert valid.variant == VariantKey("1", 100, "A", ("C",), ("rs1",))


def test_multivalue_field_exploded_with_index():
    metrics = collect(OLD_VCF, "old")
    partition = metrics.partitions[AF_A]
    at_400 = [o for o in partition.observations if o.variant.pos == 400]
    assert [(o.index, o.value) for o in at_400] == [(0, 0.4), (1, 0.5)]
    assert partition.valid_count == 4
    assert partition.missing_count == 1


# incompatible conventions ---------------------------------------------------


def test_same_name_incompatible_conventions_not_merged():
    metrics = collect(OLD_VCF, "old")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reader = vcfpy.Reader.from_stream(io.StringIO(NEW_VCF))
        metrics = collect_reader(reader, "new", metrics=metrics)
    assert metrics.conventions_for("DPX") == [DPX_FLOAT, DPX_INT]
    assert metrics.conventions_for("AF") == [AF_1, AF_A]
    # each convention keeps its own statistics
    assert metrics.partitions[DPX_INT].stats().total == 10.0
    assert metrics.partitions[DPX_FLOAT].stats().total == 10.0
    assert metrics.partitions[DPX_INT].stats().count == 1
    assert metrics.partitions[DPX_FLOAT].stats().count == 2


def test_merge_keeps_incompatible_conventions_apart():
    merged = collect(OLD_VCF, "old").merge(collect(NEW_VCF, "new"))
    assert set(merged.partitions) == {DPX_INT, DPX_FLOAT, AF_A, AF_1}
    assert merged.partitions[DPX_INT].batches == ("old",)
    assert merged.partitions[DPX_FLOAT].batches == ("new",)


def test_partition_merge_rejects_incompatible_conventions():
    with pytest.raises(IncompatibleConvention):
        make_partition([1], DPX_INT).merge(make_partition([1], DPX_FLOAT))
    with pytest.raises(IncompatibleConvention):
        make_partition([1], DPX_INT).add(
            Observation(
                DPX_FLOAT,
                "b",
                VariantKey("1", 1, "A", ("C",), ()),
                None,
                ObsStatus.VALID,
                1.0,
            )
        )


# thresholds -----------------------------------------------------------------


def test_threshold_change_reevaluates_only_affected_band():
    partition = make_partition(range(1, 11))
    stats = partition.set_threshold(4)
    assert partition.last_band == 3  # values 1..3 left the statistic
    assert (stats.count, stats.total, stats.minimum, stats.maximum) == (
        7,
        49.0,
        4.0,
        10.0,
    )
    stats = partition.set_threshold(7)
    assert partition.last_band == 3  # only values 4..6 re-evaluated
    assert (stats.count, stats.total) == (4, 34.0)
    assert stats == partition.full_stats(7)
    stats = partition.set_threshold(2)
    assert partition.last_band == 5  # only values 2..6 re-evaluated
    assert (stats.count, stats.total) == (9, 54.0)
    assert stats == partition.full_stats(2)
    partition.set_threshold(2)
    assert partition.last_band == 0


def test_threshold_stats_match_full_computation():
    partition = make_partition([3, 1, 4, 1, 5, 9, 2, 6])
    for threshold in (0, 2, 5, 5.5, 10, 1):
        assert partition.set_threshold(threshold) == partition.full_stats(threshold)


# merging --------------------------------------------------------------------


def test_merge_statistics_equal_full_computation():
    old = collect(OLD_VCF, "old")
    mid = collect(MID_VCF, "mid")
    merged = old.merge(mid)
    partition = merged.partitions[DPX_INT]
    assert partition.stats() == Stats.from_values([10.0, 20.0, 5.0])
    assert partition.stats() == partition.full_stats()
    # same-caliber comparison: merged statistics equal full computation at any threshold
    for threshold in (0, 6, 15, 100):
        assert merged.partitions[DPX_INT].set_threshold(threshold) == Stats.from_values(
            [v for v in (10.0, 20.0, 5.0) if v >= threshold]
        )
    # identical to merging the per-batch full statistics
    assert old.partitions[DPX_INT].full_stats().merge(
        mid.partitions[DPX_INT].full_stats()
    ) == Stats.from_values([10.0, 20.0, 5.0])


def test_merge_preserves_observations_and_batches():
    merged = collect(OLD_VCF, "old").merge(collect(MID_VCF, "mid"))
    partition = merged.partitions[DPX_INT]
    assert partition.batches == ("mid", "old")
    assert {o.variant.pos for o in partition.observations} == {
        100,
        200,
        300,
        400,
        500,
        600,
    }
    assert merged.source_distribution()[DPX_INT] == {"old": 4, "mid": 2}


def test_unrecoverable_missing_not_backfilled_by_merge():
    merged = collect(OLD_VCF, "old").merge(collect(MID_VCF, "mid"))
    partition = merged.partitions[DPX_INT]
    assert partition.missing_count == 2
    assert partition.failed_count == 1
    assert partition.valid_count == 3
    # statistics only cover the three valid observations
    assert partition.stats().count == 3
    # thresholds cannot resurrect missing values either
    assert partition.set_threshold(0).count == 3
    assert partition.missing_count == 2


# extraction -----------------------------------------------------------------


def test_non_numeric_field_rejected():
    text = textwrap.dedent(
        """
        ##fileformat=VCFv4.3
        ##contig=<ID=1>
        ##INFO=<ID=S,Number=1,Type=String,Description="s">
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
        1\t100\t.\tA\tC\t.\t.\tS=x
        """
    ).lstrip()
    with pytest.raises(ValueError):
        collect(text, "b", fields=["S"])


def test_flag_field_collected_as_presence():
    text = textwrap.dedent(
        """
        ##fileformat=VCFv4.3
        ##contig=<ID=1>
        ##INFO=<ID=F,Number=0,Type=Flag,Description="f">
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
        1\t100\t.\tA\tC\t.\t.\tF
        1\t200\t.\tA\tC\t.\t.\t.
        """
    ).lstrip()
    metrics = collect(text, "b")
    partition = metrics.partitions[FieldConvention("F", "Flag", "0")]
    assert partition.stats().count == 1
    assert partition.missing_count == 1
