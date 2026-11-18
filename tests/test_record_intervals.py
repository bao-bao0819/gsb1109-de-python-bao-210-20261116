# -*- coding: utf-8 -*-
"""Tests for the definite/possible interval model of ``Record``.

All intervals exposed by :py:class:`vcfpy.RecordInterval` use 0-based,
half-open coordinates (end exclusive).
"""

import pytest

import vcfpy
from vcfpy import IntervalClass, RecordInterval, SymbolicAllele, record


def make_record(alt, info=None, pos=1000, ref="N"):
    return vcfpy.Record("chr1", pos, [], ref, alt, None, [], info or {})


# Ordinary (non-symbolic) records ---------------------------------------------


def test_snv_interval_is_ref_span():
    rec = make_record([vcfpy.Substitution("SNV", "A")], pos=1000, ref="G")
    assert rec.interval.definite == (999, 1000)
    # No CIPOS/CIEND: possible interval equals the definite interval.
    assert rec.interval.possible == (999, 1000)
    assert rec.interval.is_imprecise is False


def test_explicit_deletion_uses_ref_span():
    rec = make_record([vcfpy.Substitution("DEL", "A")], pos=100, ref="ATG")
    assert rec.interval.definite == (99, 102)
    assert rec.classify_interval(99, 102) is IntervalClass.DEFINITE
    # Half-open: the base immediately after the span is not covered.
    assert rec.classify_interval(102, 103) is IntervalClass.NO_HIT


def test_substitution_insertion_is_zero_length_point():
    rec = make_record([vcfpy.Substitution("INS", "ATG")], pos=100, ref="A")
    assert rec.interval.definite == (100, 100)
    assert rec.interval.possible == (100, 100)
    # Point query covering the insertion position.
    assert rec.classify_interval(100, 101) is IntervalClass.DEFINITE
    # The anchor base before the insertion point is not covered.
    assert rec.classify_interval(99, 100) is IntervalClass.NO_HIT


def test_explicit_deletion_ignores_contradicting_end():
    # Some callers emit INFO/END for explicit deletions using a different
    # convention; the REF allele span must remain authoritative.
    rec = make_record(
        [vcfpy.Substitution("DEL", "C")],
        pos=2827693,
        ref="CCGTGGATGCGGGGACCCGCATCCCCTCTCCCTTCACAGCTGAGTGACCCACATCCCCTCTCCCCTCGCA",
        info={"SVTYPE": "DEL", "END": 2827680},
    )
    expected_end = 2827692 + len(rec.REF)
    assert rec.interval.definite == (2827692, expected_end)
    assert rec.interval.end > rec.interval.start


def test_mixed_insertion_does_not_use_point_semantics():
    rec = make_record(
        [vcfpy.Substitution("INS", "ATG"), SymbolicAllele("DEL")],
        pos=1000,
        ref="A",
        info={"END": 1100},
    )
    # Mixed allele types fall back to the spanning semantics, not a point.
    assert rec.interval.start == 999
    assert rec.interval.end == 1100


# Symbolic structural variants ------------------------------------------------


def test_symbolic_del_uses_end_not_ref_length():
    # REF is a single base but END spans a thousand bases; the REF length must
    # not be substituted for the structural variant extent.
    rec = make_record(
        [SymbolicAllele("DEL")],
        pos=1000,
        ref="N",
        info={"SVTYPE": "DEL", "END": 2000},
    )
    assert rec.interval.definite == (999, 2000)
    assert rec.interval.possible == (999, 2000)
    assert rec.interval.is_imprecise is False


def test_symbolic_del_with_confidence_intervals():
    rec = make_record(
        [SymbolicAllele("DEL")],
        pos=1000,
        ref="N",
        info={"SVTYPE": "DEL", "END": 2000, "CIPOS": [-100, 50], "CIEND": [-50, 100]},
    )
    # Definite interval uses the point estimates only.
    assert rec.interval.definite == (999, 2000)
    # Possible interval expands both breakpoints by the confidence offsets.
    assert rec.interval.possible == (899, 2100)
    assert rec.interval.is_imprecise is True

    # Queries inside the definite span are definite hits.
    assert rec.classify_interval(1500, 1600) is IntervalClass.DEFINITE
    # Queries only touching the CIPOS/CIEND edges are possible hits.
    assert rec.classify_interval(900, 950) is IntervalClass.POSSIBLE
    assert rec.classify_interval(2050, 2099) is IntervalClass.POSSIBLE
    # Queries outside even the confidence interval are no hits.
    assert rec.classify_interval(800, 899) is IntervalClass.NO_HIT
    assert rec.classify_interval(2100, 2200) is IntervalClass.NO_HIT


def test_confidence_edge_query_is_possible_not_definite():
    rec = make_record(
        [SymbolicAllele("DEL")],
        pos=1000,
        ref="N",
        info={"SVTYPE": "DEL", "END": 2000, "CIPOS": [-100, 0], "CIEND": [0, 100]},
    )
    # 1-based POS 900 == 0-based 899, exactly the left confidence edge.
    result = rec.classify_interval(899, 900)
    assert result is IntervalClass.POSSIBLE
    assert result is not IntervalClass.DEFINITE


def test_symbolic_insertion_ignores_end_for_span():
    # END equals POS for insertions; it must not turn the record into a span.
    rec = make_record(
        [SymbolicAllele("INS:ME:L1")],
        pos=9425916,
        ref="C",
        info={"SVTYPE": "INS", "END": 9425916, "SVLEN": 6027, "CIPOS": [-16, 22]},
    )
    assert rec.interval.definite == (9425916, 9425916)
    assert rec.interval.possible == (9425900, 9425938)
    assert rec.classify_interval(9425916, 9425917) is IntervalClass.DEFINITE
    assert rec.classify_interval(9425900, 9425901) is IntervalClass.POSSIBLE


def test_unknown_length_sv_keeps_end_open():
    rec = make_record([SymbolicAllele("CNV")], pos=1000, ref="N", info={"SVTYPE": "CNV"})
    start, end = rec.interval.definite
    ci_start, ci_end = rec.interval.possible
    # The anchor base is definite ...
    assert (start, end) == (999, 1000)
    # ... but without END the possible extent is open on the right side
    # instead of being limited to the REF anchor base.
    assert ci_start == 999
    assert ci_end is None
    assert rec.interval.is_imprecise is False
    # A query beyond the anchor base can only be a possible hit.
    assert rec.classify_interval(1000, 5000) is IntervalClass.POSSIBLE
    assert rec.classify_interval(999, 1000) is IntervalClass.DEFINITE


def test_imprecise_unknown_length_sv_open_end():
    rec = make_record(
        [SymbolicAllele("CNV")],
        pos=1000,
        ref="N",
        info={"SVTYPE": "CNV", "IMPRECISE": True, "CIPOS": [-50, 50]},
    )
    # CIPOS widens the left side; the missing END keeps the right side open.
    assert rec.interval.definite == (999, 1000)
    assert rec.interval.possible == (949, None)
    assert rec.interval.is_imprecise is True
    assert rec.classify_interval(950, 960) is IntervalClass.POSSIBLE
    assert rec.classify_interval(5000, 5010) is IntervalClass.POSSIBLE
    assert rec.classify_interval(948, 949) is IntervalClass.NO_HIT


def test_breakend_has_no_span_on_this_contig():
    bnd = record.BreakEnd("chr2", 5000, record.FORWARD, record.FORWARD, "A", True)
    rec = make_record([bnd], pos=1000, ref="A")
    assert rec.interval.definite == (999, 1000)
    assert rec.interval.possible == (999, 1000)
    assert rec.classify_interval(1000, 2000) is IntervalClass.NO_HIT


def test_breakend_with_cipos():
    bnd = record.BreakEnd("chr2", 5000, record.FORWARD, record.FORWARD, "A", True)
    rec = make_record([bnd], pos=1000, ref="A", info={"CIPOS": [-20, 30]})
    assert rec.interval.definite == (999, 1000)
    assert rec.interval.possible == (979, 1030)
    assert rec.classify_interval(980, 990) is IntervalClass.POSSIBLE


# Interval object and API behaviour --------------------------------------------


def test_interval_is_recomputed_after_mutation():
    rec = make_record([SymbolicAllele("DEL")], pos=1000, ref="N", info={"END": 2000})
    assert rec.interval.definite == (999, 2000)
    rec.INFO["CIPOS"] = [-100, 50]
    rec.INFO["CIEND"] = [0, 100]
    assert rec.interval.possible == (899, 2100)


def test_invalid_query_raises():
    rec = make_record([vcfpy.Substitution("SNV", "A")])
    with pytest.raises(ValueError):
        rec.classify_interval(100, 99)


def test_record_interval_dataclass_api():
    iv = RecordInterval(start=10, end=20, ci_start=5, ci_end=25)
    assert iv.definite == (10, 20)
    assert iv.possible == (5, 25)
    assert iv.overlaps(15, 16) is True
    assert iv.classify(15, 16) is IntervalClass.DEFINITE
    assert iv.classify(6, 7) is IntervalClass.POSSIBLE
    assert iv.classify(30, 40) is IntervalClass.NO_HIT
    # Frozen dataclass.
    with pytest.raises(Exception):
        iv.start = 11  # type: ignore[misc]
