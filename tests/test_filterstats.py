# -*- coding: utf-8 -*-
"""Tests for vcfpy.filterstats."""

import pytest

from vcfpy import filterstats as fs
from vcfpy.record import Call, Record, Substitution


def outcome(source, level, key, status, labels=()):
    return fs.FilterOutcome(
        source=source, level=level, key=key, status=status, labels=tuple(labels)
    )


def record_outcomes():
    """Two sources, record- and sample-level outcomes, tri-state covered."""
    return [
        outcome("A", fs.Level.RECORD, ("1", "10"), fs.FilterStatus.PASSED),
        outcome("A", fs.Level.RECORD, ("1", "20"), fs.FilterStatus.NOT_DONE),
        outcome("A", fs.Level.RECORD, ("1", "30"), fs.FilterStatus.FAILED, ("q10",)),
        outcome(
            "A", fs.Level.RECORD, ("1", "40"), fs.FilterStatus.FAILED, ("q10", "dp")
        ),
        outcome("B", fs.Level.RECORD, ("1", "10"), fs.FilterStatus.FAILED, ("q10",)),
        outcome("B", fs.Level.RECORD, ("1", "20"), fs.FilterStatus.PASSED),
        outcome(
            "A", fs.Level.SAMPLE, ("1", "10", "s1"), fs.FilterStatus.FAILED, ("ft",)
        ),
        outcome("A", fs.Level.SAMPLE, ("1", "10", "s2"), fs.FilterStatus.PASSED),
        outcome("B", fs.Level.SAMPLE, ("1", "10", "s1"), fs.FilterStatus.NOT_DONE),
    ]


def default_mapping():
    return fs.FilterMapping(
        [
            fs.MappingRule("A", "q10", "low_quality", loss=fs.Loss.RENAME),
            fs.MappingRule("B", "q10", "depth_bias", loss=fs.Loss.RENAME),
            fs.MappingRule("A", "dp", "low_quality", loss=fs.Loss.COLLAPSE),
            fs.MappingRule("A", "ft", "sample_fail"),
        ]
    )


def test_outcome_validation():
    with pytest.raises(ValueError):
        outcome("A", fs.Level.RECORD, ("1", "1"), fs.FilterStatus.FAILED)
    with pytest.raises(ValueError):
        outcome("A", fs.Level.RECORD, ("1", "1"), fs.FilterStatus.PASSED, ("q10",))


def test_allowed_loss_is_enforced():
    rule = fs.MappingRule("A", "q10", "low_quality", loss=fs.Loss.COLLAPSE)
    with pytest.raises(ValueError):
        fs.FilterMapping([rule], allowed_loss=fs.Loss.RENAME)
    fs.FilterMapping([rule], allowed_loss=fs.Loss.COLLAPSE)


def test_same_named_labels_map_per_source():
    stats = fs.UnifiedFilterStats(default_mapping())
    stats.extend(record_outcomes())

    record_stats = stats.stats(fs.Level.RECORD)[fs.Level.RECORD]
    assert record_stats.status[fs.FilterStatus.PASSED] == 2
    assert record_stats.status[fs.FilterStatus.NOT_DONE] == 1
    assert record_stats.status[fs.FilterStatus.FAILED] == 3
    # A:q10 (x2) and A:dp (x1) collapse into low_quality.
    assert record_stats.failures["low_quality"] == 3
    # Same-named B:q10 has a different definition and target.
    assert record_stats.failures["depth_bias"] == 1

    sample_stats = stats.stats(fs.Level.SAMPLE)[fs.Level.SAMPLE]
    assert sample_stats.status[fs.FilterStatus.FAILED] == 1
    assert sample_stats.status[fs.FilterStatus.PASSED] == 1
    assert sample_stats.status[fs.FilterStatus.NOT_DONE] == 1
    assert sample_stats.failures["sample_fail"] == 1


def test_unmapped_failures_stay_explicit():
    stats = fs.UnifiedFilterStats(fs.FilterMapping())
    stats.extend(record_outcomes())
    failures = stats.stats(fs.Level.RECORD)[fs.Level.RECORD].failures
    # No generic "failed" bucket; source and label remain visible.
    assert failures[fs.unmapped_category("A", "q10")] == 2
    assert failures[fs.unmapped_category("A", "dp")] == 1
    assert failures[fs.unmapped_category("B", "q10")] == 1
    assert not any("generic" in cat or cat == "failed" for cat in failures)


def test_conditional_mapping():
    cond = lambda oc: oc.key[1] == "30"  # noqa: E731
    mapping = fs.FilterMapping(
        [fs.MappingRule("A", "q10", "low_quality", condition=cond)]
    )
    stats = fs.UnifiedFilterStats(mapping)
    stats.extend(record_outcomes())
    failures = stats.stats(fs.Level.RECORD)[fs.Level.RECORD].failures
    assert failures["low_quality"] == 1  # only record at pos 30
    assert failures[fs.unmapped_category("A", "q10")] == 1  # pos 40 unmapped


def test_contributions_cover_every_item():
    stats = fs.UnifiedFilterStats(default_mapping())
    outcomes = record_outcomes()
    stats.extend(outcomes)
    contribs = stats.contributions()
    # Every item contributes; failed items contribute once per label.
    assert len(contribs) == len(outcomes) + 1  # one item has two labels
    by_item = {}
    for contrib in contribs:
        by_item.setdefault((contrib.source, contrib.level, contrib.key), []).append(
            contrib
        )
    assert len(by_item) == len(outcomes)
    mapped = [c for c in contribs if c.mapped]
    assert all(c.rule is not None and c.loss is not None for c in mapped)
    unmapped = [
        c for c in contribs if c.status is fs.FilterStatus.FAILED and not c.mapped
    ]
    assert not unmapped  # default mapping covers all failure labels here
    not_failed = [c for c in contribs if c.status is not fs.FilterStatus.FAILED]
    assert {c.category for c in not_failed} == {
        fs.CATEGORY_PASSED,
        fs.CATEGORY_NOT_DONE,
    }


def test_duplicate_item_rejected():
    stats = fs.UnifiedFilterStats()
    oc = outcome("A", fs.Level.RECORD, ("1", "10"), fs.FilterStatus.PASSED)
    stats.add(oc)
    with pytest.raises(ValueError):
        stats.add(oc)


def test_mapping_switch_recomputes_only_affected():
    stats = fs.UnifiedFilterStats(default_mapping())
    stats.extend(record_outcomes())
    before = {c.key + (c.label or "",): c for c in stats.contributions()}

    # Only the target of A:q10 changes; B:q10, A:dp, A:ft are untouched.
    mapping_v2 = fs.FilterMapping(
        [
            fs.MappingRule("A", "q10", "quality_fail", loss=fs.Loss.RENAME),
            fs.MappingRule("B", "q10", "depth_bias", loss=fs.Loss.RENAME),
            fs.MappingRule("A", "dp", "low_quality", loss=fs.Loss.COLLAPSE),
            fs.MappingRule("A", "ft", "sample_fail"),
        ]
    )
    reclassified = stats.set_mapping(mapping_v2)
    # Exactly the two items carrying A:q10 are reclassified.
    assert reclassified == 2

    after = {c.key + (c.label or "",): c for c in stats.contributions()}
    assert set(before) == set(after)
    changed = [k for k in before if before[k] != after[k]]
    assert sorted(changed) == [("1", "30", "q10"), ("1", "40", "q10")]
    failures = stats.stats(fs.Level.RECORD)[fs.Level.RECORD].failures
    assert failures["quality_fail"] == 2
    assert failures["low_quality"] == 1  # only A:dp remains
    assert failures["depth_bias"] == 1


def test_mapping_switch_to_unmapped_and_back():
    stats = fs.UnifiedFilterStats(default_mapping())
    stats.extend(record_outcomes())
    # Drop the A:q10 rule entirely.
    mapping_v2 = fs.FilterMapping(
        [
            fs.MappingRule("B", "q10", "depth_bias", loss=fs.Loss.RENAME),
            fs.MappingRule("A", "dp", "low_quality", loss=fs.Loss.COLLAPSE),
            fs.MappingRule("A", "ft", "sample_fail"),
        ]
    )
    assert stats.set_mapping(mapping_v2) == 2
    failures = stats.stats(fs.Level.RECORD)[fs.Level.RECORD].failures
    assert failures[fs.unmapped_category("A", "q10")] == 2
    # Switching back restores the previous unified statistics.
    assert stats.set_mapping(default_mapping()) == 2
    failures = stats.stats(fs.Level.RECORD)[fs.Level.RECORD].failures
    assert failures["low_quality"] == 3
    assert fs.unmapped_category("A", "q10") not in failures


def test_reconcile_denominators_and_shares():
    stats = fs.UnifiedFilterStats(default_mapping())
    stats.extend(record_outcomes())
    rec = stats.reconcile()
    assert rec.ok
    by_source = {(r.source, r.level): r for r in rec.by_source}

    a_rec = by_source[("A", fs.Level.RECORD)]
    assert a_rec.denominator == 4
    assert (a_rec.not_done, a_rec.passed, a_rec.failed) == (1, 1, 2)
    assert a_rec.failure_instances == 3
    assert (a_rec.known, a_rec.unknown) == (3, 0)
    assert a_rec.known_share == 1.0
    assert a_rec.unknown_share == 0.0

    b_rec = by_source[("B", fs.Level.RECORD)]
    assert b_rec.denominator == 2
    assert (b_rec.known, b_rec.unknown) == (1, 0)

    b_sample = by_source[("B", fs.Level.SAMPLE)]
    assert b_sample.denominator == 1
    assert b_sample.failure_instances == 0
    assert b_sample.known_share == 1.0  # vacuous: no failures


def test_reconcile_unknown_share():
    stats = fs.UnifiedFilterStats(
        fs.FilterMapping([fs.MappingRule("A", "q10", "low_quality")])
    )
    stats.extend(record_outcomes())
    rec = stats.reconcile()
    assert rec.ok
    by_source = {(r.source, r.level): r for r in rec.by_source}
    a_rec = by_source[("A", fs.Level.RECORD)]
    assert (a_rec.known, a_rec.unknown) == (2, 1)
    assert a_rec.unknown_share == pytest.approx(1 / 3)
    b_rec = by_source[("B", fs.Level.RECORD)]
    assert (b_rec.known, b_rec.unknown) == (0, 1)
    assert b_rec.unknown_share == 1.0


def make_record(filters, fts, pos=100):
    calls = [
        Call("s1", {"FT": fts[0]} if fts[0] is not None else {}),
        Call("s2", {"FT": fts[1]} if fts[1] is not None else {}),
    ]
    return Record(
        "1",
        pos,
        [],
        "A",
        [Substitution("SNV", "T")],
        60.0,
        filters,
        {},
        ["FT"],
        calls,
    )


def test_record_outcome_helper():
    assert (
        fs.record_outcome(make_record([], [None, None]), "A").status
        is fs.FilterStatus.NOT_DONE
    )
    passed = fs.record_outcome(make_record(["PASS"], [None, None]), "A")
    assert passed.status is fs.FilterStatus.PASSED
    assert passed.key == ("1", "100")
    failed = fs.record_outcome(make_record(["q10", "dp"], [None, None]), "A")
    assert failed.status is fs.FilterStatus.FAILED
    assert failed.labels == ("q10", "dp")
    assert failed.level is fs.Level.RECORD


def test_sample_outcomes_helper():
    outcomes = fs.sample_outcomes(make_record(["PASS"], [["PASS"], ["ft1"]]), "A")
    assert len(outcomes) == 2
    assert outcomes[0].status is fs.FilterStatus.PASSED
    assert outcomes[0].key == ("1", "100", "s1")
    assert outcomes[1].status is fs.FilterStatus.FAILED
    assert outcomes[1].labels == ("ft1",)
    assert all(oc.level is fs.Level.SAMPLE for oc in outcomes)
    missing = fs.sample_outcomes(make_record(["PASS"], [None, None]), "A")
    assert all(oc.status is fs.FilterStatus.NOT_DONE for oc in missing)


def test_end_to_end_from_records():
    stats = fs.UnifiedFilterStats(
        fs.FilterMapping([fs.MappingRule("A", "q10", "low_quality")])
    )
    for record in [
        make_record(["PASS"], [["PASS"], ["PASS"]]),
        make_record(["q10"], [["ft1"], [None]], pos=200),
    ]:
        stats.add(fs.record_outcome(record, "A"))
        stats.extend(fs.sample_outcomes(record, "A"))
    rec = stats.reconcile()
    assert rec.ok
    assert sum(r.denominator for r in rec.by_source if r.level is fs.Level.RECORD) == 2
    assert sum(r.denominator for r in rec.by_source if r.level is fs.Level.SAMPLE) == 4
    sample_failures = stats.stats(fs.Level.SAMPLE)[fs.Level.SAMPLE].failures
    # ft1 has no provable correspondence and stays explicit.
    assert sample_failures[fs.unmapped_category("A", "ft1")] == 1
