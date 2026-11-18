import io

import numpy as np
import pytest

import laspy
from laspy.cleaning import (
    ClassificationCleaner,
    ClassificationError,
    ClassificationLedger,
    ClassificationNotConverged,
    ClassificationRuleSet,
    ClassificationSession,
    ExtentRule,
    InvalidClassificationRule,
    NeighborRule,
    SpatialExtent,
    load_classification_history,
    store_classification_history,
)


def grid(n=5, spacing=1.0, z=0.0):
    coordinates = np.arange(n, dtype=np.float64) * spacing
    return coordinates, np.zeros(n), np.full(n, z)


def test_extent_rule_keeps_device_classification_untouched():
    las = laspy.read("tests/data/simple.las")
    original = np.array(las.classification)
    ruleset = ClassificationRuleSet(
        "v1",
        [
            ExtentRule(
                "box",
                2,
                extent=(636000, 637000, 849000, 850000, 0, 1000),
                match_classes=[1],
            )
        ],
    )

    result = laspy.clean_classification(las, ruleset)

    assert result.converged
    assert np.array_equal(np.array(las.classification), original)
    assert not np.array_equal(result.classification, original)
    result.apply_to_las(las)
    assert np.array_equal(np.array(las.classification), original)
    assert np.array_equal(np.array(las.cleaned_classification), result.classification)


def test_each_rule_change_has_a_reason_and_source_metadata():
    x, y, z = grid(3)
    classification = np.zeros(3, dtype=np.uint8)
    ruleset = ClassificationRuleSet(
        "v1", [ExtentRule("make-ground", 2, extent=(-1, 1.5, -1, 1, -1, 1))]
    )

    result = ClassificationCleaner(ruleset).clean(x, y, z, classification)

    assert result.converged
    assert len(result.rule_changes) == 2
    for change in result.rule_changes:
        assert change.rule_id == "make-ground"
        assert change.rule_version == "v1"
        assert "explicit spatial extent" in change.reason
        assert change.evidence["old_class"] == 0
        assert change.evidence["target"] == 2
    assert result.explain(0)[0].new_class == 2
    assert result.explain(2) == ()


def test_neighbor_propagation_spans_several_layers():
    x, y, z = grid(6)
    classification = np.ones(6, dtype=np.uint8)
    ruleset = ClassificationRuleSet(
        "grow",
        [
            ExtentRule("seed", 2, extent=(-0.1, 0.1, -0.1, 0.1, -0.1, 0.1)),
            NeighborRule(
                "grow",
                2,
                radius=1.1,
                neighbor_classes=[2],
                min_count=1,
                priority=-1,
            ),
        ],
    )

    result = ClassificationCleaner(ruleset).clean(x, y, z, classification)

    assert result.converged
    assert np.array_equal(result.classification, np.full(6, 2, dtype=np.uint8))
    neighbor_changes = [
        change for change in result.rule_changes if change.rule_id == "grow"
    ]
    assert len(neighbor_changes) == 5
    assert {change.round for change in neighbor_changes} == {2, 3, 4, 5, 6}
    assert all("neighbours within" in change.reason for change in neighbor_changes)


def test_rule_declaration_order_does_not_change_published_version():
    rng = np.random.default_rng(12)
    x = rng.uniform(0, 50, 200)
    y = rng.uniform(0, 50, 200)
    z = rng.uniform(0, 5, 200)
    classification = rng.integers(0, 3, 200, dtype=np.uint8)
    rules = [
        ExtentRule("high", 3, extent=(5, 40, 5, 40), match_classes=[0], priority=1),
        ExtentRule("low", 2, extent=(0, 25, 0, 25), match_classes=[0], priority=0),
        NeighborRule(
            "neighbor",
            1,
            radius=3,
            neighbor_classes=[2],
            min_count=2,
            priority=-1,
        ),
    ]

    forward = ClassificationCleaner(
        ClassificationRuleSet("ordered", rules)
    ).clean(x, y, z, classification)
    backward = ClassificationCleaner(
        ClassificationRuleSet("ordered", list(reversed(rules)))
    ).clean(x, y, z, classification)

    assert forward.converged and backward.converged
    assert np.array_equal(forward.classification, backward.classification)
    assert forward.version_id == backward.version_id


def test_priority_then_rule_id_resolves_conflicts_deterministically():
    x, y, z = grid(1)
    classification = np.zeros(1, dtype=np.uint8)
    low = ExtentRule("zzz", 2, extent=(-1, 1, -1, 1, -1, 1), priority=0)
    high = ExtentRule("aaa", 3, extent=(-1, 1, -1, 1, -1, 1), priority=5)

    result = ClassificationCleaner(
        ClassificationRuleSet("conflict", [low, high])
    ).clean(x, y, z, classification)

    assert result.classification.tolist() == [3]
    change = result.rule_changes[0]
    assert change.rule_id == "aaa"
    assert change.evidence["alternatives"][0]["rule_id"] == "zzz"


def test_cycle_is_reported_and_never_published():
    x, y, z = grid(3)
    classification = np.full(3, 2, dtype=np.uint8)
    ruleset = ClassificationRuleSet(
        "cycle",
        [
            NeighborRule(
                "a-to-one",
                1,
                radius=1.1,
                neighbor_classes=[2],
                min_count=1,
            ),
            NeighborRule(
                "b-to-two",
                2,
                radius=1.1,
                neighbor_classes=[1],
                min_count=1,
            ),
        ],
    )

    result = ClassificationCleaner(ruleset, max_iterations=5).clean(
        x, y, z, classification
    )

    assert result.status == "cycle"
    assert not result.converged
    assert result.ledger.published_versions() == ()
    assert result.ledger.latest() is None
    with pytest.raises(ClassificationNotConverged) as excinfo:
        result.raise_for_status()
    assert "no version was published" in str(excinfo.value)


def test_iteration_limit_is_reported():
    x, y, z = grid(10)
    classification = np.full(10, 1, dtype=np.uint8)
    ruleset = ClassificationRuleSet(
        "slow",
        [
            ExtentRule("seed", 2, extent=(-0.1, 0.1, -0.1, 0.1, -0.1, 0.1)),
            NeighborRule(
                "spread", 2, radius=1.1, neighbor_classes=[2], min_count=1
            ),
        ],
    )

    result = ClassificationCleaner(ruleset, max_iterations=2).clean(
        x, y, z, classification
    )

    assert result.status == "max_iterations"
    assert not result.converged
    with pytest.raises(ClassificationNotConverged):
        result.apply_to_las(laspy.create())


def test_apply_to_las_rejects_length_mismatch():
    x, y, z = grid(3)
    ruleset = ClassificationRuleSet(
        "v1", [ExtentRule("box", 2, extent=(-1, 3, -1, 1, -1, 1))]
    )
    result = ClassificationCleaner(ruleset).clean(
        x, y, z, np.zeros(3, dtype=np.uint8)
    )
    las = laspy.create()
    las.x = [0, 1]
    las.y = [0, 1]
    las.z = [0, 1]
    with pytest.raises(ClassificationError):
        result.apply_to_las(las)


def test_apply_to_las_cannot_overwrite_standard_classification():
    x, y, z = grid(3)
    ruleset = ClassificationRuleSet(
        "v1", [ExtentRule("box", 2, extent=(-1, 3, -1, 1, -1, 1))]
    )
    result = ClassificationCleaner(ruleset).clean(
        x, y, z, np.zeros(3, dtype=np.uint8)
    )
    las = laspy.create()
    las.x = [0, 1, 2]
    las.y = [0, 1, 2]
    las.z = [0, 1, 2]
    original = np.array(las.classification)

    with pytest.raises(ClassificationError):
        result.apply_to_las(las, dimension_name="classification")

    assert np.array_equal(np.array(las.classification), original)


def test_manual_correction_is_preserved_and_explained():
    x, y, z = grid(4)
    classification = np.zeros(4, dtype=np.uint8)
    ruleset = ClassificationRuleSet(
        "v1", [ExtentRule("box", 2, extent=(-1, 4, -1, 1, -1, 1))]
    )

    result = ClassificationCleaner(ruleset).clean(
        x, y, z, classification, corrections={2: 7}
    )

    assert result.converged
    assert result.classification[2] == 7
    assert result.original_classification[2] == 0
    correction = [
        change for change in result.changes if change.source == "correction"
    ]
    assert len(correction) == 1
    assert correction[0].old_class == 0
    assert correction[0].new_class == 7
    assert correction[0].evidence["original_class"] == 0


def test_session_only_recomputes_affected_closure():
    # Two clusters far apart.  The correction in the second cluster must not
    # cause recomputation of the first cluster.
    x = np.concatenate(
        [np.arange(10, dtype=np.float64), 1000 + np.arange(10, dtype=np.float64)]
    )
    y = np.zeros(20)
    z = np.zeros(20)
    classification = np.ones(20, dtype=np.uint8)
    ruleset = ClassificationRuleSet(
        "growth",
        [
            ExtentRule("seed", 2, extent=(-0.1, 0.1, -0.1, 0.1, -0.1, 0.1)),
            NeighborRule(
                "spread",
                2,
                radius=1.1,
                neighbor_classes=[2],
                min_count=1,
                priority=-1,
            ),
        ],
    )
    session = ClassificationSession(ruleset, x, y, z, classification)
    first = session.run()
    assert first.converged
    # Only the cluster around the explicit seed is touched at all; the far
    # cluster has no supporting neighbour to start from.
    assert set(first.recomputed_indices.tolist()) == set(range(10))

    session.set_correction(15, 9)
    second = session.run()

    assert second.converged
    assert second.version.parent_version_id == first.version_id
    assert second.classification[15] == 9
    assert 0 < second.version.recomputed_indices_count < 20
    assert 15 in second.recomputed_indices
    assert 0 not in second.recomputed_indices


def test_incremental_session_matches_full_recompute():
    rng = np.random.default_rng(42)
    n = 300
    x = rng.uniform(0, 40, n)
    y = rng.uniform(0, 40, n)
    z = rng.uniform(0, 5, n)
    classification = rng.integers(0, 3, n, dtype=np.uint8)
    ruleset = ClassificationRuleSet(
        "mixed",
        [
            ExtentRule(
                "box",
                3,
                extent=(5, 30, 5, 30),
                match_classes=[0, 1],
                priority=1,
            ),
            NeighborRule(
                "near-three",
                2,
                radius=2.5,
                neighbor_classes=[3],
                min_count=2,
                priority=0,
            ),
        ],
    )

    session = ClassificationSession(ruleset, x, y, z, classification)
    first = session.run()
    assert first.converged
    session.set_correction(10, 3)
    session.set_correction(20, 0)
    session.clear_correction(20)
    incremental = session.run()

    full = ClassificationCleaner(ruleset).clean(
        x, y, z, classification, corrections={10: 3}
    )

    assert incremental.converged
    assert np.array_equal(incremental.classification, full.classification)
    assert len(session.ledger.history(incremental.version_id)) == 2


def test_correction_update_order_does_not_change_published_version():
    x, y, z = grid(8)
    classification = np.ones(8, dtype=np.uint8)
    ruleset = ClassificationRuleSet(
        "order",
        [
            ExtentRule("seed", 2, extent=(-0.1, 0.1, -0.1, 0.1, -0.1, 0.1)),
            NeighborRule(
                "spread",
                2,
                radius=1.1,
                neighbor_classes=[2],
                min_count=1,
                priority=-1,
            ),
        ],
    )
    corrections = [(6, 4), (7, 5), (5, 3)]

    first_session = ClassificationSession(ruleset, x, y, z, classification)
    for index, value in corrections:
        first_session.set_correction(index, value)
    first = first_session.run()

    second_session = ClassificationSession(ruleset, x, y, z, classification)
    for index, value in reversed(corrections):
        second_session.set_correction(index, value)
    second = second_session.run()

    assert first.converged and second.converged
    assert np.array_equal(first.classification, second.classification)
    assert first.version_id == second.version_id


def test_incremental_runs_end_at_the_same_published_result():
    x = np.concatenate(
        [np.arange(10, dtype=np.float64), 1000 + np.arange(10, dtype=np.float64)]
    )
    y = np.zeros(20)
    z = np.zeros(20)
    classification = np.ones(20, dtype=np.uint8)
    ruleset = ClassificationRuleSet(
        "incremental",
        [
            ExtentRule("seed", 2, extent=(-0.1, 0.1, -0.1, 0.1, -0.1, 0.1)),
            NeighborRule(
                "spread",
                2,
                radius=1.1,
                neighbor_classes=[2],
                min_count=1,
                priority=-1,
            ),
        ],
    )

    stepwise = ClassificationSession(ruleset, x, y, z, classification)
    stepwise.run()
    stepwise.set_correction(15, 7)
    stepwise.run()
    stepwise.set_correction(12, 8)
    final = stepwise.run()

    batched = ClassificationSession(ruleset, x, y, z, classification)
    batched.set_correction(15, 7)
    batched.set_correction(12, 8)
    batch_result = batched.run()

    assert final.converged and batch_result.converged
    assert np.array_equal(final.classification, batch_result.classification)
    assert final.original_classification.tolist() == classification.tolist()
    assert final.version.corrections == {12: 8, 15: 7}
    # Different update paths keep distinct lineage rows, but the published
    # content is identical and has one content id.
    assert final.version_id != batch_result.version_id
    assert final.published_id == batch_result.published_id
    shared = stepwise.ledger.by_published_id(final.published_id)
    assert {version.version_id for version in shared} >= {final.version_id}
    assert final.published_id is not None
    assert batch_result.version.published_id == final.published_id
    x, y, z = grid(4)
    classification = np.zeros(4, dtype=np.uint8)
    ruleset = ClassificationRuleSet(
        "v1", [ExtentRule("box", 2, extent=(-1, 2.5, -1, 1, -1, 1))]
    )
    session = ClassificationSession(ruleset, x, y, z, classification)
    first = session.run()
    session.set_correction(3, 5)
    second = session.run()

    history = session.ledger.history(second.version_id)
    assert [version.version_id for version in history] == [
        first.version_id,
        second.version_id,
    ]
    assert history[0].original_digest == history[1].original_digest
    assert history[0].rule_version == history[1].rule_version
    assert second.version.corrections == {3: 5}
    assert second.original_classification[3] == 0
    assert second.classification[3] == 5
    assert len(session.ledger.versions) == 2
    assert session.ledger.latest() is second.version


def test_history_round_trips_through_vlr():
    las = laspy.read("tests/data/simple.las")
    ruleset = ClassificationRuleSet(
        "v1",
        [
            ExtentRule(
                "box",
                2,
                extent=(636000, 637000, 849000, 850000, 0, 1000),
                match_classes=[1],
            )
        ],
    )
    result = laspy.clean_classification(las, ruleset)
    result.apply_to_las(las)

    stream = io.BytesIO()
    las.write(stream)
    stream.seek(0)
    reread = laspy.read(stream)
    history = load_classification_history(reread)

    assert len(history) == 1
    assert history.latest().version_id == result.version_id
    assert np.array_equal(
        np.array(reread.cleaned_classification), result.classification
    )
    assert np.array_equal(
        np.array(reread.classification), result.original_classification
    )


def test_storing_history_replaces_previous_history():
    las = laspy.read("tests/data/simple.las")
    ledger_one = ClassificationLedger()
    ledger_two = ClassificationLedger()
    store_classification_history(las, ledger_one)
    store_classification_history(las, ledger_two)

    histories = [
        vlr
        for vlr in las.vlrs
        if vlr.user_id == "laspy.clean" and vlr.record_id == 0x0100
    ]
    assert len(histories) == 1
    assert len(load_classification_history(las)) == 0


def test_spatial_extent_is_closed_and_validated():
    extent = SpatialExtent.from_bounds({"x": (0, 1), "y": (2, 3)})
    assert extent.contains(np.array([0.0, 1.0]), np.array([2.0, 3.0])).all()

    with pytest.raises(InvalidClassificationRule):
        SpatialExtent(1, 0, 0, 1)
    with pytest.raises(InvalidClassificationRule):
        SpatialExtent(0, 1, 0, 1, zmin=0)
    with pytest.raises(InvalidClassificationRule):
        SpatialExtent(0, float("nan"), 0, 1)


@pytest.mark.parametrize("value", [True, 1.5, -1, 256])
def test_invalid_class_values_are_rejected(value):
    with pytest.raises(InvalidClassificationRule):
        ExtentRule("bad", value, extent=(0, 1, 0, 1))


def test_ruleset_round_trips_through_dict():
    ruleset = ClassificationRuleSet(
        "serialised",
        [
            ExtentRule(
                "explicit",
                2,
                extent=(0, 1, 2, 3, 4, 5),
                match_classes=[1],
                description="an explicit box",
            ),
            NeighborRule(
                "near",
                3,
                radius=2.5,
                neighbor_classes=[2],
                min_count=2,
                min_fraction=0.5,
                neighbor_source="seed",
            ),
        ],
    )

    restored = ClassificationRuleSet.from_dict(ruleset.to_dict())

    assert restored.version == ruleset.version
    assert restored.to_dict() == ruleset.to_dict()
    assert restored.max_neighbor_radius == 2.5


def test_duplicate_rule_ids_are_rejected():
    with pytest.raises(InvalidClassificationRule):
        ClassificationRuleSet(
            "v1",
            [
                ExtentRule("same", 1, extent=(0, 1, 0, 1)),
                ExtentRule("same", 2, extent=(0, 1, 0, 1)),
            ],
        )


def test_session_reports_nonconvergence_without_publishing():
    x = np.arange(6, dtype=np.float64)
    y = np.zeros(6)
    z = np.zeros(6)
    classification = np.ones(6, dtype=np.uint8)
    ruleset = ClassificationRuleSet(
        "oscillate",
        [
            NeighborRule(
                "to-two",
                2,
                radius=1.1,
                neighbor_classes=[1],
                min_count=1,
                match_classes=[1],
            ),
            NeighborRule(
                "to-one",
                1,
                radius=1.1,
                neighbor_classes=[2],
                min_count=1,
                match_classes=[2],
                priority=-1,
            ),
        ],
    )
    session = ClassificationSession(ruleset, x, y, z, classification)
    result = session.run()

    assert not result.converged
    assert result.status in {"cycle", "max_iterations"}
    assert session.ledger.published_versions() == ()
    assert session.last_result is None
