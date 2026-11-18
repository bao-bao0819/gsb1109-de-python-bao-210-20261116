import csv
import json

import numpy as np
import pytest

import laspy
from laspy.repair import (
    ReturnCandidate,
    ReturnRepairError,
    plan_return_repair,
    validate_returns,
)


def make_las(return_numbers, number_of_returns, point_format=6):
    las = laspy.create(point_format=point_format)
    count = len(return_numbers)
    las.points.array = np.zeros(count, dtype=las.points.array.dtype)
    las.return_number = np.asarray(return_numbers, dtype=np.uint8)
    las.number_of_returns = np.asarray(number_of_returns, dtype=np.uint8)
    return las


def test_out_of_range_return_number_is_reported():
    las = make_las([1, 2, 5], [3, 3, 3], point_format=3)

    issues = validate_returns(las, pulse_ids=[0, 0, 0])
    kinds = {issue.kind for issue in issues}

    assert "return_exceeds_total" in kinds


def test_minimum_cost_repair_with_unique_optimum():
    las = make_las([1, 1], [1, 1])
    candidates = {
        0: [ReturnCandidate(1, 2, cost=1.0)],
        1: [ReturnCandidate(2, 2, cost=1.0)],
    }

    plan = plan_return_repair(las, pulse_ids=[7, 7], candidates=candidates)

    assert plan.optimal_cost == pytest.approx(2.0)
    assert not plan.has_review
    assert plan.report.stats_optimal.is_valid
    assert plan.report.stats_after.is_valid
    assert np.array(plan.revised.return_number).tolist() == [1, 2]
    assert np.array(plan.revised.number_of_returns).tolist() == [2, 2]


def test_ambiguous_optima_are_left_for_review():
    las = make_las([1, 1, 1, 1], [1, 1, 1, 1])

    plan = plan_return_repair(las, pulse_ids=[0, 0, 1, 1])

    assert plan.has_review
    assert not plan.report.stats_after.is_valid
    assert plan.report.stats_optimal.is_valid
    review_fields = {
        (item.point_index, item.field) for item in plan.report.review_items
    }
    assert (0, "return_number") in review_fields


def test_source_object_is_not_modified_until_apply():
    las = make_las([3, 1], [3, 1])
    original_r = np.array(las.return_number).copy()
    original_n = np.array(las.number_of_returns).copy()

    plan = plan_return_repair(
        las,
        pulse_ids=[1, 1],
        candidates={0: [(1, 3)], 1: [(2, 3)]},
    )

    assert np.array(las.return_number).tolist() == original_r.tolist()
    assert np.array(las.number_of_returns).tolist() == original_n.tolist()

    plan.apply_in_place()
    assert np.array(plan.source.return_number).tolist() == [1, 2]
    assert np.array(plan.source.number_of_returns).tolist() == [3, 3]


def test_apply_only_changes_consensus_fields_and_keeps_original_codes():
    las = make_las([4, 1], [4, 1])
    original_r = np.array(las.return_number).copy()
    original_n = np.array(las.number_of_returns).copy()

    report = las.repair_returns([3, 3])

    assert report.original_return_numbers.tolist() == original_r.tolist()
    assert report.original_number_of_returns.tolist() == original_n.tolist()
    applied = {(change.point_index, change.field) for change in report.applied_changes}
    reviewed = {(item.point_index, item.field) for item in report.review_items}
    assert applied.isdisjoint(reviewed)


def test_bit_widths_are_taken_from_the_point_format():
    las = make_las([1, 1], [1, 1], point_format=3)

    plan = plan_return_repair(las, pulse_ids=[0, 0])
    assert plan.report.return_number_limits == (7, 7)
    assert plan.report.number_of_returns_limits == (7, 7)

    new_format = make_las([1, 1], [1, 1], point_format=6)
    plan = plan_return_repair(new_format, pulse_ids=[0, 0])
    assert plan.report.return_number_limits == (15, 15)
    assert plan.report.number_of_returns_limits == (15, 15)


def test_candidate_exceeding_point_format_width_is_rejected():
    las = make_las([1, 1], [1, 1], point_format=3)

    with pytest.raises(ReturnRepairError):
        plan_return_repair(
            las,
            pulse_ids=[0, 0],
            candidates={0: [(8, 8)], 1: [(8, 8)]},
        )

    wider = make_las([1, 1], [1, 1], point_format=6)
    plan = plan_return_repair(
        wider,
        pulse_ids=[0, 0],
        candidates={0: [(8, 9)], 1: [(9, 9)]},
    )
    assert plan.optimal_cost == pytest.approx(4.0)


def test_mixed_legacy_and_new_source_formats_narrow_the_limit():
    las = make_las([1, 1, 1], [1, 1, 1], point_format=6)

    plan = plan_return_repair(
        las,
        pulse_ids=[0, 0, 1],
        source_point_format_ids=[3, 3, 6],
    )

    assert plan.report.return_number_limits == (7, 7, 15)
    assert plan.report.number_of_returns_limits == (7, 7, 15)


def test_candidate_exceeding_width_is_rejected():
    las = make_las([1, 1, 1, 1], [1, 1, 1, 1], point_format=6)

    with pytest.raises(ReturnRepairError):
        plan_return_repair(
            las,
            pulse_ids=[0, 0, 1, 1],
            candidates={0: [(8, 8)]},
            source_point_format_ids=[3, 3, 6, 6],
        )


def test_flightline_control_counts_are_matched_exactly():
    las = make_las([1, 1, 1, 1], [1, 1, 1, 1], point_format=3)
    candidates = {
        0: [(1, 2)],
        1: [(2, 2)],
        2: [(1, 3)],
        3: [(2, 3)],
        4: [(3, 3)],
    }
    las = make_las([1, 1, 1, 1, 1], [1, 1, 1, 1, 1], point_format=3)

    plan = plan_return_repair(
        las,
        pulse_ids=[0, 0, 1, 1, 2],
        flightline_ids=["A", "A", "A", "A", "B"],
        candidates=candidates,
        return_number_counts={"A": [2, 2], "B": [0, 0, 1]},
        number_of_returns_counts={"A": [0, 2, 2], "B": [0, 0, 1]},
    )

    assert plan.optimal_cost == pytest.approx(8.0)
    assert not plan.has_review
    assert np.array(plan.revised.return_number).tolist() == [1, 2, 1, 2, 3]


def test_missing_flightline_control_is_rejected():
    las = make_las([1, 1, 1, 1], [1, 1, 1, 1], point_format=3)

    with pytest.raises(ReturnRepairError):
        plan_return_repair(
            las,
            pulse_ids=[0, 0, 1, 1],
            flightline_ids=["A", "A", "B", "B"],
            return_number_counts={"A": [1, 1]},
        )


def test_inconsistent_control_totals_are_rejected():
    las = make_las([1, 1], [1, 1], point_format=3)

    with pytest.raises(ReturnRepairError):
        plan_return_repair(
            las,
            pulse_ids=[0, 0],
            flightline_ids=["A", "A"],
            return_number_counts={"A": [1, 0, 0]},
        )


def test_pulse_marker_is_required_to_be_explicit():
    las = make_las([1, 1], [1, 1])

    with pytest.raises(ReturnRepairError):
        plan_return_repair(las, pulse_ids=[0])

    with pytest.raises(ReturnRepairError):
        plan_return_repair(las, pulse_ids=[[0], [0]])


def test_pulse_membership_is_never_inferred_from_coordinates():
    las = make_las([1, 1, 1, 1], [1, 1, 1, 1])
    las.X = np.array([0, 0, 1000, 1000], dtype=np.int32)
    las.Y = np.zeros(4, dtype=np.int32)
    las.Z = np.zeros(4, dtype=np.int32)

    plan = plan_return_repair(las, pulse_ids=[0, 1, 2, 3])

    assert plan.report.pulse_count == 4
    assert plan.optimal_cost == pytest.approx(0.0)


def test_duplicate_return_numbers_within_a_pulse_are_detected():
    las = make_las([1, 1, 2], [2, 2, 2])

    issues = validate_returns(las, pulse_ids=[0, 0, 1])

    assert any(
        issue.kind == "pulse_duplicate_return_number"
        for issue in issues
        if issue.point_index == 1
    )


def test_pulse_with_multiple_number_of_returns_is_detected():
    las = make_las([1, 2], [2, 1])

    issues = validate_returns(las, pulse_ids=[0, 0])

    assert any(
        issue.kind == "pulse_inconsistent_number_of_returns" for issue in issues
    )


def test_pulse_spanning_flightlines_is_rejected():
    las = make_las([1, 2], [2, 2])

    with pytest.raises(ReturnRepairError):
        plan_return_repair(las, pulse_ids=[0, 0], flightline_ids=["A", "B"])


def test_report_is_json_serializable_and_csv_is_written(tmp_path):
    las = make_las([3, 3, 1, 1], [3, 3, 1, 1])

    plan = plan_return_repair(las, pulse_ids=[0, 0, 1, 1])
    json_path = tmp_path / "report.json"
    csv_path = tmp_path / "report.csv"

    payload = plan.to_json(json_path)
    parsed = json.loads(payload)
    assert parsed["optimal_cost"] == pytest.approx(plan.optimal_cost)
    assert json.loads(json_path.read_text(encoding="utf-8"))["points"]

    plan.to_csv(csv_path)
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(las.points)
    assert {"original_return_number", "optimal_return_number"} <= set(rows[0])


def test_before_and_after_statistics_are_recorded():
    las = make_las([5, 1], [5, 1], point_format=3)
    original_header = las.header.number_of_points_by_return.copy()

    plan = plan_return_repair(las, pulse_ids=[0, 0], pulse_size_must_match=True)

    before = plan.report.stats_before
    after = plan.report.stats_after
    assert before.point_count == after.point_count == 2
    assert before.invalid_point_count > 0
    assert after.invalid_point_count <= before.invalid_point_count
    assert np.array_equal(las.header.number_of_points_by_return, original_header)
    assert plan.report.stats_optimal.is_valid


def test_real_file_can_be_planned_without_changing_it():
    las = laspy.read("tests/data/test1_4.las")
    original = las.points.array.copy()

    plan = las.plan_return_repair(np.arange(len(las.points)))

    assert plan.optimal_cost == pytest.approx(0.0)
    assert not plan.has_review
    assert np.array_equal(las.points.array, original)


def test_review_items_list_possible_values_and_reason():
    las = make_las([1, 1, 1, 1], [1, 1, 1, 1])

    plan = plan_return_repair(las, pulse_ids=[0, 0, 1, 1])

    for item in plan.report.review_items:
        assert len(item.possible_values) > 1
        assert item.reason == "optimal revisions disagree"
        assert item.original in (1,)


def test_consensus_and_optimal_template_stay_in_sync():
    las = make_las([1, 1, 1, 1], [1, 1, 1, 1])

    plan = plan_return_repair(las, pulse_ids=[0, 0, 1, 1])

    assert plan.report.stats_optimal.is_valid
    assert plan.report.optimal_return_numbers.shape == (4,)
    assert plan.report.optimal_number_of_returns.shape == (4,)
    for index in range(4):
        consensus_r = int(plan.report.revised_return_numbers[index])
        consensus_n = int(plan.report.revised_number_of_returns[index])
        reviewed = {
            item.field
            for item in plan.report.review_items
            if item.point_index == index
        }
        if "return_number" not in reviewed:
            assert consensus_r == int(plan.report.optimal_return_numbers[index])


def test_concrete_optimal_revision_has_the_optimal_cost():
    las = make_las([2, 2, 1, 1], [1, 4, 1, 2])
    candidates = {
        0: [(1, 3), (3, 4), (4, 4)],
        1: [(3, 3), (3, 4)],
        2: [(1, 2), (1, 4), (4, 4)],
        3: [(2, 4)],
    }

    plan = plan_return_repair(
        las, pulse_ids=[0, 0, 1, 1], candidates=candidates
    )

    optimal_cost = sum(
        int(plan.report.optimal_return_numbers[i] != las.return_number[i])
        + int(plan.report.optimal_number_of_returns[i] != las.number_of_returns[i])
        for i in range(len(las.points))
    )
    assert plan.optimal_cost == pytest.approx(optimal_cost)
    assert plan.report.stats_optimal.is_valid


def test_costs_can_be_customised():
    las = make_las([1, 1], [1, 1])

    plan = plan_return_repair(
        las,
        pulse_ids=[0, 0],
        candidates={
            0: [ReturnCandidate(1, 2, cost=10.0)],
            1: [ReturnCandidate(2, 2, cost=1.0)],
        },
    )

    assert plan.optimal_cost == pytest.approx(11.0)
