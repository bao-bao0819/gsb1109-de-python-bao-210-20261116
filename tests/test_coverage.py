"""Tests for pulse-aware scan coverage and missed-scan review."""

import json

import numpy as np
import pytest

import laspy
from laspy.coverage import (
    CellStatus,
    CoverageConfig,
    CoverageError,
    GridSpec,
    analyze_coverage,
    build_review_list,
    compare_coverage,
)
from laspy.filtering import FilterExpression


GRID = GridSpec(x_min=0.0, y_min=0.0, cell_size=1.0, width=3, height=1)


def make_source(
    *,
    x,
    y=None,
    return_number=None,
    number_of_returns=None,
    gps_time=None,
    classification=None,
    point_source_id=None,
):
    n = len(x)
    source = {
        "x": np.asarray(x, dtype=float),
        "y": np.zeros(n) if y is None else np.asarray(y, dtype=float),
    }
    if return_number is None:
        return_number = np.ones(n, dtype=np.int64)
    if number_of_returns is None:
        number_of_returns = np.ones(n, dtype=np.int64)
    source["return_number"] = np.asarray(return_number, dtype=np.int64)
    source["number_of_returns"] = np.asarray(number_of_returns, dtype=np.int64)
    if gps_time is not None:
        source["gps_time"] = np.asarray(gps_time, dtype=float)
    if classification is not None:
        source["classification"] = np.asarray(classification, dtype=np.int64)
    if point_source_id is not None:
        source["point_source_id"] = np.asarray(point_source_id, dtype=np.int64)
    return source


def test_multiple_returns_of_one_pulse_count_as_one_observation():
    source = make_source(
        x=[0.2, 0.3, 0.4, 0.5],
        return_number=[1, 2, 3, 1],
        number_of_returns=[3, 3, 3, 1],
        gps_time=[10.0, 10.0, 10.0, 20.0],
    )

    result = analyze_coverage(source, grid=GRID)

    assert result.raw_point_counts[0, 0] == 4
    assert result.raw_pulse_counts[0, 0] == 2
    assert result.retained_point_counts[0, 0] == 4
    assert result.retained_pulse_counts[0, 0] == 2
    assert result.summary()["pulses"]["input"] == 2


def test_points_of_one_pulse_in_different_cells_count_once_per_cell():
    source = make_source(
        x=[0.5, 1.5],
        return_number=[1, 2],
        number_of_returns=[2, 2],
        gps_time=[10.0, 10.0],
    )

    result = analyze_coverage(source, grid=GRID)

    assert result.raw_pulse_counts[0, 0] == 1
    assert result.raw_pulse_counts[0, 1] == 1
    assert result.summary()["pulses"]["input"] == 1


def test_without_shared_return_numbers_two_points_are_two_pulses():
    source = make_source(x=[0.2, 0.3], gps_time=[10.0, 11.0])

    result = analyze_coverage(source, grid=GRID)

    assert result.raw_pulse_counts[0, 0] == 2
    assert result.retained_pulse_counts[0, 0] == 2


def test_raw_support_and_retained_points_are_reported_separately():
    source = make_source(
        x=[0.2, 0.3, 1.2],
        gps_time=[10.0, 11.0, 12.0],
        classification=[1, 1, 2],
    )

    result = analyze_coverage(source, grid=GRID, filter_condition="classification == 2")

    assert result.raw_pulse_counts[0, 0] == 2
    assert result.retained_pulse_counts[0, 0] == 0
    assert result.filtered_point_counts[0, 0] == 2
    assert result.status[0, 0] == CellStatus.UNCONFIRMED
    assert "filter" in result.reasons[0, 0]
    assert result.status[0, 1] == CellStatus.COVERED


def test_gap_low_density_and_covered_cells_are_classified():
    source = make_source(
        x=[0.2, 1.2],
        gps_time=[10.0, 11.0],
    )

    result = analyze_coverage(source, grid=GRID, min_pulses=2)

    assert result.status[0, 0] == CellStatus.LOW_DENSITY
    assert result.status[0, 1] == CellStatus.LOW_DENSITY
    assert result.status[0, 2] == CellStatus.GAP

    review = result.review_list()
    statuses = [cell.status for cell in review]
    assert statuses.count(CellStatus.GAP) == 1
    assert statuses.count(CellStatus.LOW_DENSITY) == 2
    assert review[0].status == CellStatus.GAP


def test_min_pulses_threshold_promotes_cell_to_covered():
    source = make_source(x=[0.2, 0.3], gps_time=[10.0, 11.0])

    result = analyze_coverage(source, grid=GRID, min_pulses=2)

    assert result.status[0, 0] == CellStatus.COVERED
    assert result.candidate_cells.tolist() == [[False, True, True]]


@pytest.mark.parametrize(
    "mode, expected_points, expected_pulses",
    [
        ("all", 3, 1),
        ("first", 1, 1),
        ("last", 1, 1),
        ("single", 0, 0),
        ("first_and_last", 2, 1),
    ],
)
def test_multi_return_selection(mode, expected_points, expected_pulses):
    source = make_source(
        x=[0.2, 0.3, 0.4],
        return_number=[1, 2, 3],
        number_of_returns=[3, 3, 3],
        gps_time=[10.0, 10.0, 10.0],
    )

    result = analyze_coverage(source, grid=GRID, multi_return=mode)

    assert result.raw_pulse_counts[0, 0] == 1
    assert result.retained_point_counts[0, 0] == expected_points
    assert result.retained_pulse_counts[0, 0] == expected_pulses


def test_multi_return_requires_return_dimensions():
    source = {"x": np.array([0.2]), "y": np.array([0.0])}

    with pytest.raises(CoverageError):
        analyze_coverage(source, grid=GRID, multi_return="first")


def test_zero_return_number_is_not_silently_accepted_as_a_pulse():
    source = make_source(
        x=[0.2, 0.3],
        return_number=[0, 1],
        number_of_returns=[0, 1],
        gps_time=[10.0, 11.0],
    )

    result = analyze_coverage(source, grid=GRID)

    assert result.invalid_point_count == 1
    assert result.raw_point_counts[0, 0] == 2
    assert result.raw_pulse_counts[0, 0] == 1
    assert result.retained_point_counts[0, 0] == 1


def test_invalid_pulses_make_cell_unconfirmed():
    source = make_source(
        x=[0.2],
        return_number=[0],
        number_of_returns=[0],
        gps_time=[10.0],
    )

    result = analyze_coverage(source, grid=GRID)

    assert result.status[0, 0] == CellStatus.UNCONFIRMED
    assert "invalid" in result.reasons[0, 0]
    cell = next(cell for cell in result.review_list() if cell.col == 0)
    assert cell.status == CellStatus.UNCONFIRMED


def test_known_coverage_limits_assessed_cells():
    source = make_source(x=[0.2, 1.2], gps_time=[10.0, 11.0])

    result = analyze_coverage(
        source,
        grid=GRID,
        known_coverage=(0.0, 0.0, 2.0, 1.0),
    )

    assert result.assessed_mask.tolist() == [[True, True, False]]
    assert result.status[0, 2] == CellStatus.OUTSIDE
    assert all(cell.col != 2 for cell in result.review_list())


def test_known_coverage_polygon():
    source = make_source(x=[0.2, 1.2], gps_time=[10.0, 11.0])

    result = analyze_coverage(
        source,
        grid=GRID,
        known_coverage=[(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)],
    )

    assert result.assessed_mask.tolist() == [[True, True, False]]


def test_points_outside_explicit_grid_are_counted_separately():
    source = make_source(x=[0.2, 9.0], gps_time=[10.0, 11.0])

    result = analyze_coverage(source, grid=GRID)

    assert result.summary()["points"]["outside_grid"] == 1
    assert result.raw_point_counts.sum() == 1


def test_automatic_grid_covers_all_points_without_extra_rows():
    source = make_source(x=[0.5, 1.5], y=[0.5, 0.5], gps_time=[10.0, 11.0])

    result = analyze_coverage(source, cell_size=1.0)

    assert result.grid.shape == (1, 2)
    assert result.raw_point_counts.tolist() == [[1, 1]]
    assert result.summary()["points"]["outside_grid"] == 0


def test_gps_time_tolerance_groups_nearly_equal_times():
    source = make_source(
        x=[0.2, 0.3],
        return_number=[1, 2],
        number_of_returns=[2, 2],
        gps_time=[10.0000001, 10.0000002],
    )

    result = analyze_coverage(source, grid=GRID)

    assert result.raw_pulse_counts[0, 0] == 1
    assert result.retained_pulse_counts[0, 0] == 1


def test_explicit_pulse_id_dimension_is_used():
    source = make_source(
        x=[0.2, 0.3, 1.2],
        return_number=[1, 2, 1],
        number_of_returns=[2, 2, 1],
    )
    source["pulse_id"] = np.array([7, 7, 8])

    result = analyze_coverage(source, grid=GRID)

    assert result.raw_pulse_counts.tolist() == [[1, 1, 0]]
    assert result.pulse_identity_methods == ["pulse_id"]


def test_custom_pulse_key():
    source = make_source(
        x=[0.2, 0.3],
        gps_time=[10.0, 20.0],
        point_source_id=[5, 5],
    )

    result = analyze_coverage(source, grid=GRID, pulse_key=["point_source_id"])

    # both points share pulse key 5, even though gps_time differs
    assert result.raw_pulse_counts[0, 0] == 1


def test_return_run_fallback_marks_identity_unreliable():
    source = make_source(
        x=[0.2, 0.3, 1.2],
        return_number=[1, 2, 1],
        number_of_returns=[2, 2, 1],
    )

    result = analyze_coverage(source, grid=GRID)

    assert result.pulse_identity_methods == ["returns"]
    assert result.pulse_identity_reliable is False
    assert result.raw_pulse_counts.tolist() == [[1, 1, 0]]


def test_coverage_config_and_threshold_alias():
    source = make_source(x=[0.2], gps_time=[10.0])
    config = CoverageConfig(grid=GRID, min_pulses=2)

    result = analyze_coverage(source, config=config)

    assert result.status[0, 0] == CellStatus.LOW_DENSITY


def test_grid_and_config_disagreement_is_rejected():
    source = make_source(x=[0.2], gps_time=[10.0])
    other = GridSpec(0.0, 0.0, 1.0, 1, 1)
    config = CoverageConfig(grid=GRID)

    with pytest.raises(CoverageError):
        analyze_coverage(source, grid=other, config=config)


def test_review_list_orders_gaps_before_low_density():
    source = make_source(x=[0.2, 1.2], gps_time=[10.0, 11.0])

    result = analyze_coverage(source, grid=GRID, min_pulses=2)
    review = result.review_list()

    assert review[0].status == CellStatus.GAP
    assert review[-1].status == CellStatus.LOW_DENSITY


def test_result_to_json_and_csv(tmp_path):
    source = make_source(x=[0.2], gps_time=[10.0])
    result = analyze_coverage(source, grid=GRID, min_pulses=2)

    json_path = tmp_path / "review.json"
    csv_path = tmp_path / "review.csv"
    result.to_json(json_path)
    result.to_csv(csv_path)

    document = json.loads(json_path.read_text())
    assert document["summary"]["cells"]["low_density"] == 1
    assert any(row["cell_id"] == "r0c0" for row in document["review"])

    lines = csv_path.read_text().strip().splitlines()
    assert lines[0].startswith("cell_id,")
    assert any(line.startswith("r0c0,") for line in lines[1:])


def test_build_review_list_alias():
    source = make_source(x=[0.2], gps_time=[10.0])

    result = build_review_list(source, grid=GRID, min_pulses=2)

    cell = next(cell for cell in result.review_list() if cell.col == 0)
    assert cell.status == CellStatus.LOW_DENSITY


def test_filter_expression_supports_and_or_and_in():
    expression = FilterExpression.parse_string(
        "classification in [2, 3] and intensity >= 10"
    )
    source = make_source(
        x=[0.2, 0.3, 0.4],
        gps_time=[10.0, 11.0, 12.0],
        classification=[2, 3, 4],
    )
    source["intensity"] = np.array([20, 5, 20])

    result = analyze_coverage(source, grid=GRID, filter_condition=expression)

    assert result.retained_point_counts[0, 0] == 1
    assert result.filtered_point_counts[0, 0] == 2


def test_compare_new_coverage_and_repeat_observations():
    baseline = make_source(x=[0.2], gps_time=[10.0])
    added = make_source(x=[0.3, 1.5], gps_time=[11.0, 12.0])

    comparison = compare_coverage(baseline, added, grid=GRID)

    assert comparison.newly_covered_cells.tolist() == [[False, True, False]]
    assert comparison.repeat_observation_cells.tolist() == [[True, False, False]]
    delta = comparison.summary()["delta"]
    assert delta["newly_covered_cells"] == 1
    assert delta["repeat_observation_cells"] == 1
    assert delta["new_coverage_points"] == 1
    assert delta["repeat_observation_points"] == 1


def test_compare_does_not_report_already_covered_cell_as_new():
    baseline = make_source(x=[0.2], gps_time=[10.0])
    added = make_source(x=[0.3], gps_time=[11.0])

    comparison = compare_coverage(baseline, added, grid=GRID)

    assert comparison.new_coverage_cell_count == 0
    assert comparison.repeat_observation_cell_count == 1


def test_compare_results_can_recompute_combined_support():
    baseline = make_source(x=[0.2], gps_time=[10.0])
    added = make_source(x=[0.3, 1.5], gps_time=[11.0, 12.0])

    baseline_result = analyze_coverage(baseline, grid=GRID)
    added_result = analyze_coverage(added, grid=GRID)
    comparison = compare_coverage(baseline_result, added_result)

    assert comparison.combined.raw_pulse_counts.tolist() == [[2, 1, 0]]
    assert comparison.new_coverage_cell_count == 1
    assert comparison.repeat_observation_cell_count == 1


def test_compare_requires_combined_when_source_data_is_unavailable():
    baseline = make_source(x=[0.2], gps_time=[10.0])
    added = make_source(x=[1.2], gps_time=[11.0])
    baseline_result = analyze_coverage(baseline, grid=GRID)
    baseline_result.source_data = None

    with pytest.raises(CoverageError):
        compare_coverage(baseline_result, added)


def test_compare_remaining_candidates():
    baseline = make_source(x=[0.2], gps_time=[10.0])
    added = make_source(x=[1.2], gps_time=[11.0])

    comparison = compare_coverage(baseline, added, grid=GRID, min_pulses=2)

    assert comparison.newly_covered_cells.sum() == 0
    assert comparison.remaining_candidate_cells.tolist() == [[True, True, True]]


def test_compare_uses_pulses_not_returns_for_added_strip():
    baseline = make_source(x=[0.2], gps_time=[10.0])
    added = make_source(
        x=[1.2, 1.3, 1.4],
        return_number=[1, 2, 3],
        number_of_returns=[3, 3, 3],
        gps_time=[11.0, 11.0, 11.0],
    )

    comparison = compare_coverage(baseline, added, grid=GRID, min_pulses=2)

    assert comparison.newly_covered_cells.sum() == 0
    assert comparison.remaining_candidate_cells[0, 1]
    assert comparison.added_pulse_counts[0, 1] == 1


def test_compare_to_json_and_csv(tmp_path):
    baseline = make_source(x=[0.2], gps_time=[10.0])
    added = make_source(x=[1.2], gps_time=[11.0])
    comparison = compare_coverage(baseline, added, grid=GRID)

    json_path = tmp_path / "delta.json"
    csv_path = tmp_path / "delta.csv"
    comparison.to_json(json_path)
    comparison.to_csv(csv_path)

    document = json.loads(json_path.read_text())
    assert document["summary"]["delta"]["newly_covered_cells"] == 1
    assert document["cells"][0]["classification"] == "new_coverage"

    lines = csv_path.read_text().strip().splitlines()
    assert lines[0].startswith("cell_id,")
    assert any("new_coverage" in line for line in lines[1:])


def test_las_data_source_end_to_end(tmp_path):
    las = laspy.create(point_format=1)
    las.x = np.array([0.5, 0.6, 0.7, 1.5])
    las.y = np.zeros(4)
    las.z = np.zeros(4)
    las.return_number = np.array([1, 1, 2, 1])
    las.number_of_returns = np.array([2, 2, 2, 1])
    las.gps_time = np.array([10.0, 10.0, 10.0, 11.0])
    las.classification[:] = 2

    path = tmp_path / "strips.las"
    las.write(path)

    result = analyze_coverage(path, grid=GridSpec(0.0, 0.0, 1.0, 2, 1))

    assert result.raw_pulse_counts.tolist() == [[1, 1]]
    assert result.summary()["pulses"]["identity_methods"] == ["gps_time"]


def test_min_density_uses_cell_area():
    source = make_source(x=[0.2, 0.3], gps_time=[10.0, 11.0])

    result = analyze_coverage(source, grid=GRID, min_points=0, min_density=3.0)

    assert result.config.min_points_effective == 3.0
    assert result.status[0, 0] == CellStatus.LOW_DENSITY


def test_min_density_is_capped_by_min_points():
    source = make_source(x=[0.2, 0.3, 0.4], gps_time=[10.0, 11.0, 12.0])

    result = analyze_coverage(source, grid=GRID, min_points=2, min_density=1.0)

    assert result.config.min_points_effective == 2.0
    assert result.status[0, 0] == CellStatus.COVERED


def test_repeat_observation_requires_retained_evidence():
    baseline = make_source(x=[0.2], gps_time=[10.0], classification=[2])
    added = make_source(
        x=[0.3, 0.4],
        gps_time=[11.0, 12.0],
        classification=[1, 1],
    )

    comparison = compare_coverage(
        baseline,
        added,
        grid=GRID,
        filter_condition="classification == 2",
    )

    assert comparison.repeat_observation_cell_count == 0


def test_return_runs_never_split_a_gps_time_group():
    source = make_source(
        x=[0.2, 0.3],
        return_number=[1, 2],
        number_of_returns=[2, 2],
        gps_time=[10.0, 10.0],
    )

    result = analyze_coverage(source, grid=GRID)

    assert result.raw_pulse_counts[0, 0] == 1


def test_compare_without_explicit_grid_covers_added_strip():
    baseline = make_source(x=[0.5], y=[0.5], gps_time=[10.0])
    added = make_source(x=[4.5], y=[0.5], gps_time=[11.0])

    comparison = compare_coverage(baseline, added, cell_size=1.0)

    assert comparison.grid.width == 5
    assert comparison.newly_covered_cells.tolist() == [
        [False, False, False, False, True]
    ]
    assert comparison.summary()["delta"]["newly_covered_cells"] == 1


def test_retained_totals_cover_all_input_points():
    source = make_source(x=[0.2, 9.0], gps_time=[10.0, 11.0])

    result = analyze_coverage(source, grid=GRID)

    # Totals cover all input points; cell counts only cover the grid.
    assert result.retained_point_count == 2
    assert result.retained_point_counts.sum() == 1
    assert result.summary()["points"]["input"] == 2
    assert result.summary()["points"]["outside_grid"] == 1


def test_directory_source_is_expanded(tmp_path):
    first = laspy.create(point_format=1)
    first.x = np.array([0.5])
    first.y = np.array([0.5])
    first.z = np.zeros(1)
    first.return_number[:] = 1
    first.number_of_returns[:] = 1
    first.gps_time = np.array([10.0])
    first.write(tmp_path / "a.las")

    second = laspy.create(point_format=1)
    second.x = np.array([2.5])
    second.y = np.array([0.5])
    second.z = np.zeros(1)
    second.return_number[:] = 1
    second.number_of_returns[:] = 1
    second.gps_time = np.array([11.0])
    second.write(tmp_path / "b.las")

    result = analyze_coverage(tmp_path, cell_size=1.0)

    assert result.grid.width == 3
    assert result.summary()["pulses"]["input"] == 2
    assert result.raw_pulse_counts.tolist() == [[1, 0, 1]]


def test_comma_separated_pulse_key_fields():
    source = make_source(
        x=[0.2, 0.3],
        gps_time=[10.0, 20.0],
        point_source_id=[1, 1],
    )

    result = analyze_coverage(
        source, grid=GRID, pulse_key="point_source_id, gps_time"
    )

    # point_source_id is equal but gps_time differs -> two pulses
    assert result.raw_pulse_counts[0, 0] == 2
    assert result.pulse_identity_methods == ["custom_key"]


def test_compare_rejects_results_with_different_thresholds():
    baseline = make_source(x=[0.2], gps_time=[10.0])
    added = make_source(x=[1.2], gps_time=[11.0])
    baseline_result = analyze_coverage(baseline, grid=GRID, min_pulses=1)
    added_result = analyze_coverage(added, grid=GRID, min_pulses=2)

    with pytest.raises(CoverageError):
        compare_coverage(baseline_result, added_result)


def test_empty_source_and_all_invalid_points():
    empty = {
        "x": np.array([], dtype=float),
        "y": np.array([], dtype=float),
        "gps_time": np.array([], dtype=float),
    }
    result = analyze_coverage(empty, grid=GRID)
    assert result.input_point_count == 0
    assert result.status.tolist() == [[CellStatus.GAP] * 3]

    non_finite = {
        "x": np.array([np.nan]),
        "y": np.array([0.0]),
        "gps_time": np.array([1.0]),
    }
    result = analyze_coverage(non_finite, grid=GRID)
    assert result.raw_point_counts.sum() == 0
    assert result.status.tolist() == [[CellStatus.GAP] * 3]
