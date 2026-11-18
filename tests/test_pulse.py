from types import SimpleNamespace

import numpy as np
import pytest

import laspy
from laspy.pulse import (
    InvalidReason,
    MissingReason,
    build_pulse_groups,
    group_into_pulses,
    group_pulses,
)


def make_points(**overrides):
    data = {
        "gps_time": [10.0],
        "return_number": [1],
        "number_of_returns": [1],
        "point_source_id": [1],
        "scanner_channel": [0],
    }
    data.update(overrides)
    count = max(len(value) for value in data.values())
    return {
        name: list(value) * count if len(value) == 1 else list(value)
        for name, value in data.items()
    }


def test_single_complete_pulse_is_proven():
    result = group_pulses(
        make_points(
            gps_time=[10.0, 10.000001, 10.000002],
            return_number=[1, 2, 3],
            number_of_returns=[3, 3, 3],
        ),
        time_precision=0.00001,
    )

    assert result.ambiguous is False
    assert result.proven_pulse_count == 1
    group = result.groups[0]
    assert group.return_numbers == (1, 2, 3)
    assert group.point_indices == (0, 1, 2)
    assert group.is_complete
    assert result.pulse_ids().tolist() == [0, 0, 0]


def test_equal_time_on_different_channels_are_distinct_pulses():
    result = group_pulses(
        make_points(
            gps_time=[10.0, 10.0],
            scanner_channel=[0, 1],
        )
    )

    assert [(group.channel, group.point_indices) for group in result.groups] == [
        (0, (0,)),
        (1, (1,)),
    ]


def test_equal_time_on_different_flight_lines_are_distinct_pulses():
    result = group_pulses(
        make_points(
            gps_time=[10.0, 10.0],
            point_source_id=[3, 4],
        )
    )

    assert len(result.groups) == 2


def test_same_return_number_cannot_be_claimed_twice():
    result = group_pulses(
        make_points(
            gps_time=[10.0, 10.0],
            return_number=[1, 1],
            number_of_returns=[2, 2],
        )
    )

    assert result.ambiguous is False
    assert len(result.groups) == 2
    for group in result.groups:
        assert group.return_numbers == (1,)
        assert group.missing_returns == (2,)


def test_crossing_candidates_are_ambiguous_and_claim_no_pair():
    result = group_pulses(
        make_points(
            gps_time=[10.0, 10.04, 10.02, 10.05],
            return_number=[1, 1, 2, 2],
            number_of_returns=[2, 2, 2, 2],
        ),
        time_precision=0.06,
    )

    assert result.ambiguous
    assert result.groups == ()
    cluster = result.ambiguous_clusters[0]
    assert cluster.point_indices == (0, 1, 2, 3)
    assert cluster.minimum_pulse_count == 2
    assert cluster.exhaustive
    assert cluster.grouping_count > 1
    assert cluster.common_groups == ()
    assert cluster.common_pairs == ()
    assert result.pulse_ids().tolist() == [-1, -1, -1, -1]


def test_shared_pair_is_confirmed_when_all_groupings_keep_it():
    # The two return-2 points (2 and 3) cannot be separated from each other:
    # only one pulse can act as their partner, while the two return-1 points
    # compete for it. The pair (2, 3) is in every minimum grouping even though
    # no complete pulse membership is proven.
    result = group_pulses(
        make_points(
            gps_time=[0.4, 0.3, 0.3, 0.2],
            return_number=[2, 4, 2, 1],
            number_of_returns=[4, 4, 4, 4],
        ),
        time_precision=0.15,
        max_solutions=100,
    )

    assert result.groups == ()
    cluster = result.ambiguous_clusters[0]
    assert cluster.common_groups == ()
    assert cluster.common_pairs == ((2, 3),)


def test_file_order_does_not_decide_grouping():
    values = make_points(
        gps_time=[10.0, 10.0, 10.0, 10.0],
        return_number=[1, 2, 1, 2],
        number_of_returns=[2, 2, 2, 2],
        scanner_channel=[0, 0, 1, 1],
    )
    forward = group_pulses(values)
    reverse_values = {name: list(reversed(array)) for name, array in values.items()}
    reverse = group_pulses(reverse_values)

    forward_members = sorted(
        tuple(sorted(group.return_numbers)) for group in forward.groups
    )
    reverse_members = sorted(
        tuple(sorted(group.return_numbers)) for group in reverse.groups
    )
    assert forward_members == reverse_members
    assert [group.channel for group in forward.groups] == sorted(
        group.channel for group in forward.groups
    )


def test_missing_return_is_unknown_without_evidence():
    result = group_pulses(
        make_points(
            gps_time=[10.0, 10.0, 10.0],
            return_number=[1, 2, 3],
            number_of_returns=[4, 4, 4],
        )
    )

    group = result.groups[0]
    assert group.missing_returns == (4,)
    missing = group.missing[0]
    assert missing.return_number == 4
    assert missing.reason is MissingReason.UNKNOWN
    assert missing.filtered_indices == ()
    assert group.has_unknown_missing


def test_missing_return_is_known_filtered_when_excluded_point_matches():
    excluded = make_points(
        gps_time=[10.0],
        return_number=[4],
        number_of_returns=[4],
    )
    result = group_pulses(
        make_points(
            gps_time=[10.0, 10.0, 10.0],
            return_number=[1, 2, 3],
            number_of_returns=[4, 4, 4],
        ),
        excluded=excluded,
    )

    missing = result.groups[0].missing[0]
    assert missing.reason is MissingReason.KNOWN_FILTERED
    assert missing.filtered_indices == (0,)
    assert not result.groups[0].has_unknown_missing


def test_known_filtered_requires_matching_identity_fields():
    excluded = make_points(
        gps_time=[10.0],
        return_number=[4],
        number_of_returns=[4],
        scanner_channel=[1],
    )
    result = group_pulses(
        make_points(
            gps_time=[10.0, 10.0, 10.0],
            return_number=[1, 2, 3],
            number_of_returns=[4, 4, 4],
        ),
        excluded=excluded,
    )

    assert result.groups[0].missing[0].reason is MissingReason.UNKNOWN


def test_time_precision_is_required_for_quantised_times():
    points = make_points(
        gps_time=[10.0, 10.2],
        return_number=[1, 2],
        number_of_returns=[2, 2],
    )
    exact = group_pulses(points)
    tolerant = group_pulses(points, time_precision=0.5)

    assert len(exact.groups) == 2
    assert len(tolerant.groups) == 1
    assert tolerant.time_precision == 0.5


def test_original_identity_is_preserved():
    identity = np.array([101, 205, 309])
    result = group_pulses(
        make_points(
            gps_time=[10.0, 10.1, 10.2],
            return_number=[1, 2, 3],
            number_of_returns=[3, 3, 3],
        ),
        identity=identity,
        time_precision=0.5,
    )

    group = result.groups[0]
    assert group.point_indices == (0, 1, 2)
    assert group.original_indices == (101, 205, 309)


def test_invalid_declarations_are_reported_not_grouped():
    result = group_pulses(
        make_points(
            gps_time=[10.0, 10.0, 10.0],
            return_number=[0, 5, 1],
            number_of_returns=[1, 2, 0],
        )
    )

    reasons = {point.point_index: point.reason for point in result.invalid_points}
    assert reasons == {
        0: InvalidReason.INVALID_RETURN_NUMBER,
        1: InvalidReason.RETURN_EXCEEDS_DECLARATION,
        2: InvalidReason.INVALID_NUMBER_OF_RETURNS,
    }
    assert result.pulse_ids().tolist() == [-2, -2, -2]


def test_enumerated_minimum_grouping_respects_bounds():
    result = group_pulses(
        make_points(
            gps_time=[10.0] * 6,
            return_number=[1, 1, 1, 2, 2, 2],
            number_of_returns=[2] * 6,
        ),
        time_precision=0.02,
        max_steps=1,
    )

    assert result.ambiguous
    cluster = result.ambiguous_clusters[0]
    assert cluster.exhaustive is False
    assert cluster.common_groups == ()
    assert cluster.common_pairs == ()


def test_lasdata_integration_reads_standard_dimensions():
    header = laspy.LasHeader(point_format=6)
    las = laspy.LasData(header)
    las.x = [0.0, 0.0, 0.0]
    las.y = [0.0, 0.0, 0.0]
    las.z = [0.0, 0.0, 0.0]
    las.gps_time = [100.0, 100.000001, 100.000002]
    las.return_number = [1, 2, 3]
    las.number_of_returns = [3, 3, 3]
    las.point_source_id = [12, 12, 12]
    las.scanner_channel = [2, 2, 2]

    result = las.pulse_groups(time_precision=0.00001)

    assert len(result.groups) == 1
    group = result.groups[0]
    assert group.flight_line == 12
    assert group.channel == 2
    assert group.point_indices == (0, 1, 2)


def test_mapping_and_object_sources():
    mapping = SimpleNamespace(
        gps_time=np.array([1.0]),
        return_number=np.array([1]),
        number_of_returns=np.array([1]),
        point_source_id=np.array([0]),
        scanner_channel=np.array([0]),
    )
    assert len(group_pulses(mapping).groups) == 1


def test_aliases_build_the_same_result():
    points = make_points(
        gps_time=[1.0, 1.1],
        return_number=[1, 2],
        number_of_returns=[2, 2],
    )
    base = group_pulses(points, tolerance=0.5)
    assert build_pulse_groups(points, time_tolerance=0.5) == base
    assert group_into_pulses(points, time_precision=0.5) == base


def test_invalid_arguments():
    with pytest.raises(ValueError):
        group_pulses(make_points(), time_precision=-1.0)
    with pytest.raises(ValueError):
        group_pulses(make_points(), time_precision=float("nan"))
    with pytest.raises(ValueError):
        group_pulses(make_points(), max_solutions=0)
    with pytest.raises(ValueError):
        group_pulses({"gps_time": [1.0]})
    with pytest.raises(ValueError):
        group_pulses(
            {
                "gps_time": [1.0, 2.0],
                "return_number": [1],
                "number_of_returns": [1],
            }
        )


def test_shipped_las_file_can_be_grouped():
    las = laspy.read("tests/data/autzen.las")
    result = las.pulse_groups()

    assert result.source_point_count == len(las.points)
    assert not result.invalid_points
    assert result.proven_pulse_count > 0
