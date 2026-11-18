import numpy as np
import pytest

import laspy
from laspy.migration import (
    Degradation,
    DegradationKind,
    MigrationError,
    MigrationPolicy,
    plan_batch_migration,
)


def make_las(point_format_id, file_version, point_count=10):
    las = laspy.create(point_format=point_format_id, file_version=file_version)
    las.x = np.arange(point_count, dtype=np.float64)
    las.y = np.arange(point_count, dtype=np.float64) + 1.0
    las.z = np.arange(point_count, dtype=np.float64) + 2.0
    return las


def make_fmt6_las(classifications, gps_times=None):
    las = make_las(6, "1.4", len(classifications))
    las.classification = np.array(classifications, dtype=np.uint8)
    if gps_times is not None:
        las.gps_time = np.array(gps_times, dtype=np.float64)
    return las


def test_default_policy_quarantines_unrepresentable_points():
    las = make_fmt6_las([1, 2, 31, 64, 200], gps_times=np.arange(5.0))

    plan = plan_batch_migration([las], target_version="1.2")

    # gps_time must be preserved (default policy) -> only formats 1 and 3 qualify
    assert plan.target_point_format_id in (1, 3)
    assert plan.total_loss == 0.0
    assert plan.total_quarantined == 2  # classes 64 and 200 do not fit in 5 bits

    result = plan.execute()
    assert result.readback_verified
    file_result = result.files[0]
    assert len(file_result.migrated.points) == 3
    assert list(file_result.migrated.classification) == [1, 2, 31]
    # isolated points keep the original format and original codes
    assert file_result.quarantined is not None
    assert file_result.quarantined.point_format.id == 6
    assert sorted(file_result.quarantined.classification) == [64, 200]
    assert list(file_result.quarantined_indices) == [3, 4]
    # no point was lost
    assert len(file_result.migrated.points) + len(file_result.quarantined.points) == 5


def test_clamp_is_chosen_when_cheaper_than_quarantine():
    las = make_fmt6_las([1, 64, 200], gps_times=np.arange(3.0))
    policy = MigrationPolicy(
        field_policies={"classification": (Degradation.clamp(cost=1.0),)},
        quarantine_cost=10.0,
    )

    plan = plan_batch_migration([las], target_version="1.2", policy=policy)

    decision = {d.field: d for d in plan.decisions}["classification"]
    assert decision.degradation.kind == DegradationKind.CLAMP
    assert decision.affected_points == 2
    assert decision.quarantined_points == 0
    assert decision.loss == 2.0
    assert decision.original_codes == (64, 200)
    assert decision.stored_codes == (31, 31)

    result = plan.execute()
    assert result.readback_verified
    file_result = result.files[0]
    assert file_result.quarantined is None
    # out-of-range codes are saturated, other points are untouched
    assert list(file_result.migrated.classification) == [1, 31, 31]
    report = file_result.field_reports[0]
    assert report.field == "classification"
    assert report.loss == 2.0
    assert report.original_codes == (64, 200)
    assert report.stored_codes == (31, 31)
    assert report.readback_verified


def test_map_degradation_remaps_codes_and_reports_them():
    las = make_fmt6_las([2, 64, 200, 7], gps_times=np.arange(4.0))
    policy = MigrationPolicy(
        field_policies={
            "classification": (Degradation.map(cost=0.5, mapping={64: 2, 200: 5}),)
        },
        quarantine_cost=10.0,
    )

    plan = plan_batch_migration([las], target_version="1.2", policy=policy)

    decision = {d.field: d for d in plan.decisions}["classification"]
    assert decision.degradation.kind == DegradationKind.MAP
    assert decision.original_codes == (64, 200)
    assert decision.stored_codes == (2, 5)
    assert decision.loss == 1.0

    result = plan.execute()
    assert result.readback_verified
    assert list(result.files[0].migrated.classification) == [2, 2, 5, 7]


def test_map_with_unmapped_codes_quarantines_them():
    las = make_fmt6_las([2, 64, 200], gps_times=np.arange(3.0))
    policy = MigrationPolicy(
        field_policies={
            "classification": (Degradation.map(cost=0.5, mapping={64: 2}),)
        },
        quarantine_cost=5.0,
    )

    result = plan_batch_migration([las], target_version="1.2", policy=policy).execute()

    file_result = result.files[0]
    assert list(file_result.migrated.classification) == [2, 2]
    assert list(file_result.quarantined.classification) == [200]
    assert result.readback_verified


def test_quarantine_budget_can_force_a_degradation():
    las = make_fmt6_las([1, 64, 200], gps_times=np.arange(3.0))
    policy = MigrationPolicy(
        field_policies={"classification": (Degradation.clamp(cost=1.0),)},
        max_quarantined_points=1,
    )

    plan = plan_batch_migration([las], target_version="1.2", policy=policy)

    # preserve would quarantine 2 points, over the budget: clamp is selected
    decision = {d.field: d for d in plan.decisions}["classification"]
    assert decision.degradation.kind == DegradationKind.CLAMP
    assert plan.total_quarantined == 0


def test_infeasible_when_budget_cannot_be_met():
    las = make_fmt6_las([64, 200], gps_times=np.arange(2.0))
    policy = MigrationPolicy(max_quarantined_points=1)

    with pytest.raises(MigrationError):
        plan_batch_migration([las], target_version="1.2", policy=policy)


def test_infeasible_when_a_field_cannot_be_preserved_nor_dropped():
    las = make_las(3, "1.2", 4)
    las.red = np.arange(4, dtype=np.uint16) + 1

    # version 1.1 has no point format with color, and dropping is not allowed
    with pytest.raises(MigrationError):
        plan_batch_migration([las], target_version="1.1")


def test_preserve_only_policy_selects_formats_keeping_the_data():
    las = make_las(3, "1.2", 4)
    las.red = np.arange(4, dtype=np.uint16) + 1
    las.green = np.arange(4, dtype=np.uint16) + 1
    las.blue = np.arange(4, dtype=np.uint16) + 1
    las.gps_time = np.arange(4, dtype=np.float64) + 1.0

    plan = plan_batch_migration([las], target_version="1.2")

    # gps_time and color must be preserved -> only format 3 qualifies
    assert plan.target_point_format_id == 3
    assert plan.total_loss == 0.0


def test_drop_is_used_when_field_cannot_be_preserved():
    las = make_las(3, "1.2", 4)
    las.red = np.arange(4, dtype=np.uint16) + 1
    las.green = np.zeros(4, dtype=np.uint16)
    las.blue = np.zeros(4, dtype=np.uint16)
    las.gps_time = np.arange(4, dtype=np.float64) + 1.0

    policy = MigrationPolicy(
        field_policies={
            "red": (Degradation.drop(cost=2.0),),
            "green": (Degradation.drop(cost=2.0),),
            "blue": (Degradation.drop(cost=2.0),),
        }
    )
    plan = plan_batch_migration([las], target_version="1.1", policy=policy)

    # only red carries non-default values: loss = 2.0 * 4 points
    assert plan.target_point_format_id == 1
    assert plan.total_loss == 8.0

    result = plan.execute()
    assert result.readback_verified
    assert "red" not in result.files[0].migrated.point_format.dimension_names
    report = {r.field: r for r in result.files[0].field_reports}["red"]
    assert report.degradation == DegradationKind.DROP
    assert report.affected_points == 4
    assert report.loss == 8.0


def test_batch_uses_one_consistent_interpretation_per_field():
    las_a = make_fmt6_las([1, 64], gps_times=np.arange(2.0))
    las_b = make_fmt6_las([200, 3], gps_times=np.arange(2.0))
    policy = MigrationPolicy(
        field_policies={"classification": (Degradation.clamp(cost=1.0),)}
    )
    policy = policy._replace(quarantine_cost=10.0)

    plan = plan_batch_migration([las_a, las_b], target_version="1.2", policy=policy)

    decisions = [d for d in plan.decisions if d.field == "classification"]
    assert len(decisions) == 1
    assert decisions[0].degradation.kind == DegradationKind.CLAMP
    assert decisions[0].affected_points == 2

    result = plan.execute()
    assert result.readback_verified
    assert list(result.files[0].migrated.classification) == [1, 31]
    assert list(result.files[1].migrated.classification) == [31, 3]
    # both files were migrated to the same target point format
    assert (
        result.files[0].migrated.point_format.id
        == result.files[1].migrated.point_format.id
        == plan.target_point_format_id
    )


def test_extra_dimensions_are_preserved_consistently():
    las_a = make_las(1, "1.2", 3)
    las_a.add_extra_dim(laspy.ExtraBytesParams(name="amplitude", type="uint16"))
    las_a.amplitude = np.array([1, 2, 3], dtype=np.uint16)
    las_b = make_las(3, "1.2", 2)
    las_b.add_extra_dim(laspy.ExtraBytesParams(name="amplitude", type="uint16"))
    las_b.amplitude = np.array([4, 5], dtype=np.uint16)

    plan = plan_batch_migration([las_a, las_b], target_version="1.2")

    decision = {d.field: d for d in plan.decisions}["amplitude"]
    assert decision.degradation.kind == DegradationKind.PRESERVE

    result = plan.execute()
    assert result.readback_verified
    assert list(result.files[0].migrated.amplitude) == [1, 2, 3]
    assert list(result.files[1].migrated.amplitude) == [4, 5]
    # same target format (including extra dims) for the whole batch
    assert (
        result.files[0].migrated.point_format
        == result.files[1].migrated.point_format
    )


def test_conflicting_extra_dimensions_cannot_be_preserved():
    las_a = make_las(1, "1.2", 3)
    las_a.add_extra_dim(laspy.ExtraBytesParams(name="echo", type="uint16"))
    las_b = make_las(1, "1.2", 3)
    las_b.add_extra_dim(laspy.ExtraBytesParams(name="echo", type="float32"))

    # preserve-only policy: the conflicting definitions make the plan infeasible
    with pytest.raises(MigrationError):
        plan_batch_migration([las_a, las_b], target_version="1.2")

    # allowing the field to be dropped solves the conflict
    policy = MigrationPolicy(
        field_policies={"echo": (Degradation.drop(cost=1.0),)}
    )
    plan = plan_batch_migration([las_a, las_b], target_version="1.2", policy=policy)
    decision = {d.field: d for d in plan.decisions}["echo"]
    assert decision.degradation.kind == DegradationKind.DROP
    result = plan.execute()
    assert "echo" not in result.files[0].migrated.point_format.dimension_names


def test_scan_angle_conversion_and_quarantine():
    las = make_las(6, "1.4", 4)
    # 1500 * 0.006 = 9.0 degrees exactly, 1503 is not representable in whole
    # degrees, 30000 * 0.006 = 180 degrees does not fit in an int8 rank
    las.scan_angle = np.array([0, 1500, 1503, 30000], dtype=np.int16)
    las.gps_time = np.arange(4.0)

    plan = plan_batch_migration([las], target_version="1.2")
    assert plan.total_quarantined == 2

    result = plan.execute()
    assert result.readback_verified
    file_result = result.files[0]
    assert list(file_result.migrated.scan_angle_rank) == [0, 9]
    assert list(file_result.quarantined.scan_angle) == [1503, 30000]


def test_scan_angle_clamp_rounds_and_saturates():
    las = make_las(6, "1.4", 3)
    las.scan_angle = np.array([1503, 30000, -30000], dtype=np.int16)
    las.gps_time = np.arange(3.0)
    policy = MigrationPolicy(
        field_policies={"scan_angle": (Degradation.clamp(cost=1.0),)},
        quarantine_cost=10.0,
    )

    plan = plan_batch_migration([las], target_version="1.2", policy=policy)
    decision = {d.field: d for d in plan.decisions}["scan_angle"]
    assert decision.degradation.kind == DegradationKind.CLAMP
    assert decision.affected_points == 3

    result = plan.execute()
    assert result.readback_verified
    assert list(result.files[0].migrated.scan_angle_rank) == [9, 127, -128]


def test_scan_angle_rank_is_converted_when_upgrading():
    las = make_las(1, "1.2", 3)
    las.scan_angle_rank = np.array([0, 9, -9], dtype=np.int8)

    # version 1.5 only accepts the new-family point formats
    plan = plan_batch_migration([las], target_version="1.5")
    assert plan.target_point_format_id in (6, 7, 8, 9, 10)

    result = plan.execute()
    assert result.readback_verified
    migrated = result.files[0].migrated
    # 9 degrees -> 1500 units of 0.006 degrees
    assert list(migrated.scan_angle) == [0, 1500, -1500]


def test_shared_flags_are_carried_between_families():
    las = make_las(6, "1.4", 3)
    las.synthetic = np.array([1, 0, 1], dtype=np.uint8)
    las.withheld = np.array([0, 1, 0], dtype=np.uint8)
    las.gps_time = np.arange(3.0)

    result = plan_batch_migration([las], target_version="1.2").execute()

    migrated = result.files[0].migrated
    assert list(migrated.synthetic) == [1, 0, 1]
    assert list(migrated.withheld) == [0, 1, 0]


def test_new_family_flags_must_be_dropped_explicitly():
    las = make_las(6, "1.4", 3)
    las.overlap = np.array([1, 0, 0], dtype=np.uint8)
    las.gps_time = np.arange(3.0)

    # overlap does not exist in formats < 6 and the default policy
    # only allows preserving it
    with pytest.raises(MigrationError):
        plan_batch_migration([las], target_version="1.2")

    policy = MigrationPolicy(
        field_policies={"overlap": (Degradation.drop(cost=1.0),)}
    )
    plan = plan_batch_migration([las], target_version="1.2", policy=policy)
    decision = {d.field: d for d in plan.decisions}["overlap"]
    assert decision.degradation.kind == DegradationKind.DROP
    assert decision.affected_points == 1
    assert decision.loss == 1.0

    result = plan.execute()
    assert result.readback_verified
    report = {r.field: r for r in result.files[0].field_reports}["overlap"]
    assert report.degradation == DegradationKind.DROP
    assert report.affected_points == 1
