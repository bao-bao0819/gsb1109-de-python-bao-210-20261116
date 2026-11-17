"""
Tests for laspy.lib.merge_las
"""

import numpy as np
import pytest

import laspy
from laspy import ExtraBytesParams
from laspy.errors import LaspyException
from laspy.lib import MERGE_SOURCE_DIM_NAME, MergeConflict, merge_las, write_then_read_again
from laspy.point import record


def make_las(point_format=3, extra_dims=(), dim_values=None, point_count=0):
    las = laspy.create(point_format=point_format)
    for params in extra_dims:
        las.add_extra_dim(params)
    las.points = record.ScaleAwarePointRecord.zeros(point_count, header=las.header)
    for name, values in (dim_values or {}).items():
        las.points.array[name][:] = values
    return las


def test_merge_temperature_integer_codes_and_scaled_with_no_data():
    """One file stores raw integer codes, the other declares a scale
    and an invalid (no_data) code. The merged column must hold the
    converted values of both, with invalid values recognized on the
    raw codes *before* the scaling is applied.
    """
    las_a = make_las(
        extra_dims=[ExtraBytesParams("temperature", "i2", description="celsius")],
        dim_values={"temperature": [0, 21, 22]},
        point_count=3,
    )
    las_b = make_las(
        extra_dims=[
            ExtraBytesParams(
                "temperature",
                "u1",
                description="celsius",
                scales=[0.5],
                offsets=[-20.0],
                no_data=[255],
            )
        ],
        dim_values={"temperature": [100, 255]},
        point_count=2,
    )

    merged = merge_las([las_a, las_b])

    assert merged.conflicts == []
    assert merged.temperature.dtype == np.float64
    assert list(merged.temperature[:4]) == [0.0, 21.0, 22.0, 30.0]
    assert np.isnan(merged.temperature[4])
    assert [int(s) for s in merged.source_file_id] == [0, 0, 0, 1, 1]
    assert len(merged.points) == 5
    assert merged.header.point_count == 5


def test_merge_no_data_is_recognized_before_scaling():
    """A raw value equal to no_data must be invalid even though the
    scaled value of some other raw code could collide with it.
    """
    las = make_las(
        extra_dims=[
            ExtraBytesParams(
                "temperature", "u1", scales=[2.0], offsets=[0.0], no_data=[10]
            )
        ],
        dim_values={"temperature": [5, 10]},
        point_count=2,
    )
    other = make_las(
        extra_dims=[
            ExtraBytesParams(
                "temperature", "u1", scales=[2.0], offsets=[0.0], no_data=[10]
            )
        ],
        dim_values={"temperature": [10, 5]},
        point_count=2,
    )

    merged = merge_las([las, other])

    assert list(merged.temperature)[:1] == [10.0]
    assert np.isnan(merged.temperature[1])
    assert np.isnan(merged.temperature[2])
    assert merged.temperature[3] == 10.0


def test_merge_missing_dimension_is_nan_and_not_zero():
    """A file that does not have the dimension must produce NaN,
    which cannot be confused with an actual value of exactly 0.0.
    """
    las_with = make_las(
        extra_dims=[ExtraBytesParams("temperature", "i2")],
        dim_values={"temperature": [0, 7]},
        point_count=2,
    )
    las_without = make_las(point_count=2)

    merged = merge_las([las_with, las_without])

    assert merged.conflicts == []
    assert merged.temperature.dtype == np.float64
    assert merged.temperature[0] == 0.0
    assert merged.temperature[1] == 7.0
    assert np.all(np.isnan(np.asarray(merged.temperature)[2:]))


def test_merge_units_mismatch_is_a_conflict():
    las_celsius = make_las(
        extra_dims=[ExtraBytesParams("temperature", "i2", description="celsius")],
        dim_values={"temperature": [1, 2]},
        point_count=2,
    )
    las_fahrenheit = make_las(
        extra_dims=[ExtraBytesParams("temperature", "i2", description="fahrenheit")],
        dim_values={"temperature": [3]},
        point_count=1,
    )

    merged = merge_las([las_celsius, las_fahrenheit])

    assert len(merged.conflicts) == 1
    conflict = merged.conflicts[0]
    assert conflict.dimension == "temperature"
    assert conflict.reason == "units_mismatch"
    assert conflict.sources == (0, 1)
    assert "temperature" not in list(merged.point_format.extra_dimension_names)


def test_merge_dtype_overflow_is_a_conflict():
    """The merged column keeps the type declared by the first file,
    values that do not fit in it are a conflict, not a silent cast.
    """
    las_u1 = make_las(
        extra_dims=[ExtraBytesParams("temperature", "u1")],
        dim_values={"temperature": [1, 2]},
        point_count=2,
    )
    las_i2 = make_las(
        extra_dims=[ExtraBytesParams("temperature", "i2")],
        dim_values={"temperature": [-5]},
        point_count=1,
    )

    merged = merge_las([las_u1, las_i2])

    assert len(merged.conflicts) == 1
    conflict = merged.conflicts[0]
    assert conflict.dimension == "temperature"
    assert conflict.reason == "dtype_overflow"
    assert conflict.sources == (1,)
    assert "temperature" not in list(merged.point_format.extra_dimension_names)


def test_merge_integer_values_that_fit_are_cast():
    las_u1 = make_las(
        extra_dims=[ExtraBytesParams("temperature", "u1")],
        dim_values={"temperature": [1, 2]},
        point_count=2,
    )
    las_i2 = make_las(
        extra_dims=[ExtraBytesParams("temperature", "i2")],
        dim_values={"temperature": [100]},
        point_count=1,
    )

    merged = merge_las([las_u1, las_i2])

    assert merged.conflicts == []
    assert merged.temperature.dtype == np.uint8
    assert [int(t) for t in merged.temperature] == [1, 2, 100]


def test_merge_source_mapping():
    las_a = make_las(point_count=2)
    las_b = make_las(point_count=3)

    merged = merge_las([las_a, las_b])

    assert MERGE_SOURCE_DIM_NAME in list(merged.point_format.extra_dimension_names)
    assert [int(s) for s in merged.source_file_id] == [0, 0, 1, 1, 1]


def test_merge_keeps_standard_dimensions():
    las_a = make_las(point_count=2)
    las_a.x = [1.0, 2.0]
    las_a.classification = [1, 2]
    las_b = make_las(point_count=1)
    las_b.x = [3.0]
    las_b.classification = [5]

    merged = merge_las([las_a, las_b])

    assert list(merged.x) == [1.0, 2.0, 3.0]
    assert list(merged.classification) == [1, 2, 5]


def test_merge_rescales_xyz_when_header_scales_differ():
    las_a = make_las(point_count=1)
    las_a.header.scales = [0.01, 0.01, 0.01]
    las_a.points.scales = las_a.header.scales
    las_a.x = [1.0]
    las_b = make_las(point_count=1)
    las_b.header.scales = [0.1, 0.1, 0.1]
    las_b.points.scales = las_b.header.scales
    las_b.x = [1.0]

    merged = merge_las([las_a, las_b])

    assert np.allclose(list(merged.x), [1.0, 1.0])


def test_merge_different_point_formats_raise():
    las_a = make_las(point_format=3, point_count=1)
    las_b = make_las(point_format=6, point_count=1)

    with pytest.raises(LaspyException):
        merge_las([las_a, las_b])


def test_merge_empty_list_raises():
    with pytest.raises(LaspyException):
        merge_las([])


def test_merge_single_file_is_returned_as_is():
    las = make_las(point_count=1)
    assert merge_las(las) is las
    assert merge_las([las]) is las


def test_merge_result_can_be_written_and_read_back():
    las_a = make_las(
        extra_dims=[
            ExtraBytesParams(
                "temperature",
                "u1",
                scales=[0.5],
                offsets=[-20.0],
                no_data=[255],
            )
        ],
        dim_values={"temperature": [100, 255]},
        point_count=2,
    )
    las_b = make_las(point_count=1)

    merged = merge_las([las_a, las_b])
    merged = write_then_read_again(merged)

    assert merged.temperature[0] == 30.0
    assert np.isnan(merged.temperature[1])
    assert np.isnan(merged.temperature[2])
    assert [int(s) for s in merged.source_file_id] == [0, 0, 1]


def test_merge_conflict_namedtuple_fields():
    conflict = MergeConflict(
        dimension="temperature",
        reason="units_mismatch",
        sources=(0, 1),
        message="msg",
    )
    assert conflict.dimension == "temperature"
    assert conflict.reason == "units_mismatch"
    assert conflict.sources == (0, 1)
    assert conflict.message == "msg"
