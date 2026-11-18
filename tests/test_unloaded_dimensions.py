"""Tests for dimensions that were not decompressed when reading a LAZ file.

When a LAS 1.4 (point format 6 to 10) LAZ file is read with a
restrictive :class:`laspy.DecompressionSelection`, the fields that are
not decompressed are filled with placeholder bytes. laspy must not
present those placeholders as if they were real zero values.
"""

import io
from pathlib import Path

import numpy as np
import pytest

import laspy
from laspy.point.dims import DimensionNotLoadedWarning
from laspy.point.record import ScaleAwarePointRecord

POINT_FORMAT_ID = 6

# Dimensions that a ``DecompressionSelection.base()`` (i.e X, Y, return
# number, number of returns and scanner channel) would load.
BASE_DIMENSIONS = [
    "X",
    "Y",
    "x",
    "y",
    "return_number",
    "number_of_returns",
    "scanner_channel",
]


def make_record(loaded_dimensions=BASE_DIMENSIONS, point_count=4):
    point_format = laspy.PointFormat(POINT_FORMAT_ID)
    array = np.zeros(point_count, point_format.dtype())
    array["X"] = np.arange(point_count, dtype=array["X"].dtype) + 100
    array["Y"] = np.arange(point_count, dtype=array["Y"].dtype) + 200
    array["Z"] = np.arange(point_count, dtype=array["Z"].dtype) + 300
    array["bit_fields"] = np.full(
        point_count, 0b0001_0001, dtype=array["bit_fields"].dtype
    )
    # Placeholder values, as written by the LAZ decompressor for
    # skipped fields.
    array["intensity"] = np.zeros(point_count, dtype=array["intensity"].dtype)
    array["classification"] = np.zeros(
        point_count, dtype=array["classification"].dtype
    )
    return ScaleAwarePointRecord(
        array,
        point_format,
        scales=[1.0, 1.0, 1.0],
        offsets=[0.0, 0.0, 0.0],
        loaded_dimensions=loaded_dimensions,
    )


def make_las(loaded_dimensions=BASE_DIMENSIONS, point_count=4):
    header = laspy.LasHeader(point_format=POINT_FORMAT_ID, version="1.4")
    header.scales = np.array([1.0, 1.0, 1.0])
    header.offsets = np.array([0.0, 0.0, 0.0])
    return laspy.LasData(
        header=header,
        points=make_record(loaded_dimensions, point_count),
    )


def test_fully_loaded_record_has_all_dimensions_loaded():
    record = ScaleAwarePointRecord.zeros(
        3,
        point_format=laspy.PointFormat(POINT_FORMAT_ID),
        scales=[1.0] * 3,
        offsets=[0.0] * 3,
    )

    assert set(record.loaded_dimensions) == set(record.point_format.dimension_names)
    assert record.not_loaded_dimensions == ()
    assert record.is_dimension_loaded("intensity")
    assert record.dimensions_are_available(["X", "intensity"])


def test_unloaded_dimensions_are_reported():
    record = make_record()

    assert "intensity" in record.not_loaded_dimensions
    assert "intensity" not in record.loaded_dimensions
    assert not record.is_dimension_loaded("intensity")
    assert not record.dimensions_are_available(["X", "intensity"])
    # Loaded dimensions are still reported as such
    assert record.is_dimension_loaded("X")
    assert record.is_dimension_loaded("return_number")


def test_las_data_reports_unloaded_dimensions():
    las = make_las()

    assert "intensity" in las.not_loaded_dimensions
    assert not las.is_dimension_loaded("intensity")
    assert not las.dimensions_are_available(["intensity"])
    assert las.dimensions_are_available(["X", "Y"])


def test_require_dimensions_raises_for_unloaded_dimension():
    las = make_las()

    with pytest.raises(laspy.DimensionNotLoadedError) as exc_info:
        las.require_dimensions(["X", "intensity"])

    assert exc_info.value.missing_dimensions == ("intensity",)


def test_require_dimensions_passes_for_loaded_dimensions():
    las = make_las()

    # Must not raise
    las.require_dimensions(["X", "Y", "return_number", "scanner_channel"])


def test_unloaded_intensity_does_not_take_part_in_mean():
    las = make_las()
    # The real intensity is unknown, the array only holds placeholders
    assert np.all(las.points.array["intensity"] == 0)

    assert np.isnan(np.mean(las.intensity))
    assert np.isnan(np.sum(las.intensity))
    assert np.isnan(np.min(las.intensity))
    assert np.isnan(np.max(las.intensity))


def test_loaded_dimension_mean_is_regular_mean():
    las = make_las()
    las.points.array["intensity"] = np.array([10, 20, 30, 40], dtype=np.uint16)
    las.points.mark_dimensions_as_loaded(["intensity"])

    assert las.is_dimension_loaded("intensity")
    assert np.mean(las.intensity) == 25.0


def test_unloaded_dimension_placeholders_are_still_comparable():
    # Existing code compares unloaded fields with their placeholder
    # values. This must keep working, without pretending the values
    # are real.
    las = make_las()

    assert np.all(las.intensity == 0)
    assert np.any(las.intensity != 1)


def test_writing_unloaded_placeholders_is_refused():
    las = make_las()

    with pytest.raises(laspy.DimensionNotLoadedError) as exc_info:
        las.write(io.BytesIO())

    message = str(exc_info.value)
    assert "intensity" in message
    assert "placeholder" in message


def test_writing_after_filling_unloaded_dimension_is_allowed():
    las = make_las()

    # Every unloaded dimension must be filled with real values before
    # the file can be written.
    for dimension_name in las.not_loaded_dimensions:
        las[dimension_name][:] = 1

    assert las.not_loaded_dimensions == ()

    out = io.BytesIO()
    las.write(out)
    out.seek(0)
    written = laspy.read(out)
    assert np.all(written.intensity == 1)


def test_writing_through_view_marks_dimension_loaded():
    record = make_record()
    assert not record.is_dimension_loaded("intensity")

    record["intensity"][:] = np.array([1, 2, 3, 4], dtype=np.uint16)

    assert record.is_dimension_loaded("intensity")
    assert np.mean(record.intensity) == 2.5


def test_projection_of_a_fully_loaded_record_keeps_loaded_dimensions():
    las = laspy.read("tests/data/simple.las")

    projection = las[["X", "Y", "intensity"]]

    assert set(projection.loaded_dimensions) == {"X", "Y", "intensity"}
    assert projection.not_loaded_dimensions == ()


def test_projection_does_not_invent_loaded_dimensions():
    las = make_las()

    projection = las[["X", "Y", "intensity"]]

    # `intensity` is projected but was never loaded
    assert "intensity" in projection.not_loaded_dimensions
    assert set(projection.loaded_dimensions) == {"X", "Y"}


def test_slicing_keeps_unloaded_state():
    las = make_las()

    subset = las[0:2]

    assert "intensity" in subset.not_loaded_dimensions
    assert subset.is_dimension_loaded("X")


def test_copy_keeps_unloaded_state():
    record = make_record()

    copied = record.copy()

    assert copied.not_loaded_dimensions == record.not_loaded_dimensions


def test_xyz_masks_unloaded_coordinates():
    las = make_las()

    xyz = las.xyz

    assert isinstance(xyz, np.ma.MaskedArray)
    # X and Y were loaded, Z was not
    assert not np.any(xyz.mask[:, 0])
    assert not np.any(xyz.mask[:, 1])
    assert np.all(xyz.mask[:, 2])


def test_converting_to_numpy_warns_about_placeholders():
    record = make_record()

    with pytest.warns(DimensionNotLoadedWarning):
        array = np.array(record.intensity)

    # The raw placeholder values are still accessible
    assert np.all(array == 0)


def test_header_update_ignores_unloaded_coordinates():
    las = make_las()

    las.update_header()

    # X and Y were loaded and have real values, Z was not loaded and
    # must not contribute placeholder values to the header bounds.
    assert las.header.x_min == 100.0
    assert las.header.x_max == 103.0
    assert las.header.y_min == 200.0
    assert las.header.y_max == 203.0
    assert las.header.z_max == np.finfo(np.float64).min
    assert las.header.z_min == np.finfo(np.float64).max


def test_header_update_ignores_unloaded_gps_time():
    las = make_las()

    las.update_header()

    # gps_time is unloaded so it must not be used in the header stats,
    # the sentinel 'unset' values are kept
    f64info = np.finfo(np.float64)
    assert las.header.max_gps_time == f64info.min
    assert las.header.min_gps_time == f64info.max


def test_decompression_selection_maps_to_dimensions():
    selection = laspy.DecompressionSelection.base().decompress_intensity()
    point_format = laspy.PointFormat(7)

    loaded = selection.loaded_dimension_names(point_format)

    assert "X" in loaded
    assert "x" in loaded
    assert "Y" in loaded
    assert "return_number" in loaded
    assert "number_of_returns" in loaded
    assert "intensity" in loaded
    assert "classification" not in loaded
    assert "gps_time" not in loaded
    assert "red" not in loaded


def test_decompression_selection_all_loads_everything():
    point_format = laspy.PointFormat(7)

    loaded = laspy.DecompressionSelection.all().loaded_dimension_names(point_format)

    assert set(point_format.dimension_names).issubset(loaded)


def test_decompression_selection_flags_loads_classification_flags():
    selection = (
        laspy.DecompressionSelection.base()
        .decompress_flags()
        .decompress_classification()
    )
    point_format = laspy.PointFormat(6)

    loaded = selection.loaded_dimension_names(point_format)

    assert "classification" in loaded
    assert "synthetic" in loaded
    assert "key_point" in loaded
    assert "withheld" in loaded
    assert "overlap" in loaded


def make_reader(decompression_selection):
    with open("tests/data/simple.las", "rb") as f:
        return laspy.LasReader(
            io.BytesIO(f.read()),
            closefd=False,
            decompression_selection=decompression_selection,
        )


def test_reader_only_tracks_dimensions_for_selectively_decompressed_laz():
    reader = make_reader(
        laspy.DecompressionSelection.base().decompress_intensity()
    )
    reader.header = laspy.LasHeader(point_format=POINT_FORMAT_ID, version="1.4")
    reader.header.are_points_compressed = True

    loaded = reader._loaded_dimensions_for_read()

    assert loaded is not None
    assert "intensity" in loaded
    assert "classification" not in loaded


def test_reader_does_not_track_dimensions_for_uncompressed_files():
    reader = make_reader(laspy.DecompressionSelection.base())
    reader.header = laspy.LasHeader(point_format=POINT_FORMAT_ID, version="1.4")
    reader.header.are_points_compressed = False

    assert reader._loaded_dimensions_for_read() is None


def test_full_read_and_projection_report_consistent_loaded_dimensions():
    las = laspy.read("tests/data/simple.las")

    fully_loaded = set(las.loaded_dimensions)
    projection = las[["X", "Y", "Z", "intensity"]]

    # Every dimension that the projection reports as loaded must be
    # reported loaded by a full read of the same file, and their
    # values must be identical.
    for dimension_name in projection.loaded_dimensions:
        assert dimension_name in fully_loaded
        assert np.all(
            np.asarray(projection[dimension_name])
            == np.asarray(las[dimension_name])
        )


def test_reading_in_chunks_keeps_loaded_dimensions_consistent():
    reader = make_reader(laspy.DecompressionSelection.base())

    first = reader._read_points(10)
    second = reader._read_points(10)

    assert first.loaded_dimensions == second.loaded_dimensions
    assert set(first.loaded_dimensions) == set(
        first.point_format.dimension_names
    )


def test_extra_dimensions_are_loaded_when_selection_says_so():
    selection = laspy.DecompressionSelection.base().decompress_intensity()
    point_format = laspy.PointFormat(6)
    point_format.add_extra_dimension(laspy.ExtraBytesParams("my_extra", "int32"))

    loaded = selection.loaded_dimension_names(point_format)

    assert "my_extra" not in loaded

    selection = laspy.DecompressionSelection.base().decompress_all_extra_bytes()
    loaded = selection.loaded_dimension_names(point_format)

    assert "my_extra" in loaded


LAZ_1_4_FILE = Path(__file__).parent / "data" / "1_4_w_evlr.laz"

skip_if_no_laz_backend = pytest.mark.skipif(
    len(laspy.LazBackend.detect_available()) == 0,
    reason="No Laz backend installed",
)


@skip_if_no_laz_backend
def test_selective_laz_read_tracks_loaded_dimensions():
    selection = laspy.DecompressionSelection.base().decompress_intensity()

    fully_loaded = laspy.read(LAZ_1_4_FILE)
    partially_loaded = laspy.read(
        LAZ_1_4_FILE, decompression_selection=selection
    )

    expected_loaded = selection.loaded_dimension_names(fully_loaded.point_format)
    assert set(partially_loaded.loaded_dimensions) == set(expected_loaded)

    for dimension_name in partially_loaded.loaded_dimensions:
        assert np.all(
            np.asarray(fully_loaded[dimension_name])
            == np.asarray(partially_loaded[dimension_name])
        ), dimension_name

    # Dimensions that were not decompressed are not reported as loaded
    assert "classification" in partially_loaded.not_loaded_dimensions
    assert "gps_time" in partially_loaded.not_loaded_dimensions
    assert "intensity" in partially_loaded.loaded_dimensions


@skip_if_no_laz_backend
def test_selective_laz_read_does_not_use_unloaded_intensity_in_mean():
    selection = laspy.DecompressionSelection.base()

    partially_loaded = laspy.read(
        LAZ_1_4_FILE, decompression_selection=selection
    )

    assert "intensity" in partially_loaded.not_loaded_dimensions
    assert np.isnan(np.mean(partially_loaded.intensity))

    with pytest.raises(laspy.DimensionNotLoadedError):
        partially_loaded.require_dimensions(["intensity"])


@skip_if_no_laz_backend
def test_selective_laz_read_cannot_be_written_as_is():
    selection = laspy.DecompressionSelection.base()

    partially_loaded = laspy.read(
        LAZ_1_4_FILE, decompression_selection=selection
    )

    with pytest.raises(laspy.DimensionNotLoadedError):
        partially_loaded.write(io.BytesIO())
