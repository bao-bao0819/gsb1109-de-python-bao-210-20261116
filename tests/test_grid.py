"""Tests for the global grid indexing of point clouds.

The invariants enforced here are:

* points on grid edges are assigned to exactly one cell (half-open rule),
* the assignment only depends on the global grid definition, never on the
  bounding boxes stored in file headers,
* changing the cell size, the reading chunk size or the order of the input
  files does not change which cell a point belongs to nor the cell counts,
* every indexed point keeps its source file and its index inside that file.
"""

import struct

import numpy as np
import pytest

import laspy
from laspy.grid import (
    GridDefinition,
    GridIndexBuilder,
    GridOrigin,
    build_point_grid_index,
)


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------


def _write_las(path, xyz):
    # Use a fine scale so that values close to grid lines (e.g. -0.001,
    # 1.999) survive the LAS integer quantization unchanged.
    header = laspy.LasHeader()
    header.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(header)
    las.x = np.asarray(xyz[:, 0], dtype=np.float64)
    las.y = np.asarray(xyz[:, 1], dtype=np.float64)
    las.z = np.asarray(xyz[:, 2], dtype=np.float64)
    las.write(path)
    return las


# Points deliberately placed on grid lines / corners so that a naive
# per-file snapping of the local bbox would double count them.
POINTS_A = np.array(
    [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.5, 0.5, 0.0),
        (-1.0, 0.0, 0.0),
    ]
)

POINTS_B = np.array(
    [
        (2.0, 2.0, 0.0),
        (1.999, 2.0, 0.0),
        (-0.001, 0.5, 0.0),
        # Same coordinates as the third point of file A: it must still be
        # tracked separately through its own source file / point index.
        (1.0, 1.0, 0.0),
    ]
)


@pytest.fixture()
def two_files(tmp_path):
    path_a = tmp_path / "a.las"
    path_b = tmp_path / "b.las"
    _write_las(path_a, POINTS_A)
    _write_las(path_b, POINTS_B)
    return path_a, path_b


def _expected_cells(xyz, cell_size, origin):
    """Reference implementation of the deterministic assignment rule."""
    origin_arr = np.asarray(origin, dtype=np.float64)
    size = np.asarray(cell_size, dtype=np.float64)
    return np.floor((xyz - origin_arr) / size).astype(np.int64)


# ---------------------------------------------------------------------------
# Grid definition and the boundary point rule
# ---------------------------------------------------------------------------


def test_grid_definition_validates_cell_size():
    with pytest.raises(ValueError):
        GridDefinition(cell_size=0.0, origin=GridOrigin(0.0, 0.0))
    with pytest.raises(ValueError):
        GridDefinition(cell_size=-1.0, origin=GridOrigin(0.0, 0.0))
    with pytest.raises(ValueError):
        GridDefinition(
            cell_size=(1.0, 1.0, 1.0), origin=GridOrigin(0.0, 0.0)
        )


def test_boundary_points_go_to_a_single_cell():
    grid = GridDefinition(cell_size=1.0, origin=GridOrigin(0.0, 0.0))
    xyz = np.array(
        [
            (0.0, 0.0),  # corner, belongs to (0, 0)
            (1.0, 0.0),  # edge, belongs to (1, 0), not (0, 0)
            (1.0, 1.0),  # corner, belongs to (1, 1)
            (-1.0, -1.0),  # negative edge, belongs to (-1, -1)
            (0.999999, 0.0),  # just before the edge
        ]
    )

    cells = [tuple(row) for row in grid.cell_indices(xyz)]

    assert cells == [(0, 0), (1, 0), (1, 1), (-1, -1), (0, 0)]


def test_assignment_matches_floor_rule_for_any_origin_and_size():
    rng = np.random.default_rng(42)
    xyz = rng.normal(size=(500, 2)) * 50.0
    # Force some points exactly on grid lines.
    xyz[:20] = np.round(xyz[:20])

    for cell_size in (0.25, 1.0, 3.5, 10.0):
        for origin in ((0.0, 0.0), (-7.5, 3.25)):
            grid = GridDefinition(
                cell_size=cell_size, origin=GridOrigin(origin[0], origin[1])
            )
            np.testing.assert_array_equal(
                grid.cell_indices(xyz),
                _expected_cells(xyz, cell_size, origin),
            )


def test_3d_grid_uses_z_origin():
    grid = GridDefinition(
        cell_size=1.0, origin=GridOrigin(0.0, 0.0, 0.0)
    )
    xyz = np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, -1.0)])

    cells = [tuple(row) for row in grid.cell_indices(xyz)]

    assert cells == [(0, 0, 0), (0, 0, 1), (1, 1, -1)]


def test_non_finite_coordinates_are_rejected():
    grid = GridDefinition(cell_size=1.0, origin=GridOrigin(0.0, 0.0))
    with pytest.raises(ValueError):
        grid.cell_indices(np.array([(0.0, 0.0), (np.nan, 1.0)]))
    with pytest.raises(ValueError):
        grid.cell_indices(np.array([(np.inf, 0.0)]))


# ---------------------------------------------------------------------------
# Provenance: source file and point index
# ---------------------------------------------------------------------------


def test_each_point_keeps_source_file_and_index(two_files):
    path_a, path_b = two_files
    index = build_point_grid_index(
        [path_a, path_b], cell_size=1.0, origin=GridOrigin(0.0, 0.0)
    )

    expected = {}
    for path, points in ((path_a, POINTS_A), (path_b, POINTS_B)):
        for point_index, row in enumerate(points):
            cell = tuple(int(v) for v in np.floor(row[:2]))
            expected.setdefault(cell, []).append((str(path), point_index))

    assert index.cell_counts() == {
        cell: len(entries) for cell, entries in expected.items()
    }
    for cell, entries in expected.items():
        assert sorted(index.points_of(cell)) == sorted(entries)

    # Every point is indexed exactly once: no point lost, none counted twice.
    assert index.total_points() == len(POINTS_A) + len(POINTS_B)
    assert sum(index.cell_counts().values()) == index.total_points()


def test_duplicate_coordinates_in_different_files_are_both_indexed(two_files):
    path_a, path_b = two_files
    index = build_point_grid_index(
        [path_a, path_b], cell_size=1.0, origin=GridOrigin(0.0, 0.0)
    )

    # POINTS_A[2] and POINTS_B[3] are both (1.0, 1.0) but come from two files.
    assert (str(path_a), 2) in index.points_of((1, 1))
    assert (str(path_b), 3) in index.points_of((1, 1))
    assert index.count((1, 1)) == 2


# ---------------------------------------------------------------------------
# Invariance: file order, chunk size, cell size and header bbox
# ---------------------------------------------------------------------------


def _counts_for(paths, cell_size, points_per_chunk=None):
    index = build_point_grid_index(
        paths,
        cell_size=cell_size,
        origin=GridOrigin(0.0, 0.0),
        points_per_chunk=points_per_chunk,
    )
    return {
        cell: sorted(entries) for cell, entries in index.cells.items()
    }


def test_file_order_does_not_change_assignment(two_files):
    path_a, path_b = two_files

    forward = _counts_for([path_a, path_b], cell_size=1.0)
    reversed_order = _counts_for([path_b, path_a], cell_size=1.0)

    assert reversed_order == forward


def test_chunked_reading_matches_whole_file_reading(two_files):
    path_a, path_b = two_files

    whole = _counts_for([path_a, path_b], cell_size=1.0)
    chunked = _counts_for(
        [path_a, path_b], cell_size=1.0, points_per_chunk=2
    )

    assert chunked == whole


def test_cell_size_change_keeps_consistent_assignment(two_files):
    path_a, path_b = two_files
    all_points = np.vstack([POINTS_A, POINTS_B])

    for cell_size in (0.5, 1.0, 2.0, 7.5):
        index = build_point_grid_index(
            [path_a, path_b], cell_size=cell_size, origin=GridOrigin(0.0, 0.0)
        )
        reference = _expected_cells(all_points[:, :2], cell_size, (0.0, 0.0))

        rebuilt = {}
        for row in reference:
            rebuilt.setdefault(tuple(int(v) for v in row), 0)
            rebuilt[tuple(int(v) for v in row)] += 1

        assert index.cell_counts() == rebuilt
        assert index.total_points() == len(all_points)


def test_header_bounding_box_is_only_a_reading_hint(tmp_path, two_files):
    path_a, _ = two_files
    before = _counts_for([path_a], cell_size=1.0)

    # Corrupt the bbox stored in the LAS 1.2 header (max_x at byte 179,
    # min_x at byte 187, each a little-endian float64).
    with open(path_a, "r+b") as stream:
        stream.seek(179)
        stream.write(struct.pack("<d", 9999.0))
        stream.seek(187)
        stream.write(struct.pack("<d", -9999.0))

    with laspy.open(path_a) as reader:
        # Sanity check: the header really advertises the wrong bbox.
        assert reader.header.x_max == 9999.0
        assert reader.header.x_min == -9999.0

    after = _counts_for([path_a], cell_size=1.0)
    assert after == before


def test_builder_accepts_lasdata_and_reader_directly(two_files):
    path_a, path_b = two_files
    grid = GridDefinition(cell_size=1.0, origin=GridOrigin(0.0, 0.0))

    builder = GridIndexBuilder(grid)
    builder.add_las(laspy.read(path_a), source_file=str(path_a))
    with laspy.open(path_b) as reader:
        builder.add_reader(reader, points_per_chunk=3, source_file=str(path_b))

    direct = builder.index
    function_result = build_point_grid_index(
        [path_a, path_b], cell_size=1.0, origin=GridOrigin(0.0, 0.0)
    )

    assert direct.cell_counts() == function_result.cell_counts()
    assert sorted(
        (cell, tuple(sorted(entries)))
        for cell, entries in direct.cells.items()
    ) == sorted(
        (cell, tuple(sorted(entries)))
        for cell, entries in function_result.cells.items()
    )


def test_public_api_is_exported():
    assert laspy.GridOrigin is GridOrigin
    assert laspy.GridDefinition is GridDefinition
    assert laspy.PointGridIndex is laspy.grid.PointGridIndex
    assert laspy.GridIndexBuilder is GridIndexBuilder
    assert laspy.build_point_grid_index is build_point_grid_index


def test_index_on_shipped_las_file():
    las = laspy.read("tests/data/simple.las")
    index = build_point_grid_index(
        ["tests/data/simple.las"],
        cell_size=10.0,
        origin=(las.header.mins[0], las.header.mins[1]),
    )

    assert index.total_points() == len(las.points)
    assert all(count > 0 for count in index.cell_counts().values())
