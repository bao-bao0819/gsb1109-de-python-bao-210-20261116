"""Spatial grid indexing of point clouds.

This module assigns every point of one or several LAS/LAZ files to a cell
of a *global* grid.

The grid is defined independently of the files being indexed: its origin
(:class:`GridOrigin`) and cell size are given as input and never derived from
the bounding boxes advertised in the file headers. Header bounding boxes are
only used as reading hints. As a consequence the cell a boundary point falls
into, and the per-cell counts, stay identical when:

* the cell size changes (each size defines its own grid, but points are always
  assigned through the same deterministic rule),
* the input files are given in a different order,
* the files are read whole or split into chunks.

Points lying exactly on a grid edge (or corner) are assigned to exactly one
cell: cell intervals are half-open ``[origin + k * size, origin + (k + 1) *
size)`` and the cell index is computed with ``floor``. This removes the
double-counting that happens when each file or chunk snaps its own local
bounding box to grid lines.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Optional, Sequence, Tuple, Union

import numpy as np

from .lasdata import LasData
from .lasreader import LasReader
from .lib import open_las

__all__ = [
    "GridOrigin",
    "GridDefinition",
    "PointGridIndex",
    "GridIndexBuilder",
    "build_point_grid_index",
]


def _coordinates(points) -> np.ndarray:
    """Return the x, y, z coordinates of a LasData or a point record.

    Point records (as yielded by chunked reads) do not expose an ``xyz``
    view, but scaled ``x`` / ``y`` / ``z`` dimensions are available on both
    records and :class:`LasData`.
    """
    return np.column_stack((points.x, points.y, points.z))


@dataclass(frozen=True)
class GridOrigin:
    """Origin (x, y[, z]) of the global grid.

    The origin is part of the grid definition: moving the origin moves the
    grid lines. It must be provided explicitly instead of being inferred from
    the data bounding box so that the assignment stays independent of the
    input files and their order.
    """

    x: float
    y: float
    z: Optional[float] = None

    def as_array(self) -> np.ndarray:
        if self.z is None:
            return np.array([self.x, self.y], dtype=np.float64)
        return np.array([self.x, self.y, self.z], dtype=np.float64)


@dataclass(frozen=True)
class GridDefinition:
    """Definition of the global grid used to assign points to cells.

    Parameters
    ----------
    cell_size:
        Size of a cell along each dimension. A scalar gives cubic / square
        cells, a sequence gives one size per indexed dimension.
    origin:
        Origin of the grid. When ``z`` is ``None`` the grid is 2D (only x/y
        are indexed).
    """

    cell_size: Union[float, Sequence[float]]
    origin: GridOrigin

    def __post_init__(self) -> None:
        sizes = np.atleast_1d(np.asarray(self.cell_size, dtype=np.float64))
        if sizes.ndim != 1 or sizes.size not in (1, 2, 3):
            raise ValueError(
                "cell_size must be a scalar or a sequence of 2 or 3 values"
            )
        if np.any(~np.isfinite(sizes)) or np.any(sizes <= 0.0):
            raise ValueError("cell_size values must be finite and positive")
        dimensions = 3 if self.origin.z is not None else 2
        if sizes.size == 1:
            sizes = np.repeat(sizes, dimensions)
        elif sizes.size != dimensions:
            raise ValueError(
                f"expected {dimensions} cell_size values for a "
                f"{dimensions}D grid, got {sizes.size}"
            )
        object.__setattr__(self, "cell_size", tuple(float(s) for s in sizes))
        if not np.all(np.isfinite(self.origin.as_array())):
            raise ValueError("grid origin must be finite")

    @property
    def is_3d(self) -> bool:
        return self.origin.z is not None

    @property
    def dimensions(self) -> int:
        return 3 if self.is_3d else 2

    def cell_indices(self, xyz: np.ndarray) -> np.ndarray:
        """Return the (n_points, dimensions) integer cell indices of points.

        Cell intervals are half-open: a point located exactly on a grid line
        is assigned to the cell starting on that line (the lower index), so
        boundary points are counted in exactly one cell.
        """
        coords = np.asarray(xyz, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] < self.dimensions:
            raise ValueError(
                f"expected an (n, {self.dimensions}) coordinate array"
            )
        coords = coords[:, : self.dimensions]
        if not np.all(np.isfinite(coords)):
            raise ValueError("point coordinates must be finite to be indexed")
        origin = self.origin.as_array()
        size = np.asarray(self.cell_size, dtype=np.float64)
        # floor makes the rule explicit and keeps negative coordinates
        # consistent with the half-open [origin + k*size, origin + (k+1)*size)
        # intervals.
        return np.floor((coords - origin) / size).astype(np.int64)


class PointGridIndex:
    """Result of indexing point clouds against a global grid.

    Each entry of :attr:`cells` maps a cell index tuple (``(ix, iy)`` for 2D
    grids, ``(ix, iy, iz)`` for 3D grids) to the list of points it contains.
    Every point is represented by a ``(source_file, point_index)`` tuple where
    ``source_file`` is the path (or stream name) the point was read from and
    ``point_index`` is the zero-based index of the point within that file.
    """

    def __init__(self, grid: GridDefinition) -> None:
        self.grid = grid
        self.cells: Dict[Tuple[int, ...], list] = {}

    def add(
        self,
        cell_index: Tuple[int, ...],
        source_file: str,
        point_index: int,
    ) -> None:
        self.cells.setdefault(cell_index, []).append((source_file, point_index))

    def points_of(self, cell_index: Tuple[int, ...]) -> list:
        """Return the ``(source_file, point_index)`` tuples of a cell."""
        return list(self.cells.get(cell_index, ()))

    def count(self, cell_index: Tuple[int, ...]) -> int:
        """Return the number of points assigned to a cell."""
        return len(self.cells.get(cell_index, ()))

    def cell_counts(self) -> Dict[Tuple[int, ...], int]:
        """Return a mapping of every occupied cell to its point count."""
        return {cell_index: len(points) for cell_index, points in self.cells.items()}

    def total_points(self) -> int:
        return sum(len(points) for points in self.cells.values())

    def __len__(self) -> int:
        return len(self.cells)

    def __iter__(self) -> Iterator[Tuple[int, ...]]:
        return iter(self.cells)

    def __contains__(self, cell_index: Tuple[int, ...]) -> bool:
        return cell_index in self.cells


class GridIndexBuilder:
    """Incrementally build a :class:`PointGridIndex` against one grid.

    Files can be added one by one (whole or chunk by chunk) without changing
    the resulting assignment: points are always assigned through the same
    :class:`GridDefinition`.
    """

    def __init__(self, grid: GridDefinition) -> None:
        self.grid = grid
        self.index = PointGridIndex(grid)
        # Per-file running offsets make chunked reads produce the same
        # point indices as whole-file reads.
        self._offsets: Dict[str, int] = {}

    @staticmethod
    def source_name(source) -> str:
        name = getattr(source, "name", None)
        if isinstance(name, str):
            return os.fspath(name)
        return repr(source)

    def add_points(self, xyz: np.ndarray, source_file: str) -> None:
        """Assign points already read from ``source_file`` to cells."""
        indices = self.grid.cell_indices(xyz)
        offset = self._offsets.get(source_file, 0)
        for local_index, cell_index in enumerate(indices):
            self.index.add(
                tuple(int(v) for v in cell_index),
                source_file,
                offset + local_index,
            )
        self._offsets[source_file] = offset + len(indices)

    def add_las(self, las: LasData, source_file: Optional[str] = None) -> None:
        """Assign all the points of an in-memory :class:`LasData`."""
        if source_file is None:
            source_file = self.source_name(las)
        self.add_points(_coordinates(las), source_file)

    def add_reader(
        self,
        reader: LasReader,
        points_per_chunk: Optional[int] = None,
        source_file: Optional[str] = None,
    ) -> None:
        """Assign every point of an opened :class:`LasReader`.

        The header bounding box is ignored: points are read and assigned with
        the global grid only. When ``points_per_chunk`` is given the file is
        streamed chunk by chunk, which yields the same index as a whole read.
        """
        if source_file is None:
            source_file = self.source_name(reader._source)
        reader.seek(0)
        if points_per_chunk is None or points_per_chunk <= 0:
            self.add_points(_coordinates(reader.read()), source_file)
            return

        for chunk in reader.chunk_iterator(points_per_chunk):
            self.add_points(_coordinates(chunk), source_file)

    def add_file(
        self,
        source,
        points_per_chunk: Optional[int] = None,
        laz_backend=None,
    ) -> None:
        """Open a LAS/LAZ file and assign all its points to cells."""
        with open_las(source, laz_backend=laz_backend) as reader:
            self.add_reader(
                reader,
                points_per_chunk=points_per_chunk,
                source_file=self.source_name(reader._source),
            )


def build_point_grid_index(
    sources: Iterable,
    cell_size: Union[float, Sequence[float]],
    origin: Union[GridOrigin, Sequence[float]],
    *,
    points_per_chunk: Optional[int] = None,
) -> PointGridIndex:
    """Build a grid index for one or several LAS/LAZ files.

    Parameters
    ----------
    sources:
        Paths (or readable streams) of the LAS/LAZ files to index, in any
        order. Reordering them does not change which cell each point is
        assigned to nor the per-cell counts.
    cell_size:
        Cell size along each dimension (scalar or 2/3-element sequence).
    origin:
        :class:`GridOrigin` or ``(x, y)`` / ``(x, y, z)`` sequence defining the
        global grid origin. It is never derived from the file bounding boxes.
    points_per_chunk:
        When set, files are read in chunks of that many points; the resulting
        index is identical to reading each file whole.

    Returns
    -------
    PointGridIndex
        The occupied cells with, for each one, the ``(source_file,
        point_index)`` tuples of the points it contains.

    Examples
    --------
    >>> from laspy.grid import GridOrigin, build_point_grid_index
    >>> index = build_point_grid_index(
    ...     ["tests/data/simple.las"], cell_size=10.0, origin=GridOrigin(0.0, 0.0)
    ... )
    >>> index.total_points()
    1065
    """
    if not isinstance(origin, GridOrigin):
        values = tuple(float(v) for v in origin)
        if len(values) == 2:
            origin = GridOrigin(values[0], values[1])
        elif len(values) == 3:
            origin = GridOrigin(values[0], values[1], values[2])
        else:
            raise ValueError("origin must have 2 or 3 components")

    grid = GridDefinition(cell_size=cell_size, origin=origin)
    builder = GridIndexBuilder(grid)
    for source in sources:
        builder.add_file(source, points_per_chunk=points_per_chunk)
    return builder.index
