"""Scan coverage and missed-scan review for multi-strip lidar surveys.

The tools in this module work at the *pulse* level.  Several returns of one
emitted pulse are a single observation, so they can never be counted as
several coverage observations.  Two evidence levels are reported separately:

raw pulse support
    Every return is used to decide whether a pulse observed a grid cell.
    A pulse contributes at most one support count per cell.

retained points
    The returns that pass the filter and the multi-return selection are
    counted, and (again) deduplicated by pulse for the retained pulse count.

Cells are then classified as ``covered``, ``gap``, ``low_density`` or
``unconfirmed``.  Gaps, low-density cells and unconfirmed cells form the
review list.  Low-density cells are only listed as evidence-backed
candidates; they are never silently turned into confirmed coverage.

The module also compares a baseline survey with the same survey after one
or more strips were added.  Cells that were uncovered before and are covered
afterwards are reported as *new coverage*; observations of the added strip
in cells that were already covered are reported as *repeat observations*.

Only numpy is required.  Inputs may be :class:`laspy.LasData` objects, point
records, mappings of dimension names to arrays, paths to LAS/LAZ files, or
iterables of these.  A missing GPS time can be replaced by an explicit
``pulse_id`` dimension or by a caller supplied ``pulse_key``.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, replace
from io import StringIO
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from .filtering import compile_filter


class CoverageError(ValueError):
    """Raised when coverage inputs or options are inconsistent."""


class CellStatus:
    """Values used for the per-cell classification."""

    COVERED = "covered"
    LOW_DENSITY = "low_density"
    GAP = "gap"
    UNCONFIRMED = "unconfirmed"
    OUTSIDE = "outside"

    ALL = (COVERED, LOW_DENSITY, GAP, UNCONFIRMED, OUTSIDE)


CANDIDATE_STATUSES = (
    CellStatus.GAP,
    CellStatus.LOW_DENSITY,
    CellStatus.UNCONFIRMED,
)

MULTI_RETURN_MODES = (
    "all",
    "any",
    "first",
    "last",
    "single",
    "first_and_last",
)

_MULTI_RETURN_ALIASES = {
    "all": "all",
    "any": "all",
    "first": "first",
    "first_only": "first",
    "first-return": "first",
    "last": "last",
    "last_only": "last",
    "last-return": "last",
    "single": "single",
    "single_only": "single",
    "first_and_last": "first_and_last",
    "first-and-last": "first_and_last",
    "first+last": "first_and_last",
}


def _normalise_multi_return(mode: str) -> str:
    try:
        return _MULTI_RETURN_ALIASES[str(mode).lower()]
    except KeyError:
        raise CoverageError(
            f"multi_return must be one of {MULTI_RETURN_MODES}, got {mode!r}"
        ) from None


@dataclass(frozen=True)
class GridSpec:
    """A north-up regular square grid.

    Parameters
    ----------
    x_min, y_min:
        Lower-left corner of the grid.
    cell_size:
        Side length of a cell in the horizontal units of the data.
    width, height:
        Number of columns and rows.
    """

    x_min: float
    y_min: float
    cell_size: float
    width: int
    height: int

    def __post_init__(self) -> None:
        if not np.isfinite(self.cell_size) or self.cell_size <= 0:
            raise CoverageError("cell_size must be a positive finite number")
        if int(self.width) != self.width or int(self.width) <= 0:
            raise CoverageError("width must be a positive integer")
        if int(self.height) != self.height or int(self.height) <= 0:
            raise CoverageError("height must be a positive integer")

    @property
    def x_max(self) -> float:
        return self.x_min + self.width * self.cell_size

    @property
    def y_max(self) -> float:
        return self.y_min + self.height * self.cell_size

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)

    @property
    def cell_area(self) -> float:
        return self.cell_size * self.cell_size

    @classmethod
    def from_bounds(
        cls,
        bounds: Sequence[float],
        cell_size: float,
        *,
        align: bool = True,
    ) -> "GridSpec":
        """Build a grid covering ``(x_min, y_min, x_max, y_max)``.

        With ``align=True`` the origin is rounded down to a multiple of
        ``cell_size`` so grids built independently still line up.
        """
        if len(bounds) != 4:
            raise CoverageError("bounds must contain 4 values")
        if not np.isfinite(cell_size) or cell_size <= 0:
            raise CoverageError("cell_size must be a positive finite number")
        x_min, y_min, x_max, y_max = map(float, bounds)
        if not np.isfinite([x_min, y_min, x_max, y_max]).all():
            raise CoverageError("bounds must be finite")
        if x_max <= x_min or y_max <= y_min:
            raise CoverageError("bounds must have x_max > x_min and y_max > y_min")
        if align:
            x_min = float(np.floor(x_min / cell_size) * cell_size)
            y_min = float(np.floor(y_min / cell_size) * cell_size)
        width = max(1, int(np.ceil((x_max - x_min) / cell_size)))
        height = max(1, int(np.ceil((y_max - y_min) / cell_size)))
        return cls(x_min, y_min, float(cell_size), width, height)

    @classmethod
    def around_points(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        cell_size: float,
        *,
        padding: float = 0.0,
    ) -> "GridSpec":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if not valid.any():
            raise CoverageError("cannot build a grid from empty or non-finite points")
        if not np.isfinite(cell_size) or cell_size <= 0:
            raise CoverageError("cell_size must be a positive finite number")

        x_min = float(np.floor((x[valid].min() - padding) / cell_size) * cell_size)
        y_min = float(np.floor((y[valid].min() - padding) / cell_size) * cell_size)
        x_max = float(x[valid].max()) + padding
        y_max = float(y[valid].max()) + padding
        width = int(np.floor((x_max - x_min) / cell_size)) + 1
        height = int(np.floor((y_max - y_min) / cell_size)) + 1
        return cls(x_min, y_min, float(cell_size), max(1, width), max(1, height))

    def cell_indices(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(row, col)`` for each coordinate.

        Values outside the grid are not clipped; use :meth:`inside` to find
        which coordinates belong to the grid.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        cols = np.floor((x - self.x_min) / self.cell_size).astype(np.int64)
        rows = np.floor((y - self.y_min) / self.cell_size).astype(np.int64)
        return rows, cols

    def inside(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        rows, cols = self.cell_indices(x, y)
        return (
            (rows >= 0)
            & (rows < self.height)
            & (cols >= 0)
            & (cols < self.width)
        )

    def flat_cells(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        rows, cols = self.cell_indices(x, y)
        return rows * self.width + cols

    def cell_centers(self) -> Tuple[np.ndarray, np.ndarray]:
        xs = self.x_min + (np.arange(self.width) + 0.5) * self.cell_size
        ys = self.y_min + (np.arange(self.height) + 0.5) * self.cell_size
        return np.meshgrid(xs, ys)

    def cell_bounds(self, row: int, col: int) -> Tuple[float, float, float, float]:
        x_min = self.x_min + col * self.cell_size
        y_min = self.y_min + row * self.cell_size
        return (x_min, y_min, x_min + self.cell_size, y_min + self.cell_size)

    def cell_id(self, row: int, col: int) -> str:
        return f"r{int(row)}c{int(col)}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
            "cell_size": self.cell_size,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class CoverageConfig:
    """Options controlling coverage analysis."""

    grid: GridSpec
    filter_condition: Any = None
    multi_return: str = "all"
    min_pulses: int = 1
    min_points: int = 1
    min_density: Optional[float] = None
    known_coverage: Any = None
    pulse_key: Any = "auto"
    gps_time_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if not isinstance(self.grid, GridSpec):
            raise CoverageError("grid must be a GridSpec")
        self.multi_return = _normalise_multi_return(self.multi_return)
        if self.min_pulses < 0 or self.min_points < 0:
            raise CoverageError("min_pulses and min_points must be non negative")
        if self.min_density is not None and (
            not np.isfinite(self.min_density) or self.min_density < 0
        ):
            raise CoverageError("min_density must be a non negative number")

    @property
    def min_points_effective(self) -> float:
        """Per-cell point threshold, including the density threshold."""
        if self.min_density is None:
            return float(self.min_points)
        return max(float(self.min_points), self.min_density * self.grid.cell_area)


@dataclass
class CoverageCell:
    """One row of a coverage/review table."""

    row: int
    col: int
    status: str
    reason: str
    raw_point_count: int
    raw_pulse_count: int
    retained_point_count: int
    retained_pulse_count: int
    filtered_point_count: int = 0
    invalid_point_count: int = 0
    grid: Optional[GridSpec] = None

    @property
    def cell_id(self) -> str:
        return f"r{self.row}c{self.col}"

    @property
    def center(self) -> Tuple[float, float]:
        if self.grid is None:
            return (float("nan"), float("nan"))
        x_min, y_min, x_max, y_max = self.grid.cell_bounds(self.row, self.col)
        return ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        if self.grid is None:
            return (float("nan"),) * 4
        return self.grid.cell_bounds(self.row, self.col)

    @property
    def is_candidate(self) -> bool:
        return self.status in CANDIDATE_STATUSES

    def to_dict(self) -> Dict[str, Any]:
        x_min, y_min, x_max, y_max = self.bounds
        center_x, center_y = self.center
        return {
            "cell_id": self.cell_id,
            "row": self.row,
            "col": self.col,
            "status": self.status,
            "reason": self.reason,
            "center_x": center_x,
            "center_y": center_y,
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
            "raw_point_count": self.raw_point_count,
            "raw_pulse_count": self.raw_pulse_count,
            "retained_point_count": self.retained_point_count,
            "retained_pulse_count": self.retained_pulse_count,
            "filtered_point_count": self.filtered_point_count,
            "invalid_point_count": self.invalid_point_count,
        }


@dataclass(eq=False)
class CoverageResult:
    """Coverage analysis of one or more strips on a single grid."""

    grid: GridSpec
    config: CoverageConfig
    status: np.ndarray
    reasons: np.ndarray
    raw_point_counts: np.ndarray
    raw_pulse_counts: np.ndarray
    retained_point_counts: np.ndarray
    retained_pulse_counts: np.ndarray
    filtered_point_counts: np.ndarray
    invalid_point_counts: np.ndarray
    assessed_mask: np.ndarray
    input_point_count: int = 0
    input_pulse_count: int = 0
    retained_point_count: int = 0
    retained_pulse_count: int = 0
    invalid_point_count: int = 0
    filtered_point_count: int = 0
    outside_point_count: int = 0
    pulse_identity_methods: List[str] = field(default_factory=list)
    source_data: Optional[List[Any]] = field(default=None, repr=False, compare=False)

    # -- evidence aliases ------------------------------------------------
    @property
    def raw_pulse_support(self) -> np.ndarray:
        return self.raw_pulse_counts

    @property
    def retained_points(self) -> np.ndarray:
        return self.retained_point_counts

    @property
    def covered_cells(self) -> np.ndarray:
        return self.status == CellStatus.COVERED

    @property
    def gap_cells(self) -> np.ndarray:
        return self.status == CellStatus.GAP

    @property
    def low_density_cells(self) -> np.ndarray:
        return self.status == CellStatus.LOW_DENSITY

    @property
    def unconfirmed_cells(self) -> np.ndarray:
        return self.status == CellStatus.UNCONFIRMED

    @property
    def candidate_cells(self) -> np.ndarray:
        return np.isin(self.status, CANDIDATE_STATUSES) & self.assessed_mask

    @property
    def raw_pulse_support_cells(self) -> np.ndarray:
        return (self.raw_pulse_counts > 0) & self.assessed_mask

    @property
    def pulse_identity_reliable(self) -> bool:
        return bool(self.pulse_identity_methods) and all(
            method not in ("none", "returns") for method in self.pulse_identity_methods
        )

    @property
    def reviewed_cell_count(self) -> int:
        return int(np.count_nonzero(self.candidate_cells))

    # -- reporting -------------------------------------------------------
    def _count(self, mask: np.ndarray) -> int:
        return int(np.count_nonzero(mask & self.assessed_mask))

    def summary(self) -> Dict[str, Any]:
        """Return a JSON serialisable overview of the three evidence levels."""
        return {
            "grid": self.grid.to_dict(),
            "pulses": {
                "input": int(self.input_pulse_count),
                "retained": int(self.retained_pulse_count),
                "raw_pulse_support_cells": self._count(
                    self.raw_pulse_counts > 0
                ),
                "identity_methods": list(self.pulse_identity_methods),
                "identity_reliable": self.pulse_identity_reliable,
            },
            "points": {
                "input": int(self.input_point_count),
                "retained": int(self.retained_point_count),
                "filtered_out": int(self.filtered_point_count),
                "invalid_pulse_identity": int(self.invalid_point_count),
                "outside_grid": int(self.outside_point_count),
            },
            "cells": {
                "assessed": self._count(np.ones(self.grid.shape, dtype=bool)),
                "covered": self._count(self.covered_cells),
                "gap": self._count(self.gap_cells),
                "low_density": self._count(self.low_density_cells),
                "unconfirmed": self._count(self.unconfirmed_cells),
                "candidates": int(np.count_nonzero(self.candidate_cells)),
                "outside_known_coverage": int(
                    np.count_nonzero(~self.assessed_mask)
                ),
            },
            "thresholds": {
                "min_pulses": self.config.min_pulses,
                "min_points": self.config.min_points,
                "min_density": self.config.min_density,
                "effective_min_points": self.config.min_points_effective,
                "multi_return": self.config.multi_return,
            },
        }

    # raw / retained / unconfirmed statistics at cell level
    def evidence_summary(self) -> Dict[str, Any]:
        assigned = self.assessed_mask & (self.raw_point_counts > 0)
        return {
            "raw_pulse_support": {
                "cells_with_support": int(
                    np.count_nonzero(self.raw_pulse_support_cells)
                ),
                "unique_pulses": int(self.input_pulse_count),
                # A pulse crossing several cells supports each of them, so
                # this is cell-level support, not a count of unique pulses.
                "pulse_cell_support": int(self.raw_pulse_counts[assigned].sum()),
                "points": int(self.raw_point_counts[assigned].sum()),
            },
            "retained_points": {
                "cells": int(
                    np.count_nonzero(
                        self.assessed_mask & (self.retained_point_counts > 0)
                    )
                ),
                "points": int(self.retained_point_counts[self.assessed_mask].sum()),
                "unique_pulses": int(self.retained_pulse_count),
                "pulse_cell_support": int(
                    self.retained_pulse_counts[self.assessed_mask].sum()
                ),
            },
            "unconfirmed": {
                "cells": self._count(self.unconfirmed_cells),
                "points": int(
                    self.raw_point_counts[self.unconfirmed_cells].sum()
                    if np.any(self.unconfirmed_cells)
                    else 0
                ),
            },
        }

    # -- review list -----------------------------------------------------
    def cells(
        self,
        *,
        candidates_only: bool = True,
        statuses: Optional[Iterable[str]] = None,
    ) -> List[CoverageCell]:
        if statuses is not None:
            selected = np.isin(self.status, list(statuses)) & self.assessed_mask
        elif candidates_only:
            selected = self.candidate_cells
        else:
            selected = self.assessed_mask

        result: List[CoverageCell] = []
        for row, col in zip(*np.nonzero(selected)):
            result.append(
                CoverageCell(
                    row=int(row),
                    col=int(col),
                    status=str(self.status[row, col]),
                    reason=str(self.reasons[row, col]),
                    raw_point_count=int(self.raw_point_counts[row, col]),
                    raw_pulse_count=int(self.raw_pulse_counts[row, col]),
                    retained_point_count=int(
                        self.retained_point_counts[row, col]
                    ),
                    retained_pulse_count=int(
                        self.retained_pulse_counts[row, col]
                    ),
                    filtered_point_count=int(
                        self.filtered_point_counts[row, col]
                    ),
                    invalid_point_count=int(self.invalid_point_counts[row, col]),
                    grid=self.grid,
                )
            )
        return result

    def review_list(self) -> List[CoverageCell]:
        """Return review candidates ordered by priority.

        Complete gaps come first, then evidence-backed low-density cells,
        then cells that cannot be confirmed.  Within a class, cells with
        more raw pulse support are listed first.
        """
        priority = {
            CellStatus.GAP: 0,
            CellStatus.LOW_DENSITY: 1,
            CellStatus.UNCONFIRMED: 2,
        }
        cells = self.cells(candidates_only=True)
        cells.sort(
            key=lambda cell: (
                priority.get(cell.status, 9),
                -cell.raw_pulse_count,
                -cell.retained_point_count,
                cell.row,
                cell.col,
            )
        )
        return cells

    def review_rows(self) -> List[Dict[str, Any]]:
        return [cell.to_dict() for cell in self.review_list()]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary(),
            "evidence": self.evidence_summary(),
            "review": self.review_rows(),
        }

    def to_json(self, path: Optional[Union[str, Path]] = None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def to_csv(self, path: Optional[Union[str, Path]] = None) -> str:
        rows = self.review_rows()
        fieldnames = [
            "cell_id",
            "row",
            "col",
            "status",
            "reason",
            "center_x",
            "center_y",
            "x_min",
            "y_min",
            "x_max",
            "y_max",
            "raw_point_count",
            "raw_pulse_count",
            "retained_point_count",
            "retained_pulse_count",
            "filtered_point_count",
            "invalid_point_count",
        ]
        buffer = StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})
        text = buffer.getvalue()
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text


@dataclass(eq=False)
class CoverageComparison:
    """Difference between a baseline and the survey with added strips."""

    grid: GridSpec
    baseline: CoverageResult
    added: CoverageResult
    combined: CoverageResult
    newly_covered_cells: np.ndarray
    repeat_observation_cells: np.ndarray
    remaining_candidate_cells: np.ndarray
    added_point_counts: np.ndarray
    added_pulse_counts: np.ndarray
    new_coverage_point_counts: np.ndarray
    repeat_observation_point_counts: np.ndarray
    new_coverage_pulse_counts: np.ndarray
    repeat_observation_pulse_counts: np.ndarray

    @property
    def new_coverage(self) -> np.ndarray:
        return self.newly_covered_cells

    @property
    def repeated_observations(self) -> np.ndarray:
        return self.repeat_observation_cells

    @property
    def new_coverage_cell_count(self) -> int:
        return int(np.count_nonzero(self.newly_covered_cells))

    @property
    def repeat_observation_cell_count(self) -> int:
        return int(np.count_nonzero(self.repeat_observation_cells))

    def summary(self) -> Dict[str, Any]:
        repeat = self.repeat_observation_cells
        new = self.newly_covered_cells
        return {
            "grid": self.grid.to_dict(),
            "baseline": self.baseline.summary(),
            "added": self.added.summary(),
            "combined": self.combined.summary(),
            "delta": {
                "newly_covered_cells": int(np.count_nonzero(new)),
                "repeat_observation_cells": int(np.count_nonzero(repeat)),
                "remaining_candidate_cells": int(
                    np.count_nonzero(self.remaining_candidate_cells)
                ),
                "new_coverage_points": int(self.new_coverage_point_counts.sum()),
                "new_coverage_pulses": int(self.new_coverage_pulse_counts.sum()),
                "repeat_observation_points": int(
                    self.repeat_observation_point_counts.sum()
                ),
                "repeat_observation_pulses": int(
                    self.repeat_observation_pulse_counts.sum()
                ),
                "added_points": int(self.added_point_counts.sum()),
                "added_pulses": int(self.added_pulse_counts.sum()),
            },
        }

    def cells(self) -> List[Dict[str, Any]]:
        selected = self.newly_covered_cells | self.repeat_observation_cells
        result: List[Dict[str, Any]] = []
        for row, col in zip(*np.nonzero(selected)):
            x_min, y_min, x_max, y_max = self.grid.cell_bounds(int(row), int(col))
            if self.newly_covered_cells[row, col]:
                classification = "new_coverage"
            else:
                classification = "repeat_observation"
            result.append(
                {
                    "cell_id": self.grid.cell_id(row, col),
                    "row": int(row),
                    "col": int(col),
                    "classification": classification,
                    "center_x": (x_min + x_max) / 2.0,
                    "center_y": (y_min + y_max) / 2.0,
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                    "baseline_status": str(self.baseline.status[row, col]),
                    "combined_status": str(self.combined.status[row, col]),
                    "added_point_count": int(self.added_point_counts[row, col]),
                    "added_pulse_count": int(self.added_pulse_counts[row, col]),
                }
            )
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {"summary": self.summary(), "cells": self.cells()}

    def to_json(self, path: Optional[Union[str, Path]] = None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def to_csv(self, path: Optional[Union[str, Path]] = None) -> str:
        rows = self.cells()
        fieldnames = [
            "cell_id",
            "row",
            "col",
            "classification",
            "center_x",
            "center_y",
            "x_min",
            "y_min",
            "x_max",
            "y_max",
            "baseline_status",
            "combined_status",
            "added_point_count",
            "added_pulse_count",
        ]
        buffer = StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})
        text = buffer.getvalue()
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text


# ----------------------------------------------------------------------
# Normalisation of point sources
# ----------------------------------------------------------------------
def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _source_length(source: Any) -> int:
    if _is_mapping(source):
        for name in ("x", "y", "X", "Y"):
            if name in source:
                return len(source[name])
        raise CoverageError("point source mapping has no coordinate arrays")
    try:
        return len(source)
    except TypeError:
        x = _column(source, "x")
        if x is None:
            x = _column(source, "X")
        if x is None:
            raise CoverageError("could not determine the number of points")
        return len(x)


def _expand_path(path: Union[str, Path]) -> List[Path]:
    path = Path(path)
    if not path.is_dir():
        return [path]
    files = sorted(
        file
        for file in path.iterdir()
        if file.is_file() and file.suffix.lower() in (".las", ".laz")
    )
    if not files:
        raise CoverageError(f"directory '{path}' contains no LAS/LAZ files")
    return files


def _normalise_sources(source: Any) -> List[Any]:
    if isinstance(source, (str, Path)):
        items: List[Any] = list(_expand_path(source))
    elif _is_mapping(source):
        items = [source]
    elif isinstance(source, (list, tuple)):
        items = []
        for item in source:
            if isinstance(item, (str, Path)):
                items.extend(_expand_path(item))
            else:
                items.append(item)
    elif hasattr(source, "points") and hasattr(source, "header"):
        items = [source]
    elif hasattr(source, "chunk_iterator"):
        items = [source]
    else:
        items = [source]

    sources: List[Any] = []
    for item in items:
        if isinstance(item, (str, Path)):
            import laspy

            item = laspy.read(item)
        if hasattr(item, "chunk_iterator") and not hasattr(item, "points"):
            item = item.read()
        if hasattr(item, "points") and hasattr(item, "header"):
            item = item.points
        sources.append(item)
    if not sources:
        raise CoverageError("at least one point source is required")
    return sources


def _column(source: Any, name: str) -> Optional[np.ndarray]:
    value = None
    if _is_mapping(source):
        if name in source:
            value = source[name]
        elif name.lower() in source:
            value = source[name.lower()]
        elif name.upper() in source:
            value = source[name.upper()]
    else:
        try:
            value = source[name]
        except (KeyError, ValueError, IndexError, TypeError):
            try:
                value = getattr(source, name)
            except AttributeError:
                value = None
    if value is None:
        return None
    return np.asarray(value)


def _xy_columns(source: Any) -> Tuple[np.ndarray, np.ndarray]:
    x = _column(source, "x")
    y = _column(source, "y")
    if x is None:
        x = _column(source, "X")
        if x is not None:
            scales = getattr(source, "scales", None)
            offsets = getattr(source, "offsets", None)
            if scales is None or offsets is None:
                raise CoverageError(
                    "integer X/Y coordinates require scales and offsets"
                )
            x = np.asarray(x, dtype=float) * scales[0] + offsets[0]
    if y is None:
        y = _column(source, "Y")
        if y is not None:
            scales = getattr(source, "scales", None)
            offsets = getattr(source, "offsets", None)
            if scales is None or offsets is None:
                raise CoverageError(
                    "integer X/Y coordinates require scales and offsets"
                )
            y = np.asarray(y, dtype=float) * scales[1] + offsets[1]
    if x is None or y is None:
        raise CoverageError(
            "point sources must provide x/y coordinates (or X/Y with scales "
            "and offsets)"
        )
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise CoverageError("x and y arrays must have the same shape")
    return x, y


def _int_column(source: Any, name: str) -> Optional[np.ndarray]:
    value = _column(source, name)
    if value is None:
        return None
    return np.asarray(value).astype(np.int64)


def _round_gps(values: np.ndarray, tolerance: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if tolerance <= 0:
        return values
    if tolerance < 1.0:
        decimals = int(np.ceil(-np.log10(tolerance)))
        return np.round(values, decimals)
    return np.round(values / tolerance) * tolerance


def _factorize_keys(
    arrays: Sequence[np.ndarray], *, treat_nan_as_unique: bool = False
) -> Tuple[np.ndarray, int]:
    arrays = [np.asarray(array) for array in arrays]
    if not arrays:
        raise CoverageError("at least one key array is required")
    length = len(arrays[0])
    if any(len(array) != length for array in arrays):
        raise CoverageError("pulse key arrays must have the same length")
    if length == 0:
        return np.empty(0, dtype=np.int64), 0

    if len(arrays) == 1:
        values = arrays[0]

        if treat_nan_as_unique and np.issubdtype(values.dtype, np.floating):
            # NaN never equals NaN, so every NaN must stay a unique key.
            nan_mask = np.isnan(values)
            ids = np.full(length, -1, dtype=np.int64)
            nan_positions = np.nonzero(nan_mask)[0]
            if len(nan_positions):
                ids[nan_positions] = (
                    np.arange(len(nan_positions), dtype=np.int64)
                )
            finite_positions = np.nonzero(~nan_mask)[0]
            if len(finite_positions):
                finite_ids = _factorize_keys([values[finite_positions]])[0]
                ids[finite_positions] = finite_ids + len(nan_positions)
            return ids, int(ids.max()) + 1

        _, inverse = np.unique(values, return_inverse=True)
        inverse = np.asarray(inverse, dtype=np.int64).reshape(-1)
        return inverse, int(inverse.max()) + 1

    columns = [np.asarray(array) for array in arrays]
    if treat_nan_as_unique:
        nan_rows = np.zeros(length, dtype=bool)
        for column in columns:
            if np.issubdtype(column.dtype, np.floating):
                nan_rows |= np.isnan(column)
    else:
        nan_rows = np.zeros(length, dtype=bool)

    safe_columns = []
    for column in columns:
        if np.issubdtype(column.dtype, np.floating):
            column = np.where(nan_rows, 0.0, column)
        safe_columns.append(column)

    structured = np.zeros(
        length, dtype=[(f"k{index}", column.dtype) for index, column in enumerate(safe_columns)]
    )
    for index, column in enumerate(safe_columns):
        structured[f"k{index}"] = column

    _, inverse = np.unique(structured, return_inverse=True)
    inverse = np.asarray(inverse, dtype=np.int64).reshape(-1)

    unique_rows = len(np.unique(structured))
    if nan_rows.any():
        nan_count = int(np.count_nonzero(nan_rows))
        inverse = inverse + nan_count
        inverse[nan_rows] = np.arange(nan_count)
        return inverse, unique_rows + nan_count
    return inverse, unique_rows


@dataclass
class _PulseIdentity:
    ids: np.ndarray
    valid: np.ndarray
    reliable: bool
    method: str


def _return_run_ids(
    return_number: np.ndarray,
    number_of_returns: np.ndarray,
    point_source_id: Optional[np.ndarray],
) -> np.ndarray:
    """Group contiguous returns into pulses using return-number runs.

    This is the best effort when neither GPS time nor an explicit pulse id is
    available.  It is conservative: a new return number 1 always starts a
    new pulse, and a return that does not follow the previous one also does.
    """
    n = len(return_number)
    ids = np.empty(n, dtype=np.int64)
    current = -1
    previous_return = 0
    previous_total = 0
    previous_source: Optional[int] = None
    for index in range(n):
        rn = int(return_number[index])
        nr = int(number_of_returns[index])
        source = int(point_source_id[index]) if point_source_id is not None else None
        starts_new = current < 0
        if current >= 0:
            if source != previous_source:
                starts_new = True
            elif rn == 1:
                starts_new = True
            elif rn <= previous_return:
                starts_new = True
            elif previous_total > 0 and previous_return >= previous_total:
                starts_new = True
        if starts_new:
            current += 1
        ids[index] = current
        previous_return = rn
        previous_total = nr
        previous_source = source
    return ids


def _pulse_identity(
    source: Any,
    *,
    pulse_key: Any,
    gps_time_tolerance: float,
) -> _PulseIdentity:
    n = _source_length(source)
    return_number = _int_column(source, "return_number")
    number_of_returns = _int_column(source, "number_of_returns")
    point_source_id = _int_column(source, "point_source_id")

    if number_of_returns is not None and return_number is not None:
        valid = (
            (return_number >= 1)
            & (number_of_returns >= 1)
            & (return_number <= number_of_returns)
        )
    elif return_number is not None:
        valid = return_number >= 1
    elif number_of_returns is not None:
        valid = number_of_returns >= 1
    else:
        valid = np.ones(n, dtype=bool)

    # 1. explicit caller supplied key
    if pulse_key not in (None, "auto", "default"):
        if callable(pulse_key):
            keys = np.asarray(pulse_key(source))
        else:
            if isinstance(pulse_key, str):
                field_names = [
                    name.strip()
                    for name in pulse_key.split(",")
                    if name.strip()
                ]
            else:
                field_names = list(pulse_key)
            arrays = []
            for name in field_names:
                values = _column(source, name)
                if values is None:
                    raise CoverageError(
                        f"pulse_key references missing field '{name}'"
                    )
                if np.issubdtype(values.dtype, np.floating):
                    values = _round_gps(values, gps_time_tolerance)
                arrays.append(np.asarray(values))
            keys = arrays if len(arrays) > 1 else arrays[0]
        if isinstance(keys, list):
            ids, _ = _factorize_keys(keys)
        elif keys.ndim == 1:
            ids, _ = _factorize_keys([keys])
        else:
            ids, _ = _factorize_keys([keys[:, index] for index in range(keys.shape[1])])
        return _PulseIdentity(ids, valid, True, "custom_key")

    # 2. explicit pulse id dimension
    pulse_id = _column(source, "pulse_id")
    if pulse_id is not None:
        ids, _ = _factorize_keys([np.asarray(pulse_id)])
        return _PulseIdentity(ids, valid, True, "pulse_id")

    # 3. GPS time is the pulse emission time; all returns of one pulse share
    #    it.  `gps_time_tolerance` absorbs the small differences some
    #    producers store per return.
    gps_time = _column(source, "gps_time")
    if gps_time is not None:
        gps = np.asarray(gps_time, dtype=float)
        valid_gps = np.isfinite(gps)
        rounded = np.full(n, np.nan, dtype=float)
        rounded[valid_gps] = _round_gps(gps[valid_gps], gps_time_tolerance)
        arrays: List[np.ndarray] = [rounded]
        if point_source_id is not None:
            arrays.append(point_source_id)
        ids, _ = _factorize_keys(arrays, treat_nan_as_unique=True)
        valid = valid & valid_gps
        return _PulseIdentity(ids, valid, True, "gps_time")

    # 4. contiguous return runs
    if return_number is not None and number_of_returns is not None:
        ids = _return_run_ids(return_number, number_of_returns, point_source_id)
        return _PulseIdentity(ids, valid, False, "returns")

    # 5. no identity at all
    return _PulseIdentity(np.arange(n, dtype=np.int64), valid, False, "none")


def _return_selection_mask(
    return_number: Optional[np.ndarray],
    number_of_returns: Optional[np.ndarray],
    pulse_ids: np.ndarray,
    mode: str,
) -> np.ndarray:
    if mode == "all":
        return np.ones(len(pulse_ids), dtype=bool)

    if return_number is None or number_of_returns is None:
        raise CoverageError(
            f"multi_return={mode!r} requires return_number and "
            "number_of_returns dimensions"
        )

    first = return_number == 1
    last = return_number == number_of_returns
    if mode == "first":
        return first
    if mode == "last":
        return last
    if mode == "single":
        return number_of_returns == 1
    if mode == "first_and_last":
        return first | last
    raise CoverageError(f"invalid multi_return mode {mode!r}")


def _filter_mask(source: Any, condition: Any, n: int) -> np.ndarray:
    if condition is None:
        return np.ones(n, dtype=bool)
    if isinstance(condition, np.ndarray):
        mask = np.asarray(condition, dtype=bool)
    elif callable(condition) or hasattr(condition, "apply") or isinstance(
        condition, (str, list, tuple)
    ):
        mask = np.asarray(compile_filter(condition)(source))
    else:
        raise TypeError(
            "filter_condition must be an expression string, callable, mask, "
            "or sequence of expression strings"
        )
    if mask.shape != (n,):
        raise CoverageError("filter mask does not match the number of points")
    return mask.astype(bool, copy=False)


def _coverage_mask(known_coverage: Any, grid: GridSpec) -> np.ndarray:
    if known_coverage is None:
        return np.ones(grid.shape, dtype=bool)

    if isinstance(known_coverage, CoverageResult):
        return known_coverage.assessed_mask.copy()

    if isinstance(known_coverage, (str, Path)):
        path = Path(known_coverage)
        suffix = path.suffix.lower()
        if suffix == ".npy":
            known_coverage = np.load(path)
        elif suffix in (".json", ".geojson"):
            known_coverage = json.loads(path.read_text(encoding="utf-8"))
        elif suffix in (".csv", ".txt"):
            array = np.loadtxt(path, delimiter=",")
            known_coverage = array.astype(bool)
        else:
            raise CoverageError(
                f"unsupported known coverage file '{path}'; use .npy, "
                "GeoJSON or a CSV mask"
            )

    if isinstance(known_coverage, np.ndarray):
        array = np.asarray(known_coverage)
        if array.shape == grid.shape:
            return array.astype(bool)
        if array.size == grid.width * grid.height:
            return array.reshape(grid.shape).astype(bool)
        raise CoverageError(
            f"known coverage mask has shape {array.shape}, expected {grid.shape}"
        )

    if (
        isinstance(known_coverage, (tuple, list))
        and len(known_coverage) == 4
        and all(np.isscalar(value) for value in known_coverage)
    ):
        x_min, y_min, x_max, y_max = map(float, known_coverage)
        centers_x, centers_y = grid.cell_centers()
        return (
            (centers_x >= x_min)
            & (centers_x <= x_max)
            & (centers_y >= y_min)
            & (centers_y <= y_max)
        )

    polygons = _as_polygons(known_coverage)
    if polygons:
        centers_x, centers_y = grid.cell_centers()
        mask = np.zeros(grid.shape, dtype=bool)
        for polygon in polygons:
            mask |= _points_in_polygon(centers_x, centers_y, polygon)
        return mask

    raise CoverageError(
        "known_coverage must be a boolean grid mask, bounds, polygons, a "
        "GeoJSON file or a .npy mask"
    )


def _as_polygons(value: Any) -> List[np.ndarray]:
    if isinstance(value, np.ndarray):
        return []
    if isinstance(value, Mapping):
        if "geometry" in value:
            return _as_polygons(value["geometry"])
        if "features" in value:
            polygons: List[np.ndarray] = []
            for feature in value["features"]:
                polygons.extend(_as_polygons(feature))
            return polygons
        geo_type = value.get("type")
        coordinates = value.get("coordinates")
        if geo_type == "Polygon" and coordinates:
            return [np.asarray(coordinates[0], dtype=float)]
        if geo_type == "MultiPolygon" and coordinates:
            return [np.asarray(polygon[0], dtype=float) for polygon in coordinates]
        return []
    if hasattr(value, "__geo_interface__"):
        return _as_polygons(value.__geo_interface__)
    if isinstance(value, (list, tuple)):
        if not value:
            return []
        if all(
            isinstance(point, (list, tuple, np.ndarray)) and len(point) == 2
            for point in value
        ):
            return [np.asarray(value, dtype=float)]
        polygons = []
        for item in value:
            polygons.extend(_as_polygons(item))
        return polygons
    return []


def _points_in_polygon(
    x: np.ndarray, y: np.ndarray, polygon: np.ndarray
) -> np.ndarray:
    """Vectorised ray casting test for grid cell centres against a polygon."""
    if len(polygon) < 3:
        return np.zeros(np.shape(x), dtype=bool)
    inside = np.zeros(np.shape(x), dtype=bool)
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        crosses = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / ((yj - yi) + np.finfo(float).eps) + xi
        )
        inside ^= crosses
        j = i
    return inside


# ----------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------
@dataclass
class _PreparedSource:
    x: np.ndarray
    y: np.ndarray
    n: int
    finite: np.ndarray
    pulse_ids: np.ndarray
    identity_valid: np.ndarray
    retained: np.ndarray
    filtered: np.ndarray
    identity: _PulseIdentity


def _prepare_source(source: Any, config: CoverageConfig) -> _PreparedSource:
    x, y = _xy_columns(source)
    n = len(x)
    finite = np.isfinite(x) & np.isfinite(y)

    identity = _pulse_identity(
        source,
        pulse_key=config.pulse_key,
        gps_time_tolerance=config.gps_time_tolerance,
    )
    return_number = _int_column(source, "return_number")
    number_of_returns = _int_column(source, "number_of_returns")
    mask = _filter_mask(source, config.filter_condition, n)
    selected = _return_selection_mask(
        return_number, number_of_returns, identity.ids, config.multi_return
    )
    # Retained evidence must also have a valid pulse identity, otherwise a
    # point could look like support that cannot be attributed to any pulse.
    retained = mask & finite & identity.valid & selected
    return _PreparedSource(
        x=x,
        y=y,
        n=n,
        finite=finite,
        pulse_ids=identity.ids,
        identity_valid=identity.valid & finite,
        retained=retained,
        filtered=finite & ~mask,
        identity=identity,
    )


def _classify(
    config: CoverageConfig,
    assessed_mask: np.ndarray,
    raw_point_counts: np.ndarray,
    raw_pulse_counts: np.ndarray,
    retained_point_counts: np.ndarray,
    retained_pulse_counts: np.ndarray,
    filtered_point_counts: np.ndarray,
    invalid_point_counts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    shape = assessed_mask.shape
    status = np.full(shape, CellStatus.OUTSIDE, dtype=object)
    reasons = np.full(shape, "outside known coverage", dtype=object)

    covered = (
        assessed_mask
        & (retained_pulse_counts > 0)
        & (retained_pulse_counts >= config.min_pulses)
        & (retained_point_counts >= config.min_points_effective)
    )
    status[covered] = CellStatus.COVERED
    reasons[covered] = "retained pulse support meets threshold"

    has_raw_points = assessed_mask & (raw_point_counts > 0)
    no_raw_pulses = has_raw_points & (raw_pulse_counts == 0)
    all_filtered = has_raw_points & (filtered_point_counts >= raw_point_counts)
    no_retained = has_raw_points & (retained_point_counts == 0)

    unconfirmed = assessed_mask & ~covered & (
        no_raw_pulses | all_filtered | no_retained
    )
    status[unconfirmed] = CellStatus.UNCONFIRMED
    reasons[unconfirmed] = "retained evidence cannot confirm coverage"
    reasons[unconfirmed & no_raw_pulses] = "pulse identity is missing or invalid"
    reasons[unconfirmed & ~no_raw_pulses & all_filtered] = (
        "all raw points were removed by the filter"
    )
    reasons[
        unconfirmed & ~no_raw_pulses & ~all_filtered & no_retained
    ] = "multi-return policy kept no return in the cell"

    low_density = (
        assessed_mask
        & ~covered
        & ~unconfirmed
        & (raw_point_counts > 0)
        & (retained_point_counts > 0)
    )
    status[low_density] = CellStatus.LOW_DENSITY
    reasons[low_density] = (
        "raw evidence exists but retained support is below threshold"
    )

    gap = assessed_mask & ~covered & ~unconfirmed & ~low_density
    status[gap] = CellStatus.GAP
    reasons[gap] = "no raw pulse returns in this cell"

    return status, reasons


def _empty_counts(grid: GridSpec) -> Dict[str, np.ndarray]:
    shape = grid.shape
    return {
        name: np.zeros(shape, dtype=np.int64)
        for name in (
            "raw_point_counts",
            "raw_pulse_counts",
            "retained_point_counts",
            "retained_pulse_counts",
            "filtered_point_counts",
            "invalid_point_counts",
        )
    }


def analyze_coverage(
    sources: Any,
    *,
    grid: Optional[Union[GridSpec, Sequence[float]]] = None,
    cell_size: Optional[float] = None,
    bounds: Optional[Sequence[float]] = None,
    filter_condition: Any = None,
    multi_return: str = "all",
    min_pulses: int = 1,
    min_points: int = 1,
    min_density: Optional[float] = None,
    threshold: Optional[int] = None,
    known_coverage: Any = None,
    pulse_key: Any = "auto",
    gps_time_tolerance: float = 1e-6,
    config: Optional[CoverageConfig] = None,
) -> CoverageResult:
    """Analyse raw pulse support and retained points on a regular grid.

    Parameters
    ----------
    sources:
        One source or an iterable of sources.  A source can be a
        :class:`laspy.LasData`, a point record, a mapping of dimension names
        to arrays, a path to a LAS/LAZ file, or a readable LAS stream.
    grid:
        A :class:`GridSpec`, ``(x_min, y_min, x_max, y_max)`` bounds, or
        ``None``.  When omitted, ``cell_size`` is required and the grid is
        derived from the point extent (or from ``bounds`` if given).
    filter_condition:
        Filter expression string, callable, boolean mask, or a sequence of
        expression strings combined with ``and``.
    multi_return:
        Which returns are retained as evidence: ``all`` (default), ``first``,
        ``last``, ``single`` or ``first_and_last``.  Raw pulse support always
        uses every return; this option only affects retained points.
    min_pulses, min_points:
        Per-cell thresholds.  ``threshold`` is an alias for ``min_pulses``.
    min_density:
        Optional minimum retained points per unit area.  The effective point
        threshold is ``max(min_points, min_density * cell_area)``.
    known_coverage:
        Boolean grid mask, bounds, polygon(s), GeoJSON, or ``.npy`` file
        describing the known survey coverage.  Cells outside it are marked
        ``outside`` and are never candidates.
    pulse_key:
        ``"auto"`` (default), a field name, a list of field names, or a
        callable returning one key per point.  ``gps_time`` is used
        automatically when present; an explicit ``pulse_id`` dimension is
        used next; contiguous return-number runs are the last resort.
    """
    if threshold is not None:
        min_pulses = threshold

    if config is not None:
        if not isinstance(config, CoverageConfig):
            raise CoverageError("config must be a CoverageConfig")
        if grid is not None and isinstance(grid, GridSpec) and grid != config.grid:
            raise CoverageError("grid and config.grid disagree")
        if cell_size is not None and config.grid.cell_size != cell_size:
            raise CoverageError("cell_size and config.grid disagree")
        effective = config
    else:
        if isinstance(grid, GridSpec):
            grid_spec = grid
        elif grid is not None:
            if cell_size is None:
                raise CoverageError("cell_size is required when bounds are given")
            grid_spec = GridSpec.from_bounds(grid, cell_size)
        else:
            if cell_size is None:
                raise CoverageError("either grid or cell_size must be provided")
            if bounds is not None:
                grid_spec = GridSpec.from_bounds(bounds, cell_size)
            else:
                all_x: List[np.ndarray] = []
                all_y: List[np.ndarray] = []
                for source in _normalise_sources(sources):
                    x, y = _xy_columns(source)
                    all_x.append(x)
                    all_y.append(y)
                grid_spec = GridSpec.around_points(
                    np.concatenate(all_x), np.concatenate(all_y), cell_size
                )
        effective = CoverageConfig(
            grid=grid_spec,
            filter_condition=filter_condition,
            multi_return=multi_return,
            min_pulses=min_pulses,
            min_points=min_points,
            min_density=min_density,
            known_coverage=known_coverage,
            pulse_key=pulse_key,
            gps_time_tolerance=gps_time_tolerance,
        )

    normalised = _normalise_sources(sources)
    prepared = [_prepare_source(source, effective) for source in normalised]
    grid_spec = effective.grid
    counts = _empty_counts(grid_spec)
    size = grid_spec.width * grid_spec.height

    input_point_count = 0
    retained_point_count = 0
    invalid_point_count = 0
    filtered_point_count = 0
    outside_point_count = 0
    global_pulses: set = set()
    retained_global_pulses: set = set()
    identity_methods: List[str] = []

    for source_index, item in enumerate(prepared):
        identity_methods.append(item.identity.method)
        input_point_count += item.n
        filtered_point_count += int(np.count_nonzero(item.filtered))
        retained_point_count += int(np.count_nonzero(item.retained))
        invalid_point_count += int(
            np.count_nonzero(item.finite & ~item.identity_valid)
        )

        if not item.finite.any():
            continue

        item_x = item.x[item.finite]
        item_y = item.y[item.finite]
        inside = grid_spec.inside(item_x, item_y)
        outside_point_count += int(np.count_nonzero(~inside))
        if not inside.any():
            continue

        cells = grid_spec.flat_cells(item_x[inside], item_y[inside])
        ids = item.pulse_ids[item.finite][inside]
        valid = item.identity_valid[item.finite][inside]
        removed_by_filter = item.filtered[item.finite][inside]
        is_retained = item.retained[item.finite][inside]

        counts["raw_point_counts"] += np.bincount(
            cells, minlength=size
        ).reshape(grid_spec.shape)
        counts["filtered_point_counts"] += np.bincount(
            cells[removed_by_filter], minlength=size
        ).reshape(grid_spec.shape)
        counts["invalid_point_counts"] += np.bincount(
            cells[~valid], minlength=size
        ).reshape(grid_spec.shape)

        if valid.any():
            valid_cells = cells[valid]
            valid_ids = ids[valid]
            pairs = np.unique(
                np.stack((valid_ids, valid_cells), axis=1), axis=0
            )
            counts["raw_pulse_counts"] += np.bincount(
                pairs[:, 1], minlength=size
            ).reshape(grid_spec.shape)
            global_pulses.update(
                (source_index, int(pulse)) for pulse in np.unique(valid_ids)
            )

        counts["retained_point_counts"] += np.bincount(
            cells[is_retained], minlength=size
        ).reshape(grid_spec.shape)

        retained_and_valid = is_retained & valid
        if retained_and_valid.any():
            retained_cells = cells[retained_and_valid]
            retained_ids = ids[retained_and_valid]
            pairs = np.unique(
                np.stack((retained_ids, retained_cells), axis=1), axis=0
            )
            counts["retained_pulse_counts"] += np.bincount(
                pairs[:, 1], minlength=size
            ).reshape(grid_spec.shape)
            retained_global_pulses.update(
                (source_index, int(pulse)) for pulse in np.unique(retained_ids)
            )

    assessed_mask = _coverage_mask(effective.known_coverage, grid_spec)
    if assessed_mask.shape != grid_spec.shape:
        raise CoverageError("known coverage mask shape does not match the grid")
    assessed_mask = assessed_mask.astype(bool, copy=False)

    status, reasons = _classify(
        effective,
        assessed_mask,
        counts["raw_point_counts"],
        counts["raw_pulse_counts"],
        counts["retained_point_counts"],
        counts["retained_pulse_counts"],
        counts["filtered_point_counts"],
        counts["invalid_point_counts"],
    )

    return CoverageResult(
        grid=grid_spec,
        config=effective,
        status=status,
        reasons=reasons,
        assessed_mask=assessed_mask,
        input_point_count=input_point_count,
        input_pulse_count=len(global_pulses),
        retained_point_count=retained_point_count,
        retained_pulse_count=len(retained_global_pulses),
        invalid_point_count=invalid_point_count,
        filtered_point_count=filtered_point_count,
        outside_point_count=outside_point_count,
        pulse_identity_methods=identity_methods,
        source_data=normalised,
        **counts,
    )


def build_review_list(sources: Any, **kwargs: Any) -> CoverageResult:
    """Build a missed-scan review result.  Alias of :func:`analyze_coverage`."""
    return analyze_coverage(sources, **kwargs)


def coverage_review(sources: Any, **kwargs: Any) -> CoverageResult:
    return analyze_coverage(sources, **kwargs)


def scan_coverage(sources: Any, **kwargs: Any) -> CoverageResult:
    return analyze_coverage(sources, **kwargs)


def review_list(sources: Any, **kwargs: Any) -> List[CoverageCell]:
    """Shortcut returning only the ordered review rows."""
    return analyze_coverage(sources, **kwargs).review_list()


# ----------------------------------------------------------------------
# Comparison after adding strip(s)
# ----------------------------------------------------------------------
def _config_signature(config: CoverageConfig) -> Dict[str, Any]:
    filters = config.filter_condition
    if isinstance(filters, (str, list, tuple)):
        filters = list(filters)
    elif filters is None:
        filters = None
    else:
        filters = "<callable>"
    return {
        "multi_return": config.multi_return,
        "min_pulses": config.min_pulses,
        "min_points": config.min_points,
        "min_density": config.min_density,
        "filter": filters,
    }


def _result_from(
    value: Any, config: CoverageConfig, *, label: str
) -> CoverageResult:
    if isinstance(value, CoverageResult):
        if value.grid.to_dict() != config.grid.to_dict():
            raise CoverageError(f"{label} result uses a different grid")
        if _config_signature(value.config) != _config_signature(config):
            raise CoverageError(
                f"{label} result was computed with different thresholds or "
                "filter settings"
            )
        return value
    return analyze_coverage(value, config=config)


def _grid_from_union(sources: Sequence[Any], cell_size: float) -> GridSpec:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for source in sources:
        for item in _normalise_sources(source):
            x, y = _xy_columns(item)
            xs.append(x)
            ys.append(y)
    return GridSpec.around_points(
        np.concatenate(xs), np.concatenate(ys), cell_size
    )


def compare_coverage(
    baseline: Any,
    added: Any,
    *,
    combined: Optional[Any] = None,
    grid: Optional[Union[GridSpec, Sequence[float]]] = None,
    cell_size: Optional[float] = None,
    bounds: Optional[Sequence[float]] = None,
    filter_condition: Any = None,
    multi_return: str = "all",
    min_pulses: int = 1,
    min_points: int = 1,
    min_density: Optional[float] = None,
    threshold: Optional[int] = None,
    known_coverage: Any = None,
    pulse_key: Any = "auto",
    gps_time_tolerance: float = 1e-6,
    config: Optional[CoverageConfig] = None,
) -> CoverageComparison:
    """Compare a baseline survey with the survey after adding strip(s).

    ``baseline`` is the original survey, ``added`` contains only the newly
    added strip(s), and ``combined`` is the union of both.  Each argument may
    be a :class:`CoverageResult` or raw source data.  When ``combined`` is
    omitted it is built from the raw baseline and added data; if either side
    is already a :class:`CoverageResult` the raw data cannot be recovered and
    an explicit ``combined`` is required.

    ``newly_covered_cells`` are cells that were not covered by the baseline
    but are covered by the combined survey.  ``repeat_observation_cells`` are
    cells already covered by the baseline that also received observations
    from the added strip.  Their point/pulse totals are reported separately.
    """
    if threshold is not None:
        min_pulses = threshold

    baseline_is_result = isinstance(baseline, CoverageResult)
    added_is_result = isinstance(added, CoverageResult)

    if config is not None:
        effective = config
    elif baseline_is_result:
        effective = baseline.config
    elif added_is_result:
        effective = added.config
    elif combined is not None and isinstance(combined, CoverageResult):
        effective = combined.config
    else:
        if (
            grid is None
            and bounds is None
            and cell_size is not None
            and not isinstance(baseline, CoverageResult)
            and not isinstance(added, CoverageResult)
        ):
            # Without an explicit grid, derive it from baseline + added so
            # the added strip can never fall outside the assessed area.
            grid = _grid_from_union([baseline, added], cell_size)
        baseline_result = analyze_coverage(
            baseline,
            grid=grid,
            cell_size=cell_size,
            bounds=bounds,
            filter_condition=filter_condition,
            multi_return=multi_return,
            min_pulses=min_pulses,
            min_points=min_points,
            min_density=min_density,
            known_coverage=known_coverage,
            pulse_key=pulse_key,
            gps_time_tolerance=gps_time_tolerance,
        )
        effective = baseline_result.config

    baseline_result = _result_from(baseline, effective, label="baseline")
    added_result = _result_from(added, effective, label="added")

    if combined is None:
        baseline_sources = _source_data_of(baseline)
        added_sources = _source_data_of(added)
        if baseline_sources is not None and added_sources is not None:
            combined_result = analyze_coverage(
                list(baseline_sources) + list(added_sources),
                config=effective,
            )
        elif baseline_is_result or added_is_result:
            if baseline_is_result and added_is_result:
                # Without per-pulse membership the exact union cannot be
                # recomputed.  Combining point counts is exact, but pulse
                # support must not be double counted, so use the maximum as
                # a conservative estimate and make the limitation explicit.
                combined_result = _combine_results(baseline_result, added_result)
            else:
                raise CoverageError(
                    "combined data is required when baseline or added is "
                    "already a CoverageResult; pass combined=<raw data>"
                )
        else:
            combined_result = analyze_coverage(
                list(_iter_sources(baseline)) + list(_iter_sources(added)),
                config=effective,
            )
    else:
        combined_result = _result_from(combined, effective, label="combined")

    baseline_covered = baseline_result.covered_cells
    combined_covered = combined_result.covered_cells

    newly_covered = (
        ~baseline_covered & combined_covered & combined_result.assessed_mask
    )
    # A repeat observation needs retained evidence in the added strip; points
    # removed by the filter are not evidence that the area was observed again.
    added_evidence = (
        (added_result.retained_point_counts > 0) & combined_result.assessed_mask
    )
    repeat = baseline_covered & added_evidence
    remaining = (
        ~combined_covered
        & combined_result.assessed_mask
        & np.isin(combined_result.status, CANDIDATE_STATUSES)
    )

    new_points = np.where(
        newly_covered, added_result.retained_point_counts, 0
    )
    new_pulses = np.where(
        newly_covered, added_result.retained_pulse_counts, 0
    )
    repeat_points = np.where(repeat, added_result.retained_point_counts, 0)
    repeat_pulses = np.where(repeat, added_result.retained_pulse_counts, 0)

    return CoverageComparison(
        grid=effective.grid,
        baseline=baseline_result,
        added=added_result,
        combined=combined_result,
        newly_covered_cells=newly_covered,
        repeat_observation_cells=repeat,
        remaining_candidate_cells=remaining,
        added_point_counts=added_result.raw_point_counts,
        added_pulse_counts=added_result.raw_pulse_counts,
        new_coverage_point_counts=new_points,
        repeat_observation_point_counts=repeat_points,
        new_coverage_pulse_counts=new_pulses,
        repeat_observation_pulse_counts=repeat_pulses,
    )


def _iter_sources(value: Any) -> Iterator[Any]:
    if isinstance(value, (list, tuple)):
        yield from value
    else:
        yield value


def _source_data_of(value: Any) -> Optional[List[Any]]:
    if isinstance(value, CoverageResult):
        return value.source_data
    return list(_iter_sources(value))


def _combine_results(
    baseline: CoverageResult, added: CoverageResult
) -> CoverageResult:
    if baseline.grid.to_dict() != added.grid.to_dict():
        raise CoverageError("both results must use the same grid")

    if baseline.source_data is not None and added.source_data is not None:
        config = replace(
            baseline.config,
            known_coverage=None,
        )
        combined_before_mask = analyze_coverage(
            list(baseline.source_data) + list(added.source_data),
            config=config,
        )
        if baseline.config.known_coverage is not None:
            assessed_mask = baseline.assessed_mask.copy()
            status, reasons = _classify(
                baseline.config,
                assessed_mask,
                combined_before_mask.raw_point_counts,
                combined_before_mask.raw_pulse_counts,
                combined_before_mask.retained_point_counts,
                combined_before_mask.retained_pulse_counts,
                combined_before_mask.filtered_point_counts,
                combined_before_mask.invalid_point_counts,
            )
            combined_before_mask.status = status
            combined_before_mask.reasons = reasons
            combined_before_mask.assessed_mask = assessed_mask
        return combined_before_mask

    raw_points = baseline.raw_point_counts + added.raw_point_counts
    retained_points = baseline.retained_point_counts + added.retained_point_counts
    filtered_points = (
        baseline.filtered_point_counts + added.filtered_point_counts
    )
    invalid_points = baseline.invalid_point_counts + added.invalid_point_counts
    # Conservative: a pulse observed by both results is counted once.
    raw_pulses = np.maximum(baseline.raw_pulse_counts, added.raw_pulse_counts)
    retained_pulses = np.maximum(
        baseline.retained_pulse_counts, added.retained_pulse_counts
    )

    status, reasons = _classify(
        baseline.config,
        baseline.assessed_mask,
        raw_points,
        raw_pulses,
        retained_points,
        retained_pulses,
        filtered_points,
        invalid_points,
    )
    return CoverageResult(
        grid=baseline.grid,
        config=baseline.config,
        status=status,
        reasons=reasons,
        raw_point_counts=raw_points,
        raw_pulse_counts=raw_pulses,
        retained_point_counts=retained_points,
        retained_pulse_counts=retained_pulses,
        filtered_point_counts=filtered_points,
        invalid_point_counts=invalid_points,
        assessed_mask=baseline.assessed_mask,
        input_point_count=baseline.input_point_count + added.input_point_count,
        input_pulse_count=baseline.input_pulse_count + added.input_pulse_count,
        retained_point_count=(
            baseline.retained_point_count + added.retained_point_count
        ),
        retained_pulse_count=(
            baseline.retained_pulse_count + added.retained_pulse_count
        ),
        invalid_point_count=(
            baseline.invalid_point_count + added.invalid_point_count
        ),
        filtered_point_count=(
            baseline.filtered_point_count + added.filtered_point_count
        ),
        outside_point_count=(
            baseline.outside_point_count + added.outside_point_count
        ),
        pulse_identity_methods=(
            baseline.pulse_identity_methods + added.pulse_identity_methods
        ),
    )


def compare_strips(baseline: Any, added: Any, **kwargs: Any) -> CoverageComparison:
    """Alias of :func:`compare_coverage`."""
    return compare_coverage(baseline, added, **kwargs)


def coverage_delta(baseline: Any, added: Any, **kwargs: Any) -> CoverageComparison:
    """Alias of :func:`compare_coverage`."""
    return compare_coverage(baseline, added, **kwargs)


__all__ = [
    "CANDIDATE_STATUSES",
    "CellStatus",
    "CoverageCell",
    "CoverageComparison",
    "CoverageConfig",
    "CoverageError",
    "CoverageResult",
    "GridSpec",
    "MULTI_RETURN_MODES",
    "analyze_coverage",
    "build_review_list",
    "compare_coverage",
    "compare_strips",
    "coverage_delta",
    "coverage_review",
    "review_list",
    "scan_coverage",
]
