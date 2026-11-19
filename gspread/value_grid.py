"""
gspread.value_grid
~~~~~~~~~~~~~~~~~~

Reconcile values returned by the Sheets API with the write operations
performed through gspread.

A ``values.get`` response omits trailing empty cells and often omits fully
empty trailing rows. A short row therefore only proves that the cells beyond
the last returned value are empty *at the time of the request*; it never
proves that those cells were cleared. Likewise, an empty string in a response
may be a genuinely blank cell, the result of a formula evaluating to an empty
string, or a cell that was explicitly cleared -- the response alone cannot
tell these cases apart.

The objects in this module keep the evidence separate:

* :class:`ValueSnapshot` stores one response together with the requested
  range, the response range, its shape (rows or columns) and the source
  version, if any.
* :class:`WriteEvent` stores a write, a clear or a structural mutation with
  the ranges it actually covered.
* :class:`GridState` reconciles snapshots and write events so that a cell is
  only reported as cleared when a clear was explicitly observed, unobserved
  regions stay unknown, and two responses are only merged when they describe
  the same version of the sheet.

All row and column indices used in this module are zero-based, matching the
GridRange convention of the Sheets API.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .utils import (
    Dimension,
    StrEnum,
    a1_range_to_grid_range,
    get_a1_from_absolute_range,
    rowcol_to_a1,
)


class CellStatus(StrEnum):
    """The evidence held about a single cell."""

    unknown = "UNKNOWN"
    """No value was ever fetched for the cell and no event touched it."""

    empty = "EMPTY"
    """A response or a write showed the cell is currently blank.

    This is a statement about the current value only, not about the reason
    why the cell is blank. An empty formula result and a cell holding an
    empty string both produce this status, not :attr:`cleared`.
    """

    cleared = "CLEARED"
    """A clear operation explicitly covered the cell."""

    value = "VALUE"
    """A response or a write provided a non-empty value for the cell."""


@dataclass(frozen=True)
class GridBounds:
    """A rectangular region of a sheet using zero-based half-open indices.

    ``end_row`` / ``end_col`` may be ``None`` when the underlying range is
    unbounded on that side (e.g. the A1 range ``"A:B"`` or ``"2:2"``).
    """

    start_row: int
    end_row: Optional[int]
    start_col: int
    end_col: Optional[int]

    @classmethod
    def from_a1(cls, name: str) -> "GridBounds":
        """Build bounds from an A1 range, possibly unbounded or prefixed."""
        return grid_bounds_from_a1(name)

    def contains(self, row: int, col: int) -> bool:
        if row < self.start_row or col < self.start_col:
            return False
        if self.end_row is not None and row >= self.end_row:
            return False
        if self.end_col is not None and col >= self.end_col:
            return False
        return True

    def overlaps(self, other: "GridBounds") -> bool:
        rows_overlap = self.start_row < (
            other.end_row if other.end_row is not None else float("inf")
        ) and other.start_row < (
            self.end_row if self.end_row is not None else float("inf")
        )
        cols_overlap = self.start_col < (
            other.end_col if other.end_col is not None else float("inf")
        ) and other.start_col < (
            self.end_col if self.end_col is not None else float("inf")
        )
        return rows_overlap and cols_overlap

    def to_a1(self) -> str:
        """Best-effort A1 rendering. Unbounded sides are omitted."""
        start_label = rowcol_to_a1(self.start_row + 1, self.start_col + 1)

        end_row = self.end_row if self.end_row is not None else self.start_row + 1
        end_col = self.end_col if self.end_col is not None else self.start_col + 1
        end_label = rowcol_to_a1(end_row, end_col)

        if start_label == end_label:
            return start_label
        return "{}:{}".format(start_label, end_label)


@dataclass
class CellRecord:
    """Everything known about a single cell."""

    status: CellStatus
    value: Any = None
    source: Optional[str] = None
    sequence: int = 0


class WriteEventType(StrEnum):
    write = "WRITE"
    clear = "CLEAR"
    structural = "STRUCTURAL"


@dataclass
class WriteEvent:
    """A mutation requested on the sheet.

    ``ranges`` always contains finite bounds for :attr:`WriteEventType.write`
    and :attr:`WriteEventType.clear` events. Structural events may use
    unbounded bounds because inserting or deleting a dimension shifts every
    cell beyond a boundary.
    """

    type: WriteEventType
    ranges: List[GridBounds]
    sequence: int
    detail: Optional[str] = None


@dataclass
class ValueSnapshot:
    """One ``values.get`` response with the evidence needed to interpret it."""

    response_range: str
    requested_range: Optional[str]
    major_dimension: str
    values: List[List[Any]]
    bounds: GridBounds
    sequence: int
    version: Optional[Any] = None

    @classmethod
    def from_response(
        cls,
        response: Dict[str, Any],
        requested_range: Optional[str],
        sequence: int,
    ) -> "ValueSnapshot":
        """Build a snapshot from a raw Sheets API response body.

        A response without a ``values`` key describes a range that currently
        holds no data at all; every cell of its bounds stays unknown.
        """
        response_range = response["range"]
        major_dimension = response.get(
            "majorDimension", str(Dimension.rows)
        )
        values = response.get("values") or []
        bounds = grid_bounds_from_a1(get_a1_from_absolute_range(response_range))
        version = response.get("version", response.get("revision"))

        return cls(
            response_range=response_range,
            requested_range=requested_range,
            major_dimension=str(major_dimension),
            values=values,
            bounds=bounds,
            sequence=sequence,
            version=version,
        )

    @property
    def observed_width(self) -> int:
        """Number of columns (rows mode) or rows (columns mode) observed."""
        return max((len(row) for row in self.values), default=0)


def grid_bounds_from_a1(name: str) -> GridBounds:
    """Convert an A1 range, possibly unbounded or sheet prefixed, into bounds."""
    grid_range = a1_range_to_grid_range(get_a1_from_absolute_range(name))
    return GridBounds(
        start_row=grid_range.get("startRowIndex", 0),
        end_row=grid_range.get("endRowIndex"),
        start_col=grid_range.get("startColumnIndex", 0),
        end_col=grid_range.get("endColumnIndex"),
    )


def _as_bounds(ranges: Union[str, GridBounds, Sequence[Union[str, GridBounds]]]) -> List[GridBounds]:
    if isinstance(ranges, (str, GridBounds)):
        ranges = [ranges]
    return [
        bounds if isinstance(bounds, GridBounds) else grid_bounds_from_a1(bounds)
        for bounds in ranges
    ]


class GridState:
    """Reconciles fetched snapshots and locally issued write events.

    The state is evidence based:

    * a short row in a response only fills the cells it actually returns;
      the omitted trailing cells remain whatever they were before;
    * a cell becomes :attr:`CellStatus.cleared` only through an explicit
      clear event, never through an empty response alone;
    * structural mutations (insert/delete/resize/sort) invalidate the regions
      whose coordinates may have shifted, returning them to unknown;
    * snapshots carrying a version are only merged against state describing
      the same version, otherwise the conflicting region is reported as
      undetermined instead of overwriting older values.
    """

    def __init__(self) -> None:
        self._cells: Dict[Tuple[int, int], CellRecord] = {}
        self._events: List[WriteEvent] = []
        self._snapshots: List[ValueSnapshot] = []
        self._undetermined: List[GridBounds] = []
        self._version: Optional[Any] = None
        self._sequence = 0

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    @property
    def version(self) -> Optional[Any]:
        """Version of the sheet the current evidence belongs to, if known."""
        return self._version

    @property
    def undetermined_ranges(self) -> List[GridBounds]:
        """Regions whose value could not be settled due to a version clash."""
        return list(self._undetermined)

    @property
    def events(self) -> List[WriteEvent]:
        return list(self._events)

    @property
    def snapshots(self) -> List[ValueSnapshot]:
        return list(self._snapshots)

    def status_of(self, row: int, col: int) -> CellStatus:
        """Return the :class:`CellStatus` of a zero-based cell coordinate."""
        record = self._cells.get((row, col))
        return record.status if record is not None else CellStatus.unknown

    def value_of(self, row: int, col: int) -> Any:
        """Return the known value of a zero-based cell coordinate.

        Returns ``None`` for empty, cleared and unknown cells. Use
        :meth:`status_of` to distinguish those cases.
        """
        record = self._cells.get((row, col))
        return record.value if record is not None else None

    def is_cleared(self, row: int, col: int) -> bool:
        """Return ``True`` only when a clear explicitly covered the cell."""
        return self.status_of(row, col) is CellStatus.cleared

    def known_cells(self) -> Dict[Tuple[int, int], CellRecord]:
        """Shallow copy of every cell for which evidence exists."""
        return dict(self._cells)

    def record_snapshot(
        self,
        response: Dict[str, Any],
        requested_range: Optional[str] = None,
    ) -> ValueSnapshot:
        """Apply a raw ``values.get`` response body to the state."""
        snapshot = ValueSnapshot.from_response(
            response, requested_range, self._next_sequence()
        )

        if (
            snapshot.version is not None
            and self._version is not None
            and snapshot.version != self._version
        ):
            # The sheet changed outside the evidence collected so far. Flag
            # this snapshot's range rather than letting its short rows erase
            # older values, then adopt the new version so that further
            # snapshots agreeing on it are applied normally.
            self._undetermined.append(snapshot.bounds)
            self._version = snapshot.version
            self._snapshots.append(snapshot)
            return snapshot

        if snapshot.version is not None:
            self._version = snapshot.version

        self._apply_snapshot(snapshot)
        self._snapshots.append(snapshot)
        return snapshot

    def _apply_snapshot(self, snapshot: ValueSnapshot) -> None:
        bounds = snapshot.bounds
        columns_mode = snapshot.major_dimension == str(Dimension.cols)

        for outer_index, row in enumerate(snapshot.values):
            # Only cells the response actually enumerates become evidence.
            # An empty inner row ([]) carries no cell entries and a short row
            # stops at its last value; omitted trailing cells stay unknown.
            for inner_index, raw_value in enumerate(row):
                if columns_mode:
                    coordinate = (
                        bounds.start_row + inner_index,
                        bounds.start_col + outer_index,
                    )
                else:
                    coordinate = (
                        bounds.start_row + outer_index,
                        bounds.start_col + inner_index,
                    )
                self._observe_cell(
                    coordinate[0],
                    coordinate[1],
                    raw_value,
                    snapshot.response_range,
                    snapshot.sequence,
                )

    def _observe_cell(
        self,
        row: int,
        col: int,
        raw_value: Any,
        source: str,
        sequence: int,
    ) -> None:
        existing = self._cells.get((row, col))
        if existing is not None and existing.sequence > sequence:
            return

        if (
            existing is not None
            and existing.status is CellStatus.cleared
            and raw_value == ""
        ):
            return

        status = CellStatus.empty if raw_value == "" else CellStatus.value
        self._cells[(row, col)] = CellRecord(
            status=status,
            value=None if status is CellStatus.empty else raw_value,
            source=source,
            sequence=sequence,
        )

    def record_write(
        self,
        ranges: Union[str, GridBounds, Sequence[Union[str, GridBounds]]],
        values: Optional[List[List[Any]]] = None,
        major_dimension: Union[str, Dimension] = Dimension.rows,
        detail: Optional[str] = None,
    ) -> WriteEvent:
        """Record a value write, optionally with the matrix that was sent.

        Writing an empty string is evidence the cell is currently blank, not
        evidence that it was cleared.
        """
        bounds_list = _as_bounds(ranges)
        event = WriteEvent(
            type=WriteEventType.write,
            ranges=bounds_list,
            sequence=self._next_sequence(),
            detail=detail,
        )
        self._events.append(event)

        if values:
            self._apply_values(bounds_list[0], values, str(major_dimension), event)
        return event

    def record_clear(
        self,
        ranges: Union[str, GridBounds, Sequence[Union[str, GridBounds]]],
        detail: Optional[str] = None,
    ) -> WriteEvent:
        """Record an explicit clear of one or more finite ranges."""
        bounds_list = _as_bounds(ranges)
        for bounds in bounds_list:
            if bounds.end_row is None or bounds.end_col is None:
                raise ValueError(
                    "a clear range must be finite, got: {}".format(bounds)
                )

        event = WriteEvent(
            type=WriteEventType.clear,
            ranges=bounds_list,
            sequence=self._next_sequence(),
            detail=detail,
        )
        self._events.append(event)

        for bounds in bounds_list:
            for row in range(bounds.start_row, bounds.end_row):
                for col in range(bounds.start_col, bounds.end_col):
                    self._cells[(row, col)] = CellRecord(
                        status=CellStatus.cleared,
                        source=bounds.to_a1(),
                        sequence=event.sequence,
                    )
        return event

    def record_structural(
        self,
        ranges: Union[str, GridBounds, Sequence[Union[str, GridBounds]]],
        detail: Optional[str] = None,
    ) -> WriteEvent:
        """Record a mutation that shifts or drops cells.

        Covered cells lose their cached evidence and return to unknown.
        """
        bounds_list = _as_bounds(ranges)
        event = WriteEvent(
            type=WriteEventType.structural,
            ranges=bounds_list,
            sequence=self._next_sequence(),
            detail=detail,
        )
        self._events.append(event)

        for coordinate in list(self._cells):
            if any(bounds.contains(*coordinate) for bounds in bounds_list):
                del self._cells[coordinate]
        return event

    def _apply_values(
        self,
        bounds: GridBounds,
        values: List[List[Any]],
        major_dimension: str,
        event: WriteEvent,
    ) -> None:
        for outer_index, row in enumerate(values):
            for inner_index, raw_value in enumerate(row):
                if major_dimension == str(Dimension.cols):
                    row_index = bounds.start_row + inner_index
                    col_index = bounds.start_col + outer_index
                else:
                    row_index = bounds.start_row + outer_index
                    col_index = bounds.start_col + inner_index

                status = CellStatus.empty if raw_value == "" else CellStatus.value
                self._cells[(row_index, col_index)] = CellRecord(
                    status=status,
                    value=None if status is CellStatus.empty else raw_value,
                    source=event.detail or bounds.to_a1(),
                    sequence=event.sequence,
                )

    @classmethod
    def merge(cls, left: "GridState", right: "GridState") -> "GridState":
        """Merge two states, refusing to let a mismatched version erase data.

        When both states know their version and they differ, the snapshot
        ranges of ``right`` are reported as undetermined and its cell
        observations are discarded: its short rows must not clear values that
        only ``left`` fetched. Events and snapshots are kept in the history
        either way.
        """
        merged = cls()
        sequence_offset = left._sequence

        merged._cells = {
            coordinate: CellRecord(**record.__dict__)
            for coordinate, record in left._cells.items()
        }
        merged._events = list(left._events)
        merged._snapshots = list(left._snapshots)
        merged._undetermined = list(left._undetermined)
        merged._version = left._version
        merged._sequence = left._sequence + right._sequence

        version_clash = (
            left._version is not None
            and right._version is not None
            and left._version != right._version
        )

        for event in right._events:
            shifted = WriteEvent(
                type=event.type,
                ranges=list(event.ranges),
                sequence=event.sequence + sequence_offset,
                detail=event.detail,
            )
            merged._events.append(shifted)

        merged._snapshots.extend(right._snapshots)

        if version_clash:
            merged._undetermined.extend(
                snapshot.bounds for snapshot in right._snapshots
            )
            return merged

        if right._version is not None:
            merged._version = right._version

        for coordinate, record in right._cells.items():
            shifted = CellRecord(
                status=record.status,
                value=record.value,
                source=record.source,
                sequence=record.sequence + sequence_offset,
            )
            existing = merged._cells.get(coordinate)
            if existing is None or shifted.sequence >= existing.sequence:
                if (
                    existing is not None
                    and existing.status is CellStatus.cleared
                    and shifted.status is CellStatus.empty
                ):
                    continue
                merged._cells[coordinate] = shifted

        merged._undetermined.extend(right._undetermined)
        return merged
