"""
gspread.merged
~~~~~~~~~~~~~~

Merged-cell aware view of a value grid.

When a quotation sheet merges one amount cell across several rows,
expanding the merged value into every covered row makes a total count
the amount once per row. This module keeps the three kinds of values
apart:

* the *master* cell actually holds the value,
* *inherited* cells only display the master value,
* every row keeps a reference (``source``) back to the concrete cell
  the value comes from,

and aggregates at the business granularity the caller explicitly
confirms, so a merged amount is counted once per merge region instead
of once per covered row.

"""

from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Optional, Tuple

from .utils import StrEnum, rowcol_to_a1


class CellRole(StrEnum):
    """Role of a cell with respect to merge regions."""

    master = "master"
    """Top-left cell of a merge region. Actually holds the value."""

    inherited = "inherited"
    """Covered by a merge region. Only displays the master value."""

    plain = "plain"
    """Not part of any merge region. Holds its own value."""


class MergeGranularity(StrEnum):
    """Granularity used to aggregate a column of a merged grid.

    The caller must confirm the business granularity explicitly:
    ``merged`` counts one entry per merge region (the business fact),
    ``rows`` counts one entry per grid row (the displayed expansion).
    """

    merged = "merged"
    rows = "rows"


class MergedCell(NamedTuple):
    """One cell of a :class:`MergedGrid`.

    ``value``
        The display value of the cell. For an inherited cell this is
        the master cell's value, which may be empty.

    ``role``
        A :class:`CellRole` telling whether the cell actually holds the
        value (``master``), only displays it (``inherited``) or is not
        merged at all (``plain``).

    ``source``
        A1 notation of the concrete cell the value comes from. For
        master and plain cells this is the cell itself; for inherited
        cells it is the master cell of the merge region.
    """

    value: Any
    role: CellRole
    source: str


MergeRange = Tuple[int, int, int, int]
"""Normalized merge region: (startRowIndex, endRowIndex, startColumnIndex, endColumnIndex)."""


class MergedGrid:
    """A merged-cell aware view over a fetched grid of values.

    Rows are computed lazily and cached. When merge regions change,
    :meth:`update_merges` recomputes only the rows whose coverage
    actually changed; every other cached row is left untouched.

    Inherited cells always read the current master cell value. An empty
    master yields empty inherited cells and is never backfilled from a
    neighbouring cell or a previously displayed value.

    :param values: matrix of values as returned by the API. May be ragged.
    :param merges: merge regions as returned by the API, each a mapping
        with ``startRowIndex``, ``endRowIndex``, ``startColumnIndex``
        and ``endColumnIndex`` keys (end indices exclusive).
    :param start_row_index: index of the first row of ``values`` in the
        worksheet, e.g. 2 if the values start at row 3.
    :param start_col_index: index of the first column of ``values`` in
        the worksheet, e.g. 2 if the values start at column C.

    Example::

        >>> values = [["item", "amount"], ["a", 300], ["b", None]]
        >>> merges = [
        ...     {
        ...         "startRowIndex": 1,
        ...         "endRowIndex": 3,
        ...         "startColumnIndex": 1,
        ...         "endColumnIndex": 2,
        ...     }
        ... ]
        >>> grid = MergedGrid(values, merges)
        >>> grid.cell(2, 1)
        MergedCell(value=300, role=<CellRole.inherited: 'inherited'>, source='B2')
        >>> grid.sum_column(1, granularity=MergeGranularity.merged)
        300
    """

    def __init__(
        self,
        values: List[List[Any]],
        merges: Iterable[Mapping[str, int]] = (),
        start_row_index: int = 0,
        start_col_index: int = 0,
    ) -> None:
        self._values = values
        self._start_row_index = start_row_index
        self._start_col_index = start_col_index
        self._row_count = len(values)
        self._col_count = max((len(row) for row in values), default=0)
        self._row_cache: Dict[int, Tuple[MergedCell, ...]] = {}
        self._merges: List[MergeRange] = []
        self._coverage: Dict[int, Tuple[Optional[Tuple[int, int]], ...]] = {}
        self.update_merges(merges)

    @classmethod
    def from_metadata(
        cls,
        worksheet_metadata: Mapping[str, Any],
        values: List[List[Any]],
        start_row_index: int = 0,
        start_col_index: int = 0,
    ) -> "MergedGrid":
        """Build a grid from worksheet metadata as returned by the API.

        :param worksheet_metadata: metadata of the worksheet, should
            have a ``merges`` key.
        :param values: matrix of values as returned by the API.
        :param start_row_index: index of the first row of ``values``.
        :param start_col_index: index of the first column of ``values``.
        """
        return cls(
            values,
            merges=worksheet_metadata.get("merges", []),
            start_row_index=start_row_index,
            start_col_index=start_col_index,
        )

    @property
    def row_count(self) -> int:
        """Number of rows in the grid."""
        return self._row_count

    @property
    def col_count(self) -> int:
        """Number of columns in the grid."""
        return self._col_count

    def row(self, row_index: int) -> Tuple[MergedCell, ...]:
        """Return the records of one row of the grid.

        :param row_index: zero-based row index within the grid.
        """
        if not 0 <= row_index < self._row_count:
            raise IndexError("row index out of range: {}".format(row_index))
        if row_index not in self._row_cache:
            self._row_cache[row_index] = self._build_row(row_index)
        return self._row_cache[row_index]

    def cell(self, row_index: int, col_index: int) -> MergedCell:
        """Return the record of one cell of the grid.

        :param row_index: zero-based row index within the grid.
        :param col_index: zero-based column index within the grid.
        """
        if not 0 <= col_index < self._col_count:
            raise IndexError("column index out of range: {}".format(col_index))
        return self.row(row_index)[col_index]

    def update_merges(self, merges: Iterable[Mapping[str, int]]) -> List[int]:
        """Replace the merge regions and recompute only affected rows.

        Rows whose merge coverage is identical before and after the
        change keep their cached records; only rows that entered, left
        or moved between merge regions are recomputed.

        :param merges: merge regions as returned by the API.
        :returns: sorted zero-based row indices of the recomputed rows.
        """
        normalized = self._normalize_merges(merges)
        new_coverage = self._coverage_map(normalized)
        affected = [
            abs_row
            for abs_row in set(self._coverage) | set(new_coverage)
            if self._coverage.get(abs_row) != new_coverage.get(abs_row)
        ]
        for abs_row in affected:
            self._row_cache.pop(abs_row - self._start_row_index, None)
        self._merges = normalized
        self._coverage = new_coverage
        return sorted(abs_row - self._start_row_index for abs_row in affected)
    def column_values(
        self,
        col_index: int,
        *,
        granularity: MergeGranularity,
    ) -> List[Any]:
        """Return the values of one column at the confirmed granularity.

        With ``MergeGranularity.merged`` each merge region contributes
        its master value once; inherited cells are skipped. With
        ``MergeGranularity.rows`` every row contributes its display
        value, expanding merged values over the covered rows.

        :param col_index: zero-based column index within the grid.
        :param granularity: the business granularity confirmed by the
            caller, a :class:`MergeGranularity`.
        """
        granularity = MergeGranularity(granularity)
        values = []
        for row_index in range(self._row_count):
            record = self.cell(row_index, col_index)
            if (
                granularity is MergeGranularity.merged
                and record.role is CellRole.inherited
            ):
                continue
            values.append(record.value)
        return values

    def sum_column(
        self,
        col_index: int,
        *,
        granularity: MergeGranularity,
    ) -> Any:
        """Sum the numeric values of one column at the confirmed granularity.

        Non-numeric values (headers, blanks, text) are ignored.

        :param col_index: zero-based column index within the grid.
        :param granularity: the business granularity confirmed by the
            caller, a :class:`MergeGranularity`.
        """
        return sum(
            value
            for value in self.column_values(col_index, granularity=granularity)
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        )

    def _raw_at(self, abs_row: int, abs_col: int) -> Any:
        local_row = abs_row - self._start_row_index
        local_col = abs_col - self._start_col_index
        if 0 <= local_row < self._row_count:
            row = self._values[local_row]
            if 0 <= local_col < len(row):
                return row[local_col]
        return None

    def _normalize_merges(self, merges: Iterable[Mapping[str, int]]) -> List[MergeRange]:
        abs_row_end = self._start_row_index + self._row_count
        abs_col_end = self._start_col_index + self._col_count
        normalized = []
        for merge in merges:
            start_row = merge["startRowIndex"]
            end_row = merge["endRowIndex"]
            start_col = merge["startColumnIndex"]
            end_col = merge["endColumnIndex"]
            # the master cell must be inside the grid, otherwise its
            # value is unknown and there is nothing to inherit
            if not (
                self._start_row_index <= start_row < abs_row_end
                and self._start_col_index <= start_col < abs_col_end
            ):
                continue
            normalized.append(
                (
                    start_row,
                    min(end_row, abs_row_end),
                    start_col,
                    min(end_col, abs_col_end),
                )
            )
        return normalized

    def _coverage_map(
        self, merges: List[MergeRange]
    ) -> Dict[int, Tuple[Optional[Tuple[int, int]], ...]]:
        # a row's records only depend on the master coordinates behind
        # each of its cells, so that is the signature used to decide
        # whether a row is affected by a merge change
        coverage: Dict[int, List[Optional[Tuple[int, int]]]] = {}
        for merge in merges:
            master = (merge[0], merge[2])
            for abs_row in range(merge[0], merge[1]):
                row_map = coverage.setdefault(abs_row, [None] * self._col_count)
                for abs_col in range(merge[2], merge[3]):
                    row_map[abs_col - self._start_col_index] = master
        return {abs_row: tuple(row_map) for abs_row, row_map in coverage.items()}

    def _merge_covering(self, abs_row: int, abs_col: int) -> Optional[MergeRange]:
        for merge in self._merges:
            if merge[0] <= abs_row < merge[1] and merge[2] <= abs_col < merge[3]:
                return merge
        return None

    def _build_row(self, row_index: int) -> Tuple[MergedCell, ...]:
        abs_row = self._start_row_index + row_index
        records = []
        for local_col in range(self._col_count):
            abs_col = self._start_col_index + local_col
            address = rowcol_to_a1(abs_row + 1, abs_col + 1)
            merge = self._merge_covering(abs_row, abs_col)
            if merge is None:
                records.append(
                    MergedCell(
                        value=self._raw_at(abs_row, abs_col),
                        role=CellRole.plain,
                        source=address,
                    )
                )
                continue
            master_row, master_col = merge[0], merge[2]
            if (abs_row, abs_col) == (master_row, master_col):
                records.append(
                    MergedCell(
                        value=self._raw_at(abs_row, abs_col),
                        role=CellRole.master,
                        source=address,
                    )
                )
            else:
                records.append(
                    MergedCell(
                        value=self._raw_at(master_row, master_col),
                        role=CellRole.inherited,
                        source=rowcol_to_a1(master_row + 1, master_col + 1),
                    )
                )
        return tuple(records)
