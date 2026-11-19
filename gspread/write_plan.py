"""
gspread.write_plan
~~~~~~~~~~~~~~~~~~

Plan batch writes with per-cell input semantics and verify the
actual stored type and content after writing.

The Sheets API applies a single ``valueInputOption`` per request, so a
mixed import (e.g. the id ``00123`` that must stay text next to the
amount ``1,234`` that must be parsed using the spreadsheet locale)
cannot be sent as one naive update. This module lets callers declare
the intended semantics of every cell, groups the cells accordingly,
and reads the values back to confirm that each cell was interpreted
as agreed. Cells whose stored result does not match the declared
convention are reported individually in
:class:`WriteReport.mismatches` -- a successful HTTP response alone is
not treated as proof of a correct write.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Sequence

from .exceptions import InvalidInputValue
from .utils import StrEnum, ValueInputOption, ValueRenderOption, rowcol_to_a1

if TYPE_CHECKING:
    from .worksheet import Worksheet


class Expectation(StrEnum):
    """The kind of value a cell is expected to hold after a write."""

    any = "any"
    text = "text"
    number = "number"
    boolean = "boolean"


@dataclass
class PlannedCell:
    """One cell of a write plan.

    :param int row: Row number (1-based).
    :param int col: Column number (1-based).
    :param value: The value to write.
    :param value_input_option: How the value must be interpreted:
        ``ValueInputOption.raw`` stores the value as-is (``"00123"``
        stays the text ``"00123"``), ``ValueInputOption.user_entered``
        lets Sheets parse it as if typed in the UI (``"1,234"`` becomes
        a number according to the spreadsheet locale).
    :param expect: The kind of value the cell must hold after the
        write. Derived from ``value`` when omitted: ``raw`` cells keep
        their natural kind, ``user_entered`` strings default to
        ``Expectation.any`` unless declared otherwise.
    :param expected_value: The exact value expected when reading the
        cell back (unformatted). Defaults to ``value`` for ``raw``
        cells and for escaped text; leave unset for ``user_entered``
        cells when only the resulting type matters.
    :param bool allow_formula: Permit a ``user_entered`` string that
        starts with ``=`` to be sent as a real formula. By default such
        strings are escaped so they are stored as literal text.
    """

    row: int
    col: int
    value: Any
    value_input_option: ValueInputOption = ValueInputOption.raw
    expect: Optional[Expectation] = None
    expected_value: Any = None
    allow_formula: bool = False

    def __post_init__(self) -> None:
        if self.row < 1 or self.col < 1:
            raise InvalidInputValue(
                "row and col must be >= 1, got row={!r} col={!r}".format(
                    self.row, self.col
                )
            )
        if self.value is None:
            raise InvalidInputValue(
                "PlannedCell value must not be None, got cell {!r}".format(self.a1)
            )
        self.value_input_option = ValueInputOption(self.value_input_option)
        if self.expect is not None:
            self.expect = Expectation(self.expect)

        if self.escaped:
            # text that looks like a formula must come back as the
            # exact original string
            if self.expect is None:
                self.expect = Expectation.text
            if self.expected_value is None:
                self.expected_value = self.value

        if self.expect is None:
            self.expect = self._default_expectation()

        if (
            self.expected_value is None
            and self.value_input_option == ValueInputOption.raw
        ):
            self.expected_value = self.value

    def _default_expectation(self) -> Expectation:
        if isinstance(self.value, bool):
            return Expectation.boolean
        if isinstance(self.value, (int, float)):
            return Expectation.number
        if isinstance(self.value, str):
            if self.value_input_option == ValueInputOption.user_entered:
                # the spreadsheet locale decides how the string is parsed
                return Expectation.any
            return Expectation.text
        return Expectation.any

    @property
    def a1(self) -> str:
        """The cell address in A1 notation."""
        return rowcol_to_a1(self.row, self.col)

    @property
    def escaped(self) -> bool:
        """Whether the value must be escaped to stay literal text."""
        return (
            self.value_input_option == ValueInputOption.user_entered
            and isinstance(self.value, str)
            and self.value.startswith("=")
            and not self.allow_formula
        )

    @property
    def payload(self) -> Any:
        """The value actually sent to the API.

        A ``user_entered`` string starting with ``=`` is prefixed with
        an apostrophe so Sheets stores it as text instead of parsing it
        as a formula, unless ``allow_formula`` is set.
        """
        if self.escaped:
            return "'" + self.value
        return self.value


@dataclass
class CellResult:
    """The verified outcome of one planned cell."""

    row: int
    col: int
    a1: str
    value: Any
    value_input_option: ValueInputOption
    actual: Any


@dataclass
class CellMismatch(CellResult):
    """A cell whose stored result does not match the declared convention."""

    expect: Expectation = Expectation.any
    expected_value: Any = None
    reason: str = ""


@dataclass
class WriteReport:
    """The result of applying a write plan.

    ``responses`` holds the raw API responses of the write requests.
    HTTP success only means the requests were accepted; ``matched`` and
    ``mismatches`` reflect the per-cell verification read back from the
    sheet afterwards.
    """

    matched: List[CellResult] = field(default_factory=list)
    mismatches: List[CellMismatch] = field(default_factory=list)
    responses: List[Any] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when every planned cell verified as expected."""
        return not self.mismatches

    @property
    def total(self) -> int:
        """The number of planned cells that were verified."""
        return len(self.matched) + len(self.mismatches)


def _actual_expectation(value: Any) -> Expectation:
    if isinstance(value, bool):
        return Expectation.boolean
    if isinstance(value, (int, float)):
        return Expectation.number
    if isinstance(value, str):
        return Expectation.text
    return Expectation.any


def _content_matches(expected: Any, actual: Any) -> bool:
    # bool is a subclass of int: never equate True with 1
    if isinstance(expected, bool) != isinstance(actual, bool):
        return False
    return bool(expected == actual)


def _verify_cell(cell: PlannedCell, actual: Any) -> Optional[CellMismatch]:
    def mismatch(reason: str) -> CellMismatch:
        return CellMismatch(
            row=cell.row,
            col=cell.col,
            a1=cell.a1,
            value=cell.value,
            value_input_option=cell.value_input_option,
            actual=actual,
            expect=cell.expect or Expectation.any,
            expected_value=cell.expected_value,
            reason=reason,
        )

    if actual is None:
        return mismatch("empty: nothing stored in the cell")

    if cell.expect is not None and cell.expect != Expectation.any:
        if _actual_expectation(actual) != cell.expect:
            return mismatch(
                "type: expected {}, got {}".format(
                    cell.expect, _actual_expectation(actual)
                )
            )

    if cell.expected_value is not None and not _content_matches(
        cell.expected_value, actual
    ):
        return mismatch(
            "content: expected {!r}, got {!r}".format(cell.expected_value, actual)
        )

    return None


def apply_write_plan(
    worksheet: "Worksheet",
    cells: Sequence[PlannedCell],
) -> WriteReport:
    """Writes planned cells and verifies what was actually stored.

    Cells are grouped by their ``value_input_option`` and each group is
    sent with its own write request, so ``raw`` cells are stored as-is
    while ``user_entered`` cells are parsed using the spreadsheet
    locale. Every cell is then read back with
    ``ValueRenderOption.unformatted`` to inspect the stored type and
    content.

    :param worksheet: The :class:`~gspread.worksheet.Worksheet` to write to.
    :param cells: The :class:`PlannedCell` entries to write and verify.
    :returns: A :class:`WriteReport` whose ``mismatches`` lists every
        cell whose stored result does not match the declared
        convention, even though the write requests themselves succeeded.
    """
    cells = list(cells)
    report = WriteReport()
    if not cells:
        return report

    groups = defaultdict(list)  # type: defaultdict[ValueInputOption, List[PlannedCell]]
    for cell in cells:
        groups[cell.value_input_option].append(cell)

    for value_input_option, group in groups.items():
        data = [{"range": cell.a1, "values": [[cell.payload]]} for cell in group]
        response = worksheet.batch_update(data, value_input_option=value_input_option)
        report.responses.append(response)

    value_ranges = worksheet.batch_get(
        [cell.a1 for cell in cells],
        value_render_option=ValueRenderOption.unformatted,
    )

    for cell, value_range in zip(cells, value_ranges):
        actual = value_range[0][0] if value_range and value_range[0] else None
        mismatch = _verify_cell(cell, actual)
        if mismatch is not None:
            report.mismatches.append(mismatch)
        else:
            report.matched.append(
                CellResult(
                    row=cell.row,
                    col=cell.col,
                    a1=cell.a1,
                    value=cell.value,
                    value_input_option=cell.value_input_option,
                    actual=actual,
                )
            )

    return report
