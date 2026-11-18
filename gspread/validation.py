"""
gspread.validation
~~~~~~~~~~~~~~~~~~

Offline auditing of spreadsheet data-validation rules.

The module is intentionally independent of the Sheets API.  A caller supplies
the rules that are allowed for a batch, the option sources locked for that
batch, and the current cell values.  The audit then reports, for every cell:

* the effective constraint (the options or bounds that really apply),
* the offending value, if any,
* whether the cell is rejected, only warned, or pending review.

Two details are important for collection sheets whose dropdowns live in
another table:

``OptionSource`` / ``OptionCategory``
    A dropdown source can be split into categories and every category can
    point at a different range.  Categories carry their own revision and
    completeness flag.

``CellRule.list_range`` / ``CellRule.relative``
    A list range written relative to a cell (for example ``$A2:$A2``) is
    resolved from the cell where the rule was originally written, so expanding
    a rule over a column changes the row exactly as a copied spreadsheet
    formula would.

Incomplete sources never produce a rejected cell.  Values that cannot be
confirmed are reported as :attr:`CellStatus.pending` ("待核").  Revisions are
tracked per category, so revising one category only re-evaluates the cells that
actually used that category.

The module can be used as a command line tool::

    python -m gspread.validation batch.json --format json

Input JSON has three top-level keys::

    {
      "name": "November intake",
      "sources": [...],
      "rules": [...],
      "values": {"Sheet1!D2": "Gold", ...}
    }

See :meth:`ValidationBatch.from_dict` for the accepted source and rule shapes.
"""

import argparse
import csv
import io
import json
import re
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    TextIO,
    Tuple,
    Union,
)

from .exceptions import IncorrectCellLabel
from .utils import rowcol_to_a1

__all__ = [
    "A1Range",
    "CellResult",
    "CellRule",
    "CellStatus",
    "OptionCategory",
    "OptionSource",
    "RuleMode",
    "RuleType",
    "ValidationAudit",
    "ValidationBatch",
    "audit_from_dict",
    "load_batch",
    "main",
]


class RuleMode(str, Enum):
    """What happens when a value violates a rule."""

    reject = "reject"
    warn = "warn"


class CellStatus(str, Enum):
    """The audit outcome for one cell."""

    valid = "valid"
    invalid = "invalid"
    warned = "warned"
    pending = "pending"
    empty = "empty"


class RuleType(str, Enum):
    """The validation-rule families understood by this module."""

    list = "list"
    number = "number"
    text = "text"
    date = "date"
    boolean = "boolean"
    blank = "blank"
    custom_formula = "custom_formula"


_TOKEN_RE = re.compile(r"^(\$?)([A-Za-z]*)(\$?)(\d*)$")
_UNRESOLVED_SOURCE = "\x00unresolved-range"


def _unresolved_key(list_range: Optional[str]) -> Optional[str]:
    return list_range


def _column_to_index(label: str) -> int:
    index = 0
    for char in label.upper():
        if not ("A" <= char <= "Z"):
            raise IncorrectCellLabel(label)
        index = index * 26 + (ord(char) - 64)
    if index < 1:
        raise IncorrectCellLabel(label)
    return index


def _index_to_column(index: int) -> str:
    label = ""
    while index:
        index, remainder = divmod(index, 26)
        if remainder == 0:
            remainder = 26
            index -= 1
        label = chr(remainder + 64) + label
    return label


def _unquote_sheet(sheet: str) -> str:
    sheet = sheet.strip()
    if len(sheet) >= 2 and sheet[0] == "'" and sheet[-1] == "'":
        return sheet[1:-1].replace("''", "'")
    return sheet


def _sheet_needs_quotes(sheet: str) -> bool:
    return (
        not sheet
        or any(char in sheet for char in " !'")
        or sheet[0].isdigit()
    )


@dataclass(frozen=True)
class _Token:
    row: Optional[int]
    col: Optional[int]
    row_absolute: bool
    col_absolute: bool


def _parse_token(token: str) -> _Token:
    token = token.strip()
    match = _TOKEN_RE.match(token)
    if match is None:
        raise IncorrectCellLabel(token)
    col_absolute, col_label, row_absolute, row_label = match.groups()
    if not col_label and not row_label:
        raise IncorrectCellLabel(token)
    col = _column_to_index(col_label) if col_label else None
    row = int(row_label) if row_label else None
    if row is not None and row < 1:
        raise IncorrectCellLabel(token)
    return _Token(
        row=row,
        col=col,
        row_absolute=bool(row_absolute),
        col_absolute=bool(col_absolute),
    )


class A1Range:
    """A rectangular A1 range, optionally unbounded on either axis.

    ``Sheet1!B2:D4``, ``B2:D4``, ``B2:B`` and ``B:B`` are all accepted.
    Absolute markers (``$``) are preserved and used when a relative range is
    resolved for a concrete cell.
    """

    __slots__ = (
        "start_row",
        "start_col",
        "end_row",
        "end_col",
        "sheet",
        "start_row_absolute",
        "start_col_absolute",
        "end_row_absolute",
        "end_col_absolute",
    )

    def __init__(
        self,
        start_row: Optional[int],
        start_col: Optional[int],
        end_row: Optional[int],
        end_col: Optional[int],
        sheet: Optional[str] = None,
        start_row_absolute: bool = False,
        start_col_absolute: bool = False,
        end_row_absolute: bool = False,
        end_col_absolute: bool = False,
    ) -> None:
        for value in (start_row, start_col, end_row, end_col):
            if value is not None and value < 1:
                raise ValueError("A1 range bounds are one-based")
        if start_row is not None and end_row is not None and start_row > end_row:
            start_row, end_row = end_row, start_row
            start_row_absolute, end_row_absolute = (
                end_row_absolute,
                start_row_absolute,
            )
        if start_col is not None and end_col is not None and start_col > end_col:
            start_col, end_col = end_col, start_col
            start_col_absolute, end_col_absolute = (
                end_col_absolute,
                start_col_absolute,
            )
        self.start_row = start_row
        self.start_col = start_col
        self.end_row = end_row
        self.end_col = end_col
        self.sheet = sheet
        self.start_row_absolute = start_row_absolute
        self.start_col_absolute = start_col_absolute
        self.end_row_absolute = end_row_absolute
        self.end_col_absolute = end_col_absolute

    @classmethod
    def parse(cls, value: Union[str, "A1Range"]) -> "A1Range":
        if isinstance(value, A1Range):
            return value
        if not isinstance(value, str) or not value.strip():
            raise ValueError("range must be a non-empty A1 string")
        body = value.strip()
        sheet: Optional[str] = None
        if "!" in body:
            sheet_part, _, body = body.rpartition("!")
            sheet = _unquote_sheet(sheet_part)
        start_label, separator, end_label = body.partition(":")
        start = _parse_token(start_label)
        if separator:
            if not end_label.strip():
                raise ValueError("range {!r} has an empty end".format(value))
            end = _parse_token(end_label)
        else:
            end = start
        return cls(
            start.row,
            start.col,
            end.row,
            end.col,
            sheet,
            start.row_absolute,
            start.col_absolute,
            end.row_absolute,
            end.col_absolute,
        )

    @property
    def is_single_cell(self) -> bool:
        return (
            self.start_row is not None
            and self.start_col is not None
            and self.end_row == self.start_row
            and self.end_col == self.start_col
        )

    def contains(self, row: int, col: int, sheet: Optional[str] = None) -> bool:
        if self.sheet is not None and sheet is not None and self.sheet != sheet:
            return False
        if self.start_row is not None and row < self.start_row:
            return False
        if self.end_row is not None and row > self.end_row:
            return False
        if self.start_col is not None and col < self.start_col:
            return False
        if self.end_col is not None and col > self.end_col:
            return False
        return True

    def iter_cells(self) -> Iterator[Tuple[int, int]]:
        if None in (self.start_row, self.start_col, self.end_row, self.end_col):
            raise ValueError("cannot iterate an unbounded range: {!r}".format(self))
        assert self.start_row is not None
        assert self.start_col is not None
        assert self.end_row is not None
        assert self.end_col is not None
        for row in range(self.start_row, self.end_row + 1):
            for col in range(self.start_col, self.end_col + 1):
                yield row, col

    def resolve(
        self,
        origin_row: int,
        origin_col: int,
        target_row: int,
        target_col: int,
        *,
        relative: bool = True,
    ) -> "A1Range":
        """Resolve this range as a copied spreadsheet reference.

        Relative components keep their offset from ``origin``; absolute
        components (``$``) stay fixed.  When ``relative`` is false the range
        is returned unchanged.
        """
        if not relative:
            return self

        def resolve_component(
            value: Optional[int],
            absolute: bool,
            origin: int,
            target: int,
        ) -> Optional[int]:
            if value is None or absolute:
                return value
            return value + (target - origin)

        return A1Range(
            resolve_component(
                self.start_row, self.start_row_absolute, origin_row, target_row
            ),
            resolve_component(
                self.start_col, self.start_col_absolute, origin_col, target_col
            ),
            resolve_component(
                self.end_row, self.end_row_absolute, origin_row, target_row
            ),
            resolve_component(
                self.end_col, self.end_col_absolute, origin_col, target_col
            ),
            self.sheet,
            self.start_row_absolute,
            self.start_col_absolute,
            self.end_row_absolute,
            self.end_col_absolute,
        )

    def offset(self, row_delta: int, col_delta: int) -> "A1Range":
        """Shift every bounded component; used for relative source ranges."""
        return A1Range(
            self.start_row + row_delta if self.start_row is not None else None,
            self.start_col + col_delta if self.start_col is not None else None,
            self.end_row + row_delta if self.end_row is not None else None,
            self.end_col + col_delta if self.end_col is not None else None,
            self.sheet,
            self.start_row_absolute,
            self.start_col_absolute,
            self.end_row_absolute,
            self.end_col_absolute,
        )

    def with_sheet(self, sheet: Optional[str]) -> "A1Range":
        return A1Range(
            self.start_row,
            self.start_col,
            self.end_row,
            self.end_col,
            sheet,
            self.start_row_absolute,
            self.start_col_absolute,
            self.end_row_absolute,
            self.end_col_absolute,
        )

    def key(self) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[int], Optional[int]]:
        """Identity used to match a rule range with a locked source range."""
        return (
            self.sheet,
            self.start_row,
            self.start_col,
            self.end_row,
            self.end_col,
        )

    def body(self) -> str:
        return str(self.without_sheet())

    def without_sheet(self) -> "A1Range":
        return self.with_sheet(None)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, A1Range) and self.key() == other.key()

    def __hash__(self) -> int:
        return hash(self.key())

    def __str__(self) -> str:
        start = self._format_endpoint(
            self.start_col,
            self.start_row,
            self.start_col_absolute,
            self.start_row_absolute,
        )
        end = self._format_endpoint(
            self.end_col,
            self.end_row,
            self.end_col_absolute,
            self.end_row_absolute,
        )
        if (
            self.start_row is not None
            and self.start_col is not None
            and self.end_row == self.start_row
            and self.end_col == self.start_col
        ):
            body = start
        else:
            body = "{}:{}".format(start, end)
        if self.sheet is None:
            return body
        if _sheet_needs_quotes(self.sheet):
            escaped = self.sheet.replace("'", "''")
            return "'{}'!{}".format(escaped, body)
        return "{}!{}".format(self.sheet, body)

    @staticmethod
    def _format_endpoint(
        col: Optional[int],
        row: Optional[int],
        col_absolute: bool,
        row_absolute: bool,
    ) -> str:
        result = ""
        if col is not None:
            result += ("$" if col_absolute else "") + _index_to_column(col)
        if row is not None:
            result += ("$" if row_absolute else "") + str(row)
        return result

    def __repr__(self) -> str:
        return "A1Range({!r})".format(str(self))


@dataclass
class OptionCategory:
    """One category of a dropdown source and the range locked for it."""

    name: Optional[str]
    values: List[Any] = field(default_factory=list)
    complete: bool = True
    range: Optional[A1Range] = None
    relative_range: bool = False
    revision: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.range, str):
            self.range = A1Range.parse(self.range)
        self.values = [value for value in self.values if value not in ("", None)]

    @classmethod
    def from_any(
        cls,
        name: Optional[str],
        value: Any,
        *,
        source_complete: bool = True,
        source_revision: int = 0,
        source_relative: bool = False,
    ) -> "OptionCategory":
        if isinstance(value, OptionCategory):
            return value
        if isinstance(value, Mapping):
            return cls(
                name=name,
                values=list(value.get("values", [])),
                complete=bool(value.get("complete", source_complete)),
                range=(
                    A1Range.parse(value["range"])
                    if value.get("range") is not None
                    else None
                ),
                relative_range=bool(
                    value.get("relative_range", source_relative)
                ),
                revision=int(value.get("revision", source_revision)),
            )
        return cls(
            name=name,
            values=list(value or []),
            complete=source_complete,
            revision=source_revision,
            relative_range=source_relative,
        )

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "values": list(self.values),
            "complete": self.complete,
            "revision": self.revision,
        }
        if self.range is not None:
            result["range"] = str(self.range)
        if self.relative_range:
            result["relative_range"] = True
        return result

    def __repr__(self) -> str:
        return "OptionCategory({!r}, values={}, complete={!r})".format(
            self.name, len(self.values), self.complete
        )


class OptionSource:
    """The dropdown source locked for a batch.

    ``values``/``range`` describe the default category.  ``categories`` maps a
    category name to its own values, range, completeness and revision.  When
    the source is incomplete, every category is treated as incomplete too.

    ``relative_range`` marks a source whose range is relative to the rule's
    own position.  When true, the range is rebased for every target cell.
    """

    def __init__(
        self,
        name: str,
        *,
        values: Optional[Iterable[Any]] = None,
        range: Optional[Union[str, A1Range]] = None,
        complete: bool = True,
        categories: Optional[Mapping[str, Any]] = None,
        relative_range: bool = False,
        revision: int = 0,
        source_id: Optional[str] = None,
    ) -> None:
        if not name:
            raise ValueError("option source name must not be empty")
        self.name = name
        self.source_id = source_id or name
        self.complete = bool(complete)
        self.revision = int(revision)
        self.relative_range = bool(relative_range)
        self.default = OptionCategory(
            name=None,
            values=list(values or []),
            complete=self.complete,
            range=A1Range.parse(range) if range is not None else None,
            relative_range=self.relative_range,
            revision=self.revision,
        )
        self.categories: "OrderedDict[str, OptionCategory]" = OrderedDict()
        for category_name, category_value in (categories or {}).items():
            self.categories[str(category_name)] = OptionCategory.from_any(
                str(category_name),
                category_value,
                source_complete=self.complete,
                source_revision=self.revision,
                source_relative=self.relative_range,
            )

    @classmethod
    def from_rows(
        cls,
        name: str,
        rows: Iterable[Sequence[Any]],
        *,
        value_column: int = 0,
        category_column: Optional[int] = None,
        **kwargs: Any,
    ) -> "OptionSource":
        """Build a source from a table, optionally grouping by category."""
        values: List[Any] = []
        categories: Dict[str, List[Any]] = OrderedDict()
        for row in rows:
            if value_column >= len(row):
                continue
            value = row[value_column]
            if value in ("", None):
                continue
            values.append(value)
            if category_column is not None and category_column < len(row):
                category = str(row[category_column])
                categories.setdefault(category, []).append(value)
        return cls(name, values=values, categories=categories, **kwargs)

    def get_category(self, name: Optional[str]) -> Optional[OptionCategory]:
        if name is not None:
            return self.categories.get(str(name))
        if self.default.values or self.default.range is not None:
            return self.default
        return None

    def replace_category(self, category: OptionCategory) -> Optional[OptionCategory]:
        """Replace one category and bump its revision.

        Bumping only this category's revision means only cells whose effective
        options came from this category are re-validated.
        """
        if category.name is None:
            previous = self.default
            category.revision += 1
            self.default = category
            return previous
        key = str(category.name)
        previous_category = self.categories.get(key)
        category.revision += 1
        self.categories[key] = category
        return previous_category

    def all_categories(self) -> List[OptionCategory]:
        result: List[OptionCategory] = []
        if self.default.values or self.default.range is not None:
            result.append(self.default)
        result.extend(self.categories.values())
        return result

    def category_revisions(self) -> Dict[Optional[str], Tuple[int, int]]:
        return {
            category.name: (self.revision, category.revision)
            for category in self.all_categories()
        }

    def values_for(self, name: Optional[str]) -> Tuple[List[Any], bool]:
        category = self.get_category(name)
        if category is None:
            return [], False
        return list(category.values), self.complete and category.complete

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "name": self.name,
            "source_id": self.source_id,
            "complete": self.complete,
            "revision": self.revision,
        }
        if self.default.values:
            result["values"] = list(self.default.values)
        if self.default.range is not None:
            result["range"] = str(self.default.range)
        if self.relative_range:
            result["relative_range"] = True
        if self.categories:
            result["categories"] = {
                name: category.to_dict()
                for name, category in self.categories.items()
            }
        return result

    def __repr__(self) -> str:
        return "OptionSource({!r}, complete={!r}, categories={})".format(
            self.name, self.complete, list(self.categories)
        )


@dataclass
class CellRule:
    """One validation rule, written for one cell or a cell range.

    ``source`` names an :class:`OptionSource`; ``category`` selects one of its
    categories.  ``options`` may instead be a literal option list, or the name
    of a locked source.  ``list_range`` is the A1 range exactly as written in
    the cell.  When ``relative`` is true (or omitted and the range contains
    ``$`` markers), the range is interpreted from ``origin`` and rebased for
    every cell the rule expands to.
    """

    cell: str
    rule_type: RuleType = RuleType.list
    mode: RuleMode = RuleMode.reject
    source: Optional[str] = None
    category: Optional[str] = None
    options: Optional[Any] = None
    list_range: Optional[str] = None
    relative: Optional[bool] = None
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    allow_blank: bool = True
    message: Optional[str] = None
    origin: Optional[str] = None
    values: Optional[List[Any]] = None
    cell_range: A1Range = field(init=False)
    _relative: bool = field(init=False, default=False)
    _origin_row: int = field(init=False, default=1)
    _origin_col: int = field(init=False, default=1)

    def __post_init__(self) -> None:
        if not isinstance(self.rule_type, RuleType):
            self.rule_type = RuleType(str(self.rule_type))
        if not isinstance(self.mode, RuleMode):
            self.mode = RuleMode(str(self.mode))
        self.cell_range = A1Range.parse(self.cell)
        if (
            self.cell_range.start_row is None
            or self.cell_range.start_col is None
            or self.cell_range.end_row is None
            or self.cell_range.end_col is None
        ):
            raise ValueError("rule cell must be a bounded range: {!r}".format(self.cell))
        if self.origin is not None:
            origin = A1Range.parse(self.origin)
            if origin.start_row is None or origin.start_col is None:
                raise ValueError("rule origin must be a single cell")
            self._origin_row = origin.start_row
            self._origin_col = origin.start_col
        else:
            self._origin_row = self.cell_range.start_row
            self._origin_col = self.cell_range.start_col
        if self.relative is None:
            # Spreadsheet references are relative by default.  Absolute
            # components are still honoured by A1Range.resolve().
            self._relative = self.list_range is not None
        else:
            self._relative = bool(self.relative)
        if self.values is not None:
            self.values = list(self.values)

    @property
    def options_category(self) -> Optional[str]:
        return self.category

    @property
    def address(self) -> str:
        return str(self.cell_range)

    @property
    def origin_address(self) -> str:
        return rowcol_to_a1(self._origin_row, self._origin_col)

    def resolve_list_range(
        self, target_row: int, target_col: int
    ) -> Optional[A1Range]:
        if self.list_range is None:
            return None
        parsed = A1Range.parse(self.list_range)
        if not self._relative:
            return parsed
        return parsed.resolve(
            self._origin_row,
            self._origin_col,
            target_row,
            target_col,
            relative=True,
        )

    def for_cell(self, row: int, col: int) -> "CellRule":
        label = rowcol_to_a1(row, col)
        if self.cell_range.sheet is not None:
            label = "{}!{}".format(
                _quote_sheet(self.cell_range.sheet), label
            )
        return CellRule(
            cell=label,
            rule_type=self.rule_type,
            mode=self.mode,
            source=self.source,
            category=self.category,
            options=self.options,
            list_range=self.list_range,
            relative=self.relative,
            min_value=self.min_value,
            max_value=self.max_value,
            allow_blank=self.allow_blank,
            message=self.message,
            origin=self.origin_address,
            values=list(self.values) if self.values is not None else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "cell": str(self.cell_range),
            "rule_type": self.rule_type.value,
            "mode": self.mode.value,
            "allow_blank": self.allow_blank,
        }
        optional: Dict[str, Any] = {
            "source": self.source,
            "category": self.category,
            "options": self.options,
            "list_range": self.list_range,
            "relative": self.relative,
            "origin": self.origin,
            "min": self.min_value,
            "max": self.max_value,
            "values": self.values,
            "message": self.message,
        }
        for key, value in optional.items():
            if value is not None:
                result[key] = value
        if self.values is not None:
            result["values"] = list(self.values)
        return result


def _quote_sheet(sheet: str) -> str:
    if _sheet_needs_quotes(sheet):
        return "'{}'".format(sheet.replace("'", "''"))
    return sheet


class ValidationBatch:
    """A locked snapshot of rules and option sources for one audit."""

    def __init__(
        self,
        rules: Iterable[CellRule],
        sources: Optional[Iterable[OptionSource]] = None,
        *,
        name: str = "",
        values: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.name = name
        self.revision = 0
        self.sources: "OrderedDict[str, OptionSource]" = OrderedDict(
            (source.name, source) for source in (sources or [])
        )
        self.duplicate_cells: List[str] = []
        self._raw_rules = list(rules)
        self.rules: "OrderedDict[str, CellRule]" = OrderedDict()
        for rule in self._raw_rules:
            for row, col in rule.cell_range.iter_cells():
                expanded = rule.for_cell(row, col)
                if expanded.address in self.rules:
                    self.duplicate_cells.append(expanded.address)
                self.rules[expanded.address] = expanded
        self.values: Dict[str, Any] = dict(values or {})

    def source(self, name: str) -> OptionSource:
        try:
            return self.sources[name]
        except KeyError:
            raise KeyError("unknown option source: {!r}".format(name))

    def replace_source(self, source: OptionSource) -> Optional[OptionSource]:
        """Replace a source, revisioning only categories that changed.

        Categories whose values, range and completeness are unchanged keep
        their revision.  Their dependent cells are therefore not revalidated
        by :meth:`ValidationAudit.revalidate_changed`.
        """
        previous = self.sources.get(source.name)
        if previous is not None:
            for category in source.all_categories():
                key = _category_key(category)
                old = previous.get_category(key)
                if old is None:
                    category.revision = max(category.revision, 0) + 1
                    continue
                if _category_fingerprint(old) != _category_fingerprint(category):
                    category.revision = max(category.revision, old.revision) + 1
                else:
                    category.revision = old.revision
        self.sources[source.name] = source
        self.revision += 1
        return previous

    def source_for_range(
        self, target: A1Range
    ) -> Optional[Tuple[str, Optional[str]]]:
        """Find a locked source category whose range equals ``target``."""
        target_key = target.key()
        fallback_key = target.without_sheet().key()
        for source in self.sources.values():
            for category in source.all_categories():
                if category.range is None or category.relative_range:
                    continue
                if category.range.key() == target_key:
                    return source.name, _category_key(category)
                if (
                    category.range.sheet is None
                    and target.sheet is not None
                    and category.range.without_sheet().key() == fallback_key
                ):
                    return source.name, _category_key(category)
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "revision": self.revision,
            "rules": [rule.to_dict() for rule in self._raw_rules],
            "sources": [source.to_dict() for source in self.sources.values()],
            "values": dict(self.values),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValidationBatch":
        source_specs = data.get("sources", [])
        if isinstance(source_specs, Mapping):
            sources = [
                _source_from_dict(dict(spec, name=name))
                for name, spec in source_specs.items()
            ]
        else:
            sources = [_source_from_dict(spec) for spec in source_specs]

        rule_specs = data.get("rules", [])
        if isinstance(rule_specs, Mapping):
            rules = [
                _rule_from_dict(dict(spec, cell=cell))
                for cell, spec in rule_specs.items()
            ]
        else:
            rules = [_rule_from_dict(spec) for spec in rule_specs]

        return cls(
            rules,
            sources,
            name=str(data.get("name", "")),
            values=_values_from_any(data.get("values", {})),
        )


class CellResult:
    """The audit result for one cell."""

    __slots__ = (
        "cell",
        "status",
        "rule",
        "value",
        "effective_constraint",
        "allowed",
        "violations",
        "violating_values",
        "source",
        "category",
        "resolved_range",
        "pending_reason",
        "message",
    )

    def __init__(
        self,
        cell: str,
        status: CellStatus,
        *,
        rule: Optional[CellRule] = None,
        value: Any = None,
        effective_constraint: Optional[Dict[str, Any]] = None,
        allowed: Optional[List[Any]] = None,
        source: Optional[str] = None,
        category: Optional[str] = None,
        resolved_range: Optional[A1Range] = None,
        pending_reason: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        self.cell = cell
        self.status = status
        self.rule = rule
        self.value = value
        self.effective_constraint = effective_constraint or {}
        self.allowed = allowed
        self.violations: List[str] = []
        self.violating_values: List[Any] = []
        self.source = source
        self.category = category
        self.resolved_range = resolved_range
        self.pending_reason = pending_reason
        self.message = message

    @property
    def valid(self) -> bool:
        return self.status in (CellStatus.valid, CellStatus.empty)

    @property
    def rejected(self) -> bool:
        return self.status == CellStatus.invalid

    @property
    def warned(self) -> bool:
        return self.status == CellStatus.warned

    @property
    def pending(self) -> bool:
        return self.status == CellStatus.pending

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "cell": self.cell,
            "status": self.status.value,
            "mode": self.rule.mode.value if self.rule else None,
            "rule_type": self.rule.rule_type.value if self.rule else None,
            "value": self.value,
            "violating_values": list(self.violating_values),
            "violations": list(self.violations),
            "effective_constraint": dict(self.effective_constraint),
        }
        if self.allowed is not None:
            result["allowed"] = list(self.allowed)
        if self.source is not None:
            result["source"] = self.source
        if self.category is not None:
            result["category"] = self.category
        if self.resolved_range is not None:
            result["resolved_range"] = str(self.resolved_range)
        if self.pending_reason is not None:
            result["pending_reason"] = self.pending_reason
        if self.message is not None:
            result["message"] = self.message
        return result

    def __repr__(self) -> str:
        return "CellResult({!r}, status={!r})".format(
            self.cell, self.status.value
        )


class ValidationAudit:
    """Evaluate a :class:`ValidationBatch` and track its dependencies."""

    def __init__(self, batch: ValidationBatch) -> None:
        self.batch = batch
        self._results: "OrderedDict[str, CellResult]" = OrderedDict()
        self._values: Dict[str, Any] = {}
        self._value_bodies: Dict[str, Any] = {}
        self._dependencies: Dict[
            Tuple[str, Optional[str]], Set[str]
        ] = defaultdict(set)
        self._cell_dependencies: Dict[
            str, Tuple[str, Optional[str]]
        ] = {}
        self._advisories: Set[Tuple[str, str, str]] = set()
        self._known_revisions: Dict[
            Tuple[str, Optional[str]], Tuple[int, int]
        ] = {}
        self._remember_revisions()

    def validate(self, values: Optional[Mapping[str, Any]] = None) -> List[CellResult]:
        """Validate every rule in the batch."""
        if values is not None:
            self._set_values(values)
        for address in self.batch.rules:
            self._evaluate(address)
        self._remember_revisions()
        self._record_duplicate_advisories()
        return list(self._results.values())

    def validate_cells(
        self,
        values: Optional[Mapping[str, Any]],
        cells: Iterable[str],
    ) -> List[CellResult]:
        """Validate only the named cells; other results are preserved."""
        if values is not None:
            self._merge_values(values)
        addresses = self._ordered_addresses(cells)
        for address in addresses:
            self._evaluate(address)
        return [self._results[address] for address in addresses]

    def revalidate_changed(
        self, values: Optional[Mapping[str, Any]] = None
    ) -> List[CellResult]:
        """Re-validate only cells whose source/category revision changed.

        This is the incremental path after one or more categories are
        revised.  A cell that never depended on a changed category is not
        touched, even when it references the same source.
        """
        if values is not None:
            self._merge_values(values)
        changed = self._changed_pairs()
        addresses: List[str] = []
        for pair in changed:
            addresses.extend(self._dependencies.get(pair, set()))
        addresses.extend(self._resolvable_unresolved_addresses())
        addresses = self._ordered_addresses(addresses)
        for address in addresses:
            self._evaluate(address)
        still_unresolved = {
            address: dependency
            for address, dependency in self._cell_dependencies.items()
            if dependency[0] == _UNRESOLVED_SOURCE
        }
        for address in addresses:
            dependency = still_unresolved.get(address)
            if dependency is not None and address not in self._resolvable_unresolved_addresses():
                self._dependencies[dependency].discard(address)
                self._cell_dependencies.pop(address, None)
        self._remember_revisions()
        return [self._results[address] for address in addresses]

    def revalidate_category(
        self,
        source: str,
        category: Optional[str] = None,
        *,
        values: Optional[Mapping[str, Any]] = None,
        force: bool = False,
    ) -> List[CellResult]:
        """Re-validate the cells that used one category of one source."""
        if values is not None:
            self._merge_values(values)
        pair = (source, category)
        if not force and pair not in self._changed_pairs():
            return []
        addresses = self._ordered_addresses(self._dependencies.get(pair, set()))
        for address in addresses:
            self._evaluate(address)
        self._remember_revisions()
        return [self._results[address] for address in addresses]

    def dependents(
        self, source: str, category: Optional[str] = None
    ) -> List[str]:
        """Cells that actually used ``source`` (and ``category``)."""
        if category is None and (source, None) not in self._dependencies:
            addresses: Set[str] = set()
            for (name, _), cells in self._dependencies.items():
                if name == source:
                    addresses.update(cells)
            return self._ordered_addresses(addresses)
        return self._ordered_addresses(self._dependencies.get((source, category), set()))

    @property
    def results(self) -> List[CellResult]:
        return list(self._results.values())

    def summary(self) -> Dict[str, int]:
        counts = {status.value: 0 for status in CellStatus}
        for result in self._results.values():
            counts[result.status.value] += 1
        counts["total"] = len(self._results)
        counts["rejected"] = counts[CellStatus.invalid.value]
        counts["warnings"] = counts[CellStatus.warned.value]
        return counts

    def to_report(self) -> Dict[str, Any]:
        """Serialize every cell's effective constraint and violations."""
        cells = [result.to_dict() for result in self._results.values()]
        return {
            "batch": self.batch.name,
            "revision": self.batch.revision,
            "summary": self.summary(),
            "cells": cells,
            "by_cell": {
                result.cell: result.to_dict() for result in self._results.values()
            },
            "constraints": {
                result.cell: dict(result.effective_constraint)
                for result in self._results.values()
            },
            "violations": {
                result.cell: list(result.violating_values)
                for result in self._results.values()
                if result.violating_values
            },
            "pending": {
                result.cell: result.pending_reason
                for result in self._results.values()
                if result.pending_reason is not None
            },
            "advisories": [
                {"cell": cell, "code": code, "message": message}
                for cell, code, message in sorted(self._advisories)
            ],
        }

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_report(), ensure_ascii=False, indent=indent)

    def to_csv(self, *, include_empty: bool = True) -> str:
        output = io.StringIO()
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(
            [
                "cell",
                "status",
                "mode",
                "rule_type",
                "value",
                "violating_values",
                "effective_constraint",
                "source",
                "category",
                "resolved_range",
                "violations",
                "pending_reason",
                "message",
            ]
        )
        for result in self._results.values():
            if not include_empty and result.status == CellStatus.empty:
                continue
            writer.writerow(
                [
                    result.cell,
                    result.status.value,
                    result.rule.mode.value if result.rule else "",
                    result.rule.rule_type.value if result.rule else "",
                    _compact(result.value),
                    _compact(result.violating_values),
                    _compact(result.effective_constraint),
                    result.source or "",
                    result.category or "",
                    str(result.resolved_range) if result.resolved_range else "",
                    "; ".join(result.violations),
                    result.pending_reason or "",
                    result.message or "",
                ]
            )
        return output.getvalue()

    def _set_values(self, values: Mapping[str, Any]) -> None:
        self._values = {}
        self._value_bodies = {}
        self._merge_values(values)

    def _merge_values(self, values: Mapping[str, Any]) -> None:
        for key, value in _values_from_any(values).items():
            parsed = A1Range.parse(key)
            address = str(parsed)
            self._values[address] = value
            self._value_bodies[parsed.without_sheet().body()] = value

    def _lookup_value(self, address: str) -> Any:
        if address in self._values:
            return self._values[address]
        parsed = A1Range.parse(address)
        return self._value_bodies.get(parsed.without_sheet().body())

    def _ordered_addresses(self, cells: Iterable[str]) -> List[str]:
        wanted: Set[str] = set()
        for cell in cells:
            parsed = A1Range.parse(cell)
            address = str(parsed)
            if address in self.batch.rules:
                wanted.add(address)
                continue
            body = parsed.without_sheet().body()
            for candidate in self.batch.rules:
                if A1Range.parse(candidate).without_sheet().body() == body:
                    wanted.add(candidate)
        return [address for address in self.batch.rules if address in wanted]

    def _remember_revisions(self) -> None:
        self._known_revisions = {}
        for name, source in self.batch.sources.items():
            for category, revision in source.category_revisions().items():
                self._known_revisions[(name, category)] = revision

    def _changed_pairs(self) -> List[Tuple[str, Optional[str]]]:
        current: Dict[Tuple[str, Optional[str]], Tuple[int, int]] = {}
        for name, source in self.batch.sources.items():
            for category, revision in source.category_revisions().items():
                current[(name, category)] = revision
        changed = []
        for pair in set(current) | set(self._known_revisions):
            if current.get(pair) != self._known_revisions.get(pair):
                changed.append(pair)
        if self._unresolved_is_resolvable():
            changed.extend(
                pair
                for pair in self._dependencies
                if pair[0] == _UNRESOLVED_SOURCE
            )
        return changed

    def _unresolved_is_resolvable(self) -> bool:
        return bool(self._resolvable_unresolved_addresses())

    def _resolvable_unresolved_addresses(self) -> Set[str]:
        addresses: Set[str] = set()
        for address, (source_name, _) in self._cell_dependencies.items():
            if source_name != _UNRESOLVED_SOURCE:
                continue
            rule = self.batch.rules.get(address)
            if rule is None or rule.list_range is None:
                continue
            resolved = rule.resolve_list_range(
                rule.cell_range.start_row or 1,
                rule.cell_range.start_col or 1,
            )
            if resolved is None:
                continue
            if self.batch.source_for_range(resolved) or self._find_relative_category(
                rule, resolved
            ):
                addresses.add(address)
        return addresses

    def _record_duplicate_advisories(self) -> None:
        for address in self.batch.duplicate_cells:
            self._advisories.add(
                (
                    address,
                    "duplicate-rule",
                    "more than one rule targeted this cell; the last one wins",
                )
            )

    def _evaluate(self, address: str) -> CellResult:
        rule = self.batch.rules.get(address)
        if rule is None:
            raise KeyError("no rule for cell {!r}".format(address))
        value = self._lookup_value(address)
        result = CellResult(
            address,
            CellStatus.valid,
            rule=rule,
            value=value,
            message=rule.message,
        )

        if rule.rule_type == RuleType.list:
            self._evaluate_list(rule, result)
        elif rule.rule_type == RuleType.number:
            self._evaluate_number(rule, result)
        elif rule.rule_type == RuleType.text:
            self._evaluate_text(rule, result)
        elif rule.rule_type == RuleType.date:
            self._evaluate_date(rule, result)
        elif rule.rule_type == RuleType.boolean:
            self._evaluate_boolean(rule, result)
        elif rule.rule_type == RuleType.blank:
            self._evaluate_blank(rule, result)
        elif rule.rule_type == RuleType.custom_formula:
            self._evaluate_formula(rule, result)

        self._results[address] = result
        return result

    def _base_constraint(self, rule: CellRule) -> Dict[str, Any]:
        return {
            "type": rule.rule_type.value,
            "cell": rule.address,
            "mode": rule.mode.value,
            "allow_blank": rule.allow_blank,
        }

    def _is_blank(self, value: Any) -> bool:
        return value is None or (isinstance(value, str) and value.strip() == "")

    def _set_empty(self, rule: CellRule, result: CellResult) -> bool:
        if not self._is_blank(result.value):
            return False
        result.status = CellStatus.empty if rule.allow_blank else CellStatus.invalid
        result.effective_constraint = self._base_constraint(rule)
        if not rule.allow_blank:
            result.violations.append("value must not be blank")
            result.violating_values.append(result.value)
        return True

    def _evaluate_list(self, rule: CellRule, result: CellResult) -> None:
        constraint = self._base_constraint(rule)
        result.effective_constraint = constraint
        if self._set_empty(rule, result):
            return

        allowed, complete = self._resolve_list_options(rule, result, constraint)
        if allowed is None:
            return

        result.allowed = list(allowed)
        constraint["options"] = list(allowed)
        constraint["complete"] = complete

        matched = _match_option(result.value, allowed)
        if not complete:
            self._pending(
                result,
                constraint,
                "incomplete-source",
                rule.message
                or "option source is incomplete; value cannot be confirmed",
            )
            return
        if matched:
            result.status = CellStatus.valid
            return
        result.violations.append(
            "value {!r} is not one of the effective options".format(result.value)
        )
        result.violating_values.append(result.value)
        result.status = (
            CellStatus.invalid if rule.mode == RuleMode.reject else CellStatus.warned
        )

    def _resolve_list_options(
        self,
        rule: CellRule,
        result: CellResult,
        constraint: Dict[str, Any],
    ) -> Tuple[Optional[List[Any]], bool]:
        source_name = rule.source
        if (
            source_name is None
            and isinstance(rule.options, str)
            and rule.options in self.batch.sources
        ):
            source_name = rule.options
        if source_name is not None:
            return self._resolve_named_source(rule, result, constraint, source_name)
        if rule.values is not None:
            return list(rule.values), True
        if isinstance(rule.options, (list, tuple, set)):
            return list(rule.options), True
        if rule.list_range is not None:
            return self._resolve_range_options(rule, result, constraint)
        if rule.options is not None:
            return [rule.options], True
        self._pending(
            result,
            constraint,
            "options-missing",
            "list rule has no options and no list range",
        )
        return None, True

    def _resolve_named_source(
        self,
        rule: CellRule,
        result: CellResult,
        constraint: Dict[str, Any],
        source_name: str,
    ) -> Tuple[Optional[List[Any]], bool]:
        category_name = rule.category
        source = self.batch.sources.get(source_name)
        if source is None:
            self._bind_dependency(rule, source_name, category_name)
            result.source = source_name
            result.category = category_name
            self._pending(
                result,
                constraint,
                "source-missing",
                "option source {!r} is not locked in this batch".format(
                    source_name
                ),
            )
            return None, False
        category = source.get_category(category_name)
        if category is None:
            self._bind_dependency(rule, source_name, category_name)
            result.source = source_name
            result.category = category_name
            self._pending(
                result,
                constraint,
                "category-missing",
                "category {!r} of source {!r} is not locked".format(
                    category_name, source_name
                ),
            )
            return None, False
        category_name = _category_key(category)
        self._bind_dependency(rule, source_name, category_name)
        result.source = source_name
        result.category = category_name
        resolved = self._resolved_range(rule, category)
        result.resolved_range = resolved
        constraint["source"] = source_name
        constraint["category"] = category_name
        if resolved is not None:
            constraint["range"] = str(resolved)
        return list(category.values), source.complete and category.complete

    def _resolve_range_options(
        self,
        rule: CellRule,
        result: CellResult,
        constraint: Dict[str, Any],
    ) -> Tuple[Optional[List[Any]], bool]:
        resolved = rule.resolve_list_range(
            rule.cell_range.start_row or 1,
            rule.cell_range.start_col or 1,
        )
        result.resolved_range = resolved
        found = (
            self.batch.source_for_range(resolved) if resolved is not None else None
        )
        if found is None and rule.relative:
            found = self._find_relative_category(rule, resolved)
        if found is None:
            where = str(resolved) if resolved is not None else rule.list_range
            self._bind_dependency(
                rule, _UNRESOLVED_SOURCE, _unresolved_key(rule.list_range)
            )
            self._pending(
                result,
                constraint,
                "range-not-locked",
                "option range {!r} is not part of the locked batch".format(where),
            )
            return None, False
        source_name, category_name = found
        source = self.batch.source(source_name)
        category = source.get_category(category_name)
        assert category is not None
        self._bind_dependency(rule, source_name, category_name)
        result.source = source_name
        result.category = category_name
        constraint.update(
            {
                "source": source_name,
                "category": category_name,
                "range": str(resolved) if resolved is not None else "",
            }
        )
        return list(category.values), source.complete and category.complete

    def _bind_dependency(
        self,
        rule: CellRule,
        source_name: str,
        category_name: Optional[str],
    ) -> None:
        self._dependencies[(source_name, category_name)].add(rule.address)
        self._cell_dependencies[rule.address] = (source_name, category_name)

    def _resolved_range(
        self,
        rule: CellRule,
        category: OptionCategory,
    ) -> Optional[A1Range]:
        target_row = rule.cell_range.start_row or 1
        target_col = rule.cell_range.start_col or 1
        if rule.list_range is not None:
            return rule.resolve_list_range(target_row, target_col)
        if category.range is None:
            return None
        if category.relative_range:
            return category.range.resolve(
                rule._origin_row,
                rule._origin_col,
                target_row,
                target_col,
                relative=True,
            )
        return category.range

    def _find_relative_category(
        self, rule: CellRule, resolved: Optional[A1Range]
    ) -> Optional[Tuple[str, Optional[str]]]:
        if resolved is None:
            return None
        target_row = rule.cell_range.start_row or 1
        target_col = rule.cell_range.start_col or 1
        for source in self.batch.sources.values():
            for category in source.all_categories():
                if category.range is None:
                    continue
                if not category.relative_range:
                    continue
                candidate = category.range.resolve(
                    rule._origin_row,
                    rule._origin_col,
                    target_row,
                    target_col,
                    relative=True,
                )
                if candidate.key() == resolved.key():
                    return source.name, _category_key(category)
        return None

    def _evaluate_number(self, rule: CellRule, result: CellResult) -> None:
        constraint = self._base_constraint(rule)
        constraint.update(
            {
                "min": rule.min_value,
                "max": rule.max_value,
                "allowed": list(rule.values) if rule.values else None,
            }
        )
        result.effective_constraint = constraint
        if self._set_empty(rule, result):
            return
        number = _to_number(result.value)
        if number is None:
            self._violate(rule, result, "value is not numeric", result.value)
            return
        if rule.values is not None and not _match_option(number, rule.values):
            self._violate(
                rule,
                result,
                "value {} is not in the allowed set".format(number),
                result.value,
            )
            return
        if rule.min_value is not None and number < float(rule.min_value):
            self._violate(
                rule,
                result,
                "value {} is below minimum {}".format(number, rule.min_value),
                result.value,
            )
        if rule.max_value is not None and number > float(rule.max_value):
            self._violate(
                rule,
                result,
                "value {} is above maximum {}".format(number, rule.max_value),
                result.value,
            )

    def _evaluate_text(self, rule: CellRule, result: CellResult) -> None:
        constraint = self._base_constraint(rule)
        constraint.update(
            {
                "allowed": list(rule.values) if rule.values else None,
                "min_length": rule.min_value,
                "max_length": rule.max_value,
            }
        )
        result.effective_constraint = constraint
        if self._set_empty(rule, result):
            return
        text = str(result.value)
        if rule.values is not None and not _match_option(
            text, [str(value) for value in rule.values]
        ):
            self._violate(
                rule,
                result,
                "value {!r} is not in the allowed set".format(result.value),
                result.value,
            )
            return
        if rule.min_value is not None and len(text) < int(rule.min_value):
            self._violate(
                rule,
                result,
                "text length {} is below {}".format(len(text), rule.min_value),
                result.value,
            )
        if rule.max_value is not None and len(text) > int(rule.max_value):
            self._violate(
                rule,
                result,
                "text length {} is above {}".format(len(text), rule.max_value),
                result.value,
            )

    def _evaluate_date(self, rule: CellRule, result: CellResult) -> None:
        constraint = self._base_constraint(rule)
        constraint.update({"min": rule.min_value, "max": rule.max_value})
        result.effective_constraint = constraint
        if self._set_empty(rule, result):
            return
        parsed = _to_date(result.value)
        if parsed is None:
            self._violate(rule, result, "value is not an ISO date", result.value)
            return
        minimum = _to_date(rule.min_value) if rule.min_value is not None else None
        maximum = _to_date(rule.max_value) if rule.max_value is not None else None
        if minimum is not None and parsed < minimum:
            self._violate(
                rule,
                result,
                "date {} is before {}".format(parsed, minimum),
                result.value,
            )
        if maximum is not None and parsed > maximum:
            self._violate(
                rule,
                result,
                "date {} is after {}".format(parsed, maximum),
                result.value,
            )

    def _evaluate_boolean(self, rule: CellRule, result: CellResult) -> None:
        constraint = self._base_constraint(rule)
        constraint["allowed"] = ["true", "false"]
        result.effective_constraint = constraint
        if self._set_empty(rule, result):
            return
        if not _is_boolean(result.value):
            self._violate(rule, result, "value is not a boolean", result.value)

    def _evaluate_blank(self, rule: CellRule, result: CellResult) -> None:
        expect_blank = rule.options in (None, "blank", True)
        constraint = self._base_constraint(rule)
        constraint["expect"] = "blank" if expect_blank else "not_blank"
        result.effective_constraint = constraint
        blank = self._is_blank(result.value)
        if expect_blank and not blank:
            self._violate(rule, result, "value must be blank", result.value)
        elif not expect_blank and blank:
            self._violate(rule, result, "value must not be blank", result.value)
        else:
            result.status = CellStatus.empty if blank else CellStatus.valid

    def _evaluate_formula(self, rule: CellRule, result: CellResult) -> None:
        constraint = self._base_constraint(rule)
        constraint["formula"] = rule.options
        result.effective_constraint = constraint
        if self._set_empty(rule, result):
            return
        self._pending(
            result,
            constraint,
            "formula-pending",
            rule.message or "custom formula cannot be evaluated offline",
        )

    def _violate(
        self,
        rule: CellRule,
        result: CellResult,
        message: str,
        value: Any,
    ) -> None:
        result.violations.append(message)
        result.violating_values.append(value)
        result.status = (
            CellStatus.invalid if rule.mode == RuleMode.reject else CellStatus.warned
        )

    def _pending(
        self,
        result: CellResult,
        constraint: Dict[str, Any],
        reason: str,
        message: str,
    ) -> None:
        result.status = CellStatus.pending
        result.pending_reason = reason
        result.message = message
        result.effective_constraint = constraint
        self._advisories.add((result.cell, reason, message))


def _category_key(category: OptionCategory) -> Optional[str]:
    return category.name


def _category_fingerprint(category: OptionCategory) -> Tuple[Any, ...]:
    return (
        category.complete,
        tuple(str(value) for value in category.values),
        category.range.key() if category.range is not None else None,
        category.relative_range,
    )


def _match_option(value: Any, allowed: Sequence[Any]) -> bool:
    for option in allowed:
        if value == option:
            return True
        if str(value).strip() == str(option).strip():
            return True
        left = _to_number(value)
        right = _to_number(option)
        if left is not None and right is not None and left == right:
            return True
    return False


def _to_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip().replace(",", ""))
        except ValueError:
            return None
    return None


def _to_date(value: Any) -> Optional[date]:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip())
        except ValueError:
            return None
    return None


def _is_boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    return isinstance(value, str) and value.strip().lower() in (
        "true",
        "false",
        "yes",
        "no",
        "1",
        "0",
    )


def _compact(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


def _values_from_any(values: Any) -> Dict[str, Any]:
    if values is None:
        return {}
    if isinstance(values, Mapping):
        return dict(values)
    if isinstance(values, Iterable):
        result: Dict[str, Any] = {}
        for item in values:
            if not isinstance(item, Mapping) or "cell" not in item:
                raise ValueError("values must be a mapping or a list of {cell, value}")
            result[str(item["cell"])] = item.get("value")
        return result
    raise ValueError("values must be a mapping or a list of {cell, value}")


def _source_from_dict(data: Mapping[str, Any]) -> OptionSource:
    return OptionSource(
        str(data["name"]),
        values=data.get("values"),
        range=data.get("range"),
        complete=bool(data.get("complete", True)),
        categories=data.get("categories"),
        relative_range=bool(data.get("relative_range", False)),
        revision=int(data.get("revision", 0)),
        source_id=data.get("source_id"),
    )


def _rule_from_dict(data: Mapping[str, Any]) -> CellRule:
    if "cell" not in data and "range" not in data:
        raise ValueError("rule is missing 'cell'")
    options = data.get("options")
    values = data.get("values")
    if values is None and isinstance(options, list):
        values = options
    return CellRule(
        str(data.get("cell", data.get("range"))),
        rule_type=RuleType(str(data.get("rule_type", RuleType.list.value))),
        mode=RuleMode(str(data.get("mode", RuleMode.reject.value))),
        source=data.get("source", data.get("source_id")),
        category=data.get("category", data.get("options_category")),
        options=options if not isinstance(options, list) else None,
        list_range=data.get("list_range"),
        relative=data.get("relative"),
        min_value=data.get("min", data.get("min_value")),
        max_value=data.get("max", data.get("max_value")),
        allow_blank=bool(data.get("allow_blank", True)),
        message=data.get("message"),
        origin=data.get("origin"),
        values=values,
    )


def audit_from_dict(data: Mapping[str, Any]) -> ValidationAudit:
    """Build and run an audit from a JSON-compatible mapping."""
    batch = ValidationBatch.from_dict(data)
    audit = ValidationAudit(batch)
    audit.validate(_values_from_any(data.get("values", {})))
    return audit


def load_batch(stream: Union[str, TextIO, Mapping[str, Any]]) -> ValidationBatch:
    """Load a batch from a path/stream of JSON or an already parsed mapping."""
    if isinstance(stream, Mapping):
        return ValidationBatch.from_dict(stream)
    if hasattr(stream, "read"):
        return ValidationBatch.from_dict(json.load(stream))
    with open(str(stream), "r", encoding="utf-8") as handle:
        return ValidationBatch.from_dict(json.load(handle))


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command line entry point for batch JSON auditing."""
    parser = argparse.ArgumentParser(
        description="Audit spreadsheet data-validation rules and option sources."
    )
    parser.add_argument("input", help="batch JSON file, or '-' for stdin")
    parser.add_argument(
        "--format",
        choices=("json", "csv"),
        default="json",
        help="report format (default: json)",
    )
    parser.add_argument("-o", "--output", help="write the report to this file")
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="omit empty cells from CSV output",
    )
    args = parser.parse_args(argv)

    if args.input == "-":
        data = json.load(sys.stdin)
        audit = audit_from_dict(data)
    else:
        with open(args.input, "r", encoding="utf-8") as handle:
            audit = audit_from_dict(json.load(handle))

    if args.format == "csv":
        report = audit.to_csv(include_empty=not args.skip_empty)
    else:
        report = audit.to_json()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(report)
            if not report.endswith("\n"):
                handle.write("\n")
    else:
        sys.stdout.write(report)
        if not report.endswith("\n"):
            sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
