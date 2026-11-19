"""
gspread.references
~~~~~~~~~~~~~~~~~~

Helpers to recognize references to other worksheets.

Google Sheets stores cross-sheet references as text such as ``'Sheet 1'!A1``.
The parser in this module only recognizes *static* references: it never tries
to evaluate formulas or guess what ``INDIRECT()`` will produce at runtime.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .utils import rowcol_to_a1

# A single part of an A1 range. ``$`` is accepted on both axes.
_A1_PART_RE = re.compile(
    r"^(?:"
    r"\$?[A-Za-z]{1,3}\$?[1-9][0-9]*"  # A1
    r"|\$?[A-Za-z]{1,3}"  # A
    r"|\$?[1-9][0-9]*"  # 1
    r")$"
)

_A1_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789$:"
)
_UNQUOTED_TITLE_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_."
)


@dataclass(frozen=True)
class FormulaReference:
    """A single static cross-sheet reference found in a formula.

    :ivar sheet_title: The sheet title exactly as it appears in the formula.
    :ivar a1_range: The referenced A1 range (may be a single cell, a full row
        or a full column).
    :ivar raw: The complete reference text, e.g. ``'My Sheet'!A1:B2``.
    :ivar start: Index of the reference in the formula.
    :ivar end: Index just after the reference in the formula.
    """

    sheet_title: str
    a1_range: str
    raw: str
    start: int = 0
    end: int = 0


def is_a1_range(value: str) -> bool:
    """Return whether ``value`` is a recognizable A1 cell/range reference."""
    if not value:
        return False
    parts = value.split(":")
    if len(parts) > 2:
        return False
    if not all(_A1_PART_RE.match(part) for part in parts):
        return False
    if len(parts) == 1:
        return True
    left, right = parts
    left_is_col = not any(ch.isdigit() for ch in left)
    left_is_row = not any(ch.isalpha() for ch in left)
    right_is_col = not any(ch.isdigit() for ch in right)
    right_is_row = not any(ch.isalpha() for ch in right)
    if left_is_col and right_is_col:
        return True
    if left_is_row and right_is_row:
        return True
    # Mixed ranges such as A1:B are valid in Google Sheets.
    return True


def _skip_double_quoted(formula: str, start: int) -> int:
    """Return the index after a double-quoted string literal."""
    i = start + 1
    length = len(formula)
    while i < length:
        if formula[i] == '"':
            if i + 1 < length and formula[i + 1] == '"':
                i += 2
                continue
            return i + 1
        i += 1
    return length


def _read_quoted_title(formula: str, start: int) -> Tuple[Optional[str], int]:
    """Read a single-quoted sheet title.

    In Sheets formulas a literal quote inside a sheet name is doubled, for
    example ``'Bob''s sheet'!A1``.
    """
    i = start + 1
    length = len(formula)
    chars: List[str] = []
    while i < length:
        if formula[i] == "'":
            if i + 1 < length and formula[i + 1] == "'":
                chars.append("'")
                i += 2
                continue
            return "".join(chars), i + 1
        chars.append(formula[i])
        i += 1
    return None, length


def _read_a1_range(formula: str, start: int) -> Tuple[str, int]:
    i = start
    length = len(formula)
    while i < length and formula[i] in _A1_CHARS:
        i += 1
    value = formula[start:i]
    # Trailing ':' belongs to a malformed reference such as ``Sheet1!A1:``.
    while value.endswith(":"):
        value = value[:-1]
        i -= 1
    return value, i


def _reference_at(
    formula: str, start: int, title: str, range_start: int
) -> Optional[FormulaReference]:
    """Build a reference when ``range_start`` follows a completed ``title!``."""
    if range_start > len(formula) or formula[range_start - 1 : range_start] != "!":
        return None
    a1_range, end = _read_a1_range(formula, range_start)
    if not is_a1_range(a1_range):
        return None
    return FormulaReference(
        sheet_title=title,
        a1_range=a1_range,
        raw=formula[start:end],
        start=start,
        end=end,
    )


def _parse_quoted_reference(
    formula: str, start: int
) -> Tuple[Optional[FormulaReference], int]:
    title, after_title = _read_quoted_title(formula, start)
    if title is None:
        return None, max(after_title, start + 1)
    reference = _reference_at(formula, start, title, after_title + 1)
    if reference is None:
        return None, max(after_title, start + 1)
    return reference, reference.end


def _parse_unquoted_reference(
    formula: str, start: int
) -> Tuple[Optional[FormulaReference], int]:
    i = start
    length = len(formula)
    while i < length and formula[i] in _UNQUOTED_TITLE_CHARS:
        i += 1
    title = formula[start:i]
    if not title:
        return None, i
    reference = _reference_at(formula, start, title, i + 1)
    if reference is not None:
        return reference, reference.end
    return None, i


def extract_formula_references(formula: str) -> List[FormulaReference]:
    """Extract static cross-sheet references from a formula.

    The parser intentionally ignores ``INDIRECT()`` and text inside string
    literals. Only references that literally contain ``!`` are returned.

    Examples::

        >>> extract_formula_references("=SUM(Data!A1:B2)")
        [FormulaReference(sheet_title='Data', a1_range='A1:B2', raw='Data!A1:B2', ...)]

        >>> extract_formula_references("='Bob''s data'!A1")
        [FormulaReference(sheet_title="Bob's data", a1_range='A1', ...)]
    """
    if not isinstance(formula, str) or not formula:
        return []

    references: List[FormulaReference] = []
    i = 0
    length = len(formula)

    while i < length:
        char = formula[i]

        if char == '"':
            i = _skip_double_quoted(formula, i)
            continue

        if char == "'":
            reference, i = _parse_quoted_reference(formula, i)
        elif char in _UNQUOTED_TITLE_CHARS:
            reference, i = _parse_unquoted_reference(formula, i)
        else:
            reference = None
            i += 1

        if reference is not None:
            references.append(reference)

    return references


def extract_sheet_titles(formula: str) -> List[str]:
    """Return the distinct sheet titles referenced by a formula.

    The order of first appearance is preserved.
    """
    seen = set()
    titles: List[str] = []
    for reference in extract_formula_references(formula):
        if reference.sheet_title not in seen:
            seen.add(reference.sheet_title)
            titles.append(reference.sheet_title)
    return titles


def find_formula_cell(
    rows: Sequence[Sequence[object]],
    start_row: int = 1,
    start_col: int = 1,
) -> List[Tuple[str, str]]:
    """Return ``(cell, formula)`` pairs for ``rows``.

    ``rows`` is a matrix as returned by the Sheets values API. ``start_row``
    and ``start_col`` are one-based coordinates of the top-left cell.
    """
    result: List[Tuple[str, str]] = []
    for row_offset, row in enumerate(rows):
        for col_offset, value in enumerate(row):
            if isinstance(value, str) and value.startswith("="):
                cell = rowcol_to_a1(
                    start_row + row_offset, start_col + col_offset
                )
                result.append((cell, value))
    return result
