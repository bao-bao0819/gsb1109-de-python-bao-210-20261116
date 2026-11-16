"""Apply offline edit suggestions to a parsed wikitext document.

An edit suggestion describes a single localized cleanup operation whose
position was computed against the original (old) wikitext: deleting a
comment, replacing an external link URL, or changing a table cell value.

Suggestions are merged without touching any byte outside the targeted
spans. Overlapping suggestions and replacements that would invalidate
the syntax next to an untouched region are reported through
``EditConflictError`` and nothing is applied. The merge result does not
depend on the order in which the suggestions are supplied.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from regex import compile as rc

if TYPE_CHECKING:
    from ._cell import Cell
    from ._comment_bold_italic import Comment
    from ._externallink import ExternalLink
    from ._wikitext import WikiText

TARGET_COMMENT = 'comment'
TARGET_EXTERNAL_LINK = 'external_link'
TARGET_CELL = 'cell'
_TARGETS = frozenset((TARGET_COMMENT, TARGET_EXTERNAL_LINK, TARGET_CELL))

# Characters that cannot occur inside an external link URL.
# Mirrors INVALID_URL_CHARS in wikitextparser._spans.
_INVALID_URL_CHARS = frozenset(b' \t\r\n"<>[]')

# Substrings that would turn a cell value into new table structure if
# written through the cell's data span. A pipe is always an attribute or
# cell separator, "!!" splits a header cell, and a line break followed by
# "!" starts a new header cell or row. A single "!" is harmless.
_CELL_STRUCTURE_BREAK_SEARCH = rc(rb'\||!!|\R[^\S\r\n]*!').search


class EditError(ValueError):
    """Base class for failures while applying edit suggestions."""


class EditConflictError(EditError):
    """Suggestions overlap or would invalidate adjacent syntax."""


@dataclass(frozen=True)
class EditSuggestion:
    """A single localized edit anchored to the original wikitext.

    Every span is expressed against the original string, even when
    several suggestions are applied together.

    :param target: kind of element to edit, one of ``'comment'``,
        ``'external_link'``, or ``'cell'``.
    :param span: the ``(start, end)`` character span of the element in
        the original wikitext, as reported by ``WikiText.span``.
    :param replacement: new URL for external links or new value for
        table cells. Ignored for comments, which are always deleted.
    """

    target: str
    span: tuple[int, int]
    replacement: str | None = None

    @classmethod
    def delete_comment(cls, span: tuple[int, int]) -> EditSuggestion:
        """Suggest deleting the comment covering ``span``."""
        return cls(TARGET_COMMENT, span)

    @classmethod
    def replace_external_link(
        cls, span: tuple[int, int], new_url: str
    ) -> EditSuggestion:
        """Suggest replacing the URL of the link covering ``span``."""
        return cls(TARGET_EXTERNAL_LINK, span, new_url)

    @classmethod
    def replace_cell(
        cls, span: tuple[int, int], new_value: str
    ) -> EditSuggestion:
        """Suggest replacing the value of the cell covering ``span``."""
        return cls(TARGET_CELL, span, new_value)


# (start, end, target, suggestion) for a validated, not-yet-applied edit.
_PreparedSuggestion = tuple[int, int, str, EditSuggestion]


def apply_edit_suggestions(
    wikitext: WikiText, suggestions: Iterable[EditSuggestion]
) -> WikiText:
    """Merge offline edit suggestions into ``wikitext``.

    The targeted comments, external links, and table cells are looked up
    by their spans in the original document; every byte outside the
    selected spans is preserved unchanged.

    :raises ValueError: a suggestion has an unknown target or a malformed
        span.
    :raises EditError: a span does not point to the claimed element of
        the original wikitext (a stale suggestion).
    :raises EditConflictError: two suggestions target overlapping spans,
        or a replacement would invalidate the syntax next to an
        untouched region, such as a bracket-breaking URL or a cell value
        introducing a new cell separator. All detected conflicts are
        listed in the exception message and nothing is applied.

    The result is independent of the order of ``suggestions``.
    """
    suggestions = list(suggestions)
    if not suggestions:
        return wikitext
    original = wikitext.string
    comments, links, cells = _index_elements(wikitext)
    prepared: list[_PreparedSuggestion] = []
    syntax_problems = []
    indexes = {
        TARGET_COMMENT: comments,
        TARGET_EXTERNAL_LINK: links,
        TARGET_CELL: cells,
    }
    for suggestion in suggestions:
        target = suggestion.target
        if target not in _TARGETS:
            raise ValueError(f'unknown edit suggestion target: {target!r}')
        start, end = _checked_span(suggestion.span, len(original))
        if (start, end) not in indexes[target]:
            raise EditError(
                f'span {(start, end)} does not point to a {target} '
                'of the original wikitext'
            )
        problem = _replacement_problem(target, suggestion.replacement)
        if problem is not None:
            syntax_problems.append(
                f'{target} suggestion at span {(start, end)} {problem}'
            )
        prepared.append((start, end, target, suggestion))
    # Sorting makes both conflict detection and the final page
    # independent of the order in which suggestions were supplied.
    prepared.sort(key=lambda item: (item[0], item[1], item[2]))
    syntax_problems.extend(_overlap_problems(prepared))
    if syntax_problems:
        raise EditConflictError('; '.join(syntax_problems))
    # Disjoint spans: right-to-left application keeps every remaining
    # anchor valid regardless of the original input order.
    for start, end, target, suggestion in reversed(prepared):
        span = (start, end)
        if target == TARGET_COMMENT:
            del comments[span][:]
            continue
        # Non-comment targets require a string; _replacement_problem
        # rejected any None or non-string replacement above.
        replacement = suggestion.replacement
        assert isinstance(replacement, str)
        if target == TARGET_EXTERNAL_LINK:
            links[span].url = replacement
        else:
            cells[span].value = replacement
    return wikitext


def _index_elements(
    wikitext: WikiText,
) -> tuple[
    dict[tuple[int, int], Comment],
    dict[tuple[int, int], ExternalLink],
    dict[tuple[int, int], Cell],
]:
    comments = {comment.span: comment for comment in wikitext.comments}
    links = {link.span: link for link in wikitext.external_links}
    cells: dict[tuple[int, int], Cell] = {}
    for table in wikitext.get_tables(recursive=True):
        for row in table.cells():
            for cell in row:
                if cell is not None:
                    cells.setdefault(cell.span, cell)
    return comments, links, cells


def _checked_span(span: tuple[int, int], length: int) -> tuple[int, int]:
    if (
        not isinstance(span, tuple)
        or len(span) != 2
        or not all(isinstance(i, int) for i in span)
    ):
        raise ValueError(f'span must be a (start, end) tuple, got {span!r}')
    start, end = span
    if start < 0 or end < start or end > length:
        raise EditError(
            f'span {(start, end)} is out of range for a wikitext '
            f'of length {length}'
        )
    return start, end


def _replacement_problem(target: str, replacement: str | None) -> str | None:
    if target == TARGET_COMMENT:
        return None
    if not isinstance(replacement, str):
        return 'replacement must be a string'
    if target == TARGET_EXTERNAL_LINK:
        if not replacement:
            return 'replacement URL is empty'
        bad = sorted(
            {
                chr(code)
                for code in replacement.encode('ascii', 'replace')
                if code in _INVALID_URL_CHARS
            }
        )
        if bad:
            return (
                'replacement URL contains characters that would '
                f'break the link: {"".join(bad)!r}'
            )
        return None
    # A table cell value must not introduce new table structure,
    # otherwise the neighboring cells of the untouched region would
    # shift or change meaning.
    if _CELL_STRUCTURE_BREAK_SEARCH(replacement.encode('ascii', 'replace')):
        return (
            'replacement value introduces a new cell or row separator '
            'which would invalidate the neighboring table syntax'
        )
    return None


def _overlap_problems(prepared: list[_PreparedSuggestion]) -> list[str]:
    problems = []
    for left, right in zip(prepared, prepared[1:]):
        if left[1] > right[0]:  # left end > right start
            problems.append(
                f'overlapping suggestions at spans {left[0:2]} '
                f'({left[2]}) and {right[0:2]} ({right[2]})'
            )
    return problems
