from __future__ import annotations

from bisect import insort_right
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from re import Match
from typing import Any, Callable, TypeVar, cast, overload

from regex import DOTALL, VERBOSE

from wikitextparser._spans import SpanData

from ._cell import (
    INLINE_HAEDER_CELL_MATCH,
    INLINE_NONHAEDER_CELL_MATCH,
    NEWLINE_CELL_MATCH,
    Cell,
)
from ._spans import ATTRS_MATCH
from ._tag import SubWikiTextWithAttrs
from ._wikitext import WS, rc

CAPTION_MATCH = rc(
    rb"""
    # Everything until the caption line
    (?P<preattrs>
        # Start of table
        {\|
        (?:
            (?:
                (?!\R\s*+\|)
                [\s\S]
            )*?
        )
        # Start of caption line
        \R\s*+\|\+
    )
    # Optional caption attrs
    (?:
        (?P<attrs>[^\r\n|]*+)
        \|(?!\|)
    )?
    (?P<caption>.*?)
    (?:\R[\|\!]|\|\|)
    """,
    DOTALL | VERBOSE,
).match
T = TypeVar('T')
FIND_ROWS = rc(rb'\|-(.*)').finditer


@dataclass(frozen=True)
class CellRevision:
    """A value change for a cell at a row/column coordinate.

    Coordinates are zero-based and refer to the rendered table when
    ``span`` is true, i.e. a coordinate covered by a merged cell points to
    the anchor cell that provides its value.
    """

    row: int
    column: int
    value: str


@dataclass(frozen=True)
class RevisionWrite:
    """A value written to a single source cell."""

    anchor_row: int
    anchor_column: int
    span: tuple[int, int]
    value: str
    original_value: str
    revisions: tuple[CellRevision, ...]


@dataclass(frozen=True)
class RevisionOverlap:
    """Revisions that target the same source cell.

    ``consistent`` is true when all revisions agree on the value. In that
    case the value is written once and all revisions are reported as
    applied.
    """

    anchor_row: int
    anchor_column: int
    span: tuple[int, int]
    revisions: tuple[CellRevision, ...]

    @property
    def values(self) -> tuple[str, ...]:
        """Return the target values, in input order."""
        return tuple(revision.value for revision in self.revisions)

    @property
    def consistent(self) -> bool:
        """Return True if every revision in this overlap agrees."""
        return len({revision.value for revision in self.revisions}) <= 1


@dataclass(frozen=True)
class RevisionConflict:
    """Revisions that target the same source cell with different values."""

    anchor_row: int
    anchor_column: int
    span: tuple[int, int]
    revisions: tuple[CellRevision, ...]
    original_value: str

    @property
    def values(self) -> tuple[str, ...]:
        """Return the conflicting values, in input order."""
        return tuple(revision.value for revision in self.revisions)


@dataclass(frozen=True)
class RevisionReport:
    """The outcome of :meth:`Table.apply_revisions`."""

    writes: tuple[RevisionWrite, ...]
    unchanged: tuple[RevisionWrite, ...]
    overlaps: tuple[RevisionOverlap, ...]
    conflicts: tuple[RevisionConflict, ...]
    skipped: tuple[CellRevision, ...]

    @property
    def applied(self) -> tuple[CellRevision, ...]:
        """Return all revisions whose values were accepted."""
        return tuple(
            revision
            for write in self.writes + self.unchanged
            for revision in write.revisions
        )

    @property
    def has_conflicts(self) -> bool:
        """Return True if any source cell received conflicting values."""
        return bool(self.conflicts)

    @property
    def changed(self) -> bool:
        """Return True if at least one source cell was changed."""
        return bool(self.writes)


HEAD_DIGITS = rc(rb'\s*+\d+').match

# Captions are optional and only one should be placed between table-start
# and the first row. Others captions are not part of the table and will
# be ignored.
FIRST_NON_CAPTION_LINE = rc(rb'\R[\t \0]*+(\|(?!\+)|!)|\Z').search

FIRST_LINEBREAK = cast(Callable[..., Match[bytes]], rc(rb'\R').search)


def head_int(value):
    if value is None:
        return 1
    match = HEAD_DIGITS(value)
    return 1 if match is None else int(match[0])


class Table(SubWikiTextWithAttrs):
    __slots__ = ('_attrs_match_cache',)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._attrs_match_cache: tuple[Match, str] | tuple[None, None] = (
            None,
            None,
        )

    @property
    def nesting_level(self) -> int:
        """Return the nesting level of self.

        The minimum nesting_level is 0. Being part of any Table increases
        the level by one.
        """
        return self._nesting_level(('Table',)) - 1

    @property
    def _table_shadow(self) -> bytearray:
        """Remove Table spans from shadow and return it."""
        shadow = self._shadow[:]
        ss = self._span_data.start
        for span in self._subspans('Table'):
            s, e = span.start, span.end
            if s == ss:
                continue
            shadow[s - ss : e - ss] = b'#' * (e - s)
        return shadow

    @property
    def _match_table(self) -> list[list[Any]]:
        """Return match_table."""
        table_shadow = self._table_shadow
        # Remove table-start and table-end marks.
        pos = FIRST_LINEBREAK(table_shadow).start()
        lsp = _lstrip_increase(table_shadow, pos)
        # Remove everything until the first row
        try:
            # while condition may raise IndexError of table is empty
            while table_shadow[lsp] not in b'!|':
                nlp = FIRST_LINEBREAK(table_shadow, lsp).start()
                pos = nlp
                lsp = _lstrip_increase(table_shadow, pos)
        except IndexError:
            return [[]]
        # Start of the first row
        match_table = []
        pos = FIRST_NON_CAPTION_LINE(table_shadow, pos).start()
        rsp = _row_separator_increase(table_shadow, pos)
        pos = -1
        while pos != rsp:
            pos = rsp
            # We have a new row.
            m = NEWLINE_CELL_MATCH(table_shadow, pos)
            # Don't add a row if there are no new cells.
            if m:
                match_row: list[Any] = []
                match_table.append(match_row)
                while m is not None:
                    match_row.append(m)
                    sep = m['sep']
                    pos = m.end()
                    if sep == b'|':
                        m = INLINE_NONHAEDER_CELL_MATCH(table_shadow, pos)
                        while m is not None:
                            match_row.append(m)
                            pos = m.end()
                            m = INLINE_NONHAEDER_CELL_MATCH(table_shadow, pos)
                    elif sep == b'!':
                        m = INLINE_HAEDER_CELL_MATCH(table_shadow, pos)
                        while m is not None:
                            match_row.append(m)
                            pos = m.end()
                            m = INLINE_HAEDER_CELL_MATCH(table_shadow, pos)
                    pos = FIRST_NON_CAPTION_LINE(table_shadow, pos).start()
                    m = NEWLINE_CELL_MATCH(table_shadow, pos)
            rsp = _row_separator_increase(table_shadow, pos)
        return match_table

    @overload
    def data(
        self,
        *,
        row: int,
        column: int,
        span: bool = ...,
        strip: bool = ...,
    ) -> str | None: ...

    @overload
    def data(
        self,
        *,
        row: int,
        column: None = ...,
        span: bool = ...,
        strip: bool = ...,
    ) -> list[str | None]: ...

    @overload
    def data(
        self,
        *,
        row: None = ...,
        column: None = ...,
        span: bool = ...,
        strip: bool = ...,
    ) -> list[list[str | None]]: ...

    @overload
    def data(
        self,
        *,
        row: None = ...,
        column: int = ...,
        span: bool = ...,
        strip: bool = ...,
    ) -> list[list[str | None]]: ...

    def data(
        self,
        span: bool = True,
        row: int | None = None,
        strip: bool = True,
        column: int | None = None,
    ) -> list[list[str | None]] | list[str | None] | str | None:
        """Return a list containing lists of row values.

        :param span: If true, calculate rows according to rowspans and colspans
            attributes. Otherwise ignore them.
        :param row: Return the specified row only. Zero-based index.
        :param column: Return the specified column only. Zero-based index.
        :param strip: strip data values

        Note: Due to the lots of complications that it may cause, this function
            won't look inside templates, parser functions, etc.
            See https://www.mediawiki.org/wiki/Extension:Pipe_Escape for how
            wiki-tables can be inserted within templates.
        """
        match_table = self._match_table
        # Note string is only used for extracting data, matching is done over
        # the shadow.
        string = self.string
        table_data: list[list[str | None]] = []
        if strip:
            for match_row in match_table:
                row_data: list[str | None] = []
                table_data.append(row_data)
                for m in match_row:
                    # Spaces after the first newline can be meaningful
                    s, e = m.span('data')
                    row_data.append(string[s:e].lstrip(' ').rstrip(WS))
        else:
            for match_row in match_table:
                row_data = []
                table_data.append(row_data)
                for m in match_row:
                    s, e = m.span('data')
                    row_data.append(string[s:e])
        if table_data and span:
            table_attrs: list[list[dict[bytes, bytes]]] = []
            for match_row in match_table:
                row_attrs: list[dict[bytes, bytes]] = []
                table_attrs.append(row_attrs)
                row_attrs_append = row_attrs.append
                for m in match_row:
                    s, e = m.span('attrs')
                    captures = ATTRS_MATCH(
                        string.encode('ascii', 'replace'), s, e
                    ).captures
                    row_attrs_append(
                        dict(
                            zip(
                                captures('attr_name'),
                                captures('attr_value'),
                            )
                        )
                    )
            table_data = _apply_attr_spans(table_attrs, table_data)
        if row is None:
            if column is None:
                return table_data
            return [r[column] for r in table_data]
        if column is None:
            return table_data[row]
        return table_data[row][column]

    @overload
    def cells(
        self,
        row: int,
        column: int,
        span: bool = True,
    ) -> Cell | None: ...
    @overload
    def cells(
        self,
        row: int,
        column: None = None,
        span: bool = True,
    ) -> list[Cell | None]: ...
    @overload
    def cells(
        self,
        row: None = None,
        column: int = 0,
        span: bool = True,
    ) -> list[list[Cell | None]]: ...
    @overload
    def cells(
        self,
        row: None = None,
        column: None = None,
        span: bool = True,
    ) -> list[list[Cell | None]]: ...
    def cells(
        self,
        row: int | None = None,
        column: int | None = None,
        span: bool = True,
    ) -> list[list[Cell | None]] | list[Cell | None] | Cell | None:
        """Return a list of lists containing Cell objects.

        :param span: If is True, rearrange the result according to colspan and
            rospan attributes.
        :param row: Return the specified row only. Zero-based index.
        :param column: Return the specified column only. Zero-based index.

        If both row and column are provided, return the relevant cell object.

        If only need the values inside cells, then use the ``data`` method
        instead.
        """
        tbl_span = self._span_data
        ss = tbl_span.start
        match_table = self._match_table
        shadow = self._shadow
        type_ = id(tbl_span)
        type_to_spans = self._type_to_spans
        spans = type_to_spans.setdefault(type_, [])
        table_cells: list[list[Cell | None]] = []
        table_attrs: list[list[dict[bytes, bytes]]] = []
        attrs_match = None
        for match_row in match_table:
            row_cells: list[Cell | None] = []
            table_cells.append(row_cells)
            if span:
                row_attrs: list[dict[bytes, bytes]] = []
                table_attrs.append(row_attrs)
                row_attrs_append = row_attrs.append
            for m in match_row:
                header = m['sep'] == b'!'
                ms, me = m.span()
                cell_span = SpanData(ss + ms, ss + me, None, shadow[ms:me])
                if span:
                    s, e = m.span('attrs')
                    # Note: ATTRS_MATCH always matches, even to empty strings.
                    # Also ATTRS_MATCH should match against the cell string
                    # so that it can be used easily as cache later in Cells.
                    attrs_match = ATTRS_MATCH(shadow[ms:me], s - ms, e - ms)
                    captures = attrs_match.captures
                    row_attrs_append(  # type: ignore
                        dict(
                            zip(captures('attr_name'), captures('attr_value'))
                        )
                    )
                old_span = next((s for s in spans if s == cell_span), None)
                if old_span is None:
                    insort_right(spans, cell_span)
                else:
                    cell_span = old_span
                row_cells.append(
                    Cell(
                        self._lststr,
                        header,
                        type_to_spans,
                        cell_span,
                        type_,
                        m,
                        attrs_match,
                    )
                )
        if table_cells and span:
            table_cells = _apply_attr_spans(table_attrs, table_cells)
        if row is None:
            if column is None:
                return table_cells
            return [r[column] for r in table_cells]
        if column is None:
            return table_cells[row]
        return table_cells[row][column]

    def apply_revisions(
        self,
        revisions: (
            Iterable[CellRevision | tuple[int, int, str]]
            | Mapping[tuple[int, int], str]
        ),
        *,
        span: bool = True,
    ) -> RevisionReport:
        """Apply several cell value revisions as one local change.

        ``revisions`` can be an iterable of :class:`CellRevision` or
        ``(row, column, value)`` tuples, or a mapping of
        ``(row, column)`` to value. Coordinates are zero-based. With
        ``span=True`` (the default) they refer to the rendered table, so
        different coordinates may resolve to the same merged cell.
        Revisions are always interpreted against the table as it was
        before this call; a value inserted by an earlier revision cannot
        move the source position of a later one.

        Revisions that resolve to the same source cell are reported in
        ``overlaps``. If they all provide the same value, that value is
        written once. Otherwise the cell is left untouched and the group
        is reported in ``conflicts``.

        Cells without a source (``None`` slots in a ragged table) are
        reported in ``skipped``.

        :return: A :class:`RevisionReport` describing every write,
            overlap, conflict, and skipped revision.
        """
        grid = cast(Any, self.cells(span=span))
        string = self.string
        anchors: dict[tuple[int, int], tuple[int, int]] = {}
        for row_index, row in enumerate(grid):
            for column_index, cell in enumerate(row):
                if cell is not None:
                    anchors.setdefault(cell.span, (row_index, column_index))
        groups: dict[
            tuple[int, int],
            list[tuple[CellRevision, int, int]],
        ] = {}
        skipped: list[CellRevision] = []
        if isinstance(revisions, Mapping):
            revisions = [
                CellRevision(row, column, value)
                for (row, column), value in revisions.items()
            ]
        for item in revisions:
            revision = _coerce_revision(item)
            row, column = revision.row, revision.column
            if row < 0 or column < 0:
                raise IndexError(
                    f'revision coordinate ({row}, {column}) '
                    'is outside the table'
                )
            try:
                cell = grid[row][column]
            except IndexError:
                raise IndexError(
                    f'revision coordinate ({row}, {column}) '
                    'is outside the table'
                ) from None
            if cell is None:
                skipped.append(revision)
                continue
            # Matches cached by ``cells`` are relative to this table, while
            # ``cell.span`` is relative to the root. Keep value positions
            # table-relative for slicing and use the resolved source span
            # only for identity.
            match = cell._match
            data_start, data_end = match.span('data')
            groups.setdefault(cell.span, []).append(
                (revision, data_start, data_end)
            )

        writes: list[RevisionWrite] = []
        unchanged: list[RevisionWrite] = []
        overlaps: list[RevisionOverlap] = []
        conflicts: list[RevisionConflict] = []
        pending: list[tuple[int, int, str]] = []
        for span_range, entries in groups.items():
            revisions_group = tuple(entry[0] for entry in entries)
            original_value = string[entries[0][1] : entries[0][2]]
            anchor_row, anchor_column = anchors[span_range]
            if len(entries) > 1:
                overlap = RevisionOverlap(
                    anchor_row,
                    anchor_column,
                    span_range,
                    revisions_group,
                )
                overlaps.append(overlap)
                if not overlap.consistent:
                    conflicts.append(
                        RevisionConflict(
                            anchor_row,
                            anchor_column,
                            span_range,
                            revisions_group,
                            original_value,
                        )
                    )
                    continue
            value = revisions_group[0].value
            write = RevisionWrite(
                anchor_row,
                anchor_column,
                span_range,
                value,
                original_value,
                revisions_group,
            )
            if value == original_value:
                unchanged.append(write)
                continue
            writes.append(write)
            pending.append((entries[0][1], entries[0][2], value))

        for value_start, value_end, value in sorted(
            pending, key=lambda write: write[0], reverse=True
        ):
            self[value_start:value_end] = value

        return RevisionReport(
            tuple(writes),
            tuple(unchanged),
            tuple(overlaps),
            tuple(conflicts),
            tuple(skipped),
        )

    @property
    def caption(self) -> str | None:
        """Caption of the table. Support get and set."""
        m = CAPTION_MATCH(self._shadow)
        if m:
            return self(*m.span('caption'))
        return None

    @caption.setter
    def caption(self, newcaption: str) -> None:
        shadow = self._shadow
        m = CAPTION_MATCH(shadow)
        if m:
            s = m.end('attrs')
            self[s if s != -1 else m.end('preattrs') : m.end('caption')] = (
                newcaption
            )
            return
        # There is no caption. Create one.
        m = FIRST_LINEBREAK(shadow)
        # Insert caption after the first one.
        self.insert(m.end(), '|+' + newcaption + m[0].decode())

    @property
    def _attrs_match(self) -> Any:  # type: ignore
        cache_match, cache_string = self._attrs_match_cache
        string = self.string
        if cache_string == string:
            return cache_match
        shadow = self._shadow
        attrs_match: Match = ATTRS_MATCH(
            shadow, 2, FIRST_LINEBREAK(shadow).start()
        )
        self._attrs_match_cache = attrs_match, string
        return attrs_match

    @property
    def caption_attrs(self) -> str | None:
        """Caption attributes. Support get and set operations."""
        m = CAPTION_MATCH(self._shadow)
        if m:
            s, e = m.span('attrs')
            if s != -1:
                return self(s, e)
        return None

    @caption_attrs.setter
    def caption_attrs(self, attrs: str) -> None:
        shadow = self._shadow
        m = CAPTION_MATCH(shadow)
        if not m:  # There is no caption-line
            flb = FIRST_LINEBREAK(shadow)
            self.insert(flb.end(), '|+' + attrs + '|' + flb[0].decode())
        else:  # Caption and attrs or Caption but no attrs
            end = m.end('attrs')
            if end != -1:
                self[m.end('preattrs') : end] = attrs

    @property
    def row_attrs(self) -> list[dict[str, str]]:
        """Row attributes.

        Use the setter of this property to set attributes for all rows.
        Note that it will overwrite all the existing attr values.
        """
        shadow = self._table_shadow
        string = self.string
        attrs = []
        append = attrs.append
        for row_match in FIND_ROWS(shadow):
            s, e = row_match.span(1)
            spans = ATTRS_MATCH(shadow, s, e).spans
            append(
                {
                    string[ns:ne]: string[vs:ve]
                    for (ns, ne), (vs, ve) in zip(
                        spans('attr_name'), spans('attr_value')
                    )
                }
            )
        return attrs

    @row_attrs.setter
    def row_attrs(self, attrs: list[dict[str, str]]):
        for row_match, attrs_dict in reversed(
            [*zip(FIND_ROWS(self._table_shadow), attrs)]
        ):
            s, e = row_match.span(1)
            del self[s:e]
            self.insert(
                s,
                ''.join(
                    [
                        f' {name}="{value}"' if value else f' {name}'
                        for name, value in attrs_dict.items()
                    ]
                ),
            )


def _apply_attr_spans(
    table_attrs: list[list[dict[bytes, bytes]]], table_data: list[list[T]]
) -> list[list[T | None]]:
    """Apply row and column spans and return table_data."""
    # The following code is based on the table forming algorithm described
    # at http://www.w3.org/TR/html5/tabular-data.html#processing-model-1
    # Numeral comments indicate the steps in that algorithm.
    # 1, 2, 10
    ycurrent = yheight = xwidth = 0
    # 4
    # The xwidth and yheight variables give the table's dimensions.
    # The table is initially empty.
    table: list[list[T | None]] = []
    append_row = table.append
    # Table.data won't call this function if table_data is empty.
    # 5
    # if not table_data:
    #     return table_data
    # 11
    downward_growing_cells: list[tuple[T | None, int, int]] = []
    # 13, 18
    # Algorithm for processing rows
    for attrs_row, row in zip(table_attrs, table_data):
        # 13.1 ycurrent is never greater than yheight
        if yheight == ycurrent:
            yheight += 1
            append_row([None] * xwidth)
        # 13.2
        xcurrent = 0
        # 13.3
        # The algorithm for growing downward-growing cells
        for cell, cellx, width in downward_growing_cells:
            r = table[ycurrent]
            for x in range(cellx, cellx + width):
                r[x] = cell
        # 13.4 will be handled by the following for-loop.
        # 13.5, 13.16
        for attrs, current_cell in zip(attrs_row, row):
            # 13.6
            attrs_get = attrs.get
            while xcurrent < xwidth and table[ycurrent][xcurrent] is not None:
                xcurrent += 1
            # 13.7
            if xcurrent == xwidth:
                # xcurrent is never greater than xwidth
                xwidth += 1
                for r in table:
                    if xwidth > len(r):
                        r.extend([None] * (xwidth - len(r)))
            # 13.8
            colspan = head_int(attrs_get(b'colspan'))
            if colspan == 0:
                # Note: colspan="0" tells the browser to span the cell to
                # the last column of the column group (colgroup)
                # http://www.w3schools.com/TAGS/att_td_colspan.asp
                colspan = 1
            # 13.9
            rowspan = head_int(attrs_get(b'rowspan'))
            # 13.10
            if rowspan == 0:
                # Note: rowspan="0" tells the browser to span the cell to the
                # last row of the table.
                # http://www.w3schools.com/tags/att_td_rowspan.asp
                cell_grows_downward = True
                rowspan = 1
            else:
                cell_grows_downward = False
            # 13.11
            if xwidth < xcurrent + colspan:
                xwidth = xcurrent + colspan
                for r in table:
                    if xwidth > len(r):
                        r.extend([None] * (xwidth - len(r)))
            # 13.12
            if yheight < ycurrent + rowspan:
                yheight = ycurrent + rowspan
                while len(table) < yheight:
                    append_row([None] * xwidth)
            # 13.13
            for y in range(ycurrent, ycurrent + rowspan):
                r = table[y]
                for x in range(xcurrent, xcurrent + colspan):
                    # If any of the slots involved already had a cell
                    # covering them, then this is a table model error.
                    # Those slots now have two cells overlapping.
                    r[x] = current_cell
                    # Skipping algorithm for assigning header cells
            # 13.14
            if cell_grows_downward:
                downward_growing_cells.append(
                    (current_cell, xcurrent, colspan)
                )
            # 13.15
            xcurrent += colspan
        # 13.16
        ycurrent += 1
    # 14
    # The algorithm for ending a row group
    # 14.1
    while ycurrent < yheight:
        # 14.1.1
        # Run the algorithm for growing downward-growing cells.
        for cell, cellx, width in downward_growing_cells:
            for x in range(cellx, cellx + width):
                table[ycurrent][x] = cell
        # 14.2.2
        ycurrent += 1
    # 14.2
    # downward_growing_cells = []
    # 20 If there exists a row or column in the table containing only
    # slots that do not have a cell anchored to them,
    # then this is a table model error.
    return table


def _coerce_revision(
    item: CellRevision | tuple[int, int, str],
) -> CellRevision:
    if isinstance(item, CellRevision):
        return item
    row, column, value = item
    return CellRevision(row, column, value)


def _lstrip_increase(shadow: bytearray, pos: int) -> int:
    """Return the new position to lstrip the shadow."""
    length = len(shadow)
    while pos < length and shadow[pos] in {0, 9, 10, 13, 32}:  # \0\t\n\r space
        pos += 1
    return pos


def _row_separator_increase(shadow: bytearray, pos: int) -> int:
    """Return the position after the starting row separator line.

    Also skips any semi-caption lines before and after the separator.
    """
    # General format of row separators: r'\|-[^\n]*\n'
    ncl = FIRST_NON_CAPTION_LINE(shadow, pos).start()
    lsp = _lstrip_increase(shadow, ncl)
    while shadow[lsp : lsp + 2] == b'|-':  # type: ignore
        # We are on a row separator line.
        pos = FIRST_LINEBREAK(shadow, lsp + 2).start()
        pos = FIRST_NON_CAPTION_LINE(shadow, pos).start()
        lsp = _lstrip_increase(shadow, pos)
    return pos
