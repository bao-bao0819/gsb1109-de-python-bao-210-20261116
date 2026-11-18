# -*- coding: utf-8 -*-
#
#   pyhwp : hwp file format parser in python
#   Copyright (C) 2010-2023 mete0r <https://github.com/mete0r>
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Affero General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Affero General Public License for more details.
#
#   You should have received a copy of the GNU Affero General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
''' Stitch page-split table fragments back into one logical table.

In HWP documents a table that crosses a page boundary is stored as one
table control per page.  A merged cell (rowspan > 1) that crosses the
boundary is repeated in the next page fragment, sometimes with its text
duplicated.  Naively exporting every physical cell therefore emits the
same merged cell -- e.g. an amount in a contract table -- twice, and any
sum over the column counts it twice.

This module rebuilds the logical table:

- :class:`StitchedTable` generates logical row records from the physical
  fragments, keeping the correspondence between logical cells and the
  original physical cells (:attr:`LogicalCell.originals`), across both
  rows (rowspan) and pages (seam links).
- Positions covered by a merged cell anchored in a previous row are
  "empty continuations" (:attr:`Slot.continuation`); they are kept
  distinct from genuinely empty cells (an anchor slot whose text is
  ``''``).
- Where a dangling merged cell meets the next fragment there are often
  two plausible seams -- continue the cell across the page break, or
  clamp it and treat the repeated cell as a new one.  Both candidates
  are kept (:attr:`StitchedTable.boundaries`); nothing is force-joined
  just to complete a row.  Asking for rows while a boundary is
  ambiguous raises :class:`AmbiguousSeamError` until
  :meth:`StitchedTable.confirm` picks a candidate.
- Once seams are resolved, :meth:`StitchedTable.total` sums a column
  counting every logical cell exactly once, no matter how many rows it
  spans or how many fragments repeat it.
'''
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from collections import namedtuple
from decimal import Decimal
from decimal import InvalidOperation
import logging


logger = logging.getLogger(__name__)


class Cell(namedtuple('Cell', 'row col rowspan colspan text')):
    ''' A physical table cell as parsed from a single page fragment. '''

    def __new__(cls, row, col, rowspan=1, colspan=1, text=''):
        return super(Cell, cls).__new__(cls, row, col, rowspan, colspan,
                                        text)


class OriginalCell(namedtuple('OriginalCell',
                              'fragment page row col rowspan colspan text')):
    ''' A physical cell tied to the page fragment it was parsed from. '''

    @property
    def key(self):
        ''' Identity of this physical cell: (fragment, row, col). '''
        return (self.fragment, self.row, self.col)


class TableFragment(object):
    ''' Physical cells of a table fragment drawn on a single page. '''

    def __init__(self, cells, ncols=None, nrows=None, page=None):
        cells = tuple(cells)
        if ncols is None:
            ncols = max([c.col + c.colspan for c in cells] or [0])
        if nrows is None:
            nrows = max([c.row + 1 for c in cells] or [0])
        self.cells = cells
        self.ncols = ncols
        self.nrows = nrows
        self.page = page


class Seam(namedtuple('Seam', 'links')):
    ''' A candidate seam between two consecutive page fragments.

    ``links`` is a frozenset of ``(a_key, b_key)`` original-cell key
    pairs considered to be the same logical cell continued across the
    page break.  An empty ``links`` means the fragments do not continue
    each other: dangling merged cells of the first fragment are clamped
    at the page break and the repeated cells start new logical cells.
    '''

    @property
    def joins(self):
        ''' Whether this seam continues any cell across the boundary. '''
        return bool(self.links)


class AmbiguousSeamError(Exception):
    ''' A boundary has multiple seam candidates and none is confirmed. '''

    def __init__(self, boundary, candidates):
        self.boundary = boundary
        self.candidates = list(candidates)
        super(AmbiguousSeamError, self).__init__(
            'ambiguous seam at boundary %d: %d candidates; '
            'call confirm() to choose one' % (boundary, len(candidates)))


def _texts_compatible(a_text, b_text):
    return a_text == b_text or not a_text or not b_text


def _coverage(cells):
    ''' Map every covered (row, col) position to its cell. '''
    covered = {}
    for cell in cells:
        for r in range(cell.row, cell.row + cell.rowspan):
            for c in range(cell.col, cell.col + cell.colspan):
                covered.setdefault((r, c), cell)
    return covered


def _seam_candidates(a_cells, a_nrows, b_cells):
    ''' Generate seam candidates between two consecutive fragments.

    The clamp candidate (no links) is always kept.  A join candidate is
    added for every dangling merged cell of the first fragment whose
    continuation positions in the second fragment hold compatible cells
    (same colspan; texts equal or either side empty).  If only some of
    the dangling cells can be continued, the partial join is kept as a
    candidate of its own instead of being forced.
    '''
    dangling = [a for a in a_cells if a.row + a.rowspan > a_nrows]
    candidates = [Seam(frozenset())]
    if not dangling:
        return candidates
    covered = _coverage(b_cells)
    links = set()
    for a in dangling:
        overhang = a.row + a.rowspan - a_nrows
        seen = set()
        cell_links = []
        for j in range(overhang):
            b = covered.get((j, a.col))
            if (b is None or b.colspan != a.colspan or
                    not _texts_compatible(a.text, b.text)):
                break
            if b.key not in seen:
                seen.add(b.key)
                cell_links.append((a.key, b.key))
        else:
            links.update(cell_links)
    if links:
        candidates.append(Seam(frozenset(links)))
    return candidates


def _original_cells(fragment, index):
    page = fragment.page if fragment.page is not None else index
    return [
        OriginalCell(index, page, c.row, c.col, c.rowspan, c.colspan,
                     c.text)
        for c in fragment.cells
    ]


def find_seam_candidates(fragment_a, fragment_b):
    ''' Seam candidates between two standalone fragments.

    Original-cell keys in the returned seams use fragment indices 0 and
    1 for ``fragment_a`` and ``fragment_b`` respectively.
    '''
    a_cells = _original_cells(fragment_a, 0)
    b_cells = _original_cells(fragment_b, 1)
    return _seam_candidates(a_cells, fragment_a.nrows, b_cells)


class LogicalCell(object):
    ''' A cell of the logical table.

    Wraps one or more original cells: duplicates of the same merged
    cell repeated in the next page fragment are absorbed here, so the
    value exists exactly once no matter how many fragments carry it.
    '''

    def __init__(self, original):
        self.originals = [original]
        self.text = original.text
        self.rowspan = original.rowspan
        self.colspan = original.colspan

    def absorb(self, original):
        ''' Merge a repeated original cell into this logical cell. '''
        self.originals.append(original)
        if not self.text and original.text:
            self.text = original.text

    @property
    def fragments(self):
        ''' Indices of the fragments this cell was parsed from. '''
        return sorted(set(o.fragment for o in self.originals))

    @property
    def pages(self):
        ''' Page numbers this cell appears on. '''
        return sorted(set(o.page for o in self.originals))

    @property
    def spans_pages(self):
        ''' Whether this cell continues across a page break. '''
        return len(self.fragments) > 1

    def __repr__(self):
        return 'LogicalCell(%r, originals=%r)' % (
            self.text, [o.key for o in self.originals])


class Slot(namedtuple('Slot', 'cell continuation anchor')):
    ''' One column position of a logical row.

    ``continuation`` is True for positions covered by a merged cell
    anchored in a previous logical row -- including positions carried
    by a repeated cell in the next page fragment.  Such positions are
    empty continuations: they hold no value of their own, which is
    distinct from a genuinely empty cell (an anchor slot whose text is
    ``''``).
    '''

    @property
    def is_empty_continuation(self):
        return self.continuation

    @property
    def value(self):
        ''' Cell text for anchor slots; None for continuations. '''
        if self.anchor:
            return self.cell.text
        return None


class LogicalRow(object):
    ''' A row of the logical table.

    ``slots`` is aligned to the table columns; entries are :class:`Slot`
    or None for positions not covered by any cell.  ``sources`` holds
    the ``(fragment_index, row)`` pairs of the physical rows this
    logical row was built from.
    '''

    def __init__(self, slots, sources):
        self.slots = tuple(slots)
        self.sources = tuple(sources)

    def values(self):
        ''' Anchor texts per column; None for continuations/gaps. '''
        return [slot.value if slot is not None else None
                for slot in self.slots]

    def slot(self, col):
        return self.slots[col]


class StitchedTable(object):
    ''' A logical table stitched from page fragments. '''

    def __init__(self, fragments):
        self.fragments = list(fragments)
        self._ocells = [_original_cells(f, i)
                        for i, f in enumerate(self.fragments)]
        self.boundaries = [
            _seam_candidates(self._ocells[i], self.fragments[i].nrows,
                             self._ocells[i + 1])
            for i in range(len(self.fragments) - 1)
        ]
        self._confirmed = [None] * len(self.boundaries)
        self._rows_cache = None

    @property
    def seams(self):
        ''' Candidate seams per boundary between consecutive fragments. '''
        return self.boundaries

    @property
    def resolved(self):
        ''' Whether every boundary has exactly one usable seam. '''
        return all(self._resolved_or_none(i) is not None
                   for i in range(len(self.boundaries)))

    def _resolved_or_none(self, boundary):
        confirmed = self._confirmed[boundary]
        if confirmed is not None:
            return confirmed
        candidates = self.boundaries[boundary]
        if len(candidates) == 1:
            return candidates[0]
        return None

    def resolved_seam(self, boundary):
        ''' The confirmed seam, or the only candidate if unambiguous. '''
        seam = self._resolved_or_none(boundary)
        if seam is None:
            raise AmbiguousSeamError(boundary, self.boundaries[boundary])
        return seam

    def confirm(self, boundary, seam):
        ''' Confirm a seam for a boundary, resolving its ambiguity.

        ``seam`` is normally one of ``boundaries[boundary]``; a custom
        seam is accepted as long as its links reference original cells
        of the two adjacent fragments.
        '''
        a_keys = set(c.key for c in self._ocells[boundary])
        b_keys = set(c.key for c in self._ocells[boundary + 1])
        for a_key, b_key in seam.links:
            if a_key not in a_keys:
                raise ValueError('unknown cell key: %r' % (a_key,))
            if b_key not in b_keys:
                raise ValueError('unknown cell key: %r' % (b_key,))
        self._confirmed[boundary] = seam
        self._rows_cache = None

    def rows(self):
        ''' Logical row records generated from the logical structure. '''
        if self._rows_cache is None:
            self._rows_cache = self._build_rows()
        return self._rows_cache

    def records(self):
        ''' Logical rows as plain per-column value lists. '''
        return [row.values() for row in self.rows()]

    def _build_rows(self):
        rows = []
        cell_index = {}
        active = {}
        for fi, fragment in enumerate(self.fragments):
            links = {}
            if fi > 0:
                seam = self.resolved_seam(fi - 1)
                if not seam.joins:
                    active = {}
                for a_key, b_key in seam.links:
                    logical = cell_index.get(a_key)
                    if logical is not None:
                        links[b_key] = logical
            cells_by_row = {}
            for cell in self._ocells[fi]:
                cells_by_row.setdefault(cell.row, []).append(cell)
            for r in range(fragment.nrows):
                slots = {}
                next_active = {}
                for col in sorted(active):
                    logical, remaining = active[col]
                    slots[col] = Slot(logical, True, False)
                    if remaining > 1:
                        next_active[col] = [logical, remaining - 1]
                for cell in sorted(cells_by_row.get(r, []),
                                   key=lambda c: c.col):
                    cols = list(range(cell.col, cell.col + cell.colspan))
                    logical = links.get(cell.key)
                    if logical is not None:
                        logical.absorb(cell)
                        for c in cols:
                            slots[c] = Slot(logical, True, False)
                            next_active.pop(c, None)
                    else:
                        if any(c in slots for c in cols):
                            logger.warning(
                                'cell %r overlaps a continuation slot; '
                                'the continuation is dropped', cell.key)
                            for c in cols:
                                next_active.pop(c, None)
                        logical = LogicalCell(cell)
                        cell_index[cell.key] = logical
                        for c in cols:
                            slots[c] = Slot(logical, False, c == cell.col)
                    if cell.rowspan > 1:
                        for c in cols:
                            next_active[c] = [logical, cell.rowspan - 1]
                active = next_active
                row_slots = [slots.get(c) for c in range(fragment.ncols)]
                rows.append(LogicalRow(row_slots, [(fi, r)]))
        return rows

    def total(self, col):
        ''' Sum the numeric values of a column over the logical rows.

        Each logical cell contributes once, no matter how many logical
        rows it spans or how many page fragments repeat it: the same
        original cell is never counted twice.
        '''
        seen = set()
        result = Decimal(0)
        for row in self.rows():
            if col >= len(row.slots):
                continue
            slot = row.slots[col]
            if slot is None or not slot.anchor:
                continue
            cell = slot.cell
            if id(cell) in seen:
                continue
            seen.add(id(cell))
            amount = parse_amount(cell.text)
            if amount is not None:
                result += amount
        return result


def parse_amount(text):
    ''' Parse an amount like '1,234,500' or '₩1,234원' into Decimal.

    Returns None for texts that carry no usable number.
    '''
    if not text:
        return None
    kept = [ch for ch in text if ch.isdigit() or ch in '.-']
    cleaned = ''.join(kept)
    if cleaned in ('', '-', '.', '-.'):
        return None
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None
