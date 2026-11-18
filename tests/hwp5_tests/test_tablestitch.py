# -*- coding: utf-8 -*-
#
#   pyhwp : hwp file format parser in python
#   Copyright (C) 2010-2023 mete0r <https://github.com/mete0r>
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Affero General Public License as published to
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
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from decimal import Decimal
from unittest import TestCase

from hwp5.tablestitch import AmbiguousSeamError
from hwp5.tablestitch import Cell
from hwp5.tablestitch import Seam
from hwp5.tablestitch import StitchedTable
from hwp5.tablestitch import TableFragment
from hwp5.tablestitch import find_seam_candidates
from hwp5.tablestitch import parse_amount


def contract_fragments(duplicate_text='2,500,000'):
    ''' A contract table split across two pages.

    Page 1 holds a merged amount cell ('2,500,000', rowspan 3) that
    dangles over the page break; page 2 repeats it in its first row.
    '''
    page1 = TableFragment([
        Cell(0, 0, text='항목'), Cell(0, 1, text='금액'),
        Cell(1, 0, text='설계비'), Cell(1, 1, rowspan=3, text='2,500,000'),
        Cell(2, 0, text='감리비'),
    ], ncols=2, nrows=3, page=1)
    page2 = TableFragment([
        Cell(0, 0, text='시공비'), Cell(0, 1, text=duplicate_text),
        Cell(1, 0, text='기타비'), Cell(1, 1, text='300,000'),
    ], ncols=2, nrows=2, page=2)
    return [page1, page2]


def join_seam(table):
    for seam in table.seams[0]:
        if seam.joins:
            return seam
    raise AssertionError('no join candidate')


class SeamCandidatesTest(TestCase):

    def test_clamp_and_join_candidates_are_kept(self):
        page1, page2 = contract_fragments()
        candidates = find_seam_candidates(page1, page2)
        self.assertEqual(2, len(candidates))
        clamp, join = candidates
        self.assertEqual(frozenset(), clamp.links)
        self.assertEqual(
            frozenset([((0, 1, 1), (1, 0, 1))]),
            join.links,
        )

    def test_no_dangling_cells_means_single_candidate(self):
        fragment = TableFragment([
            Cell(0, 0, text='a'), Cell(0, 1, text='b'),
        ], ncols=2, nrows=1)
        candidates = find_seam_candidates(fragment, fragment)
        self.assertEqual([Seam(frozenset())], candidates)

    def test_incompatible_continuation_leaves_only_clamp(self):
        page1, _ = contract_fragments()
        page2 = TableFragment([
            Cell(0, 0, text='시공비'), Cell(0, 1, text='9,999'),
        ], ncols=2, nrows=1, page=2)
        candidates = find_seam_candidates(page1, page2)
        self.assertEqual([Seam(frozenset())], candidates)


class StitchedTableTest(TestCase):

    def test_ambiguous_seam_is_not_force_joined(self):
        table = StitchedTable(contract_fragments())
        self.assertFalse(table.resolved)
        with self.assertRaises(AmbiguousSeamError) as cm:
            table.rows()
        self.assertEqual(0, cm.exception.boundary)
        self.assertEqual(2, len(cm.exception.candidates))

    def test_confirmed_join_generates_logical_rows(self):
        table = StitchedTable(contract_fragments())
        table.confirm(0, join_seam(table))
        self.assertTrue(table.resolved)
        self.assertEqual([
            ['항목', '금액'],
            ['설계비', '2,500,000'],
            ['감리비', None],
            ['시공비', None],
            ['기타비', '300,000'],
        ], table.records())

    def test_confirmed_join_does_not_count_original_twice(self):
        table = StitchedTable(contract_fragments())
        table.confirm(0, join_seam(table))
        self.assertEqual(Decimal('2800000'), table.total(1))

    def test_confirmed_clamp_counts_repeated_cell_again(self):
        table = StitchedTable(contract_fragments())
        clamp = Seam(frozenset())
        table.confirm(0, clamp)
        self.assertEqual(5, len(table.rows()))
        # the repeated cell starts a new logical cell, so the amount is
        # counted twice: this is the reading confirm() exists to reject
        self.assertEqual(Decimal('5300000'), table.total(1))

    def test_empty_continuation_is_distinct_from_real_empty(self):
        page1, _ = contract_fragments()
        # page 2 carries an empty continuation cell for the merged
        # amount, plus a genuinely empty amount cell in a new row
        page2 = TableFragment([
            Cell(0, 0, text='시공비'), Cell(0, 1, text=''),
            Cell(1, 0, text='기타비'), Cell(1, 1, text=''),
        ], ncols=2, nrows=2, page=2)
        table = StitchedTable([page1, page2])
        table.confirm(0, join_seam(table))
        rows = table.rows()

        continuation = rows[3].slot(1)
        self.assertTrue(continuation.is_empty_continuation)
        self.assertIsNone(continuation.value)

        real_empty = rows[4].slot(1)
        self.assertFalse(real_empty.is_empty_continuation)
        self.assertEqual('', real_empty.value)

        # the logical cell keeps the anchor text even though the
        # continuation cell in page 2 is empty
        self.assertEqual('2,500,000', continuation.cell.text)
        self.assertEqual(Decimal('2500000'), table.total(1))

    def test_originals_mapping_across_rows_and_pages(self):
        table = StitchedTable(contract_fragments())
        table.confirm(0, join_seam(table))
        rows = table.rows()

        merged = rows[1].slot(1).cell
        self.assertIs(merged, rows[2].slot(1).cell)
        self.assertIs(merged, rows[3].slot(1).cell)
        self.assertEqual(
            [(0, 1, 1), (1, 0, 1)],
            [o.key for o in merged.originals],
        )
        self.assertEqual([1, 2], merged.pages)
        self.assertTrue(merged.spans_pages)

        self.assertEqual(((0, 1),), rows[1].sources)
        self.assertEqual(((1, 0),), rows[3].sources)

    def test_rowspan_within_one_fragment_counts_once(self):
        fragment = TableFragment([
            Cell(0, 0, text='a'), Cell(0, 1, rowspan=2, text='100'),
            Cell(1, 0, text='b'),
            Cell(2, 0, text='c'), Cell(2, 1, text='50'),
        ], ncols=2, nrows=3)
        table = StitchedTable([fragment])
        self.assertTrue(table.resolved)
        self.assertEqual(Decimal('150'), table.total(1))

    def test_confirm_rejects_unknown_cell_keys(self):
        table = StitchedTable(contract_fragments())
        with self.assertRaises(ValueError):
            table.confirm(0, Seam(frozenset([((0, 1, 1), (1, 9, 9))])))

    def test_confirm_can_be_changed(self):
        table = StitchedTable(contract_fragments())
        table.confirm(0, Seam(frozenset()))
        self.assertEqual(Decimal('5300000'), table.total(1))
        table.confirm(0, join_seam(table))
        self.assertEqual(Decimal('2800000'), table.total(1))


class ParseAmountTest(TestCase):

    def test_plain_number(self):
        self.assertEqual(Decimal('1234500'), parse_amount('1234500'))

    def test_thousands_separator(self):
        self.assertEqual(Decimal('1234500'), parse_amount('1,234,500'))

    def test_currency_marks(self):
        self.assertEqual(Decimal('1234'), parse_amount('₩1,234원'))

    def test_decimal_number(self):
        self.assertEqual(Decimal('1.5'), parse_amount('1.5'))

    def test_empty_text(self):
        self.assertIsNone(parse_amount(''))
        self.assertIsNone(parse_amount(None))

    def test_non_numeric_text(self):
        self.assertIsNone(parse_amount('합계'))
