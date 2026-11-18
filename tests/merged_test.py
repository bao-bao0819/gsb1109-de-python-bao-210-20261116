import unittest

from gspread.merged import CellRole, MergedCell, MergedGrid, MergeGranularity


def merge(start_row, end_row, start_col, end_col):
    return {
        "startRowIndex": start_row,
        "endRowIndex": end_row,
        "startColumnIndex": start_col,
        "endColumnIndex": end_col,
    }


class MergedGridTest(unittest.TestCase):
    def make_quote_grid(self):
        """A quotation sheet with one amount merged over three rows."""
        values = [
            ["item", "qty", "amount"],
            ["a", 1, 300],
            ["b", 2, None],
            ["c", 3, None],
        ]
        merges = [merge(1, 4, 2, 3)]
        return MergedGrid(values, merges)

    def test_master_inherited_and_plain_roles_are_distinct(self):
        grid = self.make_quote_grid()

        master = grid.cell(1, 2)
        self.assertEqual(master, MergedCell(300, CellRole.master, "C2"))

        inherited = grid.cell(2, 2)
        self.assertEqual(inherited.value, 300)
        self.assertEqual(inherited.role, CellRole.inherited)

        plain = grid.cell(0, 0)
        self.assertEqual(plain, MergedCell("item", CellRole.plain, "A1"))

    def test_every_row_refers_back_to_the_source_cell(self):
        grid = self.make_quote_grid()

        self.assertEqual(grid.cell(1, 2).source, "C2")
        self.assertEqual(grid.cell(2, 2).source, "C2")
        self.assertEqual(grid.cell(3, 2).source, "C2")

    def test_sum_uses_the_confirmed_business_granularity(self):
        grid = self.make_quote_grid()

        self.assertEqual(
            grid.sum_column(2, granularity=MergeGranularity.merged),
            300,
        )
        self.assertEqual(
            grid.sum_column(2, granularity=MergeGranularity.rows),
            900,
        )

    def test_column_values_merged_counts_each_region_once(self):
        grid = self.make_quote_grid()

        self.assertEqual(
            grid.column_values(2, granularity=MergeGranularity.merged),
            ["amount", 300],
        )
        self.assertEqual(
            grid.column_values(2, granularity=MergeGranularity.rows),
            ["amount", 300, 300, 300],
        )

    def test_granularity_accepts_plain_string(self):
        grid = self.make_quote_grid()

        self.assertEqual(grid.sum_column(2, granularity="merged"), 300)

    def test_sum_column_ignores_non_numeric_values(self):
        values = [["total"], [1.5], ["n/a"], [None], [2]]
        grid = MergedGrid(values)

        self.assertEqual(grid.sum_column(0, granularity=MergeGranularity.merged), 3.5)

    def test_update_merges_only_recomputes_affected_rows(self):
        values = [
            ["item", "qty", "amount", "note"],
            ["a", 1, 300, "x"],
            ["b", 2, None, "y"],
            ["c", 3, None, "z"],
            ["d", 4, 50, "w"],
        ]
        merges = [merge(1, 4, 2, 3), merge(4, 5, 0, 2)]
        grid = MergedGrid(values, merges)

        cached_rows = [grid.row(i) for i in range(grid.row_count)]

        # shrink the amount merge from rows 2-4 to rows 2-3
        affected = grid.update_merges([merge(1, 3, 2, 3), merge(4, 5, 0, 2)])

        self.assertEqual(affected, [3])
        for row_index in (0, 1, 2, 4):
            self.assertIs(grid.row(row_index), cached_rows[row_index])
        self.assertIsNot(grid.row(3), cached_rows[3])

    def test_shrunk_merge_reverts_to_own_raw_values(self):
        values = [
            ["item", "qty", "amount"],
            ["a", 1, 300],
            ["b", 2, None],
            ["c", 3, None],
        ]
        grid = MergedGrid(values, [merge(1, 4, 2, 3)])
        self.assertEqual(grid.cell(3, 2).value, 300)

        grid.update_merges([merge(1, 3, 2, 3)])

        reverted = grid.cell(3, 2)
        self.assertEqual(reverted, MergedCell(None, CellRole.plain, "C4"))

    def test_unmerge_reverts_all_rows_to_plain(self):
        grid = self.make_quote_grid()

        affected = grid.update_merges([])

        self.assertEqual(affected, [1, 2, 3])
        for row_index in range(grid.row_count):
            for record in grid.row(row_index):
                self.assertEqual(record.role, CellRole.plain)

    def test_empty_master_is_not_filled_from_neighbours(self):
        values = [
            ["item", "qty", "amount"],
            ["a", 999, None],
            ["b", 2, None],
        ]
        grid = MergedGrid(values, [merge(1, 3, 2, 3)])

        master = grid.cell(1, 2)
        self.assertEqual(master, MergedCell(None, CellRole.master, "C2"))

        inherited = grid.cell(2, 2)
        self.assertEqual(inherited.value, None)
        self.assertEqual(inherited.role, CellRole.inherited)
        self.assertEqual(inherited.source, "C2")

        self.assertEqual(grid.sum_column(2, granularity=MergeGranularity.merged), 0)

    def test_merge_outside_grid_is_ignored(self):
        values = [[1, 2], [3, 4]]
        grid = MergedGrid(values, [merge(7, 9, 7, 9)])

        for row_index in range(grid.row_count):
            for record in grid.row(row_index):
                self.assertEqual(record.role, CellRole.plain)

    def test_merge_starting_before_the_grid_is_ignored(self):
        values = [[1, 2], [3, 4]]
        grid = MergedGrid(values, [merge(0, 2, 0, 2)], start_row_index=1)

        for row_index in range(grid.row_count):
            for record in grid.row(row_index):
                self.assertEqual(record.role, CellRole.plain)

    def test_merge_is_clipped_to_the_grid(self):
        values = [[1, 2], [3, 4]]
        grid = MergedGrid(values, [merge(0, 10, 0, 10)])

        self.assertEqual(grid.cell(0, 0).role, CellRole.master)
        self.assertEqual(grid.cell(1, 1).role, CellRole.inherited)
        self.assertEqual(grid.cell(1, 1).source, "A1")

    def test_offsets_are_reflected_in_source_labels(self):
        values = [[10, None], [None, None]]
        grid = MergedGrid(
            values,
            [merge(2, 4, 3, 4)],
            start_row_index=2,
            start_col_index=3,
        )

        self.assertEqual(grid.cell(0, 0), MergedCell(10, CellRole.master, "D3"))
        self.assertEqual(grid.cell(1, 0).source, "D3")

    def test_from_metadata_reads_the_merges_key(self):
        metadata = {"properties": {"sheetId": 0}, "merges": [merge(0, 2, 0, 1)]}
        grid = MergedGrid.from_metadata(metadata, [[5, None], [None, None]])

        self.assertEqual(grid.cell(1, 0).role, CellRole.inherited)
        self.assertEqual(grid.cell(1, 0).source, "A1")

    def test_ragged_values_are_padded_with_none(self):
        values = [[1, 2], [3]]
        grid = MergedGrid(values)

        self.assertEqual(grid.cell(1, 1), MergedCell(None, CellRole.plain, "B2"))

    def test_out_of_bounds_indices_raise(self):
        grid = MergedGrid([[1]])

        with self.assertRaises(IndexError):
            grid.row(1)
        with self.assertRaises(IndexError):
            grid.cell(0, 1)


if __name__ == "__main__":
    unittest.main()
