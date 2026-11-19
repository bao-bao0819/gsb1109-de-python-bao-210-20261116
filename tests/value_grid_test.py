"""Tests for gspread.value_grid and its Worksheet wiring.

These tests use plain dictionaries and mocked HTTP clients instead of VCR
cassettes: the evidence reconciliation is pure logic over the response
shapes the Sheets API returns.
"""

import unittest
from unittest.mock import Mock

from gspread.http_client import HTTPClient
from gspread.utils import Dimension
from gspread.value_grid import (
    CellStatus,
    GridBounds,
    GridState,
    WriteEventType,
)
from gspread.worksheet import Worksheet


def make_response(
    values=None,
    response_range="Sheet1!A1:D4",
    major_dimension="ROWS",
    version=None,
):
    body = {
        "range": response_range,
        "majorDimension": major_dimension,
    }
    if values is not None:
        body["values"] = values
    if version is not None:
        body["version"] = version
    return body


def make_worksheet():
    client = HTTPClient(auth=None, session=Mock())
    client.values_get = Mock()
    client.values_batch_get = Mock()
    client.values_update = Mock()
    client.values_append = Mock()
    client.values_batch_update = Mock()
    client.values_clear = Mock()
    client.values_batch_clear = Mock()
    client.batch_update = Mock()
    properties = {
        "title": "Sheet1",
        "sheetId": 0,
        "gridProperties": {"rowCount": 1000, "columnCount": 26},
    }
    worksheet = Worksheet(
        spreadsheet=None,
        properties=properties,
        spreadsheet_id="abc123",
        client=client,
    )
    return worksheet, client


class GridBoundsTest(unittest.TestCase):
    def test_finite_range(self):
        bounds = GridBounds.from_a1("A1:B2")
        self.assertEqual(
            bounds, GridBounds(start_row=0, end_row=2, start_col=0, end_col=2)
        )

    def test_unbounded_columns(self):
        bounds = GridBounds.from_a1("A:B")
        self.assertEqual(bounds.end_row, None)
        self.assertEqual((bounds.start_col, bounds.end_col), (0, 2))

    def test_absolute_range_is_stripped(self):
        bounds = GridBounds.from_a1("Sheet1!A1:B2")
        self.assertEqual(bounds.start_row, 0)
        self.assertEqual(bounds.end_col, 2)

    def test_contains_and_overlaps(self):
        bounds = GridBounds(0, 2, 0, 2)
        self.assertTrue(bounds.contains(1, 1))
        self.assertFalse(bounds.contains(2, 1))
        other = GridBounds(1, 3, 1, 3)
        self.assertTrue(bounds.overlaps(other))
        disjoint = GridBounds(5, 6, 5, 6)
        self.assertFalse(bounds.overlaps(disjoint))

    def test_to_a1_round_trip(self):
        self.assertEqual(GridBounds.from_a1("C3:D4").to_a1(), "C3:D4")
        self.assertEqual(GridBounds.from_a1("B2").to_a1(), "B2")


class SnapshotEvidenceTest(unittest.TestCase):
    def test_short_row_does_not_mark_omitted_cells(self):
        # The API omits trailing empty cells: B1:D1 are not enumerated, so
        # there is no evidence about them at all.
        state = GridState()
        state.record_snapshot(
            make_response([["a"]], response_range="Sheet1!A1:D1"),
            requested_range="A1:D1",
        )

        self.assertEqual(state.status_of(0, 0), CellStatus.value)
        self.assertEqual(state.value_of(0, 0), "a")
        for col in range(1, 4):
            self.assertEqual(
                state.status_of(0, col),
                CellStatus.unknown,
                "omitted trailing cell must stay unknown, not be cleared",
            )

    def test_no_values_key_leaves_everything_unknown(self):
        state = GridState()
        state.record_snapshot(
            make_response(None, response_range="Sheet1!A1:B2"),
            requested_range="A1:B2",
        )
        for row in range(2):
            for col in range(2):
                self.assertEqual(state.status_of(row, col), CellStatus.unknown)

    def test_empty_string_is_empty_not_cleared(self):
        state = GridState()
        state.record_snapshot(
            make_response([[""]], response_range="Sheet1!A1:A1"),
            requested_range="A1",
        )
        self.assertEqual(state.status_of(0, 0), CellStatus.empty)
        self.assertFalse(state.is_cleared(0, 0))
        self.assertIsNone(state.value_of(0, 0))

    def test_empty_formula_result_is_empty_not_cleared(self):
        # An empty formula result and a blank cell share the same response
        # shape; only an explicit clear event may mark a cell as cleared.
        state = GridState()
        state.record_snapshot(
            make_response([["", "x"]], response_range="Sheet1!A1:B1"),
            requested_range="A1:B1",
        )
        self.assertEqual(state.status_of(0, 0), CellStatus.empty)
        self.assertEqual(state.status_of(0, 1), CellStatus.value)

    def test_row_and_column_major_dimensions_are_distinct(self):
        row_state = GridState()
        row_state.record_snapshot(
            make_response(
                [["a", "b"], ["c", "d"]],
                response_range="Sheet1!A1:B2",
                major_dimension="ROWS",
            )
        )
        col_state = GridState()
        col_state.record_snapshot(
            make_response(
                [["a", "c"], ["b", "d"]],
                response_range="Sheet1!A1:B2",
                major_dimension="COLUMNS",
            )
        )
        for row in range(2):
            for col in range(2):
                self.assertEqual(
                    row_state.value_of(row, col), col_state.value_of(row, col)
                )
        self.assertEqual(
            row_state.snapshots[0].major_dimension,
            "ROWS",
        )
        self.assertEqual(
            col_state.snapshots[0].major_dimension,
            "COLUMNS",
        )

    def test_partial_refetch_keeps_outside_evidence(self):
        state = GridState()
        state.record_snapshot(
            make_response(
                [["a", "b"], ["c", "d"]], response_range="Sheet1!A1:B2"
            ),
            requested_range="A1:B2",
        )
        # a later, narrower fetch returns a short row; it cannot erase D...
        # nor anything outside A1.
        state.record_snapshot(
            make_response([["z"]], response_range="Sheet1!A1:A1"),
            requested_range="A1",
        )
        self.assertEqual(state.value_of(0, 0), "z")
        self.assertEqual(state.value_of(0, 1), "b")
        self.assertEqual(state.value_of(1, 0), "c")
        self.assertEqual(state.value_of(1, 1), "d")

    def test_snapshot_keeps_request_and_response_ranges(self):
        state = GridState()
        state.record_snapshot(
            make_response([["a"]], response_range="Sheet1!A1:D1"),
            requested_range="A1:D1",
        )
        snapshot = state.snapshots[0]
        self.assertEqual(snapshot.requested_range, "A1:D1")
        self.assertEqual(snapshot.response_range, "Sheet1!A1:D1")


class WriteEventTest(unittest.TestCase):
    def test_write_with_empty_string_is_empty_not_cleared(self):
        state = GridState()
        state.record_write("A1:B1", [["", "x"]])
        self.assertEqual(state.status_of(0, 0), CellStatus.empty)
        self.assertEqual(state.status_of(0, 1), CellStatus.value)
        self.assertFalse(state.is_cleared(0, 0))
        event = state.events[0]
        self.assertEqual(event.type, WriteEventType.write)

    def test_explicit_clear_marks_cells_cleared(self):
        state = GridState()
        state.record_write("A1:B1", [["a", "b"]])
        state.record_clear("A1:A1")
        self.assertEqual(state.status_of(0, 0), CellStatus.cleared)
        self.assertTrue(state.is_cleared(0, 0))
        # the neighbour is untouched
        self.assertEqual(state.status_of(0, 1), CellStatus.value)

    def test_empty_read_after_clear_keeps_cleared_evidence(self):
        state = GridState()
        state.record_clear("A1:A1")
        # a later response merely shows the cell is currently blank; it must
        # not downgrade the explicit clear into a generic empty value.
        state.record_snapshot(
            make_response([[""]], response_range="Sheet1!A1:A1"),
            requested_range="A1",
        )
        self.assertEqual(state.status_of(0, 0), CellStatus.cleared)

    def test_clear_requires_finite_bounds(self):
        state = GridState()
        with self.assertRaises(ValueError):
            state.record_clear(GridBounds(0, None, 0, 1))

    def test_column_major_write(self):
        state = GridState()
        # with COLUMNS, each outer list is one column:
        # column A -> [a, c], column B -> [b, d]
        state.record_write(
            "A1:B2",
            [["a", "c"], ["b", "d"]],
            major_dimension=Dimension.cols,
        )
        self.assertEqual(state.value_of(0, 0), "a")
        self.assertEqual(state.value_of(1, 0), "c")
        self.assertEqual(state.value_of(0, 1), "b")
        self.assertEqual(state.value_of(1, 1), "d")

    def test_structural_event_invalidates_shifted_cells(self):
        state = GridState()
        state.record_write("A1:B2", [["a", "b"], ["c", "d"]])
        state.record_structural(
            GridBounds(start_row=1, end_row=None, start_col=0, end_col=None),
            detail="insert row",
        )
        # first row stays, everything below is now unknown
        self.assertEqual(state.value_of(0, 0), "a")
        self.assertEqual(state.status_of(1, 0), CellStatus.unknown)


class MergeStateTest(unittest.TestCase):
    def test_merge_same_version_combines_evidence(self):
        left = GridState()
        left.record_snapshot(
            make_response([["a"]], response_range="Sheet1!A1:A1", version="v1"),
            requested_range="A1",
        )
        right = GridState()
        right.record_snapshot(
            make_response([["b"]], response_range="Sheet1!B1:B1", version="v1"),
            requested_range="B1",
        )
        merged = GridState.merge(left, right)
        self.assertEqual(merged.value_of(0, 0), "a")
        self.assertEqual(merged.value_of(0, 1), "b")
        self.assertEqual(merged.undetermined_ranges, [])

    def test_merge_version_mismatch_protects_old_values(self):
        left = GridState()
        left.record_snapshot(
            make_response(
                [["a", "b"]], response_range="Sheet1!A1:B1", version="v1"
            ),
            requested_range="A1:B1",
        )
        right = GridState()
        # newer version, short row that would otherwise blank B1
        right.record_snapshot(
            make_response([["z"]], response_range="Sheet1!A1:B1", version="v2"),
            requested_range="A1:B1",
        )
        merged = GridState.merge(left, right)

        # the whole second response range is undetermined
        self.assertEqual(len(merged.undetermined_ranges), 1)
        self.assertEqual(merged.undetermined_ranges[0].start_col, 0)
        # right-hand observations must not have overwritten left-hand values
        self.assertEqual(merged.value_of(0, 0), "a")
        self.assertEqual(merged.value_of(0, 1), "b")

    def test_merge_without_versions_uses_newer_evidence(self):
        left = GridState()
        left.record_snapshot(
            make_response([["a"]], response_range="Sheet1!A1:A1"),
            requested_range="A1",
        )
        right = GridState()
        right.record_snapshot(
            make_response([["z"]], response_range="Sheet1!A1:A1"),
            requested_range="A1",
        )
        merged = GridState.merge(left, right)
        self.assertEqual(merged.value_of(0, 0), "z")


class WorksheetWiringTest(unittest.TestCase):
    def test_get_records_snapshot_but_lazy_state(self):
        worksheet, client = make_worksheet()
        self.assertIsNone(worksheet._grid_state)

        client.values_get.return_value = make_response(
            [["a"]], response_range="Sheet1!A1:A1"
        )
        worksheet.grid_state  # initialize tracking
        worksheet.get("A1")

        self.assertEqual(worksheet.grid_state.value_of(0, 0), "a")
        self.assertEqual(
            worksheet.grid_state.status_of(0, 1), CellStatus.unknown
        )

    def test_get_without_tracking_does_not_build_state(self):
        worksheet, client = make_worksheet()
        client.values_get.return_value = make_response(
            [["a"]], response_range="Sheet1!A1:A1"
        )
        worksheet.get("A1")
        self.assertIsNone(worksheet._grid_state)

    def test_clear_records_explicit_clear_event(self):
        worksheet, client = make_worksheet()
        client.values_clear.return_value = {"clearedRange": "Sheet1!A1:B2"}
        worksheet.grid_state
        worksheet.clear()

        self.assertTrue(worksheet.grid_state.is_cleared(0, 0))
        self.assertTrue(worksheet.grid_state.is_cleared(1, 1))
        self.assertEqual(
            worksheet.grid_state.status_of(2, 2), CellStatus.unknown
        )

    def test_update_records_written_values(self):
        worksheet, client = make_worksheet()
        client.values_update.return_value = {"updatedRange": "Sheet1!A1:B1"}
        worksheet.grid_state
        worksheet.update([["", "x"]], "A1:B1")
        self.assertEqual(
            worksheet.grid_state.status_of(0, 0), CellStatus.empty
        )
        self.assertEqual(worksheet.grid_state.value_of(0, 1), "x")

    def test_version_mismatch_through_get_marks_undetermined(self):
        worksheet, client = make_worksheet()
        client.values_get.side_effect = [
            make_response(
                [["a", "b"]], response_range="Sheet1!A1:B1", version="v1"
            ),
            make_response(
                [["z"]], response_range="Sheet1!A1:B1", version="v2"
            ),
        ]
        worksheet.grid_state
        worksheet.get("A1:B1")
        worksheet.get("A1:B1")

        self.assertEqual(len(worksheet.grid_state.undetermined_ranges), 1)
        # the short row of the mismatched response must not erase B1
        self.assertEqual(worksheet.grid_state.value_of(0, 1), "b")

    def test_snapshot_after_clash_adopts_new_version(self):
        worksheet, client = make_worksheet()
        client.values_get.side_effect = [
            make_response(
                [["a", "b"]], response_range="Sheet1!A1:B1", version="v1"
            ),
            make_response(
                [["z", "q"]], response_range="Sheet1!A1:B1", version="v2"
            ),
            make_response(
                [["w", "t"]], response_range="Sheet1!A1:B1", version="v2"
            ),
        ]
        worksheet.grid_state
        worksheet.get("A1:B1")
        worksheet.get("A1:B1")
        # a further snapshot agreeing with the new version applies normally
        worksheet.get("A1:B1")
        self.assertEqual(worksheet.grid_state.version, "v2")
        self.assertEqual(worksheet.grid_state.value_of(0, 0), "w")
        self.assertEqual(worksheet.grid_state.value_of(0, 1), "t")

    def test_batch_get_records_each_range(self):
        worksheet, client = make_worksheet()
        client.values_batch_get.return_value = {
            "valueRanges": [
                make_response([["a"]], response_range="Sheet1!A1:A1"),
                make_response([["b"]], response_range="Sheet1!C2:C2"),
            ]
        }
        worksheet.grid_state
        worksheet.batch_get(["A1", "C2"])
        self.assertEqual(worksheet.grid_state.value_of(0, 0), "a")
        self.assertEqual(worksheet.grid_state.value_of(1, 2), "b")
        # B1 sits between the two ranges and was never fetched
        self.assertEqual(
            worksheet.grid_state.status_of(0, 1), CellStatus.unknown
        )

    def test_append_records_written_values(self):
        worksheet, client = make_worksheet()
        client.values_append.return_value = {
            "updates": {"updatedRange": "Sheet1!A2:B2"}
        }
        worksheet.grid_state
        worksheet.append_rows([["", "y"]])
        self.assertEqual(worksheet.grid_state.status_of(1, 0), CellStatus.empty)
        self.assertEqual(worksheet.grid_state.value_of(1, 1), "y")

    def test_batch_clear_marks_all_ranges_cleared(self):
        worksheet, client = make_worksheet()
        client.values_batch_clear.return_value = {
            "clearedRanges": ["Sheet1!A1:A1", "Sheet1!C2:D2"]
        }
        worksheet.grid_state
        worksheet.batch_clear(["A1", "C2:D2"])
        self.assertTrue(worksheet.grid_state.is_cleared(0, 0))
        self.assertTrue(worksheet.grid_state.is_cleared(1, 2))
        self.assertTrue(worksheet.grid_state.is_cleared(1, 3))
        self.assertEqual(
            worksheet.grid_state.status_of(0, 1), CellStatus.unknown
        )

    def test_delete_rows_invalidates_shifted_region(self):
        worksheet, client = make_worksheet()
        client.batch_update.return_value = {"replies": []}
        worksheet.grid_state
        worksheet.grid_state.record_write(
            "A1:B3", [["a", "b"], ["c", "d"], ["e", "f"]]
        )
        worksheet.delete_rows(2, 2)
        # the first row is above the deleted boundary and keeps its evidence
        self.assertEqual(worksheet.grid_state.value_of(0, 0), "a")
        # every row from the deleted boundary onward shifted and is unknown
        self.assertEqual(
            worksheet.grid_state.status_of(1, 0), CellStatus.unknown
        )
        self.assertEqual(
            worksheet.grid_state.status_of(2, 0), CellStatus.unknown
        )


if __name__ == "__main__":
    unittest.main()
