import json
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pytest

from gspread.http_client import HTTPClient
from gspread.worksheet import Worksheet

SHEET_ID = 7
SHEET_TITLE = "Prices"


class FakeHTTPClient(HTTPClient):
    """HTTP client that serves fixed metadata and records batch updates."""

    def __init__(
        self,
        metadata: Mapping[str, Any],
        grid_data: Mapping[str, Any],
    ) -> None:
        self.metadata = metadata
        self.grid_data = grid_data
        self.metadata_calls: List[Tuple[str, Dict[str, Any]]] = []
        self.batch_updates: List[Tuple[str, Mapping[str, Any]]] = []

    def spreadsheets_get(
        self, id: str, params: Optional[Mapping[str, Any]] = None
    ) -> Mapping[str, Any]:
        self.metadata_calls.append((id, dict(params or {})))
        if params and params.get("includeGridData") == "true":
            return self.grid_data
        return self.metadata

    def batch_update(
        self, id: str, body: Optional[Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        self.batch_updates.append((id, body or {}))
        return {"replies": []}


def cell_data(
    value: Any = None,
    note: Optional[str] = None,
    formula: Optional[str] = None,
    number: Optional[float] = None,
    text: Optional[str] = None,
) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    if formula is not None:
        data["userEnteredValue"] = {"formulaValue": formula}
    elif number is not None:
        data["userEnteredValue"] = {"numberValue": number}
    elif text is not None:
        data["userEnteredValue"] = {"stringValue": text}
    elif value is not None:
        data["userEnteredValue"] = {"stringValue": value}
    if note is not None:
        data["note"] = note
    # The fake includes formatting to make sure it is never echoed back.
    data["userEnteredFormat"] = {"textFormat": {"bold": True}}
    return data


def make_worksheet(
    grid_rows: Optional[Dict[Tuple[int, int], Mapping[str, Any]]] = None,
    merges: Optional[List[Dict[str, int]]] = None,
    grid_spans: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Worksheet, FakeHTTPClient]:
    grid_rows = grid_rows or {}
    merges = merges or []

    metadata = {
        "sheets": [
            {
                "properties": {"sheetId": 1, "title": "Other"},
                "merges": [
                    {
                        "startRowIndex": 0,
                        "endRowIndex": 1,
                        "startColumnIndex": 0,
                        "endColumnIndex": 2,
                    }
                ],
            },
            {
                "properties": {"sheetId": SHEET_ID, "title": SHEET_TITLE},
                "merges": merges,
            },
        ]
    }

    if grid_spans is None:
        # One GridData block per requested anchor, mirroring the Sheets API
        # response: a block starts at the requested cell and holds one
        # one-cell row.
        grid_spans = [
            {
                "startRow": row,
                "startColumn": col,
                "rowData": [{"values": [data]}],
            }
            for (row, col), data in sorted(grid_rows.items())
        ]

    # The Sheets API may return one sheet entry per requested range.
    grid_data = {
        "sheets": [{"properties": {"sheetId": 1}, "data": []}]
        + [{"properties": {"sheetId": SHEET_ID}, "data": [span]} for span in grid_spans]
    }

    client = FakeHTTPClient(metadata, grid_data)
    worksheet = Worksheet(
        None,  # type: ignore[arg-type]
        {
            "sheetId": SHEET_ID,
            "title": SHEET_TITLE,
            "index": 0,
            "gridProperties": {"rowCount": 100, "columnCount": 20},
        },
        spreadsheet_id="spreadsheet-id",
        client=client,
    )
    return worksheet, client


def request_cells(client: FakeHTTPClient) -> Dict[str, Dict[str, Any]]:
    """Index the sent requests by their A1-style anchor for assertions."""
    cells = {}
    for _, body in client.batch_updates:
        for request in body["requests"]:
            cell_request = request["updateCells"]
            range_ = cell_request["range"]
            row = range_["startRowIndex"] + 1
            col = range_["startColumnIndex"] + 1
            address = "{}{}".format(chr(ord("A") + col - 1), row)
            cells[address] = cell_request
    return cells


def test_dry_run_returns_plan_and_conflicts_without_writing() -> None:
    worksheet, client = make_worksheet(
        grid_rows={
            (0, 0): cell_data(number=19.99, note="old"),
            (0, 1): cell_data(formula="=A1*2"),
            (0, 2): cell_data(),
        }
    )

    result = worksheet.writeback_cells(
        [
            {"cell": "A1", "field": "price", "expected": "19.99", "value": 20.5},
            {"cell": "A1", "field": "note", "expected": "old", "value": "new"},
            {"cell": "B1", "field": "value", "expected": "=A1*2", "value": "=A1*3"},
            {"cell": "C1", "field": "note", "expected": "old", "value": "new note"},
            {"cell": "C1", "field": "value", "expected": None, "value": 1},
        ],
        dry_run=True,
    )

    assert result["dry_run"] is True
    assert client.batch_updates == []
    assert len(result["updates"]) == 4
    assert [entry["status"] for entry in result["updates"]] == ["planned"] * 4

    conflicts = result["conflicts"]
    assert len(conflicts) == 1
    assert conflicts[0]["cell"] == "C1"
    assert conflicts[0]["field"] == "note"
    assert conflicts[0]["current"] == ""
    assert conflicts[0]["reason"] == "original_value_mismatch"

    requests = request_cells(client) if client.batch_updates else {}
    assert requests == {}
    assert result["requests"]
    by_cell = {
        request["updateCells"]["range"]["startColumnIndex"]: request["updateCells"]
        for request in result["requests"]
    }
    assert by_cell[0]["fields"] == "userEnteredValue,note"
    assert by_cell[1]["fields"] == "userEnteredValue"
    assert by_cell[2]["fields"] == "userEnteredValue"
    assert by_cell[0]["rows"][0]["values"][0] == {
        "userEnteredValue": {"numberValue": 20.5},
        "note": "new",
    }
    assert "userEnteredFormat" not in json.dumps(result["requests"])


def test_applies_only_non_conflicting_cells() -> None:
    worksheet, client = make_worksheet(
        grid_rows={
            (0, 0): cell_data(number=19.99, note="old"),
            (0, 2): cell_data(note="keep"),
        }
    )

    result = worksheet.writeback_cells(
        [
            {"cell": "A1", "field": "price", "expected": 19.99, "value": 20},
            {"cell": "A1", "field": "note", "expected": "old", "value": "new"},
            {"cell": "C1", "field": "note", "expected": "changed", "value": "x"},
        ]
    )

    assert result["dry_run"] is False
    assert len(client.batch_updates) == 1
    assert len(result["updates"]) == 2
    assert [entry["status"] for entry in result["updates"]] == ["applied", "applied"]
    assert len(result["conflicts"]) == 1
    assert result["conflicts"][0]["cell"] == "C1"

    cells = request_cells(client)
    assert set(cells) == {"A1"}
    assert cells["A1"]["fields"] == "userEnteredValue,note"
    # C1 was not written and its note remains untouched.
    assert cells["A1"]["rows"][0]["values"][0]["note"] == "new"


def test_merged_cell_is_checked_and_written_at_the_anchor() -> None:
    worksheet, client = make_worksheet(
        grid_rows={(1, 0): cell_data(number=5, note="merged")},
        merges=[
            {
                "startRowIndex": 1,
                "endRowIndex": 2,
                "startColumnIndex": 0,
                "endColumnIndex": 2,
            }
        ],
    )

    result = worksheet.writeback_cells(
        [{"cell": "B2", "field": "price", "expected": 5, "value": 6}]
    )

    assert result["conflicts"] == []
    assert result["updates"][0]["cell"] == "B2"
    assert result["updates"][0]["stored_cell"] == "A2"
    cells = request_cells(client)
    assert set(cells) == {"A2"}

    grid_request = client.metadata_calls[-1][1]
    assert grid_request["ranges"] == ["'Prices'!A2:A2"]
    assert "userEnteredValue" in grid_request["fields"]
    assert "note" in grid_request["fields"]


def test_formula_cell_compares_the_stored_formula() -> None:
    worksheet, _ = make_worksheet(grid_rows={(0, 1): cell_data(formula="=A1*2")})

    matching = worksheet.writeback_cells(
        [{"cell": "B1", "field": "value", "expected": "=A1*2", "value": "=A1*3"}],
        dry_run=True,
    )
    assert matching["conflicts"] == []
    assert matching["updates"][0]["current"] == "=A1*2"

    calculated = worksheet.writeback_cells(
        [{"cell": "B1", "field": "value", "expected": 39.98, "value": 1}],
        dry_run=True,
    )
    assert calculated["updates"] == []
    assert calculated["conflicts"][0]["reason"] == "original_value_mismatch"
    assert calculated["conflicts"][0]["current"] == "=A1*2"


def test_formula_value_is_written_as_a_formula() -> None:
    worksheet, client = make_worksheet(grid_rows={(0, 1): cell_data(formula="=A1*2")})

    worksheet.writeback_cells(
        [{"cell": "B1", "field": "value", "expected": "=A1*2", "value": "=A1*3"}]
    )

    value = client.batch_updates[0][1]["requests"][0]["updateCells"]["rows"][0][
        "values"
    ][0]
    assert value == {"userEnteredValue": {"formulaValue": "=A1*3"}}


def test_selected_field_only_keeps_other_attributes_out_of_the_request() -> None:
    worksheet, client = make_worksheet(
        grid_rows={(0, 0): cell_data(number=1, note="keep me")}
    )

    worksheet.writeback_cells(
        [{"cell": "A1", "field": "price", "expected": 1, "value": 2}]
    )

    request = client.batch_updates[0][1]["requests"][0]["updateCells"]
    assert request["fields"] == "userEnteredValue"
    assert request["rows"][0]["values"][0] == {"userEnteredValue": {"numberValue": 2}}
    assert "userEnteredFormat" not in json.dumps(client.batch_updates)
    assert "note" not in json.dumps(client.batch_updates)


def test_note_only_write_does_not_touch_the_value_or_format() -> None:
    worksheet, client = make_worksheet(
        grid_rows={(0, 0): cell_data(number=19.99, note="old")}
    )

    worksheet.writeback_cells(
        [{"cell": "A1", "field": "note", "expected": "old", "value": "new"}]
    )

    request = client.batch_updates[0][1]["requests"][0]["updateCells"]
    assert request["fields"] == "note"
    assert request["rows"][0]["values"][0] == {"note": "new"}
    assert "userEnteredValue" not in json.dumps(client.batch_updates)


def test_two_cells_in_one_merged_range_conflict_without_writing() -> None:
    worksheet, client = make_worksheet(
        grid_rows={(1, 0): cell_data(number=5)},
        merges=[
            {
                "startRowIndex": 1,
                "endRowIndex": 2,
                "startColumnIndex": 0,
                "endColumnIndex": 2,
            }
        ],
    )

    result = worksheet.writeback_cells(
        [
            {"cell": "A2", "field": "price", "expected": 5, "value": 6},
            {"cell": "B2", "field": "price", "expected": 5, "value": 7},
        ]
    )

    assert result["updates"] == []
    assert len(result["conflicts"]) == 2
    assert {entry["reason"] for entry in result["conflicts"]} == {"same_stored_cell"}
    assert client.batch_updates == []


def test_empty_and_numeric_values_are_compared_sensibly() -> None:
    worksheet, _ = make_worksheet(
        grid_rows={(0, 0): cell_data(number=0), (0, 1): cell_data()}
    )

    result = worksheet.writeback_cells(
        [
            {"cell": "A1", "field": "price", "expected": "0", "value": 1},
            {"cell": "B1", "field": "value", "expected": None, "value": 2},
        ],
        dry_run=True,
    )

    assert result["conflicts"] == []
    assert len(result["updates"]) == 2


def test_unchanged_duplicate_does_not_block_merged_write() -> None:
    worksheet, client = make_worksheet(
        grid_rows={(1, 0): cell_data(number=5)},
        merges=[
            {
                "startRowIndex": 1,
                "endRowIndex": 2,
                "startColumnIndex": 0,
                "endColumnIndex": 2,
            }
        ],
    )

    result = worksheet.writeback_cells(
        [
            {"cell": "A2", "field": "price", "expected": 5, "value": 5},
            {"cell": "B2", "field": "price", "expected": 5, "value": 6},
        ]
    )

    assert result["conflicts"] == []
    assert len(result["unchanged"]) == 1
    assert result["updates"][0]["cell"] == "B2"
    assert set(request_cells(client)) == {"A2"}


def test_exact_same_value_is_a_no_op_even_if_it_looks_equal() -> None:
    worksheet, client = make_worksheet(grid_rows={(0, 0): cell_data(text="20")})

    result = worksheet.writeback_cells(
        [{"cell": "A1", "field": "price", "expected": "20", "value": 20}]
    )

    # The number matches the original condition, but the stored
    # representation changes from text to a number, so it is written.
    assert [entry["cell"] for entry in result["updates"]] == ["A1"]
    assert client.batch_updates

    same = worksheet.writeback_cells(
        [{"cell": "A1", "field": "price", "expected": "20", "value": "20"}]
    )
    assert same["updates"] == []
    assert len(same["unchanged"]) == 1
    assert len(client.batch_updates) == 1


def test_text_values_are_not_compared_numerically() -> None:
    worksheet, _ = make_worksheet(
        grid_rows={(0, 0): cell_data(text="001"), (0, 1): cell_data(text="1")}
    )

    result = worksheet.writeback_cells(
        [
            {"cell": "A1", "field": "price", "expected": "1", "value": 2},
            {"cell": "B1", "field": "price", "expected": "1", "value": 2},
        ],
        dry_run=True,
    )

    assert result["conflicts"][0]["cell"] == "A1"
    assert result["conflicts"][0]["current"] == "001"
    assert [entry["cell"] for entry in result["updates"]] == ["B1"]


def test_invalid_updates_are_rejected_before_any_read() -> None:
    worksheet, client = make_worksheet()

    with pytest.raises(ValueError, match="unknown field"):
        worksheet.writeback_cells(
            [{"cell": "A1", "field": "format", "expected": None, "value": {}}]
        )
    assert client.metadata_calls == []

    with pytest.raises(ValueError, match="missing required key"):
        worksheet.writeback_cells([{"cell": "A1", "field": "value", "value": 1}])

    with pytest.raises(ValueError, match="another sheet"):
        worksheet.writeback_cells(
            [
                {
                    "cell": "Other!A1",
                    "field": "value",
                    "expected": None,
                    "value": 1,
                }
            ]
        )

    with pytest.raises(ValueError, match="single cell"):
        worksheet.writeback_cells(
            [{"cell": "A1:B2", "field": "value", "expected": None, "value": 1}]
        )

    with pytest.raises(ValueError, match="repeats"):
        worksheet.writeback_cells(
            [
                {"cell": "A1", "field": "value", "expected": None, "value": 1},
                {"cell": "A1", "field": "value", "expected": None, "value": 2},
            ]
        )

    with pytest.raises(TypeError, match="only strings"):
        worksheet.writeback_cells(
            [{"cell": "A1", "field": "note", "expected": "", "value": 1}]
        )


def test_sheet_selection_uses_the_current_worksheet_merges() -> None:
    worksheet, _ = make_worksheet(grid_rows={(0, 0): cell_data(number=5)})

    result = worksheet.writeback_cells(
        [{"cell": "A1", "field": "price", "expected": 5, "value": 6}],
        dry_run=True,
    )

    # Sheet id 1 in the metadata has A1:B1 merged; this worksheet does not, so
    # the update must remain on A1 and not be treated as the first sheet.
    assert result["updates"][0]["stored_cell"] == "A1"


def test_clearing_selected_field_only_sends_that_field() -> None:
    worksheet, client = make_worksheet(
        grid_rows={(0, 0): cell_data(number=19.99, note="old")}
    )

    result = worksheet.writeback_cells(
        [
            {"cell": "A1", "field": "price", "expected": "19.99", "value": None},
            {"cell": "A1", "field": "note", "expected": "old", "value": ""},
        ]
    )

    assert len(result["updates"]) == 2
    request = client.batch_updates[0][1]["requests"][0]["updateCells"]
    assert request["fields"] == "userEnteredValue,note"
    assert request["rows"][0]["values"][0] == {"userEnteredValue": {}, "note": ""}


def test_numeric_expected_matches_a_stored_number() -> None:
    worksheet, _ = make_worksheet(grid_rows={(0, 0): cell_data(number=19)})

    result = worksheet.writeback_cells(
        [{"cell": "A1", "field": "price", "expected": 19, "value": 20}],
        dry_run=True,
    )

    assert result["conflicts"] == []
    assert result["updates"][0]["current"] == 19


def test_note_is_never_matched_against_a_number() -> None:
    worksheet, _ = make_worksheet(grid_rows={(0, 0): cell_data(number=19)})

    result = worksheet.writeback_cells(
        [{"cell": "A1", "field": "note", "expected": 19, "value": "text"}],
        dry_run=True,
    )

    assert result["updates"] == []
    assert result["conflicts"][0]["reason"] == "original_value_mismatch"


def test_absolute_cell_reference_is_accepted() -> None:
    worksheet, client = make_worksheet(grid_rows={(0, 0): cell_data(number=1)})

    result = worksheet.writeback_cells(
        [{"cell": "$A$1", "field": "price", "expected": 1, "value": 2}]
    )

    assert result["conflicts"] == []
    assert len(client.batch_updates) == 1


def test_note_with_none_expected_matches_an_untouched_cell() -> None:
    worksheet, _ = make_worksheet(grid_rows={(0, 0): cell_data(number=1)})

    result = worksheet.writeback_cells(
        [{"cell": "A1", "field": "note", "expected": None, "value": "added"}],
        dry_run=True,
    )

    assert result["conflicts"] == []
    assert result["updates"][0]["current"] == ""


def test_read_ranges_group_continuous_cells_per_row() -> None:
    runs = Worksheet._writeback_read_ranges([(0, 0), (0, 1), (0, 3), (1, 2), (0, 1)])

    assert runs == [(0, 0, 0, 1), (0, 3, 0, 3), (1, 2, 1, 2)]


def test_large_batches_are_read_in_chunks() -> None:
    worksheet, client = make_worksheet(
        grid_rows={(row, 0): cell_data(number=1) for row in range(250)}
    )

    worksheet._get_writeback_cell_data({(row, 0) for row in range(250)})

    # 250 one-cell runs become three requests of at most 100 ranges each.
    assert len(client.metadata_calls) == 3
    assert [len(params["ranges"]) for _, params in client.metadata_calls] == [
        100,
        100,
        50,
    ]


def test_note_expected_numeric_string_is_not_treated_as_text() -> None:
    worksheet, _ = make_worksheet(grid_rows={(0, 0): cell_data(note="19")})

    result = worksheet.writeback_cells(
        [{"cell": "A1", "field": "note", "expected": 19, "value": "new"}],
        dry_run=True,
    )

    assert result["updates"] == []
    assert result["conflicts"][0]["reason"] == "original_value_mismatch"


def test_mixed_fields_on_one_cell_share_a_single_request() -> None:
    worksheet, client = make_worksheet(
        grid_rows={(0, 0): cell_data(number=19.99, note="old")}
    )

    worksheet.writeback_cells(
        [
            {"cell": "A1", "field": "price", "expected": "19.99", "value": 20},
            {"cell": "A1", "field": "note", "expected": "old", "value": "new"},
        ]
    )

    requests = client.batch_updates[0][1]["requests"]
    assert len(requests) == 1
    assert requests[0]["updateCells"]["fields"] == "userEnteredValue,note"


def test_empty_update_list_does_not_read_or_write() -> None:
    worksheet, client = make_worksheet()

    result = worksheet.writeback_cells([])

    assert result == {
        "dry_run": False,
        "updates": [],
        "conflicts": [],
        "unchanged": [],
        "requests": [],
    }
    assert client.metadata_calls == []
    assert client.batch_updates == []


def test_unchanged_values_are_reported_without_a_request() -> None:
    worksheet, client = make_worksheet(
        grid_rows={(0, 0): cell_data(number=19.99, note="old")}
    )

    result = worksheet.writeback_cells(
        [
            {"cell": "A1", "field": "price", "expected": "19.99", "value": 19.99},
            {"cell": "A1", "field": "note", "expected": "old", "value": "old"},
        ]
    )

    assert result["updates"] == []
    assert len(result["unchanged"]) == 2
    assert result["requests"] == []
    assert client.batch_updates == []
