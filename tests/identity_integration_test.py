import unittest
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

from gspread.identity import (
    ChangeKind,
    IdentityLedger,
    SheetEvent,
    SheetSnapshot,
)
from gspread.spreadsheet import Spreadsheet


class FakeClient:
    """Minimal stand-in for HTTPClient used by Spreadsheet."""

    def __init__(self, metadata: Dict[str, Any], values=None):
        self.metadata = metadata
        self.values = values or {}
        self.value_calls = []

    def fetch_sheet_metadata(
        self, id: str, params: Optional[Mapping[str, Any]] = None
    ) -> Mapping[str, Any]:
        return self.metadata

    def values_get(
        self, id: str, range: str, params: Optional[Mapping[str, Any]] = None
    ) -> Mapping[str, Any]:
        self.value_calls.append((range, dict(params or {})))
        title = range.strip("'").split("!")[0]
        return {
            "range": range,
            "majorDimension": "ROWS",
            "values": self.values.get(title, []),
        }


def spreadsheet_metadata(worksheets, spreadsheet_id: str = "SS"):
    return {
        "properties": {
            "id": spreadsheet_id,
            "title": "Book",
            "timeZone": "UTC",
            "locale": "en_US",
            "developerMetadata": [],
        },
        "sheets": [
            {
                "properties": {
                    "sheetId": sheet_id,
                    "title": title,
                    "index": index,
                    "gridProperties": {"rowCount": 10, "columnCount": 10},
                }
            }
            for index, (sheet_id, title) in enumerate(worksheets)
        ],
    }


class SpreadsheetIdentityTest(unittest.TestCase):
    def test_ledger_bootstraps_published_at_from_metadata(self):
        client = FakeClient(spreadsheet_metadata([(1, "Data"), (2, "Report")]))
        spreadsheet = Spreadsheet(client, {"id": "SS"})
        published = datetime(2025, 5, 1, tzinfo=timezone.utc)

        ledger = spreadsheet.identity_ledger(published_at=published)

        self.assertEqual(len(ledger.identities), 2)
        for identity in ledger.identities:
            self.assertEqual(identity.published_at, published)
            self.assertEqual(identity.sheet_id in (1, 2), True)

    def test_build_snapshot_passes_events_and_completeness(self):
        client = FakeClient(spreadsheet_metadata([(1, "Data")]))
        spreadsheet = Spreadsheet(client, {"id": "SS"})

        event = SheetEvent.delete(9)
        snapshot = spreadsheet.build_snapshot(
            events=[event], complete=False, observed_at="2026-01-01T00:00:00Z"
        )

        self.assertIsInstance(snapshot, SheetSnapshot)
        self.assertFalse(snapshot.complete)
        self.assertEqual(snapshot.events, (event,))
        self.assertEqual(
            snapshot.observed_at, datetime(2026, 1, 1, tzinfo=timezone.utc)
        )

    def test_collect_snapshot_reads_formulas(self):
        client = FakeClient(
            spreadsheet_metadata([(1, "Data"), (2, "Report")]),
            values={
                "Data": [["1", "2"]],
                "Report": [["=Data!A1", "plain"]],
            },
        )
        spreadsheet = Spreadsheet(client, {"id": "SS"})

        snapshot = spreadsheet.collect_identity_snapshot()

        by_id = {sheet.sheet_id: sheet for sheet in snapshot.sheets}
        self.assertEqual(by_id[2].formulas, {"A1": "=Data!A1"})
        self.assertEqual(by_id[1].formulas, {})
        for _, params in client.value_calls:
            self.assertEqual(params["valueRenderOption"], "FORMULA")

    def test_reconcile_identity_publishes_to_attached_ledger(self):
        client = FakeClient(spreadsheet_metadata([(1, "Data")]))
        spreadsheet = Spreadsheet(client, {"id": "SS"})
        ledger = spreadsheet.identity_ledger(
            published_at=datetime(2026, 1, 1, tzinfo=timezone.utc)
        )

        result = spreadsheet.reconcile_identity(
            at=datetime(2026, 1, 2, tzinfo=timezone.utc)
        )

        self.assertEqual(len(result.changes), 1)
        self.assertEqual(result.changes[0].kind, ChangeKind.unchanged)
        self.assertIs(spreadsheet.identity_ledger(), ledger)

    def test_attach_ledger_reuses_previous_identities(self):
        client = FakeClient(spreadsheet_metadata([(1, "Data")]))
        spreadsheet = Spreadsheet(client, {"id": "SS"})
        ledger = IdentityLedger()
        ledger.publish(
            SheetSnapshot(
                spreadsheet_id="SS",
                sheets=(),
                complete=True,
                observed_at=datetime(2025, 12, 1, tzinfo=timezone.utc),
            )
        )

        spreadsheet.attach_identity_ledger(ledger)

        self.assertIs(spreadsheet.identity_ledger(), ledger)
        self.assertEqual(ledger.identities, ())

    def test_identity_graph_delegates_to_ledger(self):
        client = FakeClient(spreadsheet_metadata([(1, "Data")]))
        spreadsheet = Spreadsheet(client, {"id": "SS"})
        spreadsheet.identity_ledger(
            published_at=datetime(2026, 1, 1, tzinfo=timezone.utc)
        )

        graph = spreadsheet.identity_graph(
            at=datetime(2026, 1, 1, tzinfo=timezone.utc)
        )

        self.assertEqual([source.title for source in graph.sources], ["Data"])
