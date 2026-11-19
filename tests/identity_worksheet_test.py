import unittest
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

from gspread.http_client import HTTPClient
from gspread.identity import ChangeKind
from gspread.spreadsheet import Spreadsheet
from gspread.worksheet import Worksheet


class RecordingClient(HTTPClient):
    def __init__(self, metadata: Dict[str, Any]):
        # Intentionally skip HTTPClient.__init__: these tests never send
        # requests over the network.
        self.metadata = metadata
        self.requests = []

    def fetch_sheet_metadata(
        self, id: str, params: Optional[Mapping[str, Any]] = None
    ) -> Mapping[str, Any]:
        return self.metadata

    def batch_update(self, id: str, body: Mapping[str, Any]) -> Mapping[str, Any]:
        self.requests.append(body)
        return {"replies": [{}]}

    def values_get(
        self, id: str, range: str, params: Optional[Mapping[str, Any]] = None
    ) -> Mapping[str, Any]:
        return {"range": range, "majorDimension": "ROWS", "values": []}


def metadata(worksheets):
    return {
        "properties": {
            "id": "SS",
            "title": "Book",
            "timeZone": "UTC",
            "locale": "en_US",
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


class WorksheetIdentityIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.client = RecordingClient(metadata([(1, "Data")]))
        self.spreadsheet = Spreadsheet(self.client, {"id": "SS"})
        self.sheet = self.spreadsheet.sheet1

    def test_uid_is_available_after_ledger_bootstrap(self):
        self.assertIsNone(self.sheet.uid)

        self.spreadsheet.identity_ledger(
            published_at=datetime(2026, 1, 1, tzinfo=timezone.utc)
        )

        self.assertEqual(self.sheet.uid, "worksheet:SS:1")
        self.assertEqual(self.sheet.identity_uid(), "worksheet:SS:1")

    def test_update_title_records_rename_without_a_new_fetch(self):
        self.spreadsheet.identity_ledger(
            published_at=datetime(2026, 1, 1, tzinfo=timezone.utc)
        )

        self.sheet.update_title("Archive")

        identity = self.spreadsheet.identity_ledger().identity("worksheet:SS:1")
        self.assertEqual(identity.title, "Archive")
        self.assertEqual(
            identity.titles, ("Data", "Archive")
        )
        self.assertEqual(len(identity.revisions), 2)
        # The UID follows the sheet ID, not the title.
        self.assertEqual(self.sheet.uid, "worksheet:SS:1")

    def test_update_title_without_ledger_still_updates_properties(self):
        self.sheet.update_title("Archive")

        self.assertEqual(self.sheet.title, "Archive")
        self.assertIsNone(self.sheet.uid)


class SpreadsheetDeletionTest(unittest.TestCase):
    def test_del_worksheet_records_delete_event(self):
        client = RecordingClient(metadata([(1, "Data"), (2, "Report")]))
        spreadsheet = Spreadsheet(client, {"id": "SS"})
        spreadsheet.identity_ledger(
            published_at=datetime(2026, 1, 1, tzinfo=timezone.utc)
        )

        spreadsheet.del_worksheet(spreadsheet.get_worksheet_by_id(1))

        identity = spreadsheet.identity_ledger().identity("worksheet:SS:1")
        self.assertFalse(identity.active)

    def test_del_worksheet_by_id_records_delete_event(self):
        client = RecordingClient(metadata([(1, "Data")]))
        spreadsheet = Spreadsheet(client, {"id": "SS"})
        spreadsheet.identity_ledger(
            published_at=datetime(2026, 1, 1, tzinfo=timezone.utc)
        )

        spreadsheet.del_worksheet_by_id(1)

        identity = spreadsheet.identity_ledger().identity("worksheet:SS:1")
        self.assertFalse(identity.active)


class WorksheetDuplicateTest(unittest.TestCase):
    class DuplicateClient(RecordingClient):
        def batch_update(self, id, body):
            self.requests.append(body)
            if "duplicateSheet" in body["requests"][0]:
                return {
                    "replies": [
                        {
                            "duplicateSheet": {
                                "properties": {
                                    "sheetId": 7,
                                    "title": "Data copy",
                                    "index": 1,
                                    "gridProperties": {
                                        "rowCount": 10,
                                        "columnCount": 10,
                                    },
                                }
                            }
                        }
                    ]
                }
            return {"replies": [{}]}

    def test_duplicate_records_parent_event(self):
        client = self.DuplicateClient(metadata([(1, "Data")]))
        spreadsheet = Spreadsheet(client, {"id": "SS"})
        spreadsheet.identity_ledger(
            published_at=datetime(2026, 1, 1, tzinfo=timezone.utc)
        )
        sheet = spreadsheet.sheet1

        copied = sheet.duplicate(new_sheet_name="Data copy")

        self.assertIsInstance(copied, Worksheet)
        self.assertEqual(copied.id, 7)
        ledger = spreadsheet.identity_ledger()
        pending = ledger.pending_duplicates()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].source_uid, "worksheet:SS:1")
        self.assertEqual(pending[0].sheet_id, 7)

    def test_duplicate_is_linked_on_next_observation(self):
        client = self.DuplicateClient(metadata([(1, "Data")]))
        spreadsheet = Spreadsheet(client, {"id": "SS"})
        spreadsheet.identity_ledger(
            published_at=datetime(2026, 1, 1, tzinfo=timezone.utc)
        )
        sheet = spreadsheet.sheet1
        sheet.duplicate(new_sheet_name="Data copy")

        # The next snapshot contains both worksheets.
        client.metadata = metadata([(1, "Data"), (7, "Data copy")])
        result = spreadsheet.reconcile_identity(
            at=datetime(2026, 1, 2, tzinfo=timezone.utc)
        )

        copied = next(change for change in result.changes if change.sheet_id == 7)
        self.assertEqual(copied.kind, ChangeKind.copied)
        self.assertEqual(copied.parent_uid, "worksheet:SS:1")
        self.assertEqual(spreadsheet.identity_ledger().pending_duplicates(), ())

    def test_stale_pending_duplicate_does_not_attach_to_a_later_sheet(self):
        client = self.DuplicateClient(metadata([(1, "Data")]))
        spreadsheet = Spreadsheet(client, {"id": "SS"})
        spreadsheet.identity_ledger(
            published_at=datetime(2026, 1, 1, tzinfo=timezone.utc)
        )
        spreadsheet.sheet1.duplicate(new_sheet_name="Data copy")

        # The copy never appears; an unrelated new sheet with another title
        # shows up instead.
        client.metadata = metadata([(1, "Data"), (9, "Unrelated")])
        result = spreadsheet.reconcile_identity(
            at=datetime(2026, 1, 2, tzinfo=timezone.utc)
        )

        unrelated = next(c for c in result.changes if c.sheet_id == 9)
        self.assertEqual(unrelated.kind, ChangeKind.new)
        self.assertIsNone(unrelated.parent_uid)
        # The pending copy is still waiting for its own sheet.
        self.assertEqual(
            len(spreadsheet.identity_ledger().pending_duplicates()), 1
        )
