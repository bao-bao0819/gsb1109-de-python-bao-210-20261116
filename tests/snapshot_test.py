import json
from unittest import TestCase
from unittest.mock import Mock

from gspread.exceptions import GSpreadException
from gspread.http_client import HTTPClient
from gspread.snapshot import (
    ReadSnapshotStatus,
    SnapshotReadBatch,
    SuspectRegion,
    reconcile_ranges,
    suspect_ranges_to_a1,
)
from gspread.worksheet import Worksheet


def make_worksheet(row_counts=None, metadata_error=None):
    client = HTTPClient(auth=None, session=Mock())
    properties = {
        "sheetId": 7,
        "title": "Sheet1",
        "index": 0,
        "gridProperties": {"rowCount": 100, "columnCount": 5},
    }

    if metadata_error is not None:
        client.fetch_sheet_metadata = Mock(side_effect=metadata_error)
    elif row_counts is not None:
        responses = [
            {
                "sheets": [
                    {
                        "properties": {
                            "sheetId": 7,
                            "title": "Sheet1",
                            "gridProperties": {
                                "rowCount": count,
                                "columnCount": 5,
                            },
                        }
                    }
                ]
            }
            for count in row_counts
        ]
        client.fetch_sheet_metadata = Mock(side_effect=responses)
    else:
        client.fetch_sheet_metadata = Mock(
            return_value={"sheets": [{"properties": properties}]}
        )

    def values_get(spreadsheet_id, range_name, params=None):
        return {"range": range_name, "majorDimension": "ROWS", "values": []}

    client.values_get = Mock(side_effect=values_get)
    worksheet = Worksheet(
        spreadsheet=None,
        properties=properties,
        spreadsheet_id="spreadsheet-id",
        client=client,
    )
    return worksheet, client


class SnapshotReadBatchTest(TestCase):
    def test_consistent_when_row_count_stable_and_ranges_match(self):
        # two reads, metadata is fetched before and after each read plus
        # once at finalization: five identical samples
        worksheet, client = make_worksheet(row_counts=[100] * 5)

        batch = SnapshotReadBatch(worksheet)
        batch.read("A1:C50")
        batch.read("A51:C100")
        report = batch.finalize()

        self.assertIs(report.status, ReadSnapshotStatus.CONSISTENT)
        self.assertTrue(report.is_consistent)
        self.assertEqual([], report.suspect_regions)
        self.assertEqual(2, len(report.reads))
        self.assertTrue(all(read.range_matches for read in report.reads))
        self.assertEqual(100, report.reads[0].row_count_before)
        self.assertEqual(100, report.reads[0].row_count_after)

        # the report is JSON-serialisable evidence that can be persisted
        encoded = json.dumps(report.to_dict())
        decoded = json.loads(encoded)
        self.assertEqual("CONSISTENT", decoded["status"])
        self.assertTrue(decoded["is_consistent"])
        self.assertEqual(
            "'Sheet1'!A1:C50", decoded["reads"][0]["requested_range"]
        )

        # a proven consistent batch does not raise
        report.raise_if_not_consistent()

        # values.get really was called with the absolute ranges
        called_ranges = [
            call.args[1] for call in client.values_get.call_args_list
        ]
        self.assertEqual(
            ["'Sheet1'!A1:C50", "'Sheet1'!A51:C100"], called_ranges
        )

    def test_inconsistent_when_row_count_changes_between_reads(self):
        # before/after read 1: 100/100, before/after read 2: 100/101,
        # finalize: 101
        worksheet, _ = make_worksheet(row_counts=[100, 100, 100, 101, 101])

        batch = SnapshotReadBatch(worksheet)
        batch.read("A1:C50")
        batch.read("A51:C100")
        report = batch.finalize()

        self.assertIs(report.status, ReadSnapshotStatus.INCONSISTENT)
        self.assertFalse(report.is_consistent)
        self.assertEqual(1, len(report.suspect_regions))
        region = report.suspect_regions[0]
        # the second read is the first unstable one, so rows 51..100
        # are the minimal region that must be re-read
        self.assertEqual(51, region.first_row)
        self.assertEqual(100, region.last_row)

        with self.assertRaises(GSpreadException):
            report.raise_if_not_consistent()

    def test_inconsistent_when_response_range_drifts(self):
        worksheet, client = make_worksheet(row_counts=[100] * 5)
        client.values_get = Mock(
            side_effect=[
                {"range": "'Sheet1'!A1:C50", "values": []},
                # a row inserted above: the same A1 range now answers one row
                # lower than requested
                {"range": "'Sheet1'!A2:C51", "values": []},
            ]
        )

        batch = SnapshotReadBatch(worksheet)
        batch.read("A1:C50")
        batch.read("A1:C50")
        report = batch.finalize()

        self.assertIs(report.status, ReadSnapshotStatus.INCONSISTENT)
        self.assertTrue(report.reads[0].range_matches)
        self.assertFalse(report.reads[1].range_matches)
        self.assertTrue(report.reads[1].issues)
        self.assertEqual(1, len(report.suspect_regions))
        self.assertEqual(1, report.suspect_regions[0].first_row)
        self.assertEqual(51, report.suspect_regions[0].last_row)

    def test_unknown_when_per_read_verification_disabled(self):
        worksheet, client = make_worksheet()

        batch = SnapshotReadBatch(
            worksheet, verify_before_each_read=False
        )
        batch.read("A1:C50")
        report = batch.finalize()

        self.assertIs(report.status, ReadSnapshotStatus.UNKNOWN)
        self.assertFalse(report.is_consistent)
        self.assertEqual([], report.suspect_regions)
        self.assertTrue(
            any("cannot be asserted" in note for note in report.notes)
        )
        with self.assertRaises(GSpreadException):
            report.raise_if_not_consistent()
        # no per-read metadata traffic, only the finalization sample
        self.assertEqual(1, client.fetch_sheet_metadata.call_count)

    def test_unknown_when_metadata_cannot_be_fetched(self):
        worksheet, _ = make_worksheet(metadata_error=RuntimeError("boom"))

        batch = SnapshotReadBatch(worksheet)
        response = batch.read("A1:C50")
        report = batch.finalize()

        # the value read itself must still succeed and be returned
        self.assertEqual("'Sheet1'!A1:C50", response["range"])
        self.assertIs(report.status, ReadSnapshotStatus.UNKNOWN)
        self.assertTrue(
            any("boom" in note for note in report.notes)
        )

    def test_empty_batch_is_unknown(self):
        worksheet, _ = make_worksheet()
        report = SnapshotReadBatch(worksheet).finalize()
        self.assertIs(report.status, ReadSnapshotStatus.UNKNOWN)

    def test_read_rejects_ranges_without_row_bounds(self):
        worksheet, _ = make_worksheet()
        batch = SnapshotReadBatch(worksheet)
        with self.assertRaises(GSpreadException):
            batch.read("A:B")

    def test_evidence_records_requested_and_answered_bounds(self):
        worksheet, client = make_worksheet(row_counts=[100] * 3)
        client.values_get = Mock(
            side_effect=[{"range": "'Sheet1'!A10:C60", "values": []}]
        )

        batch = SnapshotReadBatch(worksheet)
        batch.read("A10:C60")
        evidence = batch.reads[0]

        self.assertEqual(10, evidence.requested_start_row)
        self.assertEqual(60, evidence.requested_end_row)
        self.assertEqual(10, evidence.response_start_row)
        self.assertEqual(60, evidence.response_end_row)
        self.assertTrue(evidence.range_matches)


class ReconcileRangesTest(TestCase):
    def test_no_drift_returns_no_region(self):
        regions = reconcile_ranges(
            ["'Sheet1'!A1:C50", "'Sheet1'!A51:C100"],
            ["'Sheet1'!A1:C50", "'Sheet1'!A51:C100"],
        )
        self.assertEqual([], regions)

    def test_drift_is_circled_per_range_and_merged(self):
        regions = reconcile_ranges(
            ["'Sheet1'!A1:C50", "'Sheet1'!A51:C100"],
            ["'Sheet1'!A2:C51", "'Sheet1'!A52:C101"],
        )
        # adjacent intervals merge into one region covering both reads
        self.assertEqual(1, len(regions))
        self.assertEqual(1, regions[0].first_row)
        self.assertEqual(101, regions[0].last_row)

    def test_separate_drifts_keep_separate_regions(self):
        regions = reconcile_ranges(
            ["'Sheet1'!A1:C10", "'Sheet1'!A90:C100"],
            ["'Sheet1'!A2:C11", "'Sheet1'!A91:C101"],
        )
        self.assertEqual(2, len(regions))
        self.assertEqual((1, 11), (regions[0].first_row, regions[0].last_row))
        self.assertEqual(
            (90, 101), (regions[1].first_row, regions[1].last_row)
        )

    def test_length_mismatch_raises(self):
        with self.assertRaises(GSpreadException):
            reconcile_ranges(["'Sheet1'!A1:C10"], [])

    def test_unbounded_range_raises(self):
        with self.assertRaises(GSpreadException):
            reconcile_ranges(
                ["'Sheet1'!A:B"], ["'Sheet1'!A1:B10"]
            )

    def test_suspect_ranges_to_a1(self):
        regions = [SuspectRegion(12, 120, "changed"), SuspectRegion(200, 205, "x")]
        self.assertEqual(["12:120", "200:205"], suspect_ranges_to_a1(regions))
