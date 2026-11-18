import os
import tempfile
from typing import Any, List, Sequence
from unittest import TestCase

from requests.exceptions import Timeout

import gspread
from gspread.importing import (
    BatchImporter,
    ImportConflictError,
    ImportJournal,
    ImportStateError,
    ImportStatus,
    ImportTimeout,
)
from gspread.utils import a1_range_to_grid_range


class FakeWorksheet:
    """An in-memory worksheet with fault injection around append_rows."""

    spreadsheet_id = "spreadsheet-1"
    id = 42
    title = "Imports"

    def __init__(self) -> None:
        self.values: List[List[Any]] = []
        self.append_calls: List[List[List[Any]]] = []
        self.behavior: List[str] = []
        self.before_append = None

    def get_all_values(self, **kwargs: Any) -> List[List[Any]]:
        return [list(row) for row in self.values]

    def append_rows(self, rows: Sequence[Sequence[Any]], **kwargs: Any) -> None:
        self.append_calls.append([list(row) for row in rows])
        behavior = self.behavior.pop(0) if self.behavior else "write"

        if self.before_append is not None:
            callback, self.before_append = self.before_append, None
            callback(self)

        anchored_rows = [list(row) for row in rows]
        table_range = kwargs.get("table_range")
        if table_range:
            start_column = a1_range_to_grid_range(table_range).get(
                "startColumnIndex", 0
            )
            anchored_rows = [
                [""] * start_column + list(row) for row in anchored_rows
            ]

        if behavior in ("write", "write_then_timeout", "partial_then_timeout"):
            rows_to_write = [list(row) for row in anchored_rows]
            if behavior == "partial_then_timeout":
                rows_to_write = rows_to_write[: max(1, len(rows_to_write) // 2)]
            self.values.extend(rows_to_write)

        if behavior == "timeout":
            raise Timeout("append timed out")
        if behavior == "write_then_timeout":
            raise Timeout("response lost after write")
        if behavior == "partial_then_timeout":
            raise Timeout("response lost after partial write")


class BatchImporterTest(TestCase):
    def setUp(self) -> None:
        self.sheet = FakeWorksheet()

    def _importer(self, rows: Sequence[Sequence[Any]], **kwargs: Any) -> BatchImporter:
        kwargs.setdefault("batch_id", "batch-2026-11-19")
        kwargs.setdefault("key_columns", 0)
        return BatchImporter(self.sheet, rows, **kwargs)

    def test_successful_import_writes_receipts_and_confirms(self) -> None:
        rows = [["A-1", "alpha"], ["A-2", "beta"]]
        report = self._importer(rows).run()

        self.assertEqual(report.statuses, {0: "confirmed", 1: "confirmed"})
        self.assertTrue(report.complete)
        self.assertEqual(self.sheet.values[0][:2], ["A-1", "alpha"])
        self.assertTrue(self.sheet.values[0][2].startswith("__gimport__:batch-2026-11-19:"))
        self.assertEqual(len(self.sheet.append_calls), 1)

    def test_noop_after_success_does_not_append_again(self) -> None:
        rows = [["A-1", "alpha"]]
        importer = self._importer(rows)
        importer.run()
        calls_before = len(self.sheet.append_calls)

        report = importer.run()

        self.assertTrue(report.complete)
        self.assertEqual(len(self.sheet.append_calls), calls_before)

    def test_timeout_without_write_retries_only_unconfirmed(self) -> None:
        self.sheet.behavior = ["timeout", "write"]
        report = self._importer([["A-1", "alpha"]]).run(max_attempts=2)

        self.assertTrue(report.complete)
        self.assertTrue(report.timed_out)
        self.assertEqual(len(self.sheet.append_calls), 2)
        self.assertEqual([row[0] for row in self.sheet.values], ["A-1"])

    def test_timeout_after_write_confirms_without_duplicate(self) -> None:
        self.sheet.behavior = ["write_then_timeout"]
        report = self._importer([["A-1", "alpha"], ["A-2", "beta"]]).run()

        self.assertTrue(report.complete)
        self.assertTrue(report.timed_out)
        self.assertEqual(len(self.sheet.append_calls), 1)
        self.assertEqual([row[0] for row in self.sheet.values], ["A-1", "A-2"])

    def test_partial_timeout_retries_only_missing_rows(self) -> None:
        self.sheet.behavior = ["partial_then_timeout", "write"]
        rows = [["A-1", "alpha"], ["A-2", "beta"], ["A-3", "gamma"]]
        report = self._importer(rows).run(max_attempts=2)

        self.assertTrue(report.complete)
        self.assertEqual([row[0] for row in self.sheet.values], ["A-1", "A-2", "A-3"])
        self.assertEqual(len(self.sheet.append_calls), 2)
        retried = [row[0] for row in self.sheet.append_calls[1]]
        self.assertNotIn("A-1", retried)
        self.assertIn("A-3", retried)

    def test_same_business_row_inserted_by_someone_else_is_a_conflict(self) -> None:
        self.sheet.values.append(["A-1", "alpha", ""])

        report = self._importer([["A-1", "alpha"]]).run()

        self.assertEqual(report.statuses, {0: "conflict"})
        self.assertFalse(report.complete)
        self.assertEqual(self.sheet.append_calls, [])
        self.assertIn("without this batch's receipt", report.conflicts[0].reason)

    def test_same_business_row_inserted_during_append_is_never_our_receipt(self) -> None:
        def someone_else_writes(sheet: FakeWorksheet) -> None:
            sheet.values.append(["A-1", "alpha", ""])

        self.sheet.before_append = someone_else_writes
        report = self._importer([["A-1", "alpha"]]).run()

        # The race cannot be prevented, but it must never be reported as our
        # successful receipt.  Both rows are visible and the conflict is loud.
        self.assertEqual(report.statuses, {0: "conflict"})
        self.assertFalse(report.complete)
        self.assertEqual(report.conflicts[0].conflicting_row_number, 1)
        self.assertEqual(report.conflicts[0].row_number, 2)

    def test_other_batch_receipt_is_not_accepted_as_ours(self) -> None:
        other = BatchImporter(
            self.sheet,
            [["A-1", "alpha"]],
            batch_id="someone-elses-batch",
            key_columns=0,
        )
        other.run()
        receipt = self.sheet.values[0][2]

        report = self._importer([["A-1", "alpha"]]).run()

        self.assertEqual(report.statuses, {0: "conflict"})
        self.assertEqual(self.sheet.values[0][2], receipt)
        self.assertEqual(len(self.sheet.values), 1)

    def test_conflicting_row_present_even_after_our_receipt_is_conflict(self) -> None:
        importer = self._importer([["A-1", "alpha"]])
        importer.run()
        self.sheet.values.append(["A-1", "alpha", ""])

        report = importer.reconcile()

        self.assertEqual(report.statuses, {0: "conflict"})
        self.assertEqual(report.conflicts[0].row_number, 1)
        self.assertEqual(report.conflicts[0].conflicting_row_number, 2)

    def test_changed_row_with_our_receipt_is_conflict(self) -> None:
        importer = self._importer([["A-1", "alpha"]])
        importer.run()
        self.sheet.values[0][1] = "changed by a human"

        report = importer.reconcile()

        self.assertEqual(report.statuses, {0: "conflict"})
        self.assertIn("different values", report.conflicts[0].reason)

    def test_duplicate_receipt_is_conflict(self) -> None:
        importer = self._importer([["A-1", "alpha"]])
        importer.run()
        self.sheet.values.append(list(self.sheet.values[0]))

        report = importer.reconcile()

        self.assertEqual(report.statuses, {0: "conflict"})
        self.assertIn("more than once", report.conflicts[0].reason)

    def test_conflict_raises_when_requested(self) -> None:
        self.sheet.values.append(["A-1", "alpha", ""])

        with self.assertRaises(ImportConflictError) as context:
            self._importer([["A-1", "alpha"]]).run(raise_on_conflict=True)

        self.assertEqual(context.exception.report.statuses, {0: "conflict"})

    def test_timeout_raises_with_actionable_report_when_requested(self) -> None:
        self.sheet.behavior = ["timeout", "timeout", "timeout"]

        with self.assertRaises(ImportTimeout) as context:
            self._importer([["A-1", "alpha"]]).run(
                max_attempts=2, raise_on_timeout=True
            )

        report = context.exception.report
        self.assertTrue(report.timed_out)
        self.assertEqual(report.statuses, {0: "unconfirmed"})
        self.assertTrue(report.needs_retry)

    def test_journal_survives_restart_and_refuses_different_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "journal.json")
            rows = [["A-1", "alpha"], ["A-2", "beta"]]
            first = self._importer(rows, state_path=path)
            first.run()
            tokens = list(first.tokens)

            reopened = gspread.BatchImporter(
                self.sheet,
                [list(row) for row in self.sheet.values],
                batch_id="batch-2026-11-19",
                state_path=path,
            )
            self.assertEqual(reopened.tokens, tokens)
            self.assertEqual(
                reopened.reconcile().statuses, {0: "confirmed", 1: "confirmed"}
            )
            self.assertEqual(len(self.sheet.values), 2)
            reopened.run()
            self.assertEqual(len(self.sheet.values), 2)

            with self.assertRaises(ImportStateError):
                BatchImporter(
                    self.sheet,
                    [["A-1", "different"]],
                    batch_id="batch-2026-11-19",
                    key_columns=0,
                    state_path=path,
                )

    def test_manifest_is_written_before_append(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "journal.json")
            self.sheet.behavior = ["timeout", "timeout", "timeout"]
            importer = self._importer([["A-1", "alpha"]], state_path=path)

            with self.assertRaises(ImportTimeout):
                importer.run(max_attempts=1, raise_on_timeout=True)

            journal = ImportJournal(path)
            saved = journal.get("batch-2026-11-19")
            self.assertIsNotNone(saved)
            self.assertEqual(saved["rows"], [["A-1", "alpha"]])
            self.assertTrue(saved["tokens"][0].startswith("__gimport__:"))

    def test_dict_rows_use_headers_and_named_key_columns(self) -> None:
        rows = [
            {"order": "A-1", "amount": 10},
            {"order": "A-2", "amount": 20},
        ]
        importer = BatchImporter(
            self.sheet,
            rows,
            batch_id="dict-batch",
            headers=["order", "amount"],
            key_columns="order",
            receipt_column="receipt",
        )
        report = importer.run()

        self.assertTrue(report.complete)
        self.assertEqual(self.sheet.values[0][:2], ["A-1", 10])
        self.assertEqual(self.sheet.values[1][:2], ["A-2", 20])

    def test_business_key_must_come_from_this_batch(self) -> None:
        importer = self._importer([["A-1", "alpha"], ["A-1", "alpha"]])
        importer.run()

        report = importer.reconcile()

        self.assertEqual(
            [row.status for row in report.rows],
            [ImportStatus.confirmed, ImportStatus.confirmed],
        )

    def test_unowned_rows_are_ignored_by_reconciliation(self) -> None:
        self.sheet.values.append(["Z-9", "not ours", ""])
        report = self._importer([["A-1", "alpha"]]).run()

        self.assertTrue(report.complete)
        self.assertEqual(len(self.sheet.values), 2)

    def test_numeric_and_text_keys_match_across_api_rendering(self) -> None:
        self.sheet.values.append([101, "existing", ""])
        report = self._importer([["101", "existing"]]).run()

        self.assertEqual(report.statuses, {0: "conflict"})
        self.assertEqual(self.sheet.append_calls, [])

    def test_table_range_column_offset_is_honored(self) -> None:
        self.sheet.values.append(["", "", ""])
        importer = self._importer(
            [["A-1", "alpha"]], table_range="B1", receipt_column=4
        )
        importer.run()

        self.assertEqual(self.sheet.values[1][0:3], ["", "A-1", "alpha"])
        self.assertTrue(self.sheet.values[1][3].startswith("__gimport__:"))
        report = importer.reconcile()
        self.assertTrue(report.complete)
        self.assertEqual(report.receipt_column, 4)
