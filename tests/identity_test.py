import unittest
from datetime import datetime, timedelta, timezone

import gspread
from gspread.exceptions import IdentityError, IdentityTimelineError
from gspread.identity import (
    ChangeKind,
    DeveloperMetadata,
    Evidence,
    IdentityLedger,
    ReferenceSource,
    ReferenceStatus,
    SheetEvent,
    SheetObservation,
    SheetSnapshot,
)
from gspread.references import (
    extract_formula_references,
    extract_sheet_titles,
)


T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def at(hours: int) -> datetime:
    return T0 + timedelta(hours=hours)


def metadata(key: str, value, sheet_id: int = 1) -> DeveloperMetadata:
    return DeveloperMetadata(
        key=key, value=value, owner_sheet_id=sheet_id
    )


def sheet(
    sheet_id: int,
    title: str,
    formulas=None,
    developer_metadata=None,
    index: int = 0,
) -> SheetObservation:
    return SheetObservation(
        sheet_id=sheet_id,
        title=title,
        index=index,
        formulas=formulas,
        developer_metadata=developer_metadata,
    )


def snapshot(sheets, events=(), complete=True, observed_at=None) -> SheetSnapshot:
    return SheetSnapshot(
        spreadsheet_id="SS",
        sheets=tuple(sheets),
        events=tuple(events),
        complete=complete,
        observed_at=observed_at,
    )


class FormulaReferenceTest(unittest.TestCase):
    def test_simple_reference(self):
        refs = extract_formula_references("=SUM(Data!A1:B2)")
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].sheet_title, "Data")
        self.assertEqual(refs[0].a1_range, "A1:B2")
        self.assertEqual(refs[0].raw, "Data!A1:B2")

    def test_quoted_title_with_escaped_quote(self):
        refs = extract_formula_references("='Bob''s data'!A1")
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].sheet_title, "Bob's data")
        self.assertEqual(refs[0].a1_range, "A1")

    def test_ignores_string_literals(self):
        refs = extract_formula_references('=A1 & "Data!B2"')
        self.assertEqual(refs, [])

    def test_ignores_indirect(self):
        refs = extract_formula_references('=INDIRECT("Data!A1")')
        self.assertEqual(refs, [])

    def test_ignores_same_sheet_reference(self):
        refs = extract_formula_references("=SUM(A1:B2)")
        self.assertEqual(refs, [])

    def test_multiple_and_distinct_titles(self):
        formula = "=Data!A1 + 'My Sheet'!B2 + Data!C3"
        refs = extract_formula_references(formula)
        self.assertEqual(
            [ref.sheet_title for ref in refs],
            ["Data", "My Sheet", "Data"],
        )
        self.assertEqual(extract_sheet_titles(formula), ["Data", "My Sheet"])

    def test_row_and_column_ranges(self):
        refs = extract_formula_references("=SUM(Data!A:A) + SUM(Data!1:1)")
        self.assertEqual([ref.a1_range for ref in refs], ["A:A", "1:1"])


class IdentityLedgerTest(unittest.TestCase):
    def setUp(self):
        self.ledger = IdentityLedger()

    def test_identity_anchors_on_sheet_id_not_title(self):
        result = self.ledger.observe(
            snapshot([sheet(1, "Data"), sheet(2, "Report")], observed_at=at(0))
        )
        uids = {change.title: change.uid for change in result.changes}
        self.assertEqual(uids["Data"], "worksheet:SS:1")
        self.assertEqual(uids["Report"], "worksheet:SS:2")

        renamed = self.ledger.observe(
            snapshot([sheet(1, "Archive"), sheet(2, "Report")], observed_at=at(1))
        )
        data = next(c for c in renamed.changes if c.sheet_id == 1)
        self.assertEqual(data.kind, ChangeKind.renamed)
        self.assertEqual(data.uid, "worksheet:SS:1")
        self.assertEqual(data.previous_title, "Data")
        self.assertEqual(
            data.evidence, (Evidence.same_sheet_id, Evidence.title_changed)
        )

        identity = self.ledger.identity("worksheet:SS:1")
        self.assertEqual(identity.published_at, at(0))
        self.assertEqual(identity.titles, ("Data", "Archive"))

    def test_new_sheet_reusing_old_title_does_not_steal_identity(self):
        self.ledger.observe(
            snapshot([sheet(1, "Data")], observed_at=at(0))
        )
        self.ledger.observe(
            snapshot([sheet(1, "Archive")], observed_at=at(1))
        )
        result = self.ledger.observe(
            snapshot([sheet(1, "Archive"), sheet(3, "Data")], observed_at=at(2))
        )
        new_sheet = next(c for c in result.changes if c.sheet_id == 3)
        self.assertEqual(new_sheet.kind, ChangeKind.new)
        self.assertEqual(new_sheet.uid, "worksheet:SS:3")
        self.assertEqual(new_sheet.candidates[0].uid, "worksheet:SS:1")
        self.assertIn("historical_title", new_sheet.candidates[0].reasons)

        # The original identity still belongs to the renamed worksheet.
        self.assertEqual(
            self.ledger.identity("worksheet:SS:1").title, "Archive"
        )
        self.assertEqual(
            self.ledger.identity("worksheet:SS:3").title, "Data"
        )

    def test_copied_sheet_keeps_same_business_metadata_without_claiming_identity(
        self,
    ):
        source_metadata = (metadata("source.id", "acme"),)
        self.ledger.observe(
            snapshot(
                [sheet(1, "Data", developer_metadata=source_metadata)],
                observed_at=at(0),
            )
        )

        # A duplicate carries the same business metadata and the same title.
        result = self.ledger.observe(
            snapshot(
                [
                    sheet(1, "Data", developer_metadata=source_metadata),
                    sheet(3, "Data", developer_metadata=source_metadata),
                ],
                events=[SheetEvent.duplicate(3, 1)],
                observed_at=at(1),
            )
        )
        copy = next(c for c in result.changes if c.sheet_id == 3)
        self.assertEqual(copy.kind, ChangeKind.copied)
        self.assertEqual(copy.uid, "worksheet:SS:3")
        self.assertEqual(copy.parent_uid, "worksheet:SS:1")
        self.assertNotEqual(copy.uid, "worksheet:SS:1")

        # Both remain distinct identities, each with its own publication time.
        self.assertEqual(
            self.ledger.identity("worksheet:SS:1").published_at, at(0)
        )
        self.assertEqual(
            self.ledger.identity("worksheet:SS:3").published_at, at(1)
        )

    def test_copy_with_same_metadata_is_only_a_candidate_without_event(self):
        source_metadata = (metadata("source.id", "acme"),)
        self.ledger.observe(
            snapshot(
                [sheet(1, "Data", developer_metadata=source_metadata)],
                observed_at=at(0),
            )
        )
        result = self.ledger.observe(
            snapshot(
                [
                    sheet(1, "Archive", developer_metadata=source_metadata),
                    sheet(3, "Archive", developer_metadata=source_metadata),
                ],
                observed_at=at(1),
            )
        )
        copy = next(c for c in result.changes if c.sheet_id == 3)
        self.assertEqual(copy.kind, ChangeKind.new)
        self.assertIsNone(copy.parent_uid)
        self.assertTrue(copy.candidates)

        # The original keeps the identity even though metadata matches.
        self.assertEqual(
            self.ledger.identity("worksheet:SS:1").title, "Archive"
        )

    def test_explicit_delete_then_recreate_records_predecessor(self):
        self.ledger.observe(
            snapshot([sheet(1, "Data")], observed_at=at(0))
        )
        self.ledger.observe(
            snapshot([], events=[SheetEvent.delete(1)], observed_at=at(1))
        )
        result = self.ledger.observe(
            snapshot([sheet(2, "Data")], observed_at=at(2))
        )
        recreated = result.changes[0]
        self.assertEqual(recreated.kind, ChangeKind.recreated)
        self.assertEqual(recreated.predecessor_uid, "worksheet:SS:1")
        self.assertNotEqual(recreated.uid, "worksheet:SS:1")
        self.assertIn(Evidence.title_reused_after_delete, recreated.evidence)

    def test_recreate_does_not_inherit_when_title_is_still_occupied(self):
        self.ledger.observe(
            snapshot([sheet(1, "Data")], observed_at=at(0))
        )
        self.ledger.observe(
            snapshot([], events=[SheetEvent.delete(1)], observed_at=at(1))
        )
        self.ledger.observe(
            snapshot([sheet(5, "Data")], observed_at=at(2))
        )
        result = self.ledger.observe(
            snapshot([sheet(5, "Data"), sheet(6, "Data")], observed_at=at(3))
        )
        newest = next(c for c in result.changes if c.sheet_id == 6)
        self.assertEqual(newest.kind, ChangeKind.new)
        self.assertIsNone(newest.predecessor_uid)

    def test_absent_from_complete_snapshot_is_deleted(self):
        self.ledger.observe(
            snapshot([sheet(1, "Data"), sheet(2, "Report")], observed_at=at(0))
        )
        result = self.ledger.observe(
            snapshot([sheet(2, "Report")], observed_at=at(1))
        )
        deleted = next(c for c in result.changes if c.kind == ChangeKind.deleted)
        self.assertEqual(deleted.uid, "worksheet:SS:1")
        self.assertEqual(
            self.ledger.identity("worksheet:SS:1").retired_reason,
            Evidence.absent_from_complete_snapshot,
        )

    def test_incomplete_snapshot_does_not_delete_absent_sheets(self):
        self.ledger.observe(
            snapshot([sheet(1, "Data"), sheet(2, "Report")], observed_at=at(0))
        )
        result = self.ledger.observe(
            snapshot([sheet(2, "Report")], complete=False, observed_at=at(1))
        )
        self.assertFalse(
            [change for change in result.changes if change.kind == ChangeKind.deleted]
        )
        self.assertTrue(self.ledger.identity("worksheet:SS:1").active)

    def test_history_does_not_use_current_title(self):
        self.ledger.observe(
            snapshot(
                [
                    sheet(
                        1,
                        "Data",
                        formulas={"A1": "=1"},
                    ),
                    sheet(2, "Report", formulas={"B1": "=Data!A1"}),
                ],
                observed_at=at(0),
            )
        )
        self.ledger.observe(
            snapshot(
                [
                    sheet(1, "Archive", formulas={"A1": "=1"}),
                    sheet(2, "Report", formulas={"B1": "=Data!A1"}),
                ],
                observed_at=at(10),
            )
        )

        graph_at_zero = self.ledger.reference_graph(at(0))
        self.assertEqual(
            {source.uid: source.title for source in graph_at_zero.sources},
            {"worksheet:SS:1": "Data", "worksheet:SS:2": "Report"},
        )
        edge = graph_at_zero.edges[0]
        self.assertEqual(edge.target_title, "Data")
        self.assertEqual(edge.target_uid, "worksheet:SS:1")

        graph_now = self.ledger.reference_graph(at(10))
        self.assertEqual(
            {source.uid: source.title for source in graph_now.sources},
            {"worksheet:SS:1": "Archive", "worksheet:SS:2": "Report"},
        )
        # The newly observed formula still says Data. The title belonged to a
        # renamed object, so it cannot be resolved without explicit evidence,
        # but it is not silently dropped either.
        self.assertEqual(graph_now.edges[0].status, ReferenceStatus.ambiguous)
        self.assertEqual(graph_now.edges[0].target_title, "Data")
        self.assertIsNone(graph_now.edges[0].target_uid)
        self.assertEqual(
            graph_now.edges[0].candidates, ("worksheet:SS:1",)
        )

        # The historical edge is immutable and still points at the object.
        historical_edges = self.ledger.references_for("worksheet:SS:2")
        self.assertEqual(len(historical_edges), 2)
        self.assertEqual(
            historical_edges[0].target_uid, "worksheet:SS:1"
        )
        self.assertEqual(historical_edges[0].target_title, "Data")

    def test_explicit_uid_reference_survives_rename(self):
        reference_metadata = (
            metadata(
                "gspread.reference",
                {"targetUid": "worksheet:SS:1", "targetTitle": "Data"},
                sheet_id=2,
            ),
        )
        self.ledger.observe(
            snapshot(
                [
                    sheet(1, "Data"),
                    sheet(2, "Report", developer_metadata=reference_metadata),
                ],
                observed_at=at(0),
            )
        )
        result = self.ledger.observe(
            snapshot(
                [
                    sheet(1, "Archive"),
                    sheet(2, "Report", developer_metadata=reference_metadata),
                ],
                observed_at=at(1),
            )
        )
        edge = result.references[0]
        self.assertEqual(edge.source_kind, ReferenceSource.developer_metadata)
        self.assertEqual(edge.target_uid, "worksheet:SS:1")
        # The title recorded on the edge is the one at observation time.
        self.assertEqual(edge.target_title, "Data")

    def test_explicit_sheet_id_reference_survives_rename(self):
        reference_metadata = (
            metadata(
                "gspread.reference",
                {"targetSheetId": 1, "targetTitle": "Data"},
                sheet_id=2,
            ),
        )
        self.ledger.observe(
            snapshot(
                [
                    sheet(1, "Data"),
                    sheet(2, "Report", developer_metadata=reference_metadata),
                ],
                observed_at=at(0),
            )
        )
        result = self.ledger.observe(
            snapshot(
                [
                    sheet(1, "Archive"),
                    sheet(2, "Report", developer_metadata=reference_metadata),
                ],
                observed_at=at(1),
            )
        )
        self.assertEqual(result.references[0].target_uid, "worksheet:SS:1")
        self.assertEqual(result.references[0].status, ReferenceStatus.resolved)

    def test_title_reference_becomes_ambiguous_after_title_handover(self):
        self.ledger.observe(
            snapshot([sheet(1, "Data")], observed_at=at(0))
        )
        self.ledger.observe(
            snapshot([sheet(1, "Archive")], observed_at=at(1))
        )
        self.ledger.observe(
            snapshot([sheet(1, "Archive"), sheet(3, "Data")], observed_at=at(2))
        )
        result = self.ledger.observe(
            snapshot(
                [
                    sheet(1, "Archive"),
                    sheet(2, "Report", formulas={"B1": "=Data!A1"}),
                    sheet(3, "Data"),
                ],
                observed_at=at(3),
            )
        )
        report_edges = [
            edge for edge in result.references if edge.source_uid == "worksheet:SS:2"
        ]
        self.assertEqual(len(report_edges), 1)
        self.assertEqual(report_edges[0].status, ReferenceStatus.ambiguous)
        self.assertEqual(
            set(report_edges[0].candidates),
            {"worksheet:SS:1", "worksheet:SS:3"},
        )

    def test_business_metadata_is_never_a_reference(self):
        business_metadata = (
            metadata("source.id", {"targetUid": "worksheet:SS:1"}, sheet_id=2),
        )
        result = self.ledger.observe(
            snapshot(
                [
                    sheet(1, "Data"),
                    sheet(2, "Report", developer_metadata=business_metadata),
                ],
                observed_at=at(0),
            )
        )
        self.assertEqual(result.references, ())

    def test_time_cannot_move_backwards(self):
        self.ledger.observe(
            snapshot([sheet(1, "Data")], observed_at=at(2))
        )
        with self.assertRaises(IdentityTimelineError):
            self.ledger.observe(
                snapshot([sheet(1, "Data")], observed_at=at(1))
            )

    def test_unknown_identity_raises(self):
        with self.assertRaises(IdentityError):
            self.ledger.identity("worksheet:SS:404")

    def test_partial_observation_reuses_previous_values(self):
        self.ledger.observe(
            snapshot(
                [
                    sheet(
                        1,
                        "Data",
                        formulas={"A1": "=1"},
                        developer_metadata=(metadata("k", "v"),),
                    ),
                    sheet(2, "Report", formulas={"B1": "=Data!A1"}),
                ],
                observed_at=at(0),
            )
        )
        self.ledger.observe(
            snapshot([sheet(1, "Data"), sheet(2, "Report")], observed_at=at(1))
        )
        revision = self.ledger.identity("worksheet:SS:2").current_revision
        self.assertEqual(revision.formula_map, {"B1": "=Data!A1"})
        data_revision = self.ledger.identity("worksheet:SS:1").current_revision
        self.assertEqual(data_revision.developer_metadata[0].key, "k")

    def test_serialization_roundtrip(self):
        self.ledger.observe(
            snapshot(
                [
                    sheet(1, "Data"),
                    sheet(2, "Report", formulas={"B1": "=Data!A1"}),
                ],
                observed_at=at(0),
            )
        )
        self.ledger.observe(
            snapshot(
                [
                    sheet(1, "Archive"),
                    sheet(2, "Report", formulas={"B1": "=Data!A1"}),
                ],
                observed_at=at(1),
            )
        )

        restored = IdentityLedger.loads(self.ledger.dumps())
        self.assertEqual(
            [(i.uid, i.title) for i in restored.identities],
            [(i.uid, i.title) for i in self.ledger.identities],
        )
        self.assertEqual(
            [edge.to_dict() for edge in restored.edges],
            [edge.to_dict() for edge in self.ledger.edges],
        )
        self.assertEqual(
            restored.identity("worksheet:SS:1").published_at, at(0)
        )


class LedgerEventTest(unittest.TestCase):
    def test_record_event_before_observing_duplicate(self):
        ledger = IdentityLedger()
        ledger.observe(
            snapshot([sheet(1, "Data")], observed_at=at(0))
        )
        ledger.record_event(
            SheetEvent(
                kind="duplicate",
                sheet_id=-1,
                source_uid="worksheet:SS:1",
                title="Data",
            ),
            "SS",
        )
        self.assertEqual(len(ledger.pending_duplicates()), 1)

        result = ledger.observe(
            snapshot([sheet(1, "Data"), sheet(2, "Data")], observed_at=at(1))
        )
        copy = next(c for c in result.changes if c.sheet_id == 2)
        self.assertEqual(copy.kind, ChangeKind.copied)
        self.assertEqual(copy.parent_uid, "worksheet:SS:1")
        self.assertEqual(ledger.pending_duplicates(), ())


class PublicApiTest(unittest.TestCase):
    def test_identity_symbols_are_exported(self):
        self.assertIs(gspread.IdentityLedger, IdentityLedger)
        self.assertEqual(gspread.ChangeKind.renamed, "renamed")
        self.assertIs(gspread.extract_formula_references, extract_formula_references)


class GenerationTest(unittest.TestCase):
    def setUp(self):
        self.ledger = IdentityLedger()
        self.ledger.observe(
            snapshot([sheet(1, "A")], observed_at=at(0))
        )
        self.ledger.observe(
            snapshot([], events=[SheetEvent.delete(1)], observed_at=at(1))
        )
        self.ledger.observe(
            snapshot([sheet(1, "B")], observed_at=at(2))
        )

    def test_reused_sheet_id_gets_a_distinct_uid(self):
        self.assertEqual(
            [(i.uid, i.title) for i in self.ledger.identities],
            [
                ("worksheet:SS:1", "A"),
                ("worksheet:SS:1#2", "B"),
            ],
        )

    def test_explicit_sheet_id_resolves_to_generation_live_at_the_time(self):
        old, old_status, _, _ = self.ledger._resolve_reference(
            "SS", None, 1, None, at(0)
        )
        new, new_status, _, _ = self.ledger._resolve_reference(
            "SS", None, 1, None, at(2)
        )
        self.assertEqual(old.uid, "worksheet:SS:1")
        self.assertEqual(new.uid, "worksheet:SS:1#2")
        self.assertEqual(old_status, ReferenceStatus.resolved)
        self.assertEqual(new_status, ReferenceStatus.resolved)

    def test_historical_titles_follow_the_generation(self):
        self.assertEqual(
            [source.title for source in self.ledger.titles_at("SS", "A", at(0))],
            ["A"],
        )
        self.assertEqual(self.ledger.titles_at("SS", "A", at(2)), ())
        self.assertEqual(
            [source.title for source in self.ledger.titles_at("SS", "B", at(2))],
            ["B"],
        )


class ClockTest(unittest.TestCase):
    def test_locally_recorded_event_does_not_break_the_next_snapshot(self):
        ledger = IdentityLedger()
        ledger.observe(snapshot([sheet(1, "A")], observed_at=at(0)))
        # No explicit time: the event is stamped at the ledger's latest
        # observation, not at wall-clock "now".
        ledger.record_event(SheetEvent.delete(1), "SS")

        result = ledger.observe(snapshot([], observed_at=at(1)))

        self.assertEqual(ledger.last_observed, at(1))
        self.assertFalse(
            [change for change in result.changes if change.kind == ChangeKind.deleted]
        )

    def test_explicitly_future_event_blocks_earlier_snapshots(self):
        ledger = IdentityLedger()
        ledger.observe(snapshot([sheet(1, "A")], observed_at=at(0)))
        ledger.record_event(SheetEvent.delete(1, at=at(5)), "SS")

        with self.assertRaises(IdentityTimelineError):
            ledger.observe(snapshot([], observed_at=at(1)))

    def test_delete_event_is_reported(self):
        ledger = IdentityLedger()
        ledger.observe(snapshot([sheet(1, "A")], observed_at=at(0)))

        result = ledger.observe(
            snapshot([], events=[SheetEvent.delete(1)], observed_at=at(1))
        )

        deleted = [c for c in result.changes if c.kind == ChangeKind.deleted]
        self.assertEqual(len(deleted), 1)
        self.assertEqual(deleted[0].uid, "worksheet:SS:1")
        self.assertEqual(deleted[0].title, "A")
