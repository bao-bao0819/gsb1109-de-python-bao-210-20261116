"""Tests for the protected-range write access rehearsal.

These tests use mocked HTTP clients so they run offline without VCR
cassettes. They focus on the decision rules:

* every protected range is evaluated independently and an exemption on one
  range can never cancel another range's restriction;
* verdicts are split into allowed / denied / insufficient evidence;
* stale cached permission metadata is re-fetched rather than trusted.
"""

from unittest import TestCase
from unittest.mock import Mock

from gspread.exceptions import StaleProtectionMetadataError
from gspread.http_client import HTTPClient
from gspread.worksheet import (
    ProtectionVerdict,
    Worksheet,
    WriteAccessRehearsal,
)


class FakeAuth:
    """Minimal stand-in for google-auth credentials."""

    def __init__(
        self,
        token=None,
        service_account_email=None,
        refresh_token=None,
    ):
        self.token = token
        if service_account_email is not None:
            self.service_account_email = service_account_email
        if refresh_token is not None:
            self._refresh_token = refresh_token


def _grid_range(pr_id, start_row, end_row, start_col, end_col, **extra):
    protected_range = {
        "protectedRangeId": pr_id,
        "range": {
            "sheetId": 0,
            "startRowIndex": start_row,
            "endRowIndex": end_row,
            "startColumnIndex": start_col,
            "endColumnIndex": end_col,
        },
    }
    protected_range.update(extra)
    return protected_range


def _metadata(protected_ranges):
    return {
        "sheets": [
            {
                "properties": {"sheetId": 0},
                "protectedRanges": protected_ranges,
            }
        ]
    }


class WriteAccessRehearsalTest(TestCase):
    def _make_worksheet(self, protected_ranges, auth=None):
        client = HTTPClient(auth=None, session=Mock())
        client.fetch_sheet_metadata = Mock(
            return_value=_metadata(protected_ranges)
        )
        if auth is not None:
            client.auth = auth
        properties = {
            "sheetId": 0,
            "title": "Sheet1",
            "index": 0,
            "gridProperties": {"rowCount": 20, "columnCount": 10},
        }
        worksheet = Worksheet(
            spreadsheet=None,
            properties=properties,
            spreadsheet_id="abc123",
            client=client,
        )
        return worksheet, client

    def _verdict(self, rehearsal, cell):
        return next(
            verdict for verdict in rehearsal.verdicts if verdict.cell == cell
        )

    def test_no_protected_ranges_everything_allowed(self):
        worksheet, _ = self._make_worksheet([])

        rehearsal = worksheet.rehearse_write_access("A1:B2")

        self.assertEqual(
            rehearsal.allowed_cells, ["A1", "B1", "A2", "B2"]
        )
        self.assertEqual(rehearsal.denied_cells, [])
        self.assertEqual(rehearsal.insufficient_evidence_cells, [])
        self.assertFalse(rehearsal.stale)

    def test_warning_only_range_never_blocks(self):
        protected_range = _grid_range(
            1, 0, 2, 0, 2, warningOnly=True
        )
        worksheet, _ = self._make_worksheet([protected_range])

        rehearsal = worksheet.rehearse_write_access("A1:B2")

        self.assertEqual(len(rehearsal.allowed_cells), 4)

    def test_listed_user_editor_is_allowed(self):
        protected_range = _grid_range(
            1,
            0,
            1,
            0,
            1,
            warningOnly=False,
            editors={"users": ["alice@example.com"], "groups": []},
        )
        worksheet, _ = self._make_worksheet(
            [protected_range], auth=FakeAuth(token="opaque-token")
        )
        # identity cannot be decoded from an opaque token: inject the email
        # the way a service account would expose it
        worksheet.client.auth.service_account_email = "alice@example.com"

        rehearsal = worksheet.rehearse_write_access("A1:A1")

        self.assertEqual(rehearsal.allowed_cells, ["A1"])

    def test_non_editor_is_hard_denied(self):
        protected_range = _grid_range(
            1,
            0,
            1,
            0,
            1,
            warningOnly=False,
            editors={"users": ["alice@example.com"], "groups": []},
        )
        worksheet, _ = self._make_worksheet(
            [protected_range],
            auth=FakeAuth(
                service_account_email="bot@proj.iam.gserviceaccount.com"
            ),
        )

        rehearsal = worksheet.rehearse_write_access("A1:A1")
        verdict = self._verdict(rehearsal, "A1")

        self.assertEqual(verdict.status, ProtectionVerdict.DENIED)
        self.assertEqual(verdict.blocking_protected_range_ids, ["1"])
        self.assertEqual(verdict.evidence_protected_range_ids, [])

    def test_exemption_of_one_range_cannot_offset_another_range(self):
        # PR1 covers A1:B2 and only alice may edit it.
        pr1 = _grid_range(
            1,
            0,
            2,
            0,
            2,
            warningOnly=False,
            editors={"users": ["alice@example.com"], "groups": []},
        )
        # PR2 overlaps on B2 and explicitly exempts bob.
        pr2 = _grid_range(
            2,
            1,
            2,
            1,
            2,
            warningOnly=False,
            editors={"users": ["bob@example.com"], "groups": []},
        )
        worksheet, _ = self._make_worksheet(
            [pr1, pr2],
            auth=FakeAuth(service_account_email="bob@example.com"),
        )

        rehearsal = worksheet.rehearse_write_access("A1:B2")
        b2 = self._verdict(rehearsal, "B2")

        # bob's exemption on PR2 must not cancel PR1's restriction
        self.assertEqual(b2.status, ProtectionVerdict.DENIED)
        self.assertEqual(b2.blocking_protected_range_ids, ["1"])
        self.assertNotIn("B2", rehearsal.allowed_cells)

    def test_unknown_group_membership_is_insufficient_evidence(self):
        protected_range = _grid_range(
            1,
            0,
            1,
            0,
            1,
            warningOnly=False,
            editors={"users": [], "groups": ["team@googlegroups.com"]},
        )
        worksheet, _ = self._make_worksheet(
            [protected_range],
            auth=FakeAuth(service_account_email="bob@example.com"),
        )

        rehearsal = worksheet.rehearse_write_access("A1:A1")
        verdict = self._verdict(rehearsal, "A1")

        self.assertEqual(
            verdict.status, ProtectionVerdict.INSUFFICIENT_EVIDENCE
        )
        self.assertEqual(verdict.evidence_protected_range_ids, ["1"])
        self.assertEqual(verdict.blocking_protected_range_ids, [])

    def test_known_group_membership_allows_write(self):
        protected_range = _grid_range(
            1,
            0,
            1,
            0,
            1,
            warningOnly=False,
            editors={"users": [], "groups": ["team@googlegroups.com"]},
        )
        worksheet, _ = self._make_worksheet(
            [protected_range],
            auth=FakeAuth(service_account_email="bob@example.com"),
        )

        rehearsal = worksheet.rehearse_write_access(
            "A1:A1", known_group_emails=["team@googlegroups.com"]
        )

        self.assertEqual(rehearsal.allowed_cells, ["A1"])

    def test_hard_denial_wins_over_missing_evidence(self):
        pr1 = _grid_range(
            1,
            0,
            1,
            0,
            1,
            warningOnly=False,
            editors={"users": ["alice@example.com"], "groups": []},
        )
        pr2 = _grid_range(
            2,
            0,
            1,
            0,
            1,
            warningOnly=False,
            editors={"users": [], "groups": ["team@googlegroups.com"]},
        )
        worksheet, _ = self._make_worksheet(
            [pr1, pr2],
            auth=FakeAuth(service_account_email="bob@example.com"),
        )

        verdict = self._verdict(
            worksheet.rehearse_write_access("A1:A1"), "A1"
        )

        self.assertEqual(verdict.status, ProtectionVerdict.DENIED)
        self.assertEqual(verdict.blocking_protected_range_ids, ["1"])
        self.assertEqual(verdict.evidence_protected_range_ids, ["2"])

    def test_range_without_editors_is_insufficient_evidence(self):
        protected_range = _grid_range(1, 0, 1, 0, 1, warningOnly=False)
        worksheet, _ = self._make_worksheet([protected_range])

        verdict = self._verdict(
            worksheet.rehearse_write_access("A1:A1"), "A1"
        )

        self.assertEqual(
            verdict.status, ProtectionVerdict.INSUFFICIENT_EVIDENCE
        )

    def test_requesting_user_can_edit_allows_without_identity(self):
        protected_range = _grid_range(
            1,
            0,
            1,
            0,
            1,
            warningOnly=False,
            requestingUserCanEdit=True,
            editors={"users": [], "groups": []},
        )
        worksheet, _ = self._make_worksheet([protected_range])

        rehearsal = worksheet.rehearse_write_access("A1:A1")

        self.assertEqual(rehearsal.allowed_cells, ["A1"])

    def test_numeric_boundaries_are_supported(self):
        worksheet, _ = self._make_worksheet([])

        rehearsal = worksheet.rehearse_write_access(1, 1, 2, 2)

        self.assertEqual(
            [verdict.cell for verdict in rehearsal.verdicts],
            ["A1", "B1", "A2", "B2"],
        )

    def test_identity_decoded_from_oauth_jwt(self):
        import base64
        import json

        payload = base64.urlsafe_b64encode(
            json.dumps({"email": "alice@example.com"}).encode()
        ).decode().rstrip("=")
        token = "header.{}.signature".format(payload)
        worksheet, _ = self._make_worksheet(
            [], auth=FakeAuth(token=token, refresh_token="rt")
        )

        self.assertEqual(
            worksheet._get_caller_identity(), "alice@example.com"
        )

    def test_malformed_token_leaves_identity_unknown(self):
        worksheet, _ = self._make_worksheet(
            [], auth=FakeAuth(token="not-a-jwt")
        )

        self.assertIsNone(worksheet._get_caller_identity())

    def test_result_groups_cells_by_status(self):
        protected_range = _grid_range(
            1,
            0,
            1,
            0,
            1,
            warningOnly=False,
            editors={"users": ["alice@example.com"], "groups": []},
        )
        worksheet, _ = self._make_worksheet(
            [protected_range],
            auth=FakeAuth(service_account_email="bob@example.com"),
        )

        rehearsal = worksheet.rehearse_write_access("A1:B1")

        self.assertIsInstance(rehearsal, WriteAccessRehearsal)
        self.assertEqual(rehearsal.denied_cells, ["A1"])
        self.assertEqual(rehearsal.allowed_cells, ["B1"])
        self.assertEqual(rehearsal.insufficient_evidence_cells, [])


class CachedWriteAccessRehearsalTest(TestCase):
    def _make_worksheet(self, protected_ranges, auth):
        client = HTTPClient(auth=None, session=Mock())
        client.fetch_sheet_metadata = Mock(
            return_value=_metadata(protected_ranges)
        )
        client.auth = auth
        properties = {
            "sheetId": 0,
            "title": "Sheet1",
            "index": 0,
            "gridProperties": {"rowCount": 20, "columnCount": 10},
        }
        worksheet = Worksheet(
            spreadsheet=None,
            properties=properties,
            spreadsheet_id="abc123",
            client=client,
        )
        return worksheet, client

    def test_cache_hit_does_not_refetch(self):
        worksheet, client = self._make_worksheet(
            [], FakeAuth(token="token-1", refresh_token="rt")
        )

        worksheet.rehearse_write_access_cached("A1:B2")
        calls_after_first = client.fetch_sheet_metadata.call_count
        worksheet.rehearse_write_access_cached("A1:B2")

        self.assertEqual(
            client.fetch_sheet_metadata.call_count, calls_after_first
        )

    def test_token_refresh_invalidates_cache_and_refetches(self):
        auth = FakeAuth(token="token-1", refresh_token="rt")
        worksheet, client = self._make_worksheet([], auth)

        worksheet.rehearse_write_access_cached("A1:B2")
        calls_before = client.fetch_sheet_metadata.call_count

        # credentials rotate: the cached allowance is no longer a guarantee
        auth.token = "token-2"
        with self.assertRaises(StaleProtectionMetadataError) as context:
            worksheet.rehearse_write_access_cached("A1:B2")

        self.assertTrue(context.exception.rehearsal.stale)
        self.assertEqual(
            client.fetch_sheet_metadata.call_count, calls_before + 1
        )

        # the fresh rehearsal now served from the rebuilt cache is not stale
        rehearsal = worksheet.rehearse_write_access_cached("A1:B2")
        self.assertFalse(rehearsal.stale)
        self.assertEqual(
            client.fetch_sheet_metadata.call_count, calls_before + 1
        )

    def test_different_range_rebuilds_cache_without_stale_flag(self):
        worksheet, _ = self._make_worksheet(
            [], FakeAuth(token="token-1", refresh_token="rt")
        )

        worksheet.rehearse_write_access_cached("A1:B2")
        rehearsal = worksheet.rehearse_write_access_cached("A1:A1")

        self.assertFalse(rehearsal.stale)
        self.assertEqual(rehearsal.allowed_cells, ["A1"])

    def test_clear_write_access_cache_forces_refetch(self):
        worksheet, client = self._make_worksheet(
            [], FakeAuth(token="token-1", refresh_token="rt")
        )

        worksheet.rehearse_write_access_cached("A1:B2")
        calls_before = client.fetch_sheet_metadata.call_count
        worksheet.clear_write_access_cache()
        worksheet.rehearse_write_access_cached("A1:B2")

        self.assertEqual(
            client.fetch_sheet_metadata.call_count, calls_before + 1
        )

    def test_new_group_membership_rebuilds_cache_without_stale_flag(self):
        protected_range = _grid_range(
            1,
            0,
            1,
            0,
            1,
            warningOnly=False,
            editors={"users": [], "groups": ["team@googlegroups.com"]},
        )
        worksheet, client = self._make_worksheet(
            [protected_range], FakeAuth(token="token-1", refresh_token="rt")
        )

        first = worksheet.rehearse_write_access_cached("A1:A1")
        self.assertEqual(
            first.insufficient_evidence_cells, ["A1"]
        )
        calls_before = client.fetch_sheet_metadata.call_count

        second = worksheet.rehearse_write_access_cached(
            "A1:A1", known_group_emails=["team@googlegroups.com"]
        )

        self.assertFalse(second.stale)
        self.assertEqual(second.allowed_cells, ["A1"])
        self.assertEqual(
            client.fetch_sheet_metadata.call_count, calls_before + 1
        )
