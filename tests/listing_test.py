import json

from requests import Response

from gspread.client import Client
from gspread.exceptions import APIError
from gspread.listing import (
    UNKNOWN_CROSS_PAGE_DUPLICATE,
    UNKNOWN_PERMISSION_CHANGE,
    VISIBILITY_OWNED,
    VISIBILITY_SHARED,
    SpreadsheetFileListing,
)


def _response(payload, status_code=200):
    response = Response()
    response.status_code = status_code
    response._content = json.dumps(payload).encode("utf-8")
    response.headers["Content-Type"] = "application/json"
    return response


class FakeHTTPClient:
    """Returns queued Drive responses and records request params."""

    def __init__(self, pages):
        self.pages = list(pages)
        self.calls = []

    def request(self, method, url, params=None):
        self.calls.append(
            {"method": method, "url": url, "params": dict(params or {})}
        )
        item = self.pages.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _client(pages):
    client = Client.__new__(Client)
    client.http_client = FakeHTTPClient(pages)
    return client


def _file(file_id, name="Same title", **extra):
    data = {
        "id": file_id,
        "name": name,
        "createdTime": "2026-01-01T00:00:00Z",
        "modifiedTime": "2026-01-02T00:00:00Z",
    }
    data.update(extra)
    return data


def test_same_title_files_with_different_ids_are_both_kept():
    client = _client(
        [
            _response(
                {
                    "files": [_file("owned-id", ownedByMe=True), _file("shared-id")],
                    "nextPageToken": None,
                }
            )
        ]
    )

    files = client.list_spreadsheet_files(include_visibility=True)

    assert [file["id"] for file in files] == ["owned-id", "shared-id"]
    assert files.names["Same title"] == ["owned-id", "shared-id"]
    assert len(files) == 2
    assert files.complete
    assert files.deletion_safe


def test_cross_page_duplicate_is_collapsed_but_marks_the_listing_unstable():
    client = _client(
        [
            _response(
                {
                    "files": [_file("a", ownedByMe=True)],
                    "nextPageToken": "page-2",
                }
            ),
            _response(
                {
                    "files": [_file("a", ownedByMe=True), _file("b", ownedByMe=True)],
                    "nextPageToken": None,
                }
            ),
        ]
    )

    listing = client.list_spreadsheet_files_detailed(include_visibility=True)

    assert listing.ids == ["a", "b"]
    assert listing.duplicate_ids == ["a"]
    assert UNKNOWN_CROSS_PAGE_DUPLICATE in listing.reasons
    assert listing.complete
    assert not listing.deletion_safe
    assert listing.completeness == "unstable"
    assert not listing.can_confirm_deletions

    assert listing.page_tokens == [None, "page-2"]
    assert [cursor.file_ids for cursor in listing.page_cursors] == [
        ("a",),
        ("a", "b"),
    ]
    assert listing.page_for("b") == "page-2"
    assert listing.query_scope["title"] is None
    assert listing.query_scope["folder_id"] is None
    assert listing.query == 'mimeType="application/vnd.google-apps.spreadsheet"'


def test_cross_page_duplicate_never_reports_missing_files_as_deleted():
    previous = client_listing(
        [{"files": [_file("a"), _file("b")], "nextPageToken": None}]
    )
    # The replay repeats "a" on the second page and never returns "b".
    current = client_listing(
        [
            {"files": [_file("a")], "nextPageToken": "page-2"},
            {"files": [_file("a")], "nextPageToken": None},
        ]
    )

    diff = current.diff(previous)

    assert diff.deleted == []
    assert diff.unknown == ["b"]
    assert "query_scope_changed" not in diff.reasons


def client_listing(pages, **kwargs):
    return _client([_response(page) for page in pages]).list_spreadsheet_files_detailed(
        **kwargs
    )


def test_permission_change_mid_listing_is_unknown_not_deleted():
    forbidden = _response(
        {"error": {"code": 403, "message": "forbidden"}}, status_code=403
    )
    client = _client(
        [
            _response(
                {
                    "files": [_file("kept")],
                    "nextPageToken": "page-2",
                }
            ),
            APIError(forbidden),
        ]
    )

    current = client.list_spreadsheet_files_detailed()
    previous = SpreadsheetFileListing(
        files=[_file("kept"), _file("lost")],
        query_scope=current.query_scope,
        requested_page_token=None,
    )

    assert current.error is not None
    assert UNKNOWN_PERMISSION_CHANGE in current.reasons
    assert not current.complete
    assert current.completeness == "unknown"

    diff = current.diff(previous)
    assert diff.deleted == []
    assert diff.unknown == ["lost"]


def test_shared_file_disappearing_is_unknown_while_owned_file_is_deleted():
    previous = client_listing(
        [
            {
                "files": [
                    _file("owned", ownedByMe=True),
                    _file("shared", shared=True),
                ],
                "nextPageToken": None,
            }
        ],
        include_visibility=True,
    )
    current = client_listing(
        [{"files": [_file("owned", ownedByMe=True)], "nextPageToken": None}],
        include_visibility=True,
    )

    diff = current.diff(previous)

    assert diff.deleted == []
    assert diff.unknown == ["shared"]
    assert UNKNOWN_PERMISSION_CHANGE in diff.reasons


def test_changed_query_scope_makes_absence_unknown():
    previous = client_listing(
        [{"files": [_file("a")], "nextPageToken": None}], title="Old title"
    )
    current = client_listing(
        [{"files": [], "nextPageToken": None}], title="New title"
    )

    diff = current.diff(previous)

    assert diff.scope_matches is False
    assert diff.deleted == []
    assert diff.unknown == ["a"]
    assert "query_scope_changed" in diff.reasons


def test_starting_from_page_token_is_partial_and_not_deletion_safe():
    client = _client(
        [
            _response(
                {
                    "files": [_file("a", ownedByMe=True)],
                    "nextPageToken": None,
                }
            )
        ]
    )

    listing = client.list_spreadsheet_files_detailed(
        page_token="resume-token", include_visibility=True
    )

    assert listing.page_token == "resume-token"
    assert listing.page_tokens == ["resume-token"]
    assert not listing.complete
    assert not listing.deletion_safe
    assert listing.completeness == "partial"


def test_visibility_basis_is_recorded_from_drive_fields():
    client = _client(
        [
            _response(
                {
                    "files": [
                        _file("owned", ownedByMe=True),
                        _file("shared", shared=True),
                        _file("drive", driveId="drive-1"),
                        _file("unknown"),
                    ],
                    "nextPageToken": None,
                }
            )
        ]
    )

    listing = client.list_spreadsheet_files_detailed(include_visibility=True)

    assert listing.visibility_for("owned") == VISIBILITY_OWNED
    assert listing.visibility_for("shared") == VISIBILITY_SHARED
    assert listing.visibility_evidence["owned"] == {"ownedByMe": True}
    assert listing.visibility_evidence["drive"] == {"driveId": "drive-1"}


def test_visibility_fields_are_requested_only_when_asked_for():
    client = _client([_response({"files": [], "nextPageToken": None})])
    client.list_spreadsheet_files()
    default_params = client.http_client.calls[0]["params"]
    assert "ownedByMe" not in default_params["fields"]

    client = _client([_response({"files": [], "nextPageToken": None})])
    client.list_spreadsheet_files(include_visibility=True)
    visible_params = client.http_client.calls[0]["params"]
    assert "ownedByMe" in visible_params["fields"]
    assert "shared" in visible_params["fields"]
    assert "incompleteSearch" in visible_params["fields"]


def test_incomplete_search_is_preserved_as_unknown():
    client = _client(
        [
            _response(
                {
                    "files": [_file("a")],
                    "nextPageToken": None,
                    "incompleteSearch": True,
                }
            )
        ]
    )

    listing = client.list_spreadsheet_files_detailed()

    assert "incomplete_search" in listing.reasons
    assert not listing.complete
    assert listing.completeness == "unknown"


def test_repeated_page_token_stops_instead_of_looping():
    client = _client(
        [
            _response({"files": [_file("a")], "nextPageToken": "same"}),
            _response({"files": [_file("b")], "nextPageToken": "same"}),
        ]
    )

    listing = client.list_spreadsheet_files_detailed()

    assert listing.ids == ["a", "b"]
    assert "repeated_page_token" in listing.reasons
    assert not listing.complete
    assert len(client.http_client.calls) == 2


def test_list_result_is_still_a_list_for_existing_callers():
    client = _client(
        [_response({"files": [_file("a")], "nextPageToken": None})]
    )
    files = client.list_spreadsheet_files()

    assert isinstance(files, list)
    assert files[0]["id"] == "a"
    assert files[0] == files.listing.files[0]
    assert files["complete"] is True


def test_legacy_private_listing_still_returns_files_and_response():
    response = _response({"files": [_file("a")], "nextPageToken": None})
    client = _client([response])

    files, last_response = client._list_spreadsheet_files()

    assert [file["id"] for file in files] == ["a"]
    assert last_response is response


def test_diff_requires_both_listings_to_be_complete():
    previous = SpreadsheetFileListing(
        files=[_file("a")],
        query_scope={},
        finished=False,
    )
    current = SpreadsheetFileListing(files=[], query_scope={})

    diff = current.diff(previous)

    assert diff.deleted == []
    assert diff.unknown == ["a"]
    assert "previous_listing_stopped_early" in diff.reasons


def test_diff_accepts_a_plain_list_of_files():
    previous = [_file("a")]
    current = SpreadsheetFileListing(files=[], query_scope={})

    diff = current.diff(previous)

    assert diff.deleted == []
    assert diff.unknown == ["a"]


def test_page_size_and_fields_are_recorded_in_query_scope():
    client = _client([_response({"files": [], "nextPageToken": None})])
    listing = client.list_spreadsheet_files_detailed(page_size=7)

    assert listing.page_size == 7
    assert listing.query_params["pageSize"] == 7
    assert (
        listing.fields
        == "kind,nextPageToken,files(id,name,createdTime,modifiedTime)"
    )
