"""
gspread.listing
~~~~~~~~~~~~~~~

Models used when listing spreadsheet files through the Drive API.

Drive pagination is not a stable snapshot.  A file can be returned on more
than one page, and a file can disappear between pages because its sharing
permissions changed.  The objects in this module keep enough information to
tell those cases apart from a complete enumeration, and to avoid reporting
files as deleted when the listing is not conclusive.
"""

from collections.abc import Sequence as ABCSequence
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

# Reasons why a listing is not a complete enumeration.  They are stable strings
# so callers can inspect them without depending on exception messages.
# ``permission_change`` covers both a mid-listing 403/404/410 and a file that
# was visible earlier but is absent from a listing that is otherwise complete.
UNKNOWN_INCOMPLETE_SEARCH = "incomplete_search"
UNKNOWN_CROSS_PAGE_DUPLICATE = "cross_page_duplicate"
UNKNOWN_DUPLICATE = "duplicate_file"
UNKNOWN_PERMISSION_CHANGE = "permission_change"
UNKNOWN_REPEATED_PAGE_TOKEN = "repeated_page_token"
UNKNOWN_MISSING_IDENTITY = "missing_identity"
UNKNOWN_VISIBILITY_CHANGED = "visibility_changed"
UNKNOWN_PARTIAL_LISTING = "partial_listing"
UNKNOWN_MISSING_VISIBILITY = "missing_visibility"

# Anomalies that prevent the pages from being a complete enumeration.
_INCOMPLETE_REASONS = frozenset(
    {
        UNKNOWN_INCOMPLETE_SEARCH,
        UNKNOWN_PERMISSION_CHANGE,
        UNKNOWN_PARTIAL_LISTING,
        UNKNOWN_REPEATED_PAGE_TOKEN,
    }
)

# Anomalies that leave every page read but make the set unstable, so absence
# from the listing cannot be read as deletion.
_UNSTABLE_REASONS = frozenset(
    {
        UNKNOWN_CROSS_PAGE_DUPLICATE,
        UNKNOWN_DUPLICATE,
        UNKNOWN_VISIBILITY_CHANGED,
        UNKNOWN_MISSING_IDENTITY,
    }
)

# Basis on which a file is visible in the listing.
VISIBILITY_OWNED = "owned"
VISIBILITY_SHARED = "shared"
VISIBILITY_SHARED_DRIVE = "shared_drive"
VISIBILITY_NOT_OWNED = "not_owned"
VISIBILITY_UNKNOWN = "unknown"


def visibility_basis_for_file(file: Dict[str, Any]) -> str:
    """Return the Drive fields that explain why a file is visible.

    Drive only returns ``ownedByMe``, ``shared`` and ``driveId`` when they are
    requested.  When they are absent the caller cannot distinguish an owned
    file from a file shared with the account, so the basis is explicitly
    ``unknown``.
    """
    if file.get("ownedByMe") is True:
        return VISIBILITY_OWNED
    if file.get("shared") is True:
        return VISIBILITY_SHARED
    if file.get("driveId"):
        return VISIBILITY_SHARED_DRIVE
    if "ownedByMe" in file or "shared" in file:
        return VISIBILITY_NOT_OWNED
    return VISIBILITY_UNKNOWN


def _add_reason(reasons: List[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


@dataclass(frozen=True)
class PageCursor:
    """One page requested from the Drive files endpoint.

    ``request_token`` is the ``pageToken`` sent to Drive (``None`` for the
    first page) and ``next_token`` is the token returned by that page.  Both
    are kept so a listing can be resumed or audited without guessing how the
    pages were traversed.  ``page`` is the zero-based position of the page in
    this listing.
    """

    request_token: Optional[str]
    next_token: Optional[str]
    file_ids: Tuple[str, ...] = ()
    incomplete_search: bool = False
    error: Optional[str] = None
    page: int = 0

    @property
    def token(self) -> Optional[str]:
        """Alias for the token sent with this page."""
        return self.request_token

    @property
    def page_token(self) -> Optional[str]:
        return self.request_token

    @property
    def next_page_token(self) -> Optional[str]:
        return self.next_token

    def as_dict(self) -> Dict[str, Any]:
        return {
            "page": self.page,
            "request_token": self.request_token,
            "next_token": self.next_token,
            "file_ids": list(self.file_ids),
            "incomplete_search": self.incomplete_search,
            "error": self.error,
        }


@dataclass
class SpreadsheetFileDiff:
    """Differences between two listings.

    ``deleted`` only contains files that can actually be confirmed as deleted:
    both listings must be complete, reach the end of their pagination, cover
    the same query scope, and the file must not have been explicitly shared
    with the account.  A file known to be shared (or explicitly not owned)
    that is absent from the current listing is placed in ``unknown``; it may
    have disappeared because access was revoked, which is not a deletion.
    """

    added: List[str] = field(default_factory=list)
    retained: List[str] = field(default_factory=list)
    deleted: List[str] = field(default_factory=list)
    unknown: List[str] = field(default_factory=list)
    scope_matches: bool = True
    previous_complete: bool = False
    current_complete: bool = False
    reasons: List[str] = field(default_factory=list)

    @property
    def removed(self) -> List[str]:
        """Files confirmed as deleted (alias for :attr:`deleted`)."""
        return self.deleted

    @property
    def deleted_files(self) -> List[str]:
        return self.deleted

    @property
    def added_files(self) -> List[str]:
        return self.added

    @property
    def retained_files(self) -> List[str]:
        return self.retained

    @property
    def unknown_files(self) -> List[str]:
        """IDs whose absence cannot be explained conclusively."""
        return self.unknown

    @property
    def has_unknown(self) -> bool:
        return bool(self.unknown)

    @property
    def has_deletions(self) -> bool:
        return bool(self.deleted)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "added": list(self.added),
            "retained": list(self.retained),
            "deleted": list(self.deleted),
            "unknown": list(self.unknown),
            "scope_matches": self.scope_matches,
            "previous_complete": self.previous_complete,
            "current_complete": self.current_complete,
            "reasons": list(self.reasons),
            "complete": self.complete,
        }

    @property
    def missing(self) -> List[str]:
        """IDs absent from the current listing whose fate is unknown."""
        return self.unknown

    @property
    def absent(self) -> List[str]:
        """All IDs present before and absent now, confirmed or not."""
        return list(self.deleted) + [
            file_id for file_id in self.unknown if file_id not in self.deleted
        ]

    @property
    def permission_change_unknown(self) -> List[str]:
        """Absent IDs whose listing recorded a permission change."""
        if UNKNOWN_PERMISSION_CHANGE not in self.reasons:
            return []
        return list(self.unknown)

    @property
    def complete(self) -> bool:
        """Whether the diff can be treated as a complete reconciliation."""
        return (
            self.scope_matches
            and self.previous_complete
            and self.current_complete
            and not self.unknown
        )

    @property
    def is_complete(self) -> bool:
        return self.complete


@dataclass
class SpreadsheetFileListing:
    """A deduplicated, auditable snapshot of a paginated file listing."""

    files: List[Dict[str, Any]] = field(default_factory=list)
    query_scope: Dict[str, Any] = field(default_factory=dict)
    page_cursors: List[PageCursor] = field(default_factory=list)
    unknown_reasons: List[str] = field(default_factory=list)
    duplicate_ids: List[str] = field(default_factory=list)
    page_by_id: Dict[str, Optional[str]] = field(default_factory=dict)
    visibility: Dict[str, str] = field(default_factory=dict)
    unidentified_files: List[Dict[str, Any]] = field(default_factory=list)
    last_response: Any = None
    error: Optional[BaseException] = None
    requested_page_token: Optional[str] = None
    finished: bool = True
    partial: bool = False

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self.files)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.files[index]

    def __contains__(self, item: object) -> bool:
        if isinstance(item, str):
            return item in self.ids
        return item in self.files

    @property
    def ids(self) -> List[str]:
        """File IDs in first-seen order, with duplicate IDs removed."""
        return [file["id"] for file in self.files if file.get("id")]

    @property
    def file_ids(self) -> List[str]:
        return self.ids

    @property
    def by_id(self) -> Dict[str, Dict[str, Any]]:
        return {file["id"]: file for file in self.files if file.get("id")}

    @property
    def files_by_id(self) -> Dict[str, Dict[str, Any]]:
        return self.by_id

    @property
    def names(self) -> Dict[str, List[str]]:
        """Map each title to every file ID that uses it.

        Titles are deliberately not used as identities: two different files
        can share a title, and both belong in the listing.
        """
        names: Dict[str, List[str]] = {}
        for file in self.files:
            file_id = file.get("id")
            if file_id:
                names.setdefault(file.get("name", ""), []).append(file_id)
        return names

    @property
    def titles(self) -> Dict[str, List[str]]:
        return self.names

    @property
    def query(self) -> Optional[str]:
        """The Drive ``q`` expression used for every page."""
        value = self.query_scope.get("q")
        return value if isinstance(value, str) else None

    @property
    def q(self) -> Optional[str]:
        return self.query

    @property
    def scope(self) -> Dict[str, Any]:
        return self.query_scope

    @property
    def scope_key(self) -> Tuple[Any, ...]:
        return _scope_key(self.query_scope)

    @property
    def title(self) -> Optional[str]:
        return self.query_scope.get("title")

    @property
    def folder_id(self) -> Optional[str]:
        return self.query_scope.get("folder_id")

    @property
    def query_params(self) -> Dict[str, Any]:
        params = self.query_scope.get("params")
        return dict(params) if isinstance(params, dict) else {}

    @property
    def page_tokens(self) -> List[Optional[str]]:
        return [cursor.request_token for cursor in self.page_cursors]

    @property
    def cursors(self) -> List[Optional[str]]:
        """Page tokens sent to Drive, in request order."""
        return self.page_tokens

    @property
    def cursor_objects(self) -> List[PageCursor]:
        return list(self.page_cursors)

    @property
    def pages(self) -> List[PageCursor]:
        return list(self.page_cursors)

    @property
    def page_token(self) -> Optional[str]:
        """The token this listing started from (``None`` when unpaged)."""
        return self.requested_page_token

    @property
    def next_page_token(self) -> Optional[str]:
        """The last token returned by Drive, or ``None`` if fully enumerated."""
        if not self.page_cursors:
            return self.requested_page_token
        return self.page_cursors[-1].next_token

    @property
    def visibility_basis(self) -> Dict[str, str]:
        return dict(self.visibility)

    @property
    def duplicates(self) -> List[str]:
        return list(self.duplicate_ids)

    @property
    def reasons(self) -> List[str]:
        return list(self.unknown_reasons)

    @property
    def incomplete_reasons(self) -> List[str]:
        return [
            reason
            for reason in self.unknown_reasons
            if reason in _INCOMPLETE_REASONS
        ]

    @property
    def instability_reasons(self) -> List[str]:
        return [
            reason
            for reason in self.unknown_reasons
            if reason in _UNSTABLE_REASONS
        ]

    @property
    def reason(self) -> Optional[str]:
        """The first paging anomaly observed, if any."""
        return self.unknown_reasons[0] if self.unknown_reasons else None

    @property
    def unknown_reason(self) -> Optional[str]:
        return self.reason

    @property
    def complete(self) -> bool:
        """Whether every page was enumerated.

        A permission change, an incomplete search, a repeated page cursor or
        a cursor-started listing all prevent a complete enumeration.  A
        cross-page duplicate does not: all pages were still read, but see
        :attr:`deletion_safe`.
        """
        return self.finished and not (
            _INCOMPLETE_REASONS & set(self.unknown_reasons)
        )

    @property
    def incomplete(self) -> bool:
        return not self.complete

    @property
    def is_complete(self) -> bool:
        return self.complete

    @property
    def fully_enumerated(self) -> bool:
        """Whether all pages were read.  Does not imply the set is stable."""
        return self.complete

    @property
    def deletion_safe(self) -> bool:
        """Whether absence from this listing can be read as deletion.

        Even when all pages were read, cross-page duplicates, missing file
        identities, or a visibility basis that changed while paging make the
        result an unstable snapshot.  In those cases an absent file may have
        been skipped or hidden, so it is not reported as deleted.
        """
        return (
            self.complete
            and not self.partial
            and not (_UNSTABLE_REASONS & set(self.unknown_reasons))
        )

    @property
    def reached_end(self) -> bool:
        """Whether paging reached the last page, even if it was not stable."""
        return self.finished

    @property
    def stable(self) -> bool:
        return self.deletion_safe

    @property
    def unknown(self) -> bool:
        """Whether the enumeration itself is inconclusive."""
        return not self.complete

    @property
    def has_unknown(self) -> bool:
        return self.unknown

    @property
    def has_instability(self) -> bool:
        return bool(_UNSTABLE_REASONS & set(self.unknown_reasons))

    @property
    def has_cross_page_duplicates(self) -> bool:
        return UNKNOWN_CROSS_PAGE_DUPLICATE in self.unknown_reasons

    @property
    def has_permission_change(self) -> bool:
        return UNKNOWN_PERMISSION_CHANGE in self.unknown_reasons

    @property
    def unknown_due_to_permission_change(self) -> bool:
        return self.has_permission_change

    @property
    def unknown_due_to_cross_page_duplicates(self) -> bool:
        return self.has_cross_page_duplicates

    @property
    def complete_enumeration(self) -> bool:
        """A complete enumeration with no duplicates or visibility changes."""
        return self.complete and self.deletion_safe

    @property
    def can_confirm_deletions(self) -> bool:
        """Whether a comparison with this listing may report deletions."""
        return self.complete_enumeration

    @property
    def fully_enumerated_and_stable(self) -> bool:
        return self.complete_enumeration

    @property
    def is_unknown(self) -> bool:
        return self.unknown

    @property
    def completeness(self) -> str:
        """``complete``, ``unstable``, ``partial`` or ``unknown``.

        ``complete`` means every page was read into a stable set.  ``unstable``
        means every page was read, but a cross-page duplicate or a changed
        visibility basis makes absence inconclusive.  ``partial`` means the
        listing started from a page cursor.  ``unknown`` means a permission
        change, incomplete search, or another hard failure stopped paging.
        """
        if self.partial:
            return "partial"
        if not self.complete:
            return "unknown"
        if not self.deletion_safe:
            return "unstable"
        return "complete"

    @property
    def status(self) -> str:
        """``complete``, ``unknown`` or ``partial``; see :attr:`completeness`."""
        return self.completeness

    @property
    def visibility_bases(self) -> List[str]:
        return [self.visibility_for(file_id) for file_id in self.ids]

    @property
    def page_size(self) -> Optional[int]:
        value = self.query_params.get("pageSize")
        return value if isinstance(value, int) else None

    @property
    def fields(self) -> Optional[str]:
        value = self.query_params.get("fields")
        return value if isinstance(value, str) else None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "files": list(self.files),
            "ids": self.ids,
            "query": self.query,
            "q": self.query,
            "query_scope": dict(self.query_scope),
            "scope": dict(self.query_scope),
            "params": self.query_params,
            "cursors": self.page_tokens,
            "page_cursors": [
                cursor.as_dict() for cursor in self.page_cursors
            ],
            "page_tokens": self.page_tokens,
            "page_token": self.requested_page_token,
            "next_page_token": self.next_page_token,
            "visibility": dict(self.visibility),
            "visibility_basis": dict(self.visibility),
            "duplicates": list(self.duplicate_ids),
            "reasons": list(self.unknown_reasons),
            "unknown_reasons": list(self.unknown_reasons),
            "complete": self.complete,
            "unknown": self.unknown,
            "deletion_safe": self.deletion_safe,
            "completeness": self.completeness,
            "partial": self.partial,
        }

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.as_dict()

    @property
    def info(self) -> Dict[str, Any]:
        return self.as_dict()

    def to_dict(self) -> Dict[str, Any]:
        return self.as_dict()

    def page_for(self, file_id: str) -> Optional[str]:
        """Return the page token on which ``file_id`` was first seen.

        The first page has no request token, so it is represented by ``None``;
        use :meth:`page_index_for` when a numeric position is needed.
        """
        return self.page_by_id.get(file_id)

    def page_index_for(self, file_id: str) -> Optional[int]:
        """Return the zero-based page on which ``file_id`` was first seen."""
        for cursor in self.page_cursors:
            if file_id in cursor.file_ids:
                return cursor.page
        return None

    def visibility_for(self, file_id: str) -> str:
        if file_id in self.visibility:
            return self.visibility[file_id]
        file = self.by_id.get(file_id)
        if file is not None:
            return visibility_basis_for_file(file)
        return VISIBILITY_UNKNOWN

    @property
    def visibility_evidence(self) -> Dict[str, Dict[str, Any]]:
        """Raw Drive fields that support each file's visibility basis."""
        evidence: Dict[str, Dict[str, Any]] = {}
        for file in self.files:
            file_id = file.get("id")
            if not file_id:
                continue
            evidence[file_id] = {
                key: file[key]
                for key in ("ownedByMe", "shared", "driveId", "parents")
                if key in file
            }
        return evidence

    def get(self, file_id: str) -> Optional[Dict[str, Any]]:
        return self.by_id.get(file_id)

    def diff(self, previous: Any) -> SpreadsheetFileDiff:
        """Compare this listing with an earlier one without inventing deletes.

        A file that is absent now is only reported in ``deleted`` when both
        listings are complete enumerations of the same query scope, both
        reached the last page, and the earlier listing recorded the file as
        owned.  If either listing is partial or unstable, or the file was
        shared / explicitly not owned / of unknown visibility, the missing ID
        is reported in ``unknown`` instead.
        """
        previous = _coerce_listing(previous)
        previous_ids = set(previous.ids)
        current_ids = set(self.ids)

        added = [file_id for file_id in self.ids if file_id not in previous_ids]
        retained = [file_id for file_id in self.ids if file_id in previous_ids]
        absent = [file_id for file_id in previous.ids if file_id not in current_ids]

        scope_matches = _scope_key(self.query_scope) == _scope_key(previous.query_scope)
        listings_complete = (
            scope_matches
            and previous.complete
            and self.complete
            and previous.deletion_safe
            and self.deletion_safe
        )

        reasons: List[str] = []
        if not scope_matches:
            reasons.append("query_scope_changed")
        if not previous.complete:
            reasons.append("previous_listing_incomplete")
        if not self.complete:
            reasons.append("current_listing_incomplete")
        if not self.finished:
            reasons.append("current_listing_stopped_early")
        if not previous.finished:
            reasons.append("previous_listing_stopped_early")
        if self.complete and not self.deletion_safe:
            reasons.append("current_listing_unstable")
        if previous.complete and not previous.deletion_safe:
            reasons.append("previous_listing_unstable")
        for reason in self.unknown_reasons:
            _add_reason(reasons, reason)

        deleted: List[str] = []
        unknown: List[str] = []
        for file_id in absent:
            if not listings_complete:
                unknown.append(file_id)
                continue

            basis = previous.visibility_for(file_id)
            if basis == VISIBILITY_OWNED:
                deleted.append(file_id)
            elif basis == VISIBILITY_UNKNOWN:
                # Without an ownedByMe/shared basis, absence cannot be
                # attributed to deletion rather than a visibility change.
                unknown.append(file_id)
                _add_reason(reasons, UNKNOWN_MISSING_VISIBILITY)
            else:
                # Shared and explicitly-not-owned files may have vanished
                # because the account lost access.
                unknown.append(file_id)
                _add_reason(reasons, UNKNOWN_PERMISSION_CHANGE)

        return SpreadsheetFileDiff(
            added=added,
            retained=retained,
            deleted=deleted,
            unknown=unknown,
            scope_matches=scope_matches,
            previous_complete=previous.complete,
            current_complete=self.complete,
            reasons=reasons,
        )

    def reconcile(self, previous: "SpreadsheetFileListing") -> SpreadsheetFileDiff:
        """Alias for :meth:`diff`."""
        return self.diff(previous)

    compare = reconcile


def reconcile(previous: Any, current: Any) -> SpreadsheetFileDiff:
    """Reconcile two listings, never inventing deletions.

    ``previous`` is the earlier listing and ``current`` the newer one.
    """
    return _coerce_listing(current).diff(previous)


def diff_listings(previous: Any, current: Any) -> SpreadsheetFileDiff:
    """Alias for :func:`reconcile`."""
    return reconcile(previous, current)


def _scope_key(scope: Dict[str, Any]) -> Tuple[Any, ...]:
    """Identity of a query scope, ignoring page size, fields and page tokens."""
    return (
        scope.get("title"),
        scope.get("folder_id"),
        scope.get("mime_type"),
        scope.get("q"),
    )


def _coerce_listing(value: Any) -> SpreadsheetFileListing:
    if isinstance(value, SpreadsheetFileListing):
        return value
    if isinstance(value, SpreadsheetFileList):
        return value.listing
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("expected a sequence of file mappings, not text")
    if isinstance(value, ABCSequence):
        return SpreadsheetFileListing(list(value))
    raise TypeError("expected a SpreadsheetFileListing or a sequence of files")


class SpreadsheetFileList(list):
    """A normal list of file dictionaries with listing metadata attached.

    It intentionally subclasses :class:`list` so existing callers that expect
    ``Client.list_spreadsheet_files()`` to return a list keep working, while
    ``.listing`` exposes the query scope, page cursors, visibility basis and
    completeness information.
    """

    def __init__(
        self,
        files: Sequence[Dict[str, Any]] = (),
        listing: Optional[SpreadsheetFileListing] = None,
    ) -> None:
        super().__init__(files)
        if listing is None:
            listing = SpreadsheetFileListing(list(files))
        self.listing = listing

    @property
    def files(self) -> "SpreadsheetFileList":
        return self

    def diff(self, previous: Any) -> SpreadsheetFileDiff:
        return self.listing.diff(previous)

    def reconcile(self, previous: Any) -> SpreadsheetFileDiff:
        return self.listing.reconcile(previous)

    compare = reconcile

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, str):
            return self.listing.as_dict()[index]
        return list.__getitem__(self, index)

    def get(self, key: Any, default: Any = None) -> Any:
        if isinstance(key, str):
            return self.listing.as_dict().get(key, default)
        return list.__getitem__(self, key)

    def keys(self) -> List[str]:
        return list(self.listing.as_dict())

    def values(self) -> List[Any]:
        return list(self.listing.as_dict().values())

    def items(self) -> List[Tuple[str, Any]]:
        return list(self.listing.as_dict().items())

    def __getattr__(self, name: str) -> Any:
        listing = self.__dict__.get("listing")
        if listing is None:
            raise AttributeError(name)
        return getattr(listing, name)
