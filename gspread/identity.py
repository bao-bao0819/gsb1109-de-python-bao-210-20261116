"""
gspread.identity
~~~~~~~~~~~~~~~~

Stable identities, publication history and reference graphs for spreadsheet
worksheets (the "sources" of a collection pipeline).

Titles are presentation, not identity
--------------------------------------

A worksheet title can be changed and later reused by another worksheet within
the same spreadsheet. Any bookkeeping that keys sources by title therefore
loses the original object. This module keeps a ledger where:

* an identity is anchored to the spreadsheet ID plus the worksheet ``sheetId``;
* the first time an identity is observed becomes its immutable
  ``published_at`` timestamp;
* every observation is stored as a revision, so a title at a point in time can
  always be recovered;
* identity links are created only from structural evidence: the same
  ``sheetId`` (rename), an explicit duplicate event with its source sheet ID
  (copy), or an explicit delete followed by a new ``sheetId`` (recreation).

Titles, business developer metadata and other content matches are recorded as
*candidates* only. They never transfer an identity. This is what prevents a
copied worksheet that carries the same business metadata from claiming the
identity of the original.

Historical reads
----------------

:meth:`IdentityLedger.snapshot` and :meth:`IdentityLedger.reference_graph`
resolve titles and references as of a timestamp. They never substitute the
current title for the title that was valid then.

Reference edges
---------------

Reference edges are captured per revision from static A1 formulas and from
developer metadata entries that explicitly carry a target. An edge keeps the
target identity, revision and title from the moment it was observed. Each new
observation resolves references again; old edges are never rewritten.

Only references with an explicit target (a stable UID or a sheet ID) continue
to follow an object through a rename. A title-only reference is resolved only
while that title has a single live owner; when the title is ambiguous the edge
is kept as ``ambiguous``/``unresolved`` instead of being guessed.

Timestamps
----------

Snapshots and structural events are ordered by explicit observation times.
Publishing a snapshot earlier than the ledger's last known observation raises
:class:`~gspread.exceptions.IdentityTimelineError`. Events recorded by
gspread without an explicit time are stamped at the ledger's latest known
observation, so local calls never move the timeline into the future.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .exceptions import (
    IdentityError,
    IdentityTimelineError,
)
from .references import (
    FormulaReference,
    extract_formula_references,
    extract_sheet_titles,
    find_formula_cell,
)
from .utils import StrEnum


class ChangeKind(StrEnum):
    """How an observed worksheet relates to the ledger."""

    new = "new"
    unchanged = "unchanged"
    renamed = "renamed"
    copied = "copied"
    recreated = "recreated"
    deleted = "deleted"


class ReferenceStatus(StrEnum):
    """Resolution status of a reference edge."""

    resolved = "resolved"
    unresolved = "unresolved"
    ambiguous = "ambiguous"


class ReferenceSource(StrEnum):
    """Origin of a reference edge."""

    formula = "formula"
    developer_metadata = "developer_metadata"


class Evidence(StrEnum):
    """Evidence used in reconciliation decisions."""

    same_sheet_id = "same_sheet_id"
    duplicate_event = "duplicate_event"
    delete_event = "delete_event"
    absent_from_complete_snapshot = "absent_from_complete_snapshot"
    new_sheet_id = "new_sheet_id"
    sheet_id_reused_after_delete = "sheet_id_reused_after_delete"
    title_reused_after_delete = "title_reused_after_delete"
    title_changed = "title_changed"
    explicit_target_uid = "explicit_target_uid"
    explicit_target_sheet_id = "explicit_target_sheet_id"
    title_only = "title_only"
    historical_title = "historical_title"
    title_ambiguous = "title_ambiguous"
    title_unowned = "title_unowned"


_REFERENCE_KEY = "gspread.reference"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def coerce_datetime(
    value: Optional[Any], default: Optional[datetime] = None
) -> datetime:
    """Return a timezone-aware UTC :class:`datetime`.

    ``value`` may be ``None``, a :class:`datetime` or an ISO-8601 string.
    """
    if value is None:
        if default is not None:
            return default
        return _utc_now()

    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    raise TypeError("time must be a datetime, an ISO-8601 string or None")


def _datetime_key(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


@dataclass(frozen=True)
class DeveloperMetadata:
    """A developer metadata entry as returned by the Sheets API.

    ``value`` is deliberately kept opaque: business metadata is content, not
    identity evidence.
    """

    key: str = ""
    value: Any = ""
    metadata_id: Optional[int] = None
    visibility: Optional[str] = None
    owner_sheet_id: Optional[int] = None

    @classmethod
    def from_api(cls, entry: Mapping[str, Any]) -> "DeveloperMetadata":
        location = entry.get("location") or {}
        return cls(
            key=str(entry.get("metadataKey", "")),
            value=entry.get("metadataValue", ""),
            metadata_id=entry.get("metadataId"),
            visibility=entry.get("visibility"),
            owner_sheet_id=location.get("sheetId"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "metadataId": self.metadata_id,
            "visibility": self.visibility,
            "ownerSheetId": self.owner_sheet_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DeveloperMetadata":
        return cls(
            key=str(data.get("key", "")),
            value=data.get("value", ""),
            metadata_id=data.get("metadataId"),
            visibility=data.get("visibility"),
            owner_sheet_id=data.get("ownerSheetId"),
        )


@dataclass(frozen=True)
class SheetEvent:
    """A structural event observed alongside a snapshot.

    ``kind`` is either ``"duplicate"`` or ``"delete"``.

    For a duplicate event, ``sheet_id`` is the *new* sheet and either
    ``source_sheet_id`` or ``source_uid`` identifies the sheet it was copied
    from. For a delete event, ``sheet_id`` is the deleted sheet.
    """

    kind: str
    sheet_id: int
    source_sheet_id: Optional[int] = None
    at: Optional[datetime] = None
    reason: str = ""
    source_uid: Optional[str] = None
    title: Optional[str] = None

    @classmethod
    def duplicate(
        cls,
        new_sheet_id: Optional[int] = None,
        source_sheet_id: Optional[int] = None,
        at: Optional[Any] = None,
        source_uid: Optional[str] = None,
        title: Optional[str] = None,
    ) -> "SheetEvent":
        return cls(
            kind="duplicate",
            sheet_id=int(new_sheet_id) if new_sheet_id is not None else -1,
            source_sheet_id=(
                int(source_sheet_id) if source_sheet_id is not None else None
            ),
            at=coerce_datetime(at) if at is not None else None,
            source_uid=source_uid,
            title=title,
        )

    @classmethod
    def delete(
        cls, sheet_id: int, at: Optional[Any] = None, reason: str = ""
    ) -> "SheetEvent":
        return cls(
            kind="delete",
            sheet_id=int(sheet_id),
            at=coerce_datetime(at) if at is not None else None,
            reason=reason,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "sheetId": self.sheet_id,
            "sourceSheetId": self.source_sheet_id,
            "sourceUid": self.source_uid,
            "title": self.title,
            "at": _datetime_key(self.at) if self.at is not None else None,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SheetEvent":
        return cls(
            kind=str(data.get("kind", "")),
            sheet_id=int(data.get("sheetId", 0)),
            source_sheet_id=data.get("sourceSheetId"),
            at=coerce_datetime(data["at"]) if data.get("at") else None,
            reason=str(data.get("reason", "")),
            source_uid=data.get("sourceUid"),
            title=data.get("title"),
        )


@dataclass(frozen=True)
class SheetObservation:
    """A worksheet as seen in one snapshot.

    ``developer_metadata`` and ``formulas`` may be ``None`` when that part of
    the worksheet was not collected. A missing part is carried over from the
    previous revision instead of being interpreted as an empty value.
    """

    sheet_id: int
    title: str
    index: int = 0
    developer_metadata: Optional[Tuple[DeveloperMetadata, ...]] = None
    formulas: Optional[Mapping[str, str]] = None

    @classmethod
    def from_api(
        cls,
        sheet: Mapping[str, Any],
        formulas: Optional[Mapping[str, str]] = None,
        extra_metadata: Optional[Iterable[Mapping[str, Any]]] = None,
        metadata_observed: Optional[bool] = None,
    ) -> "SheetObservation":
        properties = sheet.get("properties") or {}
        sheet_id = int(properties["sheetId"])
        extra_metadata = list(extra_metadata or ())

        if metadata_observed is None:
            metadata_observed = (
                "developerMetadata" in sheet
                or bool(extra_metadata)
            )

        metadata = None
        if metadata_observed:
            entries = [
                DeveloperMetadata.from_api(entry)
                for entry in (sheet.get("developerMetadata") or [])
            ]
            entries.extend(
                DeveloperMetadata.from_api(entry)
                for entry in extra_metadata
                if (entry.get("location") or {}).get("sheetId") == sheet_id
            )
            metadata = tuple(entries)

        return cls(
            sheet_id=sheet_id,
            title=str(properties.get("title", "")),
            index=int(properties.get("index", 0)),
            developer_metadata=metadata,
            formulas=dict(formulas) if formulas is not None else None,
        )

    def with_formulas(self, formulas: Mapping[str, str]) -> "SheetObservation":
        return SheetObservation(
            sheet_id=self.sheet_id,
            title=self.title,
            index=self.index,
            developer_metadata=self.developer_metadata,
            formulas=dict(formulas),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sheetId": self.sheet_id,
            "title": self.title,
            "index": self.index,
            "developerMetadata": (
                [m.to_dict() for m in self.developer_metadata]
                if self.developer_metadata is not None
                else None
            ),
            "formulas": dict(self.formulas) if self.formulas is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SheetObservation":
        metadata = data.get("developerMetadata", None)
        formulas = data.get("formulas", None)
        return cls(
            sheet_id=int(data["sheetId"]),
            title=str(data.get("title", "")),
            index=int(data.get("index", 0)),
            developer_metadata=(
                tuple(DeveloperMetadata.from_dict(m) for m in metadata)
                if metadata is not None
                else None
            ),
            formulas=dict(formulas) if formulas is not None else None,
        )


@dataclass(frozen=True)
class SheetSnapshot:
    """A complete or partial view of the worksheets in one spreadsheet."""

    spreadsheet_id: str
    sheets: Tuple[SheetObservation, ...]
    complete: bool = True
    events: Tuple[SheetEvent, ...] = ()
    observed_at: Optional[datetime] = None

    @classmethod
    def from_api(
        cls,
        data: Mapping[str, Any],
        formulas: Optional[Mapping[Any, Mapping[str, str]]] = None,
        events: Iterable[SheetEvent] = (),
        complete: bool = True,
        observed_at: Optional[Any] = None,
        metadata_observed: Optional[bool] = None,
    ) -> "SheetSnapshot":
        properties = data.get("properties") or {}
        spreadsheet_id = str(properties.get("id", ""))
        global_metadata = data.get("developerMetadata") or None
        formulas = formulas or {}

        if metadata_observed is None:
            metadata_observed = (
                "developerMetadata" in data or bool(global_metadata)
            )

        sheets: List[SheetObservation] = []
        for sheet in data.get("sheets") or []:
            properties = sheet.get("properties") or {}
            sheet_id = int(properties["sheetId"])
            title = str(properties.get("title", ""))
            sheet_formulas = formulas.get(sheet_id, formulas.get(title, {}))
            sheets.append(
                SheetObservation.from_api(
                    sheet,
                    formulas=sheet_formulas,
                    extra_metadata=global_metadata,
                    metadata_observed=metadata_observed,
                )
            )

        return cls(
            spreadsheet_id=spreadsheet_id,
            sheets=tuple(sheets),
            complete=complete,
            events=tuple(events),
            observed_at=(
                coerce_datetime(observed_at) if observed_at is not None else None
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spreadsheetId": self.spreadsheet_id,
            "sheets": [sheet.to_dict() for sheet in self.sheets],
            "complete": self.complete,
            "events": [event.to_dict() for event in self.events],
            "observedAt": (
                _datetime_key(self.observed_at) if self.observed_at else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SheetSnapshot":
        return cls(
            spreadsheet_id=str(data.get("spreadsheetId", "")),
            sheets=tuple(
                SheetObservation.from_dict(sheet) for sheet in data.get("sheets", [])
            ),
            complete=bool(data.get("complete", True)),
            events=tuple(
                SheetEvent.from_dict(event) for event in data.get("events", [])
            ),
            observed_at=(
                coerce_datetime(data["observedAt"])
                if data.get("observedAt")
                else None
            ),
        )


@dataclass(frozen=True)
class IdentityRevision:
    """One observation of an identity."""

    uid: str
    revision: int
    observed_at: datetime
    title: str
    index: int
    developer_metadata: Tuple[DeveloperMetadata, ...]
    formulas: Tuple[Tuple[str, str], ...]

    @property
    def fingerprint(self) -> str:
        payload = {
            "title": self.title,
            "index": self.index,
            "metadata": [m.to_dict() for m in self.developer_metadata],
            "formulas": list(self.formulas),
        }
        digest = hashlib.sha256(_canonical_json(payload).encode("utf-8"))
        return digest.hexdigest()

    @property
    def formula_map(self) -> Dict[str, str]:
        return dict(self.formulas)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uid": self.uid,
            "revision": self.revision,
            "observedAt": _datetime_key(self.observed_at),
            "title": self.title,
            "index": self.index,
            "developerMetadata": [m.to_dict() for m in self.developer_metadata],
            "formulas": [[cell, formula] for cell, formula in self.formulas],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IdentityRevision":
        return cls(
            uid=str(data["uid"]),
            revision=int(data["revision"]),
            observed_at=coerce_datetime(data["observedAt"]),
            title=str(data.get("title", "")),
            index=int(data.get("index", 0)),
            developer_metadata=tuple(
                DeveloperMetadata.from_dict(m)
                for m in data.get("developerMetadata", [])
            ),
            formulas=tuple(
                (str(cell), str(formula))
                for cell, formula in data.get("formulas", [])
            ),
        )


@dataclass
class SourceIdentity:
    """A stable worksheet identity and all of its revisions."""

    uid: str
    spreadsheet_id: str
    sheet_id: int
    published_at: datetime
    origin: str
    revisions: List[IdentityRevision] = field(default_factory=list)
    parent_uid: Optional[str] = None
    predecessor_uid: Optional[str] = None
    retired_at: Optional[datetime] = None
    retired_reason: str = ""

    @property
    def current_revision(self) -> Optional[IdentityRevision]:
        return self.revisions[-1] if self.revisions else None

    @property
    def title(self) -> Optional[str]:
        revision = self.current_revision
        return revision.title if revision is not None else None

    @property
    def active(self) -> bool:
        return self.retired_at is None

    @property
    def titles(self) -> Tuple[str, ...]:
        seen = set()
        titles: List[str] = []
        for revision in self.revisions:
            if revision.title not in seen:
                seen.add(revision.title)
                titles.append(revision.title)
        return tuple(titles)

    def revision_at(self, at: datetime) -> Optional[IdentityRevision]:
        found = None
        for revision in self.revisions:
            if revision.observed_at <= at:
                found = revision
            else:
                break
        return found

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uid": self.uid,
            "spreadsheetId": self.spreadsheet_id,
            "sheetId": self.sheet_id,
            "publishedAt": _datetime_key(self.published_at),
            "origin": self.origin,
            "revisions": [revision.to_dict() for revision in self.revisions],
            "parentUid": self.parent_uid,
            "predecessorUid": self.predecessor_uid,
            "retiredAt": (
                _datetime_key(self.retired_at) if self.retired_at else None
            ),
            "retiredReason": self.retired_reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SourceIdentity":
        return cls(
            uid=str(data["uid"]),
            spreadsheet_id=str(data.get("spreadsheetId", "")),
            sheet_id=int(data.get("sheetId", 0)),
            published_at=coerce_datetime(data["publishedAt"]),
            origin=str(data.get("origin", "observed")),
            revisions=[
                IdentityRevision.from_dict(r) for r in data.get("revisions", [])
            ],
            parent_uid=data.get("parentUid"),
            predecessor_uid=data.get("predecessorUid"),
            retired_at=(
                coerce_datetime(data["retiredAt"])
                if data.get("retiredAt")
                else None
            ),
            retired_reason=str(data.get("retiredReason", "")),
        )


@dataclass(frozen=True)
class IdentityCandidate:
    """A non-binding similarity signal.

    Candidates are diagnostics. They never change ownership of an identity.
    """

    uid: str
    title: Optional[str]
    reasons: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {"uid": self.uid, "title": self.title, "reasons": list(self.reasons)}


@dataclass(frozen=True)
class IdentityChange:
    """The reconciliation outcome for one observed or retired worksheet."""

    uid: str
    kind: str
    sheet_id: int
    title: Optional[str]
    previous_title: Optional[str] = None
    parent_uid: Optional[str] = None
    predecessor_uid: Optional[str] = None
    evidence: Tuple[str, ...] = ()
    candidates: Tuple[IdentityCandidate, ...] = ()
    revision: Optional[int] = None
    observed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uid": self.uid,
            "kind": self.kind,
            "sheetId": self.sheet_id,
            "title": self.title,
            "previousTitle": self.previous_title,
            "parentUid": self.parent_uid,
            "predecessorUid": self.predecessor_uid,
            "evidence": list(self.evidence),
            "candidates": [c.to_dict() for c in self.candidates],
            "revision": self.revision,
            "observedAt": (
                _datetime_key(self.observed_at) if self.observed_at else None
            ),
        }


@dataclass(frozen=True)
class ReferenceEdge:
    """An immutable edge from a source revision to a referenced object.

    The edge records enough context to be read years later: the source cell,
    the formula/metadata text, the target identity and revision, and the
    target title at the time of observation.
    """

    source_uid: str
    source_revision: int
    source_sheet_id: int
    observed_at: datetime
    source_kind: str
    cell: str
    raw: str
    a1_range: str
    target_title: str
    target_uid: Optional[str] = None
    target_revision: Optional[int] = None
    target_sheet_id: Optional[int] = None
    status: str = ReferenceStatus.unresolved
    metadata_key: Optional[str] = None
    candidates: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sourceUid": self.source_uid,
            "sourceRevision": self.source_revision,
            "sourceSheetId": self.source_sheet_id,
            "observedAt": _datetime_key(self.observed_at),
            "sourceKind": self.source_kind,
            "cell": self.cell,
            "raw": self.raw,
            "a1Range": self.a1_range,
            "targetTitle": self.target_title,
            "targetUid": self.target_uid,
            "targetRevision": self.target_revision,
            "targetSheetId": self.target_sheet_id,
            "status": self.status,
            "metadataKey": self.metadata_key,
            "candidates": list(self.candidates),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReferenceEdge":
        return cls(
            source_uid=str(data["sourceUid"]),
            source_revision=int(data.get("sourceRevision", 0)),
            source_sheet_id=int(data.get("sourceSheetId", 0)),
            observed_at=coerce_datetime(data["observedAt"]),
            source_kind=str(data.get("sourceKind", ReferenceSource.formula)),
            cell=str(data.get("cell", "")),
            raw=str(data.get("raw", "")),
            a1_range=str(data.get("a1Range", "")),
            target_title=str(data.get("targetTitle", "")),
            target_uid=data.get("targetUid"),
            target_revision=data.get("targetRevision"),
            target_sheet_id=data.get("targetSheetId"),
            status=str(data.get("status", ReferenceStatus.unresolved)),
            metadata_key=data.get("metadataKey"),
            candidates=tuple(data.get("candidates", ())),
        )


@dataclass(frozen=True)
class PendingDuplicate:
    """A duplicate event whose target worksheet has not been observed yet.

    The source is resolved lazily: either by its UID or by its source sheet
    ID. ``sheet_id`` is ``None`` when the caller did not know the new sheet ID
    at event time (for example a just-completed ``duplicateSheet`` request
    whose reply was discarded). In that case ``title`` disambiguates the
    target match.
    """

    source_uid: Optional[str] = None
    at: Optional[datetime] = None
    sheet_id: Optional[int] = None
    title: Optional[str] = None
    spreadsheet_id: Optional[str] = None
    source_sheet_id: Optional[int] = None

    def matches(self, sheet: "SheetObservation") -> bool:
        if self.sheet_id is not None and self.sheet_id >= 0:
            return self.sheet_id == sheet.sheet_id
        return self.title is None or self.title == sheet.title

    def for_spreadsheet(self, spreadsheet_id: str) -> bool:
        return self.spreadsheet_id in (None, spreadsheet_id)

    def resolve_source(
        self, ledger: "IdentityLedger", spreadsheet_id: str
    ) -> Optional[SourceIdentity]:
        if self.source_uid is not None:
            return ledger.identity_for_uid(self.source_uid)
        if self.source_sheet_id is not None:
            identity = ledger.active_identity_for(
                spreadsheet_id, self.source_sheet_id
            )
            if identity is None:
                identity = ledger.identity_for(
                    spreadsheet_id, self.source_sheet_id
                )
            return identity
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spreadsheetId": self.spreadsheet_id or "",
            "sourceUid": self.source_uid,
            "sourceSheetId": self.source_sheet_id,
            "at": _datetime_key(self.at) if self.at is not None else None,
            "sheetId": self.sheet_id,
            "title": self.title,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PendingDuplicate":
        return cls(
            source_uid=data.get("sourceUid"),
            at=coerce_datetime(data["at"]) if data.get("at") else None,
            sheet_id=data.get("sheetId"),
            title=data.get("title"),
            spreadsheet_id=data.get("spreadsheetId"),
            source_sheet_id=data.get("sourceSheetId"),
        )


@dataclass(frozen=True)
class Reconciliation:
    """Result of publishing one snapshot to a ledger."""

    as_of: datetime
    changes: Tuple[IdentityChange, ...]
    references: Tuple[ReferenceEdge, ...]

    @property
    def renamed(self) -> Tuple[IdentityChange, ...]:
        return tuple(c for c in self.changes if c.kind == ChangeKind.renamed)

    @property
    def copied(self) -> Tuple[IdentityChange, ...]:
        return tuple(c for c in self.changes if c.kind == ChangeKind.copied)

    @property
    def recreated(self) -> Tuple[IdentityChange, ...]:
        return tuple(c for c in self.changes if c.kind == ChangeKind.recreated)

    @property
    def deleted(self) -> Tuple[IdentityChange, ...]:
        return tuple(c for c in self.changes if c.kind == ChangeKind.deleted)

    @property
    def unresolved_references(self) -> Tuple[ReferenceEdge, ...]:
        return tuple(
            edge
            for edge in self.references
            if edge.status != ReferenceStatus.resolved
        )


@dataclass(frozen=True)
class HistoricalSource:
    """A source as it existed at a historical timestamp."""

    uid: str
    sheet_id: int
    title: str
    revision: int
    published_at: datetime
    observed_at: datetime
    status: str
    parent_uid: Optional[str] = None
    predecessor_uid: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uid": self.uid,
            "sheetId": self.sheet_id,
            "title": self.title,
            "revision": self.revision,
            "publishedAt": _datetime_key(self.published_at),
            "observedAt": _datetime_key(self.observed_at),
            "status": self.status,
            "parentUid": self.parent_uid,
            "predecessorUid": self.predecessor_uid,
        }


@dataclass(frozen=True)
class ReferenceGraph:
    """An immutable graph of sources and reference edges at one timestamp."""

    as_of: datetime
    sources: Tuple[HistoricalSource, ...]
    edges: Tuple[ReferenceEdge, ...]

    def source(self, uid: str) -> Optional[HistoricalSource]:
        for source in self.sources:
            if source.uid == uid:
                return source
        return None

    def by_title(self, title: str) -> Tuple[HistoricalSource, ...]:
        return tuple(source for source in self.sources if source.title == title)

    def outgoing(self, uid: str) -> Tuple[ReferenceEdge, ...]:
        return tuple(edge for edge in self.edges if edge.source_uid == uid)

    def incoming(self, uid: str) -> Tuple[ReferenceEdge, ...]:
        return tuple(edge for edge in self.edges if edge.target_uid == uid)

    def unresolved(self) -> Tuple[ReferenceEdge, ...]:
        return tuple(
            edge for edge in self.edges if edge.status != ReferenceStatus.resolved
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asOf": _datetime_key(self.as_of),
            "sources": [source.to_dict() for source in self.sources],
            "edges": [edge.to_dict() for edge in self.edges],
        }


def _reference_value(value: Any) -> Any:
    """Normalize a developer metadata value into a reference mapping."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except ValueError:
            return text
    if text and not any(marker in text for marker in ("!", "://", "=")):
        return {"targetUid": text}
    return value


def _metadata_reference_specs(
    metadata: DeveloperMetadata,
) -> List[Tuple[Optional[str], Optional[int], Optional[str], str]]:
    """Return explicit ``(target_uid, target_sheet_id, target_title, range)``.

    Only the reserved ``gspread.reference`` namespace is interpreted. All
    other developer metadata is business content and is never used to link
    identities.
    """
    if not (
        metadata.key == _REFERENCE_KEY
        or metadata.key.startswith(_REFERENCE_KEY + ".")
    ):
        return []

    value = _reference_value(metadata.value)
    if not isinstance(value, Mapping):
        return []

    target_uid = value.get("targetUid") or value.get("target_uid")
    target_sheet_id = value.get("targetSheetId", value.get("target_sheet_id"))
    target_title = value.get("targetTitle", value.get("target_title"))
    a1_range = value.get("range", value.get("a1Range", value.get("cell", "")))
    if target_sheet_id is not None:
        try:
            target_sheet_id = int(target_sheet_id)
        except (TypeError, ValueError):
            target_sheet_id = None
    return [
        (
            str(target_uid) if target_uid is not None else None,
            target_sheet_id,
            str(target_title) if target_title is not None else None,
            str(a1_range or ""),
        )
    ]


class IdentityLedger:
    """Append-only ledger of worksheet identities and reference edges.

    The ledger is independent from the Sheets API: feed it snapshots using
    :meth:`observe` and it returns a reconciliation report. The ledger can be
    serialized with :meth:`to_dict` and restored with :meth:`from_dict`.
    """

    VERSION = 1

    def __init__(self) -> None:
        self._identities: Dict[str, SourceIdentity] = {}
        self._by_sheet: Dict[Tuple[str, int], List[str]] = {}
        self._edges: List[ReferenceEdge] = []
        self._last_observed: Optional[datetime] = None
        self._pending_duplicates: List[PendingDuplicate] = []
        self._events: List[Tuple[str, SheetEvent]] = []

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------
    @staticmethod
    def base_uid(spreadsheet_id: str, sheet_id: int, generation: int = 1) -> str:
        """Return the stable UID for a worksheet.

        The base UID depends only on the spreadsheet and the worksheet
        ``sheetId``; it never contains a title. If an explicit sheet ID is
        deleted and later reused, the ledger appends ``#<generation>``.
        """
        base = "worksheet:{}:{}".format(spreadsheet_id, int(sheet_id))
        return base if generation <= 1 else "{}#{}".format(base, generation)

    @property
    def identities(self) -> Tuple[SourceIdentity, ...]:
        return tuple(
            sorted(self._identities.values(), key=lambda item: item.uid)
        )

    @property
    def edges(self) -> Tuple[ReferenceEdge, ...]:
        return tuple(self._edges)

    @property
    def last_observed(self) -> Optional[datetime]:
        return self._last_observed

    def identity(self, uid: str) -> SourceIdentity:
        try:
            return self._identities[uid]
        except KeyError:
            raise IdentityError("unknown identity: {}".format(uid)) from None

    def identity_for(
        self, spreadsheet_id: str, sheet_id: int
    ) -> Optional[SourceIdentity]:
        """Return the most recent identity using ``sheet_id``."""
        uids = self._by_sheet.get((spreadsheet_id, int(sheet_id)))
        if not uids:
            return None
        return self._identities[uids[-1]]

    def identity_for_uid(self, uid: str) -> Optional[SourceIdentity]:
        """Return an identity by UID, or ``None``."""
        return self._identities.get(uid)

    def _identity_at_sheet_id(
        self, spreadsheet_id: str, sheet_id: int, at: datetime
    ) -> Optional[SourceIdentity]:
        """Return the identity that owned ``sheet_id`` at ``at``.

        A sheet ID can be reused after deletion, so the generation that was
        live at the requested time must be selected rather than the newest.
        """
        candidates: List[SourceIdentity] = []
        for uid in self._by_sheet.get((spreadsheet_id, int(sheet_id)), ()):
            identity = self._identities[uid]
            if identity.revision_at(at) is None:
                continue
            if identity.retired_at is not None and identity.retired_at <= at:
                continue
            candidates.append(identity)
        return candidates[-1] if candidates else None

    def active_identity_for(
        self, spreadsheet_id: str, sheet_id: int
    ) -> Optional[SourceIdentity]:
        identity = self.identity_for(spreadsheet_id, sheet_id)
        if identity is not None and identity.active:
            return identity
        return None

    def _active_for_spreadsheet(self, spreadsheet_id: str) -> List[SourceIdentity]:
        return [
            identity
            for identity in self._identities.values()
            if identity.spreadsheet_id == spreadsheet_id and identity.active
        ]

    def _title_owners(
        self, spreadsheet_id: str, title: str
    ) -> List[SourceIdentity]:
        return [
            identity
            for identity in self._active_for_spreadsheet(spreadsheet_id)
            if identity.title == title
        ]

    @staticmethod
    def _similarity_reasons(
        identity: SourceIdentity,
        title: str,
        metadata_pairs: Sequence[Tuple[str, str]],
    ) -> List[str]:
        reasons: List[str] = []
        if title and title in identity.titles:
            reasons.append(
                "title" if identity.title == title else "historical_title"
            )

        for revision in identity.revisions:
            for entry in revision.developer_metadata:
                if (
                    entry.key
                    and (entry.key, _canonical_json(entry.value))
                    in metadata_pairs
                ):
                    reason = "metadata:{}".format(entry.key)
                    if reason not in reasons:
                        reasons.append(reason)
        return reasons

    def _candidates(
        self,
        spreadsheet_id: str,
        title: str,
        metadata: Sequence[DeveloperMetadata],
        exclude_uid: Optional[str] = None,
    ) -> Tuple[IdentityCandidate, ...]:
        metadata_pairs = [
            (entry.key, _canonical_json(entry.value))
            for entry in metadata
            if entry.key
        ]
        candidates: List[IdentityCandidate] = []
        for identity in self.identities:
            if identity.spreadsheet_id != spreadsheet_id:
                continue
            if identity.uid == exclude_uid:
                continue

            reasons = self._similarity_reasons(identity, title, metadata_pairs)
            if reasons:
                candidates.append(
                    IdentityCandidate(
                        uid=identity.uid,
                        title=identity.title,
                        reasons=tuple(sorted(reasons)),
                    )
                )

        candidates.sort(key=lambda item: (-len(item.reasons), item.uid))
        return tuple(candidates)

    # ------------------------------------------------------------------
    # Publication
    # ------------------------------------------------------------------
    def publish(
        self, snapshot: SheetSnapshot, at: Optional[Any] = None
    ) -> Reconciliation:
        """Publish a snapshot. Alias of :meth:`observe`."""
        return self.observe(snapshot, at=at)

    def observe(
        self, snapshot: SheetSnapshot, at: Optional[Any] = None
    ) -> Reconciliation:
        """Reconcile ``snapshot`` with the ledger at time ``at``.

        ``at`` defaults to ``snapshot.observed_at`` and then to the current
        UTC time. Time must not move backwards; a
        :class:`~gspread.exceptions.IdentityTimelineError` is raised otherwise.
        """
        observed_at = coerce_datetime(
            at if at is not None else snapshot.observed_at
        )
        self._check_time(observed_at)

        changes: List[IdentityChange] = []
        spreadsheet_id = snapshot.spreadsheet_id
        for event in sorted(
            snapshot.events,
            key=lambda item: _datetime_key(item.at or observed_at),
        ):
            deleted_change = self._apply_event(
                spreadsheet_id, event, observed_at
            )
            if deleted_change is not None:
                changes.append(deleted_change)

        observed_ids = {sheet.sheet_id for sheet in snapshot.sheets}

        if snapshot.complete:
            for identity in list(self._active_for_spreadsheet(spreadsheet_id)):
                if identity.sheet_id not in observed_ids:
                    self._retire(
                        identity,
                        observed_at,
                        reason=Evidence.absent_from_complete_snapshot,
                    )
                    changes.append(
                        IdentityChange(
                            uid=identity.uid,
                            kind=ChangeKind.deleted,
                            sheet_id=identity.sheet_id,
                            title=identity.title,
                            previous_title=identity.title,
                            evidence=(Evidence.absent_from_complete_snapshot,),
                            observed_at=observed_at,
                        )
                    )

        current: Dict[int, SourceIdentity] = {}
        for sheet in snapshot.sheets:
            identity, change = self._observe_sheet(
                spreadsheet_id, sheet, observed_at
            )
            current[sheet.sheet_id] = identity
            changes.append(change)

        references = self._observe_references(
            snapshot.sheets, current, observed_at
        )

        # An explicit event may be timestamped later than the snapshot's own
        # observation time. Never move the ledger clock backwards afterwards.
        if self._last_observed is None or observed_at > self._last_observed:
            self._last_observed = observed_at
        return Reconciliation(
            as_of=observed_at,
            changes=tuple(changes),
            references=tuple(references),
        )

    def record_event(self, event: SheetEvent, spreadsheet_id: str) -> None:
        """Record a structural event outside a snapshot.

        This is used by :meth:`Spreadsheet.del_worksheet` and
        :meth:`Worksheet.duplicate` so deletion and duplication evidence is
        not lost between snapshots. A duplicate event is stored until the new
        sheet is observed.

        Events are stamped at the ledger's latest known observation time (or
        at the current time when the ledger is empty). A caller-supplied time
        is clamped forward so a locally recorded event can never make a later
        snapshot fail with :class:`~gspread.exceptions.IdentityTimelineError`.
        """
        moment = coerce_datetime(
            event.at if event.at is not None else (self._last_observed or _utc_now())
        )
        if self._last_observed is not None and moment < self._last_observed:
            moment = self._last_observed

        normalized = SheetEvent(
            kind=event.kind,
            sheet_id=event.sheet_id,
            source_sheet_id=event.source_sheet_id,
            at=moment,
            reason=event.reason,
            source_uid=event.source_uid,
            title=event.title,
        )
        self._events.append((spreadsheet_id, normalized))
        self._apply_event(spreadsheet_id, normalized, moment)

    def record_rename(
        self,
        spreadsheet_id: str,
        sheet_id: int,
        title: str,
        at: Optional[Any] = None,
    ) -> Optional[IdentityRevision]:
        """Record a rename performed through gspread.

        A new revision is appended with the same sheet ID and the new title.
        Developer metadata and formulas are carried over from the previous
        revision because they were not re-collected. Returns ``None`` when the
        ledger has never observed that worksheet.
        """
        identity = self.active_identity_for(spreadsheet_id, int(sheet_id))
        if identity is None:
            return None

        # Use the latest known point in the timeline instead of wall-clock
        # time: a later, explicitly timestamped snapshot must not be rejected
        # because a local call happened to run "now".
        moment = coerce_datetime(
            at if at is not None else (self._last_observed or _utc_now())
        )
        if self._last_observed is not None and moment < self._last_observed:
            moment = self._last_observed

        previous = identity.current_revision
        revision = self._append_revision(
            identity,
            SheetObservation(
                sheet_id=int(sheet_id),
                title=str(title),
                index=previous.index if previous is not None else 0,
            ),
            moment,
        )
        if (
            self._last_observed is None
            or revision.observed_at > self._last_observed
        ):
            self._last_observed = revision.observed_at
        return revision

    def pending_duplicates(self) -> Tuple[PendingDuplicate, ...]:
        """Return duplicate events whose target has not been observed yet."""
        return tuple(self._pending_duplicates)

    def _check_time(self, at: datetime) -> None:
        if self._last_observed is not None and at < self._last_observed:
            raise IdentityTimelineError(
                "cannot publish at {} before last observation {}".format(
                    _datetime_key(at), _datetime_key(self._last_observed)
                )
            )

    def _apply_event(
        self, spreadsheet_id: str, event: SheetEvent, default_at: datetime
    ) -> Optional[IdentityChange]:
        event_time = coerce_datetime(event.at, default=default_at)
        if event_time < default_at:
            self._check_time(event_time)

        if event.kind == "duplicate":
            if event.source_sheet_id is None and event.source_uid is None:
                raise IdentityError(
                    "duplicate event requires source_sheet_id or source_uid"
                )
            self._pending_duplicates.append(
                PendingDuplicate(
                    source_uid=event.source_uid,
                    source_sheet_id=event.source_sheet_id,
                    at=event_time,
                    sheet_id=event.sheet_id if event.sheet_id >= 0 else None,
                    title=event.title,
                    spreadsheet_id=spreadsheet_id,
                )
            )
            return None

        if event.kind == "delete":
            identity = self.active_identity_for(spreadsheet_id, event.sheet_id)
            if identity is None:
                return None
            previous_title = identity.title
            self._retire(
                identity,
                event_time,
                reason=event.reason or Evidence.delete_event,
            )
            return IdentityChange(
                uid=identity.uid,
                kind=ChangeKind.deleted,
                sheet_id=identity.sheet_id,
                title=previous_title,
                previous_title=previous_title,
                evidence=(event.reason or Evidence.delete_event,),
                observed_at=event_time,
            )

        raise IdentityError("unknown sheet event: {}".format(event.kind))

    def _retire(self, identity: SourceIdentity, at: datetime, reason: str) -> None:
        if identity.retired_at is None:
            identity.retired_at = at
            identity.retired_reason = str(reason)
            if self._last_observed is None or at > self._last_observed:
                self._last_observed = at

    def _next_uid(
        self, spreadsheet_id: str, sheet_id: int
    ) -> Tuple[str, int]:
        key = (spreadsheet_id, int(sheet_id))
        generation = len(self._by_sheet.get(key, ())) + 1
        return self.base_uid(spreadsheet_id, sheet_id, generation), generation

    def _take_duplicate(
        self, spreadsheet_id: str, sheet: SheetObservation
    ) -> Tuple[Optional[PendingDuplicate], Optional[SourceIdentity]]:
        """Pop the first pending duplicate that matches ``sheet``."""
        for index, pending in enumerate(self._pending_duplicates):
            if not pending.for_spreadsheet(spreadsheet_id):
                continue
            if not pending.matches(sheet):
                continue
            source = pending.resolve_source(self, spreadsheet_id)
            if source is None:
                continue
            self._pending_duplicates.pop(index)
            return pending, source
        return None, None

    def _recreation_predecessor(
        self, spreadsheet_id: str, sheet: SheetObservation
    ) -> Optional[SourceIdentity]:
        prior = self.identity_for(spreadsheet_id, sheet.sheet_id)
        if prior is not None and not prior.active:
            return prior

        # A different sheet ID may only inherit lineage when the old object
        # was explicitly deleted and the title is not currently held by
        # another live worksheet. Title reuse alone is never enough:
        # otherwise a copy (or an unrelated new sheet) could claim the
        # retired identity.
        if self._title_owners(spreadsheet_id, sheet.title):
            return None
        deleted = [
            identity
            for identity in self._identities.values()
            if identity.spreadsheet_id == spreadsheet_id
            and not identity.active
            and sheet.title in identity.titles
            and identity.retired_reason
            in (Evidence.delete_event, Evidence.absent_from_complete_snapshot)
        ]
        if not deleted:
            return None
        return sorted(
            deleted, key=lambda item: item.retired_at or item.published_at
        )[-1]

    def _observe_sheet(
        self,
        spreadsheet_id: str,
        sheet: SheetObservation,
        at: datetime,
    ) -> Tuple[SourceIdentity, IdentityChange]:
        existing = self.active_identity_for(spreadsheet_id, sheet.sheet_id)

        if existing is not None:
            previous_title = existing.title
            kind = (
                ChangeKind.unchanged
                if previous_title == sheet.title
                else ChangeKind.renamed
            )
            revision = self._append_revision(existing, sheet, at)
            evidence: Tuple[Any, ...] = (Evidence.same_sheet_id,)
            if kind == ChangeKind.renamed:
                evidence = evidence + (Evidence.title_changed,)
            return existing, IdentityChange(
                uid=existing.uid,
                kind=kind,
                sheet_id=sheet.sheet_id,
                title=sheet.title,
                previous_title=previous_title,
                evidence=evidence,
                revision=revision.revision,
                observed_at=at,
            )

        duplicate, duplicate_source = self._take_duplicate(
            spreadsheet_id, sheet
        )
        parent_uid: Optional[str] = None
        predecessor: Optional[SourceIdentity] = None
        origin = "observed"
        kind = ChangeKind.new
        evidence = (Evidence.new_sheet_id,)

        if duplicate is not None and duplicate_source is not None:
            parent_uid = duplicate_source.uid
            origin = ChangeKind.copied
            kind = ChangeKind.copied
            evidence = (Evidence.duplicate_event, Evidence.new_sheet_id)
        else:
            predecessor = self._recreation_predecessor(spreadsheet_id, sheet)
            if predecessor is not None:
                origin = ChangeKind.recreated
                kind = ChangeKind.recreated
                if self.identity_for(
                    spreadsheet_id, sheet.sheet_id
                ) is predecessor:
                    evidence = (
                        Evidence.sheet_id_reused_after_delete,
                        Evidence.delete_event,
                    )
                else:
                    evidence = (
                        Evidence.title_reused_after_delete,
                        Evidence.delete_event,
                    )

        uid, _generation = self._next_uid(spreadsheet_id, sheet.sheet_id)
        identity = SourceIdentity(
            uid=uid,
            spreadsheet_id=spreadsheet_id,
            sheet_id=sheet.sheet_id,
            published_at=at,
            origin=origin,
            parent_uid=parent_uid,
            predecessor_uid=predecessor.uid if predecessor is not None else None,
        )
        revision = self._append_revision(identity, sheet, at)
        self._identities[uid] = identity
        self._by_sheet.setdefault(
            (spreadsheet_id, sheet.sheet_id), []
        ).append(uid)

        previous_revision = (
            predecessor.current_revision if predecessor is not None else None
        )
        metadata = (
            revision.developer_metadata
            if revision.developer_metadata is not None
            else (previous_revision.developer_metadata if previous_revision else ())
        )
        candidates = self._candidates(
            spreadsheet_id,
            sheet.title,
            metadata,
            exclude_uid=uid,
        )
        return identity, IdentityChange(
            uid=uid,
            kind=kind,
            sheet_id=sheet.sheet_id,
            title=sheet.title,
            previous_title=None,
            parent_uid=parent_uid,
            predecessor_uid=(
                predecessor.uid if predecessor is not None else None
            ),
            evidence=evidence,
            candidates=candidates,
            revision=revision.revision,
            observed_at=at,
        )

    def _append_revision(
        self,
        identity: SourceIdentity,
        sheet: SheetObservation,
        at: datetime,
    ) -> IdentityRevision:
        previous = identity.current_revision
        if sheet.developer_metadata is not None:
            metadata = tuple(sheet.developer_metadata)
        elif previous is not None:
            metadata = previous.developer_metadata
        else:
            metadata = ()

        if sheet.formulas is not None:
            formulas = tuple(sorted(sheet.formulas.items()))
        elif previous is not None:
            formulas = previous.formulas
        else:
            formulas = ()

        revision = IdentityRevision(
            uid=identity.uid,
            revision=len(identity.revisions) + 1,
            observed_at=at,
            title=sheet.title,
            index=sheet.index,
            developer_metadata=metadata,
            formulas=formulas,
        )
        identity.revisions.append(revision)
        return revision

    # ------------------------------------------------------------------
    # References
    # ------------------------------------------------------------------
    def _resolve_reference(
        self,
        spreadsheet_id: str,
        target_uid: Optional[str],
        target_sheet_id: Optional[int],
        target_title: Optional[str],
        at: datetime,
    ) -> Tuple[Optional[SourceIdentity], str, Tuple[str, ...], Tuple[str, ...]]:
        """Resolve an explicit target specification.

        Returns ``(identity, status, evidence, candidate_uids)``.
        """
        if target_uid:
            identity = self._identities.get(str(target_uid))
            if identity is not None and identity.revision_at(at) is not None:
                return (
                    identity,
                    ReferenceStatus.resolved,
                    (Evidence.explicit_target_uid,),
                    (),
                )
            return (
                None,
                ReferenceStatus.unresolved,
                (Evidence.explicit_target_uid,),
                (),
            )

        if target_sheet_id is not None:
            identity = self._identity_at_sheet_id(
                spreadsheet_id, int(target_sheet_id), at
            )
            if identity is not None:
                return (
                    identity,
                    ReferenceStatus.resolved,
                    (Evidence.explicit_target_sheet_id,),
                    (),
                )
            sheet_generations = [
                self._identities[uid]
                for uid in self._by_sheet.get(
                    (spreadsheet_id, int(target_sheet_id)), ()
                )
            ]
            if sheet_generations:
                return (
                    None,
                    ReferenceStatus.ambiguous,
                    (Evidence.sheet_id_reused_after_delete,),
                    tuple(item.uid for item in sheet_generations),
                )
            return (
                None,
                ReferenceStatus.unresolved,
                (Evidence.explicit_target_sheet_id,),
                (),
            )

        if target_title:
            title = str(target_title)
            owners = self._title_owners(spreadsheet_id, title)
            owner_uids = {owner.uid for owner in owners}
            previous_owners = [
                identity.uid
                for identity in self.identities
                if identity.spreadsheet_id == spreadsheet_id
                and title in identity.titles
                and identity.uid not in owner_uids
            ]

            if len(owners) == 1 and not previous_owners:
                return (
                    owners[0],
                    ReferenceStatus.resolved,
                    (Evidence.title_only,),
                    (),
                )

            if len(owners) > 1:
                return (
                    None,
                    ReferenceStatus.ambiguous,
                    (Evidence.title_ambiguous,),
                    tuple(sorted(owner_uids)),
                )

            if previous_owners:
                # The title changed hands. A reference that only carries the
                # title cannot prove which object it meant, so it is kept as
                # ambiguous instead of silently following the new owner.
                return (
                    None,
                    ReferenceStatus.ambiguous,
                    (Evidence.historical_title,),
                    tuple(sorted(set(owner_uids) | set(previous_owners))),
                )

            return (
                None,
                ReferenceStatus.unresolved,
                (Evidence.title_unowned,),
                (),
            )

        return (None, ReferenceStatus.unresolved, (), ())

    def _make_edge(
        self,
        source: SourceIdentity,
        revision: IdentityRevision,
        at: datetime,
        source_kind: str,
        cell: str,
        raw: str,
        a1_range: str,
        target_title: str,
        target_uid: Optional[str] = None,
        target_sheet_id: Optional[int] = None,
        metadata_key: Optional[str] = None,
    ) -> ReferenceEdge:
        resolved, status, evidence, candidates = self._resolve_reference(
            source.spreadsheet_id,
            target_uid,
            target_sheet_id,
            target_title,
            at,
        )
        target_revision = None
        if resolved is not None:
            target_revision_obj = resolved.revision_at(at)
            target_revision = (
                target_revision_obj.revision
                if target_revision_obj is not None
                else None
            )

        return ReferenceEdge(
            source_uid=source.uid,
            source_revision=revision.revision,
            source_sheet_id=source.sheet_id,
            observed_at=at,
            source_kind=source_kind,
            cell=cell,
            raw=raw,
            a1_range=a1_range,
            target_title=target_title,
            target_uid=resolved.uid if resolved is not None else None,
            target_revision=target_revision,
            target_sheet_id=(
                resolved.sheet_id if resolved is not None else target_sheet_id
            ),
            status=status,
            metadata_key=metadata_key,
            candidates=candidates,
        )

    def _formula_edges(
        self,
        source: SourceIdentity,
        revision: IdentityRevision,
        at: datetime,
    ) -> List[ReferenceEdge]:
        edges: List[ReferenceEdge] = []
        for cell, formula in revision.formulas:
            for reference in extract_formula_references(formula):
                edges.append(
                    self._make_edge(
                        source=source,
                        revision=revision,
                        at=at,
                        source_kind=ReferenceSource.formula,
                        cell=str(cell),
                        raw=reference.raw,
                        a1_range=reference.a1_range,
                        target_title=reference.sheet_title,
                    )
                )
        return edges

    def _metadata_edges(
        self,
        source: SourceIdentity,
        revision: IdentityRevision,
        at: datetime,
    ) -> List[ReferenceEdge]:
        edges: List[ReferenceEdge] = []
        for metadata in revision.developer_metadata:
            for (
                target_uid,
                target_sheet_id,
                target_title,
                a1_range,
            ) in _metadata_reference_specs(metadata):
                edges.append(
                    self._make_edge(
                        source=source,
                        revision=revision,
                        at=at,
                        source_kind=ReferenceSource.developer_metadata,
                        cell="",
                        raw=str(metadata.value),
                        a1_range=a1_range,
                        target_title=target_title or "",
                        target_uid=target_uid,
                        target_sheet_id=target_sheet_id,
                        metadata_key=metadata.key,
                    )
                )
        return edges

    def _observe_references(
        self,
        sheets: Sequence[SheetObservation],
        current: Mapping[int, SourceIdentity],
        at: datetime,
    ) -> List[ReferenceEdge]:
        edges: List[ReferenceEdge] = []
        seen = set()

        for sheet in sheets:
            source = current.get(sheet.sheet_id)
            if source is None:
                continue
            revision = source.current_revision
            if revision is None:
                continue

            for edge in self._formula_edges(
                source, revision, at
            ) + self._metadata_edges(source, revision, at):
                key = (
                    edge.source_uid,
                    edge.source_revision,
                    edge.source_kind,
                    edge.cell,
                    edge.raw,
                    edge.target_uid,
                    edge.target_title,
                    edge.status,
                )
                if key not in seen:
                    seen.add(key)
                    edges.append(edge)

        self._edges.extend(edges)
        return edges

    # ------------------------------------------------------------------
    # Historical queries
    # ------------------------------------------------------------------
    def snapshot(self, at: Optional[Any] = None) -> ReferenceGraph:
        """Return titles, identities and references as of ``at``.

        The result never uses current titles for historical revisions. With
        ``at=None`` the latest known state is used.
        """
        as_of = (
            coerce_datetime(at)
            if at is not None
            else (self._last_observed or _utc_now())
        )

        sources: List[HistoricalSource] = []
        latest_revisions: Dict[str, IdentityRevision] = {}
        for identity in self.identities:
            revision = identity.revision_at(as_of)
            if revision is None:
                continue
            retired = identity.retired_at is not None and identity.retired_at <= as_of
            if retired:
                continue
            latest_revisions[identity.uid] = revision
            sources.append(
                HistoricalSource(
                    uid=identity.uid,
                    sheet_id=identity.sheet_id,
                    title=revision.title,
                    revision=revision.revision,
                    published_at=identity.published_at,
                    observed_at=revision.observed_at,
                    status="active",
                    parent_uid=identity.parent_uid,
                    predecessor_uid=identity.predecessor_uid,
                )
            )

        edges = tuple(
            edge
            for edge in self._edges
            if edge.source_uid in latest_revisions
            and latest_revisions[edge.source_uid].revision
            == edge.source_revision
            and edge.observed_at <= as_of
        )
        return ReferenceGraph(
            as_of=as_of,
            sources=tuple(sorted(sources, key=lambda item: (item.sheet_id, item.uid))),
            edges=edges,
        )

    def reference_graph(self, at: Optional[Any] = None) -> ReferenceGraph:
        """Alias of :meth:`snapshot`."""
        return self.snapshot(at=at)

    def history(self, uid: str) -> Tuple[IdentityRevision, ...]:
        """Return every revision of ``uid`` in publication order."""
        return tuple(self.identity(uid).revisions)

    def references_for(
        self,
        uid: str,
        at: Optional[Any] = None,
        current_only: bool = False,
    ) -> Tuple[ReferenceEdge, ...]:
        """Return reference edges emitted by ``uid``.

        With ``current_only=True`` only edges from the latest revision (at
        ``at``) are returned. Otherwise the complete immutable history is
        returned.
        """
        if current_only:
            as_of = (
                coerce_datetime(at)
                if at is not None
                else (self._last_observed or _utc_now())
            )
            identity = self.identity(uid)
            revision = identity.revision_at(as_of)
            if revision is None:
                return ()
            return tuple(
                edge
                for edge in self._edges
                if edge.source_uid == uid
                and edge.source_revision == revision.revision
                and edge.observed_at <= as_of
            )
        return tuple(edge for edge in self._edges if edge.source_uid == uid)

    def find_titled(
        self, spreadsheet_id: str, title: str
    ) -> Tuple[SourceIdentity, ...]:
        """Return identities that ever used ``title`` in that spreadsheet."""
        return tuple(
            identity
            for identity in self.identities
            if identity.spreadsheet_id == spreadsheet_id and title in identity.titles
        )

    def titles_at(
        self, spreadsheet_id: str, title: str, at: Optional[Any] = None
    ) -> Tuple[HistoricalSource, ...]:
        """Return the sources that held ``title`` at ``at``.

        An old title that has been renamed away returns no source, even if
        some other worksheet uses that title now.
        """
        graph = self.snapshot(at=at)
        return tuple(
            source
            for source in graph.sources
            if source.title == title
            and self.identity(source.uid).spreadsheet_id == spreadsheet_id
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "lastObserved": (
                _datetime_key(self._last_observed)
                if self._last_observed is not None
                else None
            ),
            "identities": [
                identity.to_dict()
                for identity in sorted(
                    self._identities.values(), key=lambda item: item.uid
                )
            ],
            "edges": [edge.to_dict() for edge in self._edges],
            "pendingDuplicates": [
                pending.to_dict() for pending in self._pending_duplicates
            ],
            "events": [
                {"spreadsheetId": spreadsheet_id, "event": event.to_dict()}
                for spreadsheet_id, event in self._events
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IdentityLedger":
        ledger = cls()
        for raw in data.get("identities", []):
            identity = SourceIdentity.from_dict(raw)
            ledger._identities[identity.uid] = identity
            ledger._by_sheet.setdefault(
                (identity.spreadsheet_id, identity.sheet_id), []
            ).append(identity.uid)
            for revision in identity.revisions:
                if (
                    ledger._last_observed is None
                    or revision.observed_at > ledger._last_observed
                ):
                    ledger._last_observed = revision.observed_at

        for key in ledger._by_sheet:
            ledger._by_sheet[key].sort(
                key=lambda uid: ledger._identities[uid].published_at
            )

        ledger._edges = [
            ReferenceEdge.from_dict(raw) for raw in data.get("edges", [])
        ]
        if data.get("lastObserved"):
            last_observed = coerce_datetime(data["lastObserved"])
            if (
                ledger._last_observed is None
                or last_observed > ledger._last_observed
            ):
                ledger._last_observed = last_observed

        for raw in data.get("pendingDuplicates", []):
            ledger._pending_duplicates.append(
                PendingDuplicate.from_dict(raw)
            )

        ledger._events = [
            (
                str(raw["spreadsheetId"]),
                SheetEvent.from_dict(raw["event"]),
            )
            for raw in data.get("events", [])
        ]
        return ledger

    def dumps(self, **kwargs: Any) -> str:
        """Serialize the ledger to JSON."""
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def loads(cls, payload: str) -> "IdentityLedger":
        """Restore a ledger from JSON produced by :meth:`dumps`."""
        return cls.from_dict(json.loads(payload))


def formulas_from_values(
    rows: Sequence[Sequence[Any]],
    start_row: int = 1,
    start_col: int = 1,
) -> Dict[str, str]:
    """Build ``{cell: formula}`` from a values matrix."""
    return dict(find_formula_cell(rows, start_row=start_row, start_col=start_col))


def references_in_formulas(
    formulas: Mapping[str, str],
) -> List[FormulaReference]:
    """Extract formula references from a ``{cell: formula}`` mapping."""
    result: List[FormulaReference] = []
    for formula in formulas.values():
        result.extend(extract_formula_references(formula))
    return result


def titles_in_formulas(formulas: Mapping[str, str]) -> List[str]:
    """Return distinct referenced sheet titles in first-appearance order."""
    result: List[str] = []
    seen = set()
    for formula in formulas.values():
        for title in extract_sheet_titles(formula):
            if title not in seen:
                seen.add(title)
                result.append(title)
    return result
