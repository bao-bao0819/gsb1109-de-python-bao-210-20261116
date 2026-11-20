# -*- coding: utf-8 -*-
"""On-disk layout, content addressed storage, and manifest handling."""
import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import rdflib

import pyshacl
from .recorder import DocumentRecord, LoadEvent, WebFetchRecord, normalize_url

ARCHIVE_VERSION = 1
MANIFEST_NAME = "manifest.json"
BLOBS_DIR = "blobs"
REPORTS_DIR = "reports"


@dataclass
class InputSpec:
    role: str
    base_iri: str
    document_id: str
    rdf_format: Optional[str]
    requested_url: str


@dataclass
class DocumentManifest:
    document_id: str
    role_occurrences: List[List[str]]
    base_iri: str
    requested_url: str
    sha256: str
    byte_length: int
    rdf_format: Optional[str]
    media_type: Optional[str]
    blob_path: str
    baked_in: bool
    relative_term_count: int
    relative_term_expansions: List[List[str]]
    relative_imports: List[str]
    resolved_imports: List[str]
    absolute_imports: List[str]
    loaded_via: List[str]


@dataclass
class EventManifest:
    sequence: int
    role: str
    depth: int
    requested_url: str
    effective_base: str
    sha256: str
    byte_length: int
    rdf_format: Optional[str]
    media_type: Optional[str]
    import_chain: List[str]
    baked_in: bool


@dataclass
class WebFetchManifest:
    sequence: int
    requested_url: str
    final_url: str
    sha256: str
    byte_length: int
    media_type: Optional[str]
    baked_in: bool


@dataclass
class ValidationOptions:
    inference: str
    do_owl_imports: bool
    advanced: bool
    allow_infos: bool
    allow_warnings: bool


@dataclass
class Manifest:
    archive_version: int
    created_at: str
    pyshacl_version: str
    rdflib_version: str
    python_version: str
    platform: str
    validation: ValidationOptions
    inputs: List[InputSpec]
    documents: List[DocumentManifest]
    load_events: List[EventManifest]
    web_fetches: List[WebFetchManifest]
    expected_report: Dict[str, Any]
    report_text_path: str
    report_ttl_path: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ArchiveStore:
    """Writes and reads a self-contained, content-addressed archive."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.blobs_dir = self.root / BLOBS_DIR
        self.reports_dir = self.root / REPORTS_DIR

    def ensure_dirs(self) -> None:
        self.blobs_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def blob_relpath(self, digest: str) -> Path:
        return Path(BLOBS_DIR) / digest[:2] / digest[2:]

    def write_blob(self, data: bytes, digest: str) -> str:
        relpath = self.blob_relpath(digest)
        target = self.root / relpath
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        return str(relpath)

    def read_blob(self, relpath: str) -> bytes:
        return (self.root / relpath).read_bytes()

    def write_text(self, relpath: str, text: str) -> str:
        target = self.root / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return relpath

    def read_text(self, relpath: str) -> str:
        return (self.root / relpath).read_text(encoding="utf-8")

    def write_manifest(self, manifest: Manifest) -> Path:
        path = self.root / MANIFEST_NAME
        path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=False), encoding="utf-8")
        return path

    def read_manifest(self) -> Manifest:
        data = json.loads((self.root / MANIFEST_NAME).read_text(encoding="utf-8"))
        if int(data.get("archive_version", 0)) != ARCHIVE_VERSION:
            raise ValueError(
                f"Unsupported archive version {data.get('archive_version')!r}, expected {ARCHIVE_VERSION}."
            )
        data["validation"] = ValidationOptions(**data["validation"])
        data["inputs"] = [InputSpec(**item) for item in data["inputs"]]
        data["documents"] = [DocumentManifest(**item) for item in data["documents"]]
        data["load_events"] = [EventManifest(**item) for item in data["load_events"]]
        data["web_fetches"] = [WebFetchManifest(**item) for item in data.get("web_fetches", [])]
        return Manifest(**data)

    def has_manifest(self) -> bool:
        return (self.root / MANIFEST_NAME).exists()


def build_manifest(
    validation: ValidationOptions,
    inputs: List[InputSpec],
    documents: List[DocumentRecord],
    events: List[LoadEvent],
    web_fetches: List[WebFetchRecord],
    store: ArchiveStore,
    expected_report: Dict[str, Any],
    report_text_path: str,
    report_ttl_path: str,
) -> Manifest:
    doc_manifests: List[DocumentManifest] = []
    blob_by_hash: Dict[str, str] = {}
    for index, doc in enumerate(documents):
        document_id = f"doc-{index + 1:04d}"
        if doc.sha256 not in blob_by_hash:
            blob_by_hash[doc.sha256] = store.write_blob(doc.raw_bytes, doc.sha256)
        role_occurrences = [[doc.role, str(doc.occurrence)]]
        doc_manifests.append(
            DocumentManifest(
                document_id=document_id,
                role_occurrences=role_occurrences,
                base_iri=doc.effective_base,
                requested_url=doc.requested_url,
                sha256=doc.sha256,
                byte_length=doc.byte_length,
                rdf_format=doc.rdf_format,
                media_type=doc.media_type,
                blob_path=blob_by_hash[doc.sha256],
                baked_in=doc.baked_in,
                relative_term_count=doc.relative_term_count,
                relative_term_expansions=[list(item) for item in doc.relative_term_expansions],
                relative_imports=list(doc.relative_imports),
                resolved_imports=list(doc.resolved_imports),
                absolute_imports=list(doc.absolute_imports),
                loaded_via=list(doc.loaded_via),
            )
        )
    merged = _merge_identical_base_documents(doc_manifests, documents)
    event_manifests = [
        EventManifest(
            sequence=event.sequence,
            role=event.role,
            depth=event.depth,
            requested_url=event.requested_url,
            effective_base=event.effective_base,
            sha256=event.sha256,
            byte_length=event.byte_length,
            rdf_format=event.rdf_format,
            media_type=event.media_type,
            import_chain=list(event.import_chain),
            baked_in=event.baked_in,
        )
        for event in sorted(events, key=lambda item: item.sequence)
    ]
    web_manifests = [
        WebFetchManifest(
            sequence=fetch.sequence,
            requested_url=fetch.requested_url,
            final_url=fetch.final_url,
            sha256=fetch.sha256,
            byte_length=fetch.byte_length,
            media_type=fetch.media_type,
            baked_in=fetch.baked_in,
        )
        for fetch in sorted(web_fetches, key=lambda item: item.sequence)
    ]
    return Manifest(
        archive_version=ARCHIVE_VERSION,
        created_at=utc_now_iso(),
        pyshacl_version=getattr(pyshacl, "__version__", "unknown"),
        rdflib_version=getattr(rdflib, "__version__", "unknown"),
        python_version=platform.python_version(),
        platform=platform.platform(),
        validation=validation,
        inputs=inputs,
        documents=merged,
        load_events=event_manifests,
        web_fetches=web_manifests,
        expected_report=expected_report,
        report_text_path=report_text_path,
        report_ttl_path=report_ttl_path,
    )


def _merge_identical_base_documents(
    documents: List[DocumentManifest], raw_documents: List[DocumentRecord]
) -> List[DocumentManifest]:
    """Collapse records that share base IRI and bytes (the two import trees)."""
    merged: Dict[Tuple[str, str], DocumentManifest] = {}
    order: List[Tuple[str, str]] = []
    for doc in documents:
        key = (doc.base_iri, doc.sha256)
        if key not in merged:
            merged[key] = doc
            order.append(key)
            continue
        existing = merged[key]
        for occurrence in doc.role_occurrences:
            if occurrence not in existing.role_occurrences:
                existing.role_occurrences.append(occurrence)
        for via in doc.loaded_via:
            if via not in existing.loaded_via:
                existing.loaded_via.append(via)
    return [merged[key] for key in order]


def document_index(manifest: Manifest) -> Dict[str, DocumentManifest]:
    return {doc.document_id: doc for doc in manifest.documents}


def documents_by_base(manifest: Manifest) -> Dict[str, DocumentManifest]:
    return {normalize_url(doc.base_iri): doc for doc in manifest.documents}
