# -*- coding: utf-8 -*-
"""Offline replay of a captured validation, plus completeness guards."""
import io
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from rdflib import Graph

from ..entrypoints import validate
from .recorder import CaptureSession, normalize_url, patch_loader_bindings, restore_loader_bindings
from .report import QualityReport, extract_report, reports_equivalent
from .store import ArchiveStore, InputSpec, Manifest, documents_by_base

ROLE_DATA = "data"
ROLE_SHAPES = "shapes"
ROLE_ONTOLOGY = "ontology"


class IncompleteReplayError(RuntimeError):
    """Raised when the archive cannot fully reproduce the captured run."""

    def __init__(self, missing_imports: List[Dict[str, str]], reason: str):
        self.missing_imports = missing_imports
        self.reason = reason
        detail = "; ".join(
            f"{item['role']}:{item['importer_base']} -> {item['import_iri']}" for item in missing_imports
        )
        super().__init__(f"{reason}: {detail}")


@dataclass
class ReplayResult:
    complete: bool
    conforms: Optional[bool]
    report: Optional[QualityReport]
    report_matches: Optional[bool]
    relative_expansions_match: bool
    event_sequence_match: bool
    missing_imports: List[Dict[str, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class _OfflineNetworkGuard:
    """Blocks every outbound network call while a replay is in progress."""

    def __init__(self) -> None:
        self._orig_urlopen = None

    def install(self) -> None:
        self._orig_urlopen = urllib.request.urlopen
        urllib.request.urlopen = self._deny  # type: ignore[assignment]

    def uninstall(self) -> None:
        if self._orig_urlopen is not None:
            urllib.request.urlopen = self._orig_urlopen  # type: ignore[assignment]
            self._orig_urlopen = None

    def _deny(self, *args, **kwargs):
        raise RuntimeError("Offline replay archive has no mapping for this URL; network access is disabled.")

    def __enter__(self) -> "_OfflineNetworkGuard":
        self.install()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.uninstall()


class ReplayResolver:
    """Redirects loader fetches to archived blobs using archived base IRIs."""

    def __init__(self, manifest: Manifest, store: ArchiveStore, excluded_ids: Optional[set] = None):
        self.manifest = manifest
        self.store = store
        self.excluded_ids = excluded_ids or set()
        self._by_base = documents_by_base(manifest)
        self._blob_bytes: Dict[str, bytes] = {}
        self._originals: List[Tuple[Any, str, Any]] = []
        self._orig_load: Optional[Callable[..., Any]] = None
        self._orig_web: Optional[Callable[..., Any]] = None
        self.installed = False

    def install(self) -> None:
        from ..rdfutil import load as load_module

        self._orig_load = load_module.load_from_source
        self._orig_web = load_module.get_rdf_from_web
        self._originals = patch_loader_bindings(self._offline_load_from_source, self._orig_web)
        self.installed = True

    def uninstall(self) -> None:
        if not self.installed:
            return
        restore_loader_bindings(self._originals)
        self._originals = []
        self._orig_load = None
        self._orig_web = None
        self.installed = False

    def __enter__(self) -> "ReplayResolver":
        self.install()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.uninstall()

    def _document_for_iri(self, iri: str):
        return self._by_base.get(normalize_url(iri))

    def _blob_for(self, document) -> Optional[bytes]:
        if document.document_id in self.excluded_ids:
            return None
        if document.sha256 not in self._blob_bytes:
            path = self.store.root / document.blob_path
            if not path.exists():
                return None
            data = path.read_bytes()
            self._blob_bytes[document.sha256] = data
        return self._blob_bytes[document.sha256]

    def _offline_load_from_source(self, source, *args, **kwargs):
        assert self._orig_load is not None
        archived_base = getattr(source, "archive_base_iri", None)
        if archived_base is None and isinstance(source, str):
            archived_base = self._map_string_source(source)
        if archived_base is not None:
            document = self._document_for_iri(archived_base)
            if document is None:
                raise RuntimeError(
                    f"Offline replay cannot resolve {source!r}: no archived document for {archived_base!r}."
                )
            raw = self._blob_for(document)
            if raw is None:
                raise RuntimeError(
                    f"Offline replay is missing blob {document.sha256} for document {document.document_id}."
                )
            stream = io.BytesIO(raw)
            kwargs = dict(kwargs)
            kwargs["identifier"] = document.base_iri
            if kwargs.get("rdf_format") is None:
                kwargs["rdf_format"] = document.rdf_format
            return self._orig_load(stream, *args, **kwargs)
        return self._orig_load(source, *args, **kwargs)

    def _map_string_source(self, source: str) -> Optional[str]:
        if source.startswith(("http://", "https://")):
            return source
        if source.startswith("file:"):
            from ..rdfutil.load import path_from_uri

            try:
                return path_from_uri(source).absolute().as_uri()
            except Exception:
                return None
        candidate = Path(source)
        if candidate.exists():
            return candidate.absolute().as_uri()
        return None


def _import_edges(document) -> List[Tuple[str, str]]:
    # A relative import is also visible in absolute form after base
    # resolution, so collapse both views onto one edge per target IRI and
    # keep the raw relative spelling when one exists.
    edges: Dict[str, str] = {}
    for ref in document.absolute_imports:
        edges.setdefault(normalize_url(ref), ref)
    for raw_ref, resolved in zip(document.relative_imports, document.resolved_imports):
        edges[normalize_url(resolved)] = raw_ref
    return [(raw_ref, target) for target, raw_ref in edges.items()]


def find_missing_imports(
    manifest: Manifest, store: ArchiveStore, excluded_ids: Optional[set] = None
) -> List[Dict[str, str]]:
    """Return every import edge whose target document is absent or excluded."""
    excluded_ids = excluded_ids or set()
    available = documents_by_base(manifest)
    missing: List[Dict[str, str]] = []
    for document in manifest.documents:
        importer_present = (
            (store.root / document.blob_path).exists() and document.document_id not in excluded_ids
        )
        if not importer_present:
            continue
        for raw_ref, import_iri in _import_edges(document):
            target = available.get(normalize_url(import_iri))
            if target is None:
                missing.append(
                    {
                        "importer_document": document.document_id,
                        "role": ",".join(part[0] for part in document.role_occurrences),
                        "importer_base": document.base_iri,
                        "raw_import": raw_ref,
                        "import_iri": import_iri,
                        "missing_document": "",
                    }
                )
                continue
            target_present = (
                (store.root / target.blob_path).exists() and target.document_id not in excluded_ids
            )
            if not target_present:
                missing.append(
                    {
                        "importer_document": document.document_id,
                        "role": ",".join(part[0] for part in document.role_occurrences),
                        "importer_base": document.base_iri,
                        "raw_import": raw_ref,
                        "import_iri": import_iri,
                        "missing_document": target.document_id,
                    }
                )
    return missing


def detect_import_cycles(manifest: Manifest) -> List[List[str]]:
    """Return the import cycles present in the archived import graph."""
    available = documents_by_base(manifest)
    graph: Dict[str, List[str]] = {}
    for document in manifest.documents:
        node = normalize_url(document.base_iri)
        targets = []
        for _raw_ref, import_iri in _import_edges(document):
            target_doc = available.get(normalize_url(import_iri))
            targets.append(normalize_url(target_doc.base_iri) if target_doc is not None else normalize_url(import_iri))
        graph[node] = targets
    cycles: List[List[str]] = []
    seen_cycles: set = set()
    state: Dict[str, int] = {}

    def visit(node: str, stack: List[str]) -> None:
        state[node] = 1
        stack.append(node)
        for neighbor in graph.get(node, []):
            if state.get(neighbor) == 1:
                start = stack.index(neighbor)
                cycle = stack[start:] + [neighbor]
                key = tuple(sorted(cycle[:-1]))
                if key not in seen_cycles:
                    seen_cycles.add(key)
                    cycles.append(cycle)
            elif state.get(neighbor) != 2:
                visit(neighbor, stack)
        stack.pop()
        state[node] = 2

    for node in graph:
        if state.get(node) != 2:
            visit(node, [])
    return cycles


def _archive_stream(manifest: Manifest, store: ArchiveStore, spec: InputSpec) -> io.BytesIO:
    document = next(doc for doc in manifest.documents if doc.document_id == spec.document_id)
    raw = (store.root / document.blob_path).read_bytes()
    stream = io.BytesIO(raw)
    stream.archive_base_iri = spec.base_iri  # type: ignore[attr-defined]
    stream.archive_role = spec.role  # type: ignore[attr-defined]
    return stream


def _compare_events(manifest: Manifest, session: CaptureSession) -> Tuple[bool, List[str]]:
    warnings: List[str] = []
    archived = [
        (event.role, event.effective_base, event.sha256, tuple(event.import_chain))
        for event in manifest.load_events
    ]
    replayed = [
        (event.role, event.effective_base, event.sha256, tuple(event.import_chain))
        for event in sorted(session.events, key=lambda item: item.sequence)
    ]
    if archived != replayed:
        warnings.append("Load event sequence (role, base IRI, bytes, import chain) differs from the archive.")
        return False, warnings
    archived_docs = {}
    for doc in manifest.documents:
        for role_name in (pair[0] for pair in doc.role_occurrences):
            archived_docs[(doc.base_iri, (role_name,))] = (
                doc.sha256,
                sorted(tuple(item) for item in doc.relative_term_expansions),
                sorted(doc.resolved_imports),
            )
    for document in session.documents:
        key = (document.effective_base, (document.role,))
        archived_doc = archived_docs.get(key)
        if archived_doc is None:
            warnings.append(f"Replay loaded an unarchived document: {document.effective_base}")
            return False, warnings
        digest, expansions, resolved = archived_doc
        if digest != document.sha256:
            warnings.append(f"Byte content changed for {document.effective_base}.")
            return False, warnings
        replayed_expansions = sorted(tuple(item) for item in document.relative_term_expansions)
        if expansions != replayed_expansions:
            warnings.append(
                f"Relative IRI expansion changed for {document.effective_base}; "
                "the replayed base IRI no longer matches the archive."
            )
            return False, warnings
        if resolved != sorted(document.resolved_imports):
            warnings.append(f"Relative owl:imports resolution changed for {document.effective_base}.")
            return False, warnings
    return True, warnings


def replay_validation(
    archive_dir,
    *,
    exclude_document_ids: Optional[List[str]] = None,
) -> ReplayResult:
    """Re-run the archived validation offline and compare it to the archive."""
    store = ArchiveStore(Path(archive_dir))
    if not store.has_manifest():
        raise FileNotFoundError(f"No manifest.json found in archive: {archive_dir}")
    manifest = store.read_manifest()
    excluded = set(exclude_document_ids or [])

    missing = find_missing_imports(manifest, store, excluded)
    if missing:
        raise IncompleteReplayError(
            missing,
            "Archive is incomplete; the validation report cannot be fully reproduced",
        )

    role_bases = {spec.role: spec.base_iri for spec in manifest.inputs}
    streams = {spec.role: _archive_stream(manifest, store, spec) for spec in manifest.inputs}
    options = manifest.validation
    resolver = ReplayResolver(manifest, store, excluded)
    guard = _OfflineNetworkGuard()
    session = CaptureSession(role_bases)
    with guard, resolver, session:
        conforms, report_graph, report_text = validate(
            streams[ROLE_DATA],
            shacl_graph=streams.get(ROLE_SHAPES),
            ont_graph=streams.get(ROLE_ONTOLOGY),
            inference=options.inference,
            do_owl_imports=options.do_owl_imports,
            advanced=options.advanced,
            allow_infos=options.allow_infos,
            allow_warnings=options.allow_warnings,
        )
    if not isinstance(report_graph, Graph):
        raise RuntimeError("Replay did not produce an RDF validation report graph.")
    replayed_report = extract_report(report_graph, report_text)
    archived_report = QualityReport(
        conforms=bool(manifest.expected_report["conforms"]),
        focus_nodes=list(manifest.expected_report["focus_nodes"]),
        violations=[
            _violation_from_dict(item) for item in manifest.expected_report.get("violations", [])
        ],
    )
    report_matches = reports_equivalent(archived_report, replayed_report)
    events_match, warnings = _compare_events(manifest, session)
    return ReplayResult(
        complete=True,
        conforms=conforms,
        report=replayed_report,
        report_matches=report_matches,
        relative_expansions_match=events_match,
        event_sequence_match=events_match,
        warnings=warnings,
    )


def _violation_from_dict(data: Dict[str, Any]):
    from .report import Violation

    return Violation.from_dict(data)
