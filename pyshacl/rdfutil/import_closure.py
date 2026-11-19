# -*- coding: utf-8 -*-
#
"""Capture and replay the exact owl:imports closure used by a validation run.

A validation report is only reproducible if the materials it was built from
are known. The shapes root document may be unchanged while a document it
imports (transitively) is replaced upstream, silently changing the result.

This module records, at capture time, the *actual* import closure of a root
RDF document: every dependency that was really fetched, its content digest,
and its version identity (owl:versionIRI / owl:versionInfo). The closure is
saved as a self-contained directory (manifest + content-addressed copies of
every dependency) so a historical report can later be rebuilt offline from
exactly the materials that were used at the time.

Rules honoured here:

- Duplicate imports of the same URI are recorded once; extra references are
  counted on the canonical record.
- Import cycles are detected and recorded, never followed forever.
- A dependency that cannot be retrieved is recorded as *unverified*. Nothing
  is silently substituted from a machine-local cache (``baked_in`` is not
  consulted here); every byte stored is fetched from the declared location
  and hashed at capture time.
- Replaying a closure verifies every digest. Missing or tampered materials,
  or unverified dependencies, are hard errors unless the caller explicitly
  opts in with ``allow_unverified=True``.
"""
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging import WARNING, Logger, getLogger
from pathlib import Path
from typing import List, Optional, Tuple, Union
from urllib import request

import rdflib
from rdflib.namespace import SDO, NamespaceManager
from rdflib.term import URIRef
from rdflib.util import guess_format

from ..errors import ReportableRuntimeError
from .load import MAX_OWL_IMPORT_DEPTH, GraphLike, load_from_source, path_from_uri

SCHEMA = SDO
MANIFEST_FILENAME = "import-closure.json"
MANIFEST_VERSION = 1

_FORMAT_EXTENSIONS = {
    'turtle': '.ttl',
    'xml': '.rdf',
    'json-ld': '.jsonld',
    'nt': '.nt',
    'n3': '.n3',
    'nquads': '.nq',
    'trig': '.trig',
    'hext': '.hext',
}


@dataclass
class ImportRecord:
    """One node of the import closure.

    ``verified`` is True only when the exact content of the dependency was
    obtained and hashed at capture time. Unverified records are the
    dependencies a rebuilt report cannot attest to.
    """

    uri: str
    kind: str = "import"  # "root" or "import"
    status: str = "resolved"  # "resolved" or "unresolved"
    depth: int = 0
    imported_by: List[str] = field(default_factory=list)
    duplicate_refs: int = 0
    cycle_refs: List[str] = field(default_factory=list)
    digest: Optional[str] = None
    content_file: Optional[str] = None
    rdf_format: Optional[str] = None
    triples: Optional[int] = None
    ontology_uri: Optional[str] = None
    version_iri: Optional[str] = None
    version_info: Optional[str] = None
    verified: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'uri': self.uri,
            'kind': self.kind,
            'status': self.status,
            'depth': self.depth,
            'imported_by': list(self.imported_by),
            'duplicate_refs': self.duplicate_refs,
            'cycle_refs': list(self.cycle_refs),
            'digest': self.digest,
            'content_file': self.content_file,
            'rdf_format': self.rdf_format,
            'triples': self.triples,
            'ontology_uri': self.ontology_uri,
            'version_iri': self.version_iri,
            'version_info': self.version_info,
            'verified': self.verified,
            'error': self.error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ImportRecord":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class ImportClosure:
    """The recorded import closure of one root document."""

    root_uri: str
    created_at: str
    records: List[ImportRecord] = field(default_factory=list)
    manifest_version: int = MANIFEST_VERSION

    @property
    def root_record(self) -> Optional[ImportRecord]:
        for r in self.records:
            if r.kind == "root":
                return r
        return None

    @property
    def unverified(self) -> List[ImportRecord]:
        """Dependencies whose content was not proven at capture time."""
        return [r for r in self.records if not r.verified]

    def to_dict(self) -> dict:
        return {
            'manifest_version': self.manifest_version,
            'created_at': self.created_at,
            'root_uri': self.root_uri,
            'records': [r.to_dict() for r in self.records],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ImportClosure":
        return cls(
            root_uri=data['root_uri'],
            created_at=data['created_at'],
            records=[ImportRecord.from_dict(r) for r in data.get('records', [])],
            manifest_version=data.get('manifest_version', MANIFEST_VERSION),
        )

    @classmethod
    def load(cls, closure_dir: Union[str, Path]) -> "ImportClosure":
        manifest_path = Path(closure_dir) / MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise ReportableRuntimeError(f"Import closure manifest not found: {manifest_path}")
        with open(manifest_path, 'r', encoding='utf-8') as fh:
            return cls.from_dict(json.load(fh))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fetch_raw(uri: str) -> Tuple[bytes, str]:
    """Fetch the raw bytes of a dependency from its declared location.

    This deliberately does not consult ``baked_in`` or any other
    machine-local cache: a missing dependency must surface as a failure,
    not be silently papered over with whatever this machine happens to hold.
    Returns (content_bytes, resolved_location).
    """
    if uri.startswith('file:'):
        path = path_from_uri(uri, relative_to=None)
        return path.read_bytes(), str(path)
    if uri.startswith('http:') or uri.startswith('https:'):
        headers = {
            'Accept': 'text/turtle, application/rdf+xml, application/ld+json, application/n-triples, text/plain',
            'Accept-Encoding': 'identity',
        }
        req = request.Request(uri, headers=headers)
        with request.urlopen(req) as resp:
            return resp.read(), resp.geturl()
    # Bare local path
    path = Path(uri)
    return path.read_bytes(), str(path.resolve())


def _guess_parse_format(location: str, content: bytes) -> Optional[str]:
    fmt = guess_format(location)
    if fmt:
        return fmt
    head = content[:256].lstrip().lower()
    if head.startswith(b'<?xml') or head.startswith(b'<rdf:'):
        return 'xml'
    if head.startswith(b'{') or head.startswith(b'['):
        return 'json-ld'
    if head.startswith(b'@prefix') or head.startswith(b'prefix') or head.startswith(b'@base') or head.startswith(b'#'):
        return 'turtle'
    return None


def _parse_content(content: bytes, location: str) -> Tuple[Optional[rdflib.Graph], Optional[str], Optional[str]]:
    """Parse fetched bytes. Returns (graph, rdf_format, parse_error)."""
    candidates: List[str] = []
    guessed = _guess_parse_format(location, content)
    if guessed:
        candidates.append(guessed)
    for fallback in ('turtle', 'xml', 'json-ld', 'nt'):
        if fallback not in candidates:
            candidates.append(fallback)
    last_error: Optional[Exception] = None
    for fmt in candidates:
        try:
            graph = rdflib.Graph()
            graph.parse(data=content, format=fmt, publicID=location)
            return graph, fmt, None
        except Exception as e:  # noqa: BLE001 - try the next candidate format
            last_error = e
    return None, None, str(last_error) if last_error else "unknown parse error"


def _extract_version_identity(graph: rdflib.Graph, import_uri: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (ontology_uri, version_iri, version_info) for the imported ontology."""
    ontologies = list(graph.subjects(rdflib.RDF.type, rdflib.OWL.Ontology))
    chosen = None
    stripped = import_uri.rstrip('#/')
    for ont in ontologies:
        if str(ont) == import_uri or str(ont).rstrip('#/') == stripped:
            chosen = ont
            break
    if chosen is None and ontologies:
        chosen = ontologies[0]
    if chosen is None:
        return None, None, None
    version_iri = graph.value(chosen, rdflib.OWL.versionIRI)
    version_info = graph.value(chosen, rdflib.OWL.versionInfo)
    return (
        str(chosen),
        str(version_iri) if version_iri is not None else None,
        str(version_info) if version_info is not None else None,
    )


def _import_targets(graph: GraphLike) -> List[Tuple[str, str]]:
    """All (importer, imported-uri) pairs declared via owl:imports.

    BNode import descriptions with schema:url are resolved like the main
    loader does.
    """
    targets: List[Tuple[str, str]] = []
    for importer, imported in graph.subject_objects(rdflib.OWL.imports):
        if isinstance(imported, rdflib.BNode):
            urls = sorted(str(u) for u in graph.objects(imported, SCHEMA.url))
            if not urls:
                continue
            file_urls = [u for u in urls if u.startswith('file:')]
            imported_uri = (file_urls or urls)[0]
        else:
            imported_uri = str(imported)
        targets.append((str(importer), imported_uri))
    return targets


def capture_import_closure(
    root: Union[GraphLike, str, bytes],
    dest_dir: Union[str, Path],
    rdf_format: Optional[str] = None,
    max_depth: int = MAX_OWL_IMPORT_DEPTH,
    logger: Optional[Logger] = None,
) -> ImportClosure:
    """Capture the actual owl:imports closure of ``root`` into ``dest_dir``.

    Writes ``import-closure.json`` plus one content-addressed file per
    resolved dependency, so the exact materials of this run can be replayed
    offline later. Dependencies that cannot be fetched or parsed are recorded
    as unverified in the manifest; they are never silently replaced with
    machine-local cached copies.
    """
    if logger is None:
        logger = getLogger("rdfutil.import_closure")
        logger.setLevel(WARNING)
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    root_graph = load_from_source(root, rdf_format=rdf_format, multigraph=False, do_owl_imports=False, logger=logger)
    root_base = getattr(root_graph, 'base', None)
    root_uri = str(root_base) if root_base else str(root_graph.identifier)

    created_at = datetime.now(timezone.utc).isoformat()
    closure = ImportClosure(root_uri=root_uri, created_at=created_at)
    records_by_uri: dict = {}

    def _store_content(content: bytes, fmt: Optional[str]) -> Tuple[str, str]:
        digest = _sha256(content)
        ext = _FORMAT_EXTENSIONS.get(fmt or '', '.ttl')
        filename = f"{digest[:16]}{ext}"
        (dest / filename).write_bytes(content)
        return digest, filename

    root_content = root_graph.serialize(format='turtle').encode('utf-8')
    root_digest, root_file = _store_content(root_content, 'turtle')
    root_record = ImportRecord(
        uri=root_uri,
        kind="root",
        status="resolved",
        depth=0,
        digest=f"sha256:{root_digest}",
        content_file=root_file,
        rdf_format='turtle',
        triples=len(root_graph),
        verified=True,
    )
    ont_uri, v_iri, v_info = _extract_version_identity(root_graph, root_uri)
    root_record.ontology_uri = ont_uri
    root_record.version_iri = v_iri
    root_record.version_info = v_info
    closure.records.append(root_record)
    records_by_uri[root_uri] = root_record

    def _visit(graph: GraphLike, importer_uri: str, depth: int, chain: Tuple[str, ...]) -> None:
        if depth > max_depth:
            return
        for importer, imported_uri in _import_targets(graph):
            existing = records_by_uri.get(imported_uri)
            if existing is not None:
                if imported_uri in chain:
                    if importer not in existing.cycle_refs:
                        existing.cycle_refs.append(importer)
                else:
                    existing.duplicate_refs += 1
                    if importer not in existing.imported_by:
                        existing.imported_by.append(importer)
                continue
            record = ImportRecord(uri=imported_uri, kind="import", depth=depth, imported_by=[importer])
            records_by_uri[imported_uri] = record
            closure.records.append(record)
            try:
                content, location = _fetch_raw(imported_uri)
            except Exception as e:  # noqa: BLE001 - any fetch failure means unverified
                record.status = "unresolved"
                record.verified = False
                record.error = f"{type(e).__name__}: {e}"
                logger.warning("Import %s could not be retrieved; recorded as unverified.", imported_uri)
                continue
            digest = _sha256(content)
            parsed, fmt, parse_error = _parse_content(content, location)
            record.digest = f"sha256:{digest}"
            if parsed is None:
                # We hold the exact bytes, but cannot read them as RDF:
                # keep the material, yet do not claim the dependency is verified.
                record.status = "unresolved"
                record.verified = False
                record.error = f"Unparseable RDF content: {parse_error}"
                ext = _FORMAT_EXTENSIONS.get(fmt or '', '.bin')
                filename = f"{digest[:16]}{ext}"
                (dest / filename).write_bytes(content)
                record.content_file = filename
                continue
            ext = _FORMAT_EXTENSIONS.get(fmt or '', '.ttl')
            filename = f"{digest[:16]}{ext}"
            (dest / filename).write_bytes(content)
            record.content_file = filename
            record.rdf_format = fmt
            record.triples = len(parsed)
            record.status = "resolved"
            record.verified = True
            ont_uri, v_iri, v_info = _extract_version_identity(parsed, imported_uri)
            record.ontology_uri = ont_uri
            record.version_iri = v_iri
            record.version_info = v_info
            _visit(parsed, imported_uri, depth + 1, chain + (imported_uri,))

    _visit(root_graph, root_uri, 1, (root_uri,))

    manifest_path = dest / MANIFEST_FILENAME
    with open(manifest_path, 'w', encoding='utf-8') as fh:
        json.dump(closure.to_dict(), fh, indent=2, sort_keys=False)
        fh.write('\n')
    return closure


def replay_import_closure(
    closure_dir: Union[str, Path],
    allow_unverified: bool = False,
    logger: Optional[Logger] = None,
) -> Tuple[GraphLike, ImportClosure]:
    """Rebuild a graph from a previously captured import closure.

    Every stored dependency is checked against its recorded digest before
    parsing, so a rebuilt report is guaranteed to use the same materials as
    the original run. Unverified dependencies (recorded as not retrievable at
    capture time) abort the rebuild unless ``allow_unverified=True`` is given,
    in which case they are skipped with a warning and remain listed on the
    returned closure's ``unverified`` property.
    """
    if logger is None:
        logger = getLogger("rdfutil.import_closure")
        logger.setLevel(WARNING)
    closure_dir = Path(closure_dir)
    closure = ImportClosure.load(closure_dir)
    unverified = closure.unverified
    if unverified and not allow_unverified:
        listing = ", ".join(r.uri for r in unverified)
        raise ReportableRuntimeError(
            f"Import closure has {len(unverified)} unverified dependenc"
            f"{'y' if len(unverified) == 1 else 'ies'} ({listing}). "
            "Rebuild aborted; pass allow_unverified=True to proceed without them."
        )
    for record in unverified:
        logger.warning("Skipping unverified dependency %s (%s)", record.uri, record.error or "no content")

    dataset = rdflib.Dataset(default_union=True)
    dataset.namespace_manager = NamespaceManager(dataset, 'core')
    dest_g = dataset.default_graph
    for record in closure.records:
        if not record.verified:
            continue
        assert record.content_file is not None and record.digest is not None
        content_path = closure_dir / record.content_file
        if not content_path.is_file():
            raise ReportableRuntimeError(
                f"Import closure material missing for {record.uri}: {content_path}. "
                "Cannot rebuild the historical report without it."
            )
        content = content_path.read_bytes()
        actual_digest = _sha256(content)
        if f"sha256:{actual_digest}" != record.digest:
            raise ReportableRuntimeError(
                f"Import closure material for {record.uri} does not match its recorded digest. "
                "The stored materials were altered; refusing to rebuild."
            )
        dest_g.parse(data=content, format=record.rdf_format or 'turtle', publicID=record.uri)
    return dataset, closure
