# -*- coding: utf-8 -*-
"""Capture-side instrumentation for reproducible RDF validation archives.

The recorder is strictly observational: it wraps the existing loader entry
points, never alters what the loader returns, and records every document that
participates in a validation run together with the base IRI actually used to
expand its relative references.
"""
import hashlib
import importlib
import io
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urldefrag, urljoin, urlsplit

import rdflib
from rdflib import Graph, URIRef

from ..rdfutil import load as load_module

OWL_IMPORTS = rdflib.OWL.imports
SENTINEL_BASE = "http://pyshacl.invalid/archive/sentinel/"

# ``load_from_source`` is re-exported and bound locally by several modules
# (including ``pyshacl.entrypoints``), so every binding must be redirected.
LOAD_PATCH_TARGETS = (
    ("pyshacl.entrypoints", "load_from_source"),
    ("pyshacl.rdfutil", "load_from_source"),
    ("pyshacl.rdfutil.load", "load_from_source"),
)
WEB_PATCH_TARGETS = (
    ("pyshacl.rdfutil", "get_rdf_from_web"),
    ("pyshacl.rdfutil.load", "get_rdf_from_web"),
)


def patch_loader_bindings(load_wrapper, web_wrapper) -> List[Tuple[Any, str, Any]]:
    originals: List[Tuple[Any, str, Any]] = []
    for module_name, attr_name in LOAD_PATCH_TARGETS:
        module = importlib.import_module(module_name)
        originals.append((module, attr_name, getattr(module, attr_name)))
        setattr(module, attr_name, load_wrapper)
    for module_name, attr_name in WEB_PATCH_TARGETS:
        module = importlib.import_module(module_name)
        originals.append((module, attr_name, getattr(module, attr_name)))
        setattr(module, attr_name, web_wrapper)
    return originals


def restore_loader_bindings(originals: List[Tuple[Any, str, Any]]) -> None:
    for module, attr_name, original in originals:
        setattr(module, attr_name, original)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_url(url: str) -> str:
    no_frag, _frag = urldefrag(str(url))
    return no_frag


@dataclass
class LoadEvent:
    """One physical document load performed by the loader."""

    sequence: int
    role: str
    depth: int
    requested_url: str
    effective_base: str
    sha256: str
    byte_length: int
    rdf_format: Optional[str]
    media_type: Optional[str]
    import_chain: List[str] = field(default_factory=list)
    baked_in: bool = False


@dataclass
class DocumentRecord:
    role: str
    occurrence: int
    requested_url: str
    effective_base: str
    sha256: str
    byte_length: int
    rdf_format: Optional[str]
    media_type: Optional[str]
    raw_bytes: bytes
    baked_in: bool
    relative_term_count: int
    relative_term_expansions: List[List[str]] = field(default_factory=list)
    relative_imports: List[str] = field(default_factory=list)
    resolved_imports: List[str] = field(default_factory=list)
    absolute_imports: List[str] = field(default_factory=list)
    loaded_via: List[str] = field(default_factory=list)


@dataclass
class WebFetchRecord:
    sequence: int
    requested_url: str
    final_url: str
    sha256: str
    byte_length: int
    media_type: Optional[str]
    baked_in: bool


def scan_document(raw_bytes: bytes, rdf_format: Optional[str], base_iri: str) -> Dict[str, Any]:
    """Inspect a document's relative references with a sentinel base.

    Parsing the same bytes against a sentinel base exposes every term that
    originated from a relative reference, including relative
    ``owl:imports`` targets, and the real base parse shows the absolute
    IRIs those references expand to.
    """
    graph_sentinel = Graph()
    graph_sentinel.parse(data=raw_bytes, format=rdf_format, publicID=SENTINEL_BASE)
    graph_real = Graph()
    graph_real.parse(data=raw_bytes, format=rdf_format, publicID=base_iri)

    expansions: Dict[str, str] = {}
    for term in _uri_terms(graph_sentinel):
        text = str(term)
        if not text.startswith(SENTINEL_BASE):
            continue
        raw_ref = text[len(SENTINEL_BASE):]
        expansions[raw_ref] = urljoin(base_iri, raw_ref)

    relative_imports: List[str] = []
    resolved_imports: List[str] = []
    for sentinel_ref in sorted(str(o) for o in graph_sentinel.objects(predicate=OWL_IMPORTS)):
        if not sentinel_ref.startswith(SENTINEL_BASE):
            continue
        raw_ref = sentinel_ref[len(SENTINEL_BASE):]
        relative_imports.append(raw_ref)
        resolved_imports.append(urljoin(base_iri, raw_ref))
    absolute_imports = sorted(
        str(o)
        for o in graph_real.objects(predicate=OWL_IMPORTS)
        if urlsplit(str(o)).scheme
    )
    return {
        "relative_term_count": len(expansions),
        "relative_term_expansions": sorted(expansions.items()),
        "relative_imports": relative_imports,
        "resolved_imports": resolved_imports,
        "absolute_imports": absolute_imports,
    }


def _uri_terms(graph: Graph):
    terms = set()
    for s, p, o in graph:
        for term in (s, p, o):
            if isinstance(term, URIRef):
                terms.add(term)
    return terms


def header_base_iri(raw_bytes: bytes, rdf_format: Optional[str]) -> Optional[str]:
    """Mirror load.py's pre-parse of ``# baseURI:`` style Turtle headers."""
    if rdf_format not in ("turtle", "n3", None):
        return None
    base_iri: Optional[str] = None
    for raw_line in raw_bytes.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if not line.startswith(b"#"):
            break
        stripped = line.lstrip(b"# ").strip()
        parts = stripped.split(b":", 1)
        if len(parts) < 2:
            continue
        keyword = parts[0].strip().lower()
        if keyword in (b"baseuri", b"base"):
            value = parts[1].strip()
            if value:
                base_iri = value.decode("utf-8")
    return base_iri


def resolve_base_iri(source: Any, kwargs: Dict[str, Any], final_urls: Dict[str, str]) -> Optional[str]:
    """Determine the base IRI the loader effectively used for *source*."""
    archived_base = getattr(source, "archive_base_iri", None)
    if archived_base:
        return str(archived_base)
    if isinstance(source, str):
        if source.startswith(("http://", "https://")):
            return final_urls.get(normalize_url(source), source)
        if source.startswith("file:"):
            try:
                return load_module.path_from_uri(source).absolute().as_uri()
            except Exception:
                return None
        candidate = Path(source)
        if candidate.exists():
            return candidate.absolute().as_uri()
        return None
    if hasattr(source, "name") and getattr(source, "name", None):
        name = str(source.name)
        if name.startswith("file:"):
            try:
                return load_module.path_from_uri(name).absolute().as_uri()
            except Exception:
                return None
        try:
            return Path(name).absolute().as_uri()
        except Exception:
            return None
    identifier = kwargs.get("identifier")
    return str(identifier) if identifier else None


class CaptureSession:
    """Observes loader activity for one :func:`pyshacl.validate` invocation."""

    def __init__(self, role_bases: Dict[str, str]):
        self._role_bases = {role: normalize_url(base) for role, base in role_bases.items()}
        self._lock = threading.RLock()
        self._role_stack: List[str] = []
        self._sequence = 0
        self._web_sequence = 0
        self.events: List[LoadEvent] = []
        self.web_fetches: List[WebFetchRecord] = []
        self._documents: Dict[Tuple[str, str], DocumentRecord] = {}
        self._web_bytes: Dict[str, Tuple[str, bytes, Optional[str]]] = {}
        self._originals: List[Tuple[Any, str, Any]] = []
        self._orig_load: Optional[Callable[..., Any]] = None
        self._orig_web: Optional[Callable[..., Any]] = None
        self.installed = False

    def install(self) -> None:
        if self.installed:
            return
        self._orig_load = load_module.load_from_source
        self._orig_web = load_module.get_rdf_from_web
        self._originals = patch_loader_bindings(
            self._wrapped_load_from_source, self._wrapped_get_rdf_from_web
        )
        self.installed = True

    def uninstall(self) -> None:
        if not self.installed:
            return
        restore_loader_bindings(self._originals)
        self._originals = []
        self._orig_load = None
        self._orig_web = None
        self.installed = False

    def __enter__(self) -> "CaptureSession":
        self.install()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.uninstall()

    @property
    def documents(self) -> List[DocumentRecord]:
        return [self._documents[key] for key in sorted(self._documents.keys())]

    def _wrapped_get_rdf_from_web(self, url):
        assert self._orig_web is not None
        result = self._orig_web(url)
        resp, filename, known_format, kind_or_raw = result
        with self._lock:
            self._web_sequence += 1
            seq = self._web_sequence
        if isinstance(resp, rdflib.Graph):
            raw = resp.serialize(format="turtle").encode("utf-8")
            digest = sha256_bytes(raw)
            final_url = str(url)
            self.web_fetches.append(WebFetchRecord(seq, str(url), final_url, digest, len(raw), "text/turtle", True))
            self._web_bytes[normalize_url(str(url))] = (final_url, raw, "text/turtle")
            return result
        raw = resp.read()
        digest = sha256_bytes(raw)
        media_type = None
        headers = getattr(resp, "headers", None)
        if headers is not None and hasattr(headers, "get_content_type"):
            try:
                media_type = headers.get_content_type()
            except Exception:
                media_type = None
        final_url = resp.geturl() if hasattr(resp, "geturl") else str(url)
        self.web_fetches.append(
            WebFetchRecord(seq, str(url), final_url, digest, len(raw), media_type, False)
        )
        self._web_bytes[normalize_url(str(url))] = (final_url, raw, media_type)
        wrapped = io.BytesIO(raw)
        wrapped.name = getattr(resp, "name", filename) or filename  # type: ignore[attr-defined]
        return wrapped, filename, known_format, kind_or_raw

    def _wrapped_load_from_source(self, source, *args, **kwargs):
        assert self._orig_load is not None
        owl_flag = kwargs.get("do_owl_imports", False)
        nested = isinstance(owl_flag, int) and not isinstance(owl_flag, bool) and int(owl_flag) > 1
        role = self._role_stack[-1] if self._role_stack else self._role_for_source(source, kwargs)
        if not nested:
            self._role_stack.append(role)
        chain_snapshot = list(kwargs.get("import_chain") or [])
        try:
            result = self._orig_load(source, *args, **kwargs)
        finally:
            if not nested:
                self._role_stack.pop()
        self._record_event(role, 2 if nested else 1, source, kwargs, result, chain_snapshot)
        return result

    def _role_for_source(self, source: Any, kwargs: Dict[str, Any]) -> str:
        stream_role = getattr(source, "archive_role", None)
        if stream_role:
            return str(stream_role)
        base = resolve_base_iri(source, kwargs, {k: v[0] for k, v in self._web_bytes.items()})
        normalized = normalize_url(base) if base else None
        for role, role_base in self._role_bases.items():
            if normalized is not None and role_base == normalized:
                return role
        return self._role_stack[-1] if self._role_stack else "unknown"

    def _record_event(
        self,
        role: str,
        depth: int,
        source: Any,
        kwargs: Dict[str, Any],
        result: Any,
        chain_snapshot: List[str],
    ) -> None:
        final_urls = {key: value[0] for key, value in self._web_bytes.items()}
        effective_base = resolve_base_iri(source, kwargs, final_urls)
        if not effective_base:
            return
        raw_bytes, media_type, rdf_format, baked = self._read_source_bytes(source, kwargs, effective_base)
        if raw_bytes is None:
            return
        override = header_base_iri(raw_bytes, rdf_format)
        if override:
            effective_base = override
        digest = sha256_bytes(raw_bytes)
        with self._lock:
            self._sequence += 1
            seq = self._sequence
        requested = normalize_url(
            str(kwargs.get("identifier") or (source if isinstance(source, str) else effective_base))
        )
        self.events.append(
            LoadEvent(
                sequence=seq,
                role=role,
                depth=depth,
                requested_url=requested,
                effective_base=effective_base,
                sha256=digest,
                byte_length=len(raw_bytes),
                rdf_format=rdf_format,
                media_type=media_type,
                import_chain=chain_snapshot,
                baked_in=baked,
            )
        )
        key = (role, effective_base)
        existing = self._documents.get(key)
        if existing is None:
            scanned = scan_document(raw_bytes, rdf_format, effective_base)
            occurrence = sum(1 for doc in self._documents.values() if doc.role == role)
            self._documents[key] = DocumentRecord(
                role=role,
                occurrence=occurrence,
                requested_url=requested,
                effective_base=effective_base,
                sha256=digest,
                byte_length=len(raw_bytes),
                rdf_format=rdf_format,
                media_type=media_type,
                raw_bytes=raw_bytes,
                baked_in=baked,
                relative_term_count=scanned["relative_term_count"],
                relative_term_expansions=[list(item) for item in scanned["relative_term_expansions"]],
                relative_imports=scanned["relative_imports"],
                resolved_imports=scanned["resolved_imports"],
                absolute_imports=scanned["absolute_imports"],
                loaded_via=[requested],
            )
        elif requested not in existing.loaded_via:
            existing.loaded_via.append(requested)

    def _read_source_bytes(
        self, source: Any, kwargs: Dict[str, Any], base: str
    ) -> Tuple[Optional[bytes], Optional[str], Optional[str], bool]:
        if isinstance(source, str) and source.startswith(("http://", "https://")):
            cached = self._web_bytes.get(normalize_url(source))
            if cached is not None:
                _final, raw, media_type = cached
                return raw, media_type, kwargs.get("rdf_format"), False
            return None, None, kwargs.get("rdf_format"), False
        if isinstance(source, str) and not source.startswith(("http://", "https://")):
            if source.startswith("file:"):
                path = load_module.path_from_uri(source)
            else:
                path = _path_from(source)
            if path is not None and path.exists():
                return path.read_bytes(), None, kwargs.get("rdf_format") or _format_from_name(path.name), False
            return None, None, kwargs.get("rdf_format"), False
        if hasattr(source, "read"):
            try:
                source.seek(0)
                data = source.read()
                if isinstance(data, str):
                    data = data.encode("utf-8")
                name = getattr(source, "name", None)
                fmt = kwargs.get("rdf_format") or (_format_from_name(str(name)) if name else None)
                return data, None, fmt, False
            except Exception:
                return None, None, kwargs.get("rdf_format"), False
        return None, None, kwargs.get("rdf_format"), False


def _path_from(source: str) -> Optional[Path]:
    try:
        return Path(source).resolve()
    except Exception:
        return None


def _format_from_name(name: str) -> Optional[str]:
    lower = name.lower()
    for suffix, fmt in (
        (".ttl", "turtle"),
        (".nt", "nt"),
        (".n3", "n3"),
        (".jsonld", "json-ld"),
        (".json", "json-ld"),
        (".nq", "nquads"),
        (".trig", "trig"),
        (".xml", "xml"),
        (".rdf", "xml"),
    ):
        if lower.endswith(suffix):
            return fmt
    return None
