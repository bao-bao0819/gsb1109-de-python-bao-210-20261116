# -*- coding: utf-8 -*-
"""Public entry points for capturing and replaying offline validation archives."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from rdflib import Graph

from ..entrypoints import validate
from .recorder import CaptureSession, normalize_url
from .report import QualityReport, extract_report, reports_equivalent
from .store import (
    ArchiveStore,
    InputSpec,
    Manifest,
    ValidationOptions,
    build_manifest,
)

PathLike = Union[str, Path]
GraphInput = Union[PathLike, Dict[str, Any]]

ROLE_DATA = "data"
ROLE_SHAPES = "shapes"
ROLE_ONTOLOGY = "ontology"


@dataclass
class CaptureResult:
    archive_dir: Path
    manifest: Manifest
    report: QualityReport
    unique_blobs: int
    stored_bytes: int


def _normalize_input(role: str, value: GraphInput) -> Dict[str, Any]:
    if isinstance(value, dict):
        spec = dict(value)
        spec.setdefault("role", role)
        if "source" not in spec:
            raise ValueError(f"Input {role!r} requires a 'source' entry.")
        return spec
    return {"role": role, "source": str(value)}


def _base_iri_for(spec: Dict[str, Any]) -> str:
    explicit = spec.get("base_iri")
    if explicit:
        return normalize_url(str(explicit))
    source = spec["source"]
    if isinstance(source, str) and source.startswith(("http://", "https://")):
        return normalize_url(source)
    if isinstance(source, str) and source.startswith("file:"):
        from ..rdfutil.load import path_from_uri

        return normalize_url(path_from_uri(source).absolute().as_uri())
    if isinstance(source, str):
        candidate = Path(source)
        if candidate.exists():
            return normalize_url(candidate.absolute().as_uri())
    raise ValueError(
        f"Input {spec.get('role')!r} is an in-memory source without 'base_iri'; "
        "the base IRI used to expand relative references must be stated explicitly."
    )


def capture_validation(
    archive_dir: PathLike,
    data_graph: GraphInput,
    *,
    shacl_graph: Optional[GraphInput] = None,
    ont_graph: Optional[GraphInput] = None,
    inference: str = "none",
    do_owl_imports: bool = True,
    advanced: bool = False,
    allow_infos: bool = False,
    allow_warnings: bool = False,
) -> CaptureResult:
    """Run a validation, capture every loaded document, and write the archive.

    Each input may be a filesystem path or a dict with ``source``, an
    explicit ``base_iri`` (required for in-memory sources), and optional
    ``rdf_format``. The returned archive is self-contained: identical bytes
    shared by the two import trees are stored once while the manifest keeps
    the distinct base IRI and role (shapes vs. ontology) of each load.
    """
    data_spec = _normalize_input(ROLE_DATA, data_graph)
    shapes_spec = _normalize_input(ROLE_SHAPES, shacl_graph) if shacl_graph is not None else None
    ontology_spec = _normalize_input(ROLE_ONTOLOGY, ont_graph) if ont_graph is not None else None
    specs = [spec for spec in (data_spec, shapes_spec, ontology_spec) if spec is not None]
    role_bases = {spec["role"]: _base_iri_for(spec) for spec in specs}

    session = CaptureSession(role_bases)
    with session:
        conforms, report_graph, report_text = validate(
            data_spec["source"],
            shacl_graph=shapes_spec["source"] if shapes_spec else None,
            ont_graph=ontology_spec["source"] if ontology_spec else None,
            data_graph_format=data_spec.get("rdf_format"),
            shacl_graph_format=shapes_spec.get("rdf_format") if shapes_spec else None,
            ont_graph_format=ontology_spec.get("rdf_format") if ontology_spec else None,
            inference=inference,
            do_owl_imports=do_owl_imports,
            advanced=advanced,
            allow_infos=allow_infos,
            allow_warnings=allow_warnings,
        )
    if not isinstance(report_graph, Graph):
        raise RuntimeError("Validation did not produce an RDF report graph; nothing was archived.")
    report = extract_report(report_graph, report_text)

    store = ArchiveStore(Path(archive_dir))
    store.ensure_dirs()
    report_text_path = store.write_text("reports/report.txt", report.report_text)
    report_ttl_path = store.write_text("reports/report.ttl", report.report_ttl)

    input_specs: List[InputSpec] = []
    documents = session.documents
    base_to_doc = {doc.effective_base: doc for doc in documents}
    for spec in specs:
        base = role_bases[spec["role"]]
        doc = base_to_doc.get(base)
        if doc is None:
            raise RuntimeError(
                f"Recorder never observed input role {spec['role']!r} at base {base!r}."
            )
        input_specs.append(
            InputSpec(
                role=spec["role"],
                base_iri=base,
                document_id="",
                rdf_format=doc.rdf_format,
                requested_url=doc.requested_url,
            )
        )

    options = ValidationOptions(
        inference=inference,
        do_owl_imports=do_owl_imports,
        advanced=advanced,
        allow_infos=allow_infos,
        allow_warnings=allow_warnings,
    )
    manifest = build_manifest(
        validation=options,
        inputs=input_specs,
        documents=documents,
        events=session.events,
        web_fetches=session.web_fetches,
        store=store,
        expected_report=report.summary(),
        report_text_path=report_text_path,
        report_ttl_path=report_ttl_path,
    )
    for spec in manifest.inputs:
        matching = next(
            doc.document_id
            for doc in manifest.documents
            if normalize_url(doc.base_iri) == spec.base_iri
        )
        spec.document_id = matching
    store.write_manifest(manifest)

    unique_blobs = len({doc.sha256 for doc in manifest.documents})
    stored_bytes = sum(
        (store.root / path).stat().st_size for path in {doc.blob_path for doc in manifest.documents}
    )
    return CaptureResult(
        archive_dir=store.root,
        manifest=manifest,
        report=report,
        unique_blobs=unique_blobs,
        stored_bytes=stored_bytes,
    )


def replay_comparison(
    archive_dir: PathLike, *, exclude_document_ids: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Convenience wrapper returning a JSON-friendly replay verdict."""
    from .replay import replay_validation

    result = replay_validation(archive_dir, exclude_document_ids=exclude_document_ids)
    return {
        "complete": result.complete,
        "conforms": result.conforms,
        "report_matches": result.report_matches,
        "relative_expansions_match": result.relative_expansions_match,
        "event_sequence_match": result.event_sequence_match,
        "missing_imports": result.missing_imports,
        "warnings": result.warnings,
        "report": result.report.summary() if result.report is not None else None,
    }


__all__ = [
    "capture_validation",
    "replay_comparison",
    "reports_equivalent",
    "CaptureResult",
]
