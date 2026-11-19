# -*- coding: utf-8 -*-
"""Tenant material manifests, validation specs and reports.

A :class:`MaterialManifest` is the per-tenant access control list for one
tenant. It only ever references materials that tenant is allowed to see:

* tenant-visible data graphs,
* public shapes plus that tenant's tenant-specific shapes,
* versioned shared dictionaries and tenant-specific dictionaries,
* optional ontology documents.

Source graphs are loaded once at registration time and owned by the manifest.
The runner (:mod:`pyshacl.multitenant.runner`) clones them anew for every
validation task, so a manifest graph is never mutated by a run and two runs can
never share a live graph, a shape cache or an inference closure.
"""
import hashlib
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, IO, List, Optional, Union

from rdflib import Dataset, Graph, URIRef
from rdflib.compare import to_canonical_graph

from ..rdfutil import load_from_source
from .errors import TenantScopeError, TenantValidationError

GraphSource = Union[Graph, Dataset, str, bytes, IO[Any]]

KIND_DATA = "data"
KIND_SHAPES = "shapes"
KIND_DICTIONARY = "dictionary"
KIND_ONTOLOGY = "ontology"

VISIBILITY_PUBLIC = "public"
VISIBILITY_TENANT = "tenant"

_VALID_KINDS = (KIND_DATA, KIND_SHAPES, KIND_DICTIONARY, KIND_ONTOLOGY)
_VALID_VISIBILITIES = (VISIBILITY_PUBLIC, VISIBILITY_TENANT)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def graph_fingerprint(graph: Graph) -> str:
    """Return a stable SHA-256 fingerprint of a graph's triples.

    The graph is first canonicalized with :func:`rdflib.compare.to_canonical_graph`,
    which relabels blank nodes based on graph isomorphism. As a result two
    parses of the same document (which would otherwise get different generated
    blank-node labels) produce the same fingerprint, while any real triple
    change does not. The canonical N-Triples lines are sorted before hashing so
    iteration order is irrelevant.
    """
    if not isinstance(graph, Graph):
        raise TypeError("graph_fingerprint expects an rdflib.Graph instance")
    canonical_graph = to_canonical_graph(graph)
    serialized = canonical_graph.serialize(format="nt")
    if isinstance(serialized, bytes):
        serialized = serialized.decode("utf-8")
    canonical = "".join(sorted(line for line in serialized.splitlines(keepends=True) if line.strip()))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_material_graph(source: GraphSource, rdf_format: Optional[str] = None) -> Graph:
    """Load a material source into a manifest-owned in-memory graph."""
    if isinstance(source, Dataset):
        raise TypeError(
            "Multi-tenant materials must be rdflib.Graph instances; combine named graphs "
            "into separate registered materials instead of passing a Dataset."
        )
    if isinstance(source, Graph):
        graph = Graph(bind_namespaces="core")
        for prefix, namespace in source.namespace_manager.namespaces():
            graph.namespace_manager.bind(prefix, namespace, override=True, replace=True)
        for triple in source:
            graph.add(triple)
        return graph
    loaded = load_from_source(source, rdf_format=rdf_format, multigraph=False, do_owl_imports=False)
    if isinstance(loaded, Dataset):
        raise TypeError("Multi-tenant materials must load into a single rdflib.Graph, not a Dataset.")
    return loaded


@dataclass
class TenantMaterial:
    """One registered piece of material a tenant is authorized to use."""

    material_id: str
    kind: str
    visibility: str
    graph: Graph
    version: Optional[str] = None
    label: Optional[str] = None
    rdf_format: Optional[str] = None
    fingerprint: str = ""
    registered_at: datetime = field(default_factory=_utcnow)

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise TenantValidationError("Unknown material kind: {!r}".format(self.kind))
        if self.visibility not in _VALID_VISIBILITIES:
            raise TenantValidationError("Unknown material visibility: {!r}".format(self.visibility))
        if not self.fingerprint:
            self.fingerprint = graph_fingerprint(self.graph)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "material_id": self.material_id,
            "kind": self.kind,
            "visibility": self.visibility,
            "version": self.version,
            "label": self.label,
            "fingerprint": self.fingerprint,
            "triples": len(self.graph),
            "registered_at": self.registered_at.astimezone(timezone.utc).isoformat(),
        }


class MaterialManifest:
    """The complete set of RDF materials one tenant is permitted to use."""

    def __init__(self, tenant_id: str) -> None:
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise TenantValidationError("tenant_id must be a non-empty string")
        self.tenant_id = tenant_id
        self._materials: Dict[str, TenantMaterial] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def _register(
        self,
        material_id: str,
        kind: str,
        visibility: str,
        source: GraphSource,
        version: Optional[str] = None,
        label: Optional[str] = None,
        rdf_format: Optional[str] = None,
    ) -> TenantMaterial:
        if not isinstance(material_id, str) or not material_id.strip():
            raise TenantValidationError("material_id must be a non-empty string")
        with self._lock:
            if material_id in self._materials:
                raise TenantValidationError(
                    "Material {!r} is already registered for tenant {!r}.".format(
                        material_id, self.tenant_id
                    )
                )
            graph = load_material_graph(source, rdf_format)
            material = TenantMaterial(
                material_id=material_id,
                kind=kind,
                visibility=visibility,
                graph=graph,
                version=version,
                label=label,
                rdf_format=rdf_format,
            )
            self._materials[material_id] = material
            return material

    def add_data_graph(
        self,
        material_id: str,
        source: GraphSource,
        *,
        visibility: str = VISIBILITY_TENANT,
        label: Optional[str] = None,
        rdf_format: Optional[str] = None,
    ) -> TenantMaterial:
        """Register a data graph this tenant is allowed to validate against."""
        return self._register(material_id, KIND_DATA, visibility, source, label=label, rdf_format=rdf_format)

    def add_public_shapes(
        self,
        material_id: str,
        source: GraphSource,
        *,
        version: Optional[str] = None,
        label: Optional[str] = None,
        rdf_format: Optional[str] = None,
    ) -> TenantMaterial:
        """Register a shape from the shared, public shape library."""
        return self._register(
            material_id, KIND_SHAPES, VISIBILITY_PUBLIC, source, version=version,
            label=label, rdf_format=rdf_format,
        )

    def add_tenant_shapes(
        self,
        material_id: str,
        source: GraphSource,
        *,
        version: Optional[str] = None,
        label: Optional[str] = None,
        rdf_format: Optional[str] = None,
    ) -> TenantMaterial:
        """Register a shape that belongs to this tenant's private shape library."""
        return self._register(
            material_id, KIND_SHAPES, VISIBILITY_TENANT, source, version=version,
            label=label, rdf_format=rdf_format,
        )

    def add_shared_dictionary(
        self,
        material_id: str,
        source: GraphSource,
        version: str,
        *,
        label: Optional[str] = None,
        rdf_format: Optional[str] = None,
    ) -> TenantMaterial:
        """Register a shared dictionary at an explicit, pinned version."""
        if not isinstance(version, str) or not version.strip():
            raise TenantValidationError("Shared dictionary {!r} requires a non-empty version.".format(material_id))
        return self._register(
            material_id, KIND_DICTIONARY, VISIBILITY_PUBLIC, source, version=version,
            label=label, rdf_format=rdf_format,
        )

    def add_tenant_dictionary(
        self,
        material_id: str,
        source: GraphSource,
        *,
        version: Optional[str] = None,
        label: Optional[str] = None,
        rdf_format: Optional[str] = None,
    ) -> TenantMaterial:
        """Register a tenant-private dictionary (for example a local code list)."""
        return self._register(
            material_id, KIND_DICTIONARY, VISIBILITY_TENANT, source, version=version,
            label=label, rdf_format=rdf_format,
        )

    def add_ontology(
        self,
        material_id: str,
        source: GraphSource,
        *,
        visibility: str = VISIBILITY_PUBLIC,
        version: Optional[str] = None,
        label: Optional[str] = None,
        rdf_format: Optional[str] = None,
    ) -> TenantMaterial:
        """Register an ontology document that may be mixed in during a run."""
        return self._register(
            material_id, KIND_ONTOLOGY, visibility, source, version=version,
            label=label, rdf_format=rdf_format,
        )

    # ------------------------------------------------------------------
    # Authorization / lookup
    # ------------------------------------------------------------------
    def get_material(self, material_id: str) -> TenantMaterial:
        material = self._materials.get(material_id)
        if material is None:
            raise TenantScopeError(
                "Tenant {!r} is not authorized to use material {!r}.".format(self.tenant_id, material_id),
                tenant_id=self.tenant_id,
                reference=material_id,
            )
        return material

    def require_material(self, material_id: str, kind: str) -> TenantMaterial:
        material = self.get_material(material_id)
        if material.kind != kind:
            raise TenantScopeError(
                "Material {!r} has kind {!r}, expected {!r}.".format(material_id, kind, material.kind),
                tenant_id=self.tenant_id,
                reference=material_id,
            )
        return material

    def material_ids(
        self, kind: Optional[str] = None, visibility: Optional[str] = None
    ) -> List[str]:
        with self._lock:
            return [
                mid
                for mid, material in self._materials.items()
                if (kind is None or material.kind == kind)
                and (visibility is None or material.visibility == visibility)
            ]

    def __contains__(self, material_id: object) -> bool:
        return isinstance(material_id, str) and material_id in self._materials

    def __len__(self) -> int:
        return len(self._materials)

    def fingerprint_snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Fingerprint and version of every registered material, keyed by id."""
        with self._lock:
            return {mid: material.snapshot() for mid, material in self._materials.items()}


@dataclass
class TenantValidationSpec:
    """A tenant-scoped validation request.

    Every referenced id must exist in the tenant's :class:`MaterialManifest`;
    the runner rejects unknown ids before any graph is touched.

    * ``data_graph_ids``: visible graphs to validate, combined into one run.
    * ``shapes_ids``: authorized shapes to apply; ``None`` means every shape
      registered on the manifest (public + tenant-specific).
    * ``focus_nodes``: explicit focus nodes, restricting target discovery.
      The same URI in two tenants is still resolved only against that
      tenant's own cloned graphs; it never implies a shared entity.
    * ``dictionary_ids``: dictionaries mixed in as ontology terms; ``None``
      means every registered dictionary.
    * ``dictionary_versions``: optional required version per dictionary id.
    """

    data_graph_ids: List[str]
    shapes_ids: Optional[List[str]] = None
    focus_nodes: Optional[List[Union[str, URIRef]]] = None
    dictionary_ids: Optional[List[str]] = None
    ontology_ids: List[str] = field(default_factory=list)
    dictionary_versions: Dict[str, str] = field(default_factory=dict)
    inference: str = "none"
    advanced: bool = False
    abort_on_first: bool = False
    allow_infos: bool = False
    allow_warnings: bool = False
    max_validation_depth: Optional[int] = None
    label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_graph_ids": list(self.data_graph_ids),
            "shapes_ids": None if self.shapes_ids is None else list(self.shapes_ids),
            "focus_nodes": None if self.focus_nodes is None else [str(n) for n in self.focus_nodes],
            "dictionary_ids": None if self.dictionary_ids is None else list(self.dictionary_ids),
            "ontology_ids": list(self.ontology_ids),
            "dictionary_versions": dict(self.dictionary_versions),
            "inference": self.inference,
            "advanced": self.advanced,
            "abort_on_first": self.abort_on_first,
            "allow_infos": self.allow_infos,
            "allow_warnings": self.allow_warnings,
            "max_validation_depth": self.max_validation_depth,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TenantValidationSpec":
        focus = data.get("focus_nodes")
        return cls(
            data_graph_ids=list(data["data_graph_ids"]),
            shapes_ids=list(data["shapes_ids"]) if data.get("shapes_ids") is not None else None,
            focus_nodes=[URIRef(n) for n in focus] if focus is not None else None,
            dictionary_ids=list(data["dictionary_ids"]) if data.get("dictionary_ids") is not None else None,
            ontology_ids=list(data.get("ontology_ids", [])),
            dictionary_versions=dict(data.get("dictionary_versions", {})),
            inference=data.get("inference", "none"),
            advanced=bool(data.get("advanced", False)),
            abort_on_first=bool(data.get("abort_on_first", False)),
            allow_infos=bool(data.get("allow_infos", False)),
            allow_warnings=bool(data.get("allow_warnings", False)),
            max_validation_depth=data.get("max_validation_depth"),
            label=data.get("label"),
        )


@dataclass
class TenantValidationReport:
    """Self-contained record of one tenant validation run.

    Stored alongside the conformance result are the full spec and the
    fingerprints/versions of every material visible to the tenant. That is
    enough to reproduce the run from the tenant's authorized materials alone,
    without consulting any other tenant's state.
    """

    tenant_id: str
    run_id: str
    conforms: bool
    spec: Dict[str, Any]
    material_snapshot: Dict[str, Dict[str, Any]]
    used_material_ids: Dict[str, List[str]]
    focus_node_visibility: Dict[str, List[str]]
    input_fingerprints: Dict[str, Optional[str]]
    report_graph_ttl: str
    report_text: str
    inference: str
    pyshacl_version: str
    started_at: str
    finished_at: str
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "run_id": self.run_id,
            "conforms": self.conforms,
            "spec": self.spec,
            "material_snapshot": self.material_snapshot,
            "used_material_ids": self.used_material_ids,
            "focus_node_visibility": self.focus_node_visibility,
            "input_fingerprints": self.input_fingerprints,
            "report_graph_ttl": self.report_graph_ttl,
            "report_text": self.report_text,
            "inference": self.inference,
            "pyshacl_version": self.pyshacl_version,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TenantValidationReport":
        return cls(**data)
