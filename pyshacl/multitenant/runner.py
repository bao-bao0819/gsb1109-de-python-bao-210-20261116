# -*- coding: utf-8 -*-
"""Tenant-scoped, concurrency-safe validation runner.

Isolation strategy
------------------
Every :meth:`TenantValidationRunner.validate` call:

1. Resolves every material id against the tenant's own
   :class:`~pyshacl.multitenant.manifest.MaterialManifest` and rejects anything
   the tenant is not authorized to see before a single triple is copied.
2. Builds a fresh clone of every authorized material. The manifest's source
   graphs, the clones, the SHACL shape cache and the inference closure are all
   private to that run; neighbouring tasks (concurrent or previously failed)
   cannot share shapes or inference state.
3. Restricts target discovery to focus nodes that actually occur in the
   tenant's visible graphs. The same focus URI used by another tenant is
   resolved only against this tenant's clones, so it never implies the same
   entity across tenants.
4. Records the full spec plus a fingerprint/version snapshot of every
   authorized material, so the report can be replayed from the tenant's
   materials alone.
"""
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from rdflib import Graph, URIRef

from ..rdfutil import clone_graph
from .errors import TenantScopeError, TenantValidationError
from .manifest import (
    KIND_DATA,
    KIND_DICTIONARY,
    KIND_ONTOLOGY,
    KIND_SHAPES,
    MaterialManifest,
    TenantMaterial,
    TenantValidationReport,
    TenantValidationSpec,
    graph_fingerprint,
)

# rdflib/pySHACL keep a small amount of process-global state (for example the
# NORMALIZE_LITERALS toggle flipped while parsing a shapes graph). Graph, shape
# and inference state are all per-run, but this lock makes the short core
# validation call deterministic when tasks from different tenants run in the
# same process. All material preparation (cloning, fingerprinting, merging)
# stays outside the lock and runs fully concurrently.
_CORE_VALIDATION_LOCK = threading.RLock()


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pyshacl_version() -> str:
    from .. import __version__

    return __version__


class TenantValidationRunner:
    """Runs validation tasks for exactly one tenant manifest."""

    def __init__(self, manifest: MaterialManifest, lock_core_validation: bool = True) -> None:
        if not isinstance(manifest, MaterialManifest):
            raise TenantValidationError("TenantValidationRunner requires a MaterialManifest instance.")
        self.manifest = manifest
        self.tenant_id = manifest.tenant_id
        # The lock only guards pySHACL's short process-global sections; graph
        # preparation always stays concurrent. Set False when the caller
        # already serializes validation across the process.
        self.lock_core_validation = bool(lock_core_validation)

    # ------------------------------------------------------------------
    # Authorization
    # ------------------------------------------------------------------
    def _resolve_ids(self, spec: TenantValidationSpec) -> Dict[str, List[TenantMaterial]]:
        if not spec.data_graph_ids:
            raise TenantValidationError("A validation spec must name at least one data graph.")
        data_materials = [
            self.manifest.require_material(mid, KIND_DATA) for mid in self._ordered(spec.data_graph_ids)
        ]
        shape_ids = spec.shapes_ids
        if shape_ids is None:
            shape_ids = self.manifest.material_ids(kind=KIND_SHAPES)
        if not shape_ids:
            raise TenantValidationError(
                "Tenant {!r} has no authorized shapes for this run.".format(self.tenant_id)
            )
        shape_materials = [self.manifest.require_material(mid, KIND_SHAPES) for mid in self._ordered(shape_ids)]
        dictionary_ids = spec.dictionary_ids
        if dictionary_ids is None:
            dictionary_ids = self.manifest.material_ids(kind=KIND_DICTIONARY)
        dictionary_materials: List[TenantMaterial] = []
        for mid in self._ordered(dictionary_ids):
            material = self.manifest.require_material(mid, KIND_DICTIONARY)
            required_version = spec.dictionary_versions.get(mid)
            if required_version is not None and material.version != required_version:
                raise TenantValidationError(
                    "Dictionary {!r} is registered at version {!r} but the spec pins version {!r}.".format(
                        mid, material.version, required_version
                    )
                )
            dictionary_materials.append(material)
        ontology_materials = [
            self.manifest.require_material(mid, KIND_ONTOLOGY) for mid in self._ordered(spec.ontology_ids)
        ]
        return {
            "data": data_materials,
            "shapes": shape_materials,
            "dictionaries": dictionary_materials,
            "ontologies": ontology_materials,
        }

    @staticmethod
    def _ordered(ids: List[str]) -> List[str]:
        # De-duplicate while giving every run a deterministic merge order.
        return sorted(set(ids))

    # ------------------------------------------------------------------
    # Per-run graph construction (fresh, disjoint clones)
    # ------------------------------------------------------------------
    @staticmethod
    def _clone_materials(materials: List[TenantMaterial]) -> Dict[str, Graph]:
        return {material.material_id: clone_graph(material.graph) for material in materials}

    @staticmethod
    def _merge_clones(clones: Dict[str, Graph], ordered_ids: List[str]) -> Graph:
        merged = Graph(bind_namespaces="core")
        for material_id in ordered_ids:
            clone = clones[material_id]
            for prefix, namespace in clone.namespace_manager.namespaces():
                merged.namespace_manager.bind(prefix, namespace, override=False)
            for triple in clone:
                merged.add(triple)
        return merged

    def _check_focus_visibility(
        self,
        focus_nodes: Optional[List[Union[str, URIRef]]],
        data_clones: Dict[str, Graph],
        ordered_data_ids: List[str],
    ) -> Tuple[Optional[List[URIRef]], Dict[str, List[str]]]:
        if not focus_nodes:
            return None, {}
        focus_refs: List[URIRef] = []
        visibility: Dict[str, List[str]] = {}
        for node in focus_nodes:
            focus_ref = node if isinstance(node, URIRef) else URIRef(str(node))
            focus_refs.append(focus_ref)
            containing = [
                material_id
                for material_id in ordered_data_ids
                if any(data_clones[material_id].triples((focus_ref, None, None)))
                or any(data_clones[material_id].triples((None, None, focus_ref)))
            ]
            if not containing:
                raise TenantScopeError(
                    "Focus node {} does not occur in any data graph visible to tenant {!r}.".format(
                        focus_ref, self.tenant_id
                    ),
                    tenant_id=self.tenant_id,
                    reference=str(focus_ref),
                )
            visibility[str(focus_ref)] = containing
        return focus_refs, visibility

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def validate(self, spec: TenantValidationSpec) -> TenantValidationReport:
        if not isinstance(spec, TenantValidationSpec):
            raise TenantValidationError("validate() expects a TenantValidationSpec instance.")
        run_id = str(uuid.uuid4())
        started_at = _utcnow_iso()
        resolved = self._resolve_ids(spec)

        data_ids = [m.material_id for m in resolved["data"]]
        shape_ids = [m.material_id for m in resolved["shapes"]]
        dictionary_ids = [m.material_id for m in resolved["dictionaries"]]
        ontology_ids = [m.material_id for m in resolved["ontologies"]]

        # Every graph below is a private, per-run clone. Nothing here is ever
        # written back to the manifest or shared with another run.
        data_clones = self._clone_materials(resolved["data"])
        shape_clones = self._clone_materials(resolved["shapes"])
        dictionary_clones = self._clone_materials(resolved["dictionaries"])
        ontology_clones = self._clone_materials(resolved["ontologies"])

        data_graph = self._merge_clones(data_clones, data_ids)
        shapes_graph = self._merge_clones(shape_clones, shape_ids)
        ontology_inputs = dict(dictionary_clones)
        ontology_inputs.update(ontology_clones)
        ontology_order = dictionary_ids + ontology_ids
        ontology_graph = self._merge_clones(ontology_inputs, ontology_order) if ontology_order else None

        focus_refs, focus_visibility = self._check_focus_visibility(
            spec.focus_nodes, data_clones, data_ids
        )

        input_fingerprints = {
            "data_graph": graph_fingerprint(data_graph),
            "shapes_graph": graph_fingerprint(shapes_graph),
            "ontology_graph": graph_fingerprint(ontology_graph) if ontology_graph is not None else None,
        }
        material_snapshot = self.manifest.fingerprint_snapshot()

        from .. import validate as pyshacl_validate

        pyshacl_version = _pyshacl_version()

        options: Dict[str, Any] = {
            "inference": spec.inference,
            "advanced": spec.advanced,
            "abort_on_first": spec.abort_on_first,
            "allow_infos": spec.allow_infos,
            "allow_warnings": spec.allow_warnings,
            "focus_nodes": focus_refs,
            "inplace": False,
        }
        if spec.max_validation_depth is not None:
            options["max_validation_depth"] = spec.max_validation_depth

        error: Optional[str] = None
        conforms = False
        report_graph_ttl = ""
        report_text = ""
        try:
            if self.lock_core_validation:
                with _CORE_VALIDATION_LOCK:
                    conforms, report_graph, report_text = pyshacl_validate(
                        data_graph,
                        shacl_graph=shapes_graph,
                        ont_graph=ontology_graph,
                        **options,
                    )
            else:
                conforms, report_graph, report_text = pyshacl_validate(
                    data_graph,
                    shacl_graph=shapes_graph,
                    ont_graph=ontology_graph,
                    **options,
                )
            report_graph_ttl = report_graph.serialize(format="turtle")
            if isinstance(report_graph_ttl, bytes):
                report_graph_ttl = report_graph_ttl.decode("utf-8")
        except Exception as exc:  # noqa: BLE001 - recorded on the report, no state leaks out
            error = "{}: {}".format(type(exc).__name__, exc)
        finished_at = _utcnow_iso()

        return TenantValidationReport(
            tenant_id=self.tenant_id,
            run_id=run_id,
            conforms=bool(conforms) and error is None,
            spec=spec.to_dict(),
            material_snapshot=material_snapshot,
            used_material_ids={
                "data": data_ids,
                "shapes": shape_ids,
                "dictionaries": dictionary_ids,
                "ontologies": ontology_ids,
            },
            focus_node_visibility=focus_visibility,
            input_fingerprints=input_fingerprints,
            report_graph_ttl=report_graph_ttl,
            report_text=report_text,
            inference=spec.inference,
            pyshacl_version=pyshacl_version,
            started_at=started_at,
            finished_at=finished_at,
            error=error,
        )

    def replay(
        self, report: TenantValidationReport, verify_fingerprints: bool = True
    ) -> TenantValidationReport:
        """Re-run a previous report from the tenant's authorized materials.

        When ``verify_fingerprints`` is set, the manifest materials referenced
        by the original report must still carry the same fingerprints and
        versions, otherwise :class:`TenantValidationError` is raised. This is
        what makes a tenant report reproducible using only its own materials:
        no other tenant's graphs, shapes or inference state are consulted.
        """
        if report.tenant_id != self.tenant_id:
            raise TenantScopeError(
                "Report belongs to tenant {!r}, not {!r}.".format(report.tenant_id, self.tenant_id),
                tenant_id=self.tenant_id,
                reference=report.run_id,
            )
        if verify_fingerprints:
            current = self.manifest.fingerprint_snapshot()
            for material_id, original in report.material_snapshot.items():
                present = current.get(material_id)
                if present is None:
                    raise TenantValidationError(
                        "Material {!r} used by report {!r} is no longer registered.".format(
                            material_id, report.run_id
                        )
                    )
                if present["fingerprint"] != original["fingerprint"]:
                    raise TenantValidationError(
                        "Material {!r} has changed since report {!r}; the run is no longer reproducible.".format(
                            material_id, report.run_id
                        )
                    )
                if present["version"] != original["version"]:
                    raise TenantValidationError(
                        "Material {!r} version changed since report {!r} ({} -> {}).".format(
                            material_id,
                            report.run_id,
                            original["version"],
                            present["version"],
                        )
                    )
        spec = TenantValidationSpec.from_dict(report.spec)
        # The fingerprint/version loop above is the reproducibility guarantee;
        # audit metadata such as registration timestamps is intentionally not
        # part of it, so a rebuilt manifest can still replay an old report.
        return self.validate(spec)

    def validate_batch(
        self,
        specs: Sequence[TenantValidationSpec],
        *,
        max_workers: Optional[int] = None,
        fail_fast: bool = False,
    ) -> List[TenantValidationReport]:
        """Validate a batch of specs for this tenant, one isolated run per spec.

        Each spec still gets its own fresh clones and its own
        :class:`TenantValidationReport`; batching only controls scheduling.
        When ``max_workers`` is greater than one, specs run in a worker pool,
        and the per-run core validation lock keeps the process-global parts of
        pySHACL deterministic while graph preparation stays concurrent.

        ``fail_fast`` does not share state between tasks: an authorization
        failure on one spec is recorded on that task's report only and, when
        set, stops scheduling the remaining specs. Already running tasks are
        left to finish, so their partial state never bleeds into another task.
        """
        if not isinstance(specs, (list, tuple)):
            raise TenantValidationError("validate_batch expects a list or tuple of TenantValidationSpec.")
        spec_list = list(specs)
        if not spec_list:
            return []
        for spec in spec_list:
            if not isinstance(spec, TenantValidationSpec):
                raise TenantValidationError("Every batch item must be a TenantValidationSpec instance.")

        reports: List[TenantValidationReport] = []
        if max_workers is None or max_workers <= 1:
            for spec in spec_list:
                reports.append(self._run_batch_item(spec))
                if fail_fast and reports[-1].error is not None:
                    break
            return reports

        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tenant-validate") as executor:
            # map() yields results in submission order, even when tasks finish
            # out of order on the worker pool.
            for report in executor.map(self._run_batch_item, spec_list):
                reports.append(report)
                if fail_fast and report.error is not None:
                    break
        return reports

    def _run_batch_item(self, spec: TenantValidationSpec) -> TenantValidationReport:
        try:
            return self.validate(spec)
        except TenantValidationError as exc:
            # Authorization/scope failures belong to this task only; record them
            # instead of letting them tear down neighbouring tasks.
            return TenantValidationReport(
                tenant_id=self.tenant_id,
                run_id=str(uuid.uuid4()),
                conforms=False,
                spec=spec.to_dict(),
                material_snapshot=self.manifest.fingerprint_snapshot(),
                used_material_ids={"data": [], "shapes": [], "dictionaries": [], "ontologies": []},
                focus_node_visibility={},
                input_fingerprints={"data_graph": None, "shapes_graph": None, "ontology_graph": None},
                report_graph_ttl="",
                report_text="",
                inference=spec.inference,
                pyshacl_version=_pyshacl_version(),
                started_at=_utcnow_iso(),
                finished_at=_utcnow_iso(),
                error="{}: {}".format(type(exc).__name__, exc),
            )
