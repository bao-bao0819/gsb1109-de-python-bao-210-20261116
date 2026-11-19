# -*- coding: utf-8 -*-
"""
Cross-shard validation of ``sh:minCount``, ``sh:maxCount`` and
``sh:uniqueLang`` on direct property paths.

Entity data is split over several *shards*. Each shard may individually
satisfy a cardinality constraint while the union of all shards violates it
(two shards each contributing one value to a ``sh:maxCount 1`` property).
This module validates the *merged* view of all shard versions without ever
assembling the merged graph:

1. Every shard summarises its slice as focus-node identities and per
   (predicate, focus) value *digests* (see :mod:`pyshacl.sharded.identity`).
   Digests are computed from precise RDF term identities, so the same value
   reported by two shards (or re-sent by the same shard) is counted once.
2. The coordinator merges the summaries and evaluates the supported
   constraints on the merged counts and language tags.
3. Only when a violation is found does the coordinator fetch the actual,
   reviewable values from the shards that contributed them, and builds the
   validation report with pySHACL's own constraint components, so the
   results are identical to validating the graph composed from the same set
   of source versions.

Versioning: contributions are keyed by shard id. Re-sending the current
version of a shard does not change the counts, publishing a new version
replaces the old contribution, and retracting a shard withdraws it.

Shapes that cannot be faithfully evaluated this way (anything beyond
``sh:minCount``/``sh:maxCount``/``sh:uniqueLang`` on direct predicate paths,
non-trivial targets, custom constraint components, ...) raise
:class:`pyshacl.errors.UnsupportedShapeError` instead of silently passing.
"""

from collections import defaultdict

from typing import Dict, List, Optional, Set, Tuple

import rdflib

from rdflib import BNode, Graph, URIRef

from pyshacl.constraints import ALL_CONSTRAINT_PARAMETERS
from pyshacl.constraints.core.cardinality_constraints import (
    MaxCountConstraintComponent,
    MinCountConstraintComponent,
)
from pyshacl.constraints.core.string_based_constraints import UniqueLangConstraintComponent
from pyshacl.consts import SH
from pyshacl.errors import ReportableRuntimeError, UnsupportedShapeError
from pyshacl.shape import Shape
from pyshacl.sharded.identity import (
    SHAPES_GRAPH_SOURCE,
    BNodeConvention,
    default_bnode_convention,
    term_from_identity,
    term_identity,
)
from pyshacl.sharded.shard import LocalShard, ShardProtocol
from pyshacl.shapes_graph import ShapesGraph
from pyshacl.validator import Validator

SH_minCount = SH.minCount
SH_maxCount = SH.maxCount
SH_uniqueLang = SH.uniqueLang
SH_property = SH.property

_SUPPORTED_PROPERTY_CONSTRAINTS = frozenset({SH_minCount, SH_maxCount, SH_uniqueLang})
_ALL_CONSTRAINT_PARAMETERS = frozenset(ALL_CONSTRAINT_PARAMETERS)

_KIND_MIN_COUNT = 'minCount'
_KIND_MAX_COUNT = 'maxCount'
_KIND_UNIQUE_LANG = 'uniqueLang'


class ViolationEvidence:
    """Reviewable, per-source evidence for one cross-shard violation.

    ``values_by_shard`` holds the actual value terms fetched from each
    contributing shard *after* the violation was detected; blank nodes are
    canonicalised according to the coordinator's blank node convention.
    """

    __slots__ = (
        'source_shape',
        'focus_node',
        'result_path',
        'constraint_component',
        'kind',
        'lang',
        'values_by_shard',
        'shard_versions',
    )

    def __init__(
        self,
        source_shape,
        focus_node,
        result_path,
        constraint_component,
        kind,
        lang,
        values_by_shard,
        shard_versions,
    ):
        self.source_shape = source_shape
        self.focus_node = focus_node
        self.result_path = result_path
        self.constraint_component = constraint_component
        self.kind = kind
        self.lang = lang
        self.values_by_shard = values_by_shard
        self.shard_versions = shard_versions

    def __repr__(self):
        return "<ViolationEvidence {} on {}->{} values_from={}>".format(
            self.kind, self.focus_node, self.result_path, sorted(self.values_by_shard)
        )


class ShardedValidationResult:
    """Result of :meth:`ShardCoordinator.validate`.

    Mirrors pySHACL's ``(conforms, report_graph, report_text)`` triple and
    additionally carries the per-violation fetchable ``evidence``.
    """

    __slots__ = ('conforms', 'report_graph', 'report_text', 'violations')

    def __init__(self, conforms: bool, report_graph: Graph, report_text: str, violations: Tuple[ViolationEvidence, ...]):
        self.conforms = conforms
        self.report_graph = report_graph
        self.report_text = report_text
        self.violations = violations

    def __iter__(self):
        # pySHACL-style unpacking: conforms, results_graph, results_text
        yield self.conforms
        yield self.report_graph
        yield self.report_text

    def __bool__(self):
        return bool(self.conforms)


class _TargetSpec:
    __slots__ = ('target_nodes', 'target_classes', 'subjects_of', 'objects_of')

    def __init__(self, target_nodes, target_classes, subjects_of, objects_of):
        self.target_nodes = target_nodes
        self.target_classes = target_classes
        self.subjects_of = subjects_of
        self.objects_of = objects_of

    def has_targets(self) -> bool:
        return bool(self.target_nodes or self.target_classes or self.subjects_of or self.objects_of)


class _PropertyShapePlan:
    __slots__ = ('shape', 'predicate', 'min_count', 'max_count', 'unique_lang', 'unique_lang_value')

    def __init__(self, shape: Shape, predicate: URIRef, min_count, max_count, unique_lang):
        self.shape = shape
        self.predicate = predicate
        self.min_count = min_count
        self.max_count = max_count
        self.unique_lang = unique_lang
        self.unique_lang_value = bool(next(iter(unique_lang.string_rules))) if unique_lang is not None else False

    def component_for(self, kind: str):
        if kind == _KIND_MIN_COUNT:
            return self.min_count
        if kind == _KIND_MAX_COUNT:
            return self.max_count
        if kind == _KIND_UNIQUE_LANG:
            return self.unique_lang
        raise KeyError(kind)


class _Violation:
    __slots__ = ('plan', 'focus', 'kind', 'lang')

    def __init__(self, plan: _PropertyShapePlan, focus: str, kind: str, lang: Optional[str]):
        self.plan = plan
        self.focus = focus
        self.kind = kind
        self.lang = lang


class ShardCoordinator:
    """Coordinates cross-shard validation for one shapes graph.

    :param shacl_graph: the SHACL shapes graph (rdflib graph or serialised
        string, turtle by default).
    :param bnode_convention: callable mapping ``(source_id, local_label)`` to
        a global blank node identity. ``source_id`` is the shard id, or
        ``None`` for terms originating in the shapes graph. The default
        scopes blank nodes to their source shard.
    """

    def __init__(
        self,
        shacl_graph,
        shacl_graph_format: Optional[str] = None,
        bnode_convention: Optional[BNodeConvention] = None,
        logger=None,
    ) -> None:
        if isinstance(shacl_graph, (rdflib.Graph, rdflib.Dataset)):
            sg_graph = shacl_graph
        elif isinstance(shacl_graph, (str, bytes)):
            sg_graph = Graph()
            sg_graph.parse(data=shacl_graph, format=shacl_graph_format or "turtle")
        else:
            raise TypeError("shacl_graph must be an rdflib Graph or a serialised RDF string.")
        self._convention: BNodeConvention = bnode_convention or default_bnode_convention
        self.sg = ShapesGraph(sg_graph, logger=logger)
        self._shards: Dict[str, ShardProtocol] = {}
        self._contexts: List[Tuple[_PropertyShapePlan, str]] = []
        self._target_specs: Dict[rdflib.term.Node, _TargetSpec] = {}
        self._needs_class_closure = False
        self._subjects_of_predicates: Set[str] = set()
        self._objects_of_predicates: Set[str] = set()
        self._canonical_bnodes: Dict[str, BNode] = {}
        self._build_plan()

    # ------------------------------------------------------------------
    # shard management
    # ------------------------------------------------------------------
    @property
    def shard_versions(self) -> Dict[str, str]:
        """Currently registered shard ids and their versions."""
        return {sid: shard.version for sid, shard in self._shards.items()}

    def update_shard(self, shard_id: str, version: str, data, data_format: Optional[str] = None) -> LocalShard:
        """Publish ``version`` of shard ``shard_id``.

        Contributions are keyed by shard id: re-sending the same version
        simply replaces the stored slice (counts are unaffected), and a new
        version supersedes the old contribution entirely.
        """
        shard = LocalShard.from_data(
            shard_id, version, data, data_format=data_format, bnode_convention=self._convention
        )
        self.add_shard(shard)
        return shard

    def add_shard(self, shard: ShardProtocol) -> None:
        """Register (or replace) a shard implementing the shard protocol."""
        self._shards[shard.shard_id] = shard

    def retract_shard(self, shard_id: str) -> None:
        """Withdraw a shard's contribution entirely."""
        try:
            del self._shards[shard_id]
        except KeyError:
            raise ReportableRuntimeError("Cannot retract unknown shard '{}'.".format(shard_id))

    # ------------------------------------------------------------------
    # shapes analysis
    # ------------------------------------------------------------------
    def _constraint_predicates(self, shape: Shape) -> Set:
        return {p for p in self.sg.graph.predicates(shape.node) if p in _ALL_CONSTRAINT_PARAMETERS}

    def _target_spec(self, shape: Shape) -> _TargetSpec:
        target_nodes = {
            term_identity(t, SHAPES_GRAPH_SOURCE, self._convention) for t in shape.target_nodes()
        }
        target_classes = {
            term_identity(t, SHAPES_GRAPH_SOURCE, self._convention)
            for t in list(shape.target_classes()) + list(shape.implicit_class_targets())
        }
        subjects_of = {str(p) for p in shape.target_subjects_of()}
        objects_of = {str(p) for p in shape.target_objects_of()}
        return _TargetSpec(target_nodes, target_classes, subjects_of, objects_of)

    def _build_property_plan(self, shape: Shape) -> _PropertyShapePlan:
        path = shape.path()
        if not isinstance(path, URIRef):
            raise UnsupportedShapeError(
                "Sharded validation only supports direct predicate paths; "
                "shape {} uses a complex sh:path.".format(shape.node)
            )
        unsupported = self._constraint_predicates(shape) - _SUPPORTED_PROPERTY_CONSTRAINTS
        if unsupported:
            raise UnsupportedShapeError(
                "Sharded validation only supports sh:minCount, sh:maxCount and sh:uniqueLang "
                "on property shapes; shape {} also uses: {}.".format(
                    shape.node, ", ".join(sorted(str(p) for p in unsupported))
                )
            )
        predicates = set(self.sg.graph.predicates(shape.node))
        min_count = MinCountConstraintComponent(shape) if SH_minCount in predicates else None
        max_count = MaxCountConstraintComponent(shape) if SH_maxCount in predicates else None
        unique_lang = UniqueLangConstraintComponent(shape) if SH_uniqueLang in predicates else None
        return _PropertyShapePlan(shape, path, min_count, max_count, unique_lang)

    def _build_plan(self) -> None:
        sg = self.sg
        if sg.custom_constraints:
            raise UnsupportedShapeError(
                "Sharded validation does not support custom SHACL constraint components."
            )
        prop_plans: Dict[rdflib.term.Node, _PropertyShapePlan] = {}
        node_shape_refs: List[Tuple[Shape, list]] = []
        for shape in sg.shapes:
            if shape.deactivated:
                continue
            spec = self._target_spec(shape)
            if shape.is_property_shape:
                plan = self._build_property_plan(shape)
                prop_plans[shape.node] = plan
                self._target_specs[shape.node] = spec
                if spec.has_targets():
                    self._contexts.append((plan, shape.node))
            else:
                unsupported = self._constraint_predicates(shape) - {SH_property}
                if unsupported:
                    raise UnsupportedShapeError(
                        "Sharded validation supports only sh:property references on node shapes; "
                        "shape {} also uses: {}.".format(
                            shape.node, ", ".join(sorted(str(p) for p in unsupported))
                        )
                    )
                self._target_specs[shape.node] = spec
                refs = list(sg.graph.objects(shape.node, SH_property))
                node_shape_refs.append((shape, refs))
        for shape, refs in node_shape_refs:
            for ref in refs:
                try:
                    pshape = sg.lookup_shape_from_node(ref)
                except (KeyError, AttributeError):
                    raise UnsupportedShapeError(
                        "sh:property on shape {} references an unknown shape {}.".format(shape.node, ref)
                    )
                if pshape.deactivated:
                    continue
                if not pshape.is_property_shape:
                    raise UnsupportedShapeError(
                        "sh:property on shape {} references {}, which is not a property shape.".format(
                            shape.node, ref
                        )
                    )
                self._contexts.append((prop_plans[ref], shape.node))
        for spec in self._target_specs.values():
            if spec.target_classes:
                self._needs_class_closure = True
            self._subjects_of_predicates |= spec.subjects_of
            self._objects_of_predicates |= spec.objects_of

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    def validate(self) -> ShardedValidationResult:
        """Validate the merged view of all current shard versions."""
        self._canonical_bnodes = {}
        shards = self._shards
        # Round 1: merge the shards' rdfs:subClassOf pairs so targetClass
        # expansion matches pySHACL's in-data-graph subclass reasoning.
        subclass_pairs: Set[Tuple[str, str]] = set()
        if self._needs_class_closure:
            for shard in shards.values():
                subclass_pairs |= shard.subclass_pairs()
        children: Dict[str, Set[str]] = defaultdict(set)
        for sub, sup in subclass_pairs:
            children[sup].add(sub)

        def expand(class_identity: str) -> Set[str]:
            found = {class_identity}
            queue = [class_identity]
            while queue:
                current = queue.pop()
                for child in children.get(current, ()):
                    if child not in found:
                        found.add(child)
                        queue.append(child)
            return found

        expanded_classes: Dict[str, Set[str]] = {}
        all_class_identities: Set[str] = set()
        for spec in self._target_specs.values():
            for class_identity in spec.target_classes:
                if class_identity not in expanded_classes:
                    expanded_classes[class_identity] = expand(class_identity)
                all_class_identities |= expanded_classes[class_identity]

        # Round 2: resolve focus nodes per shape from shard summaries.
        members: Dict[str, Set[str]] = defaultdict(set)
        subjects: Dict[str, Set[str]] = defaultdict(set)
        objects: Dict[str, Set[str]] = defaultdict(set)
        focus_sources: Dict[str, Set[str]] = defaultdict(set)
        for shard_id, shard in shards.items():
            summary = shard.target_summary(
                all_class_identities, self._subjects_of_predicates, self._objects_of_predicates
            )
            for class_identity, keys in summary.members.items():
                members[class_identity] |= keys
                for key in keys:
                    focus_sources[key].add(shard_id)
            for predicate, keys in summary.subjects.items():
                subjects[predicate] |= keys
                for key in keys:
                    focus_sources[key].add(shard_id)
            for predicate, keys in summary.objects.items():
                objects[predicate] |= keys
                for key in keys:
                    focus_sources[key].add(shard_id)

        focus_by_shape: Dict[rdflib.term.Node, Set[str]] = {}
        for shape_node, spec in self._target_specs.items():
            keys = set(spec.target_nodes)
            for class_identity in spec.target_classes:
                for expanded in expanded_classes.get(class_identity, ()):
                    keys |= members.get(expanded, set())
            for predicate in spec.subjects_of:
                keys |= subjects.get(predicate, set())
            for predicate in spec.objects_of:
                keys |= objects.get(predicate, set())
            focus_by_shape[shape_node] = keys

        # Round 3: value digests per (predicate, focus).
        queries: Dict[str, Set[str]] = defaultdict(set)
        for plan, shape_node in self._contexts:
            for focus in focus_by_shape.get(shape_node, ()):
                queries[str(plan.predicate)].add(focus)
        values: Dict[Tuple[str, str], Set[Tuple[str, Optional[str]]]] = {}
        provenance: Dict[Tuple[str, str], Dict[str, Set[str]]] = {}
        for shard_id, shard in shards.items():
            value_summary = shard.value_summary(queries)
            for predicate, per_focus in value_summary.items():
                for focus, records in per_focus.items():
                    key = (predicate, focus)
                    merged_records = values.setdefault(key, set())
                    shard_digests = provenance.setdefault(key, {}).setdefault(shard_id, set())
                    for digest, lang in records:
                        merged_records.add((digest, lang))
                        shard_digests.add(digest)
                    focus_sources[focus].add(shard_id)

        # Evaluate the supported constraints on the merged summaries.
        violations: List[_Violation] = []
        for plan, shape_node in self._contexts:
            predicate = str(plan.predicate)
            for focus in focus_by_shape.get(shape_node, ()):
                records = values.get((predicate, focus), set())
                count = len({digest for digest, _lang in records})
                if plan.min_count is not None:
                    min_value = int(plan.min_count.min_count.value)
                    if min_value > 0 and count < min_value:
                        violations.append(_Violation(plan, focus, _KIND_MIN_COUNT, None))
                if plan.max_count is not None:
                    max_value = int(plan.max_count.max_count.value)
                    if count > max_value:
                        violations.append(_Violation(plan, focus, _KIND_MAX_COUNT, None))
                if plan.unique_lang is not None and plan.unique_lang_value:
                    lang_counts: Dict[str, int] = {}
                    for _digest, lang in records:
                        if lang:
                            lang_counts[lang] = lang_counts.get(lang, 0) + 1
                    for lang, lang_count in lang_counts.items():
                        if lang_count > 1:
                            violations.append(_Violation(plan, focus, _KIND_UNIQUE_LANG, lang))

        # Fetch phase: only for violations, pull reviewable values per source.
        results = []
        evidence: List[ViolationEvidence] = []
        if violations:
            evidence_graph = Graph()
            for shard in shards.values():
                for prefix, namespace in shard.namespaces():
                    evidence_graph.namespace_manager.bind(prefix, namespace, override=False, replace=False)
            fetched_values: Dict[Tuple[str, str], Dict[str, list]] = {}
            needed = {(str(v.plan.predicate), v.focus) for v in violations}
            for predicate, focus in needed:
                per_shard: Dict[str, list] = {}
                for shard_id, digests in provenance.get((predicate, focus), {}).items():
                    shard = shards[shard_id]
                    terms = shard.fetch_values(predicate, focus, digests)
                    per_shard[shard_id] = [self._canonicalize(shard, t) for t in terms]
                fetched_values[(predicate, focus)] = per_shard
            focus_terms: Dict[str, rdflib.term.Node] = {}
            for violation in violations:
                if violation.focus not in focus_terms:
                    focus_terms[violation.focus] = self._resolve_focus(violation.focus, shards, focus_sources)
            for (predicate, focus), per_shard in fetched_values.items():
                focus_term = focus_terms[focus]
                for terms in per_shard.values():
                    for term in terms:
                        evidence_graph.add((focus_term, URIRef(predicate), term))
            for violation in violations:
                plan = violation.plan
                predicate = str(plan.predicate)
                focus_term = focus_terms[violation.focus]
                per_shard = fetched_values[(predicate, violation.focus)]
                component = plan.component_for(violation.kind)
                desc, r_node, r_triples = component.make_v_result(evidence_graph, focus_term)
                results.append((desc, r_node, r_triples))
                evidence.append(
                    ViolationEvidence(
                        source_shape=plan.shape.node,
                        focus_node=focus_term,
                        result_path=plan.predicate,
                        constraint_component=component.shacl_constraint_component,
                        kind=violation.kind,
                        lang=violation.lang,
                        values_by_shard={sid: tuple(terms) for sid, terms in per_shard.items()},
                        shard_versions={sid: shards[sid].version for sid in per_shard},
                    )
                )
        conforms = len(results) == 0
        report_graph, report_text = Validator.create_validation_report(self.sg, conforms, results)
        return ShardedValidationResult(conforms, report_graph, report_text, tuple(evidence))

    # ------------------------------------------------------------------
    # term resolution helpers
    # ------------------------------------------------------------------
    def _canonical_bnode(self, global_identity: str) -> BNode:
        if global_identity not in self._canonical_bnodes:
            self._canonical_bnodes[global_identity] = BNode()
        return self._canonical_bnodes[global_identity]

    def _canonicalize(self, shard: ShardProtocol, term):
        if isinstance(term, BNode):
            return self._canonical_bnode(self._convention(shard.shard_id, str(term)))
        return term

    def _resolve_focus(self, focus_identity: str, shards, focus_sources):
        term = term_from_identity(focus_identity)
        if term is not None:
            return term
        for shard_id in focus_sources.get(focus_identity, ()):
            term = shards[shard_id].find_term(focus_identity)
            if term is not None:
                return self._canonicalize(shards[shard_id], term)
        # A blank node focus that no shard reported (e.g. a blank node in
        # sh:targetNode): represent it by its conventional global identity.
        return self._canonical_bnode(focus_identity[2:])
