# -*- coding: utf-8 -*-
"""Quarantine support built on top of SHACL validation.

The warehouse pipeline uses pySHACL results to decide which records go to the
quarantine area. This module adds two things on top of the unchanged SHACL
validation report:

1. ``evaluate_quarantine`` runs the same input record twice, the second time
   with only a chosen shape's *own* target declarations suspended. The shape
   is still evaluated when reached through shape-expecting constraints such
   as ``sh:node`` or ``sh:or``.
2. The resulting :class:`QuarantineAssessment` explains which root rules still
   reference the suspended shape on the second run, and whether its failure on
   each path causes the root rule to reject the record or is rescued by
   another branch (for example the alternative member of an ``sh:or``).

Every piece of evidence carries the root rule and the input record, so rule
change records can be traced back to concrete validation outcomes.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import RDF, SH, XSD

from pyshacl.entrypoints import validate
from pyshacl.pytypes import TargetSuspension, ValidationTrace

Q = URIRef("urn:pyshacl:quarantine#")
QUARANTINE_NS = Q

#: Causes rejection of the record on this root rule path.
EFFECT_CAUSES_REJECTION = "causes_rejection"
#: The shape fails, but another branch makes the whole root rule pass.
EFFECT_RESCUED_BY_BRANCH = "rescued_by_alternative_branch"
#: The shape conforms on this path, so its suspension cannot change the outcome.
EFFECT_NO_EFFECT = "no_effect"

_SHAPE_REF_PREDICATES = (SH.node, SH.property, SH["not"], SH.qualifiedValueShape)
_LIST_REF_PREDICATES = (SH["and"], SH["or"], SH.xone)
_TARGET_PREDICATES = (SH.targetNode, SH.targetClass, SH.targetObjectsOf, SH.targetSubjectsOf)


@dataclass(frozen=True)
class ReferenceHop:
    """One shape-expecting hop between two nested shapes."""

    parent_shape: Any
    predicate: URIRef
    child_shape: Any


@dataclass(frozen=True)
class ReferencePath:
    """A static path from a root rule to the suspended shape."""

    root_shape: Any
    hops: Tuple[ReferenceHop, ...]

    @property
    def target_shape(self) -> Any:
        return self.hops[-1].child_shape if self.hops else self.root_shape

    @property
    def last_predicate(self) -> Optional[URIRef]:
        return self.hops[-1].predicate if self.hops else None


@dataclass
class RunOutcome:
    """One validation run against one input record."""

    label: str
    conforms: bool
    report_graph: Graph
    trace: ValidationTrace
    suspended_shapes: frozenset = frozenset()
    suspension_reason: Optional[str] = None


@dataclass
class RootReferenceOutcome:
    """How the suspended shape behaves on one root rule path."""

    root_shape: Any
    record_node: Any
    reference_path: ReferencePath
    referenced_shape_conforms: bool
    root_conforms: bool
    effect: str
    branch_outcomes: Tuple[Tuple[Any, bool], ...] = field(default_factory=tuple)
    result_signatures: Tuple[str, ...] = field(default_factory=tuple)


@dataclass
class QuarantineAssessment:
    """Comparison of the baseline run and the target-suspension run."""

    target_shape: Any
    record_node: Any
    baseline: RunOutcome
    suspension: RunOutcome
    reference_paths: Tuple[ReferencePath, ...]
    root_outcomes: Tuple[RootReferenceOutcome, ...]

    @property
    def quarantined_baseline(self) -> bool:
        return not self.baseline.conforms

    @property
    def quarantined_after_suspension(self) -> bool:
        return not self.suspension.conforms

    def baseline_top_level_signatures(self) -> List[str]:
        return top_level_result_signatures(self.baseline.report_graph)

    def suspension_top_level_signatures(self) -> List[str]:
        return top_level_result_signatures(self.suspension.report_graph)

    def removed_result_signatures(self) -> List[str]:
        """Result trees present in the baseline but gone after suspension."""
        baseline = set(structural_result_signatures(self.baseline.report_graph))
        suspended = set(structural_result_signatures(self.suspension.report_graph))
        return sorted(baseline - suspended)

    def added_result_signatures(self) -> List[str]:
        """Result trees introduced by the suspension run."""
        baseline = set(structural_result_signatures(self.baseline.report_graph))
        suspended = set(structural_result_signatures(self.suspension.report_graph))
        return sorted(suspended - baseline)

    def surviving_results_preserved(self) -> bool:
        """Every suspension-run result already existed with the same scope."""
        baseline = set(structural_result_signatures(self.baseline.report_graph))
        suspended = set(structural_result_signatures(self.suspension.report_graph))
        return suspended.issubset(baseline)

    def to_text(self) -> str:
        lines = [
            "Quarantine rule change assessment",
            f"  Suspended shape: {_term_to_text(self.target_shape)}",
            f"  Input record: {_term_to_text(self.record_node)}",
            f"  Baseline run conforms: {self.baseline.conforms}",
            f"  Suspension run conforms: {self.suspension.conforms}",
            "  Root rules still referencing the suspended shape:",
        ]
        if not self.root_outcomes:
            lines.append("    (none)")
        for outcome in self.root_outcomes:
            hop_texts = [
                f"{_term_to_text(hop.predicate)} {_term_to_text(hop.child_shape)}"
                for hop in outcome.reference_path.hops
            ]
            path_text = " -> ".join([_term_to_text(outcome.root_shape)] + hop_texts)
            lines.append(f"    - Root rule: {_term_to_text(outcome.root_shape)}")
            lines.append(f"      Reference path: {path_text}")
            lines.append(f"      Suspended shape conforms on path: {outcome.referenced_shape_conforms}")
            lines.append(f"      Root rule conforms: {outcome.root_conforms}")
            lines.append(f"      Effect: {outcome.effect}")
            for member, conforms in outcome.branch_outcomes:
                lines.append(
                    f"        branch {_term_to_text(member)}: {'pass' if conforms else 'fail'}"
                )
        return "\n".join(lines) + "\n"

    def to_graph(self) -> Graph:
        graph = Graph()
        graph.bind("q", Q)
        graph.bind("sh", SH)
        record_node = BNode()
        graph.add((record_node, RDF.type, _q("RuleChangeRecord")))
        graph.add((record_node, _q("suspendedShape"), _as_uri(self.target_shape)))
        graph.add((record_node, _q("inputRecord"), _as_uri_or_literal(self.record_node)))
        graph.add((record_node, _q("baselineConforms"), Literal(self.baseline.conforms, datatype=XSD.boolean)))
        graph.add((record_node, _q("suspensionConforms"), Literal(self.suspension.conforms, datatype=XSD.boolean)))
        if self.suspension.suspension_reason:
            graph.add((record_node, _q("suspensionReason"), Literal(self.suspension.suspension_reason)))
        for outcome in self.root_outcomes:
            reference_node = BNode()
            graph.add((record_node, _q("rootRuleReference"), reference_node))
            graph.add((reference_node, RDF.type, _q("RootRuleReference")))
            graph.add((reference_node, _q("rootRule"), _as_uri(outcome.root_shape)))
            graph.add((reference_node, _q("referencesShape"), _as_uri(self.target_shape)))
            graph.add((reference_node, _q("inputRecord"), _as_uri_or_literal(outcome.record_node)))
            for hop in outcome.reference_path.hops:
                hop_node = BNode()
                graph.add((reference_node, _q("referenceHop"), hop_node))
                graph.add((hop_node, _q("hopParent"), _as_uri(hop.parent_shape)))
                graph.add((hop_node, _q("hopPredicate"), hop.predicate))
                graph.add((hop_node, _q("hopChild"), _as_uri_or_literal(hop.child_shape)))
            graph.add(
                (
                    reference_node,
                    _q("referencedShapeConforms"),
                    Literal(outcome.referenced_shape_conforms, datatype=XSD.boolean),
                )
            )
            graph.add((reference_node, _q("rootRuleConforms"), Literal(outcome.root_conforms, datatype=XSD.boolean)))
            graph.add((reference_node, _q("failureEffect"), _q(outcome.effect)))
            for member, conforms in outcome.branch_outcomes:
                branch_node = BNode()
                graph.add((reference_node, _q("branchOutcome"), branch_node))
                graph.add((branch_node, _q("branchShape"), _as_uri_or_literal(member)))
                graph.add((branch_node, _q("branchConforms"), Literal(conforms, datatype=XSD.boolean)))
            for signature in outcome.result_signatures:
                graph.add((reference_node, _q("resultSignature"), Literal(signature)))
        return graph


def _q(local_name: str) -> URIRef:
    return URIRef(f"{Q}{local_name}")


def _as_uri(term: Any) -> URIRef:
    if isinstance(term, URIRef):
        return term
    return URIRef(str(term))


def _as_uri_or_literal(term: Any):
    if isinstance(term, URIRef):
        return term
    if isinstance(term, BNode):
        return Literal(str(term))
    if isinstance(term, Literal):
        return term
    return Literal(str(term))


def _term_to_text(term: Any) -> str:
    if term is None:
        return "(none)"
    return str(term)


def top_level_results(report_graph: Graph) -> List[Any]:
    report_nodes = list(report_graph.subjects(RDF.type, SH.ValidationReport))
    results: List[Any] = []
    for report_node in report_nodes:
        results.extend(report_graph.objects(report_node, SH.result))
    return results


def _result_signature(report_graph: Graph, result_node: Any) -> Tuple:
    def first(predicate):
        values = list(report_graph.objects(result_node, predicate))
        return str(values[0]) if values else None

    detail_signatures = tuple(
        sorted(_result_signature(report_graph, detail) for detail in report_graph.objects(result_node, SH.detail))
    )
    return (
        first(SH.focusNode),
        first(SH.sourceShape),
        first(SH.sourceConstraintComponent),
        first(SH.resultPath),
        first(SH.value),
        first(SH.resultSeverity),
        detail_signatures,
    )


def top_level_result_signatures(report_graph: Graph) -> List[str]:
    signatures = [_result_signature(report_graph, result) for result in top_level_results(report_graph)]
    return [repr(signature) for signature in sorted(signatures, key=repr)]


def _structural_signature(report_graph: Graph, result_node: Any) -> Tuple:
    """Stable, BNode-independent signature of one result tree.

    Blank node identifiers differ between independently generated validation
    reports, so they are replaced by a placeholder. The nested sh:detail
    trees are compared by component, path and value rather than node id.
    """

    def first(predicate: URIRef):
        values = list(report_graph.objects(result_node, predicate))
        if not values:
            return None
        value = values[0]
        if isinstance(value, BNode):
            return ("bnode",)
        return str(value)

    details = tuple(
        sorted(
            _structural_signature(report_graph, detail)
            for detail in report_graph.objects(result_node, SH.detail)
        )
    )
    return (
        first(SH.focusNode),
        first(SH.sourceConstraintComponent),
        first(SH.resultPath),
        first(SH.value),
        first(SH.resultSeverity),
        details,
    )


def structural_result_signatures(report_graph: Graph) -> List[str]:
    signatures = [
        _structural_signature(report_graph, result) for result in top_level_results(report_graph)
    ]
    return [repr(signature) for signature in sorted(signatures, key=repr)]


def reports_equivalent(report_a: Graph, report_b: Graph) -> bool:
    """True when both reports carry the same result trees, ignoring BNode ids."""
    return structural_result_signatures(report_a) == structural_result_signatures(report_b)


def results_for_root_and_record(report_graph: Graph, root_shape: Any, record_node: Any) -> List[Any]:
    matched: List[Any] = []
    for result in top_level_results(report_graph):
        focuses = set(report_graph.objects(result, SH.focusNode))
        sources = set(report_graph.objects(result, SH.sourceShape))
        if record_node in focuses and root_shape in sources:
            matched.append(result)
    return matched


def _has_own_target(shapes_graph: Graph, shape_node: Any) -> bool:
    """True when the shape declares at least one SHACL target of its own."""
    for predicate in _TARGET_PREDICATES:
        if next(shapes_graph.objects(shape_node, predicate), None) is not None:
            return True
    if next(shapes_graph.objects(shape_node, SH.target), None) is not None:
        return True
    return False


def find_reference_paths(
    shapes_graph: Graph, target_shape: Any, root_shapes: Optional[Sequence[Any]] = None
) -> Tuple[ReferencePath, ...]:
    """Static search for every shape path that reaches ``target_shape``.

    Follows the shape-expecting parameters sh:node, sh:property, sh:not,
    sh:qualifiedValueShape and the list-taking sh:and, sh:or, sh:xone.
    When ``root_shapes`` is omitted, the search starts at every node shape
    that declares at least one target, i.e. the root rules of the shapes graph.
    """
    found: Dict[Any, ReferencePath] = {}

    if root_shapes is None:
        root_shapes = sorted(
            (
                node
                for node in shapes_graph.subjects(RDF.type, SH.NodeShape)
                if _has_own_target(shapes_graph, node)
            ),
            key=str,
        )

    def outgoing(shape_node):
        for predicate in _SHAPE_REF_PREDICATES:
            for child in shapes_graph.objects(shape_node, predicate):
                yield predicate, child
        for predicate in _LIST_REF_PREDICATES:
            for list_node in shapes_graph.objects(shape_node, predicate):
                for child in shapes_graph.items(list_node):
                    yield predicate, child

    def walk(shape_node, hops, visited):
        if shape_node in visited:
            return
        visited = visited | {shape_node}
        for predicate, child in outgoing(shape_node):
            next_hops = hops + (ReferenceHop(shape_node, predicate, child),)
            if child == target_shape:
                root_shape = hops[0].parent_shape if hops else shape_node
                path = ReferencePath(root_shape=root_shape, hops=next_hops)
                found.setdefault((root_shape, tuple(next_hops)), path)
            else:
                walk(child, next_hops, visited)

    for root_shape in root_shapes:
        walk(root_shape, (), frozenset())
    return tuple(found.values())


def _evaluation_rows(trace: ValidationTrace, shape_node: Any, record_node: Any, root_shape: Any):
    return [
        evaluation
        for evaluation in trace.evaluations_of(shape_node)
        if evaluation.focus_node == record_node and evaluation.root_shape_node == root_shape
    ]


def _root_conforms(trace: ValidationTrace, report_graph: Graph, root_shape: Any, record_node: Any) -> bool:
    root_rows = [
        evaluation
        for evaluation in trace.shape_evaluations
        if evaluation.shape_node == root_shape
        and evaluation.focus_node == record_node
        and evaluation.shape_path == [root_shape]
    ]
    if root_rows:
        return all(row.conforms for row in root_rows)
    return not results_for_root_and_record(report_graph, root_shape, record_node)


def _branch_outcomes(
    trace: ValidationTrace, root_shape: Any, record_node: Any, target_shape: Any
) -> Tuple[Tuple[Any, bool], ...]:
    relevant = [
        branch
        for branch in trace.logical_branches
        if branch.root_shape_node == root_shape
        and branch.focus_node == record_node
        and branch.member_shape_node == target_shape
    ]
    outcomes: List[Tuple[Any, bool]] = []
    for branch in relevant:
        siblings = [
            (sibling.member_shape_node, sibling.conforms)
            for sibling in trace.logical_branches
            if sibling.root_shape_node == root_shape
            and sibling.focus_node == record_node
            and sibling.constraint == branch.constraint
        ]
        for sibling in siblings:
            if sibling not in outcomes:
                outcomes.append(sibling)
    return tuple(outcomes)


def _classify_effect(
    referenced_conforms: bool,
    root_conforms: bool,
    reference_path: ReferencePath,
    branch_outcomes: Sequence[Tuple[Any, bool]],
) -> str:
    if referenced_conforms:
        return EFFECT_NO_EFFECT
    disjunctive = reference_path.last_predicate in (SH["or"], SH.xone)
    if root_conforms and disjunctive:
        target = reference_path.target_shape
        other_branch_passes = any(
            conforms for member, conforms in branch_outcomes if member != target
        )
        if other_branch_passes:
            return EFFECT_RESCUED_BY_BRANCH
    if not root_conforms:
        return EFFECT_CAUSES_REJECTION
    return EFFECT_NO_EFFECT


def build_assessment(
    target_shape: Any,
    record_node: Any,
    baseline: RunOutcome,
    suspension: RunOutcome,
    shapes_graph: Graph,
) -> QuarantineAssessment:
    dynamic_roots = set(suspension.trace.referencing_roots(target_shape))
    reference_paths = find_reference_paths(shapes_graph, target_shape, root_shapes=sorted(dynamic_roots, key=str))
    root_outcomes: List[RootReferenceOutcome] = []
    for root_shape in sorted(dynamic_roots, key=str):
        root_paths = [path for path in reference_paths if path.root_shape == root_shape]
        if not root_paths:
            continue
        evaluations = _evaluation_rows(suspension.trace, target_shape, record_node, root_shape)
        if not evaluations:
            continue
        referenced_conforms = all(row.conforms for row in evaluations)
        root_conforms = _root_conforms(suspension.trace, suspension.report_graph, root_shape, record_node)
        branch_outcomes = _branch_outcomes(suspension.trace, root_shape, record_node, target_shape)
        for reference_path in root_paths:
            effect = _classify_effect(referenced_conforms, root_conforms, reference_path, branch_outcomes)
            result_nodes = results_for_root_and_record(suspension.report_graph, root_shape, record_node)
            signatures = tuple(
                repr(_structural_signature(suspension.report_graph, result_node)) for result_node in result_nodes
            )
            root_outcomes.append(
                RootReferenceOutcome(
                    root_shape=root_shape,
                    record_node=record_node,
                    reference_path=reference_path,
                    referenced_shape_conforms=referenced_conforms,
                    root_conforms=root_conforms,
                    effect=effect,
                    branch_outcomes=branch_outcomes,
                    result_signatures=signatures,
                )
            )
    return QuarantineAssessment(
        target_shape=target_shape,
        record_node=record_node,
        baseline=baseline,
        suspension=suspension,
        reference_paths=tuple(reference_paths),
        root_outcomes=tuple(root_outcomes),
    )


def evaluate_quarantine(
    data_graph: Union[Graph, str, bytes],
    shacl_graph: Union[Graph, str, bytes],
    target_shape: Union[str, URIRef],
    record_node: Union[str, URIRef],
    *,
    data_graph_format: Optional[str] = None,
    shacl_graph_format: Optional[str] = None,
    inference: Optional[str] = None,
    suspension_reason: Optional[str] = None,
) -> QuarantineAssessment:
    """Validate one record twice and explain the effect of suspending one shape.

    The first run is a normal SHACL validation. The second run suspends only
    the own target declarations of ``target_shape``. The standard SHACL
    reports are returned untouched; all extra evidence lives on the traces and
    the returned assessment.
    """
    if isinstance(target_shape, str):
        target_shape_ref = URIRef(target_shape)
    else:
        target_shape_ref = target_shape
    if isinstance(record_node, str):
        record_ref = URIRef(record_node)
    else:
        record_ref = record_node

    baseline_trace = ValidationTrace()
    baseline_conforms, baseline_report, _ = validate(
        data_graph,
        shacl_graph=shacl_graph,
        data_graph_format=data_graph_format,
        shacl_graph_format=shacl_graph_format,
        inference=inference,
        validation_trace=baseline_trace,
    )
    baseline = RunOutcome(
        label="baseline",
        conforms=baseline_conforms,
        report_graph=baseline_report,
        trace=baseline_trace,
    )

    suspension_trace = ValidationTrace()
    suspension = TargetSuspension(
        shape_nodes={target_shape_ref}, reason=suspension_reason
    )
    suspension_conforms, suspension_report, _ = validate(
        data_graph,
        shacl_graph=shacl_graph,
        data_graph_format=data_graph_format,
        shacl_graph_format=shacl_graph_format,
        inference=inference,
        suspend_targets=[target_shape_ref],
        suspension_reason=suspension_reason,
        validation_trace=suspension_trace,
    )
    suspension_run = RunOutcome(
        label="target_suspension",
        conforms=suspension_conforms,
        report_graph=suspension_report,
        trace=suspension_trace,
        suspended_shapes=frozenset(suspension.shape_nodes),
        suspension_reason=suspension_reason,
    )

    if isinstance(shacl_graph, Graph):
        static_shapes_graph = shacl_graph
    else:
        static_shapes_graph = Graph()
        static_shapes_graph.parse(data=shacl_graph, format=shacl_graph_format or "turtle")

    return build_assessment(
        target_shape_ref,
        record_ref,
        baseline,
        suspension_run,
        static_shapes_graph,
    )
