# -*- coding: utf-8 -*-
#
"""
Evidence package extraction for SHACL validation results.

Given a target violation from a SHACL validation report, the source data
graph, and the shapes graph, this module selects the material an auditor
needs to independently re-check that single violation, while preserving
the provenance of every fact (report excerpt, data evidence, and shape
definitions are kept in separate graphs).

Two kinds of violations need different completeness guarantees:

* Existential violations (e.g. sh:datatype, sh:pattern, sh:InConstraint):
  the violation is proven by the *presence* of the facts cited in the
  report, so only those triples are selected.
* Absence- or closure-based violations (e.g. sh:minCount, sh:hasValue,
  sh:closed, sh:class, sh:maxCount): the violation is proven by the
  *absence* of further facts, or by an exhaustive count.  For these, the
  complete relevant neighbourhood of the focus node must be included.
  Dropping facts that the report does not reference would silently change
  the proof, so a package that cannot fit them all is flagged incomplete
  and the shortfall is reported instead of claiming the same proof.

Shared facts are only counted once against the budget, even when they are
required by more than one part of the package.
"""
from typing import List, Optional, Set, Tuple

import rdflib
from rdflib import Graph, URIRef
from rdflib.term import BNode, Node

from .consts import (
    OWL_Class,
    RDF_type,
    RDFS_Class,
    SH,
    SH_focusNode,
    SH_node,
    SH_property,
    SH_resultPath,
    SH_sourceConstraint,
    SH_sourceConstraintComponent,
    SH_sourceShape,
    SH_targetClass,
    SH_targetObjectsOf,
    SH_targetSubjectsOf,
    SH_value,
)
from .errors import ReportableRuntimeError

# Provenance graph names used by EvidencePackage.to_dataset()
EVIDENCE_REPORT_GRAPH = URIRef("urn:pyshacl:evidence:report")
EVIDENCE_DATA_GRAPH = URIRef("urn:pyshacl:evidence:data")
EVIDENCE_SHAPES_GRAPH = URIRef("urn:pyshacl:evidence:shapes")

# Violations that hold because a required fact is missing.  The proof needs
# every triple of the focus node on the result path, not just cited ones.
ABSENCE_COMPONENTS = frozenset(
    (
        SH.MinCountConstraintComponent,
        SH.HasValueConstraintComponent,
        SH.QualifiedMinCountConstraintComponent,
    )
)

# Violations that hold because of everything the focus node says about
# itself; the proof needs the complete closed neighbourhood.
CLOSURE_COMPONENTS = frozenset((SH.ClosedConstraintComponent,))

# Violations that hold because of how many values exist; the proof needs
# every value on the result path so the count can be re-checked.
COUNT_COMPONENTS = frozenset(
    (
        SH.MaxCountConstraintComponent,
        SH.QualifiedMaxCountConstraintComponent,
    )
)

# Violations that hold because a required rdf:type is missing from a node.
TYPE_ABSENCE_COMPONENTS = frozenset((SH.ClassConstraintComponent,))

# Shape-to-shape links followed (in addition to blank nodes) when collecting
# the definition of a source shape.
SHAPE_LINK_PREDICATES = frozenset((SH_property, SH_node))

Triple = Tuple[Node, Node, Node]


class EvidencePackage:
    """A provenance-preserving, budget-checked evidence bundle for one violation.

    Attributes:
        violation: The sh:ValidationResult node this package was built for.
        component: The sh:sourceConstraintComponent of the violation.
        focus_node: The sh:focusNode of the violation.
        mode: One of 'existential', 'absence', 'closure', or 'count'.
        report_graph: Excerpt of the validation report (provenance).
        data_graph: Evidence triples drawn from the source data graph.
        shapes_graph: Shape definition triples drawn from the shapes graph.
        budget: The triple budget the package was built against, or None.
        cost: Number of distinct triples in the package (shared facts count once).
        complete: True when everything required fit within the budget.
        missing: Human-readable descriptions of what was left out and why.
    """

    def __init__(
        self,
        violation: Node,
        component: Optional[Node],
        focus_node: Optional[Node],
        mode: str,
        report_graph: Graph,
        data_graph: Graph,
        shapes_graph: Graph,
        budget: Optional[int],
        missing: List[str],
    ) -> None:
        self.violation = violation
        self.component = component
        self.focus_node = focus_node
        self.mode = mode
        self.report_graph = report_graph
        self.data_graph = data_graph
        self.shapes_graph = shapes_graph
        self.budget = budget
        self.missing = missing
        self.complete = len(missing) < 1
        # Shared facts are only counted once, across all three graphs.
        self.cost = len(
            set(report_graph) | set(data_graph) | set(shapes_graph)
        )

    def __repr__(self):
        return (
            "EvidencePackage(violation={!r}, mode={!r}, cost={}, budget={!r}, complete={!r})".format(
                self.violation, self.mode, self.cost, self.budget, self.complete
            )
        )

    def to_dataset(self) -> rdflib.Dataset:
        """Serialise the package as a dataset with one named graph per source."""
        ds = rdflib.Dataset()
        for graph_id, graph in (
            (EVIDENCE_REPORT_GRAPH, self.report_graph),
            (EVIDENCE_DATA_GRAPH, self.data_graph),
            (EVIDENCE_SHAPES_GRAPH, self.shapes_graph),
        ):
            named = ds.graph(graph_id)
            for triple in graph:
                named.add(triple)
        return ds

    def verify(self) -> bool:
        """Independently re-check the violation using only this package.

        Re-runs validation on the packaged data and shapes graphs and
        returns True only if an equivalent result (same focus node,
        constraint component, result path, and values) is reproduced.
        """
        from .entrypoints import validate

        _conforms, report, _text = validate(
            self.data_graph, shacl_graph=self.shapes_graph, inference='none'
        )
        expected_path = self.report_graph.value(self.violation, SH_resultPath)
        expected_values = set(self.report_graph.objects(self.violation, SH_value))
        for result in report.subjects(SH_focusNode, self.focus_node):
            if report.value(result, SH_sourceConstraintComponent) != self.component:
                continue
            if report.value(result, SH_resultPath) != expected_path:
                continue
            if set(report.objects(result, SH_value)) != expected_values:
                continue
            return True
        return False


def _bounded_description(graph: Graph, roots, follow_links: bool = False) -> Set[Triple]:
    """Collect all triples about ``roots``, recursing into blank nodes.

    When ``follow_links`` is True, sh:property and sh:node references are
    followed as well, so linked property shapes are included even when they
    are named with IRIs.
    """
    triples: Set[Triple] = set()
    seen: Set[Node] = set()
    stack = [r for r in roots if r is not None]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        for s, p, o in graph.triples((node, None, None)):
            triples.add((s, p, o))
            if isinstance(o, BNode) or (follow_links and p in SHAPE_LINK_PREDICATES):
                stack.append(o)
    return triples


def _path_data_triples(data_graph: Graph, focus: Node, path: Node) -> Set[Triple]:
    """All data triples of ``focus`` on ``path``.

    A complex (blank node) path cannot be decomposed here, so every triple
    about the focus node is returned as a safe over-approximation.
    """
    if isinstance(path, URIRef):
        return set(data_graph.triples((focus, path, None)))
    return set(data_graph.triples((focus, None, None)))


def _shape_lineage(shapes_graph: Graph, source_shape: Optional[Node]) -> List[Node]:
    """The source shape plus every shape that reaches it via sh:property/sh:node.

    Targeting declarations (sh:targetClass and friends) usually live on an
    ancestor node shape, so the lineage is needed both to interpret the
    constraint and to re-target the focus node during independent review.
    """
    lineage: List[Node] = []
    seen: Set[Node] = set()
    stack = [source_shape]
    while stack:
        node = stack.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        lineage.append(node)
        for predicate in SHAPE_LINK_PREDICATES:
            for parent in shapes_graph.subjects(predicate, node):
                stack.append(parent)
    return lineage


def _targeting_data_triples(
    shapes_graph: Graph, data_graph: Graph, source_shape: Optional[Node], focus: Node
) -> Set[Triple]:
    """Data triples that make ``focus`` a target of ``source_shape``.

    Without these the auditor's re-validation would not even select the
    focus node, so they are part of the evidence for every mode.
    """
    triples: Set[Triple] = set()
    if source_shape is None:
        return triples
    for target_class in shapes_graph.objects(source_shape, SH_targetClass):
        triples |= set(data_graph.triples((focus, RDF_type, target_class)))
    for predicate in shapes_graph.objects(source_shape, SH_targetSubjectsOf):
        triples |= set(data_graph.triples((focus, predicate, None)))
    for predicate in shapes_graph.objects(source_shape, SH_targetObjectsOf):
        triples |= set(data_graph.triples((None, predicate, focus)))
    # Implicit class targets: a shape that is itself a class.
    for class_type in (RDFS_Class, OWL_Class):
        if (source_shape, RDF_type, class_type) in shapes_graph:
            triples |= set(data_graph.triples((focus, RDF_type, source_shape)))
    return triples


def _collect_data_evidence(
    mode: str,
    data_graph: Graph,
    focus: Node,
    path: Optional[Node],
    values: List[Node],
) -> Set[Triple]:
    if mode == 'closure':
        # sh:closed: the proof is the complete set of properties used by the
        # focus node; nothing about it may be left out.
        return set(data_graph.triples((focus, None, None)))
    if mode in ('absence', 'count'):
        if path is None:
            return set(data_graph.triples((focus, None, None)))
        return _path_data_triples(data_graph, focus, path)
    if mode == 'type-absence':
        # sh:class: the proof is the complete set of rdf:type statements of
        # each cited node (or of the focus node when none are cited).
        triples: Set[Triple] = set()
        for node in values or [focus]:
            triples |= set(data_graph.triples((node, RDF_type, None)))
        return triples
    # existential: only the facts cited by the report are the proof.
    triples = set()
    for value in values:
        if isinstance(path, URIRef):
            triples |= set(data_graph.triples((focus, path, value)))
        else:
            for predicate in set(data_graph.predicates(focus, value)):
                triples.add((focus, predicate, value))
    return triples


def extract_evidence(
    report_graph: Graph,
    violation: Node,
    data_graph: Graph,
    shapes_graph: Graph,
    budget: Optional[int] = None,
) -> EvidencePackage:
    """Build an evidence package for a single validation result.

    :param report_graph: The validation report graph containing the result.
    :param violation: The sh:ValidationResult node to package evidence for.
    :param data_graph: The source data graph that was validated.
    :param shapes_graph: The shapes graph the data was validated against.
    :param budget: Maximum number of distinct triples the auditor accepts,
        or None for no limit.  Shared facts are counted once.
    :returns: An EvidencePackage.  When the budget is too small, the package
        is filled in priority order (report excerpt, shape definitions, data
        evidence), ``complete`` is False, and ``missing`` explains exactly
        what was left out and why it matters.
    """
    focus = report_graph.value(violation, SH_focusNode)
    if focus is None:
        raise ReportableRuntimeError(
            "The given violation has no sh:focusNode; cannot build an evidence package."
        )
    component = report_graph.value(violation, SH_sourceConstraintComponent)
    path = report_graph.value(violation, SH_resultPath)
    values = list(report_graph.objects(violation, SH_value))
    source_shape = report_graph.value(violation, SH_sourceShape)
    source_constraint = report_graph.value(violation, SH_sourceConstraint)

    if component in ABSENCE_COMPONENTS:
        mode = 'absence'
    elif component in CLOSURE_COMPONENTS:
        mode = 'closure'
    elif component in COUNT_COMPONENTS:
        mode = 'count'
    elif component in TYPE_ABSENCE_COMPONENTS:
        mode = 'type-absence'
    else:
        mode = 'existential'

    report_triples = _bounded_description(report_graph, [violation])
    lineage = _shape_lineage(shapes_graph, source_shape)
    shapes_triples = _bounded_description(
        shapes_graph, lineage + [source_constraint], follow_links=True
    )
    data_triples = _collect_data_evidence(mode, data_graph, focus, path, values)
    for shape in lineage:
        data_triples |= _targeting_data_triples(shapes_graph, data_graph, shape, focus)

    requirements = [
        (
            report_triples,
            "report excerpt identifying violation {!r}".format(violation),
            False,
        ),
        (
            shapes_triples,
            "shape definition triples for {!r} (required to interpret the constraint)".format(
                source_shape
            ),
            False,
        ),
        (
            data_triples,
            "data evidence triples for focus node {!r}".format(focus),
            mode != 'existential',
        ),
    ]

    selected_parts: List[Set[Triple]] = []
    missing: List[str] = []
    spent = 0
    for triples, description, completeness_critical in requirements:
        selected: Set[Triple] = set()
        remaining = sorted(triples, key=lambda t: (str(t[0]), str(t[1]), str(t[2])))
        for triple in remaining:
            # Shared facts that are already in the package cost nothing.
            already_included = any(triple in part for part in selected_parts)
            if budget is not None and not already_included and spent >= budget:
                break
            selected.add(triple)
            if not already_included:
                spent += 1
        leftover = len(triples) - len(selected)
        if leftover > 0:
            note = "missing {} of {} {}".format(leftover, len(triples), description)
            if completeness_critical:
                note += (
                    "; without the complete neighbourhood this package cannot"
                    " prove an absence/closure-based violation"
                )
            missing.append(note)
        selected_parts.append(selected)

    report_out = Graph()
    for triple in selected_parts[0]:
        report_out.add(triple)
    shapes_out = Graph()
    for triple in selected_parts[1]:
        shapes_out.add(triple)
    data_out = Graph()
    for triple in selected_parts[2]:
        data_out.add(triple)

    return EvidencePackage(
        violation=violation,
        component=component,
        focus_node=focus,
        mode=mode,
        report_graph=report_out,
        data_graph=data_out,
        shapes_graph=shapes_out,
        budget=budget,
        missing=missing,
    )
