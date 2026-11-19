# -*- coding: utf-8 -*-
#
# Tests for pyshacl.evidence: budget-constrained, provenance-preserving
# evidence packages for individual validation results.
from rdflib import Graph, Literal, Namespace
from rdflib.namespace import RDF

from pyshacl import extract_evidence, validate
from pyshacl.evidence import (
    EVIDENCE_DATA_GRAPH,
    EVIDENCE_REPORT_GRAPH,
    EVIDENCE_SHAPES_GRAPH,
    EvidencePackage,
)

SH = Namespace('http://www.w3.org/ns/shacl#')
EX = Namespace('http://example.com/ex#')

DATATYPE_SHAPES = """
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix ex: <http://example.com/ex#> .

ex:PersonShape a sh:NodeShape ;
    sh:targetClass ex:Person ;
    sh:property [ sh:path ex:age ; sh:datatype xsd:integer ] .
"""

DATATYPE_DATA = """
@prefix ex: <http://example.com/ex#> .

ex:alice a ex:Person ;
    ex:age "not a number" ;
    ex:name "Alice" .
"""

MINCOUNT_SHAPES = """
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.com/ex#> .

ex:PersonShape a sh:NodeShape ;
    sh:targetClass ex:Person ;
    sh:property [ sh:path ex:email ; sh:minCount 2 ] .
"""

MINCOUNT_DATA = """
@prefix ex: <http://example.com/ex#> .

ex:alice a ex:Person ;
    ex:email "alice@example.com" ;
    ex:name "Alice" .
"""

CLOSED_SHAPES = """
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix ex: <http://example.com/ex#> .

ex:PersonShape a sh:NodeShape ;
    sh:targetClass ex:Person ;
    sh:closed true ;
    sh:ignoredProperties ( rdf:type ) ;
    sh:property [ sh:path ex:name ] .
"""

CLOSED_DATA = """
@prefix ex: <http://example.com/ex#> .

ex:alice a ex:Person ;
    ex:name "Alice" ;
    ex:nick "Ally" .
"""


def _first_violation(report_graph):
    violations = list(report_graph.subjects(RDF.type, SH.ValidationResult))
    assert len(violations) == 1
    return violations[0]


def _validate(data_text, shapes_text):
    data_graph = Graph().parse(data=data_text, format='turtle')
    shapes_graph = Graph().parse(data=shapes_text, format='turtle')
    conforms, report_graph, _ = validate(
        data_graph, shacl_graph=shapes_graph, inference='none'
    )
    assert not conforms
    return data_graph, shapes_graph, report_graph


def test_existential_violation_selects_only_cited_facts():
    data_graph, shapes_graph, report_graph = _validate(DATATYPE_DATA, DATATYPE_SHAPES)
    violation = _first_violation(report_graph)

    package = extract_evidence(report_graph, violation, data_graph, shapes_graph)

    assert package.mode == 'existential'
    assert package.complete
    assert package.missing == []
    # Only the cited fact is needed; unrelated facts are not evidence.
    assert set(package.data_graph) == {
        (EX.alice, RDF.type, EX.Person),
        (EX.alice, EX.age, Literal("not a number")),
    }
    # The shape definition is included so the constraint can be interpreted.
    assert any(p == SH.path and o == EX.age for s, p, o in package.shapes_graph)
    assert any(p == SH.targetClass and o == EX.Person for s, p, o in package.shapes_graph)
    # The package independently reproduces the violation.
    assert package.verify()


def test_absence_violation_requires_complete_neighbourhood():
    data_graph, shapes_graph, report_graph = _validate(MINCOUNT_DATA, MINCOUNT_SHAPES)
    violation = _first_violation(report_graph)

    package = extract_evidence(report_graph, violation, data_graph, shapes_graph)

    assert package.mode == 'absence'
    assert package.complete
    # Every ex:email triple of the focus node is evidence, even though the
    # report cites no sh:value for a minCount violation.
    assert (EX.alice, EX.email, Literal("alice@example.com")) in package.data_graph
    # Facts on other paths do not affect the count and are not required.
    assert (EX.alice, EX.name, Literal("Alice")) not in package.data_graph
    assert package.verify()


def test_closure_violation_keeps_whole_focus_neighbourhood():
    data_graph, shapes_graph, report_graph = _validate(CLOSED_DATA, CLOSED_SHAPES)
    violation = _first_violation(report_graph)

    package = extract_evidence(report_graph, violation, data_graph, shapes_graph)

    assert package.mode == 'closure'
    assert package.complete
    # Both the cited offending triple and the unreferenced permitted one are
    # evidence: the closed-world proof depends on everything the node says.
    assert (EX.alice, EX.nick, Literal("Ally")) in package.data_graph
    assert (EX.alice, EX.name, Literal("Alice")) in package.data_graph
    assert package.verify()


def test_budget_shortfall_reports_what_is_missing():
    data_graph, shapes_graph, report_graph = _validate(MINCOUNT_DATA, MINCOUNT_SHAPES)
    violation = _first_violation(report_graph)
    full = extract_evidence(report_graph, violation, data_graph, shapes_graph)

    package = extract_evidence(
        report_graph, violation, data_graph, shapes_graph, budget=full.cost - 1
    )

    assert not package.complete
    assert package.cost <= full.cost - 1
    assert len(package.missing) == 1
    # The shortfall message names what is missing and why it matters.
    assert "missing 1 of" in package.missing[0]


def test_budget_shortfall_on_absence_violation_disclaims_the_proof():
    data_graph, shapes_graph, report_graph = _validate(MINCOUNT_DATA, MINCOUNT_SHAPES)
    violation = _first_violation(report_graph)

    # Only enough budget for the report excerpt and the shape definition.
    shapes_only = extract_evidence(report_graph, violation, data_graph, shapes_graph)
    data_cost = len(set(shapes_only.data_graph))
    budget = shapes_only.cost - data_cost
    package = extract_evidence(
        report_graph, violation, data_graph, shapes_graph, budget=budget
    )

    assert not package.complete
    assert len(package.data_graph) == 0
    assert any("cannot" in note and "absence" in note for note in package.missing)


def test_shared_facts_are_counted_once():
    report_graph = Graph()
    data_graph = Graph()
    shapes_graph = Graph()
    shared = (EX.alice, RDF.type, EX.Person)
    data_graph.add(shared)
    shapes_graph.add(shared)
    shapes_graph.add((EX.PersonShape, SH.targetClass, EX.Person))

    package = EvidencePackage(
        violation=EX.result1,
        component=SH.MinCountConstraintComponent,
        focus_node=EX.alice,
        mode='absence',
        report_graph=report_graph,
        data_graph=data_graph,
        shapes_graph=shapes_graph,
        budget=None,
        missing=[],
    )
    # Two graphs carry the shared fact, but the cost counts it once.
    assert len(package.data_graph) + len(package.shapes_graph) == 3
    assert package.cost == 2


def test_provenance_is_preserved_in_dataset():
    data_graph, shapes_graph, report_graph = _validate(DATATYPE_DATA, DATATYPE_SHAPES)
    violation = _first_violation(report_graph)
    package = extract_evidence(report_graph, violation, data_graph, shapes_graph)

    dataset = package.to_dataset()
    report_part = dataset.graph(EVIDENCE_REPORT_GRAPH)
    data_part = dataset.graph(EVIDENCE_DATA_GRAPH)
    shapes_part = dataset.graph(EVIDENCE_SHAPES_GRAPH)

    assert (violation, SH.focusNode, EX.alice) in report_part
    assert (EX.alice, EX.age, Literal("not a number")) in data_part
    assert any(p == SH.path and o == EX.age for s, p, o in shapes_part)
    # Facts from different sources are not mixed into one graph.
    assert (EX.alice, EX.age, Literal("not a number")) not in shapes_part
