# -*- coding: utf-8 -*-
"""Tests for target suspension and quarantine rule-change evidence.

The scenario mirrors the warehouse intake rules described in the design:
one standard address shape (S) is both a target of its own and referenced by
a warehouse rule (sh:node) and a forwarding rule (sh:or with a PO box rule).
A record that fails S but satisfies the PO box rule is validated twice; the
second run suspends only S's own target declarations.
"""

from rdflib import Graph, URIRef
from rdflib.namespace import RDF, SH

from pyshacl import ValidationTrace, evaluate_quarantine, validate
from pyshacl.quarantine import (
    EFFECT_CAUSES_REJECTION,
    EFFECT_NO_EFFECT,
    EFFECT_RESCUED_BY_BRANCH,
    find_reference_paths,
    reports_equivalent,
    structural_result_signatures,
)

EX = "http://example.org/addr#"

STANDARD_SHAPE = URIRef(EX + "StandardAddressShape")
POBOX_SHAPE = URIRef(EX + "PoBoxShape")
WAREHOUSE_RULE = URIRef(EX + "WarehouseAddressRule")
FORWARDING_RULE = URIRef(EX + "ForwardingAddressRule")
RECORD = URIRef(EX + "record1")

SHAPES_TTL = """
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.org/addr#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

ex:StandardAddressShape a sh:NodeShape ;
    sh:targetNode ex:record1 ;
    sh:property [
        a sh:PropertyShape ;
        sh:path ex:postalCode ;
        sh:datatype xsd:string ;
        sh:minCount 1 ;
    ] ;
    sh:property [
        a sh:PropertyShape ;
        sh:path ex:street ;
        sh:datatype xsd:string ;
        sh:minCount 1 ;
    ] .

ex:PoBoxShape a sh:NodeShape ;
    sh:property [
        a sh:PropertyShape ;
        sh:path ex:poBox ;
        sh:minCount 1 ;
        sh:datatype xsd:string ;
    ] .

ex:WarehouseAddressRule a sh:NodeShape ;
    sh:targetNode ex:record1 ;
    sh:node ex:StandardAddressShape .

ex:ForwardingAddressRule a sh:NodeShape ;
    sh:targetNode ex:record1 ;
    sh:or ( ex:StandardAddressShape ex:PoBoxShape ) .
"""

DATA_TTL = """
@prefix ex: <http://example.org/addr#> .

ex:record1 a ex:Address ;
    ex:poBox "12345" .
"""

CONFORMING_DATA_TTL = """
@prefix ex: <http://example.org/addr#> .

ex:record1 a ex:Address ;
    ex:street "Main Street 1" ;
    ex:postalCode "10115" ;
    ex:poBox "12345" .
"""


def _top_level_results(report_graph):
    report_nodes = list(report_graph.subjects(RDF.type, SH.ValidationReport))
    results = []
    for report_node in report_nodes:
        results.extend(report_graph.objects(report_node, SH.result))
    return results


def _components(report_graph, result):
    return set(report_graph.objects(result, SH.sourceConstraintComponent))


def _source_shapes(report_graph, result):
    return set(report_graph.objects(result, SH.sourceShape))


def test_baseline_reports_direct_s_and_warehouse_violations():
    trace = ValidationTrace()
    conforms, report, _ = validate(
        DATA_TTL,
        shacl_graph=SHAPES_TTL,
        data_graph_format="turtle",
        shacl_graph_format="turtle",
        validation_trace=trace,
    )
    assert not conforms
    results = _top_level_results(report)
    assert len(results) == 3
    components = [next(iter(_components(report, result))) for result in results]
    assert components.count(SH.MinCountConstraintComponent) == 2
    assert components.count(SH.NodeConstraintComponent) == 1
    # The forwarding rule must pass on the baseline run, via the PO box branch.
    forwarding_results = [
        result for result in results if FORWARDING_RULE in _source_shapes(report, result)
    ]
    assert forwarding_results == []
    forwarding_rows = [
        evaluation
        for evaluation in trace.shape_evaluations
        if evaluation.shape_node == FORWARDING_RULE and evaluation.shape_path == [FORWARDING_RULE]
    ]
    assert len(forwarding_rows) == 1
    assert forwarding_rows[0].conforms is True


def test_suspending_s_target_keeps_only_warehouse_violation():
    trace = ValidationTrace()
    conforms, report, _ = validate(
        DATA_TTL,
        shacl_graph=SHAPES_TTL,
        data_graph_format="turtle",
        shacl_graph_format="turtle",
        suspend_targets=[STANDARD_SHAPE],
        suspension_reason="S target paused for intake trial",
        validation_trace=trace,
    )
    assert not conforms
    results = _top_level_results(report)
    assert len(results) == 1
    remaining = results[0]
    assert _source_shapes(report, remaining) == {WAREHOUSE_RULE}
    assert _components(report, remaining) == {SH.NodeConstraintComponent}
    # The surviving warehouse result keeps the same nested violation scope.
    details = list(report.objects(remaining, SH.detail))
    assert len(details) == 2
    detail_components = set()
    for detail in details:
        detail_components.update(report.objects(detail, SH.sourceConstraintComponent))
    assert detail_components == {SH.MinCountConstraintComponent}
    # S no longer runs as a root shape, but is still reached via both rules.
    standard_rows = trace.evaluations_of(STANDARD_SHAPE)
    assert all(row.shape_path != [STANDARD_SHAPE] for row in standard_rows)
    roots = trace.referencing_roots(STANDARD_SHAPE)
    assert set(roots) == {WAREHOUSE_RULE, FORWARDING_RULE}
    # The forwarding rule still passes, with recorded branch outcomes.
    forwarding_rows = [
        evaluation
        for evaluation in trace.shape_evaluations
        if evaluation.shape_node == FORWARDING_RULE and evaluation.shape_path == [FORWARDING_RULE]
    ]
    assert forwarding_rows[0].conforms is True
    branches = {
        branch.member_shape_node: branch.conforms
        for branch in trace.logical_branches
        if branch.root_shape_node == FORWARDING_RULE and branch.constraint == SH["or"]
    }
    assert branches == {STANDARD_SHAPE: False, POBOX_SHAPE: True}


def test_default_validation_is_unchanged_without_trace():
    plain_conforms, plain_report, _ = validate(
        DATA_TTL, shacl_graph=SHAPES_TTL, data_graph_format="turtle", shacl_graph_format="turtle"
    )
    traced_conforms, traced_report, _ = validate(
        DATA_TTL,
        shacl_graph=SHAPES_TTL,
        data_graph_format="turtle",
        shacl_graph_format="turtle",
        validation_trace=ValidationTrace(),
    )
    assert plain_conforms == traced_conforms is False
    assert reports_equivalent(plain_report, traced_report)
    assert len(structural_result_signatures(plain_report)) == 3


def test_evaluate_quarantine_classifies_rejection_and_rescue():
    assessment = evaluate_quarantine(
        DATA_TTL,
        SHAPES_TTL,
        STANDARD_SHAPE,
        RECORD,
        data_graph_format="turtle",
        shacl_graph_format="turtle",
        suspension_reason="S target paused for intake trial",
    )
    assert assessment.quarantined_baseline is True
    assert assessment.quarantined_after_suspension is True
    # Only S's own two direct results disappear; the warehouse result survives.
    assert len(assessment.removed_result_signatures()) == 2
    assert assessment.added_result_signatures() == []
    assert assessment.surviving_results_preserved() is True

    outcomes = {outcome.root_shape: outcome for outcome in assessment.root_outcomes}
    assert set(outcomes) == {WAREHOUSE_RULE, FORWARDING_RULE}

    warehouse = outcomes[WAREHOUSE_RULE]
    assert warehouse.effect == EFFECT_CAUSES_REJECTION
    assert warehouse.referenced_shape_conforms is False
    assert warehouse.root_conforms is False
    assert [hop.predicate for hop in warehouse.reference_path.hops] == [SH.node]
    assert warehouse.record_node == RECORD

    forwarding = outcomes[FORWARDING_RULE]
    assert forwarding.effect == EFFECT_RESCUED_BY_BRANCH
    assert forwarding.referenced_shape_conforms is False
    assert forwarding.root_conforms is True
    assert [hop.predicate for hop in forwarding.reference_path.hops] == [SH["or"]]
    assert dict(forwarding.branch_outcomes) == {STANDARD_SHAPE: False, POBOX_SHAPE: True}


def test_rule_change_graph_carries_root_rule_and_record_evidence():
    assessment = evaluate_quarantine(
        DATA_TTL,
        SHAPES_TTL,
        STANDARD_SHAPE,
        RECORD,
        data_graph_format="turtle",
        shacl_graph_format="turtle",
    )
    evidence = assessment.to_graph()
    records = list(evidence.subjects(RDF.type, URIRef("urn:pyshacl:quarantine#RuleChangeRecord")))
    assert len(records) == 1
    record_node = records[0]
    assert (record_node, URIRef("urn:pyshacl:quarantine#suspendedShape"), STANDARD_SHAPE) in evidence
    assert (record_node, URIRef("urn:pyshacl:quarantine#inputRecord"), RECORD) in evidence
    effects = set(evidence.objects(None, URIRef("urn:pyshacl:quarantine#failureEffect")))
    assert URIRef(f"urn:pyshacl:quarantine#{EFFECT_CAUSES_REJECTION}") in effects
    assert URIRef(f"urn:pyshacl:quarantine#{EFFECT_RESCUED_BY_BRANCH}") in effects
    # Every reference is tied back to the input record.
    references = list(evidence.objects(record_node, URIRef("urn:pyshacl:quarantine#rootRuleReference")))
    assert len(references) == 2
    for reference in references:
        assert (reference, URIRef("urn:pyshacl:quarantine#inputRecord"), RECORD) in evidence


def test_find_reference_paths_distinguishes_node_and_or_hops():
    shapes_graph = Graph()
    shapes_graph.parse(data=SHAPES_TTL, format="turtle")
    paths = find_reference_paths(shapes_graph, STANDARD_SHAPE)
    by_root = {path.root_shape: path for path in paths}
    assert set(by_root) == {WAREHOUSE_RULE, FORWARDING_RULE}
    assert by_root[WAREHOUSE_RULE].hops[0].predicate == SH.node
    assert by_root[FORWARDING_RULE].hops[0].predicate == SH["or"]
    # S itself is a root rule (it has its own target), but it cannot reference
    # itself statically, so it must not appear as a reference root.
    assert STANDARD_SHAPE not in by_root


def test_conforming_record_marks_suspension_as_no_effect():
    assessment = evaluate_quarantine(
        CONFORMING_DATA_TTL,
        SHAPES_TTL,
        STANDARD_SHAPE,
        RECORD,
        data_graph_format="turtle",
        shacl_graph_format="turtle",
    )
    assert assessment.quarantined_baseline is False
    assert assessment.quarantined_after_suspension is False
    assert assessment.removed_result_signatures() == []
    assert all(outcome.effect == EFFECT_NO_EFFECT for outcome in assessment.root_outcomes)
