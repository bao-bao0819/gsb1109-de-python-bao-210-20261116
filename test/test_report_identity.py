# -*- coding: utf-8 -*-
"""
Tests for evidence-based validation result identity and merging.

Results are identified by focus node, result path, source shape, offending
value and source constraint component, never by message text. Results from
separate constraint evaluations that describe the same fact are merged with
the contributing evaluations kept as evidence, while distinct facts produced
by one evaluation stay separate.
"""
import rdflib
from rdflib.namespace import RDF, SH, XSD

import pyshacl

PYSHACL = rdflib.Namespace("urn:pyshacl:")


def _result_nodes(graph):
    return set(graph.subjects(RDF.type, SH.ValidationResult))


def test_same_message_different_components_are_not_merged():
    shapes = """
    @prefix sh: <http://www.w3.org/ns/shacl#> .
    @prefix ex: <http://example.org/> .
    ex:PersonShape a sh:NodeShape ;
      sh:targetClass ex:Person ;
      sh:property [
        sh:path ex:name ;
        sh:minLength 5 ;
        sh:pattern "^[A-Z][a-z]+$" ;
        sh:message "invalid name" ;
      ] .
    """
    data = """
    @prefix ex: <http://example.org/> .
    ex:bob a ex:Person ; ex:name "ab" .
    """
    shapes_graph = rdflib.Graph().parse(data=shapes, format="turtle")
    data_graph = rdflib.Graph().parse(data=data, format="turtle")

    conforms, report_graph, _ = pyshacl.validate(data_graph, shacl_graph=shapes_graph)

    assert not conforms
    results = _result_nodes(report_graph)
    assert len(results) == 2
    components = {
        str(next(report_graph.objects(r, SH.sourceConstraintComponent))) for r in results
    }
    assert components == {
        "http://www.w3.org/ns/shacl#MinLengthConstraintComponent",
        "http://www.w3.org/ns/shacl#PatternConstraintComponent",
    }
    # Neither result was merged, so no evidence bookkeeping is attached.
    assert not list(report_graph.objects(None, PYSHACL.evidence))
    assert not list(report_graph.objects(None, PYSHACL.resultIdentity))


def test_cross_evaluation_duplicates_merge_with_evidence():
    shapes = """
    @prefix sh: <http://www.w3.org/ns/shacl#> .
    @prefix ex: <http://example.org/> .
    ex:NameShape a sh:PropertyShape ;
      sh:targetNode ex:bob ;
      sh:path ex:name ;
      sh:minLength 5 ;
      sh:message "invalid name" .
    ex:PersonShape a sh:NodeShape ;
      sh:targetClass ex:Person ;
      sh:property ex:NameShape .
    """
    data = """
    @prefix ex: <http://example.org/> .
    ex:bob a ex:Person ; ex:name "ab" .
    """
    shapes_graph = rdflib.Graph().parse(data=shapes, format="turtle")
    data_graph = rdflib.Graph().parse(data=data, format="turtle")

    conforms, report_graph, report_text = pyshacl.validate(data_graph, shacl_graph=shapes_graph)

    assert not conforms
    results = _result_nodes(report_graph)
    assert len(results) == 1
    result = next(iter(results))

    counts = list(report_graph.objects(result, PYSHACL.evidenceCount))
    assert len(counts) == 1
    assert counts[0].datatype == XSD.integer
    assert int(counts[0]) == 2

    evidence_nodes = list(report_graph.objects(result, PYSHACL.evidence))
    assert len(evidence_nodes) == 2
    for evidence_node in evidence_nodes:
        assert (evidence_node, RDF.type, PYSHACL.Evidence) in report_graph
        assert next(report_graph.objects(evidence_node, SH.sourceConstraintComponent)) == (
            SH.MinLengthConstraintComponent
        )
        assert str(next(report_graph.objects(evidence_node, SH.sourceShape))).endswith("NameShape")
    indices = sorted(int(o) for o in report_graph.objects(None, PYSHACL.evidenceIndex))
    assert indices == [1, 2]

    identity = list(report_graph.objects(result, PYSHACL.resultIdentity))
    assert len(identity) == 1
    identity_text = str(identity[0])
    assert "MinLengthConstraintComponent" in identity_text
    assert "http://example.org/bob" in identity_text
    assert "http://example.org/name" in identity_text
    # The identity never derives from message text.
    assert "invalid name" not in identity_text

    assert "Evidence: 2 validation result source(s)" in report_text


def test_same_evaluation_results_with_shared_fields_are_not_merged():
    # Two duplicated language tags are distinct facts emitted by one
    # sh:uniqueLang evaluation and share identical standard result fields.
    shapes = """
    @prefix sh: <http://www.w3.org/ns/shacl#> .
    @prefix ex: <http://example.org/> .
    ex:TestShape a sh:NodeShape ;
      sh:targetNode ex:Invalid ;
      sh:property [
        sh:path ex:testProperty ;
        sh:uniqueLang true ;
      ] .
    """
    data = """
    @prefix ex: <http://example.org/> .
    ex:Invalid ex:testProperty "I"@en , "Me"@en , "Ich"@de , "Mich"@de .
    """
    shapes_graph = rdflib.Graph().parse(data=shapes, format="turtle")
    data_graph = rdflib.Graph().parse(data=data, format="turtle")

    conforms, report_graph, _ = pyshacl.validate(data_graph, shacl_graph=shapes_graph)

    assert not conforms
    results = _result_nodes(report_graph)
    assert len(results) == 2
    assert not list(report_graph.objects(None, PYSHACL.evidence))


def test_removing_one_constraint_leaves_the_other_result():
    # The minLength and pattern results are independent facts; dropping the
    # pattern constraint leaves the minLength result in place.
    base_data = """
    @prefix ex: <http://example.org/> .
    ex:bob a ex:Person ; ex:name "ab" .
    """
    data_graph = rdflib.Graph().parse(data=base_data, format="turtle")

    both_shapes = rdflib.Graph().parse(
        data="""
        @prefix sh: <http://www.w3.org/ns/shacl#> .
        @prefix ex: <http://example.org/> .
        ex:PersonShape a sh:NodeShape ;
          sh:targetClass ex:Person ;
          sh:property [
            sh:path ex:name ;
            sh:minLength 5 ;
            sh:pattern "^[A-Z][a-z]+$" ;
            sh:message "invalid name" ;
          ] .
        """,
        format="turtle",
    )
    conforms, report_graph, _ = pyshacl.validate(data_graph, shacl_graph=both_shapes)
    assert not conforms
    assert len(_result_nodes(report_graph)) == 2

    min_length_only = rdflib.Graph().parse(
        data="""
        @prefix sh: <http://www.w3.org/ns/shacl#> .
        @prefix ex: <http://example.org/> .
        ex:PersonShape a sh:NodeShape ;
          sh:targetClass ex:Person ;
          sh:property [
            sh:path ex:name ;
            sh:minLength 5 ;
            sh:message "invalid name" ;
          ] .
        """,
        format="turtle",
    )
    conforms, report_graph, _ = pyshacl.validate(data_graph, shacl_graph=min_length_only)
    assert not conforms
    remaining = _result_nodes(report_graph)
    assert len(remaining) == 1
    result = next(iter(remaining))
    assert next(report_graph.objects(result, SH.sourceConstraintComponent)) == (
        SH.MinLengthConstraintComponent
    )
