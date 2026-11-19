# -*- coding: utf-8 -*-
#
# Regression tests for sh:qualifiedValueShape counting semantics.
#
# Rules covered:
#   * When sh:qualifiedValueShapesDisjoint is true, a value that conforms to a
#     qualified value shape and also to an in-scope sibling shape contributes to
#     neither candidate's count (it is never counted on both sides).
#   * Sibling shapes are scoped by sh:property: qualified value shapes declared on
#     the same property shape are not siblings of one another.
#   * A conformance check that cannot produce evidence (recursion back-out,
#     out-of-scope sibling) is not treated as conformance and does not fill a
#     qualified count slot; min/max count comparisons still run.

import rdflib
from rdflib.namespace import SH

import pyshacl

PREFIX = """\
@prefix ex: <http://example.org/> .
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
"""


def _results_for(graph, source_shape, component):
    source = rdflib.URIRef(f"http://example.org/{source_shape}")
    return [
        result
        for result in graph.subjects(SH.sourceConstraintComponent, component)
        if (result, SH.sourceShape, source) in graph
    ]


def _validate(shapes_tail, data_tail, sparql_mode=False):
    shapes = rdflib.Graph().parse(data=PREFIX + shapes_tail, format="turtle")
    data = rdflib.Graph().parse(data=PREFIX + data_tail, format="turtle")
    return pyshacl.validate(
        data,
        shacl_graph=shapes,
        debug=False,
        sparql_mode=sparql_mode,
    )


DISJOINT_TWO_CANDIDATES_SHAPES = """
ex:HandShape a sh:NodeShape ; sh:targetClass ex:Hand ;
  sh:property ex:Qthumb ;
  sh:property ex:Qfinger .
ex:Qthumb a sh:PropertyShape ; sh:path ex:digit ;
  sh:qualifiedValueShape [ sh:class ex:Thumb ] ;
  sh:qualifiedMinCount 1 ; sh:qualifiedMaxCount 1 ;
  sh:qualifiedValueShapesDisjoint true .
ex:Qfinger a sh:PropertyShape ; sh:path ex:digit ;
  sh:qualifiedValueShape [ sh:class ex:Finger ] ;
  sh:qualifiedMinCount 1 ; sh:qualifiedMaxCount 4 ;
  sh:qualifiedValueShapesDisjoint true .
"""

SAME_Q_TWO_VALUE_SHAPES = """
ex:ThingShape a sh:NodeShape ; sh:targetClass ex:Thing ;
  sh:property ex:Q .
ex:Q a sh:PropertyShape ; sh:path ex:p ;
  sh:qualifiedValueShape ex:S1 ;
  sh:qualifiedValueShape ex:S2 ;
  sh:qualifiedMinCount 1 ;
  sh:qualifiedValueShapesDisjoint true .
ex:S1 a sh:NodeShape ; sh:class ex:A .
ex:S2 a sh:NodeShape ; sh:class ex:B .
"""

REUSED_Q_SHAPES = """
ex:ThingShape a sh:NodeShape ; sh:targetClass ex:Thing ;
  sh:property ex:Q ;
  sh:property ex:Q2 .
ex:OtherShape a sh:NodeShape ; sh:targetClass ex:Other ;
  sh:property ex:Q .
ex:Q a sh:PropertyShape ; sh:path ex:p ;
  sh:qualifiedValueShape ex:S1 ; sh:qualifiedMinCount 1 ;
  sh:qualifiedValueShapesDisjoint true .
ex:Q2 a sh:PropertyShape ; sh:path ex:p ;
  sh:qualifiedValueShape ex:S2 ; sh:qualifiedMinCount 1 ;
  sh:qualifiedValueShapesDisjoint true .
ex:S1 a sh:NodeShape ; sh:class ex:A .
ex:S2 a sh:NodeShape ; sh:class ex:B .
"""

RECURSIVE_SHAPES = """
ex:ThingShape a sh:NodeShape ; sh:targetClass ex:Thing ;
  sh:property ex:Qa ;
  sh:property ex:Qb .
ex:Qa a sh:PropertyShape ; sh:path ex:p ;
  sh:qualifiedValueShape ex:ShapeA ;
  sh:qualifiedMinCount 1 ;
  sh:qualifiedValueShapesDisjoint true .
ex:Qb a sh:PropertyShape ; sh:path ex:p ;
  sh:qualifiedValueShape ex:ShapeB ;
  sh:qualifiedMinCount 1 ;
  sh:qualifiedValueShapesDisjoint true .
ex:ShapeA a sh:NodeShape ; sh:class ex:A ;
  sh:property ex:Qa .
ex:ShapeB a sh:NodeShape ; sh:class ex:B .
"""


def test_disjoint_value_fitting_both_candidates_counts_for_neither():
    data = """
ex:h1 a ex:Hand ; ex:digit ex:both .
ex:both a ex:Thumb, ex:Finger .
"""
    conforms, report, _text = _validate(DISJOINT_TWO_CANDIDATES_SHAPES, data)
    assert not conforms
    thumb_violations = _results_for(report, "Qthumb", SH.QualifiedMinCountConstraintComponent)
    finger_violations = _results_for(report, "Qfinger", SH.QualifiedMinCountConstraintComponent)
    # The single value fits both candidates, so it may not be consumed by either count.
    assert len(thumb_violations) == 1
    assert len(finger_violations) == 1


def test_disjoint_value_fitting_both_candidates_sparql_mode():
    data = """
ex:h1 a ex:Hand ; ex:digit ex:both .
ex:both a ex:Thumb, ex:Finger .
"""
    conforms, report, _text = _validate(DISJOINT_TWO_CANDIDATES_SHAPES, data, sparql_mode=True)
    assert not conforms
    assert len(_results_for(report, "Qthumb", SH.QualifiedMinCountConstraintComponent)) == 1
    assert len(_results_for(report, "Qfinger", SH.QualifiedMinCountConstraintComponent)) == 1


def test_value_shapes_on_same_property_shape_are_not_siblings():
    data = """
ex:t1 a ex:Thing ; ex:p ex:v1 .
ex:v1 a ex:A, ex:B .
"""
    conforms, report, _text = _validate(SAME_Q_TWO_VALUE_SHAPES, data)
    # No sh:property sibling exists for Q, so the both-fitting value counts for S1 and S2.
    assert conforms
    assert len(list(report.subjects(SH.sourceConstraintComponent, SH.QualifiedMinCountConstraintComponent))) == 0


def test_value_shapes_on_same_property_shape_are_not_siblings_sparql_mode():
    data = """
ex:t1 a ex:Thing ; ex:p ex:v1 .
ex:v1 a ex:A, ex:B .
"""
    conforms, _report, _text = _validate(SAME_Q_TWO_VALUE_SHAPES, data, sparql_mode=True)
    assert conforms


def test_sibling_scope_follows_property_shape_parents():
    data = """
ex:t1 a ex:Thing ; ex:p ex:v1 .
ex:v1 a ex:A, ex:B .
"""
    conforms, report, _text = _validate(REUSED_Q_SHAPES, data)
    # Q is reached from ex:ThingShape, where Q2 makes S2 an in-scope sibling of S1.
    assert not conforms
    assert len(_results_for(report, "Q", SH.QualifiedMinCountConstraintComponent)) == 1
    assert len(_results_for(report, "Q2", SH.QualifiedMinCountConstraintComponent)) == 1


def test_recursive_qualified_check_backout_still_reports_min_count():
    data = """
ex:t1 a ex:Thing ; ex:p ex:v1 .
ex:v1 a ex:A .
"""
    conforms, report, _text = _validate(RECURSIVE_SHAPES, data)
    # Backing out of the recursive conformance check must not silently satisfy minCount.
    assert not conforms
    assert len(_results_for(report, "Qa", SH.QualifiedMinCountConstraintComponent)) == 1
    assert len(_results_for(report, "Qb", SH.QualifiedMinCountConstraintComponent)) == 1


if __name__ == "__main__":
    test_disjoint_value_fitting_both_candidates_counts_for_neither()
    test_disjoint_value_fitting_both_candidates_sparql_mode()
    test_value_shapes_on_same_property_shape_are_not_siblings()
    test_value_shapes_on_same_property_shape_are_not_siblings_sparql_mode()
    test_sibling_scope_follows_property_shape_parents()
    test_recursive_qualified_check_backout_still_reports_min_count()
