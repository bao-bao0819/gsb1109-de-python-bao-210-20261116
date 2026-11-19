# -*- coding: utf-8 -*-
import logging

import pytest
import rdflib
from rdflib.namespace import RDF, SH

from pyshacl.validator_conformance import compare_validation_reports

SH_NS = "http://www.w3.org/ns/shacl#"
EX_NS = "http://example.org/"


def _report_node(graph):
    nodes = set(graph.subjects(RDF.type, SH.ValidationReport))
    assert len(nodes) == 1
    return next(iter(nodes))


def _parse(turtle_text):
    graph = rdflib.Graph()
    graph.parse(data=turtle_text, format="turtle")
    return graph


def _single_result_report(*, bnode_id, inner_name, component, message):
    return """\
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.org/> .
[]
  a sh:ValidationReport ;
  sh:conforms false ;
  sh:result [
      a sh:ValidationResult ;
      sh:focusNode _:{bnode_id} ;
      sh:value _:{bnode_id} ;
      sh:resultPath ex:p ;
      sh:resultSeverity sh:Violation ;
      sh:sourceConstraintComponent sh:{component} ;
      sh:sourceShape ex:TestShape ;
      sh:resultMessage "{message}" ;
    ] .
_:{bnode_id}
  rdf:type ex:Thing ;
  ex:name "{inner_name}" .
""".format(bnode_id=bnode_id, inner_name=inner_name, component=component, message=message)


def test_renamed_blank_nodes_match_on_structure():
    expected = _parse(_single_result_report(
        bnode_id="expected1", inner_name="alpha", component="MinCountConstraintComponent", message="same"
    ))
    report = _parse(_single_result_report(
        bnode_id="actual9", inner_name="alpha", component="MinCountConstraintComponent", message="same"
    ))
    assert compare_validation_reports(
        report, expected, _report_node(expected), logging.getLogger()
    )


def test_symmetric_blank_node_subgraphs_do_not_hard_match():
    # Same top-level focus/value blank nodes, but the structure attached to the
    # blank node differs. Renaming the nodes must not hide that real change.
    expected = _parse(_single_result_report(
        bnode_id="b1", inner_name="alpha", component="MinCountConstraintComponent", message="same"
    ))
    report = _parse(_single_result_report(
        bnode_id="b1", inner_name="beta", component="MinCountConstraintComponent", message="same"
    ))
    assert not compare_validation_reports(
        report, expected, _report_node(expected), logging.getLogger()
    )


def test_same_message_different_constraint_is_different_result():
    expected = _parse(_single_result_report(
        bnode_id="b1", inner_name="alpha", component="ClassConstraintComponent", message="identical message"
    ))
    report = _parse(_single_result_report(
        bnode_id="b1", inner_name="alpha", component="DatatypeConstraintComponent", message="identical message"
    ))
    assert not compare_validation_reports(
        report, expected, _report_node(expected), logging.getLogger()
    )


_TWO_RESULTS_TEMPLATE = """\
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.org/> .
[]
  a sh:ValidationReport ;
  sh:conforms false ;
  sh:result [
      a sh:ValidationResult ;
      sh:focusNode ex:kept ;
      sh:value "kept" ;
      sh:resultPath ex:q ;
      sh:resultSeverity sh:Violation ;
      sh:sourceConstraintComponent sh:ClassConstraintComponent ;
      sh:sourceShape ex:TestShape ;
    ] ;
  sh:result [
      a sh:ValidationResult ;
      sh:focusNode {removed_or_added_focus} ;
      sh:value "other" ;
      sh:resultPath ex:r ;
      sh:resultSeverity sh:Violation ;
      sh:sourceConstraintComponent sh:MinCountConstraintComponent ;
      sh:sourceShape ex:TestShape ;
    ] .
"""


def test_only_genuinely_added_and_removed_results_are_reported(caplog):
    expected = _parse(_TWO_RESULTS_TEMPLATE.format(removed_or_added_focus="ex:removed"))
    report = _parse(_TWO_RESULTS_TEMPLATE.format(removed_or_added_focus="ex:added"))
    with caplog.at_level(logging.ERROR):
        matches = compare_validation_reports(
            report, expected, _report_node(expected), logging.getLogger()
        )
    assert not matches
    full_log = caplog.text
    assert "Expected validation result not found" in full_log
    assert "ex:removed" in full_log
    assert "Unexpected validation result present" in full_log
    assert "ex:added" in full_log
    # The unchanged shared result must not be reported as a change.
    assert "ex:kept" not in full_log


_DUPLICATE_RESULTS_TEMPLATE = """\
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.org/> .
[]
  a sh:ValidationReport ;
  sh:conforms false ;
  {result_blocks} .
"""

_RESULT_BLOCK = """\
sh:result [
      a sh:ValidationResult ;
      sh:focusNode ex:shared ;
      sh:value ex:k ;
      sh:resultPath ex:r ;
      sh:resultSeverity sh:Violation ;
      sh:sourceConstraintComponent sh:ClassConstraintComponent ;
      sh:sourceShape ex:S ;
    ]"""


def test_duplicate_result_multiplicity_is_respected():
    two_results = _DUPLICATE_RESULTS_TEMPLATE.format(
        result_blocks=" ;\n  ".join([_RESULT_BLOCK, _RESULT_BLOCK])
    )
    one_result = _DUPLICATE_RESULTS_TEMPLATE.format(result_blocks=_RESULT_BLOCK)
    expected_graph = _parse(two_results)
    report_graph = _parse(one_result)
    assert not compare_validation_reports(
        report_graph, expected_graph, _report_node(expected_graph), logging.getLogger()
    )

    equal_report = _parse(
        _DUPLICATE_RESULTS_TEMPLATE.format(
            result_blocks=" ;\n  ".join([_RESULT_BLOCK, _RESULT_BLOCK])
        )
    )
    assert compare_validation_reports(
        equal_report, expected_graph, _report_node(expected_graph), logging.getLogger()
    )


_PATH_REPORT_TEMPLATE = """\
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.org/> .
[]
  a sh:ValidationReport ;
  sh:conforms false ;
  sh:result [
      a sh:ValidationResult ;
      sh:focusNode ex:i ;
      sh:value "v" ;
      sh:resultPath {path_node} ;
      sh:resultSeverity sh:Violation ;
      sh:sourceConstraintComponent sh:MinCountConstraintComponent ;
      sh:sourceShape ex:TestShape ;
    ] .
{path_body}
"""


def test_blank_node_result_path_matches_on_structure():
    expected_path = "_:p1"
    expected_body = "_:p1 sh:inversePath ex:name ."
    report_path = "_:p2"
    report_body = "_:p2 sh:inversePath ex:name ."
    expected = _parse(_PATH_REPORT_TEMPLATE.format(path_node=expected_path, path_body=expected_body))
    report = _parse(_PATH_REPORT_TEMPLATE.format(path_node=report_path, path_body=report_body))
    assert compare_validation_reports(
        report, expected, _report_node(expected), logging.getLogger()
    )

    different_body = "_:p2 sh:alternativePath ( ex:name ex:label ) ."
    different_report = _parse(
        _PATH_REPORT_TEMPLATE.format(path_node=report_path, path_body=different_body)
    )
    assert not compare_validation_reports(
        different_report, expected, _report_node(expected), logging.getLogger()
    )


if __name__ == "__main__":
    pytest.main([__file__])
