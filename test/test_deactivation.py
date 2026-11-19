# -*- coding: utf-8 -*-
#
# Tests for sh:deactivated interaction with shape references and shape
# selection. Deactivating one target entry must not change the results that
# are produced through another entry that shares a referenced child shape.

import rdflib
from rdflib.namespace import SH

from pyshacl import validate

PREFIXES = """
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.com/ex#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
"""

EX = rdflib.Namespace("http://example.com/ex#")

DATA = PREFIXES + """
ex:EntryAInstance a ex:EntryA .

ex:EntryBInstance a ex:EntryB .

ex:UnrelatedInstance a ex:Unrelated .
"""

SHAPES = PREFIXES + """
ex:EntryA a sh:NodeShape ;
    sh:targetClass ex:EntryA ;
    sh:property ex:SharedProperty .

ex:EntryB a sh:NodeShape ;
    sh:targetClass ex:EntryB ;
    sh:property ex:SharedProperty .

ex:Unrelated a sh:NodeShape ;
    sh:targetClass ex:Unrelated ;
    sh:property ex:UnrelatedProperty .

ex:SharedProperty a sh:PropertyShape ;
    sh:path ex:sharedProperty ;
    sh:minCount 1 .

ex:UnrelatedProperty a sh:PropertyShape ;
    sh:path ex:unrelatedProperty ;
    sh:minCount 1 .
"""

SHAPES_ENTRY_A_DEACTIVATED = SHAPES.replace(
    "ex:EntryA a sh:NodeShape ;\n    sh:targetClass ex:EntryA ;",
    'ex:EntryA a sh:NodeShape ;\n    sh:deactivated "true"^^xsd:boolean ;\n    sh:targetClass ex:EntryA ;',
)

SHAPES_SHARED_CHILD_DEACTIVATED = SHAPES.replace(
    "ex:SharedProperty a sh:PropertyShape ;\n    sh:path ex:sharedProperty ;",
    'ex:SharedProperty a sh:PropertyShape ;\n    sh:deactivated "true"^^xsd:boolean ;\n    sh:path ex:sharedProperty ;',
)

CHAIN_SHAPES = PREFIXES + """
ex:EntryA a sh:NodeShape ;
    sh:targetClass ex:EntryA ;
    sh:node ex:IntermediateShape .

ex:EntryB a sh:NodeShape ;
    sh:targetClass ex:EntryB ;
    sh:node ex:IntermediateShape .

ex:IntermediateShape a sh:NodeShape ;
    sh:property ex:SharedProperty .

ex:SharedProperty a sh:PropertyShape ;
    sh:path ex:sharedProperty ;
    sh:minCount 1 .
"""


def _results(graph):
    """
    Return a set of (source shape, focus node) pairs from a validation report.
    """
    found = set()
    for result in graph.subjects(rdflib.RDF.type, SH.ValidationResult):
        source_shape = graph.value(result, SH.sourceShape)
        focus_node = graph.value(result, SH.focusNode)
        found.add((source_shape, focus_node))
    return found


def _focus_nodes(results):
    return {focus_node for _, focus_node in results}


def test_deactivated_entry_does_not_deactivate_shared_child():
    conforms, report, _ = validate(
        DATA,
        shacl_graph=SHAPES_ENTRY_A_DEACTIVATED,
        data_graph_format='turtle',
        shacl_graph_format='turtle',
    )
    results = _results(report)
    assert not conforms
    assert (EX.SharedProperty, EX.EntryBInstance) in results


def test_shared_child_results_remain_for_other_entry():
    conforms, report, text = validate(
        DATA,
        shacl_graph=SHAPES_ENTRY_A_DEACTIVATED,
        data_graph_format='turtle',
        shacl_graph_format='turtle',
    )
    results = _results(report)
    assert not conforms
    assert (EX.SharedProperty, EX.EntryBInstance) in results
    assert (EX.SharedProperty, EX.EntryAInstance) not in results
    assert "Results (2)" in text


def test_toggling_one_deactivation_flag_changes_only_its_entry_results():
    conforms_off, report_off, _ = validate(
        DATA,
        shacl_graph=SHAPES,
        data_graph_format='turtle',
        shacl_graph_format='turtle',
    )
    results_off = _results(report_off)
    assert not conforms_off
    assert (EX.SharedProperty, EX.EntryAInstance) in results_off
    assert (EX.SharedProperty, EX.EntryBInstance) in results_off

    conforms_on, report_on, _ = validate(
        DATA,
        shacl_graph=SHAPES_ENTRY_A_DEACTIVATED,
        data_graph_format='turtle',
        shacl_graph_format='turtle',
    )
    results_on = _results(report_on)
    assert not conforms_on
    assert (EX.SharedProperty, EX.EntryAInstance) not in results_on
    assert (EX.SharedProperty, EX.EntryBInstance) in results_on

    conforms_back, report_back, _ = validate(
        DATA,
        shacl_graph=SHAPES,
        data_graph_format='turtle',
        shacl_graph_format='turtle',
    )
    assert not conforms_back
    assert _results(report_back) == results_off


def test_unrelated_entry_still_reports_when_one_entry_deactivated():
    conforms, report, _ = validate(
        DATA,
        shacl_graph=SHAPES_ENTRY_A_DEACTIVATED,
        data_graph_format='turtle',
        shacl_graph_format='turtle',
    )
    results = _results(report)
    assert not conforms
    assert (EX.UnrelatedProperty, EX.UnrelatedInstance) in results


def test_use_shapes_follows_referenced_child_of_other_entry():
    conforms, report, _ = validate(
        DATA,
        shacl_graph=SHAPES_ENTRY_A_DEACTIVATED,
        data_graph_format='turtle',
        shacl_graph_format='turtle',
        use_shapes=["ex:EntryA", "ex:EntryB"],
    )
    results = _results(report)
    assert not conforms
    assert (EX.SharedProperty, EX.EntryBInstance) in results
    assert (EX.SharedProperty, EX.EntryAInstance) not in results


def test_use_shapes_follows_referenced_child_when_only_other_entry_selected():
    # Same reference chain, but EntryA is not even selected. EntryB must still
    # reach the shared child shape through its own sh:property reference.
    conforms, report, _ = validate(
        DATA,
        shacl_graph=SHAPES_ENTRY_A_DEACTIVATED,
        data_graph_format='turtle',
        shacl_graph_format='turtle',
        use_shapes=["ex:EntryB"],
    )
    results = _results(report)
    assert not conforms
    assert results == {(EX.SharedProperty, EX.EntryBInstance)}


def test_use_shapes_toggle_only_changes_results_of_that_entry():
    def _selected_results(shapes):
        conforms, report, _ = validate(
            DATA,
            shacl_graph=shapes,
            data_graph_format='turtle',
            shacl_graph_format='turtle',
            use_shapes=["ex:EntryA", "ex:EntryB"],
        )
        return conforms, _results(report)

    conforms_off, results_off = _selected_results(SHAPES)
    assert not conforms_off
    assert (EX.SharedProperty, EX.EntryAInstance) in results_off
    assert (EX.SharedProperty, EX.EntryBInstance) in results_off

    conforms_on, results_on = _selected_results(SHAPES_ENTRY_A_DEACTIVATED)
    assert not conforms_on
    assert (EX.SharedProperty, EX.EntryAInstance) not in results_on
    assert (EX.SharedProperty, EX.EntryBInstance) in results_on


def test_use_shapes_follows_multi_level_reference_chain():
    # Selecting only EntryA must pull in IntermediateShape and the shape it
    # references, while EntryB remains out of scope.
    conforms, report, _ = validate(
        DATA,
        shacl_graph=CHAIN_SHAPES,
        data_graph_format='turtle',
        shacl_graph_format='turtle',
        use_shapes=["ex:EntryA"],
    )
    results = _results(report)
    assert not conforms
    assert (EX.EntryA, EX.EntryAInstance) in results
    assert (EX.SharedProperty, EX.EntryAInstance) in results
    assert (EX.SharedProperty, EX.EntryBInstance) not in results


def test_deactivating_shared_child_affects_all_entries():
    conforms, report, _ = validate(
        DATA,
        shacl_graph=SHAPES_SHARED_CHILD_DEACTIVATED,
        data_graph_format='turtle',
        shacl_graph_format='turtle',
    )
    results = _results(report)
    assert not conforms
    assert (EX.SharedProperty, EX.EntryAInstance) not in results
    assert (EX.SharedProperty, EX.EntryBInstance) not in results
    assert (EX.UnrelatedProperty, EX.UnrelatedInstance) in results


BNODE_CHILD_SHAPES = PREFIXES + """
ex:EntryA a sh:NodeShape ;
    sh:targetClass ex:EntryA ;
    sh:property _:sharedChild .

ex:EntryB a sh:NodeShape ;
    sh:targetClass ex:EntryB ;
    sh:property _:sharedChild .

_:sharedChild a sh:PropertyShape ;
    sh:path ex:sharedProperty ;
    sh:minCount 1 .
"""


def test_bnode_shared_child_remains_active_for_other_entry():
    deactivated_entry = BNODE_CHILD_SHAPES.replace(
        "ex:EntryA a sh:NodeShape ;\n    sh:targetClass ex:EntryA ;",
        'ex:EntryA a sh:NodeShape ;\n    sh:deactivated "true"^^xsd:boolean ;\n    sh:targetClass ex:EntryA ;',
    )
    conforms, report, _ = validate(
        DATA,
        shacl_graph=deactivated_entry,
        data_graph_format='turtle',
        shacl_graph_format='turtle',
    )
    results = _results(report)
    assert not conforms
    assert _focus_nodes(results) == {EX.EntryBInstance}


def test_selecting_only_deactivated_entry_produces_no_results():
    conforms, report, text = validate(
        DATA,
        shacl_graph=SHAPES_ENTRY_A_DEACTIVATED,
        data_graph_format='turtle',
        shacl_graph_format='turtle',
        use_shapes=["ex:EntryA"],
    )
    assert conforms
    assert _results(report) == set()
    assert "Results" not in text
