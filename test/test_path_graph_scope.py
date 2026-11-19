# -*- coding: utf-8 -*-
"""
SHACL property path evaluation over multigraph datasets.

A path must keep every complete traversal inside one data-graph scope: edges
living in different named graphs are not stitched together into a single path.
Validation results that cannot be reproduced on one graph still name the graph
boundary involved, while single-graph and default-graph-only data keep the
standard validation report untouched.
"""
from rdflib import Dataset, Graph, Literal, Namespace, URIRef
from rdflib.graph import DATASET_DEFAULT_GRAPH_ID
from rdflib.namespace import SH

import pyshacl
from pyshacl.consts import PYSHACL_sourceGraph

EX = Namespace("http://example.org/")
G1 = URIRef("http://example.org/graph1")
G2 = URIRef("http://example.org/graph2")


def _two_graph_dataset():
    """G1 holds alice->bob, G2 holds the continuation bob->carol."""
    ds = Dataset()
    ds.graph(G1).parse(
        data="""
    @prefix ex: <http://example.org/> .
    ex:alice a ex:Person ; ex:knows ex:bob ; ex:name "Alice" .
    ex:bob a ex:Person ; ex:name "Bob" .
    """,
        format="turtle",
    )
    ds.graph(G2).parse(
        data="""
    @prefix ex: <http://example.org/> .
    ex:bob ex:knows ex:carol .
    ex:carol a ex:Person ; ex:name "Carol" .
    """,
        format="turtle",
    )
    return ds


def _single_graph_chain_dataset():
    """The full alice->bob->zoe chain lives inside G1."""
    ds = Dataset()
    ds.graph(G1).parse(
        data="""
    @prefix ex: <http://example.org/> .
    ex:alice a ex:Person ; ex:knows ex:bob .
    ex:bob a ex:Person ; ex:knows ex:zoe .
    ex:zoe a ex:Person ; ex:name "Zoe" .
    """,
        format="turtle",
    )
    ds.graph(G2).parse(
        data="""
    @prefix ex: <http://example.org/> .
    ex:bob ex:knows ex:carol .
    """,
        format="turtle",
    )
    return ds


def _source_graphs(report_graph):
    return set(report_graph.objects(None, PYSHACL_sourceGraph))


def test_sequence_path_does_not_stitch_across_named_graphs():
    shapes = """\
    @prefix ex: <http://example.org/> .
    @prefix sh: <http://www.w3.org/ns/shacl#> .
    ex:PersonShape a sh:NodeShape ;
        sh:targetClass ex:Person ;
        sh:property [ sh:path ( ex:knows ex:knows ) ; sh:maxCount 0 ] .
    """
    # Without scoping, G1's alice->bob plus G2's bob->carol would yield one
    # stitched value and violate maxCount 0.
    conforms, _report, text = pyshacl.validate(_two_graph_dataset(), shacl_graph=shapes)
    assert conforms, text


def test_sequence_path_still_completes_inside_one_named_graph():
    shapes = """\
    @prefix ex: <http://example.org/> .
    @prefix sh: <http://www.w3.org/ns/shacl#> .
    ex:PersonShape a sh:NodeShape ;
        sh:targetNode ex:alice ;
        sh:property [ sh:path ( ex:knows ex:knows ) ; sh:minCount 1 ] .
    """
    # The G1-internal chain alice->bob->zoe is a valid two-hop path and must
    # still be found; scoping must not over-restrict same-graph traversal.
    conforms, _report, text = pyshacl.validate(_single_graph_chain_dataset(), shacl_graph=shapes)
    assert conforms, text


def test_unreproducible_path_names_the_crossed_graph_boundary():
    shapes = """\
    @prefix ex: <http://example.org/> .
    @prefix sh: <http://www.w3.org/ns/shacl#> .
    ex:PersonShape a sh:NodeShape ;
        sh:targetClass ex:Person ;
        sh:property [ sh:path ( ex:knows ex:knows ) ; sh:minCount 1 ;
            sh:message "two-hop path must be reproducible in one graph" ] .
    """
    conforms, report_graph, text = pyshacl.validate(_two_graph_dataset(), shacl_graph=shapes)
    assert not conforms
    sources = _source_graphs(report_graph)
    # The failed attempt touched both halves, so the report names both graphs.
    assert G1 in sources
    assert G2 in sources
    assert "Source Graph:" in text


def test_alternative_paths_follow_the_same_scope_convention():
    shapes = """\
    @prefix ex: <http://example.org/> .
    @prefix sh: <http://www.w3.org/ns/shacl#> .
    ex:PersonShape a sh:NodeShape ;
        sh:targetClass ex:Person ;
        # name lives in G1, label lives in G2; each alternative resolves on its
        # own graph and the union must not invent cross-graph edges.
        sh:property [ sh:path [ sh:alternativePath ( ex:name ex:label ) ] ;
            sh:minCount 1 ] .
    """
    ds = _two_graph_dataset()
    ds.graph(G2).add((EX.alice, EX.label, Literal("Alice label")))
    conforms, _report, text = pyshacl.validate(ds, shacl_graph=shapes)
    assert conforms, text


def test_zero_length_path_steps_remain_unscoped():
    # zeroOrMore always includes the focus node via its zero-length step.
    shapes = """\
    @prefix ex: <http://example.org/> .
    @prefix sh: <http://www.w3.org/ns/shacl#> .
    ex:PersonShape a sh:NodeShape ;
        sh:targetNode ex:alice ;
        sh:property [ sh:path [ sh:zeroOrMorePath ex:knows ] ; sh:node ex:Reachable ] .
    ex:Reachable a sh:NodeShape ;
        sh:property [ sh:path ex:name ; sh:minCount 1 ] .
    """
    ds = Dataset()
    # bob is reachable inside G1 but has no name; carol is only reachable by
    # stitching the G1 alice->bob edge with the G2 bob->carol edge.
    ds.graph(G1).parse(
        data="""
    @prefix ex: <http://example.org/> .
    ex:alice a ex:Person ; ex:knows ex:bob .
    ex:bob a ex:Person .
    """,
        format="turtle",
    )
    ds.graph(G2).parse(
        data="""
    @prefix ex: <http://example.org/> .
    ex:bob ex:knows ex:carol .
    ex:carol a ex:Person ; ex:name "Carol" .
    """,
        format="turtle",
    )
    conforms, report_graph, _text = pyshacl.validate(ds, shacl_graph=shapes)
    assert not conforms
    focus_nodes = set(report_graph.objects(None, SH.focusNode))
    # bob is reached inside G1 and correctly fails for its missing name.
    assert EX.bob in focus_nodes
    # carol would only be reachable by crossing into G2, so it never appears.
    assert EX.carol not in focus_nodes


def test_zero_or_one_and_one_or_more_do_not_cross_boundary():
    shapes = """\
    @prefix ex: <http://example.org/> .
    @prefix sh: <http://www.w3.org/ns/shacl#> .
    ex:PersonShape a sh:NodeShape ;
        sh:targetClass ex:Person ;
        sh:property [ sh:path [ sh:oneOrMorePath ex:knows ] ; sh:maxCount 1 ] ;
        sh:property [ sh:path [ sh:zeroOrOnePath ex:nick ] ; sh:maxCount 1 ] .
    """
    # From alice, oneOrMore knows reaches only bob inside G1 (carol needs the
    # G2 edge); maxCount 1 therefore holds.
    conforms, _report, text = pyshacl.validate(_two_graph_dataset(), shacl_graph=shapes)
    assert conforms, text


def test_inverse_path_respects_graph_scope():
    shapes = """\
    @prefix ex: <http://example.org/> .
    @prefix sh: <http://www.w3.org/ns/shacl#> .
    ex:PersonShape a sh:NodeShape ;
        sh:targetClass ex:Person ;
        # Inverse knows of carol is bob, but that edge lives only in G2; the
        # G1-scoped lookup for carol must return nothing.
        sh:property [ sh:path [ sh:inversePath ex:knows ] ; sh:maxCount 1 ] .
    """
    conforms, _report, text = pyshacl.validate(_two_graph_dataset(), shacl_graph=shapes)
    assert conforms, text


def test_default_graph_only_dataset_emits_no_provenance_extension():
    ds = Dataset()
    ds.default_graph.add((EX.node, EX.prop, Literal("value")))
    shapes = """\
    @prefix ex: <http://example.org/> .
    @prefix sh: <http://www.w3.org/ns/shacl#> .
    ex:Shape a sh:NodeShape ; sh:targetNode ex:node ;
        sh:property [ sh:path ex:prop ; sh:minCount 2 ] .
    """
    conforms, report_graph, _text = pyshacl.validate(ds, shacl_graph=shapes)
    assert not conforms
    assert _source_graphs(report_graph) == set()
    # The default graph identifier itself must never be misreported as a scope.
    assert not any(s == DATASET_DEFAULT_GRAPH_ID for s in _source_graphs(report_graph))


def test_plain_graph_report_is_unaffected():
    data = Graph()
    data.add((EX.node, EX.prop, Literal("value")))
    shapes = """\
    @prefix ex: <http://example.org/> .
    @prefix sh: <http://www.w3.org/ns/shacl#> .
    ex:Shape a sh:NodeShape ; sh:targetNode ex:node ;
        sh:property [ sh:path ex:prop ; sh:minCount 2 ] .
    """
    conforms, report_graph, _text = pyshacl.validate(data, shacl_graph=shapes)
    assert not conforms
    assert _source_graphs(report_graph) == set()
