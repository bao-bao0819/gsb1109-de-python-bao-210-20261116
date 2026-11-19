# -*- coding: utf-8 -*-
#
"""
Tests for pyshacl.sharded: cross-shard validation of sh:minCount,
sh:maxCount and sh:uniqueLang on direct property paths.
"""
import pytest
import rdflib

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import RDF, XSD

import pyshacl

from pyshacl.consts import SH
from pyshacl.errors import ReportableRuntimeError, UnsupportedShapeError
from pyshacl.sharded import LocalShard, ShardCoordinator
from pyshacl.sharded.identity import default_bnode_convention


EX = "http://example.com/"
SHAPES_PREFIXES = """\
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.com/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
"""

MAXCOUNT_SHAPES = SHAPES_PREFIXES + """\
ex:ThingShape a sh:NodeShape ;
    sh:targetClass ex:Thing ;
    sh:property [ sh:path ex:prop ; sh:maxCount 2 ] .
"""

MINCOUNT_SHAPES = SHAPES_PREFIXES + """\
ex:ThingShape a sh:NodeShape ;
    sh:targetClass ex:Thing ;
    sh:property [ sh:path ex:prop ; sh:minCount 2 ] .
"""

UNIQUELANG_SHAPES = SHAPES_PREFIXES + """\
ex:ThingShape a sh:NodeShape ;
    sh:targetClass ex:Thing ;
    sh:property [ sh:path ex:label ; sh:uniqueLang true ] .
"""


def merged_validate(shapes_graph, shard_docs):
    """Validate the graph composed from the same source documents."""
    merged = Graph()
    for doc in shard_docs:
        merged.parse(data=doc, format="turtle")
    return pyshacl.validate(merged, shacl_graph=shapes_graph)


def extract_results(report_graph):
    """Structural view of a validation report, for parity comparison."""
    extracted = []
    for result in report_graph.subjects(RDF.type, SH.ValidationResult):
        value_node = report_graph.value(result, SH.value)
        extracted.append(
            (
                str(report_graph.value(result, SH.focusNode)),
                str(report_graph.value(result, SH.sourceShape)),
                str(report_graph.value(result, SH.sourceConstraintComponent)),
                str(report_graph.value(result, SH.resultPath)),
                str(report_graph.value(result, SH.resultSeverity)),
                tuple(sorted(str(m) for m in report_graph.objects(result, SH.resultMessage))),
                str(value_node) if value_node is not None else None,
            )
        )
    return sorted(extracted)


def sharded_validate(shapes_graph, shards, bnode_convention=None):
    coord = ShardCoordinator(shapes_graph, bnode_convention=bnode_convention)
    for shard_id, version, doc in shards:
        coord.update_shard(shard_id, version, doc)
    return coord.validate()


def assert_parity(shapes_ttl, shard_docs, shards=None, bnode_convention=None):
    """The sharded result must match validating the composed graph."""
    shapes_graph = Graph()
    shapes_graph.parse(data=shapes_ttl, format="turtle")
    if shards is None:
        shards = [("shard-{}".format(i), "v1", doc) for i, doc in enumerate(shard_docs)]
    expected_conforms, expected_graph, expected_text = merged_validate(shapes_graph, shard_docs)
    result = sharded_validate(shapes_graph, shards, bnode_convention=bnode_convention)
    assert result.conforms == expected_conforms
    assert extract_results(result.report_graph) == extract_results(expected_graph)
    assert result.report_text == expected_text
    return result


# ----------------------------------------------------------------------
# maxCount / minCount across shards
# ----------------------------------------------------------------------


def test_maxcount_passes_per_shard_but_violated_when_merged():
    shard_a = "@prefix ex: <http://example.com/> . ex:n1 a ex:Thing ; ex:prop \"a\", \"b\" ."
    shard_b = "@prefix ex: <http://example.com/> . ex:n1 ex:prop \"c\" ."
    # Each shard alone has at most 2 values, the union has 3.
    result = assert_parity(MAXCOUNT_SHAPES, [shard_a, shard_b])
    assert result.conforms is False
    assert len(result.violations) == 1
    evidence = result.violations[0]
    assert evidence.kind == 'maxCount'
    assert evidence.focus_node == URIRef(EX + "n1")
    assert evidence.result_path == URIRef(EX + "prop")
    # Values were fetched per contributing source only after the violation.
    fetched = {sid: sorted(str(v) for v in vals) for sid, vals in evidence.values_by_shard.items()}
    assert fetched == {"shard-0": ["a", "b"], "shard-1": ["c"]}
    assert evidence.shard_versions == {"shard-0": "v1", "shard-1": "v1"}


def test_maxcount_conforms_when_merged_count_within_limit():
    shard_a = "@prefix ex: <http://example.com/> . ex:n1 a ex:Thing ; ex:prop \"a\" ."
    shard_b = "@prefix ex: <http://example.com/> . ex:n1 ex:prop \"b\" ."
    result = assert_parity(MAXCOUNT_SHAPES, [shard_a, shard_b])
    assert result.conforms is True
    assert result.violations == ()


def test_mincount_satisfied_only_by_merging_shards():
    shard_a = "@prefix ex: <http://example.com/> . ex:n1 a ex:Thing ; ex:prop \"a\" ."
    shard_b = "@prefix ex: <http://example.com/> . ex:n1 ex:prop \"b\" ."
    result = assert_parity(MINCOUNT_SHAPES, [shard_a, shard_b])
    assert result.conforms is True


def test_mincount_violated_when_merged_count_too_small():
    shard_a = "@prefix ex: <http://example.com/> . ex:n1 a ex:Thing ; ex:prop \"a\" ."
    shard_b = "@prefix ex: <http://example.com/> . ex:n2 a ex:Thing ."
    result = assert_parity(MINCOUNT_SHAPES, [shard_a, shard_b])
    assert result.conforms is False
    # ex:n1 has only one value (minCount 2), ex:n2 has none at all.
    kinds = {(v.kind, str(v.focus_node)) for v in result.violations}
    assert kinds == {('minCount', EX + "n1"), ('minCount', EX + "n2")}
    n2_evidence = [v for v in result.violations if str(v.focus_node) == EX + "n2"][0]
    assert n2_evidence.values_by_shard == {}


def test_mincount_target_node_without_any_data():
    shapes = SHAPES_PREFIXES + """\
    ex:NodeShape a sh:NodeShape ;
        sh:targetNode ex:absent ;
        sh:property [ sh:path ex:prop ; sh:minCount 1 ] .
    """
    shard_a = "@prefix ex: <http://example.com/> . ex:other ex:prop \"x\" ."
    result = assert_parity(shapes, [shard_a])
    assert result.conforms is False
    assert str(result.violations[0].focus_node) == EX + "absent"


# ----------------------------------------------------------------------
# precise RDF value identity in summaries
# ----------------------------------------------------------------------


def test_same_value_sent_by_two_shards_counted_once():
    shard_a = "@prefix ex: <http://example.com/> . ex:n1 a ex:Thing ; ex:prop \"a\", \"b\" ."
    shard_b = "@prefix ex: <http://example.com/> . ex:n1 ex:prop \"a\" ."
    # "a" appears in both shards: the merged graph has 2 values, not 3.
    result = assert_parity(MAXCOUNT_SHAPES, [shard_a, shard_b])
    assert result.conforms is True


def test_literal_identity_uses_normalised_lexical_form():
    shapes = SHAPES_PREFIXES + """\
    ex:ThingShape a sh:NodeShape ;
        sh:targetNode ex:n1 ;
        sh:property [ sh:path ex:num ; sh:maxCount 1 ] .
    """
    shard_a = """@prefix ex: <http://example.com/> . @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
        ex:n1 ex:num "01"^^xsd:integer ."""
    shard_b = """@prefix ex: <http://example.com/> . @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
        ex:n1 ex:num "1"^^xsd:integer ."""
    # rdflib normalises "01"^^xsd:integer to "1"^^xsd:integer: one value.
    result = assert_parity(shapes, [shard_a, shard_b])
    assert result.conforms is True


def test_plain_literal_and_xsd_string_are_distinct_values():
    shapes = SHAPES_PREFIXES + """\
    ex:ThingShape a sh:NodeShape ;
        sh:targetNode ex:n1 ;
        sh:property [ sh:path ex:prop ; sh:maxCount 1 ] .
    """
    shard_a = "@prefix ex: <http://example.com/> . ex:n1 ex:prop \"a\" ."
    shard_b = """@prefix ex: <http://example.com/> . @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
        ex:n1 ex:prop "a"^^xsd:string ."""
    # Matches rdflib term identity (plain literal != xsd:string typed literal).
    result = assert_parity(shapes, [shard_a, shard_b])
    assert result.conforms is False


def test_language_tag_case_does_not_change_identity():
    shapes = SHAPES_PREFIXES + """\
    ex:ThingShape a sh:NodeShape ;
        sh:targetNode ex:n1 ;
        sh:property [ sh:path ex:label ; sh:maxCount 1 ] .
    """
    shard_a = "@prefix ex: <http://example.com/> . ex:n1 ex:label \"hi\"@en ."
    shard_b = "@prefix ex: <http://example.com/> . ex:n1 ex:label \"hi\"@EN ."
    result = assert_parity(shapes, [shard_a, shard_b])
    assert result.conforms is True


# ----------------------------------------------------------------------
# uniqueLang across shards
# ----------------------------------------------------------------------


def test_uniquelang_violated_across_shards():
    shard_a = "@prefix ex: <http://example.com/> . ex:n1 a ex:Thing ; ex:label \"hello\"@en ."
    shard_b = "@prefix ex: <http://example.com/> . ex:n1 ex:label \"world\"@en ."
    result = assert_parity(UNIQUELANG_SHAPES, [shard_a, shard_b])
    assert result.conforms is False
    assert len(result.violations) == 1
    evidence = result.violations[0]
    assert evidence.kind == 'uniqueLang'
    assert evidence.lang == "en"
    fetched = {sid: sorted(str(v) for v in vals) for sid, vals in evidence.values_by_shard.items()}
    assert fetched == {"shard-0": ["hello"], "shard-1": ["world"]}


def test_uniquelang_same_literal_resent_is_not_a_duplicate():
    shard_a = "@prefix ex: <http://example.com/> . ex:n1 a ex:Thing ; ex:label \"hello\"@en ."
    shard_b = "@prefix ex: <http://example.com/> . ex:n1 ex:label \"hello\"@en ."
    result = assert_parity(UNIQUELANG_SHAPES, [shard_a, shard_b])
    assert result.conforms is True


def test_uniquelang_language_match_is_case_insensitive():
    shard_a = "@prefix ex: <http://example.com/> . ex:n1 a ex:Thing ; ex:label \"hello\"@EN ."
    shard_b = "@prefix ex: <http://example.com/> . ex:n1 ex:label \"world\"@en ."
    result = assert_parity(UNIQUELANG_SHAPES, [shard_a, shard_b])
    assert result.conforms is False


def test_uniquelang_one_result_per_duplicated_language():
    shard_a = "@prefix ex: <http://example.com/> . ex:n1 a ex:Thing ; ex:label \"a\"@en, \"b\"@de ."
    shard_b = "@prefix ex: <http://example.com/> . ex:n1 ex:label \"c\"@en, \"d\"@de ."
    result = assert_parity(UNIQUELANG_SHAPES, [shard_a, shard_b])
    assert result.conforms is False
    assert sorted(v.lang for v in result.violations) == ["de", "en"]


# ----------------------------------------------------------------------
# blank node identity convention
# ----------------------------------------------------------------------


def test_bnodes_from_different_shards_are_distinct_by_default():
    shapes = SHAPES_PREFIXES + """\
    ex:ThingShape a sh:NodeShape ;
        sh:targetNode ex:n1 ;
        sh:property [ sh:path ex:prop ; sh:maxCount 1 ] .
    """
    shard_a = "@prefix ex: <http://example.com/> . ex:n1 ex:prop _:shared ."
    shard_b = "@prefix ex: <http://example.com/> . ex:n1 ex:prop _:shared ."
    # Default convention scopes blank nodes to their shard: two values.
    result = assert_parity(shapes, [shard_a, shard_b])
    assert result.conforms is False


def test_bnode_convention_can_share_identity_across_shards():
    shapes = SHAPES_PREFIXES + """\
    ex:ThingShape a sh:NodeShape ;
        sh:targetNode ex:n1 ;
        sh:property [ sh:path ex:prop ; sh:maxCount 1 ] .
    """
    # Blank node labels are source-local identifiers; build the shards with
    # controlled labels (a turtle parse would mint fresh random ones).
    def make_shard():
        graph = Graph()
        graph.add((URIRef(EX + "n1"), URIRef(EX + "prop"), BNode("shared")))
        return graph

    shared_labels = lambda source_id, label: label  # noqa: E731
    result = sharded_validate(
        shapes, [("s1", "v1", make_shard()), ("s2", "v1", make_shard())], bnode_convention=shared_labels
    )
    # With the input convention, _:shared is one and the same node.
    assert result.conforms is True
    # And that matches a composed graph where the node really is shared.
    merged = Graph()
    shared = BNode("shared")
    merged.add((URIRef(EX + "n1"), URIRef(EX + "prop"), shared))
    merged.add((URIRef(EX + "n1"), URIRef(EX + "prop"), shared))
    conforms, _g, _t = pyshacl.validate(merged, shacl_graph=shapes)
    assert conforms is True


def test_bnode_focus_nodes_follow_the_convention():
    shapes = SHAPES_PREFIXES + """\
    ex:ThingShape a sh:NodeShape ;
        sh:targetClass ex:Thing ;
        sh:property [ sh:path ex:prop ; sh:maxCount 1 ] .
    """

    def make_shard(value):
        graph = Graph()
        item = BNode("item")
        graph.add((item, RDF.type, URIRef(EX + "Thing")))
        graph.add((item, URIRef(EX + "prop"), Literal(value)))
        return graph

    shard_a, shard_b = make_shard("a"), make_shard("b")
    # Default: the two shards describe two different blank nodes.
    default_result = sharded_validate(shapes, [("s1", "v1", shard_a), ("s2", "v1", shard_b)])
    assert default_result.conforms is True
    # Shared-label convention: one focus node with two values.
    shared_labels = lambda source_id, label: label  # noqa: E731
    shared_result = sharded_validate(
        shapes, [("s1", "v1", make_shard("a")), ("s2", "v1", make_shard("b"))], bnode_convention=shared_labels
    )
    assert shared_result.conforms is False
    assert len(shared_result.violations) == 1
    # The composed graph with a genuinely shared blank node agrees.
    merged = Graph()
    shared = BNode()
    merged.add((shared, RDF.type, URIRef(EX + "Thing")))
    merged.add((shared, URIRef(EX + "prop"), Literal("a")))
    merged.add((shared, URIRef(EX + "prop"), Literal("b")))
    conforms, report_graph, _t = pyshacl.validate(merged, shacl_graph=shapes)
    assert conforms is False
    assert len(list(report_graph.subjects(RDF.type, SH.ValidationResult))) == 1


def test_default_bnode_convention_scopes_to_source():
    assert default_bnode_convention("s1", "b0") != default_bnode_convention("s2", "b0")
    assert default_bnode_convention("s1", "b0") == default_bnode_convention("s1", "b0")


# ----------------------------------------------------------------------
# versioning: re-send, replace, retract
# ----------------------------------------------------------------------


def test_resending_same_version_does_not_double_count():
    coord = ShardCoordinator(MAXCOUNT_SHAPES)
    doc = "@prefix ex: <http://example.com/> . ex:n1 a ex:Thing ; ex:prop \"a\", \"b\" ."
    coord.update_shard("s1", "v1", doc)
    assert coord.validate().conforms is True
    coord.update_shard("s1", "v1", doc)
    # A naive additive ingest would now count 4 values and fail.
    assert coord.validate().conforms is True
    assert coord.shard_versions == {"s1": "v1"}


def test_new_version_replaces_old_contribution():
    coord = ShardCoordinator(MAXCOUNT_SHAPES)
    coord.update_shard("s1", "v1", "@prefix ex: <http://example.com/> . ex:n1 a ex:Thing ; ex:prop \"a\" .")
    coord.update_shard("s2", "v1", "@prefix ex: <http://example.com/> . ex:n1 ex:prop \"b\", \"c\" .")
    assert coord.validate().conforms is False
    # s2 publishes a new version with only one value: the old one exits.
    coord.update_shard("s2", "v2", "@prefix ex: <http://example.com/> . ex:n1 ex:prop \"b\" .")
    result = coord.validate()
    assert result.conforms is True
    assert coord.shard_versions == {"s1": "v1", "s2": "v2"}


def test_retract_withdraws_contribution():
    coord = ShardCoordinator(MINCOUNT_SHAPES)
    coord.update_shard("s1", "v1", "@prefix ex: <http://example.com/> . ex:n1 a ex:Thing ; ex:prop \"a\" .")
    coord.update_shard("s2", "v1", "@prefix ex: <http://example.com/> . ex:n1 ex:prop \"b\" .")
    assert coord.validate().conforms is True
    coord.retract_shard("s2")
    # Only "a" remains: minCount 2 is no longer met.
    assert coord.validate().conforms is False
    coord.retract_shard("s1")
    # No shards left: nothing is targeted, so nothing violates.
    assert coord.validate().conforms is True
    with pytest.raises(ReportableRuntimeError):
        coord.retract_shard("s2")


# ----------------------------------------------------------------------
# targets
# ----------------------------------------------------------------------


def test_target_class_with_cross_shard_subclass_hierarchy():
    shapes = SHAPES_PREFIXES + """\
    ex:ThingShape a sh:NodeShape ;
        sh:targetClass ex:Thing ;
        sh:property [ sh:path ex:prop ; sh:minCount 1 ] .
    """
    # The subclass triple and the instance live in different shards.
    shard_a = "@prefix ex: <http://example.com/> . ex:n1 a ex:SpecialThing ."
    shard_b = """@prefix ex: <http://example.com/> .
        ex:SpecialThing a <http://www.w3.org/2000/01/rdf-schema#Class> ;
            <http://www.w3.org/2000/01/rdf-schema#subClassOf> ex:Thing ."""
    result = assert_parity(shapes, [shard_a, shard_b])
    assert result.conforms is False
    assert str(result.violations[0].focus_node) == EX + "n1"


def test_target_subjects_of_and_objects_of():
    shapes = SHAPES_PREFIXES + """\
    ex:SubjectShape a sh:NodeShape ;
        sh:targetSubjectsOf ex:flag ;
        sh:property [ sh:path ex:prop ; sh:minCount 1 ] .
    ex:ObjectShape a sh:NodeShape ;
        sh:targetObjectsOf ex:ref ;
        sh:property [ sh:path ex:prop ; sh:maxCount 0 ] .
    """
    shard_a = "@prefix ex: <http://example.com/> . ex:s1 ex:flag true ."
    shard_b = "@prefix ex: <http://example.com/> . ex:s2 ex:ref ex:o1 . ex:o1 ex:prop \"x\" ."
    result = assert_parity(shapes, [shard_a, shard_b])
    assert result.conforms is False
    kinds = {(v.kind, str(v.focus_node)) for v in result.violations}
    assert kinds == {('minCount', EX + "s1"), ('maxCount', EX + "o1")}


def test_standalone_property_shape_with_own_targets():
    shapes = SHAPES_PREFIXES + """\
    ex:PropShape a sh:PropertyShape ;
        sh:targetNode ex:n1 ;
        sh:path ex:prop ;
        sh:maxCount 1 .
    """
    shard_a = "@prefix ex: <http://example.com/> . ex:n1 ex:prop \"a\" ."
    shard_b = "@prefix ex: <http://example.com/> . ex:n1 ex:prop \"b\" ."
    result = assert_parity(shapes, [shard_a, shard_b])
    assert result.conforms is False


def test_custom_message_and_severity_are_preserved():
    shapes = SHAPES_PREFIXES + """\
    ex:ThingShape a sh:NodeShape ;
        sh:targetNode ex:n1 ;
        sh:property [
            sh:path ex:prop ;
            sh:maxCount 1 ;
            sh:message "Too many props!" ;
            sh:severity sh:Warning ;
        ] .
    """
    shard_a = "@prefix ex: <http://example.com/> . ex:n1 ex:prop \"a\" ."
    shard_b = "@prefix ex: <http://example.com/> . ex:n1 ex:prop \"b\" ."
    result = assert_parity(shapes, [shard_a, shard_b])
    assert result.conforms is False
    messages = list(result.report_graph.objects(predicate=SH.resultMessage))
    assert [str(m) for m in messages] == ["Too many props!"]
    severities = list(result.report_graph.objects(predicate=SH.resultSeverity))
    assert severities == [SH.Warning]


def test_deactivated_shapes_are_skipped():
    shapes = SHAPES_PREFIXES + """\
    ex:ThingShape a sh:NodeShape ;
        sh:targetNode ex:n1 ;
        sh:property [
            sh:path ex:prop ;
            sh:maxCount 1 ;
            sh:deactivated true ;
        ] .
    """
    shard_a = "@prefix ex: <http://example.com/> . ex:n1 ex:prop \"a\", \"b\" ."
    result = assert_parity(shapes, [shard_a])
    assert result.conforms is True


# ----------------------------------------------------------------------
# unsupported shapes must not silently pass
# ----------------------------------------------------------------------


def test_unsupported_constraint_raises():
    shapes = SHAPES_PREFIXES + """\
    ex:ThingShape a sh:NodeShape ;
        sh:targetNode ex:n1 ;
        sh:property [ sh:path ex:prop ; sh:pattern "^a" ] .
    """
    with pytest.raises(UnsupportedShapeError):
        ShardCoordinator(shapes)


def test_unsupported_complex_path_raises():
    shapes = SHAPES_PREFIXES + """\
    ex:ThingShape a sh:NodeShape ;
        sh:targetNode ex:n1 ;
        sh:property [ sh:path ( ex:a ex:b ) ; sh:maxCount 1 ] .
    """
    with pytest.raises(UnsupportedShapeError):
        ShardCoordinator(shapes)


def test_unsupported_node_shape_constraint_raises():
    shapes = SHAPES_PREFIXES + """\
    ex:ThingShape a sh:NodeShape ;
        sh:targetNode ex:n1 ;
        sh:in ( "a" "b" ) .
    """
    with pytest.raises(UnsupportedShapeError):
        ShardCoordinator(shapes)


def test_min_count_on_node_shape_raises():
    shapes = SHAPES_PREFIXES + """\
    ex:ThingShape a sh:NodeShape ;
        sh:targetNode ex:n1 ;
        sh:minCount 1 .
    """
    with pytest.raises(UnsupportedShapeError):
        ShardCoordinator(shapes)


def test_sparql_constraint_raises():
    shapes = SHAPES_PREFIXES + """\
    ex:ThingShape a sh:NodeShape ;
        sh:targetNode ex:n1 ;
        sh:sparql [ sh:select "SELECT $this WHERE { $this ex:prop ?v }" ] .
    """
    with pytest.raises(UnsupportedShapeError):
        ShardCoordinator(shapes)


def test_nested_property_shape_raises():
    shapes = SHAPES_PREFIXES + """\
    ex:ThingShape a sh:NodeShape ;
        sh:targetNode ex:n1 ;
        sh:property [
            sh:path ex:prop ;
            sh:property [ sh:path ex:other ; sh:maxCount 1 ] ;
        ] .
    """
    with pytest.raises(UnsupportedShapeError):
        ShardCoordinator(shapes)


def test_deactivated_unsupported_shape_is_ignored():
    shapes = SHAPES_PREFIXES + """\
    ex:ThingShape a sh:NodeShape ;
        sh:targetNode ex:n1 ;
        sh:deactivated true ;
        sh:in ( "a" "b" ) .
    """
    coord = ShardCoordinator(shapes)
    coord.update_shard("s1", "v1", "@prefix ex: <http://example.com/> . ex:n1 ex:prop \"x\" .")
    assert coord.validate().conforms is True


# ----------------------------------------------------------------------
# summaries do not move source data; fetches happen only on violation
# ----------------------------------------------------------------------


class CountingShard(LocalShard):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fetch_calls = 0

    def fetch_values(self, predicate, focus_identity, digests):
        self.fetch_calls += 1
        return super().fetch_values(predicate, focus_identity, digests)


def make_counting_shard(shard_id, version, doc):
    graph = Graph()
    graph.parse(data=doc, format="turtle")
    return CountingShard(shard_id, version, graph)


def test_no_value_fetch_when_conforming():
    coord = ShardCoordinator(MAXCOUNT_SHAPES)
    shard = make_counting_shard("s1", "v1", "@prefix ex: <http://example.com/> . ex:n1 a ex:Thing ; ex:prop \"a\" .")
    coord.add_shard(shard)
    assert coord.validate().conforms is True
    assert shard.fetch_calls == 0


def test_values_fetched_per_source_on_violation():
    coord = ShardCoordinator(MAXCOUNT_SHAPES)
    shard_a = make_counting_shard("s1", "v1", "@prefix ex: <http://example.com/> . ex:n1 a ex:Thing ; ex:prop \"a\", \"b\" .")
    shard_b = make_counting_shard("s2", "v1", "@prefix ex: <http://example.com/> . ex:n1 ex:prop \"c\" .")
    coord.add_shard(shard_a)
    coord.add_shard(shard_b)
    result = coord.validate()
    assert result.conforms is False
    assert shard_a.fetch_calls == 1
    assert shard_b.fetch_calls == 1
    evidence = result.violations[0]
    assert set(evidence.values_by_shard) == {"s1", "s2"}


def test_summary_contains_digests_not_terms():
    coord = ShardCoordinator(MAXCOUNT_SHAPES)
    shard = make_counting_shard("s1", "v1", "@prefix ex: <http://example.com/> . ex:n1 a ex:Thing ; ex:prop \"secret\" .")
    coord.add_shard(shard)
    summary = shard.value_summary({EX + "prop": {"U:" + EX + "n1"}})
    records = summary[EX + "prop"]["U:" + EX + "n1"]
    assert len(records) == 1
    digest, lang = next(iter(records))
    assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
    assert "secret" not in digest
    assert lang is None


# ----------------------------------------------------------------------
# larger parity scenario
# ----------------------------------------------------------------------


def test_parity_mixed_scenario():
    shapes = SHAPES_PREFIXES + """\
    ex:PersonShape a sh:NodeShape ;
        sh:targetClass ex:Person ;
        sh:property [ sh:path ex:name ; sh:minCount 1 ; sh:maxCount 3 ] ;
        sh:property [ sh:path ex:label ; sh:uniqueLang true ] .
    ex:TeamShape a sh:NodeShape ;
        sh:targetNode ex:team1 ;
        sh:property [ sh:path ex:member ; sh:minCount 2 ; sh:maxCount 3 ] .
    """
    shard_a = """@prefix ex: <http://example.com/> .
        ex:p1 a ex:Person ; ex:name "Ann" ; ex:label "Ann"@en .
        ex:p2 a ex:Person ; ex:label "Burt"@en .
        ex:team1 ex:member ex:p1 ."""
    shard_b = """@prefix ex: <http://example.com/> .
        ex:p1 ex:name "Annabelle" ; ex:label "Annelies"@nl .
        ex:p2 ex:label "Bert"@nl .
        ex:team1 ex:member ex:p2 ."""
    shard_c = """@prefix ex: <http://example.com/> .
        ex:p1 ex:name "A." , "Annie" ; ex:label "A."@en .
        ex:team1 ex:member ex:p3 , ex:p4 ."""
    result = assert_parity(shapes, [shard_a, shard_b, shard_c])
    assert result.conforms is False
    found = {(v.kind, str(v.focus_node)) for v in result.violations}
    # ex:p1 has 4 ex:name values (maxCount 3) and two @en labels.
    assert ('maxCount', EX + "p1") in found
    assert ('uniqueLang', EX + "p1") in found
    # ex:p2 has no ex:name anywhere.
    assert ('minCount', EX + "p2") in found
    # ex:team1 has 4 members (maxCount 3).
    assert ('maxCount', EX + "team1") in found


def test_result_unpacks_like_pyshacl():
    coord = ShardCoordinator(MAXCOUNT_SHAPES)
    coord.update_shard("s1", "v1", "@prefix ex: <http://example.com/> . ex:n1 a ex:Thing ; ex:prop \"a\" .")
    conforms, report_graph, report_text = coord.validate()
    assert conforms is True
    assert isinstance(report_graph, rdflib.Graph)
    assert isinstance(report_text, str)
    assert list(report_graph.objects(None, SH.conforms)) == [Literal(True)]


def test_repeated_validate_calls_are_consistent():
    coord = ShardCoordinator(MAXCOUNT_SHAPES)
    coord.update_shard("s1", "v1", "@prefix ex: <http://example.com/> . ex:n1 a ex:Thing ; ex:prop \"a\", \"b\", \"c\" .")
    first = coord.validate()
    second = coord.validate()
    assert first.conforms == second.conforms == False  # noqa: E712
    assert extract_results(first.report_graph) == extract_results(second.report_graph)


def test_implicit_class_target():
    shapes = SHAPES_PREFIXES + """\
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    ex:Thing a sh:NodeShape, rdfs:Class ;
        sh:property [ sh:path ex:prop ; sh:minCount 1 ] .
    """
    shard_a = "@prefix ex: <http://example.com/> . ex:n1 a ex:Thing ."
    result = assert_parity(shapes, [shard_a])
    assert result.conforms is False
    assert str(result.violations[0].focus_node) == EX + "n1"


def test_literal_focus_node_via_target_node():
    shapes = SHAPES_PREFIXES + """\
    ex:NodeShape a sh:NodeShape ;
        sh:targetNode "literal-focus" ;
        sh:property [ sh:path ex:prop ; sh:minCount 1 ] .
    """
    shard_a = "@prefix ex: <http://example.com/> . ex:n1 ex:prop \"x\" ."
    result = assert_parity(shapes, [shard_a])
    assert result.conforms is False
    assert result.violations[0].focus_node == Literal("literal-focus")


def test_min_count_zero_always_passes():
    shapes = SHAPES_PREFIXES + """\
    ex:ThingShape a sh:NodeShape ;
        sh:targetNode ex:n1 ;
        sh:property [ sh:path ex:prop ; sh:minCount 0 ] .
    """
    shard_a = "@prefix ex: <http://example.com/> . ex:n1 ex:other \"x\" ."
    result = assert_parity(shapes, [shard_a])
    assert result.conforms is True


def test_unique_lang_false_is_inert():
    shapes = SHAPES_PREFIXES + """\
    ex:ThingShape a sh:NodeShape ;
        sh:targetNode ex:n1 ;
        sh:property [ sh:path ex:label ; sh:uniqueLang false ] .
    """
    shard_a = "@prefix ex: <http://example.com/> . ex:n1 ex:label \"a\"@en ."
    shard_b = "@prefix ex: <http://example.com/> . ex:n1 ex:label \"b\"@en ."
    result = assert_parity(shapes, [shard_a, shard_b])
    assert result.conforms is True


def test_same_property_shape_referenced_by_two_node_shapes():
    shapes = SHAPES_PREFIXES + """\
    ex:ThingShape a sh:NodeShape ;
        sh:targetClass ex:Thing ;
        sh:property ex:PropShape .
    ex:OtherShape a sh:NodeShape ;
        sh:targetNode ex:n1 ;
        sh:property ex:PropShape .
    ex:PropShape a sh:PropertyShape ;
        sh:path ex:prop ;
        sh:maxCount 1 .
    """
    shard_a = "@prefix ex: <http://example.com/> . ex:n1 a ex:Thing ; ex:prop \"a\" ."
    shard_b = "@prefix ex: <http://example.com/> . ex:n1 ex:prop \"b\" ."
    # Both node shapes evaluate ex:PropShape against ex:n1: two results.
    result = assert_parity(shapes, [shard_a, shard_b])
    assert result.conforms is False
    assert len(result.violations) == 2
