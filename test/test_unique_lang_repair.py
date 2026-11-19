# -*- coding: utf-8 -*-
#
import rdflib
from rdflib import Literal, Namespace

from pyshacl.repair import (
    BASE_SOURCE,
    LanguageEquivalence,
    find_unique_lang_conflicts,
    repair_unique_lang,
)


EX = Namespace("http://example.org/")

TARGETED_SHAPE_TTL = """
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.org/> .
ex:LabelShape a sh:PropertyShape ;
    sh:path ex:label ;
    sh:targetNode ex:item ;
    sh:uniqueLang true .
"""

MULTI_TARGET_SHAPE_TTL = """
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.org/> .
ex:LabelShape a sh:PropertyShape ;
    sh:path ex:label ;
    sh:targetNode ex:item1, ex:item2 ;
    sh:uniqueLang true .
"""

NODE_SHAPE_TTL = """
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.org/> .
ex:PersonShape a sh:NodeShape ;
    sh:targetClass ex:Person ;
    sh:property [ sh:path ex:label ; sh:uniqueLang true ] .
"""


def shapes_graph(ttl=TARGETED_SHAPE_TTL):
    return rdflib.Graph().parse(data=ttl, format="turtle")


def data_graph(*triples):
    graph = rdflib.Graph()
    for triple in triples:
        graph.add(triple)
    return graph


def label(value, lang):
    return (EX.item, EX.label, Literal(value, lang=lang))


class TestLanguageEquivalence:
    def test_case_insensitive_by_default(self):
        equiv = LanguageEquivalence()
        assert equiv.equivalent("EN", "en")
        assert not equiv.equivalent("en", "en-US")

    def test_alias_mapping(self):
        equiv = LanguageEquivalence({"en-US": "en"})
        assert equiv.equivalent("en-US", "en")
        assert equiv.canonical("EN-us") == "en"
        assert not equiv.equivalent("en", "de")

    def test_alias_groups_are_transitive(self):
        equiv = LanguageEquivalence([["en", "en-US"], ["en-US", "en-GB"]])
        assert equiv.equivalent("en", "en-GB")
        assert equiv.canonical("en-GB") == "en"


class TestFindConflicts:
    def test_same_tag_different_literals_conflict(self):
        data = data_graph(label("Apple", "en"), label("Apfel", "en"))
        groups = find_unique_lang_conflicts(data, shapes_graph())
        assert len(groups) == 1
        assert groups[0].language == "en"
        assert set(groups[0].values) == {Literal("Apple", lang="en"), Literal("Apfel", lang="en")}

    def test_distinct_tags_do_not_conflict(self):
        data = data_graph(label("Apple", "en"), label("Apfel", "de"))
        assert find_unique_lang_conflicts(data, shapes_graph()) == []

    def test_aliases_merge_tags_into_one_group(self):
        data = data_graph(label("Apple", "en"), label("Apfel", "en-US"))
        assert find_unique_lang_conflicts(data, shapes_graph()) == []
        groups = find_unique_lang_conflicts(data, shapes_graph(), lang_aliases={"en-US": "en"})
        assert len(groups) == 1
        assert groups[0].language == "en"

    def test_same_literal_twice_is_not_a_conflict(self):
        # Identical (s, p, o) triples collapse in the graph, so one value only.
        data = data_graph(label("Apple", "en"), (EX.item, EX.title, Literal("Apple", lang="en")))
        assert find_unique_lang_conflicts(data, shapes_graph()) == []

    def test_property_shape_referenced_via_sh_property(self):
        data = data_graph(
            (EX.item, rdflib.RDF.type, EX.Person),
            label("Apple", "en"),
            label("Apfel", "en"),
        )
        groups = find_unique_lang_conflicts(data, shapes_graph(NODE_SHAPE_TTL))
        assert len(groups) == 1
        assert groups[0].focus_node == EX.item

    def test_literals_without_language_are_ignored(self):
        data = data_graph(label("Apple", "en"), (EX.item, EX.label, Literal("plain")))
        assert find_unique_lang_conflicts(data, shapes_graph()) == []


class TestRepair:
    def test_min_cost_source_is_retracted(self):
        apple, apfel = label("Apple", "en"), label("Apfel", "en")
        data = data_graph(apple, apfel)
        result = repair_unique_lang(
            data,
            shapes_graph(),
            supports={apple: {"cheap"}, apfel: {"expensive"}},
            costs={"cheap": 1.0, "expensive": 5.0},
        )
        assert result.feasible
        assert result.retracted_sources == {"cheap"}
        assert result.total_cost == 1.0
        assert apple in result.removed_triples
        assert apfel in result.repaired_graph
        assert apple not in result.repaired_graph
        # The input graph is never modified.
        assert len(data) == 2

    def test_multi_supported_fact_survives_partial_retraction(self):
        # "Apple" is backed by two sources; retracting one of them must not
        # count as deleting the fact.
        apple, apfel = label("Apple", "en"), label("Apfel", "en")
        data = data_graph(apple, apfel)
        result = repair_unique_lang(
            data,
            shapes_graph(),
            supports={apple: {"a", "b"}, apfel: {"b"}},
            costs={"a": 5.0, "b": 1.0},
        )
        assert result.feasible
        # Eliminating "Apple" would cost a+b=6; eliminating "Apfel" costs b=1.
        assert result.retracted_sources == {"b"}
        assert apple in result.repaired_graph  # still supported by "a"
        assert apfel not in result.repaired_graph

    def test_shared_source_is_paid_once_across_groups(self):
        data = data_graph(
            (EX.item1, EX.label, Literal("X", lang="en")),
            (EX.item1, EX.label, Literal("Y", lang="en")),
            (EX.item2, EX.label, Literal("P", lang="en")),
            (EX.item2, EX.label, Literal("Q", lang="en")),
        )
        supports = {
            (EX.item1, EX.label, Literal("X", lang="en")): {"a"},
            (EX.item1, EX.label, Literal("Y", lang="en")): {"b"},
            (EX.item2, EX.label, Literal("P", lang="en")): {"c"},
            (EX.item2, EX.label, Literal("Q", lang="en")): {"b"},
        }
        result = repair_unique_lang(
            data,
            shapes_graph(MULTI_TARGET_SHAPE_TTL),
            supports=supports,
            costs={"a": 1.0, "b": 1.5, "c": 1.0},
        )
        assert result.feasible
        # Retracting "b" alone (1.5) beats retracting "a" and "c" (2.0).
        assert result.retracted_sources == {"b"}
        assert result.total_cost == 1.5
        assert len(result.repaired_graph) == 2

    def test_unprovenanced_facts_are_locked(self):
        apple, apfel = label("Apple", "en"), label("Apfel", "en")
        data = data_graph(apple, apfel)
        result = repair_unique_lang(data, shapes_graph(), supports={apfel: {"a"}})
        assert result.feasible
        # "Apple" has no provenance, so only "Apfel" can be eliminated.
        assert result.retracted_sources == {"a"}
        assert apple in result.repaired_graph
        assert apfel not in result.repaired_graph

    def test_infeasible_reports_locked_evidence_and_keeps_graph(self):
        apple, apfel = label("Apple", "en"), label("Apfel", "en")
        data = data_graph(apple, apfel)
        result = repair_unique_lang(
            data,
            shapes_graph(),
            supports={apple: {"locked1"}, apfel: {"locked2"}},
            locked_sources={"locked1", "locked2"},
        )
        assert not result.feasible
        assert result.repaired_graph is None
        assert result.retracted_sources == frozenset()
        assert len(result.locked_evidence) == 1
        evidence = result.locked_evidence[0]
        assert set(evidence.locked_values) == {apple[2], apfel[2]}
        assert evidence.locking_sources == {"locked1", "locked2"}
        assert len(data) == 2  # original graph unchanged

    def test_infeasible_when_both_facts_lack_provenance(self):
        data = data_graph(label("Apple", "en"), label("Apfel", "en"))
        result = repair_unique_lang(data, shapes_graph())
        assert not result.feasible
        assert result.locked_evidence[0].locking_sources == {BASE_SOURCE}
        assert result.repaired_graph is None

    def test_retraction_removes_other_facts_that_lose_all_support(self):
        apple, apfel = label("Apple", "en"), label("Apfel", "en")
        comment = (EX.item, EX.comment, Literal("side fact"))
        data = data_graph(apple, apfel, comment)
        result = repair_unique_lang(
            data,
            shapes_graph(),
            supports={apple: {"a"}, apfel: {"b"}, comment: {"b"}},
            costs={"a": 2.0, "b": 1.0},
        )
        assert result.feasible
        assert result.retracted_sources == {"b"}
        # The side fact shared source "b", so it disappears together with it.
        assert comment in result.removed_triples
        assert comment not in result.repaired_graph

    def test_plan_only_skips_graph_materialisation(self):
        apple, apfel = label("Apple", "en"), label("Apfel", "en")
        data = data_graph(apple, apfel)
        result = repair_unique_lang(
            data,
            shapes_graph(),
            supports={apple: {"a"}, apfel: {"b"}},
            plan_only=True,
        )
        assert result.feasible
        assert result.repaired_graph is None
        assert result.retracted_sources in ({"a"}, {"b"})

    def test_no_conflicts_is_trivially_feasible(self):
        data = data_graph(label("Apple", "en"), label("Apfel", "de"))
        result = repair_unique_lang(data, shapes_graph())
        assert result.feasible
        assert result.total_cost == 0.0
        assert result.retracted_sources == frozenset()
        assert len(result.repaired_graph) == 2

    def test_negative_cost_is_rejected(self):
        data = data_graph(label("Apple", "en"), label("Apfel", "en"))
        try:
            repair_unique_lang(data, shapes_graph(), costs={"a": -1.0})
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError("negative costs must be rejected")
