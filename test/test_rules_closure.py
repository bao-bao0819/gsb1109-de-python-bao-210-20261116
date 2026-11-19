# -*- coding: utf-8 -*-
"""Tests for ordered, whole-graph SHACL Rules closure rounds.

These tests pin down the difference between:

* a rule that only fires after a *later* shape produced a fact in a previous
  round (requires whole-graph rounds, not per-shape local iteration);
* a run that truly reached a fixed point ("converged");
* a run that hit the round budget while still adding triples ("incomplete");
* a rerun over an already-expanded graph, which must add nothing and must not
  keep modifying the original input graph.
"""

import json
from os import path

from rdflib import Graph, Namespace, RDF

import pyshacl
from pyshacl.rules import (
    CLOSURE_STATUS_CONVERGED,
    CLOSURE_STATUS_INCOMPLETE,
    CLOSURE_STATUS_NO_RULES,
    apply_rules_closure,
    RuleClosureReport,
)

EX = Namespace("http://example.com/rules-closure#")

SHAPES = """
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.com/rules-closure#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

# Earlier shape: only fires once ex:b is present.
ex:NeedsBShape
    a sh:NodeShape ;
    sh:targetClass ex:Thing ;
    sh:order 0 ;
    sh:rule [
        a sh:TripleRule ;
        sh:order 0 ;
        sh:subject sh:this ;
        sh:predicate ex:c ;
        sh:object true ;
        sh:condition [
            sh:property [
                sh:path ex:b ;
                sh:hasValue true ;
            ] ;
        ] ;
    ] .

# Later shape: produces the fact that unblocks NeedsBShape next round.
ex:ProducesBShape
    a sh:NodeShape ;
    sh:targetClass ex:Thing ;
    sh:order 1 ;
    sh:rule [
        a sh:TripleRule ;
        sh:order 0 ;
        sh:subject sh:this ;
        sh:predicate ex:b ;
        sh:object true ;
    ] .
"""

NO_RULES_SHAPES = """
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.com/rules-closure#> .

ex:NoRulesShape
    a sh:NodeShape ;
    sh:targetClass ex:Thing .
"""


def make_data_graph():
    g = Graph()
    g.add((EX.a, RDF.type, EX.Thing))
    return g


def parse_shapes(text=SHAPES):
    g = Graph()
    g.parse(data=text, format="turtle")
    return g


def run_closure(data_graph, shapes=None, **kwargs):
    kwargs.setdefault("advanced", True)
    kwargs.setdefault("iterate_rules", True)
    kwargs.setdefault("return_rules_report", True)
    return pyshacl.shacl_rules(data_graph, shacl_graph=shapes or parse_shapes(), **kwargs)


def has_triple(graph, predicate):
    return any(True for _ in graph.triples((EX.a, predicate, None)))


def test_cross_round_dependency_needs_later_shape_fact():
    data = make_data_graph()
    out, report = run_closure(data)

    assert report["status"] == CLOSURE_STATUS_CONVERGED
    assert report["converged"] is True
    assert report["incomplete"] is False
    assert report["reason"] == "no_new_triples"
    assert [(r["round"], r["added"]) for r in report["rounds"]] == [(1, 1), (2, 1), (3, 0)]
    assert has_triple(out, EX.b)
    assert has_triple(out, EX.c)


def test_non_iterating_run_does_not_trigger_cross_round_rule():
    data = make_data_graph()
    out, report = run_closure(data, iterate_rules=False)

    # No ordered rounds were requested: the earlier shape runs before ex:b
    # exists and is never revisited.
    assert report is None
    assert has_triple(out, EX.b)
    assert not has_triple(out, EX.c)


def test_input_graph_is_not_polluted_by_closure():
    data = make_data_graph()
    before = set(data)

    out, report = run_closure(data)

    assert set(data) == before
    assert len(data) == 1
    assert report["initial_size"] == 1
    assert report["final_size"] == 3
    assert report["total_added"] == 2


def test_repeated_runs_are_idempotent_and_converged():
    data = make_data_graph()
    out1, report1 = run_closure(data)
    out2, report2 = run_closure(data)

    # Running the same input twice must not accumulate anything beyond the
    # closure, and the original graph stays untouched.
    assert set(out1) == set(out2)
    assert len(data) == 1

    # Re-running an already expanded graph adds nothing and is a fixed point.
    out3, report3 = run_closure(out1)
    assert set(out3) == set(out1)
    assert report3["status"] == CLOSURE_STATUS_CONVERGED
    assert report3["total_added"] == 0
    assert report3["round_count"] == 1
    assert report1["total_added"] == 2


def test_round_limit_with_pending_changes_is_incomplete():
    # Round 1 adds ex:b, round 2 adds ex:c, round 3 is the quiet round that
    # proves the fixed point. A budget of one or two rounds must be reported
    # as incomplete rather than silently published as converged.
    _, report_one = run_closure(make_data_graph(), max_rule_rounds=1)
    assert report_one["status"] == CLOSURE_STATUS_INCOMPLETE
    assert report_one["incomplete"] is True
    assert report_one["converged"] is False
    assert report_one["reason"] == "max_rounds_exhausted"
    assert report_one["round_count"] == 1
    assert report_one["rounds"][-1]["added"] > 0

    _, report_two = run_closure(make_data_graph(), max_rule_rounds=2)
    assert report_two["status"] == CLOSURE_STATUS_INCOMPLETE
    assert report_two["round_count"] == 2
    assert report_two["rounds"][-1]["added"] > 0
    assert has_triple_from_report(report_two, EX.c)

    _, report_three = run_closure(make_data_graph(), max_rule_rounds=3)
    assert report_three["status"] == CLOSURE_STATUS_CONVERGED
    assert report_three["round_count"] == 3


def has_triple_from_report(report, predicate):
    needle = "<{}>".format(predicate)
    return any(key[1] == needle for key in report["closure"])


def test_no_rules_is_distinct_from_incomplete():
    data = make_data_graph()
    out, report = run_closure(data, shapes=parse_shapes(NO_RULES_SHAPES))

    assert report["status"] == CLOSURE_STATUS_NO_RULES
    assert report["converged"] is True
    assert report["incomplete"] is False
    assert report["total_added"] == 0
    assert set(out.triples((None, None, None))) == set(data)


def test_report_contains_round_and_rule_evidence():
    _, report = run_closure(make_data_graph())

    assert isinstance(report, RuleClosureReport)
    assert len(report["rules"]) == 2
    assert [r["shape"].rsplit("#", 1)[-1] for r in report["rules"]] == [
        "NeedsBShape",
        "ProducesBShape",
    ]

    round_one = report["rounds"][0]
    assert round_one["round"] == 1
    # Every rule that ran is recorded, including the one that found no new
    # facts, so "no new facts" is backed by evidence.
    assert [f["changed"] for f in round_one["firings"]] == [False, True]
    assert round_one["firings"][0]["shape"].endswith("NeedsBShape")
    assert round_one["firings"][1]["shape"].endswith("ProducesBShape")
    assert round_one["firings"][1]["added"] == 1
    assert len(round_one["firings"][1]["triples"]) == 1

    round_two = report["rounds"][1]
    changed_two = [f for f in round_two["firings"] if f["changed"]]
    assert [f["shape"].rsplit("#", 1)[-1] for f in changed_two] == ["NeedsBShape"]

    round_three = report["rounds"][2]
    assert round_three["added"] == 0
    assert [f["changed"] for f in round_three["firings"]] == [False, False]

    # Per-triple provenance maps each closure triple back to the round and
    # rule that first produced it.
    provenances = report["provenance"]
    assert len(provenances) == 2
    for entries in provenances.values():
        assert entries[0]["round"] in (1, 2)
        assert entries[0]["rule"].startswith("n") or entries[0]["rule"].startswith("_:")


def test_report_is_json_serializable():
    _, report = run_closure(make_data_graph())
    payload = report.to_json()
    parsed = json.loads(payload)
    assert parsed["status"] == CLOSURE_STATUS_CONVERGED
    assert parsed["max_rounds"] == 100
    assert isinstance(parsed["closure_triples"], list)
    assert isinstance(parsed["rounds"], list)


def test_apply_rules_closure_direct_api():
    from pyshacl.pytypes import SHACLExecutor
    from pyshacl.rules import gather_rules
    from pyshacl.shapes_graph import ShapesGraph

    shapes_graph = ShapesGraph(parse_shapes(), False, None)
    # The ShapesGraph harvests shapes lazily; touching the property mirrors
    # what the runner does before gathering rules.
    _ = shapes_graph.shapes
    executor = SHACLExecutor(advanced_mode=True, iterate_rules=True, max_rule_rounds=3)
    rules = gather_rules(executor, shapes_graph)

    data = make_data_graph()
    added, report = apply_rules_closure(executor, rules, data)
    assert added == 2
    assert report["status"] == CLOSURE_STATUS_CONVERGED
    assert has_triple(data, EX.c)


def test_validate_can_return_rules_report():
    data = make_data_graph()
    conforms, report_graph, report_text, rules_report = pyshacl.validate(
        data,
        shacl_graph=parse_shapes(),
        advanced=True,
        iterate_rules=True,
        inplace=True,
        return_rules_report=True,
    )
    assert rules_report["status"] == CLOSURE_STATUS_CONVERGED
    assert rules_report["total_added"] == 2
    assert has_triple(data, EX.c)

    # Without the option the public three-tuple contract is unchanged.
    data2 = make_data_graph()
    result = pyshacl.validate(
        data2, shacl_graph=parse_shapes(), advanced=True, iterate_rules=True, inplace=True
    )
    assert len(result) == 3
    assert has_triple(data2, EX.c)


def test_opaque_rule_change_is_not_treated_as_no_new_facts():
    """A rule that reports changes without exposing triples must stay pending."""
    from rdflib import Literal, URIRef

    class OpaqueRule:
        def __init__(self):
            self.node = URIRef("http://example.com/rules-closure#OpaqueRule")
            self.order = 0
            self.iterate = False
            self.deactivated = False

        def apply(self, graph, focus_nodes=None):
            triple = (EX.a, EX.opaque, Literal(True))
            self.last_added = ()
            if triple in graph:
                return 0
            graph.add(triple)
            return 1

    class ShapeStub:
        node = URIRef("http://example.com/rules-closure#OpaqueShape")

        @property
        def order(self):
            return 0

    class Executor:
        max_rule_rounds = 5
        iterate_rules = True
        rule_closure = True

    data = make_data_graph()
    added, report = apply_rules_closure(Executor(), {ShapeStub(): [OpaqueRule()]}, data)
    assert report["status"] == CLOSURE_STATUS_CONVERGED
    assert report["rounds"][0]["changed"] is True
    assert report["rounds"][0]["unattributed_added"] == 1
    assert report["unattributed_added"] == 1
    assert report["total_added"] == 0
    assert has_triple(data, EX.opaque)
