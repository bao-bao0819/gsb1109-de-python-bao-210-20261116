# -*- coding: utf-8 -*-
"""
Tests for the release-gate decision layer (pyshacl.release_gate).

These cover the release policy: Warnings are allowed but Violations block the
release; nested sh:detail results inherit severity; the minimal blocking
evidence set is produced while every detail is retained; exemptions are scoped
to declared objects and time windows and never propagate; results reached via
multiple detail paths are counted once; and changing the time or exemptions
only re-judges the affected objects without ever collapsing warnings into a
single conforms flag.
"""

from datetime import datetime, timedelta, timezone

import pytest
from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF

from pyshacl import validate
from pyshacl.consts import (
    SH_conforms,
    SH_detail,
    SH_focusNode,
    SH_path,
    SH_result,
    SH_resultMessage,
    SH_resultPath,
    SH_resultSeverity,
    SH_sourceConstraintComponent,
    SH_sourceShape,
    SH_ValidationReport,
    SH_ValidationResult,
)
from pyshacl.release_gate import (
    Exemption,
    ReleaseGate,
    Severity,
    build_release_gate,
)

SH = Namespace("http://www.w3.org/ns/shacl#")
EX = Namespace("http://example.org/")

NOW = datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc)
PAST = NOW - timedelta(days=1)
FUTURE = NOW + timedelta(days=1)


def _add_result(graph, node, focus, shape, severity, path=None, message=None, details=()):
    graph.add((node, RDF.type, SH_ValidationResult))
    graph.add((node, SH_focusNode, focus))
    graph.add((node, SH_sourceShape, shape))
    graph.add((node, SH_resultSeverity, severity))
    graph.add((node, SH_sourceConstraintComponent, URIRef("urn:cc:test")))
    if path is not None:
        graph.add((node, SH_resultPath, path))
    if message is not None:
        graph.add((node, SH_resultMessage, Literal(message)))
    for detail in details:
        graph.add((node, SH_detail, detail))
    return node


def _report(graph, conforms, roots):
    report = BNode()
    graph.add((report, RDF.type, SH_ValidationReport))
    graph.add((report, SH_conforms, Literal(conforms)))
    for root in roots:
        graph.add((report, SH_result, root))
    return graph


# --------------------------------------------------------------------- policy

def test_warning_allowed_violation_blocked():
    graph = Graph()
    warn = BNode()
    viol = BNode()
    _add_result(graph, warn, EX.Bob, EX.WarnShape, SH.Warning, path=EX.tag, message="w")
    _add_result(graph, viol, EX.Bob, EX.ViolShape, SH.Violation, path=EX.age, message="v")
    _report(graph, conforms=False, roots=[warn, viol])

    decision = ReleaseGate(graph).evaluate(at=NOW)

    assert decision.allowed is False
    assert decision.blocked is True
    assert decision.counts["violation"] == 1
    assert decision.counts["warning"] == 1
    assert len(decision.blocking_evidence) == 1
    assert decision.blocking_evidence[0].focus_node == EX.Bob
    # Both results are retained; the warning is not erased by the verdict.
    assert len(decision.results) == 2
    assert len(decision.warnings) == 1


def test_only_warnings_release_is_allowed():
    graph = Graph()
    warn = BNode()
    info = BNode()
    _add_result(graph, warn, EX.Bob, EX.WarnShape, SH.Warning, message="w")
    _add_result(graph, info, EX.Bob, EX.InfoShape, SH.Info, message="i")
    _report(graph, conforms=False, roots=[warn, info])

    decision = ReleaseGate(graph).evaluate(at=NOW)

    assert decision.allowed is True
    assert decision.blocking_evidence == []
    assert decision.counts["warning"] == 1
    assert decision.counts["info"] == 1
    # Allowed does not mean the warnings disappeared.
    assert {r.focus_node for r in decision.warnings} == {EX.Bob}


def test_conforms_true_does_not_erase_warnings():
    # A report may carry sh:conforms true while still listing warning results.
    # The gate must keep them and make its own independent release decision.
    graph = Graph()
    warn = BNode()
    _add_result(graph, warn, EX.Carol, EX.WarnShape, SH.Warning, message="still here")
    _report(graph, conforms=True, roots=[warn])

    decision = ReleaseGate(graph).evaluate(at=NOW)

    assert decision.conforms is True
    assert decision.allowed is True
    assert len(decision.results) == 1
    assert decision.results[0].severity is Severity.WARNING
    assert decision.results[0].effective_severity is Severity.WARNING
    assert decision.warnings[0].messages == ("still here",)


# ------------------------------------------------------------- nested details

def test_nested_detail_inherits_ancestor_violation():
    graph = Graph()
    leaf = BNode()
    root = BNode()
    _add_result(graph, leaf, EX.Bob, EX.InnerShape, SH.Warning, path=EX.name, message="leaf warning")
    _add_result(graph, root, EX.Bob, EX.OuterShape, SH.Violation, message="root violation", details=[leaf])
    _report(graph, conforms=False, roots=[root])

    decision = ReleaseGate(graph).evaluate(at=NOW)

    leaf_result = decision.result_by_id(f"_:{leaf}")
    root_result = decision.result_by_id(f"_:{root}")
    # The warning leaf inherits the violation severity of its ancestor path.
    assert leaf_result.inherited_severity is Severity.VIOLATION
    assert leaf_result.effective_severity is Severity.VIOLATION
    assert root_result.blocks_release is True
    # The single topmost violation is the only minimal evidence; the inherited
    # leaf is already implied by it.
    assert [e.result_id for e in decision.blocking_evidence] == [f"_:{root}"]
    # ...but the leaf detail is still fully retained.
    assert len(decision.results) == 2
    assert leaf_result in decision.results


def test_minimal_blocking_evidence_prefers_topmost_per_branch():
    graph = Graph()
    deep = BNode()
    mid = BNode()
    root_a = BNode()
    root_b = BNode()
    _add_result(graph, deep, EX.Bob, EX.DeepShape, SH.Violation, message="deep")
    _add_result(graph, mid, EX.Bob, EX.MidShape, SH.Violation, message="mid", details=[deep])
    _add_result(graph, root_a, EX.Bob, EX.RootA, SH.Violation, message="a", details=[mid])
    _add_result(graph, root_b, EX.Alice, EX.RootB, SH.Violation, message="b")
    _report(graph, conforms=False, roots=[root_a, root_b])

    decision = ReleaseGate(graph).evaluate(at=NOW)

    # Two independent blocking branches -> two topmost pieces of evidence.
    assert {e.result_id for e in decision.blocking_evidence} == {f"_:{root_a}", f"_:{root_b}"}
    # All four results are preserved.
    assert decision.counts["total"] == 4
    assert decision.counts["violation"] == 4


def test_shared_detail_reached_via_two_paths_counted_once():
    graph = Graph()
    shared = BNode()
    parent_a = BNode()
    parent_b = BNode()
    _add_result(graph, shared, EX.Bob, EX.SharedShape, SH.Violation, path=EX.age, message="shared")
    _add_result(graph, parent_a, EX.Bob, EX.ShapeA, SH.Violation, message="a", details=[shared])
    _add_result(graph, parent_b, EX.Bob, EX.ShapeB, SH.Violation, message="b", details=[shared])
    _report(graph, conforms=False, roots=[parent_a, parent_b])

    gate = ReleaseGate(graph)
    decision = gate.evaluate(at=NOW)

    # One distinct result node => counted exactly once even with two parents.
    shared_id = f"_:{shared}"
    assert sum(1 for r in decision.results if r.result_id == shared_id) == 1
    assert decision.counts["violation"] == 3
    # The shared node is implied by both blocking parents, so neither it nor a
    # duplicate appears in the minimal evidence set.
    evidence_ids = {e.result_id for e in decision.blocking_evidence}
    assert evidence_ids == {f"_:{parent_a}", f"_:{parent_b}"}
    assert shared_id not in evidence_ids


# ------------------------------------------------------------------ exemptions

def test_exemption_is_scoped_and_does_not_propagate():
    graph = Graph()
    bob = BNode()
    alice = BNode()
    _add_result(graph, bob, EX.Bob, EX.ViolShape, SH.Violation, path=EX.age, message="bob")
    _add_result(graph, alice, EX.Alice, EX.ViolShape, SH.Violation, path=EX.age, message="alice")
    _report(graph, conforms=False, roots=[bob, alice])

    gate = ReleaseGate(graph)
    gate.add_exemption(Exemption(shape=EX.ViolShape, scope_nodes=(EX.Bob,), id="only-bob"))
    decision = gate.evaluate(at=NOW)

    bob_result = decision.result_by_id(f"_:{bob}")
    alice_result = decision.result_by_id(f"_:{alice}")
    assert bob_result.is_exempted is True
    assert alice_result.is_exempted is False
    # Alice still blocks; Bob's exemption did not spread to another object.
    assert decision.allowed is False
    assert [e.result_id for e in decision.blocking_evidence] == [f"_:{alice}"]
    assert decision.counts["exempted"] == 1
    # The exempted result is retained, merely suppressed from blocking.
    assert bob_result in decision.results


def test_exemption_respects_validity_window():
    graph = Graph()
    viol = BNode()
    _add_result(graph, viol, EX.Bob, EX.ViolShape, SH.Violation, message="v")
    _report(graph, conforms=False, roots=[viol])

    gate = ReleaseGate(graph)
    gate.add_exemption(
        Exemption(shape=EX.ViolShape, scope_nodes=(EX.Bob,), not_before=PAST, not_after=FUTURE, id="windowed")
    )

    inside = gate.evaluate(at=NOW)
    assert inside.allowed is True
    assert inside.counts["exempted"] == 1

    before = gate.evaluate(at=PAST - timedelta(days=2))
    assert before.allowed is False
    assert before.counts["exempted"] == 0

    # Half-open window: not_after is exclusive.
    on_edge = gate.evaluate(at=FUTURE)
    assert on_edge.allowed is False


def test_exemption_does_not_hide_unexempted_descendant_violation():
    graph = Graph()
    leaf = BNode()
    root = BNode()
    # Root is exempted (its shape + focus match), but the leaf uses a different
    # shape and therefore is not covered by the same exemption.
    _add_result(graph, leaf, EX.Bob, EX.LeafShape, SH.Violation, message="leaf")
    _add_result(graph, root, EX.Bob, EX.RootShape, SH.Violation, message="root", details=[leaf])
    _report(graph, conforms=False, roots=[root])

    gate = ReleaseGate(graph)
    gate.add_exemption(Exemption(shape=EX.RootShape, scope_nodes=(EX.Bob,), id="root-only"))
    decision = gate.evaluate(at=NOW)

    assert decision.result_by_id(f"_:{root}").is_exempted is True
    leaf_result = decision.result_by_id(f"_:{leaf}")
    assert leaf_result.is_exempted is False
    assert leaf_result.blocks_release is True
    assert decision.allowed is False
    assert [e.result_id for e in decision.blocking_evidence] == [f"_:{leaf}"]


def test_exemption_requires_scope_nodes():
    with pytest.raises(ValueError):
        Exemption(shape=EX.Shape, scope_nodes=())


def test_remove_exemption_restores_block():
    graph = Graph()
    viol = BNode()
    _add_result(graph, viol, EX.Bob, EX.ViolShape, SH.Violation, message="v")
    _report(graph, conforms=False, roots=[viol])

    gate = ReleaseGate(graph)
    gate.add_exemption(Exemption(shape=EX.ViolShape, scope_nodes=(EX.Bob,), id="temp"))
    assert gate.evaluate(at=NOW).allowed is True
    removed = gate.remove_exemption("temp")
    assert removed is not None
    assert gate.evaluate(at=NOW).allowed is False


# ------------------------------------------------------------- incremental

def _two_object_report():
    graph = Graph()
    bob = BNode()
    carol = BNode()
    _add_result(graph, bob, EX.Bob, EX.ViolShape, SH.Violation, path=EX.age, message="bob v")
    _add_result(graph, carol, EX.Carol, EX.WarnShape, SH.Warning, path=EX.tag, message="carol w")
    _report(graph, conforms=False, roots=[bob, carol])
    return graph, bob, carol


def test_reevaluate_time_change_only_rejudges_without_rebuild():
    graph, bob, carol = _two_object_report()
    gate = ReleaseGate(graph)
    first = gate.evaluate(at=PAST)
    carol_result = first.result_by_id(f"_:{carol}")

    gate.add_exemption(
        Exemption(shape=EX.ViolShape, scope_nodes=(EX.Bob,), not_before=NOW, not_after=FUTURE, id="future-fix")
    )
    # Before the window starts Bob still blocks.
    assert gate.evaluate(at=PAST).allowed is False
    # Moving the evaluation time flips Bob, but Carol's result identity is kept.
    later = gate.evaluate(at=NOW)
    assert later.allowed is True
    assert later.result_by_id(f"_:{carol}") is carol_result


def test_reevaluate_replaces_only_affected_focus_nodes():
    graph_v1, bob_v1, carol_v1 = _two_object_report()
    gate = ReleaseGate(graph_v1)
    before = gate.evaluate(at=NOW)
    carol_identity = before.result_by_id(f"_:{carol_v1}").identity_key()
    carol_object = before.result_by_id(f"_:{carol_v1}")
    assert before.allowed is False

    # A new report where Bob has been fixed (no Bob results) but Carol is unchanged.
    graph_v2 = Graph()
    carol_v2 = BNode()
    _add_result(graph_v2, carol_v2, EX.Carol, EX.WarnShape, SH.Warning, path=EX.tag, message="carol w")
    _report(graph_v2, conforms=True, roots=[carol_v2])

    after = gate.reevaluate(graph_v2, at=NOW, focus_nodes=[EX.Bob])

    # Bob's stale results are gone, Carol's retained object is untouched.
    assert all(r.focus_node != EX.Bob for r in after.results)
    retained_carol = after.result_by_id(f"_:{carol_v1}")
    assert retained_carol is carol_object
    assert retained_carol.identity_key() == carol_identity
    assert after.allowed is True
    assert after.conforms is True
    # The single remaining warning is still present.
    assert len(after.warnings) == 1


def test_full_reevaluate_rebuilds_everything():
    graph_v1, _bob, _carol = _two_object_report()
    gate = ReleaseGate(graph_v1)
    assert gate.evaluate(at=NOW).allowed is False

    graph_v2 = Graph()
    only_warn = BNode()
    _add_result(graph_v2, only_warn, EX.Dave, EX.WarnShape, SH.Warning, message="dave w")
    _report(graph_v2, conforms=True, roots=[only_warn])

    decision = gate.reevaluate(graph_v2, at=NOW)
    assert decision.allowed is True
    assert {r.focus_node for r in decision.results} == {EX.Dave}


# ----------------------------------------------------------- end-to-end pySHACL

def test_end_to_end_pyshacl_report_gating():
    shapes = Graph().parse(
        data="""
        @prefix sh: <http://www.w3.org/ns/shacl#> .
        @prefix ex: <http://example.org/> .
        ex:Inner a sh:NodeShape ;
          sh:property [ sh:path ex:name ; sh:minCount 1 ; sh:severity sh:Warning ] ,
                      [ sh:path ex:age ; sh:minCount 1 ; sh:severity sh:Violation ] .
        ex:Outer a sh:NodeShape ; sh:targetNode ex:Bob ;
          sh:node ex:Inner ; sh:severity sh:Violation .
        """,
        format="turtle",
    )
    data = Graph().parse(
        data="@prefix ex: <http://example.org/> . ex:Bob a ex:Person .",
        format="turtle",
    )
    conforms, report, _text = validate(data, shacl_graph=shapes)
    assert conforms is False

    gate = build_release_gate(report)
    decision = gate.evaluate(at=NOW)
    assert decision.allowed is False
    # Nested warning inherits the outer violation, so only the topmost node
    # result is required as minimal blocking evidence.
    assert len(decision.blocking_evidence) == 1
    # Every produced result (root + both details) is retained.
    assert len(decision.results) == 3

    age_shape = next(iter(report.subjects(SH_path, EX.age)))
    gate.add_exemption(
        Exemption(
            shape=age_shape,
            scope_nodes=(EX.Bob,),
            not_before=PAST,
            not_after=FUTURE,
            id="bob-age",
        )
    )
    exempted = gate.evaluate(at=NOW)
    # The parent node-constraint violation is a different shape, so the scoped
    # exemption on the age property does not clear it.
    assert exempted.allowed is False
    assert exempted.counts["exempted"] == 1

    gate.remove_exemption("bob-age")
    assert gate.evaluate(at=NOW).counts["exempted"] == 0


def test_describe_renders_decision():
    graph = Graph()
    viol = BNode()
    _add_result(graph, viol, EX.Bob, EX.ViolShape, SH.Violation, message="v")
    _report(graph, conforms=False, roots=[viol])
    text = ReleaseGate(graph).describe()
    assert "BLOCKED" in text
    assert "BLOCK" in text
