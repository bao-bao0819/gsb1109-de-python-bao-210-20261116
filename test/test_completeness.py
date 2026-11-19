# -*- coding: utf-8 -*-
"""
Tests for pyshacl.completeness: three-valued partial validation with
completeness declarations over entity slices.
"""
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF, XSD

from pyshacl import CompletenessDeclaration, PartialValidator, validate_partial
from pyshacl.completeness import MissingScope

EX = Namespace("http://example.org/ex#")
SH = Namespace("http://www.w3.org/ns/shacl#")

ALICE = EX.Alice
BOB = EX.Bob

SHAPES_TTL = """
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.org/ex#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

ex:PersonShape a sh:NodeShape ;
    sh:targetClass ex:Person ;
    sh:property [ sh:path ex:name ; sh:minCount 1 ; sh:maxCount 2 ; sh:datatype xsd:string ] ;
    sh:property [ sh:path ex:email ; sh:minCount 1 ] .
"""

NESTED_SHAPES_TTL = """
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.org/ex#> .

ex:PersonShape a sh:NodeShape ;
    sh:targetClass ex:Person ;
    sh:property [ sh:path ex:knows ; sh:node ex:FriendShape ] .

ex:FriendShape a sh:NodeShape ;
    sh:property [ sh:path ex:email ; sh:minCount 1 ] .
"""


def make_graph(ttl):
    graph = Graph()
    graph.parse(data=ttl, format="turtle")
    return graph


def components(items):
    return {item.component for item in items}


class TestThreeValuedSplit:
    def test_incomplete_min_count_is_unknown(self):
        data = make_graph(
            """
            @prefix ex: <http://example.org/ex#> .
            ex:Alice a ex:Person ; ex:name "Alice" .
            """
        )
        completeness = CompletenessDeclaration().declare(subject=ALICE, predicate=EX.name)
        report = validate_partial(
            data, SHAPES_TTL, focus_nodes=[ALICE], completeness=completeness
        )
        # No provable violation: the missing ex:email may live in a graph
        # that was not shipped.
        assert report.violations == ()
        assert not report.decisive
        # The ex:email minCount violation leans on incomplete evidence.
        email_unknowns = [
            u for u in report.unknowns if u.component == SH.MinCountConstraintComponent
        ]
        assert len(email_unknowns) == 1
        assert email_unknowns[0].leans == 'violation'
        assert MissingScope(ALICE, EX.email) in email_unknowns[0].missing
        # The declared-complete ex:name scopes are provably satisfied.
        name_conformances = [
            c for c in report.conformances if c.path == EX.name
        ]
        assert {c.component for c in name_conformances} == {
            SH.MinCountConstraintComponent,
            SH.MaxCountConstraintComponent,
            SH.DatatypeConstraintComponent,
        }

    def test_growth_stable_violation_is_proven_without_completeness(self):
        data = make_graph(
            """
            @prefix ex: <http://example.org/ex#> .
            ex:Alice a ex:Person ;
                ex:name "Alice", "Ally", "A." ; ex:email "a@example.org" .
            """
        )
        report = validate_partial(data, SHAPES_TTL, focus_nodes=[ALICE])
        # sh:maxCount 2 on ex:name is violated by present data; no amount of
        # additional triples can retract it, even with zero declarations.
        max_count_violations = [
            v for v in report.violations if v.component == SH.MaxCountConstraintComponent
        ]
        assert len(max_count_violations) == 1
        assert max_count_violations[0].focus == ALICE

    def test_complete_scope_turns_absence_into_proven_violation(self):
        data = make_graph(
            """
            @prefix ex: <http://example.org/ex#> .
            ex:Alice a ex:Person ; ex:name "Alice" .
            """
        )
        completeness = CompletenessDeclaration().declare(subject=ALICE, predicate=EX.email)
        report = validate_partial(
            data, SHAPES_TTL, focus_nodes=[ALICE], completeness=completeness
        )
        min_count_violations = [
            v for v in report.violations if v.component == SH.MinCountConstraintComponent
        ]
        assert len(min_count_violations) == 1
        assert min_count_violations[0].path == EX.email
        # Everything is decidable now (name scopes still open -> unknowns remain).
        assert all(
            u.path != EX.email or u.component != SH.MinCountConstraintComponent
            for u in report.unknowns
        )

    def test_everything_complete_is_fully_decisive(self):
        data = make_graph(
            """
            @prefix ex: <http://example.org/ex#> .
            ex:Alice a ex:Person ; ex:name "Alice" ; ex:email "a@example.org" .
            """
        )
        report = validate_partial(
            data,
            SHAPES_TTL,
            focus_nodes=[ALICE],
            completeness=CompletenessDeclaration.everything(),
        )
        assert report.decisive
        assert report.conforms
        assert report.unknowns == ()
        assert len(report.conformances) == 4


class TestEntitySlice:
    def test_only_requested_slice_is_validated(self):
        data = make_graph(
            """
            @prefix ex: <http://example.org/ex#> .
            ex:Alice a ex:Person ; ex:name "Alice" ; ex:email "a@example.org" .
            ex:Bob a ex:Person .
            """
        )
        completeness = CompletenessDeclaration.everything()
        report = validate_partial(
            data, SHAPES_TTL, focus_nodes=[ALICE], completeness=completeness
        )
        assert report.focus_nodes == (ALICE,)
        assert report.decisive and report.conforms
        # Bob's missing name/email never enter the report.
        assert all(item.focus == ALICE for item in report.violations)
        assert all(item.focus == ALICE for item in report.unknowns)
        # Validating the whole graph instead does surface Bob's violation.
        full_report = validate_partial(data, SHAPES_TTL, completeness=completeness)
        assert any(v.focus == BOB for v in full_report.violations)


class TestRefinement:
    def test_unknowns_resolve_after_supplementing_data(self):
        validator = PartialValidator(SHAPES_TTL)
        validator.add_data(
            make_graph(
                """
                @prefix ex: <http://example.org/ex#> .
                ex:Alice a ex:Person ; ex:name "Alice" .
                """
            ),
            completeness=CompletenessDeclaration().declare(subject=ALICE, predicate=EX.name),
        )
        first = validator.validate(focus_nodes=[ALICE])
        assert first.violations == ()
        assert any(
            u.component == SH.MinCountConstraintComponent and u.path == EX.email
            for u in first.unknowns
        )
        # A second slice arrives: Alice's email, plus its completeness claim.
        validator.add_data(
            make_graph(
                """
                @prefix ex: <http://example.org/ex#> .
                ex:Alice ex:email "a@example.org" .
                """
            ),
            completeness=CompletenessDeclaration().declare(subject=ALICE, predicate=EX.email),
        )
        second = validator.validate(focus_nodes=[ALICE])
        assert second.decisive
        assert second.conforms
        assert second.unknowns == ()
        # Previously proven conformances are carried over, not recomputed.
        assert len(second.conformances) == len(first.conformances) + 1

    def test_unknown_violation_confirmed_by_declaration_only(self):
        validator = PartialValidator(
            SHAPES_TTL,
            data_graph=make_graph(
                """
                @prefix ex: <http://example.org/ex#> .
                ex:Alice a ex:Person ; ex:name "Alice" .
                """
            ),
        )
        first = validator.validate(focus_nodes=[ALICE])
        assert first.violations == ()
        # No new data, only the completeness claim for ex:email arrives:
        # closedness applies to the declared scope from this point on.
        validator.add_completeness(
            CompletenessDeclaration().declare(subject=ALICE, predicate=EX.email)
        )
        second = validator.validate(focus_nodes=[ALICE])
        assert any(
            v.component == SH.MinCountConstraintComponent and v.path == EX.email
            for v in second.violations
        )
        assert not second.conforms


class TestNestedShapes:
    def test_nested_unknown_carries_remote_missing_scope(self):
        data = make_graph(
            """
            @prefix ex: <http://example.org/ex#> .
            ex:Alice a ex:Person ; ex:knows ex:Bob .
            """
        )
        report = validate_partial(data, NESTED_SHAPES_TTL, focus_nodes=[ALICE])
        assert report.violations == ()
        node_unknowns = [
            u for u in report.unknowns if u.component == SH.NodeConstraintComponent
        ]
        assert len(node_unknowns) == 1
        # The missing scope points at Bob's graph slice, not Alice's.
        assert MissingScope(BOB, EX.email) in node_unknowns[0].missing

    def test_nested_violation_proven_once_inner_scope_complete(self):
        data = make_graph(
            """
            @prefix ex: <http://example.org/ex#> .
            ex:Alice a ex:Person ; ex:knows ex:Bob .
            """
        )
        completeness = CompletenessDeclaration().declare(subject=BOB, predicate=EX.email)
        report = validate_partial(
            data, NESTED_SHAPES_TTL, focus_nodes=[ALICE], completeness=completeness
        )
        # Bob's email is declared complete and absent: proven violation both
        # at the inner shape and at Alice's sh:node constraint.
        assert any(
            v.component == SH.NodeConstraintComponent and v.focus == ALICE
            for v in report.violations
        )
        assert not report.conforms


class TestClassScopedDeclarations:
    def test_of_class_declaration_applies_to_instances(self):
        data = make_graph(
            """
            @prefix ex: <http://example.org/ex#> .
            ex:Alice a ex:Person ; ex:name "Alice" ; ex:email "a@example.org" .
            """
        )
        completeness = CompletenessDeclaration().declare(of_class=EX.Person, predicate=EX.name)
        report = validate_partial(
            data, SHAPES_TTL, focus_nodes=[ALICE], completeness=completeness
        )
        name_conformances = [c for c in report.conformances if c.path == EX.name]
        assert {c.component for c in name_conformances} == {
            SH.MinCountConstraintComponent,
            SH.MaxCountConstraintComponent,
            SH.DatatypeConstraintComponent,
        }
        # ex:email was not declared: its satisfaction stays unknown.
        assert any(u.path == EX.email for u in report.unknowns)
