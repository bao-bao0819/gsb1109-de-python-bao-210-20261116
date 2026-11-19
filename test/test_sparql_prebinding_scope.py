# -*- coding: utf-8 -*-

"""
Scope rules for the externally pre-bound variables of a SPARQL constraint.

The pre-binding of ``$this``/``$value``/``$currentShape``/``$shapesGraph`` must be
handled as SPARQL ``initBindings`` (an outer-scope solution join), never as a
global textual replacement of the query. This guards against:

* the same token appearing inside a string literal or comment being rewritten or
  mistaken for a pre-bound variable / forbidden clause,
* a sub-query projecting a variable with the same name as the pre-bound focus
  shadowing the outer binding,
* interleaved validation runs leaking one focus node's binding into another.
"""

from rdflib import RDF, Graph, Namespace

from pyshacl import validate
from pyshacl.errors import ValidationFailure
from pyshacl.helper.sparql_query_helper import SPARQLQueryHelper

EX = Namespace("http://example.org/")
SH = Namespace("http://www.w3.org/ns/shacl#")

PREFIX = """\
@prefix ex: <http://example.org/> .
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
"""


class _FakeShapesGraph:
    graph = Graph()


class _FakeShape:
    sg = _FakeShapesGraph()
    node = EX.FakeShape

    def __init__(self, path=None):
        self._path = path

    def path(self):
        return self._path


def _helper(select_text, path=None):
    return SPARQLQueryHelper(_FakeShape(path=path), EX.Constraint, select_text)


def _focus_nodes(report_graph):
    return {
        str(fn).replace(str(EX), "ex:")
        for result in report_graph.subjects(RDF.type, SH.ValidationResult)
        for fn in report_graph.objects(result, SH.focusNode)
    }


def test_path_token_inside_string_constant_is_not_rewritten():
    # The filter compares the value against a constant whose text contains " $PATH"
    # (a space before the token, exactly like real query code). A textual
    # replacement would corrupt that constant into the path IRI, silently flipping
    # the verdict (row filtered out -> false conforms).
    data = Graph().parse(data=PREFIX + 'ex:A ex:knows "x $PATH y" .\n', format="turtle")
    shapes = Graph().parse(
        data=PREFIX
        + """\
ex:Shape a sh:PropertyShape ;
  sh:targetNode ex:A ;
  sh:path ex:knows ;
  sh:sparql [ sh:select \"\"\"
    SELECT $this ?value WHERE {
      $this $PATH ?value .
      FILTER(CONTAINS(STR(?value), "x $PATH y"))
    }\"\"\" ] .
""",
        format="turtle",
    )
    conforms, report_graph, _ = validate(data, shacl_graph=shapes, inference="none", meta_shacl=False)
    assert conforms is False
    assert _focus_nodes(report_graph) == {"ex:A"}

    helper = _helper(
        'SELECT $this ?v WHERE { $this $PATH ?v . FILTER(?x = "x $PATH y") }', path=EX.knows
    )
    _, query_text = helper.pre_bind_variables(EX.A)
    assert "$this <http://example.org/knows> ?v" in query_text
    assert '"x $PATH y"' in query_text


def test_focus_token_inside_string_constant_neither_binds_nor_rewrites():
    # Only the real variable may source a binding; the identical text in a
    # constant is inert and must be preserved verbatim.
    helper = _helper(
        'SELECT $this WHERE { $this ex:n ?v . FILTER(?v = "see $this here") }'
    )
    bindings, query_text = helper.pre_bind_variables(EX.A)
    assert bindings == {"this": EX.A}
    assert '"see $this here"' in query_text

    string_only = _helper('SELECT ?v WHERE { ?x ex:n ?v . FILTER(?v = "see $this here") }')
    bindings, query_text = string_only.pre_bind_variables(EX.A)
    assert bindings == {}
    assert '"see $this here"' in query_text


def test_forbidden_clause_guards_ignore_strings_and_comments():
    for keyword in ("VALUES", "MINUS", "SERVICE"):
        in_string = _helper(
            'SELECT $this WHERE { $this ex:n ?v . '
            f'FILTER(CONTAINS("the {keyword} word", "{keyword}")) '
            f"# a {keyword} in a comment\n"
            "}"
        )
        in_string.pre_bind_variables(EX.A)  # must not raise

    real_clauses = {
        "VALUES": "SELECT $this WHERE { VALUES ?x { 1 } $this ex:n ?v . }",
        "MINUS": "SELECT $this WHERE { $this ex:n ?v . MINUS { ?x ?y ?z } }",
        "SERVICE": "SELECT $this WHERE { $this ex:n ?v . SERVICE <http://x/> { ?a ?b ?c } }",
    }
    for keyword, query in real_clauses.items():
        helper = _helper(query)
        try:
            helper.pre_bind_variables(EX.A)
        except ValidationFailure:
            continue
        raise AssertionError(f"real {keyword} clause was not rejected")


def test_real_clause_after_inline_string_is_still_rejected():
    # The active span begins mid-line (right after the string literal), so a clause
    # anchored at the start of a line must still be detected within that span.
    inline_clauses = [
        'SELECT $this WHERE { FILTER(?x = "a") VALUES ?y { 1 } }',
        'SELECT $this WHERE { BIND("s" AS ?z) MINUS { ?a ?b ?c } }',
        'SELECT $this WHERE { ?x ?p "VALUES" . VALUES ?y { 1 } }',
    ]
    for query in inline_clauses:
        helper = _helper(query)
        try:
            helper.pre_bind_variables(EX.A)
        except ValidationFailure:
            continue
        raise AssertionError(f"inline clause was not rejected: {query}")


def test_as_rebinding_guard_ignores_string_constants():
    in_string = _helper(
        'SELECT $this ?z WHERE { $this ex:n ?v . BIND("AS ?this note" AS ?z) }'
    )
    in_string.pre_bind_variables(EX.A)  # must not raise

    real_rebind = _helper("SELECT $this WHERE { BIND(true AS $this) }")
    try:
        real_rebind.pre_bind_variables(EX.A)
    except ValidationFailure:
        pass
    else:
        raise AssertionError("real AS $this rebinding was not rejected")


def test_long_string_literal_and_comment_are_hidden_regions():
    select = '''SELECT $this WHERE {
      # $PATH VALUES MINUS SERVICE in a comment
      $this ex:note ?n .
      FILTER(?n = """long $PATH and $this constant""")
    }'''
    helper = _helper(select, path=EX.knows)
    bindings, query_text = helper.pre_bind_variables(EX.A)
    assert bindings == {"this": EX.A}
    assert '"""long $PATH and $this constant"""' in query_text
    assert "$this ex:note ?n" in query_text


def test_iri_fragment_and_comparison_operators_are_not_confused_with_comments():
    # The '#' inside an IRI is a fragment marker, not a comment; '<=' is an operator.
    select = (
        "SELECT $this WHERE { "
        "FILTER(<http://example.org/thing#frag> <= 1) "
        "$this ex:n ?v . FILTER(\"# not a comment\" = ?x) }"
    )
    helper = _helper(select)
    bindings, query_text = helper.pre_bind_variables(EX.A)
    assert bindings == {"this": EX.A}
    assert "http://example.org/thing#frag" in query_text
    assert '"# not a comment"' in query_text


def test_subquery_local_variable_shadows_pre_bound_focus():
    # The inner SELECT projects its own $this; its solution is joined against the
    # outer pre-binding rather than being overwritten by it.
    data = Graph().parse(
        data=PREFIX + "ex:A ex:age 1 . ex:B ex:age 2 .\n", format="turtle"
    )
    shapes = Graph().parse(
        data=PREFIX
        + """\
ex:Shape a sh:NodeShape ;
  sh:targetNode ex:A, ex:B ;
  sh:sparql [ sh:select \"\"\"
    SELECT $this WHERE {
      { SELECT $this WHERE {
          $this ex:age ?age .
          FILTER($this = ex:B)
      } }
    }\"\"\" ] .
""",
        format="turtle",
    )
    conforms, report_graph, _ = validate(data, shacl_graph=shapes, inference="none", meta_shacl=False)
    assert conforms is False
    assert _focus_nodes(report_graph) == {"ex:B"}


def test_interleaved_validations_keep_each_focus_binding_independent():
    data = Graph().parse(data=PREFIX + "ex:A ex:n 1 . ex:B ex:n 2 .\n", format="turtle")
    shapes = Graph().parse(
        data=PREFIX
        + """\
ex:Shape a sh:NodeShape ;
  sh:targetNode ex:A, ex:B ;
  sh:sparql [ sh:select \"\"\"
    SELECT $this WHERE { $this ex:n ?v . FILTER($this = ex:B) }\"\"\" ] .
""",
        format="turtle",
    )
    first = validate(
        data, shacl_graph=shapes, inference="none", meta_shacl=False, focus_nodes=[EX.A]
    )
    second = validate(
        data, shacl_graph=shapes, inference="none", meta_shacl=False, focus_nodes=[EX.B]
    )
    third = validate(
        data, shacl_graph=shapes, inference="none", meta_shacl=False, focus_nodes=[EX.A]
    )
    assert first[0] is True and third[0] is True
    assert second[0] is False
    assert _focus_nodes(second[1]) == {"ex:B"}


def test_identical_token_in_constant_and_variable_reports_every_focus():
    data = Graph().parse(
        data=PREFIX
        + 'ex:A rdfs:comment "token $this inside" . ex:B rdfs:comment "token $this inside" .\n',
        format="turtle",
    )
    shapes = Graph().parse(
        data=PREFIX
        + """\
ex:Shape a sh:NodeShape ;
  sh:targetNode ex:A, ex:B ;
  sh:sparql [ sh:select \"\"\"
    SELECT $this WHERE {
      $this rdfs:comment ?c .
      FILTER(STR(?c) = "token $this inside")
    }\"\"\" ] .
""",
        format="turtle",
    )
    conforms, report_graph, _ = validate(data, shacl_graph=shapes, inference="none", meta_shacl=False)
    assert conforms is False
    assert _focus_nodes(report_graph) == {"ex:A", "ex:B"}
