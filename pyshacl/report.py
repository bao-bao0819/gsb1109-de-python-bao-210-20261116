# -*- coding: utf-8 -*-
"""
Result identity and evidence merging for validation reports.

A validation result is identified by the evidence it carries, not by the
text of its message: the focus node, result path, source shape, offending
value and source constraint component together define the reported fact.
When separate constraint evaluations produce the very same result (for
example a property shape that has its own target and is also referenced
through ``sh:property``), the duplicate results are merged into one and the
contributing evaluation sources are retained as evidence.

Results produced by a single constraint evaluation never share an identity,
even when their standard result triples coincide. For example one
``sh:uniqueLang`` evaluation reports each duplicated language separately,
and a ``sh:sparql`` query may emit one result per result row. Keeping the
evaluation origin on every result guarantees those distinct facts survive,
while results that merely repeat the same fact through different
evaluation paths are the ones that get merged.

Results produced by different constraint components never share an
identity either, because the source constraint component is part of the
identity. So ``sh:minLength`` and ``sh:pattern`` reporting the same value
stay as two results even with identical messages: removing one constraint
does not silently remove the result of the other.
"""
from typing import Any, Dict, List, Optional, Tuple

from rdflib import BNode, Literal, URIRef
from rdflib.namespace import Namespace, XSD

from .consts import (
    RDF_type,
    SH_detail,
    SH_focusNode,
    SH_resultPath,
    SH_sourceConstraint,
    SH_sourceConstraintComponent,
    SH_sourceShape,
    SH_value,
)

PYSHACL = Namespace("urn:pyshacl:")
PYSHACL_Evidence = URIRef("urn:pyshacl:Evidence")
PYSHACL_evidence = URIRef("urn:pyshacl:evidence")
PYSHACL_evidenceCount = URIRef("urn:pyshacl:evidenceCount")
PYSHACL_evidenceIndex = URIRef("urn:pyshacl:evidenceIndex")
PYSHACL_resultIdentity = URIRef("urn:pyshacl:resultIdentity")

# A report tuple as produced by ConstraintComponent.make_v_result():
# (human readable description, result blank node, result triples)
ReportTuple = Tuple[str, BNode, List[Tuple[Any, Any, Any]]]

# Predicates that identify the reported fact. sh:resultMessage is deliberately
# excluded: merging must never be driven by message text.
IDENTITY_PREDICATES: Tuple[Any, ...] = (
    SH_sourceConstraintComponent,
    SH_sourceShape,
    SH_focusNode,
    SH_resultPath,
    SH_value,
    SH_sourceConstraint,
)

# Predicates copied onto each stored evidence contribution.
EVIDENCE_PREDICATES: Tuple[Any, ...] = (
    SH_sourceConstraintComponent,
    SH_sourceShape,
    SH_sourceConstraint,
)


def _identity_terms(r_triples: List[Tuple[Any, Any, Any]]) -> Dict[Any, Any]:
    # Cross-graph terms are carried as (source_graph, node) tuples until the
    # report graph is built, so the tuples are retained in the identity.
    terms: Dict[Any, Any] = {p: None for p in IDENTITY_PREDICATES}
    for _s, predicate, obj in r_triples:
        if predicate in terms and terms[predicate] is None:
            terms[predicate] = obj
    return terms


def _identity_key(r_triples: List[Tuple[Any, Any, Any]]) -> Tuple[Any, ...]:
    terms = _identity_terms(r_triples)
    return tuple(terms[p] for p in IDENTITY_PREDICATES)


def _term_text(term: Any) -> str:
    if term is None:
        return ""
    node = term[1] if isinstance(term, tuple) else term
    return node.n3()


def _identity_string(terms: Dict[Any, Any]) -> str:
    return "|".join(
        "{}={}".format(str(predicate), _term_text(terms[predicate])) for predicate in IDENTITY_PREDICATES
    )


def _detail_closure(other_node: BNode, other_triples: List[Tuple[Any, Any, Any]]) -> set:
    # Collect the sh:detail blank-node subgraph hanging off other_node.
    # It is not carried into a merged result because the kept result already
    # carries the equivalent nested results.
    closure = set()
    frontier = {obj for s, p, obj in other_triples if s is other_node and p is SH_detail}
    while frontier:
        current = frontier.pop()
        if current in closure:
            continue
        closure.add(current)
        for s, p, obj in other_triples:
            if s is current and p is SH_detail:
                frontier.add(obj)
    return closure


def _merge_triples(
    kept_node: BNode,
    kept_triples: List[Tuple[Any, Any, Any]],
    other_node: BNode,
    other_triples: List[Tuple[Any, Any, Any]],
) -> None:
    detail_closure = _detail_closure(other_node, other_triples)
    existing = set(kept_triples)
    for subject, predicate, obj in other_triples:
        if subject in detail_closure:
            # Nested detail subgraph of the dropped duplicate; the kept result
            # has its own equivalent details.
            continue
        if subject is other_node:
            if predicate is SH_detail:
                continue
            merged_triple: Tuple[Any, Any, Any] = (kept_node, predicate, obj)
        else:
            # Triples belonging to retained nested results keep their subject.
            merged_triple = (subject, predicate, obj)
        if merged_triple not in existing:
            kept_triples.append(merged_triple)
            existing.add(merged_triple)


def _attach_evidence(
    desc: str, r_node: BNode, r_triples: List[Tuple[Any, Any, Any]], contributor_count: int
) -> str:
    terms = _identity_terms(r_triples)
    r_triples.append((r_node, PYSHACL_resultIdentity, Literal(_identity_string(terms))))
    r_triples.append((r_node, PYSHACL_evidenceCount, Literal(contributor_count, datatype=XSD.integer)))
    for index in range(1, contributor_count + 1):
        evidence_node = BNode()
        r_triples.append((r_node, PYSHACL_evidence, evidence_node))
        r_triples.append((evidence_node, RDF_type, PYSHACL_Evidence))
        r_triples.append((evidence_node, PYSHACL_evidenceIndex, Literal(index, datatype=XSD.integer)))
        for predicate in EVIDENCE_PREDICATES:
            term = terms[predicate]
            if term is not None:
                r_triples.append((evidence_node, predicate, term))
    return desc + "\tEvidence: {} validation result source(s) matched the same result identity.\n".format(
        contributor_count
    )


def merge_validation_results(
    results: List[ReportTuple], result_origins: Optional[Dict[int, int]] = None
) -> List[ReportTuple]:
    """
    Merge results sharing the same evidence-based identity.

    The order of first appearance is preserved. The identity is built from the
    source constraint component, source shape, focus node, result path,
    offending value and source constraint; message text never participates.

    Two results with the same identity are only merged when they originate
    from different constraint evaluations. Same-evaluation results (one per
    query row, duplicated language tag and so on) are always kept apart.
    Merged results gain a result identity literal, an evidence count and one
    evidence node per contributing evaluation source.
    """
    result_origins = result_origins or {}
    # identity key -> list of groups, each group tracking its contributing
    # evaluation origins and its merged report content.
    groups_by_key: Dict[Tuple[Any, ...], List[List[Any]]] = {}
    ordered_entries: List[List[Any]] = []
    for desc, r_node, r_triples in results:
        key = _identity_key(r_triples)
        origin = result_origins.get(id(r_node))
        groups = groups_by_key.get(key)
        target_group = None
        if groups is not None and origin is not None:
            for group in groups:
                if origin not in group[3]:
                    target_group = group
                    break
        if target_group is None:
            group = [desc, r_node, list(r_triples), set() if origin is None else {origin}]
            groups_by_key.setdefault(key, []).append(group)
            ordered_entries.append(group)
            continue
        _merge_triples(target_group[1], target_group[2], r_node, r_triples)
        target_group[3].add(origin)

    merged_results: List[ReportTuple] = []
    for desc, r_node, r_triples, origins in ordered_entries:
        contributor_count = len(origins)
        if contributor_count > 1:
            desc = _attach_evidence(desc, r_node, r_triples, contributor_count)
        merged_results.append((desc, r_node, r_triples))
    return merged_results
