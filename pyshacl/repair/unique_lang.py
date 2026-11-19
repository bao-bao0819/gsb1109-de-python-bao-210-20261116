# -*- coding: utf-8 -*-
#
"""
Provenance-aware, minimum-cost repair of ``sh:uniqueLang`` violations.

The SHACL ``sh:uniqueLang`` constraint requires that no two value nodes of a
property share the same language tag.  This module extends that idea to two
real-world complications:

* **Aliases** -- different language tags may be declared equivalent
  (for example ``en`` and ``en-US``), so conflict groups are built from
  language *equivalence classes* instead of raw tags.
* **Multiple sources** -- the same RDF triple can be supported by several
  input sources.  Retracting one source only deletes a fact when it was the
  fact's last remaining support.

Given per-source retraction costs, :func:`repair_unique_lang` computes a
minimum-cost set of sources to retract so that every conflict group keeps at
most one distinct literal.  When a conflict cannot be resolved (two or more
conflicting values are locked by non-retractable evidence), the result reports
the locking evidence and the original graph is left unchanged.
"""
import logging
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Tuple, Union

import rdflib
from rdflib import BNode, Literal, URIRef
from rdflib.term import Node

from pyshacl.consts import SH
from pyshacl.shapes_graph import ShapesGraph


logger = logging.getLogger(__name__)

SH_uniqueLang = SH.uniqueLang

#: Reserved name of the implicit source that supports every triple without
#: explicit provenance.  It is never retractable, so unprovenanced facts are
#: locked in place.
BASE_SOURCE = "pyshacl:base"

_BASE_ONLY = frozenset((BASE_SOURCE,))

Triple = Tuple[Node, Node, Node]


class LanguageEquivalence:
    """
    Equivalence classes over language tags.

    Tags are always compared case-insensitively (as SHACL does for
    ``sh:uniqueLang``).  ``aliases`` may additionally declare tags equivalent,
    either as a mapping of ``alias -> canonical tag`` or as an iterable of
    groups of mutually equivalent tags.  Equivalence is transitive.
    """

    __slots__ = ('_parent',)

    def __init__(self, aliases: Optional[Union[Mapping[str, str], Iterable[Iterable[str]]]] = None):
        self._parent: Dict[str, str] = {}
        if aliases is None:
            return
        if isinstance(aliases, Mapping):
            pairs = [(alias, canonical) for alias, canonical in aliases.items()]
        else:
            pairs = []
            for group in aliases:
                members = [str(member) for member in group]
                pairs.extend((members[0], member) for member in members[1:])
        for tag_a, tag_b in pairs:
            self._union(str(tag_a).lower(), str(tag_b).lower())

    def _find(self, tag: str) -> str:
        parent = self._parent
        root = tag
        while parent.get(root, root) != root:
            root = parent[root]
        while parent.get(tag, tag) != tag:
            parent[tag], tag = root, parent[tag]
        return root

    def _union(self, tag_a: str, tag_b: str) -> None:
        root_a, root_b = self._find(tag_a), self._find(tag_b)
        if root_a == root_b:
            return
        # Deterministic canonical representative: the lexicographically
        # smallest tag in the class.
        if root_b < root_a:
            root_a, root_b = root_b, root_a
        self._parent[root_b] = root_a

    def canonical(self, tag: str) -> str:
        """Return the canonical representative of ``tag``'s equivalence class."""
        return self._find(str(tag).lower())

    def equivalent(self, tag_a: str, tag_b: str) -> bool:
        """Return True when ``tag_a`` and ``tag_b`` belong to one class."""
        return self.canonical(tag_a) == self.canonical(tag_b)


@dataclass(frozen=True)
class ConflictGroup:
    """
    A set of literals that violate a (possibly alias-extended) unique-language
    rule: they are values of the same property on the same focus node and
    their language tags are equivalent.
    """

    shape: Union[URIRef, BNode]
    focus_node: Node
    predicate: URIRef
    language: str
    values: Tuple[Literal, ...]

    @property
    def triples(self) -> Tuple[Triple, ...]:
        return tuple((self.focus_node, self.predicate, value) for value in self.values)


@dataclass(frozen=True)
class LockedEvidence:
    """
    Proof that a conflict group cannot be repaired: these conflicting values
    are each backed by at least one non-retractable source, so no retraction
    can eliminate them.
    """

    group: ConflictGroup
    locked_values: Tuple[Literal, ...]
    locking_sources: FrozenSet[str]


@dataclass
class UniqueLangRepairResult:
    """Outcome of :func:`repair_unique_lang`."""

    feasible: bool
    conflict_groups: Tuple[ConflictGroup, ...] = ()
    retracted_sources: FrozenSet[str] = frozenset()
    total_cost: float = 0.0
    removed_triples: FrozenSet[Triple] = frozenset()
    repaired_graph: Optional[rdflib.Graph] = None
    locked_evidence: Tuple[LockedEvidence, ...] = ()


def _shape_requires_unique_lang(shape) -> bool:
    for val in shape.objects(SH_uniqueLang):
        if isinstance(val, Literal) and val.value is True:
            return True
    return False


def _collect_values(collected, data_graph, focus, predicate, shape_node, equiv) -> None:
    for value in data_graph.objects(focus, predicate):
        if not isinstance(value, Literal) or not value.language:
            continue
        key = (focus, predicate, equiv.canonical(str(value.language)))
        entry = collected.setdefault(key, (shape_node, set()))
        entry[1].add(value)


def find_unique_lang_conflicts(
    data_graph: rdflib.Graph,
    shacl_graph,
    *,
    lang_aliases: Optional[Union[Mapping[str, str], Iterable[Iterable[str]]]] = None,
) -> List[ConflictGroup]:
    """
    Find ``sh:uniqueLang`` conflict groups in ``data_graph``.

    Conflict groups are generated per property shape (``sh:uniqueLang true``)
    and per language equivalence class: literals whose tags are equivalent
    under ``lang_aliases`` land in the same group.  A group becomes a conflict
    when it holds two or more distinct literals.

    Property shapes are honoured both when they carry their own targets and
    when they are referenced from a node shape via ``sh:property``.  Only
    simple predicate paths (a single IRI) are supported, because repair needs
    a concrete predicate to retract; complex paths are skipped with a warning.
    """
    equiv = LanguageEquivalence(lang_aliases)
    sg = shacl_graph if isinstance(shacl_graph, ShapesGraph) else ShapesGraph(shacl_graph)
    collected: Dict[Tuple[Node, URIRef, str], Tuple[Union[URIRef, BNode], set]] = {}
    for shape in sg.shapes:
        if shape.deactivated:
            continue
        if shape.is_property_shape:
            candidate_shapes = [(shape, shape)]
        else:
            candidate_shapes = []
            for p_node in shape.property_shapes():
                try:
                    p_shape = sg.lookup_shape_from_node(p_node)
                except (KeyError, AttributeError):
                    continue
                if p_shape is not None:
                    candidate_shapes.append((shape, p_shape))
        for focus_shape, prop_shape in candidate_shapes:
            if not prop_shape.is_property_shape or prop_shape.deactivated:
                continue
            if not _shape_requires_unique_lang(prop_shape):
                continue
            predicate = prop_shape.path()
            if not isinstance(predicate, URIRef):
                logger.warning(
                    "Skipping sh:uniqueLang repair on shape %s: only simple predicate paths are supported.",
                    prop_shape.node,
                )
                continue
            for focus in focus_shape.focus_nodes(data_graph):
                _collect_values(collected, data_graph, focus, predicate, prop_shape.node, equiv)
    groups = []
    for (focus, predicate, language) in sorted(collected, key=lambda k: tuple(str(part) for part in k)):
        shape_node, literals = collected[(focus, predicate, language)]
        if len(literals) < 2:
            continue
        groups.append(
            ConflictGroup(
                shape=shape_node,
                focus_node=focus,
                predicate=predicate,
                language=language,
                values=tuple(sorted(literals, key=str)),
            )
        )
    return groups


def _normalize_supports(supports) -> Dict[Triple, FrozenSet[str]]:
    table: Dict[Triple, FrozenSet[str]] = {}
    if not supports:
        return table
    for triple, sources in supports.items():
        subject, predicate, obj = triple
        if isinstance(sources, str):
            source_set = frozenset((sources,))
        else:
            source_set = frozenset(str(source) for source in sources)
        if not source_set:
            source_set = _BASE_ONLY
        table[(subject, predicate, obj)] = source_set
    return table


def _solve_component(options_per_group: List[List[FrozenSet[str]]], cost_of) -> Tuple[FrozenSet[str], float]:
    """Exact branch-and-bound over the survivor choices of one component."""

    def sort_key(source_set):
        return (cost_of(source_set), len(source_set), sorted(source_set))

    # Greedy incumbent for a good starting bound.
    current = frozenset()  # type: FrozenSet[str]
    for options in options_per_group:
        current = min((current | option for option in options), key=sort_key)
    best_set, best_cost = current, cost_of(current)

    def dfs(index, current, current_cost):
        nonlocal best_set, best_cost
        if current_cost >= best_cost:
            return
        if index == len(options_per_group):
            best_set, best_cost = current, current_cost
            return
        ordered = sorted(options_per_group[index], key=lambda option: sort_key(current | option))
        for option in ordered:
            new_set = current | option
            dfs(index + 1, new_set, cost_of(new_set))

    dfs(0, frozenset(), 0.0)
    return best_set, best_cost


def _min_cost_retraction(
    options_per_group: List[List[FrozenSet[str]]], cost_of
) -> Tuple[FrozenSet[str], float]:
    """
    Choose one option (a set of sources to retract) per conflict group such
    that the total cost of the union of chosen sources is minimal.

    Groups that share no sources are independent, so the problem is decomposed
    into connected components that are solved exactly and combined.
    """
    count = len(options_per_group)
    parent = list(range(count))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    owner: Dict[str, int] = {}
    for group_index, options in enumerate(options_per_group):
        for option in options:
            for source in option:
                if source in owner:
                    root_a, root_b = find(group_index), find(owner[source])
                    if root_a != root_b:
                        parent[root_b] = root_a
                else:
                    owner[source] = group_index
    components: Dict[int, List[int]] = {}
    for group_index in range(count):
        components.setdefault(find(group_index), []).append(group_index)
    best_sources = set()  # type: set
    best_cost = 0.0
    for component in components.values():
        component_options = [options_per_group[group_index] for group_index in component]
        sources, cost = _solve_component(component_options, cost_of)
        best_sources |= sources
        best_cost += cost
    return frozenset(best_sources), best_cost


def repair_unique_lang(
    data_graph: rdflib.Graph,
    shacl_graph,
    *,
    supports: Optional[Mapping[Triple, Union[str, Iterable[str]]]] = None,
    costs: Optional[Mapping[str, float]] = None,
    locked_sources: Iterable[str] = (),
    lang_aliases: Optional[Union[Mapping[str, str], Iterable[Iterable[str]]]] = None,
    plan_only: bool = False,
) -> UniqueLangRepairResult:
    """
    Compute a minimum-cost retraction plan that repairs all ``sh:uniqueLang``
    conflicts in ``data_graph`` and (unless ``plan_only``) apply it to a copy
    of the graph.

    :param data_graph: The RDF data graph.  It is never modified.
    :param shacl_graph: Shapes graph (or :class:`ShapesGraph`) holding the
        ``sh:uniqueLang`` property shapes.
    :param supports: Mapping of ``(s, p, o)`` triple to the names of the
        sources that support it.  Triples without an entry are backed only by
        the implicit, non-retractable base source.
    :param costs: Retraction cost per source name; defaults to ``1.0``.
    :param locked_sources: Source names that must never be retracted.
    :param lang_aliases: Language tag equivalences, either a mapping of
        ``alias -> canonical tag`` or an iterable of equivalent-tag groups.
    :param plan_only: When True, do not materialise the repaired graph.

    A fact counts as deleted only when *all* of its supporting sources are
    retracted; retracting one of several supporters leaves the fact in place.
    When a conflict group contains two or more values backed by
    non-retractable sources, no feasible retraction exists: the result has
    ``feasible == False``, :attr:`UniqueLangRepairResult.locked_evidence`
    points at the locking values and sources, and no repaired graph is
    produced (the original graph stays unchanged).
    """
    costs = dict(costs or {})
    for name, cost in costs.items():
        if cost < 0:
            raise ValueError("Retraction cost of source {!r} must not be negative.".format(name))
    locked = {str(source) for source in locked_sources}
    locked.add(BASE_SOURCE)
    support_table = _normalize_supports(supports)

    def supporters_of(triple: Triple) -> FrozenSet[str]:
        return support_table.get(triple, _BASE_ONLY)

    def cost_of(source_set: FrozenSet[str]) -> float:
        return float(sum(costs.get(source, 1.0) for source in source_set))

    groups = find_unique_lang_conflicts(data_graph, shacl_graph, lang_aliases=lang_aliases)
    if not groups:
        repaired = None if plan_only else _rebuilt_graph(data_graph, frozenset())
        return UniqueLangRepairResult(feasible=True, repaired_graph=repaired)

    options_per_group: List[List[FrozenSet[str]]] = []
    evidence: List[LockedEvidence] = []
    for group in groups:
        supporters_per_value = [supporters_of(triple) for triple in group.triples]
        locked_values = []
        locking_sources = set()  # type: set
        for value, supporters in zip(group.values, supporters_per_value):
            locking = supporters & locked
            if locking:
                locked_values.append(value)
                locking_sources |= locking
        if len(locked_values) >= 2:
            evidence.append(
                LockedEvidence(
                    group=group,
                    locked_values=tuple(locked_values),
                    locking_sources=frozenset(locking_sources),
                )
            )
            continue
        options = set()  # type: set
        for index, value in enumerate(group.values):
            if locked_values and value not in locked_values:
                # A locked value can never be eliminated, so it must survive.
                continue
            retract = frozenset().union(
                *(supporters_per_value[other] for other in range(len(group.values)) if other != index)
            )
            options.add(retract)
        options_per_group.append(sorted(options, key=lambda option: (len(option), sorted(option))))

    if evidence:
        return UniqueLangRepairResult(
            feasible=False,
            conflict_groups=tuple(groups),
            locked_evidence=tuple(evidence),
        )

    retracted, total_cost = _min_cost_retraction(options_per_group, cost_of)
    removed = frozenset(
        triple
        for triple in data_graph.triples((None, None, None))
        if supporters_of(triple) and supporters_of(triple) <= retracted
    )
    repaired = None if plan_only else _rebuilt_graph(data_graph, removed)
    return UniqueLangRepairResult(
        feasible=True,
        conflict_groups=tuple(groups),
        retracted_sources=retracted,
        total_cost=total_cost,
        removed_triples=removed,
        repaired_graph=repaired,
    )


def _rebuilt_graph(data_graph: rdflib.Graph, removed: FrozenSet[Triple]) -> rdflib.Graph:
    repaired = rdflib.Graph()
    for prefix, namespace in data_graph.namespace_manager.namespaces():
        repaired.namespace_manager.bind(prefix, namespace)
    for triple in data_graph.triples((None, None, None)):
        if triple not in removed:
            repaired.add(triple)
    return repaired
