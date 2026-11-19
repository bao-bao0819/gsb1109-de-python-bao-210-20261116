# -*- coding: utf-8 -*-
"""
Provenance tracking for SHACL property path evaluation over multigraph datasets.

When the data graph is an RDF dataset (a default graph plus named graphs), every
edge used while evaluating a SHACL path is recorded together with the graph that
contains it. This lets path evaluation keep each complete path inside a single
graph scope, and lets validation results name the graph(s) that support them.

A path result that does not consume any edge (the zero-length step of
``sh:zeroOrMorePath`` and ``sh:zeroOrOnePath``) is marked as
:data:`UNSCOPED_GRAPH`; such a step is compatible with every graph scope.
"""
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

from rdflib import URIRef
from rdflib.graph import DATASET_DEFAULT_GRAPH_ID

from pyshacl.consts import PYSHACL_PFX, SH_focusNode, SH_resultPath, SH_value, PYSHACL_sourceGraph
from pyshacl.pytypes import RDFNode

# Sentinel for path results reached without traversing an edge (the zero-length
# path step). A result carrying this scope does not pin the path to a graph.
UNSCOPED_GRAPH = None

GraphId = Optional[Any]


def intersect_scopes(left: Iterable[GraphId], right: Iterable[GraphId]) -> Set[GraphId]:
    """Combine the graph scopes of two consecutive path steps.

    A sequence path can only be completed when every step is supported by the
    same graph. An unscoped step (a zero-length step) does not restrict the
    scope of the other step.
    """
    left_set = set(left)
    right_set = set(right)
    if UNSCOPED_GRAPH in left_set:
        return set(right_set)
    if UNSCOPED_GRAPH in right_set:
        return left_set
    return left_set & right_set


def graph_id_for_report(graph_id: GraphId):
    """Return the term used to identify a supporting graph in a report."""
    if graph_id == UNSCOPED_GRAPH:
        return None
    if graph_id == DATASET_DEFAULT_GRAPH_ID:
        return DATASET_DEFAULT_GRAPH_ID
    if isinstance(graph_id, URIRef):
        return graph_id
    return URIRef(str(graph_id))


def data_graph_has_named_graphs(data_graph) -> bool:
    """True when the data graph is a dataset containing at least one named graph.

    Path provenance is only relevant when distinct graph boundaries exist; a
    dataset holding only a default graph is treated like a plain graph. Graphs
    created by PySHACL itself (inference, inoculation) are internal machinery
    and do not count as user-supplied named graphs.
    """
    if not (hasattr(data_graph, "is_multigraph") and data_graph.is_multigraph()):
        return False
    # Prefer the wrapper's graphs() (both the RDFLib and Oxigraph DataGraph
    # backends expose it) and fall back to the wrapped implementation.
    graphs = getattr(data_graph, "graphs", None)
    if not callable(graphs):
        impl = getattr(data_graph, "impl", data_graph)
        graphs = getattr(impl, "graphs", None)
    if not callable(graphs):
        # Fall back to quad inspection when the backend does not expose graphs().
        return hasattr(data_graph, "quads")
    for graph in graphs():
        identifier = getattr(graph, "identifier", None)
        if identifier == DATASET_DEFAULT_GRAPH_ID:
            continue
        if isinstance(identifier, str) and identifier.startswith(PYSHACL_PFX):
            continue
        return True
    return False


def _unwrap_report_object(value):
    # Report triples encode data/shapes graph nodes as (source_graph, node) pairs.
    if isinstance(value, tuple) and len(value) == 2:
        return value[1]
    return value


class PathScopeTracker:
    """Collects the supporting graph(s) of every value node produced by a path.

    The tracker is shared across all shape evaluations of one validation run,
    so nested shapes (``sh:node`` / ``sh:property``) contribute to the same
    provenance map. Keys are ``(focus_node, path_node)`` pairs.
    """

    def __init__(self) -> None:
        # (focus, path) -> value node -> set of graph ids supporting the value
        self._provenance: Dict[Tuple[RDFNode, Any], Dict[RDFNode, Set[GraphId]]] = {}
        # (focus, path) -> graph ids touched while attempting the path, even
        # when no complete path could be reproduced inside one graph.
        self._attempted_scopes: Dict[Tuple[RDFNode, Any], Set[GraphId]] = {}

    def attach(
        self,
        focus: RDFNode,
        path_node: Any,
        provenance: Dict[RDFNode, Set[GraphId]],
    ) -> None:
        if not provenance:
            return
        key = (focus, path_node)
        existing = self._provenance.setdefault(key, {})
        for value_node, graph_ids in provenance.items():
            bucket = existing.setdefault(value_node, set())
            bucket.update(graph_ids)

    def attach_attempts(self, focus: RDFNode, path_node: Any, graph_ids: Set[GraphId]) -> None:
        scoped = {g for g in graph_ids if g != UNSCOPED_GRAPH}
        if not scoped:
            return
        bucket = self._attempted_scopes.setdefault((focus, path_node), set())
        bucket.update(scoped)

    def sources_for(self, focus: RDFNode, path_node: Any) -> Dict[RDFNode, FrozenSet[GraphId]]:
        prov = self._provenance.get((focus, path_node))
        if not prov:
            return {}
        return {value_node: frozenset(graph_ids) for value_node, graph_ids in prov.items()}

    def graph_ids_for_focus_path(self, focus: RDFNode, path_node: Any) -> FrozenSet[GraphId]:
        ids: Set[GraphId] = set()
        prov = self._provenance.get((focus, path_node))
        if prov:
            for graph_ids in prov.values():
                ids.update(graph_ids)
        # Attempted scopes include every graph touched while evaluating the
        # path, even edges that could not be joined into a complete path, so
        # they name the boundary of un-reproducible results.
        ids.update(self._attempted_scopes.get((focus, path_node), set()))
        ids.discard(UNSCOPED_GRAPH)
        return frozenset(ids)

    def sources_for_result(
        self,
        focus_node: RDFNode,
        result_path: Any,
        value_node: Optional[RDFNode] = None,
    ) -> FrozenSet[GraphId]:
        """Graph ids supporting a validation result's focus/path combination.

        With a value node, only the graphs supporting that exact path result
        are returned. A zero-length path step (which carries no graph) falls
        back to every graph touched while evaluating the path. Without a value
        node (e.g. cardinality results), the same fallback applies.
        """
        if value_node is not None:
            exact = self.sources_for(focus_node, result_path).get(value_node, frozenset())
            scoped = {g for g in exact if g != UNSCOPED_GRAPH}
            if scoped:
                return frozenset(scoped)
        return self.graph_ids_for_focus_path(focus_node, result_path)

    def annotate_reports(self, reports: List[Tuple]) -> List[Tuple]:
        """Add source-graph provenance to each validation result in ``reports``.

        Every result node found in the report triples (including nodes linked via
        ``sh:detail``) is annotated independently. The human-readable report text
        gains a ``Source Graph:`` line per supporting graph, so a result that
        cannot be reproduced on a single named graph still names the graph
        boundary involved.
        """
        annotated_reports = []
        for description, report_node, report_triples in reports:
            parts_by_subject: Dict[Any, List[Tuple]] = {}
            for subject, predicate, obj in report_triples:
                parts_by_subject.setdefault(subject, []).append((predicate, obj))
            description_suffix = ""
            for subject, parts in parts_by_subject.items():
                predicates = {predicate for predicate, _obj in parts}
                # Nested sh:detail results are annotated by their own shape run.
                if PYSHACL_sourceGraph in predicates:
                    continue
                focus_node = None
                result_path = None
                value_node = None
                for predicate, obj in parts:
                    if predicate == SH_focusNode:
                        focus_node = _unwrap_report_object(obj)
                    elif predicate == SH_resultPath:
                        result_path = _unwrap_report_object(obj)
                    elif predicate == SH_value:
                        value_node = _unwrap_report_object(obj)
                if focus_node is None or result_path is None:
                    continue
                graph_ids = self.sources_for_result(focus_node, result_path, value_node)
                indentation = "\t" if subject == report_node else "\t\t"
                for graph_id in sorted(graph_ids, key=lambda g: str(g)):
                    report_term = graph_id_for_report(graph_id)
                    if report_term is None:
                        continue
                    report_triples.append((subject, PYSHACL_sourceGraph, report_term))
                    description_suffix += f"{indentation}Source Graph: {report_term}\n"
            if description_suffix:
                description = description + description_suffix
            annotated_reports.append((description, report_node, report_triples))
        return annotated_reports
