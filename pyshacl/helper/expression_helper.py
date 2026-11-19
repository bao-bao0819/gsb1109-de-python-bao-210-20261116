import itertools
import operator
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union
from warnings import warn

import rdflib
from rdflib import Literal, URIRef
from rdflib.graph import DATASET_DEFAULT_GRAPH_ID

from pyshacl.consts import (
    RDF,
    SH_alternativePath,
    SH_filterShape,
    SH_intersection,
    SH_inversePath,
    SH_message,
    SH_nodes,
    SH_oneOrMorePath,
    SH_path,
    SH_this,
    SH_union,
    SH_zeroOrMorePath,
    SH_zeroOrOnePath,
)
from pyshacl.errors import ReportableRuntimeError, ShapeLoadError
from pyshacl.helper.path_scope import UNSCOPED_GRAPH, GraphId, intersect_scopes

if TYPE_CHECKING:
    from pyshacl.helper.path_scope import PathScopeTracker
    from pyshacl.pytypes import GraphLike, RDFNode
    from pyshacl.shapes_graph import ShapesGraph

# A value node maps to the set of graph ids supporting a complete path to it.
# A result reached without consuming an edge carries UNSCOPED_GRAPH.
ProvenanceMap = Dict[Any, Set[GraphId]]


def _is_multigraph(target_graph) -> bool:
    is_multi = getattr(target_graph, "is_multigraph", None)
    if callable(is_multi):
        return bool(is_multi())
    # RDFLib Datasets expose quads() and are multigraph even without the wrapper.
    return hasattr(target_graph, "quads") and hasattr(target_graph, "default_graph")


def _edge_objects(target_graph, focus, predicate, inverse: bool) -> List[Tuple[Any, GraphId]]:
    """Return (node, graph_id) pairs for one direct path step.

    On a multigraph dataset the graph id is the named graph (or the default
    graph) containing the edge. On a plain graph every edge is unscoped because
    there is no graph boundary to enforce.
    """
    if not _is_multigraph(target_graph):
        if inverse:
            return [(node, UNSCOPED_GRAPH) for node in target_graph.subjects(predicate, focus)]
        return [(node, UNSCOPED_GRAPH) for node in target_graph.objects(focus, predicate)]
    if inverse:
        quads = target_graph.quads((None, predicate, focus, None))
        results = [(s, _context_id(context)) for s, _p, _o, context in quads]
    else:
        quads = target_graph.quads((focus, predicate, None, None))
        results = [(o, _context_id(context)) for _s, _p, o, context in quads]
    return results


def _context_id(context) -> GraphId:
    # RDFLib yields None for the default graph; the DataGraph wrapper yields
    # DATASET_DEFAULT_GRAPH_ID instead. Normalize both to the same scope.
    if context is None:
        return DATASET_DEFAULT_GRAPH_ID
    identifier = getattr(context, "identifier", context)
    if identifier is None:
        return DATASET_DEFAULT_GRAPH_ID
    return identifier


def _merge_provenance(into: ProvenanceMap, node, graph_ids: Set[GraphId]) -> None:
    bucket = into.setdefault(node, set())
    bucket.update(graph_ids)


def value_nodes_from_path_with_provenance(
    sg,
    focus,
    path_val,
    target_graph,
    inverse: bool = False,
    recursion: int = 0,
    _attempts: Optional[Set[GraphId]] = None,
) -> ProvenanceMap:
    """Evaluate a SHACL path, keeping every full path inside one graph scope.

    The result maps each reached value node to the graph ids that support a
    complete path from the focus node to it. Edges taken from different graphs
    are never joined into a single path; see
    https://www.w3.org/TR/shacl/#property-paths for the path semantics.

    When ``_attempts`` is given, every graph touched by an edge traversal is
    recorded in it, including traversals that could not be joined into a
    complete path, so reports can name the boundary of an un-reproducible path.
    """
    # Link: https://www.w3.org/TR/shacl/#property-paths
    if isinstance(path_val, URIRef):
        provenance: ProvenanceMap = {}
        edges = _edge_objects(target_graph, focus, path_val, inverse)
        if _attempts is not None:
            _attempts.update(graph_id for _node, graph_id in edges)
        for node, graph_id in edges:
            _merge_provenance(provenance, node, {graph_id})
        return provenance
    if isinstance(path_val, Literal):
        raise ShapeLoadError(
            "Values of a property path cannot be a Literal.",
            "https://www.w3.org/TR/shacl/#property-paths",
        )
    # At this point, path_val _must_ be a BNode
    # TODO, the path_val BNode must be value of exactly one sh:path subject in the SG.
    if recursion >= 10:
        raise ReportableRuntimeError("Path traversal depth is too much!")
    find_list = set(sg.graph.objects(path_val, RDF.first))
    if len(find_list) > 0:
        first_node = next(iter(find_list))
        rest_nodes = set(sg.graph.objects(path_val, RDF.rest))
        go_deeper = True
        if len(rest_nodes) < 1:
            if recursion == 0:
                raise ReportableRuntimeError("A list of SHACL Paths must contain at least two path items.")
            else:
                go_deeper = False
        rest_node = next(iter(rest_nodes))
        if rest_node == RDF.nil:
            if recursion == 0:
                raise ReportableRuntimeError("A list of SHACL Paths must contain at least two path items.")
            else:
                go_deeper = False
        first_provenance = value_nodes_from_path_with_provenance(
            sg, focus, first_node, target_graph, inverse=inverse, recursion=recursion + 1, _attempts=_attempts
        )
        if not go_deeper:
            return first_provenance
        sequence_provenance: ProvenanceMap = {}
        for intermediate_node, first_scopes in first_provenance.items():
            rest_provenance = value_nodes_from_path_with_provenance(
                sg,
                intermediate_node,
                rest_node,
                target_graph,
                inverse=inverse,
                recursion=recursion + 1,
                _attempts=_attempts,
            )
            for value_node, rest_scopes in rest_provenance.items():
                joined_scopes = intersect_scopes(first_scopes, rest_scopes)
                if joined_scopes:
                    _merge_provenance(sequence_provenance, value_node, joined_scopes)
        return sequence_provenance

    find_inverse = set(sg.graph.objects(path_val, SH_inversePath))
    if len(find_inverse) > 0:
        inverse_path = next(iter(find_inverse))
        return value_nodes_from_path_with_provenance(
            sg,
            focus,
            inverse_path,
            target_graph,
            inverse=not inverse,
            recursion=recursion + 1,
            _attempts=_attempts,
        )

    find_alternatives = set(sg.graph.objects(path_val, SH_alternativePath))
    if len(find_alternatives) > 0:
        alternatives_list = next(iter(find_alternatives))
        alternative_provenance: ProvenanceMap = {}
        visited_alternatives = 0
        for alternative in sg.graph.items(alternatives_list):
            found = value_nodes_from_path_with_provenance(
                sg,
                focus,
                alternative,
                target_graph,
                inverse=inverse,
                recursion=recursion + 1,
                _attempts=_attempts,
            )
            visited_alternatives += 1
            for node, graph_ids in found.items():
                _merge_provenance(alternative_provenance, node, graph_ids)
        if visited_alternatives < 2:
            raise ReportableRuntimeError("List of SHACL alternate paths must have at least two path items.")
        return alternative_provenance

    find_zero_or_more = set(sg.graph.objects(path_val, SH_zeroOrMorePath))
    if len(find_zero_or_more) > 0:
        zm_path = next(iter(find_zero_or_more))
        # The zero-length step always reaches the focus node without an edge.
        collection: ProvenanceMap = {focus: {UNSCOPED_GRAPH}}
        frontier = value_nodes_from_path_with_provenance(
            sg, focus, zm_path, target_graph, inverse=inverse, recursion=recursion + 1, _attempts=_attempts
        )
        visited_nodes = {focus}
        while frontier:
            next_frontier: ProvenanceMap = {}
            for current_node, current_scopes in frontier.items():
                if current_node in visited_nodes:
                    continue
                visited_nodes.add(current_node)
                _merge_provenance(collection, current_node, current_scopes)
                deeper = value_nodes_from_path_with_provenance(
                    sg,
                    current_node,
                    zm_path,
                    target_graph,
                    inverse=inverse,
                    recursion=recursion + 1,
                    _attempts=_attempts,
                )
                for deeper_node, deeper_scopes in deeper.items():
                    joined_scopes = intersect_scopes(current_scopes, deeper_scopes)
                    if joined_scopes:
                        _merge_provenance(next_frontier, deeper_node, joined_scopes)
            frontier = next_frontier
        return collection

    find_one_or_more = set(sg.graph.objects(path_val, SH_oneOrMorePath))
    if len(find_one_or_more) > 0:
        one_or_more_path = next(iter(find_one_or_more))
        collection: ProvenanceMap = {}
        frontier = value_nodes_from_path_with_provenance(
            sg,
            focus,
            one_or_more_path,
            target_graph,
            inverse=inverse,
            recursion=recursion + 1,
            _attempts=_attempts,
        )
        visited_nodes = set()
        while frontier:
            next_frontier: ProvenanceMap = {}
            for current_node, current_scopes in frontier.items():
                if current_node in visited_nodes:
                    continue
                visited_nodes.add(current_node)
                _merge_provenance(collection, current_node, current_scopes)
                deeper = value_nodes_from_path_with_provenance(
                    sg,
                    current_node,
                    one_or_more_path,
                    target_graph,
                    inverse=inverse,
                    recursion=recursion + 1,
                    _attempts=_attempts,
                )
                for deeper_node, deeper_scopes in deeper.items():
                    joined_scopes = intersect_scopes(current_scopes, deeper_scopes)
                    if joined_scopes:
                        _merge_provenance(next_frontier, deeper_node, joined_scopes)
            frontier = next_frontier
        return collection

    find_zero_or_one = set(sg.graph.objects(path_val, SH_zeroOrOnePath))
    if len(find_zero_or_one) > 0:
        zero_or_one_path = next(iter(find_zero_or_one))
        # The zero-length step always reaches the focus node without an edge.
        collection: ProvenanceMap = {focus: {UNSCOPED_GRAPH}}
        found = value_nodes_from_path_with_provenance(
            sg,
            focus,
            zero_or_one_path,
            target_graph,
            inverse=inverse,
            recursion=recursion + 1,
            _attempts=_attempts,
        )
        for node, graph_ids in found.items():
            _merge_provenance(collection, node, graph_ids)
        return collection
    remaining = set(sg.graph.predicate_objects(path_val))
    if len(remaining) > 0:
        raise ShapeLoadError(
            "{} is not a known property for a sh:path. Malformed shape?".format(str(next(iter(remaining))[0])),
            "https://www.w3.org/TR/shacl/#property-paths",
        )

    raise ShapeLoadError(
        "Cannot get any values from sh:path property. Malformed shape?", "https://www.w3.org/TR/shacl/#property-paths"
    )


def value_nodes_from_path(
    sg,
    focus,
    path_val,
    target_graph,
    inverse: bool = False,
    recursion: int = 0,
    tracker: Optional["PathScopeTracker"] = None,
) -> Set:
    """Backward-compatible wrapper returning the reached value nodes as a set.

    When a :class:`PathScopeTracker` is supplied, the supporting graph of each
    value node is recorded on it for report provenance, together with every
    graph touched while attempting the path so that un-reproducible results can
    still name the graph boundary involved.
    """
    attempts: Set[GraphId] = set()
    provenance = value_nodes_from_path_with_provenance(
        sg,
        focus,
        path_val,
        target_graph,
        inverse=inverse,
        recursion=recursion,
        _attempts=attempts,
    )
    if tracker is not None:
        tracker.attach(focus, path_val, provenance)
        tracker.attach_attempts(focus, path_val, attempts)
    return set(provenance.keys())


def nodes_from_node_expression(
    expr, focus_node, data_graph: 'GraphLike', sg: 'ShapesGraph', recurse_depth=0
) -> Union[Set[Union['RDFNode', None]], List[Union['RDFNode', None]]]:
    # https://www.w3.org/TR/shacl-af/#node-expressions
    if expr == SH_this:
        return [focus_node]
    if isinstance(expr, (rdflib.URIRef, rdflib.Literal)):
        return [expr]
    elif isinstance(expr, rdflib.BNode):
        unions = set(sg.objects(expr, SH_union))
        intersections = set(sg.objects(expr, SH_intersection))
        if len(unions) and len(intersections):
            raise ReportableRuntimeError("Cannot have sh:intersection and sh:union on the same bnode.")
        if recurse_depth > 8 and (len(unions) or len(intersections)):
            warn(Warning("sh:union, sh:intersection, or sh:function args depth too deep. Won't capture all of it!"))
            return []
        if len(unions):
            union_list = next(iter(unions))
            parts = list(sg.graph.items(union_list))
            all_nodes: Set[Union['RDFNode', None]] = set()
            for p in parts:
                new_parts = nodes_from_node_expression(p, focus_node, data_graph, sg, recurse_depth=recurse_depth + 1)
                all_nodes = all_nodes.union(new_parts)
            return all_nodes
        if len(intersections):
            inter_list = next(iter(intersections))
            parts = list(sg.graph.items(inter_list))
            inter_nodes: Set[Union['RDFNode', None]] = set()
            new = True
            for p in parts:
                new_parts = nodes_from_node_expression(p, focus_node, data_graph, sg, recurse_depth=recurse_depth + 1)
                if new is True:
                    inter_nodes = set(iter(new_parts))
                    new = False
                else:
                    inter_nodes = inter_nodes.intersection(new_parts)
            return inter_nodes
        path_nodes = set(sg.objects(expr, SH_path))
        if len(path_nodes) > 0:
            path_results: Set[Union['RDFNode', None]] = set()
            for p in path_nodes:
                vals = value_nodes_from_path(sg, focus_node, p, data_graph)
                path_results.update(vals)
            return path_results
        filter_shapes = set(sg.objects(expr, SH_filterShape))
        nodes_nodes = set(sg.objects(expr, SH_nodes))
        if len(filter_shapes) > 0:  # pragma: no cover
            # Note: There's no tests for this whole filterShapes feature!
            if len(nodes_nodes) > 1:
                warn(Warning("More than one sh:nodes found. Using the first one."))
            elif len(nodes_nodes) < 1:
                raise ReportableRuntimeError("The Node FilterShape {} does not have sh:nodes.".format(expr))
            filter_shape = next(iter(filter_shapes))
            filter_shape = sg.lookup_shape_from_node(filter_shape)
            nodes_expr = next(iter(nodes_nodes))
            to_filter = nodes_from_node_expression(
                nodes_expr, focus_node, data_graph, sg, recurse_depth=recurse_depth + 1
            )
            passes = set()
            for n in to_filter:
                conforms, reports = filter_shape.validate(data_graph, n)
                if conforms:
                    passes.add(n)
            return passes
        # Got to here, the only other possibility is this is a FunctionExpression.
        remain_pairs = set(sg.predicate_objects(expr))
        fn_pairs = set()
        for rk, rv in remain_pairs:
            if rk == SH_message:
                # expressions can have a message, it doesn't affect the function
                continue
            elif isinstance(rv, rdflib.Literal):
                # pair is not list-valued, it can't be a function
                continue
            else:
                has_first = list(sg.objects(rv, RDF.first))
                if len(has_first) > 0:
                    fn_pairs.add((rk, rv))
        if len(fn_pairs) > 1:
            warn(Warning("More than one FunctionExpression found. Using the first one."))
        fnexpr, fnargslist = next(iter(fn_pairs))
        # find the function!
        try:
            function, optionals = sg.get_shacl_function(fnexpr)
        except KeyError:
            raise ReportableRuntimeError(
                "The SHACLFunction {} was not defined in this SHACL Shapes file.".format(fnexpr)
            )
        argslist_parts = list(sg.graph.items(fnargslist))
        args_sets: List[Union[List[Union['RDFNode', None]], Set[Union['RDFNode', None]]]] = [
            nodes_from_node_expression(p, focus_node, data_graph, sg, recurse_depth=recurse_depth + 1)
            for p in argslist_parts
        ]
        num_args_sets = len(args_sets)
        num_expected_args = len(optionals)
        if num_args_sets > num_expected_args:
            raise ReportableRuntimeError("Too many arguments given for {}".format(fnexpr))
        if num_args_sets < num_expected_args:
            # not enough, but some might be optional
            num_diff = num_expected_args - num_args_sets
            args_sets_slice = args_sets[0 - num_diff :]
            all_empty = itertools.accumulate(
                {True if len(a) < 1 else False for a in args_sets_slice}, func=operator.or_
            )
            if all_empty:
                args_sets = args_sets[:num_expected_args]
            else:
                raise ReportableRuntimeError(
                    "Too few arguments given for {}, with non-optional arguments.".format(fnexpr)
                )
        add_nones = set()
        for i, a in enumerate(args_sets):
            if len(a) < 1:
                if optionals[i] is False:
                    warn(Warning("Got an empty set of nodes for a non-optional argument in {}.".format(fnexpr)))
                    return []
                else:
                    add_nones.add(i)
        for i in add_nones:
            args_sets[i] = [None]
        args_permutations = list(itertools.product(*args_sets))
        responses = set()
        for permus in args_permutations:
            result = function(data_graph, *permus)
            responses.add(result)
        return responses
    else:
        raise NotImplementedError("Unsupported expression {}".format(expr))
