# -*- coding: utf-8 -*-
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Tuple, Type, Union

from rdflib import BNode, URIRef

from ..consts import RDF_type, SH_rule, SH_SPARQLRule, SH_TripleRule
from ..errors import ReportableRuntimeError, RuleLoadError
from ..pytypes import RDFNode, SHACLExecutor
from ..rules.sparql import SPARQLRule
from ..rules.triple import TripleRule

if TYPE_CHECKING:
    from ..graph_abstraction import DataGraph
    from ..shape import Shape
    from ..shapes_graph import ShapesGraph
    from .shacl_rule import SHACLRule


def gather_rules(
    executor: SHACLExecutor,
    shacl_graph: 'ShapesGraph',
    from_shapes: Union[Sequence[Union[URIRef, BNode]], None] = None,
) -> Dict['Shape', List['SHACLRule']]:
    """
    :param executor:
    :type executor: SHACLExecutor
    :param shacl_graph:
    :type shacl_graph: ShapesGraph
    :return:
    :rtype: Dict[Shape, List[SHACLRule]]
    """
    triple_rule_nodes = set(shacl_graph.subjects(RDF_type, SH_TripleRule))
    sparql_rule_nodes = set(shacl_graph.subjects(RDF_type, SH_SPARQLRule))
    if shacl_graph.js_enabled:
        from pyshacl.extras.js.rules import JSRule, SH_JSRule

        js_rule_nodes = set(shacl_graph.subjects(RDF_type, SH_JSRule))
        use_JSRule: Union[bool, Type] = JSRule
    else:
        use_JSRule = False
        js_rule_nodes = set()
    overlaps = triple_rule_nodes.intersection(sparql_rule_nodes)
    if len(overlaps) > 0:
        raise RuleLoadError(
            "A SHACL Rule cannot be both a TripleRule and a SPARQLRule.",
            "https://www.w3.org/TR/shacl-af/#rules-syntax",
        )
    overlaps = triple_rule_nodes.intersection(js_rule_nodes)
    if len(overlaps) > 0:
        raise RuleLoadError(
            "A SHACL Rule cannot be both a TripleRule and a JSRule.",
            "https://www.w3.org/TR/shacl-af/#rules-syntax",
        )
    overlaps = sparql_rule_nodes.intersection(js_rule_nodes)
    if len(overlaps) > 0:
        raise RuleLoadError(
            "A SHACL Rule cannot be both a SPARQLRule and a JSRule.",
            "https://www.w3.org/TR/shacl-af/#rules-syntax",
        )
    used_rules = shacl_graph.subject_objects(SH_rule)
    ret_rules = defaultdict(list)
    for sub, obj in used_rules:
        if from_shapes is not None and sub not in from_shapes:
            # Skipping rule that is not in the whitelist of Shapes to use
            continue
        try:
            shape: Shape = shacl_graph.lookup_shape_from_node(sub)
        except (AttributeError, KeyError):
            raise RuleLoadError(
                "The shape that rule is attached to is not a valid SHACL Shape.",
                "https://www.w3.org/TR/shacl-af/#rules-syntax",
            )
        if obj in triple_rule_nodes:
            rule: SHACLRule = TripleRule(executor, shape, obj, iterate=executor.iterate_rules)
        elif obj in sparql_rule_nodes:
            rule = SPARQLRule(executor, shape, obj)
        elif use_JSRule and callable(use_JSRule) and obj in js_rule_nodes:
            rule = use_JSRule(executor, shape, obj)
        else:
            raise RuleLoadError(
                "when using sh:rule, the Rule must be defined as either a TripleRule or SPARQLRule.",
                "https://www.w3.org/TR/shacl-af/#rules-syntax",
            )
        ret_rules[shape].append(rule)
    return ret_rules


RULES_ITERATE_LIMIT = 100
DEFAULT_MAX_RULE_ROUNDS = RULES_ITERATE_LIMIT

CLOSURE_STATUS_CONVERGED = "converged"
CLOSURE_STATUS_INCOMPLETE = "incomplete"
CLOSURE_STATUS_NO_RULES = "no_rules"


def _order_value(value: Any) -> str:
    return str(value)


def _term_n3(term: Any) -> str:
    try:
        return term.n3()
    except Exception:  # pragma: no cover - defensive, all RDF terms support n3()
        return str(term)


def _triple_n3(triple: Sequence[Any]) -> str:
    return "{} {} {} .".format(_term_n3(triple[0]), _term_n3(triple[1]), _term_n3(triple[2]))


def _triple_key(triple: Sequence[Any]) -> Tuple[str, str, str]:
    return (_term_n3(triple[0]), _term_n3(triple[1]), _term_n3(triple[2]))


def _graph_size(graph: Any) -> Union[int, None]:
    """Return the graph size without triggering an expensive store query.

    On an Oxigraph-backed graph ``len()`` runs a ``COUNT`` query over the whole
    store. Size is only used as a fallback for rule implementations that do not
    expose the triples they added, so it is not worth that cost there.
    """

    if getattr(graph, "is_oxigraph", False):
        return None
    try:
        return len(graph)
    except Exception:  # pragma: no cover - exotic graph implementations
        return None


class RuleClosureReport(dict):
    """Evidence for one SHACL Rules closure run.

    The report is a ``dict`` for easy JSON serialization and also exposes its
    keys as attributes (``report.status``, ``report.rounds``, ...).

    ``status`` is one of:

    * ``"converged"`` -- a complete round produced no new triples. The
      ``closure`` is a fixed point for the rules that were executed.
    * ``"incomplete"`` -- the resource limit (``max_rounds``) was reached
      while the last executed round still produced new triples, so no fixed
      point was established.
    * ``"no_rules"`` -- there were no active rules to execute.

    ``reason`` explains the status: ``"no_new_triples"`` for a converged
    run, ``"max_rounds_exhausted"`` for an incomplete run, and ``None`` when
    there were no rules.
    """

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - normal AttributeError path
            raise AttributeError(item) from exc

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable copy of the report."""
        data = dict(self)
        data["closure"] = list(data.get("closure", ()))
        data["closure_triples"] = [_triple_n3(t) for t in data.get("closure_triples", ())]
        return data

    def to_json(self, **kwargs: Any) -> str:
        import json

        kwargs.setdefault("sort_keys", True)
        return json.dumps(self.to_dict(), **kwargs)


def _new_report(max_rounds: int, rules: List[Dict[str, Any]]) -> RuleClosureReport:
    return RuleClosureReport(
        status=CLOSURE_STATUS_NO_RULES,
        converged=False,
        incomplete=False,
        reason=None,
        max_rounds=max_rounds,
        round_count=0,
        total_added=0,
        total_rule_firings=0,
        unattributed_added=0,
        rules=rules,
        closure=[],
        closure_triples=[],
        closure_complete=True,
        provenance={},
        initial_size=None,
        final_size=None,
        rounds=[],
    )


def _sorted_rules(shapes_rules: Dict) -> List[Tuple[Any, Any]]:
    """Return (shape, rule) pairs in the order they must execute.

    Shapes are ordered by ``sh:order`` and then by node IRI so that runs are
    deterministic even when several shapes share the same order. Rules inside
    a shape follow the same rule. Rules are then executed in that order in
    every round.
    """

    ordered: List[Tuple[Any, Any]] = []
    sorted_shapes = sorted(shapes_rules.items(), key=lambda x: (x[0].order, str(x[0].node)))
    for shape, rules in sorted_shapes:
        sorted_rules = sorted(rules, key=lambda x: (x.order, str(x.node)))
        for rule in sorted_rules:
            if rule.deactivated:
                continue
            ordered.append((shape, rule))
    return ordered


def apply_rules_closure(
    executor: SHACLExecutor,
    shapes_rules: Dict,
    data_graph: 'DataGraph',
    focus_nodes: Union[Sequence[RDFNode], None] = None,
    max_rounds: Union[int, None] = None,
) -> Tuple[int, RuleClosureReport]:
    """Run SHACL Rules to a fixed point in ordered global rounds.

    Every round executes all active rules in ``sh:order`` order. A rule later
    in the round (or in a later round) can therefore consume facts produced by
    an earlier rule. Rounds continue until a full round adds no new triple.

    The return value is ``(total_added, report)``. When the round budget is
    exhausted while the last round still changed the graph, the report status
    is ``"incomplete"``; a run is never reported as converged merely because
    the final round did not raise an exception.
    """

    if max_rounds is None:
        max_rounds = int(getattr(executor, "max_rule_rounds", DEFAULT_MAX_RULE_ROUNDS))
    try:
        max_rounds = int(max_rounds)
    except (TypeError, ValueError):
        raise ReportableRuntimeError(f"Invalid maximum number of SHACL Rules rounds: {max_rounds!r}")

    ordered = _sorted_rules(shapes_rules)
    rule_descriptors = [
        {
            "shape": str(shape.node),
            "rule": str(rule.node),
            "shape_order": _order_value(shape.order),
            "rule_order": _order_value(rule.order),
            "type": rule.__class__.__name__,
        }
        for shape, rule in ordered
    ]
    report = _new_report(max_rounds, rule_descriptors)

    if max_rounds < 1:
        if ordered:
            report["status"] = CLOSURE_STATUS_INCOMPLETE
            report["incomplete"] = True
            report["reason"] = "max_rounds_exhausted"
        else:
            report["status"] = CLOSURE_STATUS_NO_RULES
            report["converged"] = True
        return 0, report

    if not ordered:
        report["status"] = CLOSURE_STATUS_NO_RULES
        report["converged"] = True
        return 0, report

    closure: Dict[Tuple[str, str, str], Tuple[Any, Any, Any]] = {}
    provenance: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    total_added = 0
    total_firings = 0

    try:
        initial_size: Union[int, None] = _graph_size(data_graph)
    except Exception:  # pragma: no cover - exotic graph implementations
        initial_size = None
    report["initial_size"] = initial_size

    for round_number in range(1, max_rounds + 1):
        round_new: Dict[Tuple[str, str, str], Tuple[Any, Any, Any]] = {}
        firings: List[Dict[str, Any]] = []
        unattributed_changes = 0
        for shape, rule in ordered:
            # Closure rounds provide the outer iteration. Disabling the
            # rule-local loop avoids a hidden 100-iteration budget inside a
            # single rule and keeps one round equal to one ordered pass.
            previous_iterate = getattr(rule, "iterate", False)
            size_before = _graph_size(data_graph)
            try:
                rule.iterate = False
                modified = rule.apply(data_graph, focus_nodes=focus_nodes)
            finally:
                rule.iterate = previous_iterate

            added_triples = [t for t in getattr(rule, "last_added", ()) if len(t) >= 3]
            added_keys = {_triple_key(t) for t in added_triples}
            for triple in added_triples:
                key = _triple_key(triple)
                if key in closure or key in round_new:
                    continue
                round_new[key] = (triple[0], triple[1], triple[2])
                provenance[_triple_n3(triple)].append(
                    {
                        "round": round_number,
                        "shape": str(shape.node),
                        "rule": str(rule.node),
                        "shape_order": _order_value(shape.order),
                        "rule_order": _order_value(rule.order),
                        "type": rule.__class__.__name__,
                    }
                )
            # Fallback for rule implementations that report a modification but
            # do not expose the triples they added (including third-party
            # subclasses of SHACLRule that never populate ``last_added``). Use
            # the graph size delta when cheaply available, otherwise the
            # reported modification count. An unverifiable change must never
            # masquerade as "no new facts".
            size_increased = False
            if modified and not added_triples:
                size_after = _graph_size(data_graph)
                if size_before is not None and size_after is not None and size_after > size_before:
                    size_increased = True
                    unattributed_changes += size_after - size_before
                else:
                    size_increased = True
                    unattributed_changes += max(int(modified), 1)
            if added_triples:
                total_firings += 1
                firings.append(
                    {
                        "shape": str(shape.node),
                        "rule": str(rule.node),
                        "shape_order": _order_value(shape.order),
                        "rule_order": _order_value(rule.order),
                        "type": rule.__class__.__name__,
                        "changed": True,
                        "added": len(added_keys),
                        "triples": sorted({_triple_n3(t) for t in added_triples}),
                    }
                )
            elif modified or size_increased:
                total_firings += 1
                firings.append(
                    {
                        "shape": str(shape.node),
                        "rule": str(rule.node),
                        "shape_order": _order_value(shape.order),
                        "rule_order": _order_value(rule.order),
                        "type": rule.__class__.__name__,
                        "changed": True,
                        "added": int(modified) if not size_increased else max(int(modified), 1),
                        "triples": [],
                        "unattributed": True,
                    }
                )
            else:
                # Record rules that ran but produced nothing: a "no new
                # facts" conclusion is only meaningful together with the
                # evidence that every rule was executed in this round.
                firings.append(
                    {
                        "shape": str(shape.node),
                        "rule": str(rule.node),
                        "shape_order": _order_value(shape.order),
                        "rule_order": _order_value(rule.order),
                        "type": rule.__class__.__name__,
                        "changed": False,
                        "added": 0,
                        "triples": [],
                    }
                )

        round_added = len(round_new)
        round_changed = round_added > 0 or unattributed_changes > 0
        report["rounds"].append(
            {
                "round": round_number,
                "added": round_added,
                "unattributed_added": unattributed_changes,
                "changed": round_changed,
                "firings": firings,
                "new_triples": [key for key in sorted(round_new)],
            }
        )
        report["round_count"] = round_number
        closure.update(round_new)
        total_added += round_added
        report["unattributed_added"] = report.get("unattributed_added", 0) + unattributed_changes

        if not round_changed:
            report["status"] = CLOSURE_STATUS_CONVERGED
            report["converged"] = True
            report["incomplete"] = False
            report["reason"] = "no_new_triples"
            break

        if round_number >= max_rounds:
            # The last allowed round still changed the graph. There may be
            # more facts to derive, so this is explicitly NOT a closure.
            report["status"] = CLOSURE_STATUS_INCOMPLETE
            report["converged"] = False
            report["incomplete"] = True
            report["reason"] = "max_rounds_exhausted"
            break

    try:
        report["final_size"] = _graph_size(data_graph)
    except Exception:  # pragma: no cover - exotic graph implementations
        report["final_size"] = None
    report["total_added"] = total_added
    report["total_rule_firings"] = total_firings
    report["closure_complete"] = not any(r.get("unattributed_added", 0) for r in report["rounds"])
    report["closure"] = sorted(closure.keys())
    report["closure_triples"] = [closure[key] for key in sorted(closure)]
    report["provenance"] = {key: provenance[key] for key in sorted(provenance)}
    return total_added, report


def apply_rules(
    executor: SHACLExecutor,
    shapes_rules: Dict,
    data_graph: 'DataGraph',
    focus_nodes: Union[Sequence[RDFNode], None] = None,
    return_report: bool = False,
) -> Union[int, Tuple[int, RuleClosureReport]]:
    closure_mode = bool(getattr(executor, "iterate_rules", False) or getattr(executor, "rule_closure", False))
    if closure_mode:
        total_modified, report = apply_rules_closure(
            executor,
            shapes_rules,
            data_graph,
            focus_nodes=focus_nodes,
            max_rounds=getattr(executor, "max_rule_rounds", None),
        )
        try:
            setattr(executor, "rules_report", report)
        except Exception:  # pragma: no cover - only for exotic executor objects
            pass
        if return_report:
            return total_modified, report
        return total_modified

    # Legacy single-pass behaviour: each shape's rules run in ``sh:order``
    # order, but facts produced by a later shape do not cause an earlier
    # shape to run again.
    sorted_shapes_rules: List[Tuple[Any, Any]] = sorted(
        shapes_rules.items(), key=lambda x: (x[0].order, str(x[0].node))
    )
    total_modified = 0
    for shape, rules in sorted_shapes_rules:
        rules = sorted(rules, key=lambda x: (x.order, str(x.node)))
        _iterate_limit = int(RULES_ITERATE_LIMIT)
        while True:
            if _iterate_limit < 1:
                raise ReportableRuntimeError(
                    f"SHACL Shape Rule iteration exceeded iteration limit of {RULES_ITERATE_LIMIT}."
                )
            _iterate_limit -= 1
            this_modified = 0
            for r in rules:
                if r.deactivated:
                    continue
                n_modified = r.apply(data_graph, focus_nodes=focus_nodes)
                this_modified += n_modified
            if this_modified > 0:
                total_modified += this_modified
                if executor.iterate_rules:
                    continue
                else:
                    break
            break
    return total_modified
