# -*- coding: utf-8 -*-
"""
Release-gate decision layer built on top of a SHACL validation report.

A SHACL validation report only tells you ``sh:conforms`` true/false. A release
pipeline needs richer semantics:

* a release is allowed when there are **Warnings** (or Infos) but blocked when
  there is at least one active **Violation**;
* nested ``sh:detail`` results carry their own severity, and a detail inherits
  the severity of every ancestor result on the path(s) that reach it;
* individual shapes can be exempted for a bounded set of objects and a bounded
  time window, and an exemption never propagates outside its declared object
  scope;
* the decision must keep every result (warnings included) while exposing the
  minimal set of evidence that actually blocks the release;
* a result reached through more than one detail path is counted exactly once;
* changing the evaluation time or the exemptions only re-judges the affected
  objects, and a single ``conforms`` boolean is never allowed to hide warnings.

The layer is intentionally pure with respect to the RDF report graph: it parses
the graph produced by :func:`pyshacl.validate` into an identity-indexed tree and
performs all gating logic on top of that. The original report graph is left
untouched, so no warning is ever erased.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import RDF

from pyshacl.consts import (
    SH_conforms,
    SH_detail,
    SH_focusNode,
    SH_result,
    SH_resultMessage,
    SH_resultPath,
    SH_resultSeverity,
    SH_sourceShape,
    SH_ValidationReport,
    SH_ValidationResult,
    SH_value,
    SH_Violation,
    SH_Info,
    SH_Warning,
)
from pyshacl.rdfutil import stringify_node

__all__ = [
    "Severity",
    "Exemption",
    "GateResult",
    "BlockingEvidence",
    "ReleaseDecision",
    "ReleaseGate",
    "build_release_gate",
]


class Severity(Enum):
    """Rankable SHACL severities. Higher value means more severe."""

    INFO = 1
    WARNING = 2
    VIOLATION = 3

    @property
    def is_blocking(self) -> bool:
        return self is Severity.VIOLATION

    @classmethod
    def from_uri(cls, uri: Optional[Any]) -> "Severity":
        if uri == SH_Violation:
            return cls.VIOLATION
        if uri == SH_Warning:
            return cls.WARNING
        if uri == SH_Info:
            return cls.INFO
        # An unknown / custom severity cannot be proven to block a release,
        # so it is treated as the weakest, non-blocking level.
        return cls.INFO


def _coerce_utc(moment: Optional[datetime]) -> Optional[datetime]:
    if moment is None:
        return None
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _node_id(node: Any) -> str:
    """Stable textual identity for an RDF term within a single report graph."""
    if isinstance(node, BNode):
        return f"_:{node}"
    return str(node)


def _term_sort_key(node: Any) -> str:
    return _node_id(node)


@dataclass(frozen=True)
class Exemption:
    """
    A scoped, time-bounded shape exemption.

    The exemption suppresses results that satisfy *all* of:

    * their ``sh:sourceShape`` is ``shape``;
    * their focus node is one of ``scope_nodes`` (the exempted object range);
    * the evaluation time is inside the half-open validity window.

    An exemption never applies to a focus node that is not explicitly listed,
    therefore it cannot propagate across object scope.
    """

    shape: Any
    scope_nodes: Tuple[Any, ...]
    not_before: Optional[datetime] = None
    not_after: Optional[datetime] = None
    reason: Optional[str] = None
    id: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.shape, (URIRef, BNode)):
            raise TypeError("Exemption.shape must be a URIRef or BNode identifying a shape.")
        if len(self.scope_nodes) < 1:
            raise ValueError("An exemption must declare at least one scope node.")
        # dataclass is frozen; bypass to normalise timezone-aware datetimes.
        if self.not_before is not None:
            object.__setattr__(self, "not_before", _coerce_utc(self.not_before))
        if self.not_after is not None:
            object.__setattr__(self, "not_after", _coerce_utc(self.not_after))

    def covers_object(self, focus_node: Any) -> bool:
        return focus_node in self.scope_nodes

    def active_at(self, moment: Optional[datetime]) -> bool:
        moment = _coerce_utc(moment) or datetime.now(timezone.utc)
        if self.not_before is not None and moment < self.not_before:
            return False
        if self.not_after is not None and moment >= self.not_after:
            return False
        return True

    def matches(self, focus_node: Any, source_shape: Any, moment: Optional[datetime]) -> bool:
        if not self.active_at(moment):
            return False
        if source_shape != self.shape:
            return False
        return self.covers_object(focus_node)


@dataclass
class GateResult:
    """
    A single validation result, enriched with release-gate bookkeeping.

    The identity is the validation-result node itself. Two parent results that
    both nest the same detail node therefore resolve to the same ``GateResult``
    and are counted once.
    """

    node: Any
    result_id: str
    focus_node: Optional[Any]
    source_shape: Optional[Any]
    result_path: Optional[Any]
    value_node: Optional[Any]
    severity: Severity
    messages: Tuple[str, ...]
    detail_ids: List[str] = field(default_factory=list)
    is_root: bool = False
    # Effective severity after ancestor inheritance and exemption resolution.
    effective_severity: Severity = Severity.INFO
    inherited_severity: Severity = Severity.INFO
    exemption: Optional[Exemption] = None
    exemption_ids: Tuple[str, ...] = ()

    @property
    def is_exempted(self) -> bool:
        return self.exemption is not None

    @property
    def blocks_release(self) -> bool:
        return (not self.is_exempted) and self.effective_severity.is_blocking

    def identity_key(self) -> Tuple[Any, ...]:
        """Content identity used to detect unchanged results across runs."""
        return (
            _node_id(self.focus_node),
            _node_id(self.source_shape),
            _node_id(self.result_path),
            _node_id(self.value_node),
            self.severity,
            tuple(sorted(self.messages)),
        )


@dataclass
class BlockingEvidence:
    """One minimal piece of evidence blocking the release."""

    result_id: str
    focus_node: Optional[Any]
    source_shape: Optional[Any]
    result_path: Optional[Any]
    value_node: Optional[Any]
    severity: Severity
    effective_severity: Severity
    inherited_severity: Severity
    messages: Tuple[str, ...]
    detail_ids: Tuple[str, ...]

    @classmethod
    def from_result(cls, result: GateResult) -> "BlockingEvidence":
        return cls(
            result_id=result.result_id,
            focus_node=result.focus_node,
            source_shape=result.source_shape,
            result_path=result.result_path,
            value_node=result.value_node,
            severity=result.severity,
            effective_severity=result.effective_severity,
            inherited_severity=result.inherited_severity,
            messages=result.messages,
            detail_ids=tuple(result.detail_ids),
        )


@dataclass
class ReleaseDecision:
    """The full release verdict. Warnings are retained, never collapsed."""

    allowed: bool
    conforms: bool
    evaluated_at: datetime
    results: List[GateResult]
    blocking_evidence: List[BlockingEvidence]
    counts: Dict[str, int]
    exempted_result_ids: Tuple[str, ...]
    warnings: List[GateResult]

    @property
    def blocked(self) -> bool:
        return not self.allowed

    def result_by_id(self, result_id: str) -> Optional[GateResult]:
        for result in self.results:
            if result.result_id == result_id:
                return result
        return None


class ReleaseGate:
    """
    Holds the parsed report tree plus the current exemptions and produces
    :class:`ReleaseDecision` objects.

    The same gate can be re-evaluated after the evaluation time changes or the
    exemption set changes. Re-judgement is scoped to affected objects, so
    unaffected :class:`GateResult` identities are preserved across runs.
    """

    def __init__(
        self,
        report_graph: Graph,
        exemptions: Optional[Iterable[Exemption]] = None,
        at: Optional[datetime] = None,
    ) -> None:
        self.report_graph = report_graph
        self._exemptions: List[Exemption] = list(exemptions or [])
        self._results: Dict[str, GateResult] = {}
        self._by_focus: Dict[str, Set[str]] = {}
        self._root_ids: List[str] = []
        self._parse_report(report_graph)
        conforms_node = self._read_conforms(report_graph)
        self._conforms: bool = bool(conforms_node)
        self._last_decision: Optional[ReleaseDecision] = None
        # Seed effective severities / exemption state for the initial time.
        self.evaluate(at=at)

    # ------------------------------------------------------------------ parse
    @staticmethod
    def _read_conforms(report_graph: Graph) -> Optional[Literal]:
        for report in report_graph.subjects(RDF.type, SH_ValidationReport):
            value = report_graph.value(report, SH_conforms)
            if value is not None:
                return value
        return None

    def _first_object(self, node: Any, predicate: Any) -> Optional[Any]:
        return self.report_graph.value(node, predicate)

    def _parse_report(self, report_graph: Graph) -> None:
        result_nodes = set(report_graph.subjects(RDF.type, SH_ValidationResult))
        # Build one GateResult per distinct validation-result node.
        for node in result_nodes:
            if isinstance(node, Literal):
                continue
            result_id = _node_id(node)
            if result_id in self._results:
                continue
            severity_uri = self._first_object(node, SH_resultSeverity)
            focus = self._first_object(node, SH_focusNode)
            source_shape = self._first_object(node, SH_sourceShape)
            result_path = self._first_object(node, SH_resultPath)
            value_node = self._first_object(node, SH_value)
            messages = tuple(
                str(m) for m in sorted(report_graph.objects(node, SH_resultMessage), key=lambda m: str(m))
            )
            detail_ids = [_node_id(d) for d in report_graph.objects(node, SH_detail)]
            result = GateResult(
                node=node,
                result_id=result_id,
                focus_node=focus,
                source_shape=source_shape,
                result_path=result_path,
                value_node=value_node,
                severity=Severity.from_uri(severity_uri),
                messages=messages,
                detail_ids=detail_ids,
            )
            self._results[result_id] = result
            focus_key = _node_id(focus)
            self._by_focus.setdefault(focus_key, set()).add(result_id)

        report_nodes = set(report_graph.subjects(RDF.type, SH_ValidationReport))
        rooted: Set[str] = set()
        for report in report_nodes:
            for root in report_graph.objects(report, SH_result):
                rid = _node_id(root)
                if rid in self._results and rid not in rooted:
                    self._root_ids.append(rid)
                    rooted.add(rid)
        for rid in rooted:
            self._results[rid].is_root = True

    # ------------------------------------------------------------ exemptions
    @property
    def exemptions(self) -> Tuple[Exemption, ...]:
        return tuple(self._exemptions)

    def add_exemption(self, exemption: Exemption) -> None:
        self._exemptions.append(exemption)

    def remove_exemption(self, exemption_id: str) -> Optional[Exemption]:
        for index, exemption in enumerate(self._exemptions):
            if exemption.id == exemption_id:
                return self._exemptions.pop(index)
        return None

    def affected_focus_nodes(self, exemption: Exemption) -> Set[Any]:
        """Focus nodes currently known to fall inside an exemption's scope."""
        affected: Set[Any] = set()
        for scope_node in exemption.scope_nodes:
            for rid in self._by_focus.get(_node_id(scope_node), set()):
                focus = self._results[rid].focus_node
                if focus is not None:
                    affected.add(focus)
        return affected

    def _matching_exemption(self, result: GateResult, moment: Optional[datetime]) -> Optional[Exemption]:
        if result.focus_node is None or result.source_shape is None:
            return None
        for exemption in self._exemptions:
            if exemption.matches(result.focus_node, result.source_shape, moment):
                return exemption
        return None

    # ------------------------------------------------------- severity inherit
    def _propagate_severity(self) -> None:
        """
        Walk every root-to-detail path, carrying the maximum ancestor severity
        down to the leaves. A shared detail reachable from several parents
        inherits the strongest severity found on any of the reaching paths.
        """
        # Reset inherited state before recomputing.
        for result in self._results.values():
            result.inherited_severity = result.severity
            result.effective_severity = result.severity

        def walk(node_id: str, ancestor_max: Severity, on_path: Set[str]) -> None:
            result = self._results[node_id]
            combined = max(result.severity, ancestor_max, key=lambda s: s.value)
            if combined.value > result.inherited_severity.value:
                result.inherited_severity = combined
            result.effective_severity = combined
            next_on_path = on_path | {node_id}
            for detail_id in result.detail_ids:
                if detail_id not in self._results or detail_id in on_path:
                    continue
                walk(detail_id, combined, next_on_path)

        for root_id in self._root_ids:
            walk(root_id, Severity.INFO, set())
        # Results unreachable from any root keep their own severity, which was
        # already assigned during the reset above.

    def _apply_exemptions(self, moment: Optional[datetime]) -> None:
        for result in self._results.values():
            exemption = self._matching_exemption(result, moment)
            result.exemption = exemption
            result.exemption_ids = (exemption.id,) if exemption is not None and exemption.id else ()

    def _blocking_ancestor_ids(self) -> Set[str]:
        """
        Ids of results that have at least one active, non-exempted blocking
        (effective Violation) ancestor on any reaching detail path.

        Exempted ancestors do not count, so an exempted parent can never hide a
        non-exempted violating descendant. A shared detail reached through both
        a blocking and a non-blocking parent is still marked, because it is
        already covered by the blocking branch.
        """
        marked: Set[str] = set()

        def walk(node_id: str, ancestor_blocks: bool, on_path: Set[str]) -> None:
            if ancestor_blocks:
                marked.add(node_id)
            result = self._results[node_id]
            child_blocks = ancestor_blocks or result.blocks_release
            for detail_id in result.detail_ids:
                if detail_id not in self._results or detail_id in on_path:
                    continue
                walk(detail_id, child_blocks, on_path | {node_id})

        for root_id in self._root_ids:
            walk(root_id, False, set())
        return marked

    # ------------------------------------------------------------- evaluation
    def evaluate(self, at: Optional[datetime] = None) -> ReleaseDecision:
        moment = _coerce_utc(at) or datetime.now(timezone.utc)
        self._propagate_severity()
        self._apply_exemptions(moment)

        ordered = self._ordered_results()
        warnings: List[GateResult] = []
        counts = {"violation": 0, "warning": 0, "info": 0, "exempted": 0, "total": len(ordered)}
        exempted_ids: List[str] = []
        active_violations: List[GateResult] = []

        for result in ordered:
            if result.is_exempted:
                counts["exempted"] += 1
                exempted_ids.append(result.result_id)
                continue
            if result.effective_severity is Severity.VIOLATION:
                counts["violation"] += 1
                active_violations.append(result)
            elif result.effective_severity is Severity.WARNING:
                counts["warning"] += 1
                warnings.append(result)
            else:
                counts["info"] += 1

        # Minimal, irredundant blocking evidence: keep only active violations
        # that are not already implied by a blocking ancestor on some path.
        redundant_ids = self._blocking_ancestor_ids()
        blocking = [
            BlockingEvidence.from_result(result)
            for result in active_violations
            if result.result_id not in redundant_ids
        ]

        allowed = len(active_violations) == 0
        decision = ReleaseDecision(
            allowed=allowed,
            conforms=self._conforms,
            evaluated_at=moment,
            results=ordered,
            blocking_evidence=blocking,
            counts=counts,
            exempted_result_ids=tuple(exempted_ids),
            warnings=warnings,
        )
        self._last_decision = decision
        return decision

    def _ordered_results(self) -> List[GateResult]:
        # Roots first, then the remaining detail nodes, deterministic by id.
        ordered_ids = list(self._root_ids)
        seen = set(ordered_ids)
        for rid in sorted(self._results):
            if rid not in seen:
                ordered_ids.append(rid)
                seen.add(rid)
        return [self._results[rid] for rid in ordered_ids if rid in self._results]

    # --------------------------------------------------------- incremental
    def reevaluate(
        self,
        report_graph: Optional[Graph] = None,
        at: Optional[datetime] = None,
        focus_nodes: Optional[Sequence[Any]] = None,
    ) -> ReleaseDecision:
        """
        Re-judge after a change.

        When ``focus_nodes`` is supplied, only the results whose focus node is
        in that set are rebuilt from ``report_graph``; every other result (and
        its identity) is retained. When ``focus_nodes`` is ``None`` the time /
        exemption change is applied to the whole held graph without rebuilding.
        """
        if report_graph is not None and focus_nodes is not None:
            self._replace_subtree(report_graph, set(focus_nodes))
            self.report_graph = report_graph
            self._conforms = bool(self._read_conforms(report_graph))
        elif report_graph is not None:
            self.report_graph = report_graph
            self._results = {}
            self._by_focus = {}
            self._root_ids = []
            self._parse_report(report_graph)
            self._conforms = bool(self._read_conforms(report_graph))
        return self.evaluate(at=at)

    def _replace_subtree(self, report_graph: Graph, focus_nodes: Set[Any]) -> None:
        target_keys = {_node_id(node) for node in focus_nodes}
        stale_ids = [
            rid
            for rid, result in self._results.items()
            if _node_id(result.focus_node) in target_keys
        ]
        for rid in stale_ids:
            result = self._results.pop(rid)
            bucket = self._by_focus.get(_node_id(result.focus_node))
            if bucket is not None:
                bucket.discard(rid)
                if not bucket:
                    self._by_focus.pop(_node_id(result.focus_node), None)
            if rid in self._root_ids:
                self._root_ids.remove(rid)

        # Re-parse the incoming graph, but only adopt results for affected focus
        # nodes plus any detail nodes hanging underneath them.
        incoming = ReleaseGate(report_graph, at=None)
        incoming_ids: Set[str] = set()
        for key in target_keys:
            incoming_ids.update(incoming._by_focus.get(key, set()))
        # Expand to full subtrees so nested details are carried across.
        expanded: Set[str] = set()
        stack = list(incoming_ids)
        while stack:
            rid = stack.pop()
            if rid in expanded or rid not in incoming._results:
                continue
            expanded.add(rid)
            result = incoming._results[rid]
            stack.extend(result.detail_ids)
        for rid in expanded:
            source = incoming._results[rid]
            self._results[rid] = source
            focus_key = _node_id(source.focus_node)
            self._by_focus.setdefault(focus_key, set()).add(rid)
            if source.is_root and rid not in self._root_ids:
                self._root_ids.append(rid)
        for rid in self._root_ids:
            if rid in self._results:
                self._results[rid].is_root = True

    # -------------------------------------------------------------- rendering
    def describe(self, graph: Optional[Graph] = None) -> str:
        """Human-readable rendering of the most recent decision."""
        if self._last_decision is None:
            self.evaluate()
        decision = self._last_decision
        assert decision is not None
        graph = graph or self.report_graph
        lines = [
            f"Release {'ALLOWED' if decision.allowed else 'BLOCKED'} "
            f"(sh:conforms={decision.conforms}) at {decision.evaluated_at.isoformat()}",
            "counts: " + ", ".join(f"{key}={value}" for key, value in sorted(decision.counts.items())),
        ]
        for evidence in decision.blocking_evidence:
            lines.append(
                "BLOCK "
                + self._format_node(evidence.focus_node, graph)
                + " shape="
                + self._format_node(evidence.source_shape, graph)
                + f" severity={evidence.effective_severity.name.lower()}"
            )
        for result in decision.warnings:
            lines.append("WARN  " + self._format_node(result.focus_node, graph))
        return "\n".join(lines)

    @staticmethod
    def _format_node(node: Optional[Any], graph: Graph) -> str:
        if node is None:
            return "<none>"
        try:
            return stringify_node(graph, node)
        except (LookupError, ValueError):
            return str(node)


def build_release_gate(
    report_graph: Graph,
    exemptions: Optional[Iterable[Exemption]] = None,
    at: Optional[datetime] = None,
) -> ReleaseGate:
    """Convenience factory matching the public module entry points."""
    return ReleaseGate(report_graph, exemptions=exemptions, at=at)
