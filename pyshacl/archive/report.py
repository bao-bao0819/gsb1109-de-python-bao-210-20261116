# -*- coding: utf-8 -*-
"""Canonical extraction and comparison of SHACL validation reports."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from rdflib import BNode, Graph, URIRef
from rdflib.namespace import RDF

from ..consts import SH

# Blank node labels are freshly minted on every run, so they cannot be part of
# a stable report signature; normalize them to this marker instead.
BNODE_MARKER = "urn:pyshacl:archive:blank-node"


@dataclass(frozen=True)
class Violation:
    focus_node: Optional[str]
    source_shape: Optional[str]
    constraint_component: Optional[str]
    result_path: Optional[str]
    severity: Optional[str]
    messages: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "focus_node": self.focus_node,
            "source_shape": self.source_shape,
            "constraint_component": self.constraint_component,
            "result_path": self.result_path,
            "severity": self.severity,
            "messages": list(self.messages),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Violation":
        return cls(
            focus_node=data.get("focus_node"),
            source_shape=data.get("source_shape"),
            constraint_component=data.get("constraint_component"),
            result_path=data.get("result_path"),
            severity=data.get("severity"),
            messages=tuple(data.get("messages", [])),
        )


@dataclass
class QualityReport:
    conforms: bool
    focus_nodes: List[str]
    violations: List[Violation] = field(default_factory=list)
    report_text: str = ""
    report_ttl: str = ""

    def summary(self) -> Dict[str, Any]:
        return {
            "conforms": self.conforms,
            "focus_nodes": list(self.focus_nodes),
            "violation_count": len(self.violations),
            "violations": [violation.to_dict() for violation in self.violations],
        }


def _object_str(graph: Graph, subject: Any, predicate: URIRef) -> Optional[str]:
    values = sorted(BNODE_MARKER if isinstance(obj, BNode) else str(obj) for obj in graph.objects(subject, predicate))
    if not values:
        return None
    return values[0]


def extract_report(report_graph: Graph, report_text: str) -> QualityReport:
    conforms_triples = list(report_graph.subjects(SH.conforms, None))
    conforms = False
    for subject in conforms_triples:
        value = next(iter(report_graph.objects(subject, SH.conforms)))
        conforms = bool(value.toPython()) if hasattr(value, "toPython") else bool(value)
    violations: List[Violation] = []
    focus_nodes: Set[str] = set()
    for result in report_graph.subjects(RDF.type, SH.ValidationResult):
        focus = _object_str(report_graph, result, SH.focusNode)
        if focus is not None:
            focus_nodes.add(focus)
        messages = tuple(sorted(str(obj) for obj in report_graph.objects(result, SH.message)))
        violations.append(
            Violation(
                focus_node=focus,
                source_shape=_object_str(report_graph, result, SH.sourceShape),
                constraint_component=_object_str(report_graph, result, SH.sourceConstraintComponent),
                result_path=_object_str(report_graph, result, SH.resultPath),
                severity=_object_str(report_graph, result, SH.resultSeverity),
                messages=messages,
            )
        )
    violations.sort(key=lambda item: item.to_dict().__repr__())
    report_ttl = report_graph.serialize(format="turtle")
    return QualityReport(
        conforms=conforms,
        focus_nodes=sorted(focus_nodes),
        violations=violations,
        report_text=report_text or "",
        report_ttl=report_ttl,
    )


def reports_equivalent(first: QualityReport, second: QualityReport) -> bool:
    """Same pass conclusion, focus nodes, and violation signatures."""
    if first.conforms != second.conforms:
        return False
    if first.focus_nodes != second.focus_nodes:
        return False
    if len(first.violations) != len(second.violations):
        return False
    return {_violation_key(item) for item in first.violations} == {
        _violation_key(item) for item in second.violations
    }


def _violation_key(violation: Violation) -> Tuple[Any, ...]:
    return (
        violation.focus_node,
        violation.source_shape,
        violation.constraint_component,
        violation.result_path,
        violation.severity,
        violation.messages,
    )
