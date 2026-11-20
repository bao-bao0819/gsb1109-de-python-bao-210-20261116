# -*- coding: utf-8 -*-
#

from dataclasses import dataclass, field
from typing import Any, List, Optional, Set, Union

from rdflib import Dataset, Graph, Literal
from rdflib.term import IdentifiedNode, URIRef

GraphLike = Union[Dataset, Graph]
RDFNode = Union[IdentifiedNode, Literal]


@dataclass
class TargetSuspension:
    """Records that a shape's own target declarations are suspended for a run.

    Only the top-level (root) targeting of the named shapes is suspended. The
    shapes remain reachable through shape-expecting parameters such as
    sh:node, sh:property, sh:or and sh:not, where they are still evaluated.
    """

    shape_nodes: Set[URIRef]
    reason: Optional[str] = None


@dataclass
class ShapeEvaluationTrace:
    """One evaluation of a shape against a focus node, with its call path.

    ``shape_path`` holds the chain of nested shapes that led to this
    evaluation, outermost first. An empty path means the shape was evaluated
    as a root shape, via its own targets or a manually supplied focus node.
    """

    shape_node: Any
    focus_node: Any
    conforms: bool
    shape_path: List[Any]
    depth: int
    root_shape_node: Any


@dataclass
class LogicalBranchTrace:
    """Outcome of one member shape inside a sh:and/sh:or/sh:xone/sh:not list."""

    root_shape_node: Any
    focus_node: Any
    constraint: URIRef
    member_shape_node: Any
    conforms: bool
    shape_path: List[Any]


@dataclass
class ValidationTrace:
    """Collected evidence of how every shape evaluation was reached.

    The trace is opt-in. When present, shape validation and the logical
    constraint components append structured records that can be correlated
    back to root shapes and input focus nodes, without altering the standard
    SHACL validation report.
    """

    shape_evaluations: List[ShapeEvaluationTrace] = field(default_factory=list)
    logical_branches: List[LogicalBranchTrace] = field(default_factory=list)

    def record_evaluation(
        self,
        shape_node: Any,
        focus_node: Any,
        conforms: bool,
        shape_path: List[Any],
    ) -> None:
        path_nodes = [shape.node if hasattr(shape, 'node') else shape for shape in shape_path]
        root_shape_node = path_nodes[0] if path_nodes else shape_node
        self.shape_evaluations.append(
            ShapeEvaluationTrace(
                shape_node=shape_node,
                focus_node=focus_node,
                conforms=conforms,
                shape_path=path_nodes,
                depth=len(path_nodes),
                root_shape_node=root_shape_node,
            )
        )

    def record_logical_branch(
        self,
        root_shape_node: Any,
        focus_node: Any,
        constraint: URIRef,
        member_shape_node: Any,
        conforms: bool,
        shape_path: List[Any],
    ) -> None:
        path_nodes = [shape.node if hasattr(shape, 'node') else shape for shape in shape_path]
        self.logical_branches.append(
            LogicalBranchTrace(
                root_shape_node=root_shape_node,
                focus_node=focus_node,
                constraint=constraint,
                member_shape_node=member_shape_node,
                conforms=conforms,
                shape_path=path_nodes,
            )
        )

    def evaluations_of(self, shape_node: Any) -> List[ShapeEvaluationTrace]:
        return [evaluation for evaluation in self.shape_evaluations if evaluation.shape_node == shape_node]

    def referencing_roots(self, shape_node: Any) -> List[Any]:
        roots: List[Any] = []
        for evaluation in self.shape_evaluations:
            if evaluation.shape_node == shape_node and evaluation.root_shape_node != shape_node:
                root = evaluation.root_shape_node
                if root not in roots:
                    roots.append(root)
        return roots


@dataclass
class SHACLExecutor:
    validator: Optional[object] = None
    advanced_mode: bool = False
    abort_on_first: bool = False
    allow_infos: bool = False
    allow_warnings: bool = False
    iterate_rules: bool = False
    debug: bool = False
    sparql_mode: bool = False
    max_validation_depth: int = 15
    focus_nodes: Optional[List[URIRef]] = None
    target_suspension: Optional[TargetSuspension] = None
    validation_trace: Optional[ValidationTrace] = None
