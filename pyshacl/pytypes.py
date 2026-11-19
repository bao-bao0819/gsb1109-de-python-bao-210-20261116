# -*- coding: utf-8 -*-
#

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from rdflib import Dataset, Graph, Literal
from rdflib.term import IdentifiedNode, URIRef

GraphLike = Union[Dataset, Graph]
RDFNode = Union[IdentifiedNode, Literal]


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
    # Maps id() of a validation result blank node to the id of the constraint
    # evaluation that produced it. Results produced by the same constraint
    # evaluation (for example one per duplicate language tag) are distinct
    # facts and never merge, even when their standard result triples coincide.
    result_origins: Dict[int, int] = field(default_factory=dict)
    # Monotonic counter handing out a unique origin per constraint evaluation.
    next_result_origin: int = 0

    def take_result_origin(self) -> int:
        origin = self.next_result_origin
        self.next_result_origin += 1
        return origin
