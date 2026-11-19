# -*- coding: utf-8 -*-
#

from dataclasses import dataclass
from typing import List, Optional, Union

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
    # Per-run provenance tracker for path evaluation over multigraph datasets.
    # None for plain (single-graph) data graphs and SPARQL remote mode.
    path_scope_tracker: Optional[object] = None
