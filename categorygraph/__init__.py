"""Category membership and parent-edge graph built from wikitext sources.

The input is a batch of page sources plus the site namespace declaration.
Only ``[[Category:Name]]``-style links declared directly on a page create
membership. Ordinary links, leading-colon escaped links and protected text
(``nowiki``/``pre``/comments) never create membership.
"""

from .graph import (
    AncestorConclusion,
    CategoryGraph,
    EvidenceHop,
    MemberEdge,
    NamespaceDeclaration,
    ParentEdge,
    build_graph,
)

__all__ = [
    'AncestorConclusion',
    'CategoryGraph',
    'EvidenceHop',
    'MemberEdge',
    'NamespaceDeclaration',
    'ParentEdge',
    'build_graph',
]
