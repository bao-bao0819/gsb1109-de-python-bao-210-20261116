"""Build the category graph and answer ancestor queries with evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from .extract import CategoryReference, extract_category_references


@dataclass(frozen=True)
class NamespaceDeclaration:
    """The site's category namespace declaration.

    ``canonical`` is the canonical local name (for example ``Category`` or
    ``Kategorie``); ``aliases`` lists any extra accepted names. Matching is
    case-insensitive, as namespace names are in MediaWiki.
    """

    canonical: str
    aliases: tuple[str, ...] = ()

    @property
    def alias_set(self) -> frozenset[str]:
        names = (self.canonical, *self.aliases)
        return frozenset(
            name.strip().lower() for name in names if name.strip()
        )

    def split_category_title(self, title: str) -> str | None:
        """Return the bare category name when ``title`` is a category page."""
        normalized = title.replace('_', ' ').strip()
        if ':' not in normalized:
            return None
        namespace, _, name = normalized.partition(':')
        if namespace.lower() not in self.alias_set:
            return None
        return name.strip() or None


@dataclass(frozen=True)
class MemberEdge:
    """A direct membership: ``page`` declares a link to ``category``."""

    page: str
    category: str


@dataclass(frozen=True)
class ParentEdge:
    """A category-to-category parent declaration.

    The edge points ``category -> parent`` and was found on ``source_page``,
    the category page that contains the declaration.
    """

    category: str
    parent: str
    source_page: str


@dataclass(frozen=True)
class EvidenceHop:
    """One hop of an evidence chain.

    ``source_page`` is the page containing the declaration and ``category``
    is the category the hop arrives at.
    """

    source_page: str
    category: str


@dataclass(frozen=True)
class AncestorConclusion:
    """The conclusion that ``page`` belongs to ``ancestor``.

    Every distinct declaration chain reaching the ancestor is retained in
    ``chains``; the membership itself is reported once even when several
    paths reach it.
    """

    page: str
    ancestor: str
    chains: tuple[tuple[EvidenceHop, ...], ...]


@dataclass(frozen=True)
class CategoryGraph:
    member_edges: tuple[MemberEdge, ...]
    parent_edges: tuple[ParentEdge, ...]
    missing_categories: frozenset[str]
    cyclic_groups: tuple[frozenset[str], ...]
    _members: Mapping[str, frozenset[str]] = field(repr=False)
    _parents: Mapping[str, tuple[ParentEdge, ...]] = field(repr=False)
    _conclusions: Mapping[str, tuple[AncestorConclusion, ...]] = field(
        repr=False
    )

    def members_of(self, category: str) -> frozenset[str]:
        """Pages directly declaring membership in ``category``."""
        return self._members.get(category, frozenset())

    def parents_of(self, category: str) -> tuple[ParentEdge, ...]:
        """Parent edges declared by the category's own page."""
        return self._parents.get(category, ())

    def ancestors_of(self, page: str) -> tuple[AncestorConclusion, ...]:
        """All ancestor conclusions for ``page``, each with source evidence."""
        return self._conclusions.get(page.replace('_', ' ').strip(), ())

    def is_cyclic(self) -> bool:
        return bool(self.cyclic_groups)


def _normalize_title(title: str) -> str:
    return title.replace('_', ' ').strip()


def build_graph(
    pages: Mapping[str, str],
    namespace: NamespaceDeclaration,
) -> CategoryGraph:
    """Build a category graph from a batch of page sources.

    ``pages`` maps full page titles to their wikitext source. Category pages
    are recognized through ``namespace``; a referenced category whose page is
    absent from the batch is reported in ``missing_categories`` and simply
    has no known outgoing parent edges.
    """
    aliases = namespace.alias_set
    member_edges: set[MemberEdge] = set()
    parent_edges: set[ParentEdge] = set()
    members: dict[str, set[str]] = {}
    parents: dict[str, list[ParentEdge]] = {}
    direct_categories: dict[str, list[str]] = {}
    known_category_pages: set[str] = set()
    referenced_categories: set[str] = set()

    normalized_pages = {
        _normalize_title(title): source for title, source in pages.items()
    }

    for page, source in normalized_pages.items():
        bare_name = namespace.split_category_title(page)
        if bare_name is not None:
            known_category_pages.add(bare_name)
        references: list[CategoryReference] = extract_category_references(
            source, aliases
        )
        page_direct: list[str] = []
        for reference in references:
            category = reference.target
            referenced_categories.add(category)
            page_direct.append(category)
            member_edges.add(MemberEdge(page, category))
            members.setdefault(category, set()).add(page)
            if bare_name is not None:
                edge = ParentEdge(bare_name, category, page)
                if edge not in parent_edges:
                    parent_edges.add(edge)
                    parents.setdefault(bare_name, []).append(edge)
        direct_categories[page] = page_direct

    missing = frozenset(referenced_categories - known_category_pages)
    cyclic = _find_cyclic_groups(parents)
    conclusions = {
        page: _ancestors_of(page, tuple(categories), parents)
        for page, categories in direct_categories.items()
    }

    return CategoryGraph(
        member_edges=tuple(
            sorted(member_edges, key=lambda edge: (edge.category, edge.page))
        ),
        parent_edges=tuple(
            sorted(
                parent_edges,
                key=lambda edge: (edge.category, edge.parent),
            )
        ),
        missing_categories=missing,
        cyclic_groups=tuple(sorted(cyclic, key=lambda group: sorted(group))),
        _members={
            category: frozenset(entries)
            for category, entries in members.items()
        },
        _parents={
            category: tuple(edges) for category, edges in parents.items()
        },
        _conclusions=conclusions,
    )


def _find_cyclic_groups(
    parents: Mapping[str, list[ParentEdge]],
) -> list[frozenset[str]]:
    """Find strongly connected components containing an internal edge.

    Edges are never removed; the groups only describe where the parent graph
    deviates from a tree. Parent targets without their own page are treated as
    leaf nodes, because their outgoing edges are unknown rather than absent.
    """
    index_counter = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    groups: list[frozenset[str]] = []

    def strong_connect(node: str) -> None:
        nonlocal index_counter
        indices[node] = lowlinks[node] = index_counter
        index_counter += 1
        stack.append(node)
        on_stack.add(node)
        for edge in parents.get(node, ()):
            successor = edge.parent
            if successor not in indices:
                strong_connect(successor)
                lowlinks[node] = min(lowlinks[node], lowlinks[successor])
            elif successor in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[successor])
        if lowlinks[node] == indices[node]:
            component: list[str] = []
            while True:
                successor = stack.pop()
                on_stack.remove(successor)
                component.append(successor)
                if successor == node:
                    break
            group = frozenset(component)
            if len(group) > 1 or any(
                edge.parent == node for edge in parents.get(node, ())
            ):
                groups.append(group)

    for node in sorted(parents):
        if node not in indices:
            strong_connect(node)
    return groups


def _ancestors_of(
    page: str,
    direct: tuple[str, ...],
    parents: Mapping[str, list[ParentEdge]],
) -> tuple[AncestorConclusion, ...]:
    """Collect reachable categories, retaining every evidence chain.

    A chain never repeats a category, so cycles are traversed safely without
    deleting any edge. Each ancestor yields one conclusion even when reached
    through several paths.
    """
    chains_by_ancestor: dict[str, set[tuple[EvidenceHop, ...]]] = {}

    def walk(
        category: str,
        chain: tuple[EvidenceHop, ...],
        chain_categories: frozenset[str],
    ) -> None:
        chains_by_ancestor.setdefault(category, set()).add(chain)
        for edge in parents.get(category, ()):
            if edge.parent in chain_categories:
                continue
            hop = EvidenceHop(edge.source_page, edge.parent)
            walk(
                edge.parent,
                (*chain, hop),
                chain_categories | {edge.parent},
            )

    for category in dict.fromkeys(direct):
        hop = EvidenceHop(page, category)
        walk(category, (hop,), frozenset({category}))

    return tuple(
        AncestorConclusion(
            page=page,
            ancestor=ancestor,
            chains=tuple(
                sorted(chains, key=_chain_key),
            ),
        )
        for ancestor, chains in sorted(chains_by_ancestor.items())
    )


def _chain_key(chain: tuple[EvidenceHop, ...]) -> tuple[tuple[str, str], ...]:
    return tuple((hop.source_page, hop.category) for hop in chain)
