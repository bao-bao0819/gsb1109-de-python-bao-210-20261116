"""Extract category references from individual page sources."""

from __future__ import annotations

from dataclasses import dataclass

from wikitextparser import parse


@dataclass(frozen=True)
class CategoryReference:
    """A category link declared directly by a page.

    ``target`` is the normalized category name without the namespace prefix
    or a sort key. ``raw_target`` keeps the page's exact spelling for
    diagnostics.
    """

    target: str
    raw_target: str


def _contains(outer: tuple[int, int], inner: tuple[int, int]) -> bool:
    return outer[0] <= inner[0] and inner[1] <= outer[1]


def _split_namespace(title: str, aliases: frozenset[str]) -> str | None:
    """Return the category name when ``title`` uses a category alias."""
    if ':' not in title:
        return None
    namespace, _, name = title.partition(':')
    if namespace.lower() not in aliases:
        return None
    return name.strip()


def extract_category_references(
    source: str, category_aliases: frozenset[str]
) -> list[CategoryReference]:
    """Return the category references declared directly on one page.

    Excluded on purpose:

    * links inside protected text (``nowiki``/``pre`` and HTML comments,
      already dropped by the parser),
    * links inside templates, parser functions and other wikilinks, whose
      rendered membership is not knowable from this page's source alone,
    * leading-colon links such as ``[[:Category:X]]``, which are ordinary
      links instead of membership declarations.
    """
    parsed = parse(source)
    containers: list[tuple[int, int]] = [
        node.span for node in parsed.templates
    ]
    containers.extend(node.span for node in parsed.parser_functions)
    containers.extend(node.span for node in parsed.wikilinks)

    references: list[CategoryReference] = []
    seen: set[str] = set()
    for link in parsed.wikilinks:
        span = link.span
        if any(
            container != span and _contains(container, span)
            for container in containers
        ):
            continue
        if link.target.startswith(':'):
            continue
        normalized = link.title.replace('_', ' ').strip()
        name = _split_namespace(normalized, category_aliases)
        if not name or name in seen:
            continue
        seen.add(name)
        references.append(CategoryReference(name, link.target))
    return references
