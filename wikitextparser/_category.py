"""Collect category declarations per page view.

``<noinclude>``, ``<includeonly>`` and ``<onlyinclude>`` make the wikitext
of a page differ from the text that is emitted when the page is transcluded.
The helpers in this module report the category links (``[[Category:...]]``)
that belong to each of the two views without ever expanding templates, so a
category produced by a template is never mistaken for a category written
directly on the page.

Categories hidden inside a comment (``<!-- ... -->``) or a protected
extension tag (e.g. ``<nowiki>``, ``<pre>``, ``<source>``) are not real
declarations and are ignored in both views.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from regex import IGNORECASE, compile as rc

from ._tag import Tag
from ._wikilink import WikiLink
from ._wikitext import WikiText

# Names of the extension tags that control the two views.
_TRANSCLUSION_TAGS = frozenset(('noinclude', 'includeonly', 'onlyinclude'))
# Extension tags whose contents are not parsed as wikitext. Category-like
# text inside them is protected text rather than a declaration.
_UNPARSABLE_TAGS = frozenset(
    (
        'ce',
        'charinsert',
        'chem',
        'graph',
        'hiero',
        'languages',
        'mapframe',
        'maplink',
        'math',
        'nowiki',
        'pagelist',
        'pagequality',
        'pages',
        'pre',
        'score',
        'source',
        'syntaxhighlight',
        'templatedata',
        'templatestyles',
        'timeline',
    )
)

# A control tag start, end, or self-closing tag. Group ``name`` captures the
# tag name and group ``slash`` is non-empty for self-closing tags.
_CONTROL_TAG_FINDITER = rc(
    rb'</?(?P<name>noinclude|includeonly|onlyinclude)\b'
    rb'[^>]*?(?P<slash>/?)>',
    IGNORECASE,
).finditer

# Category namespace aliases. English ``Category`` is always recognized;
# additional aliases can be supplied by the caller for localized wikis.
_DEFAULT_CATEGORY_NAMESPACES = frozenset(('category',))

# A link nested inside one of these constructs is produced by transclusion
# syntax rather than written directly on the page, so it must not be reported
# as a declaration of this page (templates are deliberately not expanded).
_BRACE_TYPES = ('Template', 'ParserFunction', 'Parameter')


@dataclass(frozen=True)
class TransclusionRegion:
    """A transclusion-control region affecting a piece of wikitext.

    :ivar name: one of ``'noinclude'``, ``'includeonly'`` or
        ``'onlyinclude'``.
    :ivar start: start offset of the affected content (just past the opening
        tag), relative to the parsed page.
    :ivar end: end offset of the affected content (at the closing tag, or at
        the end of the page for an unclosed region).
    :ivar complete: ``True`` when a matching closing tag was found. An
        unclosed or mismatched region is kept with ``complete=False`` so the
        affected range stays visible instead of being silently dropped.
    :ivar tag: the parsed :class:`Tag` backing a complete region, or ``None``
        for an incomplete region that has no well-formed tag.
    """

    name: str
    start: int
    end: int
    complete: bool
    tag: Tag | None

    @property
    def span(self) -> tuple[int, int]:
        """Return the ``(start, end)`` offsets of the affected content."""
        return self.start, self.end


@dataclass(frozen=True)
class CategoryDeclaration:
    """A single ``[[Category:…]]`` declaration and its source.

    :ivar link: the parsed wikilink of the declaration.
    :ivar region: the innermost transclusion-control region containing the
        link, or ``None`` for ordinary page text.
    """

    link: WikiLink
    region: TransclusionRegion | None


@dataclass(frozen=True)
class CategoryReport:
    """The category declarations of a page, split by view.

    A declaration that belongs to both views appears in both lists. The lists
    keep document order. ``regions`` lists every transclusion-control region
    found on the page (including nested, empty or incomplete ones) so the
    affected range is always available for inspection.
    """

    page: list[CategoryDeclaration]
    transclusion: list[CategoryDeclaration]
    regions: list[TransclusionRegion]


def _protected_ranges(
    comments: list, tags: list[Tag]
) -> list[tuple[int, int]]:
    """Return spans that must never be treated as live wikitext.

    Comments and unparsable extension tags (``<nowiki>`` etc.) hide their
    contents from the parser, so any category-looking text or control tag
    inside them is ignored.
    """
    ranges: list[tuple[int, int]] = [comment.span for comment in comments]
    for tag in tags:
        if tag.name.lower() in _UNPARSABLE_TAGS:
            ranges.append(tag.span)
    ranges.sort()
    return ranges


def _is_masked(start: int, end: int, masked: list[tuple[int, int]]) -> bool:
    for mask_start, mask_end in masked:
        if end <= mask_start:
            break
        if start < mask_end and end > mask_start:
            return True
    return False


def _control_regions(
    text: str, masked: list[tuple[int, int]]
) -> list[TransclusionRegion]:
    """Determine the extent of every transclusion-control region.

    Tags nested inside comments or protected text are ignored. Unclosed or
    mismatched tags are kept (reported as incomplete, reaching the end of the
    text) instead of being silently dropped, so the affected range stays
    visible to the caller.
    """
    byte_array = bytearray(text, 'ascii', 'replace')
    # name, content start, tag start
    stack: list[tuple[str, int, int]] = []
    partial: list[tuple[str, int, int, bool]] = []
    for match in _CONTROL_TAG_FINDITER(byte_array):
        start, end = match.span()
        if _is_masked(start, end, masked):
            continue
        name = match.group('name').decode().lower()
        closing = byte_array[start + 1] == 47  # ord('/')
        if closing:
            for depth in range(len(stack) - 1, -1, -1):
                open_name, content_start, _tag_start = stack[depth]
                if open_name != name:
                    continue
                # Inner tags without a matching close are incomplete and run
                # up to this closing tag.
                for inner in reversed(stack[depth + 1 :]):
                    partial.append((inner[0], inner[1], start, False))
                partial.append((name, content_start, start, True))
                del stack[depth:]
                break
            # A stray end tag without a matching start tag is ignored.
            continue
        if match.group('slash'):
            # A self-closing control tag encloses no content.
            continue
        stack.append((name, end, start))
    text_length = len(text)
    for name, content_start, _tag_start in reversed(stack):
        partial.append((name, content_start, text_length, False))
    partial.sort(key=lambda item: (item[1], item[2]))
    return [
        TransclusionRegion(name, start, end, complete, None)
        for name, start, end, complete in partial
    ]


def _attach_region_tags(
    tags: list[Tag], regions: list[TransclusionRegion]
) -> list[TransclusionRegion]:
    """Pair complete regions with their parsed extension tags."""
    control_tags = [
        tag for tag in tags if tag.name.lower() in _TRANSCLUSION_TAGS
    ]
    resolved: list[TransclusionRegion] = []
    for region in regions:
        tag: Tag | None = None
        if region.complete:
            best_span: tuple[int, int] | None = None
            for candidate in control_tags:
                if candidate.name.lower() != region.name:
                    continue
                candidate_start, candidate_end = candidate.span
                if (
                    candidate_start <= region.start
                    and region.end <= candidate_end
                ):
                    if best_span is None or (
                        candidate_end - candidate_start
                        < best_span[1] - best_span[0]
                    ):
                        tag = candidate
                        best_span = candidate.span
        resolved.append(
            TransclusionRegion(
                region.name,
                region.start,
                region.end,
                region.complete,
                tag,
            )
        )
    return resolved


def _view_masks(
    length: int, regions: list[TransclusionRegion]
) -> tuple[bytearray, bytearray]:
    """Build page-view and transclusion-view inclusion masks."""
    # The page view shows everything except the contents of ``includeonly``.
    page_mask = bytearray(b'\x01') * length
    # When the page contains at least one ``onlyinclude`` region, transclusion
    # emits only those regions; otherwise the whole page is eligible.
    onlyinclude = [r for r in regions if r.name == 'onlyinclude']
    transclusion_mask = (
        bytearray(b'\x00') * length
        if onlyinclude
        else bytearray(b'\x01') * length
    )
    for region in regions:
        size = region.end - region.start
        if region.name == 'includeonly':
            page_mask[region.start : region.end] = b'\x00' * size
        elif region.name == 'onlyinclude':
            transclusion_mask[region.start : region.end] = b'\x01' * size
        else:  # noinclude
            # ``noinclude`` wins over any enclosing ``onlyinclude``.
            transclusion_mask[region.start : region.end] = b'\x00' * size
    return page_mask, transclusion_mask


def _innermost_region(
    start: int, end: int, regions: list[TransclusionRegion]
) -> TransclusionRegion | None:
    innermost: TransclusionRegion | None = None
    for region in regions:
        if region.start > start:
            break
        if region.start <= start and end <= region.end:
            if innermost is None or (
                region.start >= innermost.start and region.end <= innermost.end
            ):
                innermost = region
    return innermost


def _is_category_link(link: WikiLink, namespaces: frozenset[str]) -> bool:
    title = link.title
    # A leading colon (e.g. ``[[:Category:Foo]]``) turns the link into an
    # ordinary link to the category page instead of a category declaration.
    if title.startswith(':'):
        return False
    namespace, separator, _rest = title.partition(':')
    if not separator:
        return False
    return namespace.strip().lower() in namespaces


def _is_transcluded_syntax(link: WikiLink) -> bool:
    """Return True for links nested inside ``{{...}}`` constructs.

    Such links are emitted by the referenced template/parser function and are
    not written directly on the page; expanding the construct would wrongly
    attribute its output to the page.
    """
    return any(
        type(ancestor).__name__ in _BRACE_TYPES
        for ancestor in link.ancestors()
    )


def category_declarations(
    wikitext: WikiText,
    *,
    category_namespaces: Iterable[str] = _DEFAULT_CATEGORY_NAMESPACES,
) -> CategoryReport:
    """Return the category declarations of ``wikitext`` for both views.

    Templates are not expanded: only links written directly in the wikitext
    are considered, so a category emitted by a template is never attributed
    to this page. Categories inside comments or protected extension tags are
    pseudo-categories and are excluded.

    :param wikitext: the parsed page.
    :param category_namespaces: recognized category namespace names, matched
        case-insensitively. Defaults to ``('category',)``; pass localized
        aliases (e.g. ``('category', 'κατηγορία')``) for non-English wikis.
    """
    text = wikitext.string
    length = len(text)
    tags = wikitext.get_tags()
    masked = _protected_ranges(wikitext.comments, tags)
    regions = _attach_region_tags(tags, _control_regions(text, masked))
    page_mask, transclusion_mask = _view_masks(length, regions)
    for mask_start, mask_end in masked:
        page_mask[mask_start:mask_end] = b'\x00' * (mask_end - mask_start)
        transclusion_mask[mask_start:mask_end] = b'\x00' * (
            mask_end - mask_start
        )

    namespaces = frozenset(
        namespace.strip().lower() for namespace in category_namespaces
    )
    page_declarations: list[CategoryDeclaration] = []
    transclusion_declarations: list[CategoryDeclaration] = []
    for link in wikitext.wikilinks:
        start, end = link.span
        if _is_masked(start, end, masked):
            continue
        if _is_transcluded_syntax(link):
            continue
        if not _is_category_link(link, namespaces):
            continue
        in_page = any(page_mask[start:end])
        in_transclusion = any(transclusion_mask[start:end])
        if not in_page and not in_transclusion:
            continue
        declaration = CategoryDeclaration(
            link=link,
            region=_innermost_region(start, end, regions),
        )
        if in_page:
            page_declarations.append(declaration)
        if in_transclusion:
            transclusion_declarations.append(declaration)
    return CategoryReport(
        page=page_declarations,
        transclusion=transclusion_declarations,
        regions=regions,
    )
