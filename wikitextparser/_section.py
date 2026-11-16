from __future__ import annotations

from dataclasses import dataclass, field
from html import unescape
from re import DOTALL, compile as re_compile
from urllib.parse import unquote

from regex import Match

from ._wikitext import SubWikiText, WikiText, rc

HEADER_MATCH = rc(rb'\0*+(={1,6})([^\r\n]+?)\1[ \t\0]*+(\R|\Z)').match

_COMMENT_RE = re_compile(r'<!--.*?-->', DOTALL)
_TAG_RE = re_compile(r'<[^>]*>')
_WIKILINK_RE = re_compile(r'\[\[(?:[^|\]]*\|)?([^\]]*)\]\]')
_WHITESPACE_RE = re_compile(r'\s')


def normalize_section_title(title: str) -> str:
    """Return the MediaWiki-like anchor base for a heading.

    Comments and tags are removed, character references are decoded,
    wikilinks are replaced by their display text, bold/italic markup is
    stripped, and whitespace is converted to underscores.  Templates and
    parser functions cannot be expanded offline, so their markup is kept.
    """
    text = _COMMENT_RE.sub('', title)
    text = _TAG_RE.sub('', text)
    text = _WIKILINK_RE.sub(r'\1', text)
    text = text.replace("'''", '').replace("''", '')
    text = unescape(text).replace('\xa0', ' ')
    return _WHITESPACE_RE.sub('_', text.strip())


def normalize_section_anchor(fragment: str) -> str:
    """Normalize a link fragment so it can be compared with an anchor."""
    return normalize_section_title(unquote(fragment))


@dataclass
class SectionHeadingInfo:
    """Heading metadata for one revision of a page.

    ``ordinal`` is the numeric suffix used for the generated anchor (1 for
    the first occurrence) and corresponds to MediaWiki's ``_2``, ``_3``, ...
    anchor suffixes.  ``sibling_ordinal`` is the occurrence among same-titled
    siblings only.
    """

    section: Section
    index: int
    title: str | None
    base_anchor: str | None
    anchor: str | None
    ordinal: int
    sibling_ordinal: int
    level: int
    heading_index: int = 0
    parent: SectionHeadingInfo | None = field(default=None, repr=False)
    children: list[SectionHeadingInfo] = field(default_factory=list)

    @property
    def path(self) -> tuple[str, ...]:
        """Return the base anchors from the top-level heading to this one."""
        path: list[str] = []
        info: SectionHeadingInfo | None = self
        while info is not None:
            if info.base_anchor:
                path.append(info.base_anchor)
            info = info.parent
        path.reverse()
        return tuple(path)

    @property
    def parent_path(self) -> tuple[str, ...]:
        """Return the path of the parent heading, if any."""
        return self.parent.path if self.parent is not None else ()

    @property
    def parent_base_anchor(self) -> str | None:
        """Return the parent's base anchor, if any."""
        return self.parent.base_anchor if self.parent is not None else None


def section_heading_infos(wikitext) -> list[SectionHeadingInfo]:
    """Return heading metadata for *wikitext*, in document order.

    The lead section is included as the first item with ``anchor=None`` and
    ``level=0``.  Subsection bodies are not included in a heading's own
    ``section.contents`` because ``include_subsections=False`` is used.
    """
    sections = wikitext.get_sections(include_subsections=False)
    counts: dict[str, int] = {}
    used_anchors: set[str] = set()
    infos: list[SectionHeadingInfo] = []
    stack: list[SectionHeadingInfo] = []
    for index, section in enumerate(sections):
        level = section.level
        if level == 0:
            infos.append(
                SectionHeadingInfo(section, index, None, None, None, 0, 0, 0)
            )
            stack.clear()
            continue
        title = section.title
        base = normalize_section_title(title or '')
        ordinal = counts.get(base, 0) + 1
        counts[base] = ordinal
        anchor = base if ordinal == 1 else f'{base}_{ordinal}'
        suffix = ordinal
        while anchor in used_anchors:
            suffix += 1
            anchor = f'{base}_{suffix}'
        used_anchors.add(anchor)
        ordinal = suffix
        while stack and stack[-1].level >= level:
            stack.pop()
        parent = stack[-1] if stack else None
        siblings = (
            parent.children
            if parent is not None
            else [i for i in infos if i.level and i.parent is None]
        )
        sibling_ordinal = (
            sum(1 for sibling in siblings if sibling.base_anchor == base) + 1
        )
        info = SectionHeadingInfo(
            section,
            index,
            title,
            base,
            anchor,
            ordinal,
            sibling_ordinal,
            level,
            len(siblings),
            parent,
        )
        if parent is not None:
            parent.children.append(info)
        infos.append(info)
        stack.append(info)
    return infos


def _section_for_info(
    section: Section, info: SectionHeadingInfo | None
) -> Section | None:
    """Return a Section bound to *section*'s parse tree, if possible."""
    if info is None:
        return None
    lststr = section._lststr
    type_to_spans = section._type_to_spans
    start = info.section.span[0]
    for span in type_to_spans.get('Section', ()):
        if span.start != start:
            continue
        candidate = Section(lststr, type_to_spans, span, 'Section')
        if candidate.level == info.level:
            return candidate
    return info.section


def _section_heading_info(section: Section) -> SectionHeadingInfo | None:
    """Return the heading metadata belonging to *section*."""
    infos = section_heading_infos(WikiText(section._lststr[0]))
    start = section.span[0]
    level = section.level
    for info in infos:
        if (
            info.level
            and info.level == level
            and info.section.span[0] == start
        ):
            return info
    return None


class Section(SubWikiText):
    __slots__ = ('_header_match_cache',)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._header_match_cache: (
            tuple[Match, bytearray] | tuple[None, None]
        ) = None, None

    @property
    def _header_match(self):
        cached_match, cached_shadow = self._header_match_cache
        shadow = self._shadow
        if cached_shadow == shadow:
            return cached_match
        m = HEADER_MATCH(shadow)
        self._header_match_cache = m, shadow
        return m

    @property
    def level(self) -> int:
        """The level of this section.

        getter: Return level which as an int in range(1,7) or 0 for the lead
            section.
        setter: Change the level.
        """
        m = self._header_match
        if m:
            return len(m[1])
        return 0

    @level.setter
    def level(self, value: int) -> None:
        m: Match[bytes] = self._header_match  # type: ignore
        level_diff = len(m[1]) - value
        if level_diff == 0:
            return
        if level_diff < 0:
            new_equals = '=' * -level_diff
            self.insert(0, new_equals)
            self.insert(m.end(2) - level_diff, new_equals)
            return
        del self[:level_diff]
        del self[m.end(2) : m.end(2) + level_diff]

    @property
    def title(self) -> str | None:
        """The title of this section.

        getter: Return the title or None for lead sections or sections that
           don't have any title.
        setter: Set a new title.
        deleter: Remove the title, including the equal sign and the newline
           after it.
        """
        m = self._header_match
        if m is None:
            return None
        return self(m.start(2), m.end(2))

    @title.setter
    def title(self, value: str) -> None:
        m = self._header_match
        if m is None:
            raise RuntimeError(
                "Can't set title for a lead section. "
                'Try adding it to contents.'
            )
        self[m.start(2) : m.end(2)] = value

    @title.deleter
    def title(self) -> None:
        m = self._header_match
        if m is None:
            return
        del self[m.start() : m.end()]

    @property
    def contents(self) -> str:
        """Contents of this section.

        getter: return the contents
        setter: Set contents to a new string value.
        """
        m = self._header_match
        if m is None:
            return self(0, None)
        return self(m.end(), None)

    @contents.setter
    def contents(self, value: str) -> None:
        m = self._header_match
        if m is None:
            self[:] = value
            return
        self[m.end() :] = value

    @property
    def base_anchor(self) -> str | None:
        """The normalized heading anchor, without duplicate suffixes.

        Return ``None`` for a lead section.
        """
        if self.level == 0:
            return None
        title = self.title
        if title is None:
            return None
        return normalize_section_title(title)

    @property
    def anchor(self) -> str | None:
        """The unique section anchor used by in-page links.

        Duplicate headings receive ``_2``, ``_3``, ... in document order,
        matching MediaWiki's generated section anchors.  If ``沿革_2`` is
        already a literal heading, the second ``沿革`` gets ``沿革_3`` so
        every anchor stays unique.  Return ``None`` for a lead section.
        """
        info = _section_heading_info(self)
        if info is not None:
            return info.anchor
        return self.base_anchor

    @property
    def anchor_ordinal(self) -> int:
        """The numeric suffix used in :attr:`anchor` (1 for the first)."""
        info = _section_heading_info(self)
        return info.ordinal if info is not None else 1

    @property
    def parent_section(self) -> Section | None:
        """Return the nearest enclosing heading, or ``None``."""
        info = _section_heading_info(self)
        if info is None or info.parent is None:
            return None
        return _section_for_info(self, info.parent)

    @property
    def subsections(self) -> list[Section]:
        """Return the direct child headings of this section."""
        info = _section_heading_info(self)
        if info is None:
            return []
        return [_section_for_info(self, child) for child in info.children]

    @property
    def section_path(self) -> tuple[str, ...]:
        """Return normalized base anchors from the root heading to this one."""
        info = _section_heading_info(self)
        return info.path if info is not None else ()

    @property
    def heading_index(self) -> int:
        """Return the 0-based position among headings with the same parent."""
        info = _section_heading_info(self)
        return info.heading_index if info is not None else 0
