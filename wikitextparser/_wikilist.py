from __future__ import annotations

from collections.abc import Iterable, MutableSequence
from operator import attrgetter

from regex import Match, compile as regex_compile, escape, fullmatch

from wikitextparser._spans import SpanData

from ._spans import TypeToSpans, parse_to_spans
from ._wikitext import EXTERNAL_LINK_FINDITER, SubWikiText

# See includes/parser/BlockLevelPass.php for how MW parses list blocks.
SUBLIST_PATTERN = (
    rb'(?>(?<=\R|\A)' rb'(?&pattern)' rb'[:;#*][^\r\n]*+' rb'(?>\R|\Z)' rb')*+'
)
SUBLIST_WITH_SECOND_PATTERN = (
    rb'[*#;:][^\r\n]*+(?>\R|\Z)'
    rb'(?>'
    rb'(?&pattern)[*#;:][^\r\n]*+(?>\R|\Z)'
    rb')*+'
)
LIST_PATTERN_FORMAT = (
    rb'(?<fullitem>(?<=\R|\A)'
    rb'(?<pattern>{pattern})'
    rb'(?>'
    rb'(?(?<=;\s*+)'
    # mark inline definition as an item
    rb'(?<item>[^:\r\n]*+)(?<fullitem>:(?<item>[^\r\n]*+))?+'
    rb'(?>\R|\Z)' + SUBLIST_PATTERN + rb'|'
    # non-definition
    rb'(?>'
    rb'(?<item>)'
    + SUBLIST_WITH_SECOND_PATTERN
    + rb'|(?<item>[^\r\n]*+)(?>\R|\Z)'
    + SUBLIST_PATTERN
    + rb')'
    rb')'
    rb'))++'
)

# A line of a list block: content followed by any kind of line terminator.
_LINE_FINDITER = regex_compile(rb'([^\r\n]*+)(?>\r\n|[\r\n]|\Z)').finditer
_MARKER_BYTES = b'#*;:'


class ListItem:
    """Represent a single entry (item) of a mixed wiki list.

    Do not create instances directly; use `parse_list_items` or the
    ``get_items`` method of `WikiList`.

    All spans are (start, end) tuples of offsets into the source string
    that was passed to the parser (``WikiList.string`` for the
    ``get_items`` method). The `span` of an item only covers its own
    marker line and any continuation lines; sub-items are separate
    `ListItem` objects linked through `parent` and `children`.
    """

    __slots__ = (
        '_source',
        'body_span',
        'children',
        'definition_span',
        'marker',
        'name_span',
        'parent',
        'span',
    )

    def __init__(
        self,
        source: str,
        marker: str,
        span: tuple[int, int],
        body_span: tuple[int, int],
        name_span: tuple[int, int] | None,
        definition_span: tuple[int, int] | None,
    ) -> None:
        self._source = source
        self.marker = marker
        self.span = span
        self.body_span = body_span
        self.name_span = name_span
        self.definition_span = definition_span
        self.parent: ListItem | None = None
        self.children: list[ListItem] = []

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.marker!r}, {self.span!r})'

    @property
    def level(self) -> int:
        """Return the nesting level, i.e. the number of marker characters."""
        return len(self.marker)

    @property
    def body(self) -> str:
        """Return the item text after the marker, incl. continuation lines."""
        s, e = self.body_span
        return self._source[s:e]

    @property
    def name(self) -> str | None:
        """Return the term of a definition item, None for other items."""
        span = self.name_span
        if span is None:
            return None
        s, e = span
        return self._source[s:e]

    @property
    def definition(self) -> str | None:
        """Return the definition of a definition item, None if not found."""
        span = self.definition_span
        if span is None:
            return None
        s, e = span
        return self._source[s:e]


def _parse_list_items(
    shadow: bytes | bytearray, string: str
) -> list[ListItem]:
    """Parse list items from *shadow*, slicing actual text from *string*.

    *shadow* must be a position-preserving shadow of *string* where
    comments, templates, wikilinks and external links are blanked out, so
    that marker characters inside them are not mistaken for list syntax.
    """
    items: list[ListItem] = []
    stack: list[ListItem] = []
    current: ListItem | None = None
    for match in _LINE_FINDITER(shadow):
        content = match[1]
        cs, ce = match.span(1)
        if not content.strip():
            # A blank line ends the current item; a following non-marker
            # line is then an orphan continuation and will be ignored.
            current = None
        elif content[0] in _MARKER_BYTES:
            k = 1
            while k < len(content) and content[k] in _MARKER_BYTES:
                k += 1
            marker = string[cs : cs + k]
            body_span = (cs + k, ce)
            if marker[-1] == ';':
                # definition term, optionally followed by ': definition'
                index = shadow.find(b':', cs + k, ce)
                if index == -1:
                    name_span: tuple[int, int] | None = body_span
                    definition_span: tuple[int, int] | None = None
                else:
                    name_span = (cs + k, index)
                    definition_span = (index + 1, ce)
            elif marker[-1] == ':':
                # definition without a term
                name_span = None
                definition_span = body_span
            else:
                name_span = definition_span = None
            item = ListItem(
                string, marker, (cs, ce), body_span, name_span, definition_span
            )
            # The parent is the nearest previous item whose marker is a
            # proper prefix of this item's marker; a single character
            # change in indentation can therefore re-parent a whole block.
            while stack and not (
                len(stack[-1].marker) < k
                and marker.startswith(stack[-1].marker)
            ):
                stack.pop()
            if stack:
                item.parent = stack[-1]
                stack[-1].children.append(item)
            stack.append(item)
            items.append(item)
            current = item
        elif current is not None:
            # A wrapped description line; extend the preceding item.
            current.span = (current.span[0], ce)
            current.body_span = (current.body_span[0], ce)
            if current.definition_span is not None:
                ds = current.definition_span
                current.definition_span = (ds[0], ce)
            elif current.name_span is not None:
                ns = current.name_span
                current.name_span = (ns[0], ce)
        # else: orphan continuation line; do not make an item out of it.
    return items


def parse_list_items(text: str) -> list[ListItem]:
    r"""Parse *text* as a mixed wiki list and return its items in source order.

    Lines may start with any run of the ``#``, ``*``, ``;`` and ``:``
    markers. The parent of each `ListItem` is derived from these marker
    prefixes, so changing the indentation of one line can re-parent the
    items that follow it. Non-marker lines are treated as wrapped
    continuations of the preceding item; orphan continuation lines (those
    not following an item, e.g. at the start or after a blank line) are
    ignored and do not form new items. For definition items (marker
    ending in ``;`` or ``:``) the name and definition are available
    separately through `ListItem.name` and `ListItem.definition`.

    The returned items are in source order and their spans refer to
    *text*, so the original source can be recovered from them.
    """
    shadow = bytearray(text, 'ascii', 'replace')
    parse_to_spans(shadow)
    for m in EXTERNAL_LINK_FINDITER(shadow):
        s, e = m.span()
        shadow[s:e] = b'_' * (e - s)
    return _parse_list_items(shadow, text)


class WikiList(SubWikiText):
    """Class to represent ordered, unordered, and definition lists."""

    __slots__ = '_match_cache', 'pattern'

    def __init__(
        self,
        string: str | MutableSequence[str],
        pattern: str,
        _match: Match | None = None,
        _type_to_spans: TypeToSpans | None = None,
        _span: SpanData | None = None,
        _type: str | None = None,
    ) -> None:
        super().__init__(string, _type_to_spans, _span, _type)
        self.pattern = pattern
        if _match:
            self._match_cache = _match, self.string
        else:
            self._match_cache = (
                fullmatch(
                    LIST_PATTERN_FORMAT.replace(
                        b'{pattern}', pattern.encode(), 1
                    ),
                    self._list_shadow,
                ),
                self.string,
            )

    @property
    def _list_shadow(self):
        shadow_copy = self._shadow[:]
        if ':' in self.pattern:
            for m in EXTERNAL_LINK_FINDITER(shadow_copy):
                s, e = m.span()
                shadow_copy[s:e] = b'_' * (e - s)
        return shadow_copy

    @property
    def _match(self) -> Match[bytes]:
        """Return the match object for the current list."""
        cache_match, cache_string = self._match_cache
        string = self.string
        if cache_string == string:
            return cache_match  # type: ignore
        cache_match = fullmatch(
            LIST_PATTERN_FORMAT.replace(
                b'{pattern}', self.pattern.encode(), 1
            ),
            self._list_shadow,
        )
        self._match_cache = cache_match, string
        return cache_match  # type: ignore

    @property
    def items(self) -> list[str]:
        """Return items as a list of strings.

        Do not include sub-items and the start pattern.
        """
        items: list[str] = []
        append = items.append
        string = self.string
        match = self._match
        ms = match.start()
        for s, e in match.spans('item'):
            append(string[s - ms : e - ms])
        return items

    @property
    def fullitems(self) -> list[str]:
        """Return list of item strings. Includes their start and sub-items."""
        fullitems: list[str] = []
        append = fullitems.append
        string = self.string
        match = self._match
        ms = match.start()
        # Sort because "fullitem" can be flipped compared to "items" in case
        # of a definition list with the LIST_PATTERN_FORMAT regex.
        for s, e in sorted(match.spans('fullitem')):
            append(string[s - ms : e - ms])
        return fullitems

    @property
    def level(self) -> int:
        """Return level of nesting for the current list.

        Level is a one-based index, for example the level for `* a` will be 1.
        """
        return len(self._match['pattern'])

    def sublists(
        self,
        i: int | None = None,
        pattern: str | Iterable[str] = (r'\#', r'\*', '[:;]'),
    ) -> list[WikiList]:
        """Return the Lists inside the item with the given index.

        :param i: The index of the item which its sub-lists are desired.
        :param pattern: The starting symbol for the desired sub-lists.
            The `pattern` of the current list will be automatically added
            as prefix.
        """
        if isinstance(pattern, str):
            patterns = (pattern,)
        else:
            patterns = pattern
        self_pattern = self.pattern
        get_lists = super().get_lists
        sublists: list[WikiList] = []
        sublists_append = sublists.append
        if i is None:
            # Any sublist is acceptable
            for ptrn in patterns:
                for lst in get_lists(self_pattern + ptrn):
                    sublists_append(lst)
        else:
            # Only return sub-lists that are within the given item
            match = self._match
            fullitem_spans = match.spans('fullitem')
            ss = self._span_data.start
            ms = match.start()
            s, e = fullitem_spans[i]
            e -= ms - ss
            s -= ms - ss
            for ptrn in patterns:
                for lst in get_lists(self_pattern + ptrn):
                    # noinspection PyProtectedMember
                    sd = lst._span_data
                    if s <= sd.start and sd.end <= e:
                        sublists_append(lst)
        sublists.sort(key=attrgetter('_span_data'))
        return sublists

    def get_items(self) -> list[ListItem]:
        """Return a flat list of `ListItem` objects for every entry.

        Unlike `items`, which only returns the top-level item strings,
        this method returns every entry of the (possibly nested and
        mixed) list in source order, each linked to its `parent` and
        exposing its own `span` and `body_span`. All spans are relative
        to the start of this list's string. See `parse_list_items` for
        details on how entries, continuations and definitions are
        handled.
        """
        shadow = self._shadow[:]
        for m in EXTERNAL_LINK_FINDITER(shadow):
            s, e = m.span()
            shadow[s:e] = b'_' * (e - s)
        return _parse_list_items(shadow, self.string)

    def convert(self, newstart: str) -> None:
        """Convert to another list type by replacing starting pattern."""
        match = self._match
        ms = match.start()
        for s, e in reversed(match.spans('pattern')):
            self[s - ms : e - ms] = newstart
        self.pattern = escape(newstart)

    def get_lists(
        self, pattern: str | Iterable[str] = (r'\#', r'\*', '[:;]')
    ) -> list[WikiList]:
        return self.sublists(pattern=pattern)
