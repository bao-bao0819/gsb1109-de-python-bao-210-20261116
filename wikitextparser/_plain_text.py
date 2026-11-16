"""Plain-text results with a mapping back to the wikitext source."""

from __future__ import annotations

import re
from dataclasses import dataclass
from html import entities, unescape
from typing import Iterable, Mapping

# This is the character-reference pattern used by ``html.unescape``.  Keeping
# the same pattern makes the mapped and unmapped results agree on ambiguous
# references such as ``&notit;`` (``html.unescape`` turns that into ``¬it;``).
CHARREF = re.compile(
    r'&(#[0-9]+;?|#[xX][0-9a-fA-F]+;?|[^\t\n\f <&#;]{1,32};?)'
)

Span = tuple[int, int]
Spans = tuple[Span, ...]


def merge_spans(spans: Iterable[Span]) -> list[Span]:
    """Merge overlapping or directly adjacent source spans.

    A gap in the merged result means that the source contains characters
    that do not produce any plain text (for example a hidden template).
    Merging across such a gap would create a continuous range that includes
    unrelated source content.
    """
    ordered = sorted(spans)
    if not ordered:
        return []
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        last = merged[-1]
        if start <= last[1]:
            if end > last[1]:
                last[1] = end
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


@dataclass(frozen=True)
class PlainTextChar:
    """One character of plain text and the source spans it came from.

    ``char`` is a single Unicode code point.  ``spans`` normally contains a
    single half-open source range, but may contain several ranges when the
    character is produced by a source region that is interrupted by
    markup which itself produces no text.  A character written as an HTML
    entity (for example ``&Sigma;`` or ``&#931;``) maps to the whole entity
    token, not just to the ``&``.
    """

    char: str
    spans: Spans

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            'spans',
            tuple((int(start), int(end)) for start, end in self.spans),
        )

    @property
    def start(self) -> int:
        """Start of the first source span."""
        return self.spans[0][0]

    @property
    def end(self) -> int:
        """End of the last source span."""
        return self.spans[-1][1]

    @property
    def source_start(self) -> int:
        """Alias for :attr:`start`."""
        return self.start

    @property
    def source_end(self) -> int:
        """Alias for :attr:`end`."""
        return self.end

    @property
    def text(self) -> str:
        """Alias for :attr:`char`."""
        return self.char


class PlainText(str):
    """A plain-text string that knows where each character came from.

    It is a normal :class:`str` in every other respect, so existing code
    that compares it with, slices, or searches the result keeps working.
    Offsets are relative to the start of the root wikitext, the same
    convention as :attr:`WikiText.span`.
    """

    __slots__ = (
        '_chars',
        '_origins',
        '_source_end',
        '_source_start',
        '_specials',
    )

    _chars: tuple[PlainTextChar, ...] | None

    def __new__(
        cls,
        text: str,
        origins: list[int],
        specials: Mapping[int, Spans],
        source_start: int,
        source_end: int,
    ) -> PlainText:
        self = super().__new__(cls, text)
        self._origins = tuple(origins)
        self._specials = dict(specials)
        self._chars = None
        self._source_start = source_start
        self._source_end = source_end
        return self

    def __reduce__(self):
        return (
            type(self),
            (
                str(self),
                list(self._origins),
                dict(self._specials),
                self._source_start,
                self._source_end,
            ),
        )

    def __copy__(self) -> PlainText:
        return self

    def __deepcopy__(self, memo) -> PlainText:
        return self

    def _spans_at(self, index: int) -> Spans:
        special = self._specials.get(index)
        if special is not None:
            return special
        start = self._origins[index]
        return ((start, start + 1),)

    @property
    def chars(self) -> tuple[PlainTextChar, ...]:
        """Characters of the result in display order."""
        chars = self._chars
        if chars is None:
            chars = self._chars = tuple(
                PlainTextChar(char, self._spans_at(index))
                for index, char in enumerate(self)
            )
        return chars

    @property
    def char_spans(self) -> tuple[Span, ...]:
        """The first and last source offsets of each character."""
        return tuple(
            (spans[0][0], spans[-1][1])
            for spans in map(self._spans_at, range(len(self)))
        )

    @property
    def offsets(self) -> tuple[Span, ...]:
        """Alias for :attr:`char_spans`."""
        return self.char_spans

    @property
    def spans(self) -> tuple[Span, ...]:
        """Alias for :attr:`char_spans`."""
        return self.char_spans

    @property
    def range_map(self) -> tuple[tuple[str, Spans], ...]:
        """Pairs of each visible character and all of its source spans."""
        return tuple(
            (char, spans)
            for char, spans in (
                (char, self._spans_at(index))
                for index, char in enumerate(self)
            )
        )

    @property
    def source_start(self) -> int:
        """Start offset of the source region that was extracted."""
        return self._source_start

    @property
    def source_end(self) -> int:
        """End offset of the source region that was extracted."""
        return self._source_end

    def source_spans(
        self, start: int = 0, end: int | None = None
    ) -> list[Span]:
        """Map a half-open plain-text range to source ranges.

        ``start`` and ``end`` are indexes into this string, following normal
        Python slicing rules including negative indexes.  The returned
        half-open ranges are relative to the root wikitext, in ascending
        order, and do not overlap.

        Only source characters that produce text in the selected range are
        returned.  If hidden markup lies between two selected characters,
        the result contains separate ranges for them instead of one range
        that would include the hidden markup.

        An empty selection returns a single zero-width insertion point:
        the start of the next character, or :attr:`source_end` when the
        selection is at the end of the string.
        """
        length = len(self)
        if end is None:
            end = length
        if start < 0:
            start += length
        if end < 0:
            end += length
        if start < 0 or end < 0 or start > length or end > length:
            raise IndexError('plain-text range out of bounds')
        if start > end:
            raise IndexError('start is after end')

        if start == end:
            if start < length:
                point = self._spans_at(start)[0][0]
            else:
                point = self._source_end
            return [(point, point)]

        return merge_spans(
            span
            for index in range(start, end)
            for span in self._spans_at(index)
        )

    def raw_spans(
        self, start: int = 0, end: int | None = None
    ) -> list[Span]:
        """Alias for :meth:`source_spans`."""
        return self.source_spans(start, end)


def _expand_match(
    raw: str, spans: Spans
) -> list[tuple[str, Spans]] | None:
    """Expand one character reference, preserving its source spans.

    Return ``None`` when *raw* is not actually a character reference; the
    original characters and their individual spans should then be kept.
    """
    name = raw[1:]
    if name[0] == '#':
        replacement = unescape(raw)
        token_spans = tuple(merge_spans(spans))
        return [(char, token_spans) for char in replacement]

    html5 = entities.html5
    if name in html5:
        token_spans = tuple(merge_spans(spans))
        return [(char, token_spans) for char in html5[name]]

    # Match the longest named reference, exactly as html.unescape does.
    for size in range(len(name) - 1, 1, -1):
        prefix = name[:size]
        if prefix not in html5:
            continue
        prefix_end = 1 + size  # include the leading '&'
        prefix_spans = tuple(merge_spans(spans[:prefix_end]))
        result = [(char, prefix_spans) for char in html5[prefix]]
        # The rest of the match is literal text and keeps its own mapping.
        result.extend(
            (raw[index], (spans[index],))
            for index in range(prefix_end, len(raw))
        )
        return result
    return None


def expand_html_entities(
    text: str,
    origins: list[int],
    specials: Mapping[int, Spans],
) -> tuple[str, list[int], dict[int, Spans]]:
    """Replace HTML entities while retaining source spans.

    The behavior matches :func:`html.unescape`, but an expanded character
    maps back to the full source token rather than only to the ``&`` or to
    a single character of it.
    """
    if '&' not in text:
        return text, origins, dict(specials)

    def spans_at(index: int) -> Spans:
        special = specials.get(index)
        if special is not None:
            return special
        start = origins[index]
        return ((start, start + 1),)

    pieces: list[str] = []
    new_origins: list[int] = []
    new_specials: dict[int, Spans] = {}

    def emit_plain(start: int, end: int) -> None:
        shift = len(new_origins) - start
        new_origins.extend(origins[start:end])
        pieces.append(text[start:end])
        for index in range(start, end):
            special = specials.get(index)
            if special is not None:
                new_specials[index + shift] = special

    def emit(char: str, spans: Spans) -> None:
        index = len(new_origins)
        new_origins.append(spans[0][0])
        if len(spans) != 1 or spans[0][1] != spans[0][0] + 1:
            new_specials[index] = spans
        pieces.append(char)

    position = 0
    for match in CHARREF.finditer(text):
        start, end = match.span()
        emit_plain(position, start)
        token_spans = tuple(
            span for index in range(start, end) for span in spans_at(index)
        )
        expanded = _expand_match(match[0], token_spans)
        if expanded is None:
            # Not a character reference: keep each character's own mapping.
            for offset, char in enumerate(match[0]):
                emit(char, spans_at(start + offset))
        else:
            for char, spans in expanded:
                emit(char, spans)
        position = end

    emit_plain(position, len(text))
    return ''.join(pieces), new_origins, new_specials
