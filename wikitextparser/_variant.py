"""Project wikitext onto a language variant without evaluating templates.

The projection is source preserving.  Text outside language conversion markup
is copied verbatim, including templates and parser functions.  Only
``-{ ... }-`` constructs are resolved, so an offline collector can decide
later how to handle transclusions.

Every emitted fragment carries the half-open range of the original source that
produced it and the rule that was applied.  Conversion rules are scoped to
their own construct: a rule never changes text after its closing ``}-``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Iterator

from ._config import (
    _parsable_tag_extensions,
    _unparsable_tag_extensions,
    _valid_html_tag_names,
)

# Variant codes understood by MediaWiki's LanguageConverter implementations.
# Selectors outside this set are unexplained markup rather than valid rules.
KNOWN_VARIANTS = frozenset(
    {
        'ban',
        'ban-bali',
        'ban-x-dharma',
        'ban-x-palmleaf',
        'ban-x-pku',
        'crh',
        'crh-cyrl',
        'crh-latn',
        'en',
        'en-x-piglatin',
        'gan',
        'gan-hans',
        'gan-hant',
        'ike-cans',
        'ike-latn',
        'iu',
        'ku',
        'ku-arab',
        'ku-latn',
        'lzh',
        'lzh-hans',
        'lzh-hant',
        'sh-cyrl',
        'sh-latn',
        'shi',
        'shi-latn',
        'shi-tfng',
        'sr',
        'sr-ec',
        'sr-el',
        'tg',
        'tg-latn',
        'tly',
        'tly-cyrl',
        'uz',
        'uz-cyrl',
        'uz-latn',
        'wuu',
        'wuu-hans',
        'wuu-hant',
        'zh',
        'zh-cn',
        'zh-hans',
        'zh-hant',
        'zh-hk',
        'zh-mo',
        'zh-my',
        'zh-sg',
        'zh-tw',
    }
)

# Chinese regional variants are not independent scripts.  A request for zh-cn
# reads a zh-hans branch when no zh-cn branch exists, and likewise for the
# traditional variants.  The base-language fallback covers the other
# converters (wuu-hans -> wuu, sr-ec -> sr, ...).
VARIANT_FALLBACKS = {
    'zh': ('zh-hans', 'zh-cn', 'zh-sg', 'zh-my'),
    'zh-hans': ('zh-cn', 'zh-sg', 'zh-my', 'zh'),
    'zh-hant': ('zh-tw', 'zh-hk', 'zh-mo', 'zh'),
    'zh-cn': ('zh-hans', 'zh-sg', 'zh-my', 'zh'),
    'zh-sg': ('zh-hans', 'zh-cn', 'zh-my', 'zh'),
    'zh-my': ('zh-hans', 'zh-cn', 'zh-sg', 'zh'),
    'zh-tw': ('zh-hant', 'zh-hk', 'zh-mo', 'zh'),
    'zh-hk': ('zh-hant', 'zh-mo', 'zh-tw', 'zh'),
    'zh-mo': ('zh-hant', 'zh-hk', 'zh-tw', 'zh'),
}

# Tags whose contents MediaWiki does not parse as ordinary wikitext, or which
# are displayed literally.  Conversion markers inside them have no effect.
PROTECTED_TAGS = frozenset(_unparsable_tag_extensions)

_TAG_NAMES = '|'.join(
    sorted(PROTECTED_TAGS, key=len, reverse=True)
)
_PROTECTED_TAG_RE = re.compile(
    r'<(?P<name>' + _TAG_NAMES + r')\b'
    r'(?P<attrs>[^>]*?)'
    r'(?:/>|>(?P<content>.*?)(?:</(?P=name)\s*>|\Z))',
    re.DOTALL | re.IGNORECASE,
)
# MediaWiki only recognizes these tag names.  Unknown ``<foo>`` markup is
# escaped and displayed literally, so it must not shield conversion markers.
_MARKUP_TAG_NAMES = (
    PROTECTED_TAGS | _parsable_tag_extensions | _valid_html_tag_names
)
_HTML_TAG_MARKUP_RE = re.compile(
    r'</?(?:'
    + '|'.join(sorted(_MARKUP_TAG_NAMES, key=len, reverse=True))
    + r')\b(?:[^>"\']|"[^"]*"|\'[^\']*\')*>',
    re.IGNORECASE,
)
_COMMENT_RE = re.compile(r'<!--.*?(?:-->|\Z)', re.DOTALL)
_FLAG_RE = re.compile(r'\A([A-Za-z])\s*\|', re.DOTALL)
_SELECTOR_RE = re.compile(r'\A[a-z][a-z0-9-]*\Z')

# Rule names used in VariantSegment.rule.  They are English identifiers so
# they can be consumed directly by downstream corpus pipelines.
LITERAL = 'literal'
COMMENT = 'comment'
PROTECTED = 'protected'
TEMPLATE = 'template'
VARIANT = 'variant'
VARIANT_FALLBACK = 'variant-fallback'
REPLACE = 'replace'
REPLACE_FALLBACK = 'replace-fallback'
DEFAULT = 'default'
DEFAULT_REPLACE = 'default-replace'
HIDDEN = 'hidden'
TITLE = 'title'
RAW = 'raw'
NO_CONVERT = 'no-convert'
EMPTY = 'empty'
UNKNOWN = 'unexplained'

# Human-readable labels for the rules, useful in reports and audit output.
RULE_DESCRIPTIONS = {
    LITERAL: 'source text outside conversion markup',
    COMMENT: 'HTML comment, never converted',
    PROTECTED: 'literal tag contents, never converted',
    TEMPLATE: 'template or parser function, not evaluated',
    VARIANT: 'selected variant branch',
    VARIANT_FALLBACK: 'branch selected through variant fallback',
    REPLACE: 'selected local replacement target',
    REPLACE_FALLBACK: 'replacement target selected through fallback',
    DEFAULT: 'default branch when no variant matches',
    DEFAULT_REPLACE: 'default replacement target',
    HIDDEN: 'hidden conversion rule, produces no text',
    TITLE: 'display-title rule, produces no text',
    RAW: 'raw rule, contents emitted without conversion',
    NO_CONVERT: 'explicit no-conversion form',
    EMPTY: 'empty conversion construct',
    UNKNOWN: 'unknown form, left as original source',
}




@dataclass(frozen=True)
class VariantSegment:
    """One fragment of a variant projection.

    ``text`` is the projected text.  ``start`` and ``end`` are half-open
    offsets into the original source of the text that produced this fragment.
    For a selected branch they point at the branch text, not at the discarded
    alternatives.  ``construct_start`` and ``construct_end`` describe the whole
    ``-{ ... }-`` construct when the fragment came from one; the delimiters and
    the discarded branches are inside that range.

    ``branch`` names the selected variant selector, if any.  ``tag`` names the
    protected tag for ``rule == 'protected'``.
    """

    text: str
    start: int
    end: int
    rule: str
    branch: str | None = None
    tag: str | None = None
    construct_start: int | None = None
    construct_end: int | None = None

    @property
    def span(self) -> tuple[int, int]:
        """Return the source range of the text as a half-open tuple."""
        return self.start, self.end

    @property
    def range(self) -> tuple[int, int]:
        """Alias for ``span``."""
        return self.start, self.end

    @property
    def text_span(self) -> tuple[int, int]:
        """Alias for ``span``, the source of the emitted text."""
        return self.start, self.end

    @property
    def source_start(self) -> int:
        """Return the start offset of the text that produced the fragment."""
        return self.start

    @property
    def source_end(self) -> int:
        """Return the exclusive end offset of that text."""
        return self.end

    @property
    def construct_span(self) -> tuple[int, int] | None:
        """Return the whole conversion construct's range, if any."""
        if self.construct_start is None or self.construct_end is None:
            return None
        return self.construct_start, self.construct_end

    @property
    def block_span(self) -> tuple[int, int] | None:
        """Alias for ``construct_span``."""
        return self.construct_span

    @property
    def kind(self) -> str:
        """Alias for ``rule``."""
        return self.rule

    @property
    def selected(self) -> str | None:
        """Alias for ``branch``."""
        return self.branch

    @property
    def source(self) -> str:
        """Alias for ``text``."""
        return self.text

    @property
    def explained(self) -> bool:
        """Return False when the construct was left as unexplained source."""
        return self.rule != UNKNOWN

    @property
    def description(self) -> str:
        """Return a human-readable description of the rule."""
        return RULE_DESCRIPTIONS.get(self.rule, self.rule)

    def as_tuple(self) -> tuple[str, int, int, str]:
        """Return ``(text, start, end, rule)``."""
        return self.text, self.start, self.end, self.rule

    def as_dict(self) -> dict:
        """Return a JSON-serializable representation."""
        return {
            'text': self.text,
            'start': self.start,
            'end': self.end,
            'rule': self.rule,
            'branch': self.branch,
            'tag': self.tag,
            'construct_start': self.construct_start,
            'construct_end': self.construct_end,
        }


class VariantProjection(str):
    """Projected text with provenance for every fragment.

    The object is a ``str`` subclass, so it can be passed directly to code
    that expects the projected text.  ``segments`` retains the mapping back to
    the wikitext source.
    """

    def __new__(
        cls,
        text: str,
        *,
        variant: str,
        source: str,
        segments: Iterable[VariantSegment],
        source_start: int = 0,
        source_end: int | None = None,
    ) -> VariantProjection:
        projection = super().__new__(cls, text)
        projection.variant = variant
        projection.source = source
        projection.segments = tuple(segments)
        projection.source_start = source_start
        projection.source_end = (
            len(source) if source_end is None else source_end
        )
        return projection

    @property
    def text(self) -> str:
        """Return the projected text."""
        return str(self)

    @property
    def fragments(self) -> tuple[VariantSegment, ...]:
        """Alias for ``segments``."""
        return self.segments

    @property
    def rules(self) -> tuple[str, ...]:
        """Return the distinct rules in first-use order."""
        return tuple(dict.fromkeys(segment.rule for segment in self.segments))

    @property
    def rule_counts(self) -> dict[str, int]:
        """Return how many fragments each rule produced."""
        counts: dict[str, int] = {}
        for segment in self.segments:
            counts[segment.rule] = counts.get(segment.rule, 0) + 1
        return counts

    def source_spans(
        self, start: int | None = None, end: int | None = None
    ) -> list[tuple[int, int]]:
        """Map a half-open projected-character range to source ranges.

        Adjacent and overlapping source ranges are merged.  A range that
        crosses different branches of one conversion construct therefore
        returns one range covering both, while a range that crosses an omitted
        block returns separate ranges around it.
        """
        if start is None:
            start = 0
        if end is None:
            end = len(self)
        length = len(self)
        if start < 0:
            start += length
        if end < 0:
            end += length
        if start < 0 or end < 0 or start > length or end > length:
            raise IndexError('projected range out of bounds')
        if start > end:
            raise IndexError('start is after end')
        if start == end:
            return []

        spans: list[tuple[int, int]] = []
        out_start = 0
        for segment in self.segments:
            out_end = out_start + len(segment.text)
            if segment.text and out_end > start and out_start < end:
                left = max(start, out_start) - out_start
                right = min(end, out_end) - out_start
                span = (segment.start + left, segment.start + right)
                if spans and span[0] <= spans[-1][1]:
                    spans[-1] = (spans[-1][0], max(spans[-1][1], span[1]))
                else:
                    spans.append(span)
            out_start = out_end
        return spans

    def source_ranges(
        self, start: int | None = None, end: int | None = None
    ) -> list[tuple[int, int]]:
        """Alias for ``source_spans``."""
        return self.source_spans(start, end)

    def spans(self, start: int | None = None, end: int | None = None):
        """Alias for ``source_spans``."""
        return self.source_spans(start, end)

    @property
    def char_spans(self) -> tuple[tuple[tuple[int, int], ...], ...]:
        """Return the source range(s) for each projected character."""
        char_spans: list[tuple[tuple[int, int], ...]] = []
        for segment in self.segments:
            if not segment.text:
                continue
            char_spans.extend(
                ((segment.start + i, segment.start + i + 1),)
                for i in range(len(segment.text))
            )
        return tuple(char_spans)

    @property
    def range_map(self) -> tuple[tuple[int, int, VariantSegment], ...]:
        """Return ``(projected_start, projected_end, segment)`` tuples."""
        result = []
        out_start = 0
        for segment in self.segments:
            out_end = out_start + len(segment.text)
            result.append((out_start, out_end, segment))
            out_start = out_end
        return tuple(result)

    @property
    def mapping(self) -> tuple[tuple[str, int, int, str], ...]:
        """Return ``(text, start, end, rule)`` tuples for every fragment."""
        return tuple(
            (segment.text, segment.start, segment.end, segment.rule)
            for segment in self.segments
        )

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation."""
        return {
            'variant': self.variant,
            'text': self.text,
            'segments': [segment.as_dict() for segment in self.segments],
        }

    def __repr__(self) -> str:
        return (
            f'{type(self).__name__}({str(self)!r}, '
            f'variant={self.variant!r})'
        )


def _skip_comment(source: str, index: int) -> int:
    match = _COMMENT_RE.match(source, index)
    return len(source) if match is None else match.end()


def _find_template_end(source: str, index: int) -> int:
    """Return the end of a balanced template or parameter starting at index."""
    stack: list[str] = []
    i = index
    length = len(source)
    while i < length:
        if source.startswith('<!--', i):
            i = _skip_comment(source, i)
        elif source.startswith('{{{', i):
            stack.append('parameter')
            i += 3
        elif source.startswith('{{', i):
            stack.append('template')
            i += 2
        elif source.startswith('}}}', i) and stack:
            if stack[-1] == 'parameter':
                stack.pop()
                i += 3
            else:
                stack.pop()
                i += 2
            if not stack:
                return i
        elif source.startswith('}}', i) and stack:
            stack.pop()
            i += 2
            if not stack:
                return i
        else:
            i += 1
    return length


def _find_wikilink_end(source: str, index: int) -> int:
    """Return the end of a balanced ``[[ ... ]]`` starting at index."""
    depth = 0
    i = index
    length = len(source)
    while i < length:
        if source.startswith('<!--', i):
            i = _skip_comment(source, i)
        elif source.startswith('[[', i):
            depth += 1
            i += 2
        elif source.startswith(']]', i) and depth:
            depth -= 1
            i += 2
            if depth == 0:
                return i
        else:
            i += 1
    return length


def _find_conversion_end(source: str, index: int) -> int | None:
    """Return the end of a balanced ``-{ ... }-`` construct."""
    depth = 1
    i = index + 2
    length = len(source)
    while i < length:
        if source.startswith('<!--', i):
            i = _skip_comment(source, i)
        elif source.startswith('{{', i):
            i = _find_template_end(source, i)
        elif source.startswith('-{', i):
            depth += 1
            i += 2
        elif source.startswith('}-', i):
            depth -= 1
            i += 2
            if depth == 0:
                return i
        else:
            i += 1
    return None


def _normalize_selector(selector: str) -> str:
    return selector.strip().lower().replace('_', '-')


@dataclass(frozen=True)
class _Rule:
    selector: str
    text: str
    text_start: int
    text_end: int


def _skip_nested(source: str, index: int, end: int) -> int:
    """Skip a nested construct that may contain rule separators."""
    if source.startswith('<!--', index):
        return min(_skip_comment(source, index), end)
    if source.startswith('{{', index):
        return min(_find_template_end(source, index), end)
    if source.startswith('-{', index):
        nested_end = _find_conversion_end(source, index)
        return end if nested_end is None else min(nested_end, end)
    if source.startswith('[[', index):
        return min(_find_wikilink_end(source, index), end)
    if source[index] == '<':
        match = _PROTECTED_TAG_RE.match(source, index)
        if match is not None:
            return min(match.end(), end)
        markup = _HTML_TAG_MARKUP_RE.match(source, index)
        if markup is not None:
            return min(markup.end(), end)
    return index


def _parse_rules(body: str, body_start: int) -> list[_Rule] | None:
    """Parse top-level ``selector:text`` rules and keep their offsets.

    ``None`` means the construct is not a valid conversion table.  Nested
    templates and conversion constructs are skipped while looking for
    separators, so their semicolons and colons do not split a rule.
    """
    parts: list[tuple[int, int]] = []
    part_start = 0
    i = 0
    length = len(body)
    while i < length:
        skipped = _skip_nested(body, i, length)
        if skipped != i:
            i = skipped
            continue
        if body[i] == ';':
            # A semicolon only separates rules when a selector and a colon
            # follow it.  Semicolons inside branch prose stay put.
            head_start = i + 1
            while head_start < length and body[head_start].isspace():
                head_start += 1
            colon = body.find(':', head_start)
            if colon != -1:
                selector = _normalize_selector(body[head_start:colon])
                if _SELECTOR_RE.match(selector):
                    parts.append((part_start, i))
                    part_start = i + 1
        i += 1
    parts.append((part_start, length))

    rules: list[_Rule] = []
    indices: dict[str, int] = {}
    has_content = False
    has_selector = False
    for part_start, part_end in parts:
        if part_end == length:
            # One optional trailing semicolon is punctuation before ``}-``,
            # not part of the last branch.
            trailing = body[part_start:part_end].rstrip()
            if trailing.endswith(';'):
                part_end = part_start + len(trailing) - 1
        raw = body[part_start:part_end]
        stripped = raw.strip()
        if not stripped:
            continue
        left = part_start + (len(raw) - len(raw.lstrip()))
        colon = body.find(':', left, part_end)
        if colon == -1:
            if has_content:
                return None
            # ``-{default text; zh-hans:text}-`` supplies the fallback
            # branch.  The selector for it is the empty string.
            has_content = True
            rules.append(
                _Rule(
                    '',
                    stripped,
                    left + body_start,
                    left + len(stripped) + body_start,
                )
            )
            continue
        selector = _normalize_selector(body[left:colon])
        if not _SELECTOR_RE.match(selector):
            return None
        if selector not in KNOWN_VARIANTS:
            # An unsupported selector makes the whole construct unexplained.
            return None
        has_content = True
        has_selector = True
        value_raw = body[colon + 1 : part_end]
        value = value_raw.strip()
        value_start = colon + 1 + (len(value_raw) - len(value_raw.lstrip()))
        value_end = value_start + len(value)
        rule = _Rule(
            selector,
            value,
            value_start + body_start,
            value_end + body_start,
        )
        index = indices.get(selector)
        if index is None:
            indices[selector] = len(rules)
            rules.append(rule)
        else:
            # MediaWiki keeps the last rule for a repeated selector.
            rules[index] = rule
    if not has_content or not has_selector:
        return None
    return rules


def _fallback_chain(variant: str) -> Iterator[str]:
    seen = {variant}
    for candidate in VARIANT_FALLBACKS.get(variant, ()):
        if candidate not in seen:
            seen.add(candidate)
            yield candidate
    base = variant.partition('-')[0]
    if base and base not in seen:
        seen.add(base)
        yield base


def _select_branch(
    rules: list[_Rule], variant: str
) -> tuple[_Rule, str] | None:
    """Return the matching rule and the kind of match that was used.

    The kind is one of ``exact``, ``fallback``, or ``default``.  When no
    branch applies, ``None`` is returned and the construct remains
    unexplained source instead of inventing text.
    """
    for rule in rules:
        if rule.selector == variant:
            return rule, 'exact'
    for candidate in _fallback_chain(variant):
        for rule in rules:
            if rule.selector == candidate:
                return rule, 'fallback'
    for rule in rules:
        if not rule.selector:
            return rule, 'default'
    # A well-formed table with no matching branch displays its first branch,
    # matching MediaWiki's behavior when conversion is unavailable.
    return rules[0], 'fallback'


def _split_replacement(rule: _Rule) -> tuple[str, int, int, bool]:
    """Interpret ``source=>target`` as a local replacement rule.

    Only the target is projected.  The source side is intentionally not
    remembered: a local rule ends at the closing ``}-`` and never applies to
    later text.  The returned offsets locate the target inside the source.
    """
    before, separator, after = rule.text.partition('=>')
    if not separator:
        return rule.text, rule.text_start, rule.text_end, False
    stripped = after.strip()
    if not stripped:
        # ``source=>`` has no target; keep the source side only, with an
        # exact span so per-character provenance stays correct.
        return (
            before,
            rule.text_start,
            rule.text_start + len(before),
            False,
        )
    start = rule.text_start + len(before) + 2
    start += len(after) - len(after.lstrip())
    return stripped, start, start + len(stripped), True


class _Projector:
    """Single-pass scanner that emits variant fragments."""

    __slots__ = ('segments', 'source', 'variant')

    def __init__(self, source: str, variant: str) -> None:
        self.source = source
        self.variant = variant
        self.segments: list[VariantSegment] = []

    def emit(
        self,
        text: str,
        start: int,
        end: int,
        rule: str,
        branch: str | None = None,
        tag: str | None = None,
        construct_start: int | None = None,
        construct_end: int | None = None,
    ) -> None:
        self.segments.append(
            VariantSegment(
                text,
                start,
                end,
                rule,
                branch=branch,
                tag=tag,
                construct_start=construct_start,
                construct_end=construct_end,
            )
        )

    def literal(self, start: int, end: int) -> None:
        if end > start:
            self.emit(self.source[start:end], start, end, LITERAL)

    def scan(self, start: int, end: int) -> None:
        source = self.source
        literal_start = start
        i = start
        while i < end:
            if source.startswith('<!--', i):
                comment_end = min(_skip_comment(source, i), end)
                self.literal(literal_start, i)
                self.emit(source[i:comment_end], i, comment_end, COMMENT)
                i = literal_start = comment_end
                continue
            if source[i] == '<':
                match = _PROTECTED_TAG_RE.match(source, i)
                if match is not None and match.end() <= end:
                    tag_end = match.end()
                    self.literal(literal_start, i)
                    self.emit(
                        source[i:tag_end],
                        i,
                        tag_end,
                        PROTECTED,
                        tag=match.group('name').lower(),
                    )
                    i = literal_start = tag_end
                    continue
                markup = _HTML_TAG_MARKUP_RE.match(source, i)
                if markup is not None and markup.end() <= end:
                    # Attribute values are not rendered text.  Skip the tag
                    # markup so a conversion marker inside an attribute is
                    # not interpreted.
                    i = markup.end()
                    continue
            if source.startswith('{{', i):
                template_end = min(_find_template_end(source, i), end)
                self.literal(literal_start, i)
                self.emit(source[i:template_end], i, template_end, TEMPLATE)
                i = literal_start = template_end
                continue
            if source.startswith('-{', i):
                construct_end = _find_conversion_end(source, i)
                if construct_end is None:
                    self.literal(literal_start, i)
                    self.emit(source[i:end], i, end, UNKNOWN)
                    i = literal_start = end
                    break
                if construct_end > end:
                    # The requested region ends inside a construct; include
                    # the whole construct so the projection never depends on
                    # a truncated branch table.
                    self.literal(literal_start, i)
                    self.conversion(i, construct_end)
                    i = literal_start = end
                    break
                self.literal(literal_start, i)
                self.conversion(i, construct_end)
                i = literal_start = construct_end
                continue
            i += 1
        self.literal(literal_start, end)

    def conversion(self, start: int, end: int) -> None:
        source = self.source
        body_start = start + 2
        body_end = end - 2
        body = source[body_start:body_end]
        if not body.strip():
            self.emit(
                '', start, end, EMPTY,
                construct_start=start, construct_end=end,
            )
            return

        flag_match = _FLAG_RE.match(body)
        if flag_match is not None:
            flag = flag_match.group(1).upper()
            content_start = body_start + flag_match.end()
            content = source[content_start:body_end]
            if flag == 'H':
                hidden_rules = _parse_rules(content, content_start)
                if hidden_rules is None or not any(
                    rule.selector for rule in hidden_rules
                ):
                    self.unexplained(start, end)
                else:
                    self.emit(
                        '',
                        start,
                        end,
                        HIDDEN,
                        construct_start=start,
                        construct_end=end,
                    )
                return
            if flag == 'T':
                # ``-{T|title}-`` supplies a display title and emits no body.
                self.emit(
                    '',
                    start,
                    end,
                    TITLE,
                    construct_start=start,
                    construct_end=end,
                )
                return
            if flag == 'R':
                self.emit(
                    content,
                    content_start,
                    body_end,
                    RAW,
                    construct_start=start,
                    construct_end=end,
                )
                return
            if flag == 'D':
                # ``-{D|text}-`` marks description text that is not converted.
                self.emit(
                    content,
                    content_start,
                    body_end,
                    NO_CONVERT,
                    construct_start=start,
                    construct_end=end,
                )
                return
            # Any other flag is not understood; keep the original construct.
            self.unexplained(start, end)
            return

        rules = _parse_rules(body, body_start)
        if rules is None:
            if ':' not in body:
                # ``-{literal text}-`` disables conversion for its contents.
                # A bare ``-{source=>target}-`` is a default replacement.
                text = body
                rule = NO_CONVERT
                text_start = body_start
                text_end = body_end
                if '=>' in body:
                    before, _, after = body.partition('=>')
                    target = after.strip()
                    if target:
                        text = target
                        text_start = (
                            body_start
                            + len(before)
                            + 2
                            + len(after)
                            - len(after.lstrip())
                        )
                        text_end = text_start + len(target)
                        rule = DEFAULT_REPLACE
                    else:
                        text = before
                        text_end = body_start + len(before)
                self.emit(
                    text,
                    text_start,
                    text_end,
                    rule,
                    construct_start=start,
                    construct_end=end,
                )
            else:
                # An unsupported selector or malformed rule is unexplained.
                self.unexplained(start, end)
            return

        if all(not rule.selector for rule in rules):
            # A body made only of the default branch is the explicit
            # ``-{literal text}-`` no-conversion form.
            self.emit(
                body,
                body_start,
                body_end,
                NO_CONVERT,
                construct_start=start,
                construct_end=end,
            )
            return

        selected = _select_branch(rules, self.variant)
        if selected is None:
            # No branch matches the requested variant and there is no
            # default.  Keep the construct as unexplained source text.
            self.unexplained(start, end)
            return
        rule, match = selected
        text, text_start, text_end, replaced = _split_replacement(rule)
        if match == 'default':
            applied = DEFAULT_REPLACE if replaced else DEFAULT
        elif match == 'fallback':
            applied = REPLACE_FALLBACK if replaced else VARIANT_FALLBACK
        else:
            applied = REPLACE if replaced else VARIANT
        if not text:
            self.emit(
                '',
                text_start,
                text_end,
                applied,
                branch=rule.selector or None,
                construct_start=start,
                construct_end=end,
            )
            return
        self.selected_region(
            text_start,
            text_end,
            applied,
            rule.selector or None,
            start,
            end,
        )

    def unexplained(self, start: int, end: int) -> None:
        """Emit a construct that the projection does not understand."""
        self.emit(
            self.source[start:end],
            start,
            end,
            UNKNOWN,
            construct_start=start,
            construct_end=end,
        )

    def selected_region(
        self,
        start: int,
        end: int,
        rule: str,
        branch: str | None,
        construct_start: int,
        construct_end: int,
    ) -> None:
        """Emit a selected branch, resolving nested conversion constructs."""
        source = self.source
        literal_start = start
        i = start
        while i < end:
            if source.startswith('<!--', i):
                i = min(_skip_comment(source, i), end)
                continue
            if source.startswith('{{', i):
                # A template inside the branch is not evaluated and not
                # recursively converted.
                i = min(_find_template_end(source, i), end)
                continue
            if source[i] == '<':
                # Conversion markers in attribute values are not text.
                markup = _HTML_TAG_MARKUP_RE.match(source, i)
                if markup is not None and markup.end() <= end:
                    i = markup.end()
                    continue
                tag = _PROTECTED_TAG_RE.match(source, i)
                if tag is not None and tag.end() <= end:
                    i = tag.end()
                    continue
            if source.startswith('-{', i):
                nested_end = _find_conversion_end(source, i)
                if nested_end is None or nested_end > end:
                    break
                if i > literal_start:
                    self.emit(
                        source[literal_start:i],
                        literal_start,
                        i,
                        rule,
                        branch=branch,
                        construct_start=construct_start,
                        construct_end=construct_end,
                    )
                self.conversion(i, nested_end)
                i = literal_start = nested_end
                continue
            i += 1
        if end > literal_start:
            self.emit(
                source[literal_start:end],
                literal_start,
                end,
                rule,
                branch=branch,
                construct_start=construct_start,
                construct_end=construct_end,
            )


def project_variant(
    source: str,
    variant: str | None = None,
    *,
    language_variant: str | None = None,
    start: int | None = None,
    end: int | None = None,
) -> VariantProjection:
    """Project wikitext onto a language variant.

    :param source: The wikitext to project.  ``WikiText`` objects are accepted
        through their string representation.
    :param variant: A MediaWiki variant code such as ``zh-hans``, ``zh-hant``,
        ``zh-cn``, or ``zh-tw``.  Chinese regional variants fall back to their
        script variant when necessary.
    :param start: Optional start offset of the region to project.  Offsets in
        the result remain relative to ``source``.
    :param end: Optional exclusive end offset of the region to project.
    :return: A :class:`VariantProjection`, a ``str`` subclass whose
        ``segments`` attribute records the original source range and the rule
        used for every fragment.

    Conversion constructs have the form ``-{zh-hans:简体; zh-hant:繁體}-``.
    ``-{text}-`` is an explicit no-conversion form, and
    ``-{source=>target}-`` is a local default replacement.  The ``H`` and
    ``T`` flags describe hidden and title rules and therefore emit no body
    text; ``R`` and ``D`` emit their contents without conversion.  Unknown
    flags, unknown variant selectors, and unbalanced constructs are preserved
    as unexplained source.  Templates and parser functions are copied
    verbatim and are never expanded.  Comments and the contents of literal
    tags such as ``nowiki``, ``pre``, and ``syntaxhighlight`` are never
    converted.

    Rules are local: a replacement selected inside one construct never
    affects text after its closing ``}-``.
    """
    if variant is None:
        variant = language_variant
    if not isinstance(variant, str):
        raise TypeError('variant must be a string')
    normalized = _normalize_selector(variant)
    if not normalized:
        raise ValueError('variant must not be empty')
    source_text = str(source)
    length = len(source_text)
    if start is None:
        start = 0
    if end is None:
        end = length
    if start < 0:
        start += length
    if end < 0:
        end += length
    if start < 0 or end < 0 or start > length or end > length:
        raise IndexError('source range out of bounds')
    if start > end:
        raise IndexError('start is after end')
    projector = _Projector(source_text, normalized)
    projector.scan(start, end)
    text = ''.join(segment.text for segment in projector.segments)
    return VariantProjection(
        text,
        variant=normalized,
        source=source_text,
        segments=projector.segments,
        source_start=start,
        source_end=end,
    )


def project_variant_segments(
    source: str, variant: str
) -> tuple[VariantSegment, ...]:
    """Return only the provenance fragments for a variant projection."""
    return project_variant(source, variant).segments


def project_segments(
    source: str, variant: str
) -> tuple[VariantSegment, ...]:
    """Alias for :func:`project_variant_segments`."""
    return project_variant_segments(source, variant)


# Aliases for callers that name the operation differently.
variant_projection = project_variant
to_variant = project_variant
convert_variant = project_variant
project_text = project_variant
variant_text = project_variant
project = project_variant
language_variant = project_variant
render_variant = project_variant
resolve_variant = project_variant
VariantText = VariantProjection
