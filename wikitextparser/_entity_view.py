"""Build a plain-text view of wikitext for entity retrieval.

Company names and other entities often contain apostrophes (``O'Brien``,
``Lloyd's``) while wikitext uses runs of consecutive apostrophes as
bold/italic markup. This module provides :func:`entity_view`, which
produces an :class:`EntityTextView`:

* ``text`` is the plain text to be indexed. Unambiguous names appear
  here verbatim, so they can be indexed as usual.
* ``segments`` maps every view range back to its source range, keeping
  the correspondence between name text and the original wikitext.
* ``styles`` records bold/italic ranges, again mapped to both view and
  source offsets.
* ``ambiguities`` reports apostrophe runs whose local interpretation is
  not unique (e.g. ``''''``). Such runs are *not* silently dropped: the
  view text uses the primary (MediaWiki-compatible) interpretation and
  every candidate interpretation is returned together with the affected
  source range, so the indexer can decide how to handle them.

All offsets are relative to the start of the parsed string.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

from ._wikitext import WikiText

__all__ = [
    'Ambiguity',
    'AmbiguityCandidate',
    'EntityTextView',
    'StyleRange',
    'ViewSegment',
    'entity_view',
]

# Toggles applied by the canonical (unambiguous) quote runs.
_RUN_TOGGLES = {2: ('italic',), 3: ('bold',), 5: ('bold', 'italic')}
# Toggles applied by each part a quote run can be decomposed into:
# 1 -> literal apostrophe, 2 -> italic, 3 -> bold, 5 -> bold + italic.
_PART_TOGGLES = {1: (), 2: ('italic',), 3: ('bold',), 5: ('bold', 'italic')}
_MAX_CANDIDATES = 8


@dataclass(frozen=True)
class ViewSegment:
    """Map a range of the view text to its range in the source string.

    Within a segment the correspondence is one-to-one:
    ``view_start + k`` corresponds to ``source_start + k``.
    """

    view_start: int
    view_end: int
    source_start: int
    source_end: int


@dataclass(frozen=True)
class StyleRange:
    """A range of the view text that is bold and/or italic."""

    view_start: int
    view_end: int
    source_start: int
    source_end: int
    styles: frozenset[str]  # a non-empty subset of {'bold', 'italic'}


@dataclass(frozen=True)
class AmbiguityCandidate:
    """One possible local interpretation of an ambiguous quote run.

    ``text`` is the literal text the run contributes to the view under
    this interpretation; ``toggles`` are the style toggles it applies,
    in order.
    """

    text: str
    toggles: tuple[str, ...]


@dataclass(frozen=True)
class Ambiguity:
    """An ambiguous quote run and its possible interpretations.

    The first candidate is the primary interpretation, i.e. the one used
    in :attr:`EntityTextView.text`. It follows MediaWiki's own
    convention (4 quotes -> ``'`` + bold, 6 or more quotes -> the
    leading apostrophes + bold-italic).
    """

    view_start: int
    view_end: int
    source_start: int
    source_end: int
    candidates: tuple[AmbiguityCandidate, ...]


@dataclass(frozen=True)
class EntityTextView:
    """Plain-text view of a wikitext string, for entity retrieval."""

    text: str
    segments: tuple[ViewSegment, ...]
    styles: tuple[StyleRange, ...]
    ambiguities: tuple[Ambiguity, ...]

    def source_span(
        self, view_start: int, view_end: int | None = None
    ) -> tuple[int, int] | None:
        """Return the source range corresponding to a view range.

        ``view_end`` defaults to ``view_start + 1``. Return None if the
        view contains no text. Positions falling on removed markup are
        clamped to the nearest mapped source position.
        """
        segments = self.segments
        if not segments:
            return None
        if view_end is None:
            view_end = view_start + 1
        starts = [segment.view_start for segment in segments]

        def clamp(pos: int) -> int:
            segment = segments[max(bisect_right(starts, pos) - 1, 0)]
            offset = pos - segment.view_start
            if offset < 0:
                return segment.source_start
            if offset > segment.source_end - segment.source_start:
                return segment.source_end
            return segment.source_start + offset

        if view_end > view_start:
            # map the inclusive end (the last character of the range)
            return clamp(view_start), clamp(view_end - 1) + 1
        return clamp(view_start), clamp(view_end)


@dataclass(frozen=True)
class _Checkpoint:
    """Style state after a markup token, with the token's source span."""

    view_pos: int
    token_start: int
    token_end: int
    styles: frozenset[str]


def _compositions_exact(n: int, parts_count: int):
    """Yield compositions of n into exactly parts_count parts of {1,2,3,5}."""
    if parts_count == 1:
        if n in _PART_TOGGLES:
            yield (n,)
        return
    for part in (1, 2, 3, 5):
        if part < n:
            for rest in _compositions_exact(n - part, parts_count - 1):
                yield (part, *rest)


def _quote_run_candidates(length: int) -> tuple[AmbiguityCandidate, ...]:
    """Return the candidate interpretations for a run of `length` quotes.

    Only called for ambiguous lengths (4 or >= 6). The first candidate
    is the primary interpretation, following MediaWiki's convention.
    Candidates are deduplicated by their (text, toggles) effect.
    """
    if length == 4:
        canonical = (1, 3)
    else:  # length >= 6
        canonical = (1,) * (length - 5) + (5,)
    seen = set()
    candidates = []

    def add(composition: tuple[int, ...]) -> None:
        text = "'" * composition.count(1)
        toggles = tuple(
            toggle for part in composition for toggle in _PART_TOGGLES[part]
        )
        key = (text, toggles)
        if key in seen or len(candidates) >= _MAX_CANDIDATES:
            return
        seen.add(key)
        candidates.append(AmbiguityCandidate(text=text, toggles=toggles))

    add(canonical)
    add((1,) * length)  # all-literal fallback, e.g. a possessive cluster
    parts_count = 1
    while len(candidates) < _MAX_CANDIDATES and parts_count < length:
        parts_count += 1
        for composition in _compositions_exact(length, parts_count):
            if len(candidates) >= _MAX_CANDIDATES:
                break
            add(composition)
    return tuple(candidates)


class _ViewBuilder:
    __slots__ = (
        'active',
        'ambiguities',
        'checkpoints',
        'segments',
        'source',
        'view_len',
    )

    def __init__(self, source: str) -> None:
        self.source = source
        self.segments: list[ViewSegment] = []
        self.checkpoints: list[_Checkpoint] = []
        self.ambiguities: list[Ambiguity] = []
        self.active: set[str] = set()
        self.view_len = 0

    def emit_literal(self, start: int, end: int) -> None:
        """Emit source[start:end] as literal view text."""
        if start >= end:
            return
        segments = self.segments
        if segments and segments[-1].source_end == start:
            last = segments[-1]
            segments[-1] = ViewSegment(
                last.view_start,
                last.view_end + end - start,
                last.source_start,
                end,
            )
        else:
            segments.append(
                ViewSegment(
                    self.view_len, self.view_len + end - start, start, end
                )
            )
        self.view_len += end - start

    def apply_toggles(
        self, toggles: tuple[str, ...], token_start: int, token_end: int
    ) -> None:
        active = self.active
        for toggle in toggles:
            if toggle in active:
                active.discard(toggle)
            else:
                active.add(toggle)
        self.checkpoints.append(
            _Checkpoint(
                self.view_len, token_start, token_end, frozenset(active)
            )
        )

    def emit_run(self, start: int, end: int, length: int) -> None:
        """Emit a quote run of `length` apostrophes.

        `start`/`end` delimit the run in the source; the run may span
        comments, which are transparent to quote detection.
        """
        if length == 1:
            # A lone apostrophe is always literal text (e.g. O'Brien).
            self.emit_literal(start, start + 1)
            return
        if length in _RUN_TOGGLES:
            self.apply_toggles(_RUN_TOGGLES[length], start, end)
            return
        # Ambiguous run: keep the primary interpretation in the view
        # text instead of dropping the sequence, and record candidates.
        candidates = _quote_run_candidates(length)
        primary = candidates[0]
        literal_length = len(primary.text)
        view_start = self.view_len
        if literal_length:
            self.emit_literal(start, start + literal_length)
        self.apply_toggles(primary.toggles, start, end)
        self.ambiguities.append(
            Ambiguity(
                view_start,
                view_start + literal_length,
                start,
                end,
                candidates,
            )
        )


def entity_view(source: str | WikiText) -> EntityTextView:
    """Return an EntityTextView of the given string or WikiText object.

    The view's text keeps unambiguous names (including apostrophes such
    as ``O'Brien``) intact for indexing, maps bold/italic style ranges
    back to the source, and reports ambiguous apostrophe runs with their
    candidate interpretations instead of deleting them.
    """
    wt = source if isinstance(source, WikiText) else WikiText(source)
    base = wt._span_data.start
    string = wt.string
    length = len(string)
    comments = sorted(
        (comment.span[0] - base, comment.span[1] - base)
        for comment in wt.comments
    )
    builder = _ViewBuilder(string)
    i = 0
    ci = 0
    n_comments = len(comments)
    text_from = None
    while i < length:
        if ci < n_comments and comments[ci][0] == i:
            if text_from is not None:
                builder.emit_literal(text_from, i)
                text_from = None
            i = comments[ci][1]
            ci += 1
            continue
        if string[i] == "'":
            if text_from is not None:
                builder.emit_literal(text_from, i)
                text_from = None
            j = i
            run = 0
            while True:
                while j < length and string[j] == "'":
                    j += 1
                    run += 1
                # Comments between quotes are transparent (COMMA_COMMENT).
                if ci < n_comments and comments[ci][0] == j:
                    comment_end = comments[ci][1]
                    if comment_end < length and string[comment_end] == "'":
                        ci += 1
                        j = comment_end
                        continue
                break
            builder.emit_run(i, j, run)
            i = j
        else:
            if text_from is None:
                text_from = i
            i += 1
    if text_from is not None:
        builder.emit_literal(text_from, length)

    styles = []
    previous = _Checkpoint(0, 0, 0, frozenset())
    for checkpoint in builder.checkpoints:
        if previous.styles and checkpoint.view_pos > previous.view_pos:
            styles.append(
                StyleRange(
                    previous.view_pos,
                    checkpoint.view_pos,
                    previous.token_end,
                    checkpoint.token_start,
                    previous.styles,
                )
            )
        previous = checkpoint
    if previous.styles and builder.view_len > previous.view_pos:
        styles.append(
            StyleRange(
                previous.view_pos,
                builder.view_len,
                previous.token_end,
                length,
                previous.styles,
            )
        )

    text = ''.join(
        string[segment.source_start : segment.source_end]
        for segment in builder.segments
    )
    return EntityTextView(
        text=text,
        segments=tuple(builder.segments),
        styles=tuple(styles),
        ambiguities=tuple(builder.ambiguities),
    )
