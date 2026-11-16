from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from ._spans import external_link_target_end
from ._wikitext import BRACKET_EXTERNAL_LINK_URL, IGNORECASE, SubWikiText, rc

URL_MATCH = rc(BRACKET_EXTERNAL_LINK_URL, IGNORECASE).match

# Punctuation that may either end a sentence or belong to a free URL. The
# full-width variants are included because they are common in the wikitext this
# parser is used on; keeping them here only widens the set of reported
# candidates, it never changes a bracketed link's proven boundary.
AMBIGUOUS_TARGET_PUNCTUATION = '.,;:!?\'"。，、；：！？'


@dataclass(frozen=True)
class ExternalLinkCandidate:
    """One possible interpretation of a free external link boundary.

    Spans are absolute offsets into the root wikitext. ``link_span`` includes
    the whole occurrence. ``target_span`` is the target under this candidate.
    For a bracketed link there is only one candidate; its boundary comes from
    the ``[`` and ``]`` delimiters and is therefore syntactically proven.
    """

    link_span: tuple[int, int]
    target_span: tuple[int, int]
    display_span: tuple[int, int] | None
    target: str
    display: str | None
    is_canonical: bool
    reason: str


@dataclass(frozen=True)
class ExternalLinkInfo:
    """Boundaries, display range, canonical target, and alternatives."""

    link_span: tuple[int, int]
    target_span: tuple[int, int]
    display_span: tuple[int, int] | None
    target: str
    display: str | None
    in_brackets: bool
    boundary_proven: bool
    display_equals_target: bool
    target_in_display: bool
    display_in_target: bool
    display_repeats_target: bool
    trailing_text: str
    candidates: tuple[ExternalLinkCandidate, ...] = field(
        default_factory=tuple
    )

    @property
    def boundary(self) -> tuple[int, int]:
        """Alias for :attr:`link_span`."""
        return self.link_span

    def as_dict(self) -> dict:
        """Return the analysis as a JSON-friendly dictionary."""
        return {
            'link_span': self.link_span,
            'target_span': self.target_span,
            'display_span': self.display_span,
            'target': self.target,
            'display': self.display,
            'in_brackets': self.in_brackets,
            'boundary_proven': self.boundary_proven,
            'display_equals_target': self.display_equals_target,
            'target_in_display': self.target_in_display,
            'display_in_target': self.display_in_target,
            'display_repeats_target': self.display_repeats_target,
            'trailing_text': self.trailing_text,
            'candidates': [
                {
                    'link_span': candidate.link_span,
                    'target_span': candidate.target_span,
                    'display_span': candidate.display_span,
                    'target': candidate.target,
                    'display': candidate.display,
                    'is_canonical': candidate.is_canonical,
                    'reason': candidate.reason,
                }
                for candidate in self.candidates
            ],
        }


@dataclass(frozen=True)
class ExternalLinkReplacement:
    """A target replacement applied to one link occurrence."""

    old_target: str
    new_target: str
    link_span_before: tuple[int, int]
    target_span_before: tuple[int, int]
    display_span_before: tuple[int, int] | None
    link_span_after: tuple[int, int]
    target_span_after: tuple[int, int]
    display_span_after: tuple[int, int] | None
    display_unchanged: bool
    reparsed: bool


class ExternalLinkReplacements(list[ExternalLinkReplacement]):
    """Applied replacements, including keys that did not match any link."""

    def __init__(self, *args, unmatched=()):
        super().__init__(*args)
        self.unmatched: list[str] = list(unmatched)


class ExternalLink(SubWikiText):
    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(args) > 1 or kwargs.get('_span') is not None:
            return
        string = self.string
        if string.startswith('['):
            return
        end = external_link_target_end(string)
        if end < len(string):
            self._span_data.end = self._span_data.start + end

    @property
    def _bracket_target_end(self) -> int:
        """Return the target end relative to self for a bracketed link."""
        match = URL_MATCH(self._ext_link_shadow, 1)
        if match is None:  # pragma: no cover - defensive
            return len(self.string) - (1 if self.string.endswith(']') else 0)
        return match.end()

    @property
    def link_span(self) -> tuple[int, int]:
        """Absolute span of the whole link occurrence.

        A bracketed link is bounded by its ``[`` and ``]``. A free link is
        bounded at the canonical target end; ambiguous trailing characters
        remain available through :attr:`candidates`.
        """
        start = self._span_data.start
        if self(0) == '[':
            return start, self._span_data.end
        return start, self._span_data.end

    @property
    def boundary(self) -> tuple[int, int]:
        """Alias for :attr:`link_span`."""
        return self.link_span

    @property
    def link_boundary(self) -> tuple[int, int]:
        """Alias for :attr:`link_span`."""
        return self.link_span

    @property
    def target_span(self) -> tuple[int, int]:
        """Absolute span of the canonical target bytes."""
        start = self._span_data.start
        if self(0) == '[':
            return start + 1, start + self._bracket_target_end
        return start, self._span_data.end

    @property
    def canonical_target_span(self) -> tuple[int, int]:
        """Alias for :attr:`target_span`."""
        return self.target_span

    @property
    def raw_target(self) -> str:
        """The target bytes under the widest boundary candidate."""
        candidates = self.candidates
        if len(candidates) > 1:
            return candidates[-1].target
        return self.target

    @property
    def target(self) -> str:
        """The canonical target, excluding brackets and display text."""
        start, end = self.target_span
        return self._lststr[0][start:end]

    @property
    def canonical_target(self) -> str:
        """Alias for :attr:`target`."""
        return self.target

    @property
    def display_span(self) -> tuple[int, int] | None:
        """Absolute span of the display text, or None when there is none."""
        string = self.string
        if not string.startswith('['):
            return None
        start = self._span_data.start
        url_end = self._bracket_target_end
        if url_end >= len(string) - 1:
            return None
        if string[url_end] == ' ':
            return start + url_end + 1, start + len(string) - 1
        return start + url_end, start + len(string) - 1

    @property
    def text_span(self) -> tuple[int, int] | None:
        """Alias for :attr:`display_span`."""
        return self.display_span

    @property
    def display(self) -> str | None:
        """The display text, or None for a bare or textless link."""
        span = self.display_span
        if span is None:
            return None
        return self._lststr[0][span[0] : span[1]]

    @property
    def display_text(self) -> str | None:
        """Alias for :attr:`display`."""
        return self.display

    @property
    def candidates(self) -> tuple[ExternalLinkCandidate, ...]:
        """Return the canonical interpretation followed by alternatives.

        Bracketed links have one syntactically proven interpretation. Free
        links may be followed by sentence punctuation or closing parentheses
        that could also belong to the URL; every such trim is retained.
        """
        link_start, link_end = self.link_span
        target_start, target_end = self.target_span
        display_span = self.display_span
        display = self.display
        target = self.target
        canonical_reason = (
            'bracket_delimiter' if self(0) == '[' else 'scheme_boundary'
        )
        candidates = [
            ExternalLinkCandidate(
                (link_start, link_end),
                (target_start, target_end),
                display_span,
                target,
                display,
                True,
                canonical_reason,
            )
        ]
        if self(0) == '[':
            return tuple(candidates)

        root = self._lststr[0]
        end = target_end
        root_len = len(root)
        while end < root_len:
            char = root[end]
            if char == ')':
                reason = 'ambiguous_closing_parenthesis'
            elif char in AMBIGUOUS_TARGET_PUNCTUATION:
                reason = 'ambiguous_sentence_punctuation'
            else:
                break
            end += 1
            candidates.append(
                ExternalLinkCandidate(
                    (link_start, end),
                    (target_start, end),
                    None,
                    root[target_start:end],
                    None,
                    False,
                    reason,
                )
            )
        if len(candidates) > 1:
            canonical = candidates[0]
            candidates[0] = ExternalLinkCandidate(
                canonical.link_span,
                canonical.target_span,
                canonical.display_span,
                canonical.target,
                canonical.display,
                True,
                'ambiguous_trim_kept_as_candidate',
            )
        return tuple(candidates)

    @property
    def boundary_candidates(self) -> tuple[ExternalLinkCandidate, ...]:
        """Alias for :attr:`candidates`."""
        return self.candidates

    def analyze(self) -> ExternalLinkInfo:
        """Return boundaries, display range, canonical target, and candidates."""
        display = self.display
        target = self.target
        link_span = self.link_span
        candidates = self.candidates
        trailing_text = (
            self._lststr[0][link_span[1] : candidates[-1].link_span[1]]
            if len(candidates) > 1
            else ''
        )
        return ExternalLinkInfo(
            link_span,
            self.target_span,
            self.display_span,
            target,
            display,
            self.in_brackets,
            self.in_brackets,
            display == target if display is not None else False,
            display is not None and target in display,
            display is not None and display in target,
            display is not None
            and (display == target or display in target or target in display),
            trailing_text,
            candidates,
        )

    @property
    def info(self) -> ExternalLinkInfo:
        """Property form of :meth:`analyze`."""
        return self.analyze()

    @property
    def url(self) -> str:
        """URL of the current ExternalLink object.

        getter: Return the canonical target.
        setter: Set a new value for the target. Add brackets for bare links.
        """
        return self.target

    @url.setter
    def url(self, newurl: str) -> None:
        self.replace_target(newurl)

    @property
    def text(self) -> str | None:
        """The text part (the part after the url).

        getter: Return None if this is a bare link or has no associated text.
        setter: Automatically put the ExternalLink in brackets if it's not
            already.
        deleter: Delete self.text, including the space before it.
        """
        return self.display

    @text.setter
    def text(self, newtext: str) -> None:
        string = self.string
        if string[0] == '[':
            text = self.text
            if text:
                self[-len(text) - 1 : -1] = newtext
                return
            self.insert(-1, ' ' + newtext)
            return
        self.insert(len(string), ' ' + newtext + ']')
        self.insert(0, '[')

    @text.deleter
    def text(self) -> None:
        string = self.string
        if string[0] != '[':
            return
        text = self.text
        if text:
            del self[-len(text) - 2 : -1]

    @property
    def in_brackets(self) -> bool:
        """Return true if the ExternalLink is in brackets. False otherwise."""
        return self(0) == '['

    def replace_target(self, new_target: str) -> None:
        """Replace only the target bytes, leaving display text untouched."""
        _validate_new_target(self, new_target)
        start, end = self.target_span
        base = self._span_data.start
        old_len = end - start
        self[start - base : end - base] = new_target
        delta = len(new_target) - old_len
        if delta and not self.in_brackets:
            # Keep the object's own span on the new target so that further
            # mutations of this link remain valid without reparsing.
            self._span_data.end = start + len(new_target)

    def retarget(self, new_target: str) -> None:
        """Alias for :meth:`replace_target`."""
        self.replace_target(new_target)

    @property
    def external_links(self) -> list[ExternalLink]:
        return []


def _validate_new_target(link: ExternalLink, new_target: str) -> None:
    """Raise ValueError if ``new_target`` cannot remain the same link."""
    from ._wikitext import WikiText

    if not new_target:
        raise ValueError('an external link target cannot be empty')
    if '\n' in new_target or '\r' in new_target:
        raise ValueError(
            f'external link target contains a line break: {new_target!r}'
        )
    if link.in_brackets:
        if ']' in new_target:
            raise ValueError(
                f'external link target cannot contain "]": {new_target!r}'
            )
        link_text = link.string
        target_start = link.target_span[0] - link._span_data.start
        target_end = link.target_span[1] - link._span_data.start
        probe = link_text[:target_start] + new_target + link_text[target_end:]
        parsed = WikiText(probe).external_links
        if len(parsed) != 1 or parsed[0].target != new_target:
            raise ValueError(
                'replacement target would not be reparsed as the same '
                f'bracketed external link: {new_target!r}'
            )
        if parsed[0].display != link.display:
            raise ValueError(
                'replacement target would change the display text: '
                f'{new_target!r}'
            )
        return

    if external_link_target_end(new_target) != len(new_target):
        raise ValueError(
            'replacement target would lose trailing bytes when reparsed: '
            f'{new_target!r}'
        )
    if new_target.count('(') != new_target.count(')'):
        raise ValueError(
            f'replacement target has unbalanced parentheses: {new_target!r}'
        )
    parsed = WikiText(new_target).external_links
    if len(parsed) != 1 or parsed[0].target != new_target:
        raise ValueError(
            'replacement target would not be reparsed as one free external '
            f'link: {new_target!r}'
        )


def _shift_for(position: int, deltas: Mapping[int, int]) -> int:
    return sum(delta for start, delta in deltas.items() if start < position)


def replace_external_link_targets(
    wikitext,
    replacements: Mapping[str, str] | Iterable[tuple[str, str]],
    *,
    with_report: bool = False,
):
    """Replace external-link targets by exact canonical target match.

    ``wikitext`` may be a :class:`WikiText` instance, which is mutated in
    place, or a string, in which case the replaced string is returned. Only
    the target span of each matching occurrence is written; identical bytes in
    display text, body prose, or another link's text are never searched.

    All replacements are validated before the first byte is changed. A new
    target that would not be recognized as the same link occurrence after
    reparsing raises ``ValueError``. ``replacements`` may be a mapping or an
    iterable of ``(old, new)`` pairs; the first value wins for duplicate keys.

    :keyword with_report: if True, return ``(result, report)``. ``report`` is
        an :class:`ExternalLinkReplacements` list in document order whose
        ``unmatched`` attribute lists replacement keys with no matching link.
    """
    from ._wikitext import WikiText

    is_string = isinstance(wikitext, str)
    wt = WikiText(wikitext) if is_string else wikitext
    root = wt._lststr[0]

    pairs = (
        list(replacements.items())
        if isinstance(replacements, Mapping)
        else list(replacements)
    )
    replacement_by_old: dict[str, str] = {}
    for old, new in pairs:
        if old in replacement_by_old:
            continue
        replacement_by_old[old] = new

    all_links = wt.external_links
    plans = []
    for link in all_links:
        old_target = link.target
        if old_target not in replacement_by_old:
            continue
        new_target = replacement_by_old[old_target]
        _validate_new_target(link, new_target)
        plans.append((link, old_target, new_target))

    matched = {old for _, old, _ in plans}
    unmatched = [old for old in replacement_by_old if old not in matched]

    # Compute the new root string by editing only the target spans.
    deltas: dict[int, int] = {}
    ordered = sorted(
        plans, key=lambda plan: plan[0].target_span[0], reverse=True
    )
    new_root = root
    for link, _, new_target in ordered:
        start, end = link.target_span
        new_root = new_root[:start] + new_target + new_root[end:]
        deltas[start] = start + len(new_target) - end

    # Validate every occurrence in the fully replaced text.
    reparsed_root = WikiText(new_root)
    reparsed_links = reparsed_root.external_links
    expected_records = []
    for link, old_target, new_target in plans:
        old_link_span = link.link_span
        old_target_span = link.target_span
        old_display_span = link.display_span
        new_target_start = old_target_span[0] + _shift_for(
            old_target_span[0], deltas
        )
        new_target_end = new_target_start + len(new_target)
        if link.in_brackets:
            new_link_start = old_link_span[0] + _shift_for(
                old_link_span[0], deltas
            )
            new_link_end = old_link_span[1] + _shift_for(
                old_link_span[1], deltas
            )
        else:
            new_link_start = old_link_span[0] + _shift_for(
                old_link_span[0], deltas
            )
            new_link_end = new_target_end
        if old_display_span is None:
            new_display_span = None
        else:
            new_display_span = (
                old_display_span[0] + _shift_for(old_display_span[0], deltas),
                old_display_span[1] + _shift_for(old_display_span[1], deltas),
            )
        expected_records.append(
            [
                old_target,
                new_target,
                old_link_span,
                old_target_span,
                old_display_span,
                (new_link_start, new_link_end),
                (new_target_start, new_target_end),
                new_display_span,
                link.display,
            ]
        )

    for expected in expected_records:
        new_link_span = expected[5]
        new_target = expected[1]
        old_display = expected[8]
        reparsed = next(
            (
                link
                for link in reparsed_links
                if link.link_span == new_link_span
            ),
            None,
        )
        if reparsed is None or reparsed.target != new_target:
            raise ValueError(
                'replacement target did not reparse as the same link '
                f'occurrence: {new_target!r}'
            )
        if reparsed.display != old_display:
            raise ValueError(
                f'replacement target changed the display text: {new_target!r}'
            )
        reparsed_target_span = reparsed.target_span
        reparsed_display_span = reparsed.display_span
        expected[6] = reparsed_target_span
        expected[7] = reparsed_display_span
        expected[5] = reparsed.link_span

    # Apply the validated replacements to the caller's object, right to left
    # so earlier absolute spans remain valid.
    base = wt._span_data.start
    for link, _, new_target in ordered:
        start, end = link.target_span
        wt[start - base : end - base] = new_target

    records = ExternalLinkReplacements(
        sorted(
            [
                ExternalLinkReplacement(
                    old_target,
                    new_target,
                    old_link_span,
                    old_target_span,
                    old_display_span,
                    new_link_span,
                    new_target_span,
                    new_display_span,
                    True,
                    True,
                )
                for (
                    old_target,
                    new_target,
                    old_link_span,
                    old_target_span,
                    old_display_span,
                    new_link_span,
                    new_target_span,
                    new_display_span,
                    _,
                ) in expected_records
            ],
            key=lambda record: record.link_span_before,
        ),
        unmatched=unmatched,
    )
    if with_report:
        return (wt.string if is_string else wt), records
    if is_string:
        return wt.string
    return records


def replace_external_link_targets_in_text(
    text: str,
    replacements: Mapping[str, str] | Iterable[tuple[str, str]],
) -> str:
    """Return ``text`` with matching external-link targets replaced."""
    return replace_external_link_targets(text, replacements)


def analyze_external_links(wikitext) -> list[ExternalLinkInfo]:
    """Return link boundaries, display ranges, and canonical targets."""
    return [link.analyze() for link in wikitext.external_links]
