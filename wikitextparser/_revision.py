"""Compare sections and in-page links between two revisions of a page.

Matching is evidence-based.  A section is never identified merely because
its title is the same: duplicate headings, parent hierarchy, sibling order,
and body text are all considered.  When the evidence cannot distinguish two
candidates, both are kept instead of choosing the first one.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Literal

from ._section import (
    Section,
    SectionHeadingInfo,
    normalize_section_anchor,
    section_heading_infos,
)
from ._wikitext import WikiText

Status = Literal['matched', 'renamed', 'ambiguous', 'added', 'removed']
LinkStatus = Literal['resolved', 'ambiguous', 'unresolved', 'unchanged']

# Two candidates whose scores differ by less than this are indistinguishable.
_TIE_MARGIN = 0.05
# Pairs below this score are not considered corresponding sections at all.
_MIN_SCORE = 0.35


def _body_text(section: Section) -> str:
    """Return normalized plain text of a section body, excluding headings."""
    return ' '.join(section.contents.split())


def _similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _token_overlap(a: str, b: str) -> float:
    """Return the Jaccard similarity of character bigrams in two bodies."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    aa = {a[i : i + 2] for i in range(len(a) - 1)}
    bb = {b[i : i + 2] for i in range(len(b) - 1)}
    union = aa | bb
    if not union:
        return 1.0 if a == b else 0.0
    return len(aa & bb) / len(union)


@dataclass
class SectionPair:
    """One section from the old revision and its possible new counterparts."""

    old: SectionHeadingInfo
    candidates: list[SectionHeadingInfo]
    status: Status
    evidence: dict[str, float] = field(default_factory=dict)

    @property
    def new(self) -> SectionHeadingInfo | None:
        """The uniquely identified new section, or ``None``."""
        if self.status == 'ambiguous' or not self.candidates:
            return None
        return self.candidates[0]

    @property
    def old_anchor(self) -> str | None:
        """The old section's generated anchor."""
        return self.old.anchor

    @property
    def new_anchor(self) -> str | None:
        """The new anchor when the correspondence is unique."""
        target = self.new
        return target.anchor if target is not None else None

    @property
    def candidate_anchors(self) -> list[str]:
        """Anchors of all plausible counterparts, in score order."""
        return [
            info.anchor for info in self.candidates if info.anchor is not None
        ]


@dataclass
class LinkResolution:
    """Result of resolving one section-link fragment across revisions."""

    old_fragment: str
    new_fragment: str | None
    status: LinkStatus
    candidates: list[str] = field(default_factory=list)
    old_anchor: str | None = None
    reason: str = ''


def _features(
    old: SectionHeadingInfo,
    new: SectionHeadingInfo,
    old_title_counts: Counter[str],
    new_title_counts: Counter[str],
) -> tuple[float, dict[str, float]]:
    """Score how strongly *new* corresponds to *old* and return the parts."""
    old_title = old.base_anchor or ''
    title_match = 1.0 if old_title == (new.base_anchor or '') else 0.0
    # A title shared by several headings has little discriminative power.
    title_weight = (
        0.25
        if old_title_counts[old_title] == 1
        and new_title_counts[old_title] == 1
        else 0.0
    )
    parent_match = 1.0 if old.parent_path == new.parent_path else 0.0
    level_match = 1.0 if old.level == new.level else 0.0
    ordinal_match = 1.0 if old.sibling_ordinal == new.sibling_ordinal else 0.0
    # A parent that matches but whose occurrence among its siblings is not
    # the same is weak evidence: two identical children may have been
    # exchanged and the parent alone cannot tell them apart.
    identity_contradiction = bool(
        parent_match
        and old.base_anchor == new.base_anchor
        and not ordinal_match
    )
    old_body = _body_text(old.section)
    new_body = _body_text(new.section)
    body_seq = _similarity(old_body, new_body)
    body_overlap = _token_overlap(old_body, new_body)
    body_score = 0.5 * body_seq + 0.5 * body_overlap
    if old_body == new_body:
        # Identical text alone cannot identify a section; it is very common
        # for duplicated headings to share or even swap bodies.
        body_score = 1.0
    index_match = 1.0 if old.heading_index == new.heading_index else 0.0
    score = (
        title_weight * title_match
        + 0.55 * body_score
        + 0.12 * parent_match
        + 0.02 * index_match
        + 0.03 * ordinal_match
        + 0.02 * level_match
    )
    if identity_contradiction:
        score -= 0.02
    if title_match and parent_match:
        # A heading with the same title under the same parent is always a
        # plausible counterpart, even when its body was rewritten.  Treating
        # it as an unrelated add/remove pair would hide a real correspondence.
        score = max(score, _MIN_SCORE)
    evidence = {
        'title': title_match,
        'parent': parent_match,
        'level': level_match,
        'ordinal': ordinal_match,
        'index': index_match,
        'body': body_score,
        'body_sequence': body_seq,
        'body_overlap': body_overlap,
        'score': score,
    }
    return score, evidence


class RevisionComparison:
    """Pair the sections of two revisions and resolve in-page links.

    ``old`` and ``new`` are parsed :class:`~wikitextparser.WikiText`
    objects (or strings accepted by :func:`compare_revisions`).  The result
    exposes ``section_pairs``, ``added``, ``removed``, and link resolution
    helpers.  A pair whose counterpart is not uniquely determined has status
    ``'ambiguous'`` and keeps every plausible candidate.
    """

    def __init__(self, old: WikiText, new: WikiText) -> None:
        self.old = old
        self.new = new
        self.old_headings = [
            info for info in section_heading_infos(old) if info.level
        ]
        self.new_headings = [
            info for info in section_heading_infos(new) if info.level
        ]
        self._old_title_counts = Counter(
            info.base_anchor or '' for info in self.old_headings
        )
        self._new_title_counts = Counter(
            info.base_anchor or '' for info in self.new_headings
        )
        self._new_by_anchor: dict[str, SectionHeadingInfo] = {
            info.anchor: info
            for info in self.new_headings
            if info.anchor is not None
        }
        self._old_by_anchor: dict[str, SectionHeadingInfo] = {
            info.anchor: info
            for info in self.old_headings
            if info.anchor is not None
        }
        self._pairs_by_old: dict[int, SectionPair] = {}
        self._pairs_by_new: dict[int, SectionPair] = {}
        self._ambiguous_new: set[int] = set()
        self.section_pairs: list[SectionPair] = []
        self._match_sections()

    # ------------------------------------------------------------------
    # Section matching
    # ------------------------------------------------------------------
    def _match_sections(self) -> None:
        scores: dict[tuple[int, int], float] = {}
        evidence: dict[tuple[int, int], dict[str, float]] = {}
        for old in self.old_headings:
            for new in self.new_headings:
                score, parts = _features(
                    old,
                    new,
                    self._old_title_counts,
                    self._new_title_counts,
                )
                scores[id(old), id(new)] = score
                evidence[id(old), id(new)] = parts

        # Rank all plausible pairs.  Document order is never used as a
        # tie-breaker: two sections with the same title, the same parent,
        # and equally similar bodies stay ambiguous even if one appears
        # earlier in the page.
        ranked: dict[int, list[tuple[float, SectionHeadingInfo]]] = {}
        for old in self.old_headings:
            items = [
                (scores[id(old), id(new)], new)
                for new in self.new_headings
                if scores[id(old), id(new)] >= _MIN_SCORE
            ]
            items.sort(key=lambda item: item[0], reverse=True)
            ranked[id(old)] = items

        def equal_fit(
            old_a: SectionHeadingInfo,
            old_b: SectionHeadingInfo,
            new: SectionHeadingInfo,
        ) -> bool:
            """True when two old headings fit the same new heading equally."""
            pa = evidence[id(old_a), id(new)]
            pb = evidence[id(old_b), id(new)]
            return (
                pa['title'] == pb['title'] == 1.0
                and pa['parent'] == pb['parent'] == 1.0
                and abs(pa['body'] - pb['body']) < 1e-9
            )

        # Decide each old heading's own candidate set first.
        proposals: dict[int, SectionPair] = {}
        for old in self.old_headings:
            items = ranked[id(old)]
            if not items:
                continue
            top_score, top_new = items[0]
            tied = [
                new
                for score, new in items
                if top_score - score < _TIE_MARGIN
                or (
                    evidence[id(old), id(new)]['title']
                    == evidence[id(old), id(top_new)]['title']
                    == 1.0
                    and evidence[id(old), id(new)]['parent']
                    == evidence[id(old), id(top_new)]['parent']
                    == 1.0
                    and abs(
                        evidence[id(old), id(new)]['body']
                        - evidence[id(old), id(top_new)]['body']
                    )
                    < 1e-9
                )
            ]
            if len(tied) > 1:
                pair = SectionPair(
                    old, tied, 'ambiguous', evidence[id(old), id(tied[0])]
                )
            else:
                pair = SectionPair(
                    old,
                    [top_new],
                    'matched'
                    if old.base_anchor == top_new.base_anchor
                    else 'renamed',
                    evidence[id(old), id(top_new)],
                )
            proposals[id(old)] = pair

        # Each new heading may be proven for at most one old heading.
        by_new: dict[int, list[SectionPair]] = {}
        for pair in proposals.values():
            if pair.status != 'ambiguous' and pair.candidates:
                by_new.setdefault(id(pair.candidates[0]), []).append(pair)

        resolved: dict[int, SectionPair] = dict(proposals)
        for claimants in by_new.values():
            if len(claimants) == 1:
                continue
            claimants.sort(
                key=lambda pair: pair.evidence.get('score', 0.0),
                reverse=True,
            )
            top = claimants[0]
            top_score = top.evidence.get('score', 0.0)
            tied = [
                pair
                for pair in claimants
                if top_score - pair.evidence.get('score', 0.0) < _TIE_MARGIN
                or equal_fit(pair.old, top.old, top.candidates[0])
            ]
            if len(tied) > 1:
                # No way to tell the claimants apart: keep every candidate.
                for pair in tied:
                    pair.status = 'ambiguous'
                    pair.candidates = [top.candidates[0]]
                    resolved[id(pair.old)] = pair
                continue
            # A weaker claimant loses this new heading.  If it has no other
            # plausible heading, it was removed rather than renamed.
            for pair in claimants[1:]:
                alternatives = [
                    new
                    for score, new in ranked[id(pair.old)]
                    if id(new) != id(top.candidates[0]) and score >= _MIN_SCORE
                ]
                if alternatives:
                    pair.candidates = alternatives
                    pair.status = 'ambiguous'
                else:
                    pair.candidates = []
                    pair.status = 'removed'
                resolved[id(pair.old)] = pair

        # Reconcile ambiguity with proven matches.  A new heading that is
        # uniquely identified for one old section cannot also be a candidate
        # for another one.  Removing it may collapse an ambiguous pair to a
        # single remaining counterpart.
        proven_new = {
            id(pair.new): pair
            for pair in resolved.values()
            if pair.status in ('matched', 'renamed') and pair.new is not None
        }
        changed = True
        while changed:
            changed = False
            for pair in resolved.values():
                if pair.status != 'ambiguous':
                    continue
                filtered = [
                    candidate
                    for candidate in pair.candidates
                    if id(candidate) not in proven_new
                ]
                if len(filtered) == len(pair.candidates):
                    continue
                changed = True
                if not filtered:
                    pair.candidates = []
                    pair.status = 'removed'
                elif len(filtered) == 1:
                    candidate = filtered[0]
                    pair.candidates = [candidate]
                    pair.status = (
                        'matched'
                        if pair.old.base_anchor == candidate.base_anchor
                        else 'renamed'
                    )
                    proven_new[id(candidate)] = pair
                else:
                    pair.candidates = filtered

        for old in self.old_headings:
            pair = resolved.get(id(old))
            if pair is None:
                pair = SectionPair(old, [], 'removed', {'score': 0.0})
            self.section_pairs.append(pair)
        self._pairs_by_old = resolved

        # Keep old-revision document order for a stable, inspectable result.
        self.section_pairs.sort(key=lambda pair: pair.old.index)
        for pair in self.section_pairs:
            if pair.status == 'ambiguous':
                for candidate in pair.candidates:
                    self._ambiguous_new.add(id(candidate))
            elif pair.new is not None:
                self._pairs_by_new[id(pair.new)] = pair

    @property
    def matched(self) -> list[SectionPair]:
        """Pairs with exactly one identified new section."""
        return [
            pair
            for pair in self.section_pairs
            if pair.status in ('matched', 'renamed')
        ]

    @property
    def ambiguous(self) -> list[SectionPair]:
        """Old sections whose new identity cannot be proven."""
        return [
            pair for pair in self.section_pairs if pair.status == 'ambiguous'
        ]

    @property
    def removed(self) -> list[SectionPair]:
        """Old sections with no plausible counterpart."""
        return [
            pair for pair in self.section_pairs if pair.status == 'removed'
        ]

    @property
    def added(self) -> list[SectionHeadingInfo]:
        """New headings that are neither matched nor ambiguous candidates."""
        return [
            info
            for info in self.new_headings
            if id(info) not in self._pairs_by_new
            and id(info) not in self._ambiguous_new
        ]

    def pair_for_old_anchor(self, anchor: str) -> SectionPair | None:
        """Return the pair for an old-revision section anchor."""
        normalized = normalize_section_anchor(anchor)
        for pair in self.section_pairs:
            if pair.old.anchor is not None and (
                normalize_section_anchor(pair.old.anchor) == normalized
            ):
                return pair
        return None

    # ------------------------------------------------------------------
    # In-page link resolution
    # ------------------------------------------------------------------
    def resolve_section_link(self, fragment: str) -> LinkResolution:
        """Map an old-revision fragment to a new-revision anchor.

        The old fragment is first resolved to the old section it pointed at;
        that section's identity (not merely its title) determines the new
        anchor.  This prevents a link from attaching to an unrelated section
        that happens to share the title.
        """
        normalized = normalize_section_anchor(fragment)
        old_info = self._old_by_anchor.get(normalized)
        if old_info is None:
            # Some fragments may already have been written with the new
            # anchor, or may refer to an explicit non-heading anchor.
            if normalized in self._new_by_anchor:
                return LinkResolution(
                    fragment,
                    self._new_by_anchor[normalized].anchor,
                    'unchanged',
                    [self._new_by_anchor[normalized].anchor or ''],
                    reason='not an old heading; exact new anchor exists',
                )
            return LinkResolution(
                fragment,
                None,
                'unresolved',
                [],
                reason='no old section has this anchor',
            )
        pair = self._pairs_by_old.get(id(old_info))
        if pair is None:
            pair = self.pair_for_old_anchor(normalized)
        if pair is None:
            return LinkResolution(
                fragment,
                None,
                'unresolved',
                [],
                old_info.anchor,
                reason='old section is not part of the comparison',
            )
        if pair.status == 'ambiguous':
            return LinkResolution(
                fragment,
                None,
                'ambiguous',
                pair.candidate_anchors,
                old_info.anchor,
                reason='old section has several plausible counterparts',
            )
        if pair.status == 'removed':
            return LinkResolution(
                fragment,
                None,
                'unresolved',
                [],
                old_info.anchor,
                reason='old section was removed',
            )
        target = pair.new
        if target is None:
            return LinkResolution(
                fragment,
                None,
                'ambiguous',
                pair.candidate_anchors,
                old_info.anchor,
            )
        status: LinkStatus = (
            'unchanged' if target.anchor == old_info.anchor else 'resolved'
        )
        return LinkResolution(
            fragment,
            target.anchor,
            status,
            [target.anchor] if target.anchor else [],
            old_info.anchor,
        )

    def resolve_link(self, link) -> LinkResolution:
        """Resolve a ``WikiLink`` from the old revision."""
        fragment = link.fragment
        if fragment is None:
            return LinkResolution(
                '', None, 'unresolved', [], reason='link has no fragment'
            )
        return self.resolve_section_link(fragment)

    def resolve_links(self) -> dict[str, LinkResolution]:
        """Resolve every distinct section fragment used in the old revision."""
        results: dict[str, LinkResolution] = {}
        for link in self.old.wikilinks:
            fragment = link.fragment
            if fragment is None or fragment in results:
                continue
            results[fragment] = self.resolve_section_link(fragment)
        return results

    def __repr__(self) -> str:
        return (
            f'{type(self).__name__}(matched={len(self.matched)}, '
            f'ambiguous={len(self.ambiguous)}, '
            f'removed={len(self.removed)}, added={len(self.added)})'
        )


def compare_revisions(
    old: str | WikiText, new: str | WikiText
) -> RevisionComparison:
    """Compare two revisions and return a :class:`RevisionComparison`."""
    old_wikitext = old if isinstance(old, WikiText) else WikiText(old)
    new_wikitext = new if isinstance(new, WikiText) else WikiText(new)
    return RevisionComparison(old_wikitext, new_wikitext)
