from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from types import MappingProxyType
from typing import Any, Literal

LeadSide = Literal["left", "right"]

# The maximum number of same-label leads per session for which the exact
# assignment search is performed. Duplicate lead labels are rare, so this is
# intentionally small enough to keep the search exhaustive and predictable.
_MAX_GROUP_SIZE = 12
_NEGATIVE_INFINITY = float("-inf")


class LeadMatchingError(ValueError):
    """Raised when leads cannot be matched unambiguously or consistently."""


@dataclass(frozen=True)
class LeadRef:
    """
    Reference to one ordinary signal in one of the two sessions.

    Parameters
    ----------
    session : {"left", "right"}
        The session the lead belongs to.
    index : int
        The index in that session's ordinary signal sequence (the same indices
        used by :attr:`Edf.signals`).
    label : str
        The signal label.
    occurrence : int
        Zero-based occurrence of this label within the session. This
        distinguishes same-label leads without relying on physical order.
    """

    session: LeadSide
    index: int
    label: str
    occurrence: int

    @property
    def side(self) -> LeadSide:
        """Alias for :attr:`session`."""
        return self.session

    def __str__(self) -> str:
        return f"{self.session}[{self.index}] {self.label} (#{self.occurrence})"


@dataclass(frozen=True)
class LeadMetadata:
    """
    Metadata used as evidence when matching same-label leads.

    The fields mirror the EDF signal header fields. They can be obtained from a
    signal with :meth:`from_signal`.
    """

    label: str
    transducer_type: str
    physical_dimension: str
    physical_min: float
    physical_max: float
    digital_min: int
    digital_max: int
    prefiltering: str
    sampling_frequency: float

    @classmethod
    def from_signal(cls, signal: Any) -> LeadMetadata:
        """Extract lead metadata from an EDF/BDF signal without touching its data."""
        return cls(
            label=str(signal.label),
            transducer_type=str(signal.transducer_type),
            physical_dimension=str(signal.physical_dimension),
            physical_min=float(signal.physical_min),
            physical_max=float(signal.physical_max),
            digital_min=int(signal.digital_min),
            digital_max=int(signal.digital_max),
            prefiltering=str(signal.prefiltering),
            sampling_frequency=float(signal.sampling_frequency),
        )

    @property
    def physical_range(self) -> tuple[float, float]:
        """The physical range as ``(physical_min, physical_max)``."""
        return (self.physical_min, self.physical_max)

    @property
    def digital_range(self) -> tuple[int, int]:
        """The digital range as ``(digital_min, digital_max)``."""
        return (self.digital_min, self.digital_max)


@dataclass(frozen=True)
class LeadPair:
    """
    A one-to-one correspondence between a lead in the left and right session.

    ``LeadPair`` is also used for confirmed anchors. Instances passed as
    ``anchors`` are treated as hard evidence: the pair is always part of the
    result, even if metadata alone would have preferred a different assignment.
    """

    left: LeadRef
    right: LeadRef

    @property
    def label(self) -> str:
        """The shared label of the two leads."""
        return self.left.label

    def __str__(self) -> str:
        return f"{self.label}: {self.left} <-> {self.right}"


# A confirmed correspondence is semantically a pair. The alias keeps the
# public API readable for callers that think in terms of evidence.
LeadAnchor = LeadPair
Anchor = LeadPair


@dataclass(frozen=True)
class InterchangeableGroup:
    """
    A set of leads that still can be permuted without contradicting the evidence.

    A group contains all leads on both sessions that take part in the remaining
    ambiguity for one label. If the group has size ``n``, any of the ``n!``
    assignments between its left and right members is compatible with the
    metadata and anchors used to build the result.
    """

    label: str
    left: tuple[LeadRef, ...]
    right: tuple[LeadRef, ...]
    possible_pairs: tuple[LeadPair, ...]

    @property
    def leads(self) -> tuple[LeadRef, ...]:
        """All leads in the group, left side first."""
        return self.left + self.right

    @property
    def size(self) -> int:
        """Number of leads on each side of the group."""
        return len(self.left)

    @property
    def left_indices(self) -> tuple[int, ...]:
        """Indices of the left-session members."""
        return tuple(ref.index for ref in self.left)

    @property
    def right_indices(self) -> tuple[int, ...]:
        """Indices of the right-session members."""
        return tuple(ref.index for ref in self.right)

    def __str__(self) -> str:
        left_indices = ", ".join(str(index) for index in self.left_indices)
        right_indices = ", ".join(str(index) for index in self.right_indices)
        return (
            f"{self.label}: left[{left_indices}] <-> right[{right_indices}] "
            f"({self.size}! interchangeable assignments)"
        )


@dataclass(frozen=True)
class LeadMatching:
    """
    Result of matching leads across two sessions.

    ``pairs`` contains only correspondences that are forced by the available
    evidence. ``interchangeable`` contains the remaining ambiguity, grouped by
    label. In particular, with no metadata difference and no anchors, no pair
    is reported just because two signals happen to have the same position in
    the two files.

    The original recordings and signal arrays are never modified.
    """

    left: tuple[LeadRef, ...]
    right: tuple[LeadRef, ...]
    pairs: tuple[LeadPair, ...]
    interchangeable: tuple[InterchangeableGroup, ...]
    anchors: tuple[LeadPair, ...] = ()
    unmatched: tuple[LeadRef, ...] = ()
    metadata_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_METADATA_WEIGHTS)
    )
    _candidates: tuple[tuple[LeadRef, tuple[LeadRef, ...]], ...] = field(
        default=(), repr=False, compare=False
    )
    _matcher: LeadMatcher | None = field(default=None, repr=False, compare=False)

    @property
    def is_unique(self) -> bool:
        """Whether every same-label lead has a single forced correspondence."""
        return not self.interchangeable

    @property
    def left_refs(self) -> tuple[LeadRef, ...]:
        """References to the left session's ordinary signals."""
        return self.left

    @property
    def right_refs(self) -> tuple[LeadRef, ...]:
        """References to the right session's ordinary signals."""
        return self.right

    @property
    def unique_pairs(self) -> tuple[LeadPair, ...]:
        """Alias for :attr:`pairs`."""
        return self.pairs

    @property
    def ambiguous(self) -> tuple[InterchangeableGroup, ...]:
        """Alias for :attr:`interchangeable`."""
        return self.interchangeable

    @property
    def swappable(self) -> tuple[InterchangeableGroup, ...]:
        """Alias for :attr:`interchangeable`."""
        return self.interchangeable

    @property
    def possible_pairs(self) -> tuple[LeadPair, ...]:
        """All pairs that occur in at least one evidence-compatible assignment."""
        pairs = list(self.pairs)
        seen = set(pairs)
        for group in self.interchangeable:
            for pair in group.possible_pairs:
                if pair not in seen:
                    seen.add(pair)
                    pairs.append(pair)
        return tuple(
            sorted(pairs, key=lambda pair: (pair.left.index, pair.right.index))
        )

    def candidates_for(self, ref: LeadRef) -> tuple[LeadRef, ...]:
        """Return the leads that can still correspond to ``ref``."""
        for candidate, partners in self._candidates:
            if candidate == ref:
                return partners
        return ()

    def add_anchor(self, anchor: LeadPair | tuple[Any, Any]) -> LeadMatching:
        """Return a new matching with one additional confirmed anchor."""
        if self._matcher is None:  # pragma: no cover - defensive
            raise LeadMatchingError("This matching is not attached to a matcher")
        return self._matcher.add_anchor(anchor)

    def with_anchor(self, anchor: LeadPair | tuple[Any, Any]) -> LeadMatching:
        """Alias for :meth:`add_anchor`."""
        return self.add_anchor(anchor)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the result."""
        return {
            "unique": self.is_unique,
            "pairs": [
                {
                    "label": pair.label,
                    "left_index": pair.left.index,
                    "right_index": pair.right.index,
                }
                for pair in self.pairs
            ],
            "interchangeable": [
                {
                    "label": group.label,
                    "left_indices": list(group.left_indices),
                    "right_indices": list(group.right_indices),
                    "size": group.size,
                }
                for group in self.interchangeable
            ],
            "unmatched": [
                {
                    "session": ref.session,
                    "index": ref.index,
                    "label": ref.label,
                    "occurrence": ref.occurrence,
                }
                for ref in self.unmatched
            ],
        }

    def describe(self) -> str:
        """Return a compact human-readable description of the result."""
        lines: list[str] = []
        if self.pairs:
            lines.append("Unique pairs:")
            lines.extend(
                f"  {pair.label}: left[{pair.left.index}] -> right[{pair.right.index}]"
                for pair in self.pairs
            )
        if self.interchangeable:
            lines.append("Interchangeable:")
            lines.extend(f"  {group}" for group in self.interchangeable)
        if self.unmatched:
            lines.append("Unmatched:")
            lines.extend(
                f"  {ref.session}[{ref.index}] {ref.label}" for ref in self.unmatched
            )
        if not lines:
            return "No corresponding leads."
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.describe()


DEFAULT_METADATA_WEIGHTS: Mapping[str, float] = MappingProxyType(
    {
        "transducer_type": 1.0,
        "physical_dimension": 3.0,
        "prefiltering": 2.0,
        "sampling_frequency": 4.0,
        "physical_range": 2.0,
        "digital_range": 1.0,
    }
)


def _normalise_weights(
    metadata_weights: Mapping[str, float] | None,
) -> dict[str, float]:
    weights = dict(DEFAULT_METADATA_WEIGHTS)
    if metadata_weights is None:
        return weights
    unknown = set(metadata_weights) - set(DEFAULT_METADATA_WEIGHTS)
    if unknown:
        raise ValueError(f"Unknown metadata weight(s): {sorted(unknown)}")
    for key, weight in metadata_weights.items():
        converted = float(weight)
        if converted < 0:
            raise ValueError(f"Metadata weight for {key!r} must be non-negative")
        weights[key] = converted
    return weights


def _is_annotation_signal(signal: Any) -> bool:
    try:
        return bool(signal._is_annotation_signal)
    except AttributeError:
        return str(getattr(signal, "label", "")) in (
            "EDF Annotations",
            "BDF Annotations",
        )


def _signals_from(recording: Any) -> tuple[Any, ...]:
    if hasattr(recording, "signals"):
        signals = tuple(recording.signals)
    else:
        try:
            signals = tuple(recording)
        except TypeError as exc:  # pragma: no cover - defensive
            raise TypeError(
                "Expected an Edf/Bdf object or a sequence of signals"
            ) from exc
    return tuple(signal for signal in signals if not _is_annotation_signal(signal))


def _make_refs(signals: Sequence[Any], session: LeadSide) -> tuple[LeadRef, ...]:
    occurrences: dict[str, int] = defaultdict(int)
    refs: list[LeadRef] = []
    for index, signal in enumerate(signals):
        label = str(signal.label)
        occurrence = occurrences[label]
        occurrences[label] += 1
        refs.append(LeadRef(session, index, label, occurrence))
    return tuple(refs)


def _metadata_score(
    left: LeadMetadata, right: LeadMetadata, weights: Mapping[str, float]
) -> float:
    score = 0.0
    for field_name, weight in weights.items():
        if getattr(left, field_name) == getattr(right, field_name):
            score += weight
    return score


def _best_score(
    scores: Sequence[Sequence[float]],
    fixed: Mapping[int, int],
    forbidden: frozenset[tuple[int, int]] = frozenset(),
) -> float:
    n = len(scores)
    if n == 0:
        return 0.0
    if n > _MAX_GROUP_SIZE:
        raise LeadMatchingError(
            f"Too many same-label leads to match exactly: {n} > {_MAX_GROUP_SIZE}"
        )
    fixed_items = dict(fixed)
    used_columns: set[int] = set()
    for row, column in fixed_items.items():
        if row < 0 or row >= n or column < 0 or column >= n:
            return _NEGATIVE_INFINITY
        if column in used_columns:
            return _NEGATIVE_INFINITY
        used_columns.add(column)

    @cache
    def search(row: int, mask: int) -> float:
        if row == n:
            return 0.0
        if row in fixed_items:
            column = fixed_items[row]
            if mask & (1 << column) or (row, column) in forbidden:
                return _NEGATIVE_INFINITY
            rest = search(row + 1, mask | (1 << column))
            if rest == _NEGATIVE_INFINITY:
                return _NEGATIVE_INFINITY
            return scores[row][column] + rest
        best = _NEGATIVE_INFINITY
        for column in range(n):
            if mask & (1 << column) or (row, column) in forbidden:
                continue
            rest = search(row + 1, mask | (1 << column))
            if rest == _NEGATIVE_INFINITY:
                continue
            best = max(best, scores[row][column] + rest)
        return best

    result = search(0, 0)
    search.cache_clear()
    return result


def _scores_equal(first: float, second: float) -> bool:
    return math.isclose(first, second, rel_tol=1e-9, abs_tol=1e-12)


def _forced_score(
    scores: Sequence[Sequence[float]],
    fixed: Mapping[int, int],
    row: int,
    column: int,
) -> float:
    if row in fixed and fixed[row] != column:
        return _NEGATIVE_INFINITY
    if column in fixed.values():
        return _NEGATIVE_INFINITY
    forced = dict(fixed)
    forced[row] = column
    return _best_score(scores, forced)


def _coerce_ref(session: LeadSide, value: Any, refs: Sequence[LeadRef]) -> LeadRef:
    if isinstance(value, LeadRef):
        if value.session != session:
            raise LeadMatchingError(
                f"Expected a {session} lead, got {value.session} lead {value}"
            )
        if not 0 <= value.index < len(refs):
            raise LeadMatchingError(f"Lead index out of range: {value.index}")
        ref = refs[value.index]
        if ref.label != value.label or ref.occurrence != value.occurrence:
            raise LeadMatchingError(
                f"Lead {value} does not belong to this {session} session"
            )
        return ref
    if isinstance(value, bool):
        raise LeadMatchingError(f"Invalid lead identifier: {value!r}")
    if isinstance(value, int):
        if not 0 <= value < len(refs):
            raise LeadMatchingError(f"Lead index out of range: {value}")
        return refs[value]
    if isinstance(value, str):
        matches = [ref for ref in refs if ref.label == value]
        if not matches:
            raise LeadMatchingError(f"No signal with label {value!r}")
        if len(matches) > 1:
            raise LeadMatchingError(
                f"Ambiguous label {value!r} identifies indices "
                f"{[ref.index for ref in matches]}; use an index or (label, occurrence)"
            )
        return matches[0]
    if (
        isinstance(value, tuple)
        and len(value) == _PAIR_LENGTH
        and isinstance(value[0], str)
        and isinstance(value[1], int)
    ):
        matches = [ref for ref in refs if ref.label == value[0]]
        occurrence = value[1]
        if not 0 <= occurrence < len(matches):
            raise LeadMatchingError(
                f"Lead occurrence out of range for label {value[0]!r}: {occurrence}"
            )
        return matches[occurrence]
    raise LeadMatchingError(f"Invalid lead identifier: {value!r}")


def _normalise_anchor(
    anchor: LeadPair | Sequence[Any],
    left_refs: Sequence[LeadRef],
    right_refs: Sequence[LeadRef],
) -> LeadPair:
    if isinstance(anchor, LeadPair):
        left_value: Any = anchor.left
        right_value: Any = anchor.right
    elif isinstance(anchor, (tuple, list)) and len(anchor) == _PAIR_LENGTH:
        left_value, right_value = anchor
    else:
        raise LeadMatchingError(f"Invalid anchor: {anchor!r}")
    left = _coerce_ref("left", left_value, left_refs)
    right = _coerce_ref("right", right_value, right_refs)
    if left.label != right.label:
        raise LeadMatchingError(
            f"Anchor must connect equal labels, got {left.label!r} and {right.label!r}"
        )
    return LeadPair(left, right)


def _validate_anchor_uniqueness(anchors: Sequence[LeadPair]) -> None:
    left_seen: set[LeadRef] = set()
    right_seen: set[LeadRef] = set()
    for anchor in anchors:
        if anchor.left in left_seen:
            raise LeadMatchingError(f"Multiple anchors for left lead {anchor.left}")
        if anchor.right in right_seen:
            raise LeadMatchingError(f"Multiple anchors for right lead {anchor.right}")
        left_seen.add(anchor.left)
        right_seen.add(anchor.right)


_PAIR_LENGTH = 2


def _build_scores(
    left_metadata: Sequence[LeadMetadata],
    right_metadata: Sequence[LeadMetadata],
    weights: Mapping[str, float],
) -> list[list[float]]:
    return [
        [_metadata_score(left, right, weights) for right in right_metadata]
        for left in left_metadata
    ]


def _infer_possible_and_definite(
    scores: Sequence[Sequence[float]],
    fixed: Mapping[int, int],
    global_score: float,
) -> tuple[list[list[bool]], list[list[bool]]]:
    n = len(scores)
    possible = [[False] * n for _ in range(n)]
    definite = [[False] * n for _ in range(n)]
    fixed_columns = set(fixed.values())
    for row in range(n):
        for column in range(n):
            if row in fixed:
                if fixed[row] == column:
                    possible[row][column] = True
                    definite[row][column] = True
                continue
            if column in fixed_columns:
                continue
            if not _scores_equal(
                _forced_score(scores, fixed, row, column), global_score
            ):
                continue
            possible[row][column] = True
            without_score = _best_score(scores, fixed, frozenset({(row, column)}))
            if not _scores_equal(without_score, global_score):
                definite[row][column] = True
    return possible, definite


def _components(
    possible: Sequence[Sequence[bool]],
    definite: Sequence[Sequence[bool]],
) -> list[tuple[list[int], list[int]]]:
    n = len(possible)
    ambiguous_rows = [
        row for row in range(n) if not any(definite[row][column] for column in range(n))
    ]
    adjacency: dict[tuple[str, int], set[tuple[str, int]]] = defaultdict(set)
    for row in ambiguous_rows:
        for column in range(n):
            if possible[row][column]:
                adjacency[("left", row)].add(("right", column))
                adjacency[("right", column)].add(("left", row))

    components: list[tuple[list[int], list[int]]] = []
    visited: set[tuple[str, int]] = set()
    for start in sorted(adjacency):
        if start in visited:
            continue
        stack = [start]
        component: list[tuple[str, int]] = []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            stack.extend(adjacency[node] - visited)
        components.append(
            (
                sorted(index for session, index in component if session == "left"),
                sorted(index for session, index in component if session == "right"),
            )
        )
    return components


def _match_group_from_metadata(
    label: str,
    left_refs: Sequence[LeadRef],
    left_metadata: Sequence[LeadMetadata],
    right_refs: Sequence[LeadRef],
    right_metadata: Sequence[LeadMetadata],
    anchors: Sequence[LeadPair],
    weights: Mapping[str, float],
) -> tuple[
    list[LeadPair],
    list[InterchangeableGroup],
    dict[LeadRef, tuple[LeadRef, ...]],
]:
    n = len(left_refs)
    if n != len(right_refs):
        raise LeadMatchingError(
            f"Different number of {label!r} leads in the two sessions: "
            f"{n} and {len(right_refs)}"
        )
    if n == 0:  # pragma: no cover - callers only pass non-empty groups
        return [], [], {}

    left_position = {ref: index for index, ref in enumerate(left_refs)}
    right_position = {ref: index for index, ref in enumerate(right_refs)}
    fixed = {
        left_position[anchor.left]: right_position[anchor.right] for anchor in anchors
    }
    _validate_anchor_uniqueness(anchors)

    scores = _build_scores(left_metadata, right_metadata, weights)
    global_score = _best_score(scores, fixed)
    if global_score == _NEGATIVE_INFINITY:
        raise LeadMatchingError(f"No valid assignment for label {label!r}")

    possible, definite = _infer_possible_and_definite(scores, fixed, global_score)
    for row in range(n):
        if not any(possible[row]):
            raise LeadMatchingError(f"No valid assignment for label {label!r}")

    pairs = [
        LeadPair(left_refs[row], right_refs[column])
        for row in range(n)
        for column in range(n)
        if definite[row][column]
    ]
    groups = [
        InterchangeableGroup(
            label,
            tuple(left_refs[row] for row in component_rows),
            tuple(right_refs[column] for column in component_columns),
            tuple(
                LeadPair(left_refs[row], right_refs[column])
                for row in component_rows
                for column in component_columns
                if possible[row][column]
            ),
        )
        for component_rows, component_columns in _components(possible, definite)
    ]

    candidates: dict[LeadRef, tuple[LeadRef, ...]] = {}
    for row in range(n):
        candidates[left_refs[row]] = tuple(
            right_refs[column] for column in range(n) if possible[row][column]
        )
    for column in range(n):
        candidates[right_refs[column]] = tuple(
            left_refs[row] for row in range(n) if possible[row][column]
        )

    pairs.sort(key=lambda pair: (pair.left.index, pair.right.index))
    groups.sort(key=lambda group: (group.left_indices, group.right_indices))
    return pairs, groups, candidates


class LeadMatcher:
    """
    Match same-label leads across two sessions using metadata and anchors.

    Parameters
    ----------
    left, right : Edf | Bdf | Sequence[signal]
        The two sessions. Passing an :class:`Edf`/:class:`Bdf` uses its
        ordinary signals; passing a sequence uses that sequence directly.
    anchors : Iterable[LeadPair | tuple], default: ()
        Confirmed correspondences. Each anchor is either a :class:`LeadPair` or
        a ``(left, right)`` pair, where each side is a signal index, a unique
        label string, a ``(label, occurrence)`` tuple, or a :class:`LeadRef`.
        Anchors are hard evidence and always appear in the result.
    metadata_weights : Mapping[str, float] | None, default: None
        Optional weights for the metadata fields. Missing fields keep their
        default weight; unknown fields raise a :class:`ValueError`.

    Notes
    -----
    Matching never reorders, rewrites, or otherwise touches the signal data.
    The result only describes possible identities.
    """

    def __init__(
        self,
        left: Any,
        right: Any,
        *,
        anchors: Iterable[LeadPair | Sequence[Any]] = (),
        metadata_weights: Mapping[str, float] | None = None,
    ):
        self._left_signals = _signals_from(left)
        self._right_signals = _signals_from(right)
        self._left_refs = _make_refs(self._left_signals, "left")
        self._right_refs = _make_refs(self._right_signals, "right")
        self._left_metadata = tuple(
            LeadMetadata.from_signal(signal) for signal in self._left_signals
        )
        self._right_metadata = tuple(
            LeadMetadata.from_signal(signal) for signal in self._right_signals
        )
        self._weights = _normalise_weights(metadata_weights)
        self._raw_anchors = tuple(anchors)
        self._anchors = tuple(
            _normalise_anchor(anchor, self._left_refs, self._right_refs)
            for anchor in self._raw_anchors
        )
        _validate_anchor_uniqueness(self._anchors)

    @property
    def left_refs(self) -> tuple[LeadRef, ...]:
        """References to the left session's ordinary signals."""
        return self._left_refs

    @property
    def right_refs(self) -> tuple[LeadRef, ...]:
        """References to the right session's ordinary signals."""
        return self._right_refs

    @property
    def anchors(self) -> tuple[LeadPair, ...]:
        """The normalized confirmed anchors."""
        return self._anchors

    def with_anchor(self, anchor: LeadPair | Sequence[Any]) -> LeadMatcher:
        """Return a new matcher with one additional anchor."""
        return LeadMatcher(
            self._left_signals,
            self._right_signals,
            anchors=(*self._raw_anchors, anchor),
            metadata_weights=self._weights,
        )

    def add_anchor(self, anchor: LeadPair | Sequence[Any]) -> LeadMatching:
        """Return a new matching with one additional anchor."""
        return self.with_anchor(anchor).match()

    def match(self) -> LeadMatching:
        """Compute the matching and the remaining interchangeable sets."""
        left_by_label: dict[str, list[int]] = defaultdict(list)
        right_by_label: dict[str, list[int]] = defaultdict(list)
        for index, ref in enumerate(self._left_refs):
            left_by_label[ref.label].append(index)
        for index, ref in enumerate(self._right_refs):
            right_by_label[ref.label].append(index)

        anchors_by_label: dict[str, list[LeadPair]] = defaultdict(list)
        for anchor in self._anchors:
            anchors_by_label[anchor.left.label].append(anchor)

        pairs: list[LeadPair] = []
        groups: list[InterchangeableGroup] = []
        candidates: dict[LeadRef, tuple[LeadRef, ...]] = {}
        unmatched: list[LeadRef] = []

        for label in sorted(set(left_by_label) | set(right_by_label)):
            left_indices = left_by_label.get(label, [])
            right_indices = right_by_label.get(label, [])
            if not left_indices or not right_indices:
                unmatched.extend(self._left_refs[index] for index in left_indices)
                unmatched.extend(self._right_refs[index] for index in right_indices)
                continue
            left_refs = tuple(self._left_refs[index] for index in left_indices)
            right_refs = tuple(self._right_refs[index] for index in right_indices)
            left_metadata = tuple(self._left_metadata[index] for index in left_indices)
            right_metadata = tuple(
                self._right_metadata[index] for index in right_indices
            )
            group_pairs, group_groups, group_candidates = _match_group_from_metadata(
                label,
                left_refs,
                left_metadata,
                right_refs,
                right_metadata,
                tuple(anchors_by_label.get(label, [])),
                self._weights,
            )
            pairs.extend(group_pairs)
            groups.extend(group_groups)
            candidates.update(group_candidates)

        pairs.sort(key=lambda pair: (pair.left.index, pair.right.index))
        groups.sort(
            key=lambda group: (group.label, group.left_indices, group.right_indices)
        )
        unmatched.sort(key=lambda ref: (ref.session, ref.index))
        candidate_items = tuple(
            sorted(
                candidates.items(), key=lambda item: (item[0].session, item[0].index)
            )
        )
        return LeadMatching(
            left=self._left_refs,
            right=self._right_refs,
            pairs=tuple(pairs),
            interchangeable=tuple(groups),
            anchors=self._anchors,
            unmatched=tuple(unmatched),
            metadata_weights=MappingProxyType(dict(self._weights)),
            _candidates=candidate_items,
            _matcher=self,
        )


def match_leads(
    left: Any,
    right: Any,
    *,
    anchors: Iterable[LeadPair | Sequence[Any]] = (),
    metadata_weights: Mapping[str, float] | None = None,
) -> LeadMatching:
    """
    Match same-label leads across two EDF/BDF sessions.

    Parameters
    ----------
    left, right : Edf | Bdf | Sequence[signal]
        The two sessions to compare.
    anchors : Iterable[LeadPair | tuple], default: ()
        Confirmed correspondences, either as :class:`LeadPair` objects or
        ``(left, right)`` tuples. Each side can be an ordinary signal index, a
        unique label, a ``(label, occurrence)`` tuple, or a :class:`LeadRef`.
    metadata_weights : Mapping[str, float] | None, default: None
        Optional weights overriding :data:`DEFAULT_METADATA_WEIGHTS`.

    Returns
    -------
    LeadMatching
        The forced pairs and the remaining interchangeable groups. Signal data
        is never modified.

    Examples
    --------
    >>> result = match_leads(session_a, session_b, anchors=[(0, 1)])
    >>> result.pairs
    (...)
    >>> result.interchangeable
    (...)
    """
    return LeadMatcher(
        left, right, anchors=anchors, metadata_weights=metadata_weights
    ).match()
