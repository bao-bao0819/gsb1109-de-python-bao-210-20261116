"""Lossless migration of enum and flag states between firmware definition versions.

Firmware revisions rename enum members, add aliases, merge or split flag bits and add
values older parsers do not know. This module compares an old and a new definition and
classifies every old state instead of silently renumbering it:

``exact``
    The numeric value is preserved and every known old name survives in the new
    definition. A reverse migration reproduces the original value and known names.
``candidate``
    One or more new states are plausible, but choosing one changes the value or drops
    a known name. Candidates are reported for the caller to decide and are never
    applied automatically.
``rejected``
    No lossless or plausible mapping exists. Every rejection carries one concrete
    :class:`Conflict` with the old state and a real state from the new definition.
    Unnamed bits are never folded into a synthetic ``UNKNOWN`` value; the offending
    numeric value and bit mask are part of the evidence.

Inputs are ordinary ``{name: value}`` definitions, so definitions from any firmware
revision can be compared. A cstruct ``Enum``/``Flag`` class is accepted as well, which
keeps alias and bit members exactly as declared. Field values observed in firmware can
be supplied with ``observed``; they are treated as unnamed states and are either
matched by value in the new definition or rejected individually.

Flags are compared by member and bit-group coverage, not by copying bit positions. If a
new member covers several old bits (or an old member covers several new bits), the
result is classified as a candidate or a conflict and the lost bits are recorded.
``exclusive`` expresses caller-supplied mutual-exclusion constraints as groups of
member names. A state that sets two members of the same group is rejected with those
two members as evidence, and candidate values that would violate a constraint are
discarded.

Example::

    old = {"IDLE": 0, "RUN": 1, "HALT": 1, "LEGACY": 8}
    new = {"IDLE": 0, "RUNNING": 1, "STOPPED": 1}
    result = migrate(old, new)
    result.exact_values  # {0: 0}
    result.candidates_for(1)  # RUN/HALT names were lost at the same value
    result.conflicts_for(8)  # concrete new state used as evidence

    old = {"READ": 1, "WRITE": 2, "READ_WRITE": 3}
    new = {"READ": 1, "WRITE": 2}
    result = migrate(old, new, kind="flag", exclusive=[("READ", "WRITE")])
    result.conflicts_for(3)  # READ_WRITE violates READ/WRITE exclusion
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

__all__ = [
    "Conflict",
    "MigrationError",
    "MigrationResult",
    "State",
    "StateClassification",
    "StateMapping",
    "migrate",
    "migrate_enum",
    "migrate_flag",
    "verify_roundtrip",
]

_MAX_COMBINATIONS = 256


class MigrationError(ValueError):
    """Raised for invalid migration inputs."""


class StateClassification:
    """Classification labels returned by :meth:`MigrationResult.classify`."""

    EXACT = "exact"
    CANDIDATE = "candidate"
    REJECTED = "rejected"


@dataclass(frozen=True)
class State:
    """A numeric state and every name known for it in one domain."""

    value: int
    names: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        """The first known name, or an empty string for an unnamed state."""
        return self.names[0] if self.names else ""

    @property
    def named(self) -> bool:
        """Whether the state has at least one known name."""
        return bool(self.names)

    def __repr__(self) -> str:
        names = ", ".join(self.names) if self.names else "<unnamed>"
        return f"State(value={self.value:#x}, names=({names}))"


@dataclass(frozen=True)
class StateMapping:
    """A lossless or candidate mapping from an old state to a new state."""

    old: State
    new: State
    preserved_names: tuple[str, ...] = ()
    lost_names: tuple[str, ...] = ()
    lost_bits: int = 0
    reason: str = ""

    @property
    def value_changed(self) -> bool:
        """Whether the numeric value changes during this migration."""
        return self.old.value != self.new.value

    @property
    def lossless(self) -> bool:
        """Whether the value and all known names survive the migration."""
        return not self.value_changed and not self.lost_names and not self.lost_bits

    def __repr__(self) -> str:
        return (
            f"StateMapping(old={self.old.value:#x}, new={self.new.value:#x}, "
            f"preserved={self.preserved_names}, lost={self.lost_names}, "
            f"lost_bits={self.lost_bits:#x}, reason={self.reason!r})"
        )


@dataclass(frozen=True)
class Conflict:
    """Concrete evidence that an old state cannot be migrated.

    ``new`` is a real state from the new definition, not a synthetic ``UNKNOWN``.
    ``lost_names`` and ``lost_bits`` state exactly what the old state would lose.
    """

    old: State
    new: State | None
    lost_names: tuple[str, ...] = ()
    lost_bits: int = 0
    reason: str = ""

    def __repr__(self) -> str:
        new = "<none>" if self.new is None else f"{self.new.value:#x}({self.new.names})"
        return (
            f"Conflict(old={self.old.value:#x}({self.old.names}), new={new}, "
            f"lost_names={self.lost_names}, lost_bits={self.lost_bits:#x}, reason={self.reason!r})"
        )


@dataclass
class MigrationResult:
    """The complete classification of an old definition against a new one."""

    kind: str
    old_name: str
    new_name: str
    exact: list[StateMapping] = field(default_factory=list)
    candidates: list[StateMapping] = field(default_factory=list)
    rejected: list[Conflict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def exact_values(self) -> dict[int, int]:
        """The lossless ``old value -> new value`` mapping."""
        return {mapping.old.value: mapping.new.value for mapping in self.exact}

    @property
    def candidate_values(self) -> dict[int, list[int]]:
        """All candidate values per old value, preserving preference order."""
        result: dict[int, list[int]] = {}
        for mapping in self.candidates:
            values = result.setdefault(mapping.old.value, [])
            if mapping.new.value not in values:
                values.append(mapping.new.value)
        return result

    def classify(self, value: int) -> str | None:
        """Return the classification for an old numeric value."""
        if any(mapping.old.value == value for mapping in self.exact):
            return StateClassification.EXACT
        if any(mapping.old.value == value for mapping in self.candidates):
            return StateClassification.CANDIDATE
        if any(conflict.old.value == value for conflict in self.rejected):
            return StateClassification.REJECTED
        return None

    def candidates_for(self, value: int) -> list[StateMapping]:
        """Return candidate mappings for one old value without applying them."""
        return [mapping for mapping in self.candidates if mapping.old.value == value]

    def conflicts_for(self, value: int) -> list[Conflict]:
        """Return rejection evidence for one old value."""
        return [conflict for conflict in self.rejected if conflict.old.value == value]

    def apply_exact(self) -> dict[int, int]:
        """Return only the lossless mappings. Candidates are intentionally excluded."""
        return self.exact_values

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary suitable for logs or reports."""
        return {
            "kind": self.kind,
            "old": self.old_name,
            "new": self.new_name,
            "exact": [
                {"old": mapping.old.value, "new": mapping.new.value, "names": mapping.old.names}
                for mapping in self.exact
            ],
            "candidates": [
                {
                    "old": mapping.old.value,
                    "old_names": mapping.old.names,
                    "new": mapping.new.value,
                    "new_names": mapping.new.names,
                    "lost_names": mapping.lost_names,
                    "lost_bits": mapping.lost_bits,
                    "reason": mapping.reason,
                }
                for mapping in self.candidates
            ],
            "rejected": [
                {
                    "old": conflict.old.value,
                    "old_names": conflict.old.names,
                    "new": None if conflict.new is None else conflict.new.value,
                    "new_names": () if conflict.new is None else conflict.new.names,
                    "lost_names": conflict.lost_names,
                    "lost_bits": conflict.lost_bits,
                    "reason": conflict.reason,
                }
                for conflict in self.rejected
            ],
            "notes": self.notes,
        }

    def __bool__(self) -> bool:
        return bool(self.exact or self.candidates or self.rejected)


@dataclass
class _Domain:
    name: str
    kind: str
    names_by_value: dict[int, tuple[str, ...]]
    values_by_name: dict[str, tuple[int, ...]]


@dataclass
class _Groups:
    names: Sequence[set[str]]
    old_masks: list[list[tuple[str, int]]]
    new_masks: list[list[tuple[str, int]]]


def _as_items(definitions: Mapping[str, int] | Iterable[State] | type) -> list[tuple[str, int]]:
    if isinstance(definitions, Mapping):
        return [(str(name), value) for name, value in definitions.items()]

    if hasattr(definitions, "__members__"):
        return [(str(name), int(member.value)) for name, member in definitions.__members__.items()]

    items: list[tuple[str, int]] = []
    for state in definitions:
        if isinstance(state, State):
            if state.names:
                items.extend((name, state.value) for name in state.names)
            else:
                items.append(("", state.value))
        elif isinstance(state, tuple) and len(state) == 2:
            items.append((str(state[0]), int(state[1])))
        else:
            raise MigrationError(f"Unsupported state definition: {state!r}")
    return items


def _make_domain(definitions: Mapping[str, int] | Iterable[State] | type, name: str, kind: str) -> _Domain:
    names_by_value: dict[int, list[str]] = {}
    values_by_name: dict[str, list[int]] = {}
    seen: dict[str, int] = {}

    for member_name, value in _as_items(definitions):
        if isinstance(value, bool) or not isinstance(value, int):
            raise MigrationError(f"Value for {member_name!r} must be an integer, got {type(value).__name__}")
        if value < 0 and kind == "flag":
            raise MigrationError(f"Flag value for {member_name!r} must not be negative")

        if not member_name:
            names_by_value.setdefault(value, [])
            continue

        if member_name in seen and seen[member_name] != value:
            raise MigrationError(f"Member {member_name!r} has conflicting values {seen[member_name]} and {value}")
        seen[member_name] = value

        value_names = names_by_value.setdefault(value, [])
        if member_name not in value_names:
            value_names.append(member_name)

        name_values = values_by_name.setdefault(member_name, [])
        if value not in name_values:
            name_values.append(value)

    return _Domain(
        name=name,
        kind=kind,
        names_by_value={value: tuple(names) for value, names in sorted(names_by_value.items())},
        values_by_name={member_name: tuple(values) for member_name, values in values_by_name.items()},
    )


def _normalise_kind(kind: str | type | None, old: object, new: object) -> str:
    if kind is None:
        for definition in (old, new):
            if isinstance(definition, type) and any(base.__name__ == "Flag" for base in definition.__mro__):
                return "flag"
        return "enum"

    if isinstance(kind, str):
        normalised = kind.lower()
    elif isinstance(kind, type):
        normalised = "flag" if any(base.__name__ == "Flag" for base in kind.__mro__) else "enum"
    else:
        raise MigrationError(f"Unsupported migration kind: {kind!r}")

    if normalised not in ("enum", "flag"):
        raise MigrationError(f"Unsupported migration kind: {kind!r}")
    return normalised


def _exclusive_groups(exclusive: Sequence[Sequence[str]] | Mapping[str, Sequence[str]]) -> list[set[str]]:
    if not exclusive:
        return []

    groups: list[set[str]] = []
    iterable = exclusive.values() if isinstance(exclusive, Mapping) else exclusive
    for group in iterable:
        names = {str(name) for name in group}
        if len(names) < 2:
            raise MigrationError(f"Exclusive group must contain at least two members: {group!r}")
        groups.append(names)
    return groups


def _resolve_groups(
    groups: Sequence[set[str]],
    old: _Domain,
    new: _Domain,
) -> _Groups:
    old_masks: list[list[tuple[str, int]]] = []
    new_masks: list[list[tuple[str, int]]] = []

    for group in groups:
        unknown = sorted(name for name in group if name not in old.values_by_name and name not in new.values_by_name)
        if unknown:
            raise MigrationError(f"Exclusive group contains unknown member(s): {', '.join(unknown)}")

        resolved_old = [(name, old.values_by_name[name][0]) for name in group if name in old.values_by_name]
        resolved_new = [(name, new.values_by_name[name][0]) for name in group if name in new.values_by_name]
        if not resolved_old and not resolved_new:
            raise MigrationError(f"Exclusive group contains no known member: {sorted(group)}")
        old_masks.append(resolved_old)
        new_masks.append(resolved_new)

    return _Groups(names=groups, old_masks=old_masks, new_masks=new_masks)


def _group_violation(masks: Sequence[Sequence[tuple[str, int]]], value: int) -> tuple[str, str] | None:
    for entries in masks:
        active = [(name, mask) for name, mask in entries if mask and mask & value == mask]
        for (left_name, left_mask), (right_name, right_mask) in itertools.combinations(active, 2):
            if left_mask != right_mask:
                return left_name, right_name
    return None


def _group_violation_for(groups: _Groups, domain: str, value: int) -> tuple[str, str] | None:
    masks = groups.old_masks if domain == "old" else groups.new_masks
    return _group_violation(masks, value)


def _enum_names(domain: _Domain, value: int) -> tuple[str, ...]:
    return domain.names_by_value.get(value, ())


def _flag_names(domain: _Domain, value: int) -> tuple[str, ...]:
    """Return every known member name contained in a flag value.

    Each member whose bits are fully set is a known name of the state, including
    aliases and combined members. For ``A = 1, B = 2, AB = 3`` the value 3 therefore
    knows ``A``, ``B`` and ``AB``; losing any of them means the state is not lossless.
    """
    if value == 0:
        return domain.names_by_value.get(0, ())

    names: list[str] = []
    for member_value, member_names in domain.names_by_value.items():
        if member_value != 0 and member_value & value == member_value:
            names.extend(name for name in member_names if name not in names)
    return tuple(names)


def _covered_bits(domain: _Domain, value: int) -> int:
    covered = 0
    for member_value in domain.names_by_value:
        if member_value != 0 and member_value & value == member_value:
            covered |= member_value
    return covered


def _unknown_bits(domain: _Domain, value: int) -> int:
    if value == 0:
        return 0
    return value & ~_covered_bits(domain, value)


def _nearest_state(domain: _Domain, value: int, *, prefer_newer: bool = False) -> State | None:
    if value in domain.names_by_value:
        return State(value, domain.names_by_value[value])
    if not domain.names_by_value:
        return None

    candidates = [candidate for candidate in domain.names_by_value if candidate != value]
    if not candidates:
        candidates = list(domain.names_by_value)
    tie_break = (lambda candidate: (candidate < value, candidate)) if prefer_newer else (lambda candidate: candidate)
    nearest = min(candidates, key=lambda candidate: (abs(candidate - value), tie_break(candidate)))
    return State(nearest, domain.names_by_value[nearest])


def _evidence_state(new: _Domain, value: int) -> State | None:
    """Pick a concrete new state as rejection evidence.

    Prefer a state at the same value, then the new state whose member bits overlap the
    rejected value the most, then the numerically nearest state. The returned state is
    always a real member from the new definition.
    """
    if value in new.names_by_value:
        return State(value, new.names_by_value[value])

    best: tuple[int, int, int] | None = None
    best_state: State | None = None
    for member_value, member_names in new.names_by_value.items():
        score = (member_value & value).bit_count()
        distance = abs(member_value - value)
        rank = (score, -distance, -member_value)
        if best is None or rank > best:
            best = rank
            best_state = State(member_value, member_names)

    if best_state is not None and best is not None and best[0] > 0:
        return best_state

    return _nearest_state(new, value, prefer_newer=True)


def _enum_mapping(
    value: int, old: _Domain, new: _Domain
) -> tuple[StateMapping | None, list[StateMapping], Conflict | None]:
    old_names = _enum_names(old, value)
    old_state = State(value, old_names)

    candidates: list[StateMapping] = []
    if value in new.names_by_value:
        new_names = _enum_names(new, value)
        if set(old_names) <= set(new_names):
            return StateMapping(old_state, State(value, new_names), old_names, (), 0, "same numeric value"), [], None

        candidates.append(
            StateMapping(
                old_state,
                State(value, new_names),
                tuple(name for name in old_names if name in new_names),
                tuple(name for name in old_names if name not in new_names),
                0,
                "same numeric value, but known names differ",
            )
        )

    for candidate_value, candidate_names in new.names_by_value.items():
        if candidate_value == value:
            continue
        shared = tuple(name for name in old_names if name in candidate_names)
        if not shared:
            continue
        candidates.append(
            StateMapping(
                old_state,
                State(candidate_value, candidate_names),
                shared,
                tuple(name for name in old_names if name not in candidate_names),
                0,
                f"value changed but name(s) survived: {', '.join(shared)}",
            )
        )

    if candidates:
        candidates.sort(key=lambda mapping: (mapping.new.value != value, mapping.new.value))
        return None, candidates, None

    evidence = _evidence_state(new, value)
    if not old_names:
        reason = "unnamed value has no matching value in the new definition; it is not mapped to UNKNOWN"
    else:
        reason = "no member with the same value or a shared name in the new definition"

    return (
        None,
        [],
        Conflict(old_state, evidence, old_names, 0, reason),
    )


def _compose(
    names: Sequence[str],
    target: _Domain,
    groups: _Groups,
) -> tuple[set[int], tuple[str, ...]]:
    """Compose target values from names without copying bit positions."""
    choices: list[tuple[int, ...]] = []
    missing: list[str] = []

    for name in dict.fromkeys(names):
        values = target.values_by_name.get(name, ())
        if not values:
            missing.append(name)
        else:
            choices.append(values)

    if missing or not choices:
        return set(), tuple(missing)

    result: set[int] = set()
    for combination in itertools.islice(itertools.product(*choices), _MAX_COMBINATIONS):
        value = 0
        for part in combination:
            value |= part
        if _group_violation_for(groups, "new", value) is None:
            result.add(value)

    return result, ()


def _name_sets(domain: _Domain, value: int, names: Sequence[str]) -> list[tuple[tuple[str, ...], str]]:
    """Return independent name interpretations of an old flag value.

    Exact members define one interpretation. A multi-bit member also gets an
    interpretation by its constituent bits, so an old ``AB = 3`` can be compared with a
    new domain that defines ``A = 1`` and ``B = 2`` separately. Values without an exact
    member are interpreted by a minimal cover of their member bits.
    """
    exact = domain.names_by_value.get(value, ())
    if exact:
        result = [((name,), "exact member name") for name in exact]

        exact_masks = {
            member_value
            for member_value, member_names in domain.names_by_value.items()
            if member_names and member_value == value
        }
        constituent = _minimal_cover_names(domain, value, exclude=exact_masks)
        if constituent and constituent != set(exact):
            result.append((tuple(sorted(constituent)), "constituent member bits"))
        return result

    cover = _minimal_cover_names(domain, value)
    return [(tuple(sorted(cover)), "minimal member cover")] if cover else []


def _minimal_cover_names(domain: _Domain, value: int, *, exclude: Iterable[int] = ()) -> set[str]:
    """Return a minimal set of member names whose masks exactly cover ``value``."""
    excluded = set(exclude)
    members = [
        (member_value, member_names)
        for member_value, member_names in domain.names_by_value.items()
        if member_names and member_value not in excluded and 0 < member_value & value == member_value
    ]
    members.sort(key=lambda item: (-item[0], item[1]))

    selected: set[str] = set()
    remaining = value
    for member_value, member_names in members:
        if remaining & member_value != member_value:
            continue
        selected.add(member_names[0])
        remaining &= ~member_value
        if remaining == 0:
            return selected

    return set()


def _direct_flag_names(domain: _Domain, value: int) -> set[str]:
    """Return the names that directly describe a flag state.

    Aliases of an exactly defined value are direct names. For a combination without an
    exact member, the minimal non-overlapping member cover is used.
    """
    exact = domain.names_by_value.get(value, ())
    if exact:
        return set(exact)
    if value == 0:
        return set()
    return _minimal_cover_names(domain, value)


def _candidate_values(
    value: int,
    old: _Domain,
    new: _Domain,
    names: Sequence[str],
    groups: _Groups,
) -> dict[int, str]:
    """Return plausible candidate values for an old flag state."""
    candidates: dict[int, str] = {}
    direct = _direct_flag_names(old, value)

    def add(candidate_value: int, reason: str, *, require_direct: bool = True) -> None:
        if _group_violation_for(groups, "new", candidate_value) is not None:
            return
        if _unknown_bits(new, candidate_value):
            return

        candidate_names = set(_flag_names(new, candidate_value))
        if require_direct and not direct <= candidate_names:
            return
        candidates.setdefault(candidate_value, reason)

    if names:
        for name_set, label in _name_sets(old, value, names):
            composed, _ = _compose(name_set, new, groups)
            for composed_value in composed:
                add(composed_value, f"composed from {label}")

        for name in names:
            for candidate_value in new.values_by_name.get(name, ()):
                add(candidate_value, f"shared member name {name!r}")

    for member_value, member_names in new.names_by_value.items():
        if not member_value or member_value == value:
            continue
        if member_value & value == member_value:
            add(
                member_value,
                f"new member(s) {', '.join(member_names)} cover fewer bits than the old state; "
                "bit positions are not copied one-for-one",
            )
        elif value & member_value == value:
            add(
                member_value,
                f"new member(s) {', '.join(member_names)} cover more bits than the old state; "
                "bit positions are not copied one-for-one",
            )

    if value in new.names_by_value or _covered_bits(new, value) == value:
        add(value, "numeric value preserved; known names differ; semantics require confirmation", require_direct=False)

    return candidates


def _mapping_from_candidates(
    value: int,
    old: _Domain,
    new: _Domain,
    names: Sequence[str],
    candidate_values: Mapping[int, str],
) -> list[StateMapping]:
    old_state = State(value, names)
    mappings: list[StateMapping] = []

    for candidate_value, reason in sorted(candidate_values.items(), key=lambda item: (item[0] != value, item[0])):
        candidate_names = _flag_names(new, candidate_value)
        lost_bits = value & ~_covered_bits(new, candidate_value)
        mappings.append(
            StateMapping(
                old_state,
                State(candidate_value, candidate_names),
                tuple(name for name in names if name in candidate_names),
                tuple(name for name in names if name not in candidate_names),
                lost_bits,
                reason,
            )
        )

    return mappings


def _flag_mapping(
    value: int,
    old: _Domain,
    new: _Domain,
    groups: _Groups,
) -> tuple[StateMapping | None, list[StateMapping], Conflict | None]:
    names = _flag_names(old, value)
    old_state = State(value, names)
    unknown = _unknown_bits(old, value)

    old_violation = _group_violation_for(groups, "old", value) if names else None
    if old_violation is not None:
        return (
            None,
            [],
            Conflict(
                old_state,
                _evidence_state(new, value),
                names,
                unknown,
                f"old state sets mutually exclusive members {old_violation[0]!r} and {old_violation[1]!r}",
            ),
        )

    new_violation = _group_violation_for(groups, "new", value)
    new_unknown = _unknown_bits(new, value)
    if unknown and new_unknown:
        return (
            None,
            [],
            Conflict(
                old_state,
                _evidence_state(new, value),
                names,
                unknown & new_unknown,
                f"unnamed bit(s) {unknown & new_unknown:#x} have no matching state; refusing to substitute UNKNOWN",
            ),
        )

    new_names = _flag_names(new, value)
    if new_violation is None and set(names) <= set(new_names):
        return (
            StateMapping(old_state, State(value, new_names), names, (), 0, "same numeric value"),
            [],
            None,
        )

    candidate_values = _candidate_values(value, old, new, names, groups)
    candidates = _mapping_from_candidates(value, old, new, names, candidate_values)
    if candidates:
        return None, candidates, None

    if new_violation is not None:
        reason = (
            f"same value sets mutually exclusive members {new_violation[0]!r} and {new_violation[1]!r} "
            "in the new definition"
        )
    elif unknown:
        reason = f"unnamed bit(s) {unknown:#x} are not defined by the new definition either"
    elif names:
        missing = [name for name in names if name not in new.values_by_name]
        if missing:
            reason = f"known member name(s) absent from the new definition: {', '.join(missing)}"
        else:
            reason = "known member names cannot be composed into a state accepted by the new definition"
    else:
        reason = "unnamed state has no matching value in the new definition; it is not mapped to UNKNOWN"

    return (
        None,
        [],
        Conflict(old_state, _evidence_state(new, value), names, unknown, reason),
    )


def migrate_enum(
    old: Mapping[str, int] | Iterable[State] | type,
    new: Mapping[str, int] | Iterable[State] | type,
    *,
    old_name: str = "old",
    new_name: str = "new",
    observed: Iterable[int] = (),
) -> MigrationResult:
    """Classify enum states between two definition versions.

    Enums are atomic: values are compared as a whole, never decomposed into bits.
    Duplicate values are aliases; every alias name must survive for a state to be
    lossless. Values in ``observed`` that are not in the old definition are unnamed
    states: they keep their own numeric value and are never collapsed into one
    ``UNKNOWN`` member.
    """
    old_domain = _make_domain(old, old_name, "enum")
    new_domain = _make_domain(new, new_name, "enum")

    result = MigrationResult("enum", old_name, new_name)
    values = set(old_domain.names_by_value)
    values.update(int(value) for value in observed)

    for value in sorted(values):
        exact, candidates, conflict = _enum_mapping(value, old_domain, new_domain)
        if exact is not None:
            result.exact.append(exact)
        result.candidates.extend(candidates)
        if conflict is not None:
            result.rejected.append(conflict)

    return result


def migrate_flag(
    old: Mapping[str, int] | Iterable[State] | type,
    new: Mapping[str, int] | Iterable[State] | type,
    *,
    old_name: str = "old",
    new_name: str = "new",
    exclusive: Sequence[Sequence[str]] | Mapping[str, Sequence[str]] = (),
    observed: Iterable[int] = (),
) -> MigrationResult:
    """Classify flag states between two definition versions.

    Flags use member and bit-group coverage. A new member that covers several old bits
    is matched by name or by bit coverage, never by copying positions one-for-one.

    ``exclusive`` is a sequence of member-name groups (or a mapping of group names to
    member names). Members in one group may not be set together. A violating old state
    is rejected, and a candidate that would set two members of a group is discarded.

    ``observed`` may contain combined flag values from firmware data, including values
    that mix known members with unnamed bits. Values with unnamed bits are rejected
    individually with the bit mask as evidence.
    """
    old_domain = _make_domain(old, old_name, "flag")
    new_domain = _make_domain(new, new_name, "flag")
    groups = _resolve_groups(_exclusive_groups(exclusive), old_domain, new_domain)

    result = MigrationResult("flag", old_name, new_name)
    values = set(old_domain.names_by_value)
    values.update(int(value) for value in observed)

    for value in sorted(values):
        exact, candidates, conflict = _flag_mapping(value, old_domain, new_domain, groups)
        if exact is not None:
            result.exact.append(exact)
        result.candidates.extend(candidates)
        if conflict is not None:
            result.rejected.append(conflict)

    return result


def migrate(
    old: Mapping[str, int] | Iterable[State] | type,
    new: Mapping[str, int] | Iterable[State] | type,
    *,
    kind: str | type | None = None,
    old_name: str = "old",
    new_name: str = "new",
    exclusive: Sequence[Sequence[str]] | Mapping[str, Sequence[str]] = (),
    observed: Iterable[int] = (),
) -> MigrationResult:
    """Classify states between two definitions, dispatching to enum or flag handling.

    ``kind`` may be ``"enum"``, ``"flag"`` or a cstruct ``Enum``/``Flag`` class. When
    omitted it is inferred from the definition classes, defaulting to ``"enum"``.
    """
    kind_name = _normalise_kind(kind, old, new)
    if kind_name == "flag":
        return migrate_flag(
            old,
            new,
            old_name=old_name,
            new_name=new_name,
            exclusive=exclusive,
            observed=observed,
        )
    return migrate_enum(old, new, old_name=old_name, new_name=new_name, observed=observed)


def verify_roundtrip(
    result: MigrationResult,
    old: Mapping[str, int] | Iterable[State] | type,
    new: Mapping[str, int] | Iterable[State] | type,
    *,
    new_parser: Callable[[int], Any] | None = None,
    old_parser: Callable[[int], Any] | None = None,
    include_candidates: bool = False,
) -> list[str]:
    """Verify migrated states against original values and known names.

    Exact mappings must reproduce the old numeric value, retain every known old name
    and (when supplied) parse correctly through both domains. Candidate mappings are
    skipped by default because the caller has not chosen one; with
    ``include_candidates`` they are checked too.

    ``new_parser``/``old_parser`` may be cstruct type classes such as ``cs.NewEnum``
    and ``cs.OldEnum``. Parsed values are compared using their ``value`` attribute.
    When no parsers are supplied, cstruct classes passed as ``old``/``new`` are used as
    parsers automatically, so alias names are checked through the real enum classes.

    Returns readable failures. An empty list means verification succeeded.
    """
    old_parser = old_parser or (old if isinstance(old, type) else None)
    new_parser = new_parser or (new if isinstance(new, type) else None)

    old_domain = _make_domain(old, result.old_name, result.kind)
    new_domain = _make_domain(new, result.new_name, result.kind)
    failures: list[str] = []

    mappings = list(result.exact)
    if include_candidates:
        mappings.extend(result.candidates)

    for mapping in mappings:
        old_value = mapping.old.value
        new_value = mapping.new.value
        old_names = _enum_names(old_domain, old_value) if result.kind == "enum" else _flag_names(old_domain, old_value)
        new_names = _enum_names(new_domain, new_value) if result.kind == "enum" else _flag_names(new_domain, new_value)

        if mapping in result.exact:
            if new_value != old_value:
                failures.append(f"exact mapping {old_value:#x} produced {new_value:#x}")
            if mapping.lost_names:
                failures.append(f"exact mapping {old_value:#x} lost names: {', '.join(mapping.lost_names)}")
            if mapping.lost_bits:
                failures.append(f"exact mapping {old_value:#x} lost bits: {mapping.lost_bits:#x}")

        missing = [name for name in old_names if name not in new_names]
        if missing:
            failures.append(f"state {old_value:#x} lost known names in {new_value:#x}: {', '.join(missing)}")

        if new_value != old_value:
            failures.append(f"state {old_value:#x} changed to {new_value:#x}")
        else:
            reverse = _reverse_value(result.kind, new_value, new_domain, old_domain, old_value)
            if reverse != old_value:
                failures.append(f"reverse migration of {new_value:#x} returned {reverse!r}, expected {old_value:#x}")

        if new_parser is not None:
            parsed_new = new_parser(new_value)
            parsed_new_value = getattr(parsed_new, "value", parsed_new)
            if parsed_new_value != new_value:
                failures.append(f"new parser returned {parsed_new_value!r}, expected {new_value:#x}")

        if old_parser is not None:
            parsed_old = old_parser(old_value)
            parsed_old_value = getattr(parsed_old, "value", parsed_old)
            if parsed_old_value != old_value:
                failures.append(f"old parser returned {parsed_old_value!r}, expected {old_value:#x}")

    return failures


def _reverse_value(kind: str, value: int, new: _Domain, old: _Domain, original: int) -> int | None:
    if value == original:
        return value

    if kind == "enum":
        names = _enum_names(old, original)
        for candidate_value in old.names_by_value:
            if names and set(names) <= set(_enum_names(old, candidate_value)):
                return candidate_value
        return None

    names = _flag_names(new, value)
    choices = [old.values_by_name[name] for name in names if name in old.values_by_name]
    if not choices:
        return None

    possible: set[int] = set()
    for combination in itertools.islice(itertools.product(*choices), _MAX_COMBINATIONS):
        composed = 0
        for part in combination:
            composed |= part
        possible.add(composed)

    return original if original in possible else (min(possible) if possible else None)
