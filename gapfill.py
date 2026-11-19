"""Exhaustive gap-completion solver for truncated binary samples.

A binary *sample* is missing a small segment (the *gap*).  The caller
supplies a finite list of candidate byte strings for the missing segment
plus a set of structural validation conditions.  This module splices every
candidate into the sample, re-parses the *whole* resulting buffer (because a
candidate whose length differs from the gap shifts every field that follows
it), evaluates every validation condition, and then aggregates the results:

* the field values shared by *all* legal completions (``common``);
* the still-possible domain of every field (``domains``);
* per-candidate evidence, kept for audit (``evidence``);
* when no completion is legal, the minimal sets of mutually conflicting
  conditions (``conflicts``).

The solver never short-circuits on the first buffer that merely parses: a
completion is only legal when it parses *and* satisfies every condition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable, Mapping, Sequence

__all__ = [
    "Gap",
    "Candidate",
    "ConstraintSpec",
    "CandidateEvidence",
    "FieldDomain",
    "ConflictReport",
    "SolveResult",
    "solve",
    "format_report",
]

# A parser turns a complete buffer into a field mapping, or raises.
Parser = Callable[[bytes], Mapping[str, Any]]
# A condition checks a parsed field mapping; returning False (or raising)
# means the completion violates the condition.
Condition = Callable[[Mapping[str, Any]], bool]


@dataclass(frozen=True)
class Gap:
    """Half-open byte range ``[start, end)`` of the missing segment."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid gap range [{self.start}, {self.end})")


@dataclass(frozen=True)
class Candidate:
    """One proposed byte string for the missing segment."""

    label: str
    data: bytes


@dataclass(frozen=True)
class ConstraintSpec:
    """A named structural validation condition."""

    name: str
    check: Condition
    description: str = ""


@dataclass
class CandidateEvidence:
    """Audit trail for a single candidate."""

    candidate: Candidate
    buffer: bytes
    parse_error: str | None = None
    fields: dict[str, Any] | None = None
    # condition name -> (passed, note); note is empty on success
    condition_results: dict[str, tuple[bool, str]] = field(default_factory=dict)

    @property
    def parsed(self) -> bool:
        return self.fields is not None

    @property
    def failed_conditions(self) -> list[str]:
        return [name for name, (ok, _) in self.condition_results.items() if not ok]

    @property
    def valid(self) -> bool:
        return self.parsed and not self.failed_conditions


@dataclass
class FieldDomain:
    """The values a field still may take across all legal completions."""

    field: str
    values: list[Any]

    @property
    def is_fixed(self) -> bool:
        return len(self.values) == 1

    @property
    def common(self) -> Any:
        return self.values[0] if self.is_fixed else None

    @property
    def is_numeric(self) -> bool:
        return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in self.values)

    @property
    def minimum(self) -> Any:
        return min(self.values) if self.is_numeric else None

    @property
    def maximum(self) -> Any:
        return max(self.values) if self.is_numeric else None


@dataclass
class ConflictReport:
    """A minimal set of conditions that no parseable candidate satisfies jointly."""

    conditions: tuple[str, ...]
    # condition name -> labels of parseable candidates passing it
    satisfiers: dict[str, list[str]]


@dataclass
class SolveResult:
    evidence: list[CandidateEvidence]
    solutions: list[CandidateEvidence]
    common: dict[str, Any]
    domains: dict[str, FieldDomain]
    conflicts: list[ConflictReport]
    unparseable: list[CandidateEvidence]

    @property
    def solved(self) -> bool:
        return bool(self.solutions)


def _splice(sample: bytes, gap: Gap, data: bytes) -> bytes:
    """Replace the gap with ``data``; everything after the gap shifts."""
    if gap.end > len(sample):
        raise ValueError(f"gap [{gap.start}, {gap.end}) exceeds sample of {len(sample)} bytes")
    return sample[: gap.start] + data + sample[gap.end :]


def _distinct(values: list[Any]) -> list[Any]:
    out: list[Any] = []
    for value in values:
        if not any(value == seen for seen in out):
            out.append(value)
    try:
        return sorted(out)
    except TypeError:
        return sorted(out, key=repr)


def _evaluate(
    sample: bytes,
    gap: Gap,
    parser: Parser,
    conditions: Sequence[ConstraintSpec],
    candidate: Candidate,
) -> CandidateEvidence:
    buffer = _splice(sample, gap, candidate.data)
    evidence = CandidateEvidence(candidate=candidate, buffer=buffer)
    try:
        parsed = parser(buffer)
    except Exception as exc:  # parse failures are evidence, not crashes
        evidence.parse_error = f"{type(exc).__name__}: {exc}"
        return evidence
    evidence.fields = dict(parsed)
    for spec in conditions:
        try:
            ok = bool(spec.check(evidence.fields))
            note = ""
        except Exception as exc:  # a condition that cannot run counts as failed
            ok, note = False, f"{type(exc).__name__}: {exc}"
        evidence.condition_results[spec.name] = (ok, note)
    return evidence


def _minimal_conflicts(
    conditions: Sequence[ConstraintSpec],
    parseable: list[CandidateEvidence],
) -> list[ConflictReport]:
    """Minimal condition sets whose satisfier sets do not intersect."""
    satisfiers: dict[str, set[str]] = {}
    for spec in conditions:
        satisfiers[spec.name] = {
            ev.candidate.label
            for ev in parseable
            if ev.condition_results.get(spec.name, (False, ""))[0]
        }
    names = [spec.name for spec in conditions]
    cores: list[tuple[str, ...]] = []
    for size in range(1, len(names) + 1):
        for combo in combinations(names, size):
            if any(set(core) <= set(combo) for core in cores):
                continue
            union_ok = set.intersection(*(satisfiers[n] for n in combo)) if combo else set()
            if not union_ok:
                cores.append(combo)
    return [
        ConflictReport(
            conditions=core,
            satisfiers={name: sorted(satisfiers[name]) for name in core},
        )
        for core in cores
    ]


def solve(
    sample: bytes,
    gap: Gap,
    candidates: Sequence[Candidate],
    parser: Parser,
    conditions: Sequence[ConstraintSpec],
) -> SolveResult:
    """Try every candidate and aggregate all legal completions.

    Every candidate is spliced into the sample and the resulting buffer is
    re-parsed from the start, so candidates whose length differs from the
    gap correctly shift the position of every trailing field.
    """
    evidence = [
        _evaluate(sample, gap, parser, conditions, candidate) for candidate in candidates
    ]
    solutions = [ev for ev in evidence if ev.valid]
    unparseable = [ev for ev in evidence if not ev.parsed]

    common: dict[str, Any] = {}
    domains: dict[str, FieldDomain] = {}
    if solutions:
        field_names: list[str] = []
        for ev in solutions:
            for name in ev.fields or ():
                if name not in field_names:
                    field_names.append(name)
        for name in field_names:
            values = _distinct([(ev.fields or {}).get(name) for ev in solutions])
            domain = FieldDomain(field=name, values=values)
            domains[name] = domain
            if domain.is_fixed:
                common[name] = domain.common

    conflicts: list[ConflictReport] = []
    if not solutions:
        parseable = [ev for ev in evidence if ev.parsed]
        if parseable and conditions:
            conflicts = _minimal_conflicts(conditions, parseable)

    return SolveResult(
        evidence=evidence,
        solutions=solutions,
        common=common,
        domains=domains,
        conflicts=conflicts,
        unparseable=unparseable,
    )


def _fmt_value(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex(" ")
    return repr(value)


def format_report(result: SolveResult) -> str:
    """Render a human-readable report, keeping all candidate evidence."""
    lines: list[str] = []
    lines.append("== candidate evidence ==")
    for ev in result.evidence:
        lines.append(
            f"candidate {ev.candidate.label!r}: data={ev.candidate.data.hex(' ') or '(empty)'}"
        )
        if not ev.parsed:
            lines.append(f"  parse: FAILED ({ev.parse_error})")
            continue
        lines.append("  parse: ok")
        for name, (ok, note) in ev.condition_results.items():
            status = "pass" if ok else "FAIL"
            suffix = f" ({note})" if note else ""
            lines.append(f"  condition {name}: {status}{suffix}")
        lines.append(f"  => {'LEGAL' if ev.valid else 'rejected'}")

    if result.solutions:
        lines.append("")
        lines.append(f"== {len(result.solutions)} legal completion(s) ==")
        for ev in result.solutions:
            lines.append(f"  {ev.candidate.label!r}: buffer={ev.buffer.hex(' ')}")
        lines.append("")
        lines.append("== fields common to all legal completions ==")
        for name, value in result.common.items():
            lines.append(f"  {name} = {_fmt_value(value)}")
        lines.append("")
        lines.append("== remaining possible domains ==")
        for name, domain in result.domains.items():
            if domain.is_fixed:
                continue
            rendered = ", ".join(_fmt_value(v) for v in domain.values)
            lines.append(f"  {name}: {{{rendered}}}")
            if domain.is_numeric:
                lines.append(f"    range: [{domain.minimum}, {domain.maximum}]")
    else:
        lines.append("")
        lines.append("== no legal completion ==")
        if result.unparseable and len(result.unparseable) == len(result.evidence):
            lines.append("  every candidate fails to parse; see parse errors above")
        for conflict in result.conflicts:
            joined = " + ".join(conflict.conditions)
            lines.append(f"  conflicting conditions: {joined}")
            for name, labels in conflict.satisfiers.items():
                lines.append(f"    {name} satisfied only by: {labels or '(none)'}")
    return "\n".join(lines)
