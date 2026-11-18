# -*- coding: utf-8 -*-
"""Unified filter statistics across sources with auditable label mappings.

This module aggregates filter outcomes from multiple sources (e.g. two
callers or two annotation pipelines) into one set of statistics.

The model distinguishes, at both record and sample level:

- **not done** -- filtering was not applied (VCF ``.``),
- **passed** -- filtering was applied and passed (VCF ``PASS``),
- **failed** -- one or more concrete failure labels.

Two sources may use the same failure label with different definitions.
The caller therefore supplies a :class:`FilterMapping` of conditional
:class:`MappingRule` objects together with the information loss they are
willing to accept.  Failures without a provable correspondence are never
folded into a generic failure bucket; they are reported under an explicit
``@unmapped:<source>:<label>`` category.

Statistics are incremental: after switching the mapping, only items whose
``(source, label)`` pair is affected by the change are reclassified.
:meth:`UnifiedFilterStats.reconcile` reports per-source denominators and
known/unknown shares so every unified figure can be traced back.
"""

from __future__ import annotations

import enum
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

#: Unified category for items where filtering was not applied.
CATEGORY_NOT_DONE = "@not_done"
#: Unified category for items that passed filtering.
CATEGORY_PASSED = "@passed"
#: Prefix for failure labels without a provable mapping.
UNMAPPED_PREFIX = "@unmapped:"


def unmapped_category(source: str, label: str) -> str:
    """Return the explicit category for an unmappable failure label.

    Unmapped failures keep their source and label so they are never
    swallowed into a generic failure bucket.
    """
    return "%s%s:%s" % (UNMAPPED_PREFIX, source, label)


class Level(enum.Enum):
    """Level at which a filter outcome was observed."""

    RECORD = "record"
    SAMPLE = "sample"


class FilterStatus(enum.Enum):
    """Tri-state outcome of filtering a single item."""

    NOT_DONE = "not_done"
    PASSED = "passed"
    FAILED = "failed"


class Loss(enum.IntEnum):
    """Information loss incurred by a mapping rule."""

    #: Source and target label have identical semantics.
    NONE = 0
    #: One-to-one rename, distinction is preserved.
    RENAME = 1
    #: Many-to-one merge, per-source distinction is lost.
    COLLAPSE = 2


@dataclass(frozen=True)
class FilterOutcome:
    """Filter outcome of one item (record or record/sample pair)."""

    #: Name of the source the outcome originates from.
    source: str
    #: Record-level or sample-level outcome.
    level: Level
    #: Hashable item identifier, unique per ``(source, level)``.
    key: Tuple[str, ...]
    #: Tri-state filter status.
    status: FilterStatus
    #: Concrete failure labels; non-empty iff ``status`` is ``FAILED``.
    labels: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status is FilterStatus.FAILED and not self.labels:
            raise ValueError("FAILED outcomes require at least one label")
        if self.status is not FilterStatus.FAILED and self.labels:
            raise ValueError("labels are only allowed for FAILED outcomes")


@dataclass(frozen=True)
class MappingRule:
    """Conditional mapping of one source failure label to a unified label."""

    #: Source the rule applies to.
    source: str
    #: Failure label in the source's vocabulary.
    label: str
    #: Unified target label.
    target: str
    #: Information loss declared for this rule.
    loss: Loss = Loss.NONE
    #: Optional additional condition on the concrete outcome.
    condition: Optional[Callable[[FilterOutcome], bool]] = None

    def matches(self, outcome: FilterOutcome, label: str) -> bool:
        """Return whether the rule applies to ``label`` of ``outcome``."""
        if self.source != outcome.source or self.label != label:
            return False
        return self.condition is None or bool(self.condition(outcome))


class FilterMapping:
    """Immutable set of mapping rules bounded by an allowed information loss."""

    def __init__(
        self,
        rules: Iterable[MappingRule] = (),
        allowed_loss: Loss = Loss.COLLAPSE,
    ) -> None:
        rules = tuple(rules)
        for rule in rules:
            if rule.loss > allowed_loss:
                raise ValueError(
                    "rule %s:%s -> %s declares loss %s beyond allowed loss %s"
                    % (rule.source, rule.label, rule.target, rule.loss, allowed_loss)
                )
        #: Rules in the order given by the caller; first match wins.
        self.rules: Tuple[MappingRule, ...] = rules
        #: Maximum information loss the caller accepts.
        self.allowed_loss = allowed_loss

    def classify(self, outcome: FilterOutcome, label: str) -> Optional[int]:
        """Return the index of the first rule matching ``label``, if any."""
        for idx, rule in enumerate(self.rules):
            if rule.matches(outcome, label):
                return idx
        return None

    def _signatures(self) -> Dict[Tuple[str, str], Tuple]:
        """Return per ``(source, label)`` classification signatures."""
        result: Dict[Tuple[str, str], List[Tuple]] = {}
        for rule in self.rules:
            key = (rule.source, rule.label)
            # Conditions are compared by identity; rebuilding a mapping with
            # fresh callables is treated as a change (conservative, safe).
            result.setdefault(key, []).append(
                (rule.target, rule.loss, id(rule.condition) if rule.condition else None)
            )
        return {key: tuple(entries) for key, entries in result.items()}

    def affected_pairs(self, other: "FilterMapping") -> frozenset:
        """Return ``(source, label)`` pairs that may classify differently."""
        mine, theirs = self._signatures(), other._signatures()
        return frozenset(
            key for key in set(mine) | set(theirs) if mine.get(key) != theirs.get(key)
        )


@dataclass(frozen=True)
class LabelContribution:
    """Contribution of one item (and label) to the unified statistics."""

    #: Item identifier, as given in the :class:`FilterOutcome`.
    key: Tuple[str, ...]
    #: Source the item originates from.
    source: str
    #: Record-level or sample-level.
    level: Level
    #: Tri-state status of the item.
    status: FilterStatus
    #: Original failure label, ``None`` for non-failed items.
    label: Optional[str]
    #: Unified category the item contributes to.
    category: str
    #: Index of the applied :class:`MappingRule`, ``None`` if unmapped.
    rule: Optional[int]
    #: Information loss incurred, ``None`` if unmapped or not failed.
    loss: Optional[Loss]
    #: Whether the failure label had a provable correspondence.
    mapped: bool


@dataclass(frozen=True)
class LevelStats:
    """Unified statistics for one level."""

    #: Item-level counts per :class:`FilterStatus`.
    status: Counter
    #: Label-instance-level counts per unified failure category.
    failures: Counter


@dataclass(frozen=True)
class SourceReconciliation:
    """Per source and level breakdown for auditing the unified statistics."""

    source: str
    level: Level
    #: Total number of items contributed by this source.
    denominator: int
    not_done: int
    passed: int
    failed: int
    #: Total number of failure label instances.
    failure_instances: int
    #: Failure label instances with a provable mapping.
    known: int
    #: Failure label instances without a provable mapping.
    unknown: int

    @property
    def known_share(self) -> float:
        """Share of failure instances with a provable mapping."""
        return self.known / self.failure_instances if self.failure_instances else 1.0

    @property
    def unknown_share(self) -> float:
        """Share of failure instances without a provable mapping."""
        return self.unknown / self.failure_instances if self.failure_instances else 0.0


@dataclass(frozen=True)
class Reconciliation:
    """Audit result tying unified statistics back to per-source denominators."""

    #: Per ``(source, level)`` breakdowns, sorted for determinism.
    by_source: Tuple[SourceReconciliation, ...]
    #: Detected inconsistencies; empty if everything reconciles.
    problems: Tuple[str, ...]

    @property
    def ok(self) -> bool:
        """Return whether all denominators and shares reconcile."""
        return not self.problems


class UnifiedFilterStats:
    """Incrementally maintained unified filter statistics.

    Raw outcomes are stored once; only the classification into unified
    categories depends on the current :class:`FilterMapping`.  Switching
    the mapping via :meth:`set_mapping` reclassifies only the items whose
    ``(source, label)`` pair is affected by the change.
    """

    def __init__(self, mapping: Optional[FilterMapping] = None) -> None:
        self._mapping = mapping or FilterMapping()
        self._items: Dict[Tuple, FilterOutcome] = {}
        self._cache: Dict[Tuple, Tuple[LabelContribution, ...]] = {}

    @property
    def mapping(self) -> FilterMapping:
        """Return the currently active mapping."""
        return self._mapping

    @staticmethod
    def _item_id(outcome: FilterOutcome) -> Tuple:
        return (outcome.source, outcome.level, outcome.key)

    def add(self, outcome: FilterOutcome) -> None:
        """Add one outcome; raises on duplicate ``(source, level, key)``."""
        item_id = self._item_id(outcome)
        if item_id in self._items:
            raise ValueError("duplicate outcome for %r" % (item_id,))
        self._items[item_id] = outcome
        self._cache[item_id] = self._classify(outcome)

    def extend(self, outcomes: Iterable[FilterOutcome]) -> None:
        """Add several outcomes."""
        for outcome in outcomes:
            self.add(outcome)

    def _classify(self, outcome: FilterOutcome) -> Tuple[LabelContribution, ...]:
        base = {
            "key": outcome.key,
            "source": outcome.source,
            "level": outcome.level,
            "status": outcome.status,
        }
        if outcome.status is FilterStatus.NOT_DONE:
            return (
                LabelContribution(
                    label=None,
                    category=CATEGORY_NOT_DONE,
                    rule=None,
                    loss=None,
                    mapped=False,
                    **base,
                ),
            )
        if outcome.status is FilterStatus.PASSED:
            return (
                LabelContribution(
                    label=None,
                    category=CATEGORY_PASSED,
                    rule=None,
                    loss=None,
                    mapped=False,
                    **base,
                ),
            )
        result = []
        for label in outcome.labels:
            rule_idx = self._mapping.classify(outcome, label)
            if rule_idx is None:
                # No provable correspondence: keep source and label explicit
                # instead of swallowing the failure into a generic bucket.
                result.append(
                    LabelContribution(
                        label=label,
                        category=unmapped_category(outcome.source, label),
                        rule=None,
                        loss=None,
                        mapped=False,
                        **base,
                    )
                )
            else:
                rule = self._mapping.rules[rule_idx]
                result.append(
                    LabelContribution(
                        label=label,
                        category=rule.target,
                        rule=rule_idx,
                        loss=rule.loss,
                        mapped=True,
                        **base,
                    )
                )
        return tuple(result)

    def set_mapping(self, mapping: FilterMapping) -> int:
        """Switch the mapping, reclassifying only affected items.

        Items that did not fail and items whose ``(source, label)`` pairs
        are untouched by the change keep their cached classification.

        :return: number of items that were reclassified.
        """
        affected = self._mapping.affected_pairs(mapping)
        self._mapping = mapping
        reclassified = 0
        for item_id, outcome in self._items.items():
            if outcome.status is not FilterStatus.FAILED:
                continue
            if any((outcome.source, label) in affected for label in outcome.labels):
                self._cache[item_id] = self._classify(outcome)
                reclassified += 1
        return reclassified

    def contributions(self) -> List[LabelContribution]:
        """Return every item's contribution, in insertion order."""
        result: List[LabelContribution] = []
        for item_id in self._items:
            result.extend(self._cache[item_id])
        return result

    def stats(self, level: Optional[Level] = None) -> Dict[Level, LevelStats]:
        """Return unified statistics, separately per level.

        :param level: restrict to one level; by default both are returned.
        """
        status: Dict[Level, Counter] = {lv: Counter() for lv in Level}
        failures: Dict[Level, Counter] = {lv: Counter() for lv in Level}
        for item_id, contribs in self._cache.items():
            item_level = item_id[1]
            if level is not None and item_level is not level:
                continue
            status[item_level][contribs[0].status] += 1
            for contrib in contribs:
                if contrib.status is FilterStatus.FAILED:
                    failures[item_level][contrib.category] += 1
        levels = [level] if level is not None else list(Level)
        return {
            lv: LevelStats(status=status[lv], failures=failures[lv]) for lv in levels
        }

    def reconcile(self) -> Reconciliation:
        """Audit unified statistics against per-source denominators.

        Every source's denominator, status counts, and known/unknown
        failure shares are recomputed from the stored contributions and
        cross-checked against the unified statistics.
        """
        by_source: Dict[Tuple[str, Level], SourceReconciliation] = {}
        status: Dict[Tuple[str, Level], Counter] = {}
        known: Dict[Tuple[str, Level], int] = {}
        unknown: Dict[Tuple[str, Level], int] = {}
        for item_id, contribs in self._cache.items():
            source, item_level = item_id[0], item_id[1]
            group = (source, item_level)
            status.setdefault(group, Counter())[contribs[0].status] += 1
            for contrib in contribs:
                if contrib.status is not FilterStatus.FAILED:
                    continue
                if contrib.mapped:
                    known[group] = known.get(group, 0) + 1
                else:
                    unknown[group] = unknown.get(group, 0) + 1
        for group, counts in status.items():
            failure_instances = known.get(group, 0) + unknown.get(group, 0)
            by_source[group] = SourceReconciliation(
                source=group[0],
                level=group[1],
                denominator=sum(counts.values()),
                not_done=counts.get(FilterStatus.NOT_DONE, 0),
                passed=counts.get(FilterStatus.PASSED, 0),
                failed=counts.get(FilterStatus.FAILED, 0),
                failure_instances=failure_instances,
                known=known.get(group, 0),
                unknown=unknown.get(group, 0),
            )
        problems = []
        unified = self.stats()
        for lv in Level:
            groups = [g for g in by_source if g[1] is lv]
            denominator = sum(by_source[g].denominator for g in groups)
            unified_items = sum(unified[lv].status.values())
            if denominator != unified_items:
                problems.append(
                    "level %s: source denominators %d != unified item count %d"
                    % (lv.value, denominator, unified_items)
                )
            instances = sum(by_source[g].failure_instances for g in groups)
            unified_failures = sum(unified[lv].failures.values())
            if instances != unified_failures:
                problems.append(
                    "level %s: failure instances %d != unified failure count %d"
                    % (lv.value, instances, unified_failures)
                )
        return Reconciliation(
            by_source=tuple(
                by_source[g]
                for g in sorted(by_source, key=lambda g: (g[0], g[1].value))
            ),
            problems=tuple(problems),
        )


def record_outcome(
    record, source: str, key: Optional[Tuple[str, ...]] = None
) -> FilterOutcome:
    """Build a record-level :class:`FilterOutcome` from a VCF record.

    An empty ``FILTER`` (VCF ``.``) means filtering was not done, a sole
    ``PASS`` means passed, anything else yields the concrete failure
    labels.
    """
    labels = tuple(f for f in (record.FILTER or []) if f != "PASS")
    if key is None:
        key = (record.CHROM, str(record.POS))
    if not record.FILTER:
        status = FilterStatus.NOT_DONE
    elif not labels:
        status = FilterStatus.PASSED
    else:
        status = FilterStatus.FAILED
    return FilterOutcome(
        source=source,
        level=Level.RECORD,
        key=tuple(key),
        status=status,
        labels=labels if status is FilterStatus.FAILED else (),
    )


def sample_outcomes(record, source: str) -> List[FilterOutcome]:
    """Build sample-level :class:`FilterOutcome` objects from a VCF record.

    Uses each call's ``FT`` field with the same tri-state semantics as
    :func:`record_outcome`.
    """
    result = []
    for call in record.calls:
        ft = call.data.get("FT") or []
        labels = tuple(f for f in ft if f != "PASS")
        if not ft:
            status = FilterStatus.NOT_DONE
        elif not labels:
            status = FilterStatus.PASSED
        else:
            status = FilterStatus.FAILED
        result.append(
            FilterOutcome(
                source=source,
                level=Level.SAMPLE,
                key=(record.CHROM, str(record.POS), call.sample),
                status=status,
                labels=labels if status is FilterStatus.FAILED else (),
            )
        )
    return result
