# -*- coding: utf-8 -*-
"""Comparable cross-batch metric partitions for VCF INFO fields.

When metrics are collected across batches (VCF files), two problems show up:

* missing values (``.`` or absent keys) are silently counted as zero, and
* header definitions for the same field may change type or cardinality
  between batches, making the values incomparable.

This module provides the building blocks to do better:

* :py:class:`FieldConvention` captures the per-batch header contract
  ``(field, type, number)`` of a field.  Values are partitioned by
  convention, so same-name quantities with incompatible type or cardinality
  are never aggregated together.
* :py:class:`Observation` keeps one scalar value together with its status
  (valid / missing / conversion-failed), the raw text, and the identity of
  the whole variant it came from.
* :py:class:`Partition` aggregates the valid observations of one convention.
  Missing and failed observations are preserved and counted, but never enter
  the numeric statistics (in particular, they are never counted as zero and
  cannot be back-filled with zero later).
* :py:class:`CrossBatchMetrics` is a collection of partitions keyed by
  convention.  Merging two collections only merges partitions with identical
  conventions; the merged statistics are identical to computing them over
  the full combined data with the same threshold.

Thresholds filter which valid observations enter the statistics
(``value >= threshold``).  When the threshold changes, only the valid
observations between the old and the new threshold are re-evaluated; the
number of re-evaluated observations is reported as
:py:attr:`Partition.last_band`.
"""

import bisect
import enum
import math
from dataclasses import dataclass
from typing import Iterable, Iterator

from vcfpy.exceptions import VCFPyException
from vcfpy.header import FieldInfo, Header
from vcfpy.reader import Reader
from vcfpy.record import Record

__author__ = "Manuel Holtgrewe <manuel.holtgrewe@bihealth.de>"

__all__ = [
    "IncompatibleConvention",
    "ObsStatus",
    "VariantKey",
    "FieldConvention",
    "Observation",
    "Stats",
    "Partition",
    "CrossBatchMetrics",
    "collect_reader",
]


class IncompatibleConvention(VCFPyException):
    """Raised when quantities with incompatible field conventions are aggregated."""


@enum.unique
class ObsStatus(enum.Enum):
    """Status of a single extracted observation."""

    #: value present and converted according to the declared type
    VALID = "valid"
    #: value missing (``.`` or key absent); never counted as zero
    MISSING = "missing"
    #: value present but could not be converted to the declared type
    FAILED = "failed"


@dataclass(frozen=True)
class VariantKey:
    """Identity of a whole variant record."""

    chrom: str
    pos: int
    ref: str
    alts: tuple[str, ...]
    ids: tuple[str, ...]

    @classmethod
    def from_record(cls, record: Record) -> "VariantKey":
        """Build the key from a :py:class:`vcfpy.record.Record`."""
        return cls(
            chrom=record.CHROM,
            pos=record.POS,
            ref=record.REF,
            alts=tuple(alt.serialize() for alt in record.ALT),
            ids=tuple(record.ID),
        )


@dataclass(frozen=True)
class FieldConvention:
    """Header contract of a field within one batch: name, type, cardinality."""

    field: str
    type: str
    number: str

    @classmethod
    def from_field_info(cls, field_id: str, field_info: FieldInfo) -> "FieldConvention":
        """Build the convention from a header :py:class:`vcfpy.header.FieldInfo`."""
        return cls(field=field_id, type=field_info.type, number=str(field_info.number))

    @property
    def is_numeric(self) -> bool:
        """Whether values of this convention can enter numeric statistics."""
        return self.type in ("Integer", "Float", "Flag")


@dataclass(frozen=True)
class Observation:
    """One scalar value extracted from a record, with status and identity.

    Missing and failed observations never carry a value, so unrecoverable
    missing data cannot be back-filled with zeros.
    """

    #: convention of the field in the source batch
    convention: FieldConvention
    #: label of the source batch
    batch: str
    #: identity of the variant the value was observed on
    variant: VariantKey
    #: index within a multi-value field, ``None`` for scalar fields
    index: int | None
    #: status of the observation
    status: ObsStatus
    #: numeric value, only set for valid observations
    value: float | None = None
    #: raw text of failed conversions, for audit
    raw: str | None = None

    def __post_init__(self):
        if self.status is ObsStatus.VALID:
            if self.value is None or not math.isfinite(self.value):
                raise ValueError("valid observations need a finite value")
        elif self.value is not None:
            raise ValueError("missing/failed observations must not carry a value")


@dataclass(frozen=True)
class Stats:
    """Associative statistics over the valid observations of a partition.

    Merging two :py:class:`Stats` gives the same result as computing the
    statistics over the combined values.
    """

    count: int = 0
    total: float = 0.0
    minimum: float | None = None
    maximum: float | None = None

    @property
    def mean(self) -> float | None:
        """Mean over valid observations, ``None`` if there are none."""
        return None if self.count == 0 else self.total / self.count

    def merge(self, other: "Stats") -> "Stats":
        """Combine with statistics of a disjoint set of values."""
        if self.count == 0:
            return other
        if other.count == 0:
            return self
        return Stats(
            count=self.count + other.count,
            total=self.total + other.total,
            minimum=min(self.minimum, other.minimum),
            maximum=max(self.maximum, other.maximum),
        )

    @classmethod
    def from_values(cls, values: Iterable[float]) -> "Stats":
        """Compute statistics from scratch over ``values``."""
        values = list(values)
        if not values:
            return cls()
        return cls(
            count=len(values),
            total=math.fsum(values),
            minimum=min(values),
            maximum=max(values),
        )


class Partition:
    """Observations and statistics for one :py:class:`FieldConvention`."""

    def __init__(
        self, convention: FieldConvention, observations: Iterable[Observation] = ()
    ):
        #: convention all observations in this partition adhere to
        self.convention = convention
        #: all observations, including missing and failed ones
        self.observations: list[Observation] = []
        #: number of missing observations
        self.missing_count = 0
        #: number of failed conversions
        self.failed_count = 0
        #: number of valid observations re-evaluated at the last threshold change
        self.last_band = 0
        self._values: list[float] = []  # valid values, sorted
        self._threshold = -math.inf
        self._stats = Stats()
        for observation in observations:
            self.add(observation)

    @property
    def valid_count(self) -> int:
        """Number of valid observations."""
        return len(self._values)

    @property
    def threshold(self) -> float:
        """Currently applied threshold."""
        return self._threshold

    @property
    def batches(self) -> tuple[str, ...]:
        """Sorted labels of the batches contributing to this partition."""
        return tuple(sorted({obs.batch for obs in self.observations}))

    def add(self, observation: Observation) -> None:
        """Add an observation, which must follow this partition's convention."""
        if observation.convention != self.convention:
            raise IncompatibleConvention(
                "cannot add {} observation to {} partition".format(
                    observation.convention, self.convention
                )
            )
        self.observations.append(observation)
        if observation.status is ObsStatus.VALID:
            bisect.insort(self._values, observation.value)
            if observation.value >= self._threshold:
                self._stats = self._with_min_max(
                    Stats(
                        count=self._stats.count + 1,
                        total=self._stats.total + observation.value,
                    )
                )
        elif observation.status is ObsStatus.MISSING:
            self.missing_count += 1
        else:
            self.failed_count += 1

    def _with_min_max(self, stats: Stats) -> Stats:
        """Fill in min/max for the values at or above the current threshold."""
        if stats.count == 0:
            return Stats()
        i = bisect.bisect_left(self._values, self._threshold)
        return Stats(stats.count, stats.total, self._values[i], self._values[-1])

    def set_threshold(self, threshold: float) -> Stats:
        """Move the threshold, re-evaluating only the affected valid observations.

        Only the valid observations between the old and the new threshold
        change membership and are re-evaluated; their number is reported as
        :py:attr:`last_band`.  The result is identical to
        ``full_stats(threshold)``.
        """
        threshold = float(threshold)
        old = self._threshold
        if threshold == old:
            self.last_band = 0
            return self._stats
        lo, hi = (threshold, old) if threshold < old else (old, threshold)
        i = bisect.bisect_left(self._values, lo)
        j = bisect.bisect_left(self._values, hi)
        band = self._values[i:j]
        count = len(band)
        total = math.fsum(band)
        if threshold < old:  # widening: the band enters the statistic
            stats = Stats(self._stats.count + count, self._stats.total + total)
        else:  # narrowing: the band leaves the statistic
            stats = Stats(self._stats.count - count, self._stats.total - total)
        self._threshold = threshold
        self._stats = self._with_min_max(stats)
        self.last_band = count
        return self._stats

    def stats(self) -> Stats:
        """Statistics at the current threshold."""
        return self._stats

    def full_stats(self, threshold: float | None = None) -> Stats:
        """Statistics recomputed from scratch, for same-caliber comparison."""
        threshold = self._threshold if threshold is None else float(threshold)
        i = bisect.bisect_left(self._values, threshold)
        return Stats.from_values(self._values[i:])

    def copy(self) -> "Partition":
        """Return an independent copy."""
        result = Partition(self.convention)
        result.observations = list(self.observations)
        result.missing_count = self.missing_count
        result.failed_count = self.failed_count
        result._values = list(self._values)
        result._threshold = self._threshold
        result._stats = self._stats
        result.last_band = self.last_band
        return result

    def merge(self, other: "Partition") -> "Partition":
        """Merge with a partition of the same convention.

        The merged partition holds the observations of both sides and its
        statistics are computed over all valid observations (threshold reset
        to none), identical to a full computation over the combined data.
        """
        if self.convention != other.convention:
            raise IncompatibleConvention(
                "refusing to merge incompatible conventions {} and {}".format(
                    self.convention, other.convention
                )
            )
        result = Partition(self.convention)
        result.observations = self.observations + other.observations
        result.missing_count = self.missing_count + other.missing_count
        result.failed_count = self.failed_count + other.failed_count
        result._values = sorted(self._values + other._values)
        result._stats = result.full_stats()
        return result


class CrossBatchMetrics:
    """Metric partitions collected across batches, keyed by convention."""

    def __init__(self):
        #: mapping from field convention to its partition
        self.partitions: dict[FieldConvention, Partition] = {}

    def add(self, observation: Observation) -> None:
        """Add an observation to the partition of its convention."""
        partition = self.partitions.get(observation.convention)
        if partition is None:
            partition = self.partitions[observation.convention] = Partition(
                observation.convention
            )
        partition.add(observation)

    def extend(self, observations: Iterable[Observation]) -> None:
        """Add several observations."""
        for observation in observations:
            self.add(observation)

    def conventions_for(self, field: str) -> list[FieldConvention]:
        """All conventions seen for ``field``; more than one means the header
        definition changed between batches and the quantities are kept apart."""
        return sorted(
            (c for c in self.partitions if c.field == field),
            key=lambda c: (c.type, c.number),
        )

    def source_distribution(self) -> dict[FieldConvention, dict[str, int]]:
        """Per convention, the number of observations contributed by each batch."""
        result: dict[FieldConvention, dict[str, int]] = {}
        for convention, partition in self.partitions.items():
            counts: dict[str, int] = {}
            for observation in partition.observations:
                counts[observation.batch] = counts.get(observation.batch, 0) + 1
            result[convention] = counts
        return result

    def set_threshold(self, convention: FieldConvention, threshold: float) -> Stats:
        """Move the threshold of one partition, re-evaluating only affected
        valid observations."""
        return self.partitions[convention].set_threshold(threshold)

    def stats(self) -> dict[FieldConvention, Stats]:
        """Current statistics of all partitions."""
        return {c: p.stats() for c, p in self.partitions.items()}

    def merge(self, other: "CrossBatchMetrics") -> "CrossBatchMetrics":
        """Merge with another collection.

        Only partitions with identical conventions are merged; same-name
        fields with incompatible type or cardinality stay in separate
        partitions.  Merged statistics equal a full computation over the
        combined data, and missing observations stay missing.
        """
        result = CrossBatchMetrics()
        for convention in self.partitions.keys() | other.partitions.keys():
            left = self.partitions.get(convention)
            right = other.partitions.get(convention)
            if left is None:
                result.partitions[convention] = right.copy()
            elif right is None:
                result.partitions[convention] = left.copy()
            else:
                result.partitions[convention] = left.merge(right)
        return result


#: sentinel distinguishing "key absent" from "present but missing (.)"
_ABSENT = object()


def _observations_for(
    value, convention: FieldConvention, batch: str, variant: VariantKey
) -> Iterator[Observation]:
    """Build observations for one parsed INFO value."""

    def make(status, index=None, val=None, raw=None):
        return Observation(
            convention=convention,
            batch=batch,
            variant=variant,
            index=index,
            status=status,
            value=val,
            raw=raw,
        )

    def convert(scalar, index=None):
        if scalar is None:
            yield make(ObsStatus.MISSING, index=index)
        elif convention.type == "Flag":
            yield make(ObsStatus.VALID, index=index, val=1.0)
        elif isinstance(scalar, bool) or not isinstance(scalar, (int, float)):
            # parser kept a value it could not convert as string
            yield make(ObsStatus.FAILED, index=index, raw=str(scalar))
        elif not math.isfinite(scalar):
            yield make(ObsStatus.FAILED, index=index, raw=repr(scalar))
        else:
            yield make(ObsStatus.VALID, index=index, val=float(scalar))

    if value is _ABSENT or value is None or value == []:
        yield make(ObsStatus.MISSING)
    elif isinstance(value, list):
        for index, scalar in enumerate(value):
            yield from convert(scalar, index=index)
    else:
        yield from convert(value)


def collect_reader(
    reader: Reader,
    batch: str,
    fields: Iterable[str] | None = None,
    metrics: CrossBatchMetrics | None = None,
) -> CrossBatchMetrics:
    """Collect comparable metric partitions for INFO fields from ``reader``.

    :param reader: open :py:class:`vcfpy.Reader` of one batch
    :param batch: label of the batch, kept on every observation
    :param fields: INFO fields to collect; defaults to all numeric and flag
        INFO fields declared in the header
    :param metrics: collection to add to; a new one is created if omitted
    :return: the :py:class:`CrossBatchMetrics` collection
    """
    header: Header = reader.header
    if fields is None:
        fields = [
            field_id
            for field_id in header.info_ids()
            if FieldConvention.from_field_info(
                field_id, header.get_info_field_info(field_id)
            ).is_numeric
        ]
    conventions = {}
    for field_id in fields:
        convention = FieldConvention.from_field_info(
            field_id, header.get_info_field_info(field_id)
        )
        if not convention.is_numeric:
            raise ValueError(
                "field {} of type {} is not numeric".format(field_id, convention.type)
            )
        conventions[field_id] = convention
    if metrics is None:
        metrics = CrossBatchMetrics()
    for record in reader:
        variant = VariantKey.from_record(record)
        for field_id, convention in conventions.items():
            value = record.INFO.get(field_id, _ABSENT)
            metrics.extend(_observations_for(value, convention, batch, variant))
    return metrics
