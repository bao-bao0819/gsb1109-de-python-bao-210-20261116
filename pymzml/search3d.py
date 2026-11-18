#!/usr/bin/env python3
# -*- coding: latin-1 -*-
"""
Three-dimensional peak search for retention time, m/z and ion mobility.

The module combines three pieces:

* :class:`Peak3D` stores one original peak with all three coordinates.
* :class:`MobilityCalibration` maps measured ion mobility to calibrated
  mobility with a piecewise calibration table. Every segment carries its own
  error range, which is propagated into the hit evidence.
* :class:`PeakIndex3D` searches original peaks jointly. A hit is always based
  on the three coordinates of **one** peak; coordinates taken from different
  peaks are never combined.

Hits are classified as ``definite`` when the complete observed interval of
each dimension lies inside the query window, and as ``possible`` when every
dimension overlaps its query window but at least one observed interval is not
fully contained (for example because the mobility calibration segment has a
large error range).

Calibration corrections are applied with :meth:`PeakIndex3D.update_calibration`
or :meth:`PeakIndex3D.correct_calibration_segment`. Only peaks whose calibrated
value or error interval changed are recalibrated and invalidated; peaks in
unaffected segments keep their cached query results. Attached
:class:`Projection2D` views are updated in the same operation, so the
two-dimensional retention time/m/z view never keeps stale mobility values or
hit annotations.
"""

# Python mzML module - pymzml
# Copyright (C) 2010-2019 M. Kösters, C. Fufezan
#     The MIT License (MIT)

import math
import weakref
from collections import OrderedDict
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "Search3DError",
    "CalibrationError",
    "UncalibratedMobilityError",
    "PeakValidationError",
    "DuplicatePeakError",
    "MobilityCalibrationSegment",
    "CalibrationSegment",
    "MobilityCalibration",
    "CalibrationTable",
    "Peak3D",
    "SearchQuery",
    "Query3D",
    "DimensionEvidence",
    "CalibrationBasis",
    "Hit3D",
    "SearchResult",
    "RecalibrationReport",
    "ProjectionRow",
    "Projection2D",
    "PeakIndex3D",
    "ThreeDSearchIndex",
    "extract_peaks3d",
    "extract_reader_peaks3d",
]

_EPS = 1e-12
_DIMENSIONS = ("retention_time", "mz", "mobility")
_TIME_UNIT_TO_MINUTES = {
    "millisecond": 1.0 / 1000.0 / 60.0,
    "milliseconds": 1.0 / 1000.0 / 60.0,
    "ms": 1.0 / 1000.0 / 60.0,
    "second": 1.0 / 60.0,
    "seconds": 1.0 / 60.0,
    "s": 1.0 / 60.0,
    "minute": 1.0,
    "minutes": 1.0,
    "min": 1.0,
    "hour": 60.0,
    "hours": 60.0,
    "h": 60.0,
}


class Search3DError(ValueError):
    """Base error for the three-dimensional search helpers."""


class CalibrationError(Search3DError):
    """Raised for an invalid or incomplete mobility calibration table."""


class UncalibratedMobilityError(CalibrationError):
    """Raised when a mobility value is outside every calibration segment."""


class PeakValidationError(Search3DError):
    """Raised when a peak does not carry usable three-dimensional data."""


class DuplicatePeakError(Search3DError):
    """Raised when two original peaks use the same peak identifier."""


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _require_finite(name: str, value: Any) -> float:
    if not _is_finite_number(value):
        raise PeakValidationError("{0} must be a finite number, got {1!r}".format(name, value))
    return float(value)


def _require_bound(name: str, value: Any) -> float:
    """A calibration bound may be infinite, but never NaN."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CalibrationError("{0} must be a number, got {1!r}".format(name, value))
    value = float(value)
    if math.isnan(value):
        raise CalibrationError("{0} must not be NaN".format(name))
    return value


def _require_non_negative(name: str, value: Any) -> float:
    value = _require_finite(name, value)
    if value < 0:
        raise PeakValidationError("{0} must not be negative, got {1!r}".format(name, value))
    return value


def _require_calibration_error(name: str, value: Any) -> float:
    if not _is_finite_number(value) or float(value) < 0:
        raise CalibrationError(
            "{0} must be a finite non-negative number, got {1!r}".format(name, value)
        )
    return float(value)


def _le(a: float, b: float) -> bool:
    """Inclusive comparison that tolerates floating point round-off."""
    return a <= b or math.isclose(a, b, rel_tol=1e-12, abs_tol=_EPS)


def _ge(a: float, b: float) -> bool:
    return _le(b, a)


def _interval_contains(
    outer_low: float, outer_high: float, inner_low: float, inner_high: float
) -> bool:
    return _le(outer_low, inner_low) and _le(inner_high, outer_high)


def _interval_overlaps(
    low_a: float, high_a: float, low_b: float, high_b: float
) -> bool:
    return _le(low_a, high_b) and _le(low_b, high_a)


@dataclass(frozen=True)
class MobilityCalibrationSegment:
    """
    One piece of a piecewise ion mobility calibration.

    Within ``[lower, upper)`` the calibrated mobility is::

        calibrated = slope * measured + intercept

    ``error_low`` and ``error_high`` form the calibration error range in
    calibrated mobility units. If they are omitted, the symmetric ``error``
    half-width is used on both sides. The last segment includes its upper
    bound; all other segments are half-open so that boundaries are
    unambiguous.

    Arguments:
        lower (float): lower measured mobility bound (inclusive).
        upper (float): upper measured mobility bound (exclusive, except for
            the last segment).
        slope (float): slope of the segment calibration.
        intercept (float): intercept of the segment calibration.
        error (float): symmetric calibration error half-width.
        error_low (float, optional): lower calibration error. Defaults to
            ``error``.
        error_high (float, optional): upper calibration error. Defaults to
            ``error``.
        label (str): human readable segment label, stored as evidence.
    """

    lower: float
    upper: float
    slope: float = 1.0
    intercept: float = 0.0
    error: float = 0.0
    error_low: Optional[float] = None
    error_high: Optional[float] = None
    label: str = ""

    def __post_init__(self):
        lower = _require_bound("segment lower", self.lower)
        upper = _require_bound("segment upper", self.upper)
        if not upper > lower:
            raise CalibrationError(
                "calibration segment upper bound must be greater than its "
                "lower bound ({0!r} <= {1!r})".format(upper, lower)
            )
        slope = _require_finite("segment slope", self.slope)
        intercept = _require_finite("segment intercept", self.intercept)
        error = _require_calibration_error("segment error", self.error)
        error_low = (
            error
            if self.error_low is None
            else _require_calibration_error("segment error_low", self.error_low)
        )
        error_high = (
            error
            if self.error_high is None
            else _require_calibration_error("segment error_high", self.error_high)
        )
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "slope", slope)
        object.__setattr__(self, "intercept", intercept)
        object.__setattr__(self, "error", error)
        object.__setattr__(self, "error_low", error_low)
        object.__setattr__(self, "error_high", error_high)
        object.__setattr__(self, "label", "" if self.label is None else str(self.label))

    @property
    def lower_error(self) -> float:
        """Calibration error propagated to lower calibrated values."""
        return self.error_low

    @property
    def upper_error(self) -> float:
        """Calibration error propagated to upper calibrated values."""
        return self.error_high

    def contains(self, measured_mobility: float, *, last: bool = False) -> bool:
        """Return whether a measured mobility belongs to this segment."""
        if last:
            return _ge(measured_mobility, self.lower) and _le(
                measured_mobility, self.upper
            )
        return _ge(measured_mobility, self.lower) and measured_mobility < self.upper

    @property
    def signature(self) -> Tuple[Any, ...]:
        """Complete signature of the segment, including its evidence label."""
        return (
            self.lower,
            self.upper,
            self.slope,
            self.intercept,
            self.error_low,
            self.error_high,
            self.label,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lower": None if math.isinf(self.lower) else self.lower,
            "upper": None if math.isinf(self.upper) else self.upper,
            "slope": self.slope,
            "intercept": self.intercept,
            "error": self.error,
            "error_low": self.error_low,
            "error_high": self.error_high,
            "label": self.label,
        }


CalibrationSegment = MobilityCalibrationSegment


@dataclass(frozen=True)
class _CalibratedMobility:
    measured: float
    value: float
    lower: float
    upper: float
    segment_index: int
    segment: MobilityCalibrationSegment


@dataclass(frozen=True)
class MobilityCalibration:
    """
    Piecewise calibration table for measured to calibrated ion mobility.

    Segments are sorted by their measured mobility range. Gaps are allowed;
    peaks falling into a gap are reported as uncalibrated and never become
    hits. Overlapping segments are rejected.

    Arguments:
        segments (iterable): :class:`MobilityCalibrationSegment` objects.
        name (str): calibration table name, stored with every hit.
        revision (str or int, optional): caller supplied table revision.
    """

    segments: Tuple[MobilityCalibrationSegment, ...]
    name: str = "ion mobility calibration"
    revision: Optional[Any] = None

    def __post_init__(self):
        segments = tuple(self.segments)
        if len(segments) == 0:
            raise CalibrationError("calibration table requires at least one segment")
        for segment in segments:
            if not isinstance(segment, MobilityCalibrationSegment):
                raise CalibrationError(
                    "segments must be MobilityCalibrationSegment instances, got "
                    "{0!r}".format(segment)
                )
        segments = tuple(sorted(segments, key=lambda item: (item.lower, item.upper)))
        for previous, current in zip(segments, segments[1:]):
            if current.lower < previous.upper:
                raise CalibrationError(
                    "calibration segments overlap: {0!r}-{1!r} and {2!r}-{3!r}".format(
                        previous.lower,
                        previous.upper,
                        current.lower,
                        current.upper,
                    )
                )
        object.__setattr__(self, "segments", segments)
        object.__setattr__(self, "name", str(self.name))

    def __len__(self) -> int:
        return len(self.segments)

    def __iter__(self):
        return iter(self.segments)

    def __getitem__(self, item):
        return self.segments[item]

    def segment_index_for(self, measured_mobility: float) -> Optional[int]:
        """Return the segment index for a measured mobility, or None."""
        measured_mobility = _require_finite("measured mobility", measured_mobility)
        last_index = len(self.segments) - 1
        for index, segment in enumerate(self.segments):
            if segment.contains(measured_mobility, last=index == last_index):
                return index
        return None

    def segment_for(self, measured_mobility: float) -> Optional[MobilityCalibrationSegment]:
        """Return the segment for a measured mobility, or None."""
        index = self.segment_index_for(measured_mobility)
        if index is None:
            return None
        return self.segments[index]

    def calibrate(self, measured_mobility: float) -> _CalibratedMobility:
        """
        Calibrate one measured mobility and return its error interval.

        Raises:
            UncalibratedMobilityError: if the value lies outside every segment.
        """
        measured_mobility = _require_finite("measured mobility", measured_mobility)
        index = self.segment_index_for(measured_mobility)
        if index is None:
            raise UncalibratedMobilityError(
                "measured mobility {0!r} is outside calibration table {1!r}".format(
                    measured_mobility, self.name
                )
            )
        segment = self.segments[index]
        value = segment.slope * measured_mobility + segment.intercept
        return _CalibratedMobility(
            measured=measured_mobility,
            value=value,
            lower=value - segment.error_low,
            upper=value + segment.error_high,
            segment_index=index,
            segment=segment,
        )

    def with_segment(
        self, segment_index: int, *, revision: Optional[Any] = None, **changes
    ) -> "MobilityCalibration":
        """
        Return a corrected copy of this table.

        The original table is not modified. ``changes`` accepts any field of
        :class:`MobilityCalibrationSegment` (for example ``slope``, ``error``
        or ``upper``). ``revision`` names the corrected table; it is stored in
        the saved calibration basis of every hit.

        Arguments:
            segment_index (int): index into the current sorted segment list.

        Keyword Arguments:
            revision: optional new table revision. If omitted, the current
                revision is kept.

        Returns:
            calibration (MobilityCalibration): corrected copy.
        """
        if not isinstance(segment_index, int):
            raise CalibrationError("segment_index must be an integer")
        if not -len(self.segments) <= segment_index < len(self.segments):
            raise CalibrationError(
                "segment_index {0} is out of range for {1} segments".format(
                    segment_index, len(self.segments)
                )
            )
        index = segment_index % len(self.segments)
        segments = list(self.segments)
        changes = dict(changes)
        if "error" in changes:
            # A symmetric error correction replaces both sides unless the
            # caller explicitly corrects one side as well.
            changes.setdefault("error_low", changes["error"])
            changes.setdefault("error_high", changes["error"])
        try:
            segments[index] = replace(segments[index], **changes)
        except TypeError as error:
            raise CalibrationError(
                "unknown calibration segment field in {0}".format(
                    sorted(changes)
                )
            ) from error
        return MobilityCalibration(
            segments=tuple(segments),
            name=self.name,
            revision=self.revision if revision is None else revision,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "revision": self.revision,
            "segments": [segment.to_dict() for segment in self.segments],
        }


CalibrationTable = MobilityCalibration


@dataclass(frozen=True)
class Peak3D:
    """
    One original peak with retention time, m/z and measured ion mobility.

    The values always belong together. Search code never borrows a coordinate
    from another peak.

    Arguments:
        peak_id (hashable): unique identifier of the original peak.
        retention_time (float): retention time (keep one unit per index).
        mz (float): mass over charge.
        measured_mobility (float): measured ion mobility before calibration.
        intensity (float): peak intensity.
        retention_time_uncertainty (float): optional half-width of the
            retention time measurement interval.
        mz_uncertainty (float): optional half-width of the m/z measurement
            interval.
        source (str, optional): spectrum or file the peak came from.
        metadata (mapping): additional evidence kept with the peak.
    """

    peak_id: Hashable
    retention_time: float
    mz: float
    measured_mobility: float
    intensity: float = 0.0
    retention_time_uncertainty: float = 0.0
    mz_uncertainty: float = 0.0
    source: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self):
        if self.peak_id is None:
            raise PeakValidationError("peak_id must not be None")
        try:
            hash(self.peak_id)
        except TypeError as error:
            raise PeakValidationError("peak_id must be hashable") from error
        object.__setattr__(
            self, "retention_time", _require_finite("retention_time", self.retention_time)
        )
        mz = _require_finite("mz", self.mz)
        if mz <= 0:
            raise PeakValidationError("mz must be positive, got {0!r}".format(mz))
        object.__setattr__(self, "mz", mz)
        object.__setattr__(
            self,
            "measured_mobility",
            _require_finite("measured_mobility", self.measured_mobility),
        )
        object.__setattr__(self, "intensity", _require_finite("intensity", self.intensity))
        object.__setattr__(
            self,
            "retention_time_uncertainty",
            _require_non_negative(
                "retention_time_uncertainty", self.retention_time_uncertainty
            ),
        )
        object.__setattr__(
            self,
            "mz_uncertainty",
            _require_non_negative("mz_uncertainty", self.mz_uncertainty),
        )
        if self.source is not None:
            object.__setattr__(self, "source", str(self.source))
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})
        elif not isinstance(self.metadata, Mapping):
            raise PeakValidationError("metadata must be a mapping")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "peak_id": self.peak_id,
            "retention_time": self.retention_time,
            "mz": self.mz,
            "measured_mobility": self.measured_mobility,
            "intensity": self.intensity,
            "retention_time_uncertainty": self.retention_time_uncertainty,
            "mz_uncertainty": self.mz_uncertainty,
            "source": self.source,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SearchQuery:
    """
    Joint three-dimensional query window.

    A peak is a definite hit only when its complete observed interval in all
    three dimensions lies inside the respective query window. It is a possible
    hit when every observed interval overlaps its window, but at least one is
    not fully contained.

    Arguments:
        retention_time (float): target retention time.
        mz (float): target m/z.
        mobility (float): target **calibrated** ion mobility.
        rt_tolerance (float): retention time half-width.
        mz_tolerance (float): m/z half-width in m/z units.
        mobility_tolerance (float): calibrated mobility half-width.
        mz_ppm (float, optional): relative m/z half-width in ppm. When given,
            it replaces ``mz_tolerance``.
    """

    retention_time: float
    mz: float
    mobility: float
    rt_tolerance: float
    mz_tolerance: float = 0.0
    mobility_tolerance: float = 0.0
    mz_ppm: Optional[float] = None

    def __post_init__(self):
        object.__setattr__(
            self, "retention_time", _require_finite("retention_time", self.retention_time)
        )
        mz = _require_finite("mz", self.mz)
        if mz <= 0:
            raise PeakValidationError("mz must be positive, got {0!r}".format(mz))
        object.__setattr__(self, "mz", mz)
        object.__setattr__(self, "mobility", _require_finite("mobility", self.mobility))
        object.__setattr__(
            self, "rt_tolerance", _require_non_negative("rt_tolerance", self.rt_tolerance)
        )
        if self.mz_ppm is not None:
            mz_ppm = _require_non_negative("mz_ppm", self.mz_ppm)
            object.__setattr__(self, "mz_ppm", mz_ppm)
            object.__setattr__(self, "mz_tolerance", mz * mz_ppm * 1e-6)
        else:
            object.__setattr__(
                self,
                "mz_tolerance",
                _require_non_negative("mz_tolerance", self.mz_tolerance),
            )
        object.__setattr__(
            self,
            "mobility_tolerance",
            _require_non_negative("mobility_tolerance", self.mobility_tolerance),
        )

    @property
    def retention_time_tolerance(self) -> float:
        """Alias for :attr:`rt_tolerance`."""
        return self.rt_tolerance

    @property
    def rt_window(self) -> Tuple[float, float]:
        return (
            self.retention_time - self.rt_tolerance,
            self.retention_time + self.rt_tolerance,
        )

    @property
    def mz_window(self) -> Tuple[float, float]:
        return (self.mz - self.mz_tolerance, self.mz + self.mz_tolerance)

    @property
    def mobility_window(self) -> Tuple[float, float]:
        return (
            self.mobility - self.mobility_tolerance,
            self.mobility + self.mobility_tolerance,
        )

    @property
    def fingerprint(self) -> Tuple[float, ...]:
        return (
            self.retention_time,
            self.mz,
            self.mobility,
            self.rt_tolerance,
            self.mz_tolerance,
            self.mobility_tolerance,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "retention_time": self.retention_time,
            "mz": self.mz,
            "mobility": self.mobility,
            "rt_tolerance": self.rt_tolerance,
            "mz_tolerance": self.mz_tolerance,
            "mobility_tolerance": self.mobility_tolerance,
            "mz_ppm": self.mz_ppm,
        }


Query3D = SearchQuery


@dataclass(frozen=True)
class DimensionEvidence:
    """Evidence for one dimension of one original peak."""

    dimension: str
    observed: float
    observed_interval: Tuple[float, float]
    target: float
    tolerance: float
    target_interval: Tuple[float, float]
    status: str
    delta: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "observed": self.observed,
            "observed_interval": list(self.observed_interval),
            "target": self.target,
            "tolerance": self.tolerance,
            "target_interval": list(self.target_interval),
            "status": self.status,
            "delta": self.delta,
        }


@dataclass(frozen=True)
class CalibrationBasis:
    """
    Calibration evidence attached to a hit.

    The basis records the exact segment, coefficients, measured mobility and
    propagated error interval that produced the calibrated mobility.
    """

    calibration_name: str
    calibration_revision: Optional[Any]
    segment_index: int
    segment_label: str
    measured_mobility: float
    calibrated_mobility: float
    calibrated_lower: float
    calibrated_upper: float
    segment_lower: float
    segment_upper: float
    slope: float
    intercept: float
    error_low: float
    error_high: float

    @property
    def calibrated_interval(self) -> Tuple[float, float]:
        return (self.calibrated_lower, self.calibrated_upper)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calibration_name": self.calibration_name,
            "calibration_revision": self.calibration_revision,
            "segment_index": self.segment_index,
            "segment_label": self.segment_label,
            "measured_mobility": self.measured_mobility,
            "calibrated_mobility": self.calibrated_mobility,
            "calibrated_interval": list(self.calibrated_interval),
            "segment_lower": (
                None if math.isinf(self.segment_lower) else self.segment_lower
            ),
            "segment_upper": (
                None if math.isinf(self.segment_upper) else self.segment_upper
            ),
            "slope": self.slope,
            "intercept": self.intercept,
            "error_low": self.error_low,
            "error_high": self.error_high,
        }


@dataclass(frozen=True)
class Hit3D:
    """
    One joint three-dimensional hit on one original peak.

    ``status`` is ``"definite"`` or ``"possible"``. The ``dimensions`` mapping
    always contains retention time, m/z and mobility from the same
    ``peak_id``.
    """

    peak_id: Hashable
    status: str
    retention_time: float
    mz: float
    intensity: float
    measured_mobility: float
    calibrated_mobility: float
    calibration: CalibrationBasis
    dimensions: Mapping[str, DimensionEvidence]
    score: float
    reasons: Tuple[str, ...] = ()
    source: Optional[str] = None
    same_peak: bool = True

    @property
    def peak(self) -> Hashable:
        """Alias for :attr:`peak_id`."""
        return self.peak_id

    @property
    def calibration_basis(self) -> CalibrationBasis:
        """Alias for :attr:`calibration`."""
        return self.calibration

    @property
    def is_definite(self) -> bool:
        return self.status == "definite"

    @property
    def is_possible(self) -> bool:
        return self.status == "possible"

    _DIMENSION_ALIASES = {
        "rt": "retention_time",
        "retention time": "retention_time",
        "retention_time": "retention_time",
        "m/z": "mz",
        "mz": "mz",
        "mobility": "mobility",
        "ion_mobility": "mobility",
        "ion mobility": "mobility",
    }

    def __getitem__(self, dimension: str) -> DimensionEvidence:
        key = self._DIMENSION_ALIASES.get(str(dimension).strip().lower(), dimension)
        return self.dimensions[key]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "peak_id": self.peak_id,
            "status": self.status,
            "same_peak": self.same_peak,
            "retention_time": self.retention_time,
            "mz": self.mz,
            "intensity": self.intensity,
            "measured_mobility": self.measured_mobility,
            "calibrated_mobility": self.calibrated_mobility,
            "score": self.score,
            "reasons": list(self.reasons),
            "source": self.source,
            "dimensions": {
                name: evidence.to_dict()
                for name, evidence in self.dimensions.items()
            },
            "calibration": self.calibration.to_dict(),
        }


@dataclass(frozen=True)
class SearchResult:
    """Result of one joint three-dimensional query."""

    query: SearchQuery
    hits: Tuple[Hit3D, ...]
    uncalibrated_peak_ids: Tuple[Hashable, ...]
    evaluated_peak_count: int
    recomputed_peak_count: int
    cached_peak_count: int
    calibration: MobilityCalibration
    calibration_revision: int
    index_revision: int

    @property
    def definite_hits(self) -> Tuple[Hit3D, ...]:
        """Hits whose complete observed intervals lie inside all windows."""
        return tuple(hit for hit in self.hits if hit.status == "definite")

    @property
    def possible_hits(self) -> Tuple[Hit3D, ...]:
        """Hits that overlap all windows but are not fully contained."""
        return tuple(hit for hit in self.hits if hit.status == "possible")

    @property
    def definite(self) -> Tuple[Hit3D, ...]:
        """Alias for :attr:`definite_hits`."""
        return self.definite_hits

    @property
    def possible(self) -> Tuple[Hit3D, ...]:
        """Alias for :attr:`possible_hits`."""
        return self.possible_hits

    @property
    def confirmed_hits(self) -> Tuple[Hit3D, ...]:
        """Readable alias for :attr:`definite_hits`."""
        return self.definite_hits

    @property
    def tentative_hits(self) -> Tuple[Hit3D, ...]:
        """Readable alias for :attr:`possible_hits`."""
        return self.possible_hits

    @property
    def all_hits(self) -> Tuple[Hit3D, ...]:
        """Alias for :attr:`hits`."""
        return self.hits

    @property
    def hit_peak_ids(self) -> Tuple[Hashable, ...]:
        return tuple(hit.peak_id for hit in self.hits)

    def by_peak_id(self, peak_id: Hashable) -> Optional[Hit3D]:
        for hit in self.hits:
            if hit.peak_id == peak_id:
                return hit
        return None

    def __iter__(self):
        return iter(self.hits)

    def __len__(self) -> int:
        return len(self.hits)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query.to_dict(),
            "definite_hits": [hit.to_dict() for hit in self.definite_hits],
            "possible_hits": [hit.to_dict() for hit in self.possible_hits],
            "hits": [hit.to_dict() for hit in self.hits],
            "uncalibrated_peak_ids": list(self.uncalibrated_peak_ids),
            "evaluated_peak_count": self.evaluated_peak_count,
            "recomputed_peak_count": self.recomputed_peak_count,
            "cached_peak_count": self.cached_peak_count,
            "calibration": self.calibration.to_dict(),
            "calibration_revision": self.calibration_revision,
            "index_revision": self.index_revision,
        }


@dataclass(frozen=True)
class RecalibrationReport:
    """Summary of an incremental calibration correction."""

    previous_calibration_revision: int
    calibration_revision: int
    index_revision: int
    changed_segment_indices: Tuple[int, ...]
    changed_segment_labels: Tuple[str, ...]
    recomputed_peak_ids: Tuple[Hashable, ...]
    unaffected_peak_ids: Tuple[Hashable, ...]
    projection_rows_recomputed: int
    projections_synchronized: int

    @property
    def affected_peak_ids(self) -> Tuple[Hashable, ...]:
        return self.recomputed_peak_ids

    def to_dict(self) -> Dict[str, Any]:
        return {
            "previous_calibration_revision": self.previous_calibration_revision,
            "calibration_revision": self.calibration_revision,
            "index_revision": self.index_revision,
            "changed_segment_indices": list(self.changed_segment_indices),
            "changed_segment_labels": list(self.changed_segment_labels),
            "recomputed_peak_ids": list(self.recomputed_peak_ids),
            "unaffected_peak_ids": list(self.unaffected_peak_ids),
            "projection_rows_recomputed": self.projection_rows_recomputed,
            "projections_synchronized": self.projections_synchronized,
        }


_PEAK_CACHE_SIZE = 64


@dataclass
class _PeakState:
    peak: Peak3D
    calibration: Optional[_CalibratedMobility]
    basis: Optional[CalibrationBasis]
    cache: "OrderedDict[Tuple[float, ...], Any]" = field(
        default_factory=OrderedDict
    )
    recomputations: int = 0

    def cached(self, key):
        """Return a cached outcome and mark it as recently used."""
        try:
            value = self.cache.pop(key)
        except KeyError:
            return None, False
        self.cache[key] = value
        return value, True

    def remember(self, key, value):
        self.cache[key] = value
        while len(self.cache) > _PEAK_CACHE_SIZE:
            self.cache.popitem(last=False)


@dataclass(frozen=True)
class _MatchOutcome:
    status: Optional[str]
    dimensions: Dict[str, DimensionEvidence]
    score: float
    reasons: Tuple[str, ...]


@dataclass(frozen=True)
class ProjectionRow:
    """
    One original peak projected onto the retention time/m/z plane.

    The row keeps the calibrated mobility interval so the two-dimensional
    view can show that the projection is still tied to the mobility
    calibration. ``hit_status`` is the optional overlay set by
    :meth:`Projection2D.annotate`.
    """

    peak_id: Hashable
    retention_time: float
    mz: float
    intensity: float
    measured_mobility: float
    calibrated_mobility: Optional[float]
    calibrated_interval: Optional[Tuple[float, float]]
    segment_index: Optional[int]
    segment_label: Optional[str]
    peak_revision: int
    hit_status: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "peak_id": self.peak_id,
            "retention_time": self.retention_time,
            "mz": self.mz,
            "intensity": self.intensity,
            "measured_mobility": self.measured_mobility,
            "calibrated_mobility": self.calibrated_mobility,
            "calibrated_interval": (
                None
                if self.calibrated_interval is None
                else list(self.calibrated_interval)
            ),
            "segment_index": self.segment_index,
            "segment_label": self.segment_label,
            "peak_revision": self.peak_revision,
            "hit_status": self.hit_status,
        }


class Projection2D:
    """
    Live retention time/m/z projection of an indexed set of original peaks.

    Rows are kept per original peak, so peaks that share retention time and
    m/z are not merged. The view is attached to its index and is synchronized
    automatically after every calibration correction. Only rows of recomputed
    peaks are rebuilt; other rows keep their values.
    """

    def __init__(
        self,
        index: "PeakIndex3D",
        axes: Tuple[str, str] = ("retention_time", "mz"),
    ):
        if tuple(axes) != ("retention_time", "mz"):
            raise Search3DError(
                "the supported two-dimensional projection axes are "
                "('retention_time', 'mz')"
            )
        self._index_ref = weakref.ref(index)
        self.axes = tuple(axes)
        self._rows: Dict[Hashable, ProjectionRow] = {}
        self._annotations: Dict[Hashable, str] = {}
        self.calibration_revision = index.calibration_revision
        self.index_revision = index.revision
        self.last_recomputed_peak_ids: Tuple[Hashable, ...] = ()
        self.last_refreshed_peak_ids: Tuple[Hashable, ...] = ()
        self.rows_rebuilt = 0
        for peak_id, state in index._states.items():
            self._rows[peak_id] = self._make_row(state, None)
        index._register_projection(self)

    @property
    def index(self) -> "PeakIndex3D":
        index = self._index_ref()
        if index is None:
            raise Search3DError("the projection index is no longer available")
        return index

    @property
    def is_synchronized(self) -> bool:
        return (
            self.calibration_revision == self.index.calibration_revision
            and self.index_revision == self.index.revision
        )

    @property
    def annotations(self) -> Dict[Hashable, str]:
        """Current hit status overlay, keyed by original peak id."""
        return dict(self._annotations)

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self):
        return iter(self.rows)

    def rows(self):
        """Return projection rows sorted by retention time and m/z.

        Ties keep the original insertion order of the index, so rows are
        stable across repeated reads and calibration corrections.
        """
        self.synchronize()
        return tuple(
            sorted(
                self._rows.values(),
                key=lambda row: (row.retention_time, row.mz),
            )
        )

    @property
    def row_list(self) -> Tuple[ProjectionRow, ...]:
        return self.rows()

    def row_for(self, peak_id: Hashable) -> ProjectionRow:
        self.synchronize()
        return self._rows[peak_id]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "axes": list(self.axes),
            "index_revision": self.index_revision,
            "calibration_revision": self.calibration_revision,
            "synchronized": self.is_synchronized,
            "rows": [row.to_dict() for row in self.rows()],
            "annotations": dict(self._annotations),
        }

    def to_plot_data(self) -> Dict[str, Any]:
        """
        Return the current view as aligned plotting columns.

        The returned arrays always cover the same original peaks in the same
        order, including ``definite``/``possible`` hit overlays when
        :meth:`annotate` was called with a current search result.
        """
        rows = self.rows()
        return {
            "x": [row.retention_time for row in rows],
            "y": [row.mz for row in rows],
            "intensity": [row.intensity for row in rows],
            "calibrated_mobility": [row.calibrated_mobility for row in rows],
            "calibrated_interval": [row.calibrated_interval for row in rows],
            "hit_status": [row.hit_status for row in rows],
            "peak_id": [row.peak_id for row in rows],
            "calibration_revision": self.calibration_revision,
            "index_revision": self.index_revision,
            "synchronized": self.is_synchronized,
        }

    def status_counts(self) -> Dict[str, int]:
        """Count ``definite`` and ``possible`` hit overlays in this view."""
        counts = {"definite": 0, "possible": 0, "none": 0}
        for row in self.rows():
            if row.hit_status in ("definite", "possible"):
                counts[row.hit_status] += 1
            else:
                counts["none"] += 1
        return counts

    def annotate(self, result: SearchResult) -> "Projection2D":
        """
        Overlay the hit status of a search result on this view.

        Annotations are only accepted for the current index and calibration
        revision. The view is synchronized first, so a result computed just
        before a table correction cannot be drawn on corrected rows.
        """
        if not isinstance(result, SearchResult):
            raise Search3DError("annotate expects a SearchResult")
        self.synchronize()
        if result.index_revision != self.index.revision:
            raise Search3DError(
                "search result was computed before the index changed"
            )
        if result.calibration_revision != self.index.calibration_revision:
            raise Search3DError(
                "search result was computed with an older calibration revision"
            )
        previous_ids = set(self._annotations)
        self._annotations = {
            hit.peak_id: hit.status for hit in result.hits if hit.peak_id in self._rows
        }
        for peak_id in previous_ids | set(self._annotations):
            state = self.index._states.get(peak_id)
            if state is None:
                continue
            row = self._rows[peak_id]
            status = self._annotations.get(peak_id)
            if row.hit_status != status:
                self._rows[peak_id] = replace(row, hit_status=status)
        return self

    def synchronize(self) -> "Projection2D":
        """Apply any pending calibration update to this view."""
        index = self.index
        if not self.is_synchronized:
            index._synchronize_projection(self)
        return self

    def _make_row(self, state: _PeakState, hit_status: Optional[str]) -> ProjectionRow:
        calibration = state.calibration
        basis = state.basis
        return ProjectionRow(
            peak_id=state.peak.peak_id,
            retention_time=state.peak.retention_time,
            mz=state.peak.mz,
            intensity=state.peak.intensity,
            measured_mobility=state.peak.measured_mobility,
            calibrated_mobility=None if calibration is None else calibration.value,
            calibrated_interval=(
                None if calibration is None else (calibration.lower, calibration.upper)
            ),
            segment_index=None if basis is None else basis.segment_index,
            segment_label=None if basis is None else basis.segment_label,
            peak_revision=state.recomputations,
            hit_status=hit_status,
        )

    def _apply_update(self, report: RecalibrationReport) -> int:
        index = self.index
        recomputed = set(report.recomputed_peak_ids)
        # Peaks whose segment index or label changed keep the same calibrated
        # numbers but need refreshed row metadata.
        refreshed = []
        rebuilt = 0
        for peak_id, row in list(self._rows.items()):
            state = index._states.get(peak_id)
            if state is None:
                del self._rows[peak_id]
                self._annotations.pop(peak_id, None)
                continue
            hit_status = self._annotations.get(peak_id)
            if peak_id in recomputed:
                self._annotations.pop(peak_id, None)
                hit_status = None
                self._rows[peak_id] = self._make_row(state, hit_status)
                rebuilt += 1
                refreshed.append(peak_id)
                continue
            basis = state.basis
            segment_index = None if basis is None else basis.segment_index
            segment_label = None if basis is None else basis.segment_label
            if row.segment_index != segment_index or row.segment_label != segment_label:
                self._rows[peak_id] = replace(
                    row,
                    segment_index=segment_index,
                    segment_label=segment_label,
                )
                refreshed.append(peak_id)
        for peak_id, state in index._states.items():
            if peak_id not in self._rows:
                self._rows[peak_id] = self._make_row(
                    state, self._annotations.get(peak_id)
                )
                rebuilt += 1
                refreshed.append(peak_id)
        self.calibration_revision = report.calibration_revision
        self.index_revision = report.index_revision
        self.last_recomputed_peak_ids = tuple(report.recomputed_peak_ids)
        self.last_refreshed_peak_ids = tuple(refreshed)
        self.rows_rebuilt += rebuilt
        return rebuilt


class PeakIndex3D:
    """
    Index of original three-dimensional peaks with a piecewise calibration.

    Each peak is calibrated once. Searches evaluate a peak only when its
    calibration values changed or when the query has not been seen for that
    peak; results for unaffected peaks are reused. A calibration correction
    therefore recalibrates exactly the peaks whose segment changed.

    Arguments:
        peaks (iterable): :class:`Peak3D` objects.
        calibration (MobilityCalibration): piecewise mobility calibration.
            If omitted, an identity table covering all finite mobilities is
            used with zero error.

    Note:
        Peaks that fall outside every calibration segment are kept as
        uncalibrated and are reported by :attr:`SearchResult.uncalibrated_peak_ids`.
        They are never silently assigned to a neighbouring segment.
    """

    def __init__(
        self,
        peaks: Iterable[Peak3D] = (),
        calibration: Optional[MobilityCalibration] = None,
    ):
        self._states: Dict[Hashable, _PeakState] = {}
        self._calibration = (
            calibration if calibration is not None else _identity_calibration()
        )
        self._projections: "weakref.WeakSet[Projection2D]" = weakref.WeakSet()
        self.revision = 0
        self.calibration_revision = 0
        self._last_report: Optional[RecalibrationReport] = None
        self.add_peaks(peaks)

    @property
    def calibration(self) -> MobilityCalibration:
        return self._calibration

    @property
    def calibration_revision(self) -> int:
        return self._calibration_revision

    @calibration_revision.setter
    def calibration_revision(self, value: int):
        self._calibration_revision = int(value)

    @property
    def peaks(self) -> Tuple[Peak3D, ...]:
        return tuple(state.peak for state in self._states.values())

    def __len__(self) -> int:
        return len(self._states)

    def __iter__(self):
        return iter(self.peaks)

    def __contains__(self, peak_id: Hashable) -> bool:
        return peak_id in self._states

    def get_peak(self, peak_id: Hashable) -> Peak3D:
        return self._states[peak_id].peak

    def add_peak(self, peak: Peak3D) -> "PeakIndex3D":
        """Add one original peak and calibrate it."""
        return self.add_peaks((peak,))

    def add_peaks(self, peaks: Iterable[Peak3D]) -> "PeakIndex3D":
        """
        Add several original peaks in one revision.

        All peaks are validated before any peak is inserted, so a duplicate
        or invalid peak leaves the index unchanged.
        """
        pending = []
        seen = set()
        for peak in peaks:
            if not isinstance(peak, Peak3D):
                raise PeakValidationError("add_peaks expects Peak3D instances")
            if peak.peak_id in self._states or peak.peak_id in seen:
                raise DuplicatePeakError(
                    "peak_id {0!r} is already present".format(peak.peak_id)
                )
            seen.add(peak.peak_id)
            pending.append(peak)
        if not pending:
            return self
        added_ids = []
        for peak in pending:
            state = _PeakState(peak=peak, calibration=None, basis=None)
            self._states[peak.peak_id] = state
            self._initialise_state(state, self._calibration)
            added_ids.append(peak.peak_id)
        self.revision += 1
        report = RecalibrationReport(
            previous_calibration_revision=self.calibration_revision,
            calibration_revision=self.calibration_revision,
            index_revision=self.revision,
            changed_segment_indices=(),
            changed_segment_labels=(),
            recomputed_peak_ids=tuple(added_ids),
            unaffected_peak_ids=tuple(
                peak_id for peak_id in self._states if peak_id not in seen
            ),
            projection_rows_recomputed=0,
            projections_synchronized=len(self._projections),
        )
        for projection in list(self._projections):
            projection._apply_update(report)
        return self

    def remove_peak(self, peak_id: Hashable) -> Peak3D:
        """Remove one original peak from the index and from all views."""
        state = self._states.pop(peak_id)
        self.revision += 1
        for projection in list(self._projections):
            projection._apply_update(
                RecalibrationReport(
                    previous_calibration_revision=self.calibration_revision,
                    calibration_revision=self.calibration_revision,
                    index_revision=self.revision,
                    changed_segment_indices=(),
                    changed_segment_labels=(),
                    recomputed_peak_ids=(),
                    unaffected_peak_ids=tuple(self._states),
                    projection_rows_recomputed=0,
                    projections_synchronized=len(self._projections),
                )
            )
        return state.peak

    def _initialise_state(self, state: _PeakState, calibration: MobilityCalibration):
        try:
            calibrated = calibration.calibrate(state.peak.measured_mobility)
        except UncalibratedMobilityError:
            state.calibration = None
            state.basis = None
            state.cache.clear()
            return
        state.calibration = calibrated
        state.basis = self._basis(calibrated)
        state.cache.clear()

    def _basis(
        self,
        calibrated: _CalibratedMobility,
        calibration: Optional[MobilityCalibration] = None,
    ) -> CalibrationBasis:
        calibration = self._calibration if calibration is None else calibration
        segment = calibrated.segment
        return CalibrationBasis(
            calibration_name=calibration.name,
            calibration_revision=calibration.revision,
            segment_index=calibrated.segment_index,
            segment_label=segment.label,
            measured_mobility=calibrated.measured,
            calibrated_mobility=calibrated.value,
            calibrated_lower=calibrated.lower,
            calibrated_upper=calibrated.upper,
            segment_lower=segment.lower,
            segment_upper=segment.upper,
            slope=segment.slope,
            intercept=segment.intercept,
            error_low=segment.error_low,
            error_high=segment.error_high,
        )

    def _register_projection(self, projection: Projection2D):
        self._projections.add(projection)

    def projection_2d(self) -> Projection2D:
        """Create a new live two-dimensional projection view."""
        return Projection2D(self)

    def _synchronize_projection(self, projection: Projection2D):
        """Bring a view that missed an update back in line with the index."""
        index = self
        rebuilt = 0
        for peak_id, row in list(projection._rows.items()):
            state = index._states.get(peak_id)
            if state is None:
                del projection._rows[peak_id]
                projection._annotations.pop(peak_id, None)
                continue
            if row.peak_revision != state.recomputations:
                projection._annotations.pop(peak_id, None)
                projection._rows[peak_id] = projection._make_row(state, None)
                rebuilt += 1
            elif row.segment_index != (
                None if state.basis is None else state.basis.segment_index
            ) or row.segment_label != (
                None if state.basis is None else state.basis.segment_label
            ):
                projection._rows[peak_id] = replace(
                    row,
                    segment_index=None if state.basis is None else state.basis.segment_index,
                    segment_label=None if state.basis is None else state.basis.segment_label,
                )
        for peak_id, state in index._states.items():
            if peak_id not in projection._rows:
                projection._rows[peak_id] = projection._make_row(
                    state, projection._annotations.get(peak_id)
                )
                rebuilt += 1
        projection.calibration_revision = self.calibration_revision
        projection.index_revision = self.revision
        projection.last_recomputed_peak_ids = ()
        projection.last_refreshed_peak_ids = ()
        projection.rows_rebuilt += rebuilt

    def _changed_segments(
        self, old: MobilityCalibration, new: MobilityCalibration
    ) -> Tuple[Tuple[int, ...], Tuple[str, ...]]:
        changed_indices = []
        changed_labels = []
        count = max(len(old), len(new))
        for index in range(count):
            old_segment = old.segments[index] if index < len(old) else None
            new_segment = new.segments[index] if index < len(new) else None
            old_signature = None if old_segment is None else old_segment.signature
            new_signature = None if new_segment is None else new_segment.signature
            if old_signature != new_signature:
                changed_indices.append(index)
                label = (
                    new_segment.label
                    if new_segment is not None and new_segment.label
                    else (old_segment.label if old_segment is not None else "")
                )
                changed_labels.append(label)
        return tuple(changed_indices), tuple(changed_labels)

    def update_calibration(self, calibration: MobilityCalibration) -> RecalibrationReport:
        """
        Replace the calibration table and update only affected original peaks.

        A peak is affected when it was previously uncalibrated and now falls
        into a segment, when it now falls outside every segment, or when the
        numeric calibration values of its segment changed. Peaks in unchanged
        segments keep their calibrated values and cached query evaluations.

        Returns:
            report (RecalibrationReport): affected and unaffected peak ids and
            the projection synchronization counts.
        """
        if not isinstance(calibration, MobilityCalibration):
            raise CalibrationError("update_calibration expects a MobilityCalibration")
        old_calibration = self._calibration
        previous_revision = self.calibration_revision
        changed_indices, changed_labels = self._changed_segments(
            old_calibration, calibration
        )
        recomputed: List[Hashable] = []
        unaffected: List[Hashable] = []
        for peak_id, state in self._states.items():
            measured = state.peak.measured_mobility
            new_index = calibration.segment_index_for(measured)
            if new_index is None:
                if state.calibration is None:
                    unaffected.append(peak_id)
                else:
                    state.calibration = None
                    state.basis = None
                    state.cache.clear()
                    state.recomputations += 1
                    recomputed.append(peak_id)
                continue
            new_calibration = calibration.calibrate(measured)
            unchanged = (
                state.calibration is not None
                and state.calibration.value == new_calibration.value
                and state.calibration.lower == new_calibration.lower
                and state.calibration.upper == new_calibration.upper
            )
            if unchanged:
                # The numbers that produced this peak's calibrated mobility
                # are identical. Keep cached query results and only refresh
                # the evidence metadata.
                state.calibration = new_calibration
                state.basis = self._basis(state.calibration, calibration)
                unaffected.append(peak_id)
            else:
                state.calibration = new_calibration
                state.basis = self._basis(state.calibration, calibration)
                state.cache.clear()
                state.recomputations += 1
                recomputed.append(peak_id)
        self._calibration = calibration
        self.calibration_revision = previous_revision + 1
        self.revision += 1
        recomputed_tuple = tuple(recomputed)
        unaffected_tuple = tuple(unaffected)
        projections = list(self._projections)
        rows_recomputed = sum(
            projection._apply_update(
                RecalibrationReport(
                    previous_calibration_revision=previous_revision,
                    calibration_revision=self.calibration_revision,
                    index_revision=self.revision,
                    changed_segment_indices=changed_indices,
                    changed_segment_labels=changed_labels,
                    recomputed_peak_ids=recomputed_tuple,
                    unaffected_peak_ids=unaffected_tuple,
                    projection_rows_recomputed=0,
                    projections_synchronized=len(projections),
                )
            )
            for projection in projections
        )
        report = RecalibrationReport(
            previous_calibration_revision=previous_revision,
            calibration_revision=self.calibration_revision,
            index_revision=self.revision,
            changed_segment_indices=changed_indices,
            changed_segment_labels=changed_labels,
            recomputed_peak_ids=recomputed_tuple,
            unaffected_peak_ids=unaffected_tuple,
            projection_rows_recomputed=rows_recomputed,
            projections_synchronized=len(projections),
        )
        self._last_report = report
        return report

    def correct_calibration_segment(
        self, segment_index: int, *, revision: Optional[Any] = None, **changes
    ) -> RecalibrationReport:
        """Correct one segment and update the index incrementally."""
        return self.update_calibration(
            self._calibration.with_segment(
                segment_index, revision=revision, **changes
            )
        )

    @property
    def last_recalibration(self) -> Optional[RecalibrationReport]:
        return self._last_report

    def calibration_basis_for(self, peak_id: Hashable) -> Optional[CalibrationBasis]:
        """Return the current calibration evidence for one original peak."""
        return self._states[peak_id].basis

    def _evaluate(self, state: _PeakState, query: SearchQuery) -> Optional[_MatchOutcome]:
        calibration = state.calibration
        if calibration is None:
            return None
        peak = state.peak
        rt_low = peak.retention_time - peak.retention_time_uncertainty
        rt_high = peak.retention_time + peak.retention_time_uncertainty
        mz_low = peak.mz - peak.mz_uncertainty
        mz_high = peak.mz + peak.mz_uncertainty
        mobility_low = calibration.lower
        mobility_high = calibration.upper

        dimensions = {
            "retention_time": self._dimension_evidence(
                "retention_time",
                peak.retention_time,
                (rt_low, rt_high),
                query.retention_time,
                query.rt_tolerance,
                query.rt_window,
            ),
            "mz": self._dimension_evidence(
                "mz",
                peak.mz,
                (mz_low, mz_high),
                query.mz,
                query.mz_tolerance,
                query.mz_window,
            ),
            "mobility": self._dimension_evidence(
                "mobility",
                calibration.value,
                (mobility_low, mobility_high),
                query.mobility,
                query.mobility_tolerance,
                query.mobility_window,
            ),
        }
        statuses = [evidence.status for evidence in dimensions.values()]
        if any(status == "miss" for status in statuses):
            return None
        status = (
            "definite"
            if all(item == "definite" for item in statuses)
            else "possible"
        )
        reasons = []
        for name in _DIMENSIONS:
            evidence = dimensions[name]
            if evidence.status == "possible":
                reasons.append(
                    "{0} interval {1!r} overlaps but is not contained in "
                    "target window {2!r}".format(
                        name,
                        evidence.observed_interval,
                        evidence.target_interval,
                    )
                )
        score = max(evidence.delta for evidence in dimensions.values())
        return _MatchOutcome(
            status=status,
            dimensions=dimensions,
            score=score,
            reasons=tuple(reasons),
        )

    @staticmethod
    def _dimension_evidence(
        dimension: str,
        observed: float,
        observed_interval: Tuple[float, float],
        target: float,
        tolerance: float,
        target_interval: Tuple[float, float],
    ) -> DimensionEvidence:
        low, high = observed_interval
        target_low, target_high = target_interval
        if _interval_contains(target_low, target_high, low, high):
            status = "definite"
        elif _interval_overlaps(low, high, target_low, target_high):
            status = "possible"
        else:
            status = "miss"
        if target < low:
            delta = low - target
        elif target > high:
            delta = target - high
        else:
            delta = 0.0
        return DimensionEvidence(
            dimension=dimension,
            observed=observed,
            observed_interval=observed_interval,
            target=target,
            tolerance=tolerance,
            target_interval=target_interval,
            status=status,
            delta=delta,
        )

    def search(
        self, query: SearchQuery, *, projection: Optional[Projection2D] = None
    ) -> SearchResult:
        """
        Search all original peaks with one joint three-dimensional condition.

        Each peak is evaluated as one unit. The returned hits are unique per
        original peak; the three dimensions are never assembled from
        different peaks.

        Arguments:
            query (SearchQuery): joint query window.

        Keyword Arguments:
            projection (Projection2D, optional): annotate this view with the
                resulting hit statuses.

        Returns:
            result (SearchResult): definite and possible hits with saved
            calibration evidence.
        """
        if not isinstance(query, SearchQuery):
            raise Search3DError("search expects a SearchQuery")
        key = query.fingerprint
        hits: List[Hit3D] = []
        uncalibrated: List[Hashable] = []
        recomputed = 0
        cached = 0
        for peak_id, state in self._states.items():
            if state.calibration is None:
                uncalibrated.append(peak_id)
                continue
            outcome, was_cached = state.cached(key)
            if was_cached:
                cached += 1
            else:
                outcome = self._evaluate(state, query)
                # Cache misses too; an unaffected peak keeps the same outcome.
                state.remember(key, outcome)
                recomputed += 1
            if outcome is None:
                continue
            peak = state.peak
            hits.append(
                Hit3D(
                    peak_id=peak_id,
                    status=outcome.status,
                    retention_time=peak.retention_time,
                    mz=peak.mz,
                    intensity=peak.intensity,
                    measured_mobility=state.calibration.measured,
                    calibrated_mobility=state.calibration.value,
                    calibration=state.basis,
                    dimensions=outcome.dimensions,
                    score=outcome.score,
                    reasons=outcome.reasons,
                    source=peak.source,
                )
            )
        # Sort is stable: equal-status, equal-score hits keep the original
        # peak insertion order.
        hits.sort(key=lambda hit: (0 if hit.status == "definite" else 1, hit.score))
        result = SearchResult(
            query=query,
            hits=tuple(hits),
            uncalibrated_peak_ids=tuple(uncalibrated),
            evaluated_peak_count=len(self._states),
            recomputed_peak_count=recomputed,
            cached_peak_count=cached,
            calibration=self._calibration,
            calibration_revision=self.calibration_revision,
            index_revision=self.revision,
        )
        if projection is not None:
            projection.synchronize()
            projection.annotate(result)
        return result

    def calibration_basis_records(self) -> List[Dict[str, Any]]:
        """
        Export the saved calibration evidence of every original peak.

        Each record contains the peak coordinates, the calibrated mobility
        interval and the full segment evidence used to produce it.
        """
        records = []
        for peak_id, state in self._states.items():
            records.append(
                {
                    "peak_id": peak_id,
                    "peak": state.peak.to_dict(),
                    "calibrated": state.calibration is not None,
                    "calibration_revision": self._calibration.revision,
                    "basis": None if state.basis is None else state.basis.to_dict(),
                }
            )
        return records

    def save_calibration_basis(self, path: str) -> None:
        """Write :meth:`calibration_basis_records` as UTF-8 JSON."""
        import json

        payload = {
            "calibration": self._calibration.to_dict(),
            "calibration_revision": self.calibration_revision,
            "index_revision": self.revision,
            "peaks": self.calibration_basis_records(),
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "revision": self.revision,
            "calibration_revision": self.calibration_revision,
            "calibration": self._calibration.to_dict(),
            "peaks": [state.peak.to_dict() for state in self._states.values()],
            "calibration_basis": {
                str(peak_id): None if state.basis is None else state.basis.to_dict()
                for peak_id, state in self._states.items()
            },
        }


ThreeDSearchIndex = PeakIndex3D


def _identity_calibration() -> MobilityCalibration:
    return MobilityCalibration(
        segments=(
            MobilityCalibrationSegment(
                lower=-float("inf"),
                upper=float("inf"),
                slope=1.0,
                intercept=0.0,
                error=0.0,
                label="identity",
            ),
        ),
        name="identity ion mobility calibration",
    )


def _time_to_minutes(value: float, unit: Optional[str]) -> float:
    if unit is None:
        return value
    key = str(unit).strip().lower()
    try:
        factor = _TIME_UNIT_TO_MINUTES[key]
    except KeyError as error:
        raise Search3DError(
            "unknown retention time unit {0!r}; expected one of {1}".format(
                unit, sorted(_TIME_UNIT_TO_MINUTES)
            )
        ) from error
    return value * factor


def extract_peaks3d(
    spectrum,
    *,
    retention_time: Optional[float] = None,
    retention_time_unit: Optional[str] = None,
    mobility_array_name: Optional[str] = None,
    peak_type: str = "centroided",
    peak_id_prefix: Optional[str] = None,
    rt_uncertainty: float = 0.0,
    mz_uncertainty: Optional[float] = None,
    mz_ppm: Optional[float] = None,
) -> List[Peak3D]:
    """
    Build original three-dimensional peaks from one pymzML spectrum.

    Every returned peak uses one m/z value, its intensity and the mobility
    value at the same array position. No coordinates are combined across
    peaks.

    Arguments:
        spectrum: pymzML :class:`~pymzml.spec.Spectrum` (or compatible
            object) exposing ``peaks``, ``scan_time``, ``ID`` and
            ``get_ion_mobility``.

    Keyword Arguments:
        retention_time (float, optional): explicit retention time. If
            omitted, the spectrum scan time is used and converted to minutes.
        retention_time_unit (str, optional): unit of ``retention_time``.
            When given, the value is converted to minutes.
        mobility_array_name (str, optional): explicit mzML mobility array
            name. When omitted, the spectrum's preferred mobility array is
            used.
        peak_type (str): ``peaks`` type passed to the spectrum.
        peak_id_prefix (str, optional): prefix for generated peak ids.
            Defaults to the spectrum native id.
        rt_uncertainty (float): optional retention time half-width applied to
            every peak.
        mz_uncertainty (float, optional): optional absolute m/z half-width.
        mz_ppm (float, optional): optional relative m/z half-width in ppm.
            Cannot be combined with ``mz_uncertainty``.

    Returns:
        peaks (list[Peak3D]): one peak per spectrum peak.

    Raises:
        Search3DError: if no mobility array is present or the array length
            does not match the peak count.
    """
    peaks = np.asarray(spectrum.peaks(peak_type), dtype=float)
    if peaks.ndim != 2 or peaks.shape[1] < 2:
        raise Search3DError("spectrum peaks must be m/z and intensity pairs")
    get_mobility = getattr(spectrum, "get_ion_mobility", None)
    if get_mobility is None:
        raise Search3DError("spectrum does not provide get_ion_mobility()")
    mobility = get_mobility(mobility_array_name)
    if mobility is None:
        available = ()
        get_arrays = getattr(spectrum, "get_all_arrays_in_spec", None)
        if get_arrays is not None:
            available = tuple(get_arrays())
        raise Search3DError(
            "spectrum has no ion mobility array; available arrays: "
            "{0}".format(list(available))
        )
    mobility = np.asarray(mobility, dtype=float)
    if mobility.ndim != 1 or len(mobility) != len(peaks):
        raise Search3DError(
            "ion mobility array length {0} does not match peak count {1}".format(
                len(mobility), len(peaks)
            )
        )

    if retention_time is None:
        scan_time = getattr(spectrum, "scan_time", None)
        if scan_time is None:
            raise Search3DError(
                "spectrum has no scan time; pass retention_time explicitly"
            )
        value, unit = scan_time
        if value is None:
            raise Search3DError(
                "spectrum has no scan time; pass retention_time explicitly"
            )
        retention_time = _time_to_minutes(float(value), unit)
    elif retention_time_unit is not None:
        retention_time = _time_to_minutes(float(retention_time), retention_time_unit)
    else:
        retention_time = float(retention_time)

    if mz_uncertainty is not None and mz_ppm is not None:
        raise Search3DError("pass either mz_uncertainty or mz_ppm, not both")
    rt_uncertainty = _require_non_negative("rt_uncertainty", rt_uncertainty)
    peak_mz_uncertainty = (
        0.0 if mz_uncertainty is None else float(mz_uncertainty)
    )
    measured_precision = getattr(spectrum, "measured_precision", None)

    spectrum_id = getattr(spectrum, "ID", None)
    if peak_id_prefix is None:
        peak_id_prefix = "peak" if spectrum_id is None else str(spectrum_id)
    source = None if spectrum_id is None else str(spectrum_id)
    result = []
    for index, (mz, intensity) in enumerate(peaks[:, :2]):
        result.append(
            Peak3D(
                peak_id="{0}:{1}".format(peak_id_prefix, index),
                retention_time=retention_time,
                mz=float(mz),
                measured_mobility=float(mobility[index]),
                intensity=float(intensity),
                retention_time_uncertainty=rt_uncertainty,
                mz_uncertainty=(
                    float(mz) * float(mz_ppm) * 1e-6
                    if mz_ppm is not None
                    else peak_mz_uncertainty
                ),
                source=source,
                metadata={
                    "array_index": index,
                    "peak_type": peak_type,
                    "measured_precision": measured_precision,
                },
            )
        )
    return result


def extract_reader_peaks3d(
    reader,
    *,
    ms_level: Optional[int] = None,
    max_spectra: Optional[int] = None,
    calibrate_retention_time=None,
    **spectrum_kwargs,
) -> List[Peak3D]:
    """
    Build three-dimensional peaks from every spectrum of a pymzML Reader.

    Spectra without an ion mobility array are skipped; spectra with an
    unusable array (for example a length mismatch) are not silently ignored
    and raise :class:`Search3DError` from :func:`extract_peaks3d`.

    Arguments:
        reader: opened :class:`pymzml.run.Reader` (or any iterable of
            spectra).

    Keyword Arguments:
        ms_level (int, optional): only use spectra of this MS level.
        max_spectra (int, optional): stop after this many spectra carrying
            mobility data.
        calibrate_retention_time (callable, optional): function receiving the
            spectrum and returning an explicit retention time. By default the
            spectrum scan time is converted to minutes.
        **spectrum_kwargs: passed to :func:`extract_peaks3d`.

    Returns:
        peaks (list[Peak3D]): all original peaks in iteration order.
    """
    peaks: List[Peak3D] = []
    collected = 0
    base_kwargs = dict(spectrum_kwargs)
    for spectrum in reader:
        if ms_level is not None and getattr(spectrum, "ms_level", None) != ms_level:
            continue
        if getattr(spectrum, "get_ion_mobility", None) is None:
            continue
        if spectrum.get_ion_mobility(base_kwargs.get("mobility_array_name")) is None:
            continue
        spectrum_kwargs = base_kwargs
        if calibrate_retention_time is not None:
            spectrum_kwargs = dict(base_kwargs)
            spectrum_kwargs["retention_time"] = calibrate_retention_time(spectrum)
        peaks.extend(extract_peaks3d(spectrum, **spectrum_kwargs))
        collected += 1
        if max_spectra is not None and collected >= max_spectra:
            break
    return peaks
