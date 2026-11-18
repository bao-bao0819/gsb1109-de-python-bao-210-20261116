#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blank subtraction for spectra sampled on different m/z grids.

The sample and the blank spectrum usually do not share an m/z grid.  This
module aligns both spectra point by point:

* a sample m/z that matches a blank m/z within :attr:`tolerance` is subtracted
  with that blank value directly;
* otherwise the blank intensity is linearly interpolated between the two
  blank points that bracket the sample m/z;
* where the blank provides no bracketing information (outside its covered
  range, inside a gap wider than ``max_gap``, or with an empty blank) the
  result is *unknown*.  Unknown is reported explicitly and is never converted
  into a blank intensity of zero, because absence of a measurement is not
  evidence of zero background.

The scaled blank contribution is ``scale(mz) * blank_intensity`` and the
reported value is ``sample_intensity - scaled_blank_intensity``.  Negative
values are kept: they are valid results of the subtraction and are only
clipped when ``clip=True`` is requested explicitly.

Every output point records the blank sources on both sides (m/z, intensity and
interpolation weight), so a value can be audited back to the measurements it
came from.  Uncovered m/z regions are returned as :class:`UnknownInterval`
objects.

Processing is independent of how the sample is split into chunks: the blank
grid is built once and each sample point is evaluated against the global
neighbours, so ``subtract(chunk_size=...)`` and :meth:`BlankSubtractor.iter_chunks`
produce exactly the same values as a single full pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from typing import Callable, Iterator, Sequence

import numpy as np

__all__ = [
    "BlankSource",
    "BlankSubtractionPoint",
    "BlankSubtractionResult",
    "BlankSubtractor",
    "UnknownInterval",
    "subtract_blank",
    "subtract_blank_arrays",
]

_MATCHED = "matched"
_INTERPOLATED = "interpolated"
_UNKNOWN = "unknown"

_LEFT_OF_BLANK = "left_of_blank"
_RIGHT_OF_BLANK = "right_of_blank"
_GAP = "gap"
_NO_BLANK = "no_blank"
_INVALID_SAMPLE = "invalid_sample"
_INVALID_SCALE = "invalid_scale"

_SCALE_TYPE = float | Callable[[float], float]


@dataclass(frozen=True)
class BlankSource:
    """
    One blank measurement used for a sample point.

    Attributes:
        index (int): position of the blank point in the sorted blank grid.
        mz (float): m/z of the blank measurement.
        intensity (float): measured blank intensity (before scaling).
        weight (float or None): interpolation weight in
            :attr:`BlankSubtractionPoint.sources`.  Bracket sources that were
            not used carry ``None``.
    """

    index: int
    mz: float
    intensity: float
    weight: float | None


@dataclass(frozen=True)
class BlankSubtractionPoint:
    """
    Result for a single sample point.

    Attributes:
        mz (float): sample m/z.
        sample_intensity (float): measured sample intensity.
        blank_intensity (float or None): scaled blank contribution used for
            the subtraction; ``None`` when the result is unknown.
        value (float or None): ``sample_intensity - blank_intensity``;
            ``None`` when the blank does not cover this m/z.  Negative values
            are preserved.
        status (str): ``"matched"``, ``"interpolated"`` or ``"unknown"``.
        left (BlankSource or None): measured blank point at or left of the
            sample m/z (the left bracket).
        right (BlankSource or None): measured blank point at or right of the
            sample m/z (the right bracket).
        sources (tuple): blank measurements actually used for the value: one
            point for a direct match, two points for interpolation, empty for
            an unknown result.
        unknown_reason (str or None): why the point is unknown.
        sample_index (int): position of the point in the input sequence.
    """

    mz: float
    sample_intensity: float
    blank_intensity: float | None
    value: float | None
    status: str
    left: BlankSource | None
    right: BlankSource | None
    sources: tuple[BlankSource, ...]
    unknown_reason: str | None
    sample_index: int

    @property
    def is_known(self) -> bool:
        """Return ``True`` when a subtraction value could be computed."""
        return self.value is not None

    def to_record(self) -> dict:
        """Return a JSON-friendly dictionary with the point and its sources."""

        def source_record(source):
            if source is None:
                return None
            return {
                "index": source.index,
                "mz": source.mz,
                "intensity": source.intensity,
                "weight": source.weight,
            }

        return {
            "sample_index": self.sample_index,
            "mz": self.mz,
            "sample_intensity": self.sample_intensity,
            "blank_intensity": self.blank_intensity,
            "value": self.value,
            "status": self.status,
            "unknown_reason": self.unknown_reason,
            "left": source_record(self.left),
            "right": source_record(self.right),
            "sources": [source_record(source) for source in self.sources],
        }


@dataclass(frozen=True)
class UnknownInterval:
    """
    A contiguous m/z range for which no blank background can be computed.

    Attributes:
        mz_start (float): start of the uncovered range, clipped to the sample.
        mz_end (float): end of the uncovered range, clipped to the sample.
        reason (str): ``"left_of_blank"``, ``"right_of_blank"``, ``"gap"`` or
            ``"no_blank"``.
        sample_indices (tuple): indices of the sample points inside the range.
    """

    mz_start: float
    mz_end: float
    reason: str
    sample_indices: tuple[int, ...]

    @property
    def count(self) -> int:
        """Number of sample points inside the interval."""
        return len(self.sample_indices)

    def to_record(self) -> dict:
        """Return a JSON-friendly dictionary."""
        return {
            "mz_start": self.mz_start,
            "mz_end": self.mz_end,
            "reason": self.reason,
            "count": self.count,
            "sample_indices": list(self.sample_indices),
        }


def _coerce_peaks(data, peak_type: str, name: str) -> np.ndarray:
    """Convert a spectrum-like object or an Nx2 sequence into a peak array."""
    if data is None:
        return np.empty((0, 2), dtype=float)
    peaks = getattr(data, "peaks", None)
    if peaks is not None:
        if callable(peaks):
            try:
                data = peaks(peak_type)
            except TypeError:
                data = peaks()
        else:
            data = peaks
    if data is None:
        return np.empty((0, 2), dtype=float)
    try:
        array = np.asarray(data, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "{0} must be an Nx2 peak array or a spectrum-like object".format(name)
        ) from error
    if array.size == 0:
        return np.empty((0, 2), dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(
            "{0} must have shape (n, 2), got {1}".format(name, array.shape)
        )
    return array


class _BlankGrid:
    """Sorted, de-duplicated blank measurements plus their covered ranges."""

    def __init__(self, peaks: np.ndarray, max_gap: float | None):
        mz = peaks[:, 0]
        intensity = peaks[:, 1]
        finite = np.isfinite(mz) & np.isfinite(intensity)
        mz = mz[finite]
        intensity = intensity[finite]
        if mz.size:
            order = np.argsort(mz, kind="stable")
            mz = mz[order]
            intensity = intensity[order]
            unique_mz = np.unique(mz)
            if unique_mz.size != mz.size:
                summed = np.zeros(unique_mz.size, dtype=float)
                np.add.at(summed, np.searchsorted(unique_mz, mz), intensity)
                mz, intensity = unique_mz, summed
        self.mz = np.ascontiguousarray(mz, dtype=float)
        self.intensity = np.ascontiguousarray(intensity, dtype=float)
        self.max_gap = max_gap
        self.coverage = self._build_coverage()
        self._starts = tuple(start for start, _ in self.coverage)

    def _build_coverage(self) -> tuple[tuple[float, float], ...]:
        if self.mz.size == 0:
            return ()
        if self.mz.size == 1:
            return ((float(self.mz[0]), float(self.mz[0])),)
        coverage = []
        start = float(self.mz[0])
        for previous, current in zip(self.mz[:-1], self.mz[1:]):
            if self.max_gap is not None and current - previous > self.max_gap:
                coverage.append((start, float(previous)))
                start = float(current)
        coverage.append((start, float(self.mz[-1])))
        return tuple(coverage)

    def uncovered_regions(self) -> list[tuple[float, float, str]]:
        """Return uncovered m/z ranges as ``(start, end, reason)``."""
        if not self.coverage:
            return [(-math.inf, math.inf, _NO_BLANK)]
        regions = [(-math.inf, self.coverage[0][0], _LEFT_OF_BLANK)]
        for left, right in zip(self.coverage[:-1], self.coverage[1:]):
            regions.append((left[1], right[0], _GAP))
        regions.append((self.coverage[-1][1], math.inf, _RIGHT_OF_BLANK))
        return regions

    def classify(self, mz: float) -> str | None:
        """Return the uncovered-region reason for ``mz`` or ``None`` if covered."""
        if not self.coverage:
            return _NO_BLANK
        if mz < self.coverage[0][0]:
            return _LEFT_OF_BLANK
        if mz > self.coverage[-1][1]:
            return _RIGHT_OF_BLANK
        index = int(np.searchsorted(self._starts, mz, side="right")) - 1
        lower, upper = self.coverage[index]
        if lower <= mz <= upper:
            return None
        return _GAP

    def _source(self, index: int, weight: float | None) -> BlankSource:
        return BlankSource(
            index=index,
            mz=float(self.mz[index]),
            intensity=float(self.intensity[index]),
            weight=weight,
        )

    def brackets(self, mz: float) -> tuple[BlankSource | None, BlankSource | None]:
        """
        Return the blank points immediately left of and right of ``mz``.

        Both sides are reported independently of whether they are used for the
        value, so an unknown m/z still shows which measurements surround it.
        """
        if self.mz.size == 0:
            return None, None
        right_index = int(np.searchsorted(self.mz, mz, side="left"))
        left_index = right_index - 1
        if right_index < self.mz.size and self.mz[right_index] == mz:
            source = self._source(right_index, 1.0)
            return source, source
        left = self._source(left_index, None) if left_index >= 0 else None
        right = (
            self._source(right_index, None)
            if right_index < self.mz.size
            else None
        )
        return left, right

    def evaluate(self, mz: float, tolerance: float) -> tuple:
        """
        Return ``(status, reason, left, right, sources, interpolated)``.

        ``left``/``right`` are the bracketing :class:`BlankSource` objects and
        ``sources`` are the measurements actually used for the value.
        ``interpolated`` is the unscaled blank intensity, ``None`` if unknown.
        """
        left, right = self.brackets(mz)
        if self.mz.size == 0:
            return _UNKNOWN, _NO_BLANK, left, right, (), None

        position = int(np.searchsorted(self.mz, mz, side="left"))
        candidates = []
        if position < self.mz.size:
            candidates.append(position)
        if position > 0:
            candidates.append(position - 1)
        if candidates:
            nearest = min(candidates, key=lambda index: abs(self.mz[index] - mz))
            if abs(self.mz[nearest] - mz) <= tolerance:
                source = self._source(nearest, 1.0)
                return _MATCHED, None, left, right, (source,), source.intensity

        reason = self.classify(mz)
        if reason is not None:
            return _UNKNOWN, reason, left, right, (), None

        right_index = int(np.searchsorted(self.mz, mz, side="right"))
        left_index = right_index - 1
        if left_index < 0 or right_index >= self.mz.size:
            reason = self.classify(mz) or _NO_BLANK
            return _UNKNOWN, reason, left, right, (), None
        left_mz = float(self.mz[left_index])
        right_mz = float(self.mz[right_index])
        span = right_mz - left_mz
        if span <= 0:
            return _UNKNOWN, _GAP, left, right, (), None
        right_weight = (mz - left_mz) / span
        left_source = self._source(left_index, 1.0 - right_weight)
        right_source = self._source(right_index, right_weight)
        interpolated = (
            left_source.weight * left_source.intensity
            + right_source.weight * right_source.intensity
        )
        return (
            _INTERPOLATED,
            None,
            left,
            right,
            (left_source, right_source),
            interpolated,
        )


class BlankSubtractionResult:
    """
    Result of a blank subtraction.

    The result keeps one :class:`BlankSubtractionPoint` per input sample point
    (in input order), the covered blank ranges and the uncovered intervals.
    """

    def __init__(
        self,
        points: Sequence[BlankSubtractionPoint],
        coverage: Sequence[tuple[float, float]],
        sample_range: tuple[float, float] | None,
        options: dict,
        uncovered_regions: Sequence[tuple[float, float, str]] = (),
    ):
        self.points = tuple(points)
        self.coverage = tuple(coverage)
        self.sample_range = sample_range
        self.options = dict(options)
        self.uncovered_regions = tuple(uncovered_regions)
        self.unknown_intervals = self._build_unknown_intervals()

    def __iter__(self) -> Iterator[BlankSubtractionPoint]:
        return iter(self.points)

    def __len__(self) -> int:
        return len(self.points)

    @property
    def known_points(self) -> tuple[BlankSubtractionPoint, ...]:
        """Points for which a value could be computed."""
        return tuple(point for point in self.points if point.is_known)

    @property
    def unknown_points(self) -> tuple[BlankSubtractionPoint, ...]:
        """Points without blank coverage."""
        return tuple(point for point in self.points if not point.is_known)

    @property
    def peaks(self) -> np.ndarray:
        """Known ``(mz, value)`` pairs as an Nx2 array; negatives preserved."""
        known = self.known_points
        if not known:
            return np.empty((0, 2), dtype=float)
        return np.array([[point.mz, point.value] for point in known], dtype=float)

    @property
    def all_peaks(self) -> np.ndarray:
        """``(mz, value)`` pairs with ``NaN`` for unknown values."""
        if not self.points:
            return np.empty((0, 2), dtype=float)
        return np.array(
            [
                [point.mz, point.value if point.is_known else math.nan]
                for point in self.points
            ],
            dtype=float,
        )

    @property
    def blank_coverage(self) -> tuple[tuple[float, float], ...]:
        """m/z ranges where the blank provides a background estimate."""
        return self.coverage

    def _build_unknown_intervals(self) -> tuple[UnknownInterval, ...]:
        if not self.points or self.sample_range is None:
            return ()
        sample_min, sample_max = self.sample_range
        buckets: dict[tuple[float, float, str], list[int]] = {}
        for point in self.points:
            if point.is_known or point.unknown_reason in (
                _INVALID_SAMPLE,
                _INVALID_SCALE,
            ):
                continue
            if not math.isfinite(point.mz):
                continue
            for region_start, region_end, reason in self.uncovered_regions:
                if region_start < point.mz < region_end or (
                    region_start == region_end == point.mz
                ):
                    buckets.setdefault(
                        (region_start, region_end, reason), []
                    ).append(point.sample_index)
                    break
        intervals = []
        for (region_start, region_end, reason), indices in buckets.items():
            start = max(region_start, sample_min)
            end = min(region_end, sample_max)
            if not math.isfinite(start):
                start = sample_min
            if not math.isfinite(end):
                end = sample_max
            if end < start:
                continue
            intervals.append(
                UnknownInterval(
                    mz_start=float(start),
                    mz_end=float(end),
                    reason=reason,
                    sample_indices=tuple(indices),
                )
            )
        intervals.sort(key=lambda interval: (interval.mz_start, interval.mz_end))
        return tuple(intervals)

    def to_records(self) -> list[dict]:
        """Return all points as JSON-friendly dictionaries."""
        return [point.to_record() for point in self.points]

    def to_json(self, **kwargs) -> str:
        """Serialize the result (points, coverage and unknown intervals)."""
        payload = {
            "options": self.options,
            "sample_range": self.sample_range,
            "blank_coverage": [list(interval) for interval in self.coverage],
            "unknown_intervals": [
                interval.to_record() for interval in self.unknown_intervals
            ],
            "points": self.to_records(),
        }
        kwargs.setdefault("ensure_ascii", False)
        return json.dumps(payload, **kwargs)

    def as_arrays(self, include_unknown: bool = False) -> tuple[np.ndarray, ...]:
        """
        Return ``(mz, sample_intensity, blank_intensity, value)`` arrays.

        Unknown entries are ``NaN`` in the numeric arrays.  With
        ``include_unknown=False`` only known points are returned.
        """
        points = self.points if include_unknown else self.known_points
        if not points:
            empty = np.empty(0, dtype=float)
            return empty, empty.copy(), empty.copy(), empty.copy()
        mz = np.array([point.mz for point in points], dtype=float)
        sample = np.array([point.sample_intensity for point in points], dtype=float)
        blank = np.array(
            [
                point.blank_intensity if point.blank_intensity is not None else math.nan
                for point in points
            ],
            dtype=float,
        )
        value = np.array(
            [point.value if point.value is not None else math.nan for point in points],
            dtype=float,
        )
        return mz, sample, blank, value

    def to_spectrum(self):
        """Return a :class:`pymzml.spec.Spectrum` of the known values."""
        from .spec import Spectrum

        spectrum = Spectrum()
        spectrum.set_peaks(self.peaks, "raw")
        return spectrum

    @classmethod
    def combine(cls, results: Sequence["BlankSubtractionResult"]):
        """
        Combine consecutive chunk results into one result.

        The values of the chunks are kept as they are; unknown intervals are
        recomputed over the combined points so intervals split at chunk
        boundaries are reported once.
        """
        results = list(results)
        if not results:
            return cls((), (), None, {})
        points = [point for result in results for point in result.points]
        sample_range = results[0].sample_range
        finite_mz = [point.mz for point in points if math.isfinite(point.mz)]
        if finite_mz:
            sample_range = (min(finite_mz), max(finite_mz))
        return cls(
            points,
            results[0].coverage,
            sample_range,
            results[0].options,
            results[0].uncovered_regions,
        )


class BlankSubtractor:
    """
    Align a blank spectrum onto a sample spectrum and subtract its background.

    Arguments:
        sample: sample peaks (Nx2 array, list of pairs, Spectrum or
            Chromatogram).
        blank: blank peaks in the same accepted forms.

    Keyword Arguments:
        scale (float or callable): factor applied to the blank intensity.  A
            callable is evaluated per sample m/z, which allows an m/z
            dependent scaling rule.
        tolerance (float): absolute m/z tolerance for a direct blank match.
            Points outside the tolerance are linearly interpolated.
        max_gap (float, optional): widest blank gap that may be bridged by
            interpolation.  Wider gaps are reported as unknown.  ``None``
            bridges every internal gap between two measured blank points.
        clip (bool): clip negative values at zero.  Defaults to ``False``
            because negative values are valid subtraction results.
        sample_peak_type (str): peak type requested from a sample Spectrum.
        blank_peak_type (str): peak type requested from a blank Spectrum.

    Example:

    >>> subtractor = BlankSubtractor(sample, blank, scale=0.5)
    >>> result = subtractor.subtract()
    >>> result.peaks
    """

    def __init__(
        self,
        sample,
        blank,
        *,
        scale: _SCALE_TYPE = 1.0,
        tolerance: float = 0.0,
        max_gap: float | None = None,
        clip: bool = False,
        sample_peak_type: str = "raw",
        blank_peak_type: str = "raw",
    ):
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        if max_gap is not None and max_gap < 0:
            raise ValueError("max_gap must be non-negative or None")
        if not callable(scale) and not math.isfinite(float(scale)):
            raise ValueError("scale must be finite")
        self.sample_peaks = _coerce_peaks(sample, sample_peak_type, "sample")
        self.blank_peaks = _coerce_peaks(blank, blank_peak_type, "blank")
        self.scale = scale
        self.tolerance = float(tolerance)
        self.max_gap = None if max_gap is None else float(max_gap)
        self.clip = bool(clip)
        self.sample_peak_type = sample_peak_type
        self.blank_peak_type = blank_peak_type
        self.blank_grid = _BlankGrid(self.blank_peaks, self.max_gap)
        finite = self.sample_peaks[np.isfinite(self.sample_peaks[:, 0])][:, 0]
        self._sample_range = (
            (float(finite.min()), float(finite.max())) if finite.size else None
        )

    def _scale_at(self, mz: float) -> float:
        factor = self.scale(mz) if callable(self.scale) else self.scale
        return float(factor)

    def _evaluate_point(self, index: int, mz: float, intensity: float):
        if not (math.isfinite(mz) and math.isfinite(intensity)):
            return BlankSubtractionPoint(
                mz=mz,
                sample_intensity=intensity,
                blank_intensity=None,
                value=None,
                status=_UNKNOWN,
                left=None,
                right=None,
                sources=(),
                unknown_reason=_INVALID_SAMPLE,
                sample_index=index,
            )
        status, reason, left, right, sources, interpolated = self.blank_grid.evaluate(
            mz, self.tolerance
        )
        if status == _UNKNOWN:
            return BlankSubtractionPoint(
                mz=mz,
                sample_intensity=intensity,
                blank_intensity=None,
                value=None,
                status=_UNKNOWN,
                left=left,
                right=right,
                sources=(),
                unknown_reason=reason,
                sample_index=index,
            )
        factor = self._scale_at(mz)
        if not math.isfinite(factor):
            return BlankSubtractionPoint(
                mz=mz,
                sample_intensity=intensity,
                blank_intensity=None,
                value=None,
                status=_UNKNOWN,
                left=left,
                right=right,
                sources=sources,
                unknown_reason=_INVALID_SCALE,
                sample_index=index,
            )
        blank_intensity = factor * interpolated
        value = intensity - blank_intensity
        if self.clip and value < 0:
            value = 0.0
        return BlankSubtractionPoint(
            mz=mz,
            sample_intensity=intensity,
            blank_intensity=blank_intensity,
            value=value,
            status=status,
            left=left,
            right=right,
            sources=sources,
            unknown_reason=None,
            sample_index=index,
        )

    def blank_value(self, mz: float) -> dict:
        """
        Return the aligned and scaled blank estimate for a single m/z.

        This exposes the alignment and scaling steps on their own, without a
        sample intensity.  The returned dictionary contains ``status``,
        ``blank_intensity`` (``None`` when unknown), ``unknown_reason``, the
        bracketing ``left``/``right`` sources and the ``sources`` used.
        """
        status, reason, left, right, sources, interpolated = self.blank_grid.evaluate(
            mz, self.tolerance
        )
        if status == _UNKNOWN:
            return {
                "status": _UNKNOWN,
                "blank_intensity": None,
                "unknown_reason": reason,
                "left": left,
                "right": right,
                "sources": (),
            }
        factor = self._scale_at(mz)
        if not math.isfinite(factor):
            return {
                "status": _UNKNOWN,
                "blank_intensity": None,
                "unknown_reason": _INVALID_SCALE,
                "left": left,
                "right": right,
                "sources": sources,
            }
        return {
            "status": status,
            "blank_intensity": factor * interpolated,
            "unknown_reason": None,
            "left": left,
            "right": right,
            "sources": sources,
        }

    def _evaluate(self, start: int, stop: int) -> list[BlankSubtractionPoint]:
        points = []
        for index in range(start, stop):
            mz, intensity = self.sample_peaks[index]
            points.append(self._evaluate_point(index, float(mz), float(intensity)))
        return points

    def _result(self, points: Sequence[BlankSubtractionPoint]):
        options = {
            "scale": None if callable(self.scale) else float(self.scale),
            "scale_is_callable": callable(self.scale),
            "tolerance": self.tolerance,
            "max_gap": self.max_gap,
            "clip": self.clip,
        }
        return BlankSubtractionResult(
            points,
            self.blank_grid.coverage,
            self._sample_range,
            options,
            self.blank_grid.uncovered_regions(),
        )

    def subtract(self, chunk_size: int | None = None) -> BlankSubtractionResult:
        """
        Subtract the blank and return a :class:`BlankSubtractionResult`.

        Args:
            chunk_size (int, optional): evaluate the sample in chunks of this
                many points.  The result is identical to a full pass because
                every point is aligned against the global blank grid.
        """
        total = len(self.sample_peaks)
        if chunk_size is None:
            return self._result(self._evaluate(0, total))
        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        points = []
        for start in range(0, total, chunk_size):
            points.extend(self._evaluate(start, min(start + chunk_size, total)))
        return self._result(points)

    def iter_chunks(self, chunk_size: int) -> Iterator[BlankSubtractionResult]:
        """
        Yield one result per chunk of sample points.

        Chunk results can be merged with :meth:`BlankSubtractionResult.combine`.
        """
        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        for start in range(0, len(self.sample_peaks), chunk_size):
            stop = min(start + chunk_size, len(self.sample_peaks))
            yield self._result(self._evaluate(start, stop))


def subtract_blank(
    sample,
    blank,
    *,
    scale: _SCALE_TYPE = 1.0,
    tolerance: float = 0.0,
    max_gap: float | None = None,
    clip: bool = False,
    chunk_size: int | None = None,
    sample_peak_type: str = "raw",
    blank_peak_type: str = "raw",
) -> BlankSubtractionResult:
    """
    Subtract ``blank`` from ``sample`` after aligning both m/z grids.

    Arguments:
        sample: sample peaks or a Spectrum/Chromatogram.
        blank: blank peaks or a Spectrum/Chromatogram.

    Keyword Arguments:
        scale: blank scaling factor or ``callable(mz) -> factor``.
        tolerance: absolute m/z tolerance for a direct match.
        max_gap: widest blank gap that may be interpolated across.
        clip: clip negative values at zero (default: keep negatives).
        chunk_size: optional sample chunk size; results are chunk-invariant.
        sample_peak_type: peak type read from a sample Spectrum.
        blank_peak_type: peak type read from a blank Spectrum.

    Returns:
        BlankSubtractionResult: values, both-sided sources and unknown ranges.
    """
    return BlankSubtractor(
        sample,
        blank,
        scale=scale,
        tolerance=tolerance,
        max_gap=max_gap,
        clip=clip,
        sample_peak_type=sample_peak_type,
        blank_peak_type=blank_peak_type,
    ).subtract(chunk_size=chunk_size)


def subtract_blank_arrays(
    sample_mz: Sequence[float],
    sample_intensity: Sequence[float],
    blank_mz: Sequence[float],
    blank_intensity: Sequence[float],
    **kwargs,
) -> BlankSubtractionResult:
    """Convenience wrapper for separate m/z and intensity arrays."""
    sample = np.column_stack(
        (np.asarray(sample_mz, dtype=float), np.asarray(sample_intensity, dtype=float))
    )
    blank = np.column_stack(
        (np.asarray(blank_mz, dtype=float), np.asarray(blank_intensity, dtype=float))
    )
    return subtract_blank(sample, blank, **kwargs)


def _read_peak_file(path: str) -> np.ndarray:
    """Read two-column peak data (comma, tab or whitespace separated)."""
    rows = []
    with open(path, newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = [
                field
                for field in csv.reader([stripped], delimiter=",").__next__()
                for field in field.split()
            ]
            if len(fields) < 2:
                raise ValueError(
                    "{0}:{1}: expected two columns".format(path, line_number)
                )
            rows.append((float(fields[0]), float(fields[1])))
    if not rows:
        return np.empty((0, 2), dtype=float)
    return np.array(rows, dtype=float)


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point for ``python -m pymzml.blank``."""
    parser = argparse.ArgumentParser(
        description=(
            "Subtract a blank spectrum from a sample spectrum on differing "
            "m/z grids, keeping both-sided sources and unknown intervals."
        )
    )
    parser.add_argument("--sample", required=True, help="sample two-column file")
    parser.add_argument("--blank", required=True, help="blank two-column file")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--max-gap", type=float, default=None)
    parser.add_argument("--clip", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--format", choices=("tsv", "json"), default="tsv")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    result = subtract_blank(
        _read_peak_file(args.sample),
        _read_peak_file(args.blank),
        scale=args.scale,
        tolerance=args.tolerance,
        max_gap=args.max_gap,
        clip=args.clip,
        chunk_size=args.chunk_size,
    )
    if args.format == "json":
        text = result.to_json(indent=2)
    else:
        lines = [
            "mz\tsample_i\tblank_i\tvalue\tstatus\tleft_mz\tleft_i\tright_mz\tright_i"
        ]
        for point in result:
            def number(value):
                return "NA" if value is None else repr(float(value))

            lines.append(
                "\t".join(
                    (
                        number(point.mz),
                        number(point.sample_intensity),
                        number(point.blank_intensity),
                        number(point.value),
                        point.status,
                        number(point.left.mz if point.left else None),
                        number(point.left.intensity if point.left else None),
                        number(point.right.mz if point.right else None),
                        number(point.right.intensity if point.right else None),
                    )
                )
            )
        text = "\n".join(lines) + "\n"

    if args.output:
        with open(args.output, "w") as handle:
            handle.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
