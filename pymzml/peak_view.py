#!/usr/bin/env python3
# -*- coding: latin-1 -*-
"""
Versioned peak views for calibrated mass spectra.

A :class:`PeakView` is an immutable snapshot of one spectrum's peaks under one
calibration.  Every view keeps

* its own calibrated peak positions and intensities,
* the index of the original peak each calibrated peak came from,
* its own measured precision and the derived query index, and
* the calibration that was used to create it.

Views never write back into the spectrum or into another view.  This makes it
safe to keep the original calibration and one or more revised calibrations of
the same peak data side by side, and to interleave queries against them.

The module also provides :class:`Calibration`, a small piecewise-linear (or
callable) m/z mapping with explicit reporting of monotonicity and reverse
lookup multiplicity.
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator

import numpy as np

PROTON = 1.00727646677


class CalibrationError(ValueError):
    """Base error for invalid calibration data."""


class VersionConflictError(CalibrationError):
    """Raised when a version would silently overwrite an existing view."""


class NonInvertibleError(CalibrationError):
    """Raised when a unique inverse was requested but does not exist."""


@dataclass(frozen=True)
class CalibrationSegment:
    """One linear segment of a calibration mapping."""

    index: int
    x0: float
    x1: float
    y0: float
    y1: float

    @property
    def slope(self) -> float:
        dx = self.x1 - self.x0
        if dx == 0:
            return float("inf") if self.y1 != self.y0 else 0.0
        return (self.y1 - self.y0) / dx

    @property
    def monotonic(self) -> bool:
        return self.slope > 0 or self.slope < 0

    @property
    def flat(self) -> bool:
        return self.y0 == self.y1

    def contains(self, value: float) -> bool:
        low, high = sorted((self.y0, self.y1))
        return low <= value <= high

    def solve(self, value: float) -> list[float]:
        """Return x positions in this segment which map to *value*."""
        if self.flat:
            return [self.x0, self.x1] if value == self.y0 else []
        t = (value - self.y0) / (self.y1 - self.y0)
        if -1e-12 <= t <= 1 + 1e-12:
            return [self.x0 + t * (self.x1 - self.x0)]
        return []

    def as_dict(self) -> dict[str, float]:
        return {
            "index": self.index,
            "x0": self.x0,
            "x1": self.x1,
            "y0": self.y0,
            "y1": self.y1,
            "slope": self.slope,
        }


@dataclass
class ReverseLookup:
    """
    Result of a reverse lookup.

    ``preimages`` contains the original positions (or, for a peak view, the
    matching peak records).  ``unique`` is only true if exactly one preimage
    exists.  The object is list-like so callers can treat it as the result
    list while still inspecting the explanation.
    """

    value: float
    preimages: list[Any] = field(default_factory=list)
    unique: bool = False
    monotonic: bool | None = None
    reason: str = ""
    non_monotonic_segments: tuple[CalibrationSegment, ...] = ()

    @property
    def is_unique(self) -> bool:
        return self.unique

    @property
    def explanation(self) -> str:
        return self.explain()

    @property
    def non_monotonic(self) -> bool:
        return bool(self.non_monotonic_segments)

    @property
    def monotonic_calibration(self) -> bool | None:
        return self.monotonic

    @property
    def calibration_ambiguous(self) -> bool:
        return bool(self.non_monotonic_segments) and not self.unique

    @property
    def count(self) -> int:
        return len(self.preimages)

    @property
    def solutions(self) -> list[Any]:
        return self.preimages

    @property
    def results(self) -> list[Any]:
        return self.preimages

    @property
    def originals(self) -> list[Any]:
        return self.preimages

    @property
    def matches(self) -> list[Any]:
        return self.preimages

    @property
    def original(self) -> Any:
        return self.preimages[0] if self.unique else None

    def explain(self) -> str:
        if self.reason:
            return self.reason
        if self.unique:
            return "unique reverse lookup"
        if not self.preimages:
            return "no reverse lookup result"
        return "ambiguous reverse lookup: {0} candidates".format(len(self.preimages))

    def __iter__(self) -> Iterator[Any]:
        return iter(self.preimages)

    def __len__(self) -> int:
        return len(self.preimages)

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in {
                "value",
                "preimages",
                "solutions",
                "matches",
                "unique",
                "is_unique",
                "monotonic",
                "reason",
                "explanation",
                "non_monotonic_segments",
                "non_monotonic_ranges",
                "count",
            }:
                if item == "non_monotonic_ranges":
                    return tuple(
                        (seg.x0, seg.x1) for seg in self.non_monotonic_segments
                    )
                return getattr(self, item)
            raise KeyError(item)
        return self.preimages[item]

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        return (
            "value",
            "preimages",
            "unique",
            "monotonic",
            "reason",
            "non_monotonic_segments",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "preimages": list(self.preimages),
            "unique": self.unique,
            "monotonic": self.monotonic,
            "reason": self.explain(),
            "non_monotonic_segments": tuple(
                seg.as_dict() for seg in self.non_monotonic_segments
            ),
        }

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ReverseLookup):
            return self.as_dict() == other.as_dict()
        if isinstance(other, Mapping):
            return self.as_dict() == dict(other)
        if isinstance(other, (list, tuple)):
            return list(self.preimages) == list(other)
        return NotImplemented

    def __bool__(self) -> bool:
        return bool(self.preimages)


class Calibration:
    """
    A versioned m/z calibration mapping.

    Arguments:
        mapping: one of

            * a callable ``mz -> calibrated_mz``,
            * a mapping ``{original_mz: calibrated_mz}`` (piecewise linear),
            * a sequence of ``(original_mz, calibrated_mz)`` control points,
            * a mapping with ``scale``/``offset`` keys (affine), or
            * ``None`` for the identity mapping.

        name/version: label of the calibration.
        inverse: optional analytic inverse callable.  If omitted, the inverse
            is solved from the control points (piecewise linear) or, for a
            linear callable, from its slope and intercept.
        domain: optional ``(min, max)`` domain for callable mappings.  It is
            used for monotonicity checks and numerical reverse lookup.
        measured_precision/precision: optional default precision in ppm
            (5e-6 == 5 ppm).  A view can override this value.

    Notes:
        A calibration does not have to be monotonic.  Non-monotonic segments
        are reported, and every reverse lookup carries a ``unique`` flag and
        an explanation instead of silently picking one result.
    """

    def __init__(
        self,
        mapping: Any = None,
        name: str | None = None,
        *,
        version: str | None = None,
        inverse: Callable[[float], float] | None = None,
        domain: tuple[float, float] | None = None,
        measured_precision: float | None = None,
        precision: float | None = None,
        offset: float | None = None,
        scale: float | None = None,
        description: str | None = None,
        x: Iterable[float] | None = None,
        y: Iterable[float] | None = None,
        **kwargs: Any,
    ) -> None:
        if mapping is None:
            for key in ("control_points", "points", "mz_map", "map"):
                if kwargs.get(key) is not None:
                    mapping = kwargs[key]
                    break
        if x is not None or y is not None:
            if x is None or y is None:
                raise CalibrationError("Both x and y control values are required")
            x_values, y_values = np.asarray(x).ravel(), np.asarray(y).ravel()
            if len(x_values) != len(y_values):
                raise CalibrationError(
                    "Calibration x and y arrays must have the same length"
                )
            mapping = list(zip(x_values, y_values))
        # Be forgiving about Calibration(name="v2", mapping=...) and about
        # Calibration(x_values, y_values) positional use.
        if (
            name is not None
            and isinstance(mapping, Sequence)
            and not isinstance(mapping, (str, bytes))
            and isinstance(name, Sequence)
            and not isinstance(name, (str, bytes))
        ):
            x_values, y_values = mapping, name
            if len(x_values) != len(y_values):
                raise CalibrationError(
                    "Calibration x and y arrays must have the same length"
                )
            mapping = list(zip(x_values, y_values))
            name = None
        if isinstance(mapping, str) and name is None and version is None:
            name = mapping
            mapping = None
        self.name = version if version is not None else name
        self.version = self.name
        self.calibration_version = self.name
        self.label = self.name
        self.description = description or kwargs.get("label")
        self.domain = domain
        self.kind = "identity"
        self._inverse_func = inverse
        self._func: Callable[[float], float] | None = None
        self._control_points: tuple[tuple[float, float], ...] | None = None
        self._segments: tuple[CalibrationSegment, ...] = ()
        self._control_xs: tuple[float, ...] = ()
        self._affine: tuple[float, float] | None = None
        self.measured_precision = (
            measured_precision if measured_precision is not None else precision
        )

        if mapping is None:
            mapping = {}
        if callable(mapping):
            self.kind = "callable"
            self._func = mapping
            if self.domain is not None:
                # Sampling turns the callable into a piecewise-linear mapping,
                # which gives exact checks for monotonicity and preimages.
                sampled = self._sample_callable(
                    mapping, self.domain, kwargs.get("samples", 2001)
                )
                self.kind = "sampled-callable"
                self._set_control_points(sampled)
            else:
                self._detect_affine_callable(mapping)
        elif isinstance(mapping, Mapping):
            if "scale" in mapping or "offset" in mapping or offset is not None:
                self.kind = "affine"
                sc = float(mapping.get("scale", scale if scale is not None else 1.0))
                off = float(mapping.get("offset", offset if offset is not None else 0.0))
                self._affine = (sc, off)
                self._func = lambda x, sc=sc, off=off: sc * x + off
            else:
                self.kind = "piecewise"
                self._set_control_points(mapping.items())
        elif isinstance(mapping, Sequence) and not isinstance(mapping, (str, bytes)):
            self.kind = "piecewise"
            self._set_control_points(mapping)
        elif isinstance(mapping, np.ndarray):
            self.kind = "piecewise"
            self._set_control_points(_points_from_array(mapping))
        else:
            raise CalibrationError(
                "Unsupported calibration mapping type: {0}".format(type(mapping))
            )

        if scale is not None or offset is not None:
            self.kind = "affine"
            sc = float(scale if scale is not None else 1.0)
            off = float(offset if offset is not None else 0.0)
            self._affine = (sc, off)
            self._func = lambda x, sc=sc, off=off: sc * x + off
            self._control_points = None
            self._control_xs = ()
            self._segments = ()

    @staticmethod
    def _sample_callable(
        func: Callable[[float], float], domain: tuple[float, float], samples: int
    ) -> tuple[tuple[float, float], ...]:
        lo, hi = sorted((float(domain[0]), float(domain[1])))
        if samples < 2:
            samples = 2
        xs = np.linspace(lo, hi, int(samples))
        return tuple((float(x), float(func(float(x)))) for x in xs)

    def _detect_affine_callable(self, func: Callable[[float], float]) -> None:
        try:
            y0 = float(func(0.0))
            y1 = float(func(1.0))
            y2 = float(func(2.0))
        except Exception:
            return
        slope = y1 - y0
        if abs((y2 - y1) - slope) <= 1e-9 * max(1.0, abs(slope)):
            self._affine = (slope, y0)

    def _set_control_points(self, points: Iterable[Any]) -> None:
        parsed = []
        for point in points:
            if isinstance(point, Mapping) and {"x0", "x1", "y0", "y1"} <= set(point):
                parsed.extend(
                    [
                        (float(point["x0"]), float(point["y0"])),
                        (float(point["x1"]), float(point["y1"])),
                    ]
                )
                continue
            try:
                values = list(point)
            except TypeError:
                raise CalibrationError(
                    "Calibration control points must be (original, calibrated) pairs"
                )
            if len(values) == 4:
                # Accept both (x0, x1, y0, y1) and (x0, y0, x1, y1) segment
                # tuples; the second form is unambiguous when x0 < y0 < x1.
                a, b, c, d = (float(value) for value in values)
                if b <= a and c <= a:
                    raise CalibrationError(
                        "Calibration segment tuple needs two increasing endpoints"
                    )
                if b <= a:
                    x0, y0, x1, y1 = a, b, c, d
                else:
                    x0, x1, y0, y1 = a, b, c, d
                if x1 <= x0:
                    raise CalibrationError(
                        "Calibration segments require x1 > x0"
                    )
                parsed.extend([(x0, y0), (x1, y1)])
                continue
            if len(values) != 2:
                raise CalibrationError(
                    "Calibration control points must be (original, calibrated) pairs"
                )
            parsed.append((float(values[0]), float(values[1])))
        if not parsed:
            self._control_points = ()
            self._segments = ()
            return
        parsed.sort(key=lambda pair: pair[0])
        # Adjacent segments normally share an endpoint; drop the duplicate
        # rather than rejecting the calibration.
        deduped = []
        for point in parsed:
            if deduped and point[0] == deduped[-1][0]:
                if point[1] != deduped[-1][1]:
                    raise CalibrationError(
                        "Calibration has conflicting values at original m/z "
                        "{0!r}".format(point[0])
                    )
                continue
            deduped.append(point)
        parsed = deduped
        for (x0, _), (x1, _) in zip(parsed, parsed[1:]):
            if x1 <= x0:
                raise CalibrationError(
                    "Calibration control points require strictly increasing original m/z"
                )
        self._control_points = tuple(parsed)
        self._control_xs = tuple(pair[0] for pair in parsed)
        self._segments = tuple(
            CalibrationSegment(i, x0, x1, y0, y1)
            for i, ((x0, y0), (x1, y1)) in enumerate(zip(parsed, parsed[1:]))
        )

    # ------------------------------------------------------------------
    # Basic mapping
    # ------------------------------------------------------------------
    def apply(self, mz: float) -> float:
        """Map one original m/z value to its calibrated position."""
        if self._segments or len(self.control_points) == 1:
            return self._apply_piecewise(float(mz))
        if self._affine is not None:
            sc, off = self._affine
            return sc * float(mz) + off
        if self._func is not None:
            return float(self._func(float(mz)))
        return float(mz)

    def __call__(self, value: float | None = None):
        if value is None:
            return self
        return self.apply(value)

    map = apply
    forward = apply

    def apply_array(self, values: Iterable[float]) -> np.ndarray:
        """Vectorised :meth:`apply`."""
        arr = np.asarray(values, dtype=float)
        if self._segments or len(self.control_points) == 1:
            flat = arr.ravel()
            result = np.asarray([self._apply_piecewise(x) for x in flat], dtype=float)
            return result.reshape(arr.shape)
        if self._affine is not None:
            sc, off = self._affine
            return arr * sc + off
        if self._func is not None:
            flat = arr.ravel()
            result = np.asarray([float(self._func(x)) for x in flat], dtype=float)
            return result.reshape(arr.shape)
        return arr.copy()

    def transform(self, value: Iterable[float] | float):
        if np.isscalar(value):
            return self.apply(float(value))
        return self.apply_array(value)

    map_array = apply_array
    transform_array = apply_array

    def _apply_piecewise(self, mz: float) -> float:
        cps = self._control_points
        if not cps:
            return mz
        if len(cps) == 1:
            return cps[0][1]
        if mz <= cps[0][0]:
            seg = self._segments[0]
        elif mz >= cps[-1][0]:
            seg = self._segments[-1]
        else:
            # locate segment by original x (control points are sorted)
            idx = bisect.bisect_right(self._control_xs, mz) - 1
            idx = min(max(idx, 0), len(self._segments) - 1)
            seg = self._segments[idx]
        if seg.x1 == seg.x0:
            return seg.y0
        return seg.y0 + (mz - seg.x0) * seg.slope

    # ------------------------------------------------------------------
    # Monotonicity and reverse lookup
    # ------------------------------------------------------------------
    @property
    def control_points(self) -> tuple[tuple[float, float], ...]:
        return self._control_points or ()

    @property
    def scale(self) -> float | None:
        return self._affine[0] if self._affine is not None else None

    @property
    def offset(self) -> float | None:
        return self._affine[1] if self._affine is not None else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "kind": self.kind,
            "scale": self.scale,
            "offset": self.offset,
            "control_points": self.control_points,
            "measured_precision": self.measured_precision,
            "monotonic": self.is_monotonic,
            "injective": self.is_injective,
        }

    @property
    def segments(self) -> tuple[CalibrationSegment, ...]:
        return self._segments

    @classmethod
    def from_points(cls, points: Iterable[Any], **kwargs: Any) -> "Calibration":
        return cls(points, **kwargs)

    @classmethod
    def from_mapping(cls, mapping: Any, **kwargs: Any) -> "Calibration":
        return cls(mapping, **kwargs)

    @classmethod
    def from_arrays(
        cls, x: Iterable[float], y: Iterable[float], **kwargs: Any
    ) -> "Calibration":
        return cls(x=x, y=y, **kwargs)

    @classmethod
    def from_callable(cls, func: Callable[[float], float], **kwargs: Any) -> "Calibration":
        return cls(func, **kwargs)

    @classmethod
    def affine(cls, scale: float = 1.0, offset: float = 0.0, **kwargs: Any) -> "Calibration":
        return cls(scale=scale, offset=offset, **kwargs)

    @classmethod
    def linear(cls, scale: float = 1.0, offset: float = 0.0, **kwargs: Any) -> "Calibration":
        return cls(scale=scale, offset=offset, **kwargs)

    @classmethod
    def shift(cls, offset: float, **kwargs: Any) -> "Calibration":
        return cls(offset=offset, **kwargs)

    @classmethod
    def piecewise(cls, points: Iterable[Any], **kwargs: Any) -> "Calibration":
        return cls(points, **kwargs)

    @classmethod
    def identity(cls, **kwargs: Any) -> "Calibration":
        return cls(None, **kwargs)

    @property
    def is_identity(self) -> bool:
        return (
            not self._segments
            and not self.control_points
            and self._affine is None
            and self._func is None
        )

    @property
    def monotonic_direction(self) -> int | None:
        """1 for increasing, -1 for decreasing, 0 for flat, None if mixed."""
        if not self._segments and len(self.control_points) == 1:
            return 0
        if self._segments:
            slopes = [seg.slope for seg in self._segments]
        elif self._affine is not None:
            slopes = [self._affine[0]]
        elif self._func is not None and self.domain is not None:
            slopes = [seg.slope for seg in self._segments]
        else:
            return None
        if not slopes:
            return 1
        positive = any(s > 0 for s in slopes)
        negative = any(s < 0 for s in slopes)
        if positive and negative:
            return None
        if positive:
            return 1
        if negative:
            return -1
        return 0

    @property
    def is_monotonic(self) -> bool | None:
        """True for monotonic (including flat), False for mixed, None unknown."""
        if self._segments or self._affine is not None:
            return self.monotonic_direction is not None
        if self._func is not None and self.domain is not None:
            return self.monotonic_direction is not None
        # An unconstrained callable cannot be proven monotonic without samples.
        return None

    @property
    def monotonic(self) -> bool | None:
        return self.is_monotonic

    @property
    def is_strictly_monotonic(self) -> bool | None:
        if self._segments:
            return all(seg.slope != 0 for seg in self._segments) and (
                self.is_monotonic is not False
            )
        return self.is_injective

    @property
    def is_injective(self) -> bool | None:
        """
        Whether the calibration maps distinct original m/z values to distinct
        calibrated values.

        For piecewise-linear mappings this is exact.  Flat segments and
        direction changes make the mapping non-injective.  ``None`` means it
        could not be determined (unconstrained callable).
        """
        if not self._segments and len(self.control_points) == 1:
            return False
        if self._segments:
            slopes = [seg.slope for seg in self._segments]
            if not slopes:
                return True
            positive = any(s > 0 for s in slopes)
            negative = any(s < 0 for s in slopes)
            if positive and negative:
                return False
            return all(s > 0 for s in slopes) or all(s < 0 for s in slopes)
        if self._affine is not None:
            return self._affine[0] != 0
        if self._func is not None and self.domain is not None:
            return self.is_injective_approx()
        return None

    @property
    def invertible(self) -> bool | None:
        return self.is_injective

    def non_monotonic_segments(self) -> tuple[CalibrationSegment, ...]:
        """
        Return segments which break monotonicity.

        A segment breaks monotonicity when its slope has the opposite sign of
        the first non-flat segment.  If no consistent direction exists, all
        segments with a sign different from the preceding segment are
        returned.  Flat segments are not counted as direction changes, but
        they do prevent unique inversion.
        """
        if not self._segments:
            return ()
        slopes = [seg.slope for seg in self._segments]
        signs = [1 if s > 0 else -1 if s < 0 else 0 for s in slopes]
        nonzero = [s for s in signs if s != 0]
        if not nonzero:
            return ()
        direction = nonzero[0]
        if all(sign == direction for sign in nonzero):
            return ()
        bad = [
            seg for seg, sign in zip(self._segments, signs) if sign not in (0, direction)
        ]
        return tuple(bad)

    @property
    def non_monotonic(self) -> bool:
        return bool(self.non_monotonic_segments())

    is_non_monotonic = non_monotonic

    @property
    def has_non_monotonic_segments(self) -> bool:
        return bool(self.non_monotonic_segments())

    @property
    def non_monotonic_ranges(self) -> tuple[tuple[float, float], ...]:
        return tuple((seg.x0, seg.x1) for seg in self.non_monotonic_segments())

    def _callable_inverse(self, value: float) -> ReverseLookup:
        if self._inverse_func is not None:
            try:
                return ReverseLookup(
                    value,
                    [float(self._inverse_func(value))],
                    True,
                    self.is_monotonic,
                    "",
                    self.non_monotonic_segments(),
                )
            except Exception as exc:
                return ReverseLookup(
                    value,
                    [],
                    False,
                    self.is_monotonic,
                    "inverse callable failed: {0}".format(exc),
                    self.non_monotonic_segments(),
                )
        if self._affine is not None:
            sc, off = self._affine
            if sc == 0:
                if value == off:
                    return ReverseLookup(
                        value,
                        [],
                        False,
                        True,
                        "reverse lookup is not unique: this affine calibration "
                        "maps every original position to the same value",
                        (),
                    )
                return ReverseLookup(
                    value,
                    [],
                    False,
                    True,
                    "no original position maps to this calibrated value",
                    (),
                )
            return ReverseLookup(
                value,
                [(value - off) / sc],
                True,
                self.is_monotonic,
                "",
                self.non_monotonic_segments(),
            )
        if self._func is not None and self.domain is not None:
            xs = np.linspace(*sorted(self.domain), 2001)
            ys = np.asarray([self.apply(float(x)) for x in xs])
            candidates = []
            for i in range(len(xs) - 1):
                y0, y1 = ys[i], ys[i + 1]
                low, high = sorted((y0, y1))
                if low - 1e-12 <= value <= high + 1e-12:
                    if y1 == y0:
                        candidates.extend([float(xs[i]), float(xs[i + 1])])
                    else:
                        t = (value - y0) / (y1 - y0)
                        if -1e-9 <= t <= 1 + 1e-9:
                            candidates.append(float(xs[i] + t * (xs[i + 1] - xs[i])))
            unique = _dedupe(candidates)
            return ReverseLookup(
                value,
                unique,
                len(unique) == 1,
                self.is_monotonic,
                "" if len(unique) == 1 else "reverse lookup is not unique",
                self.non_monotonic_segments(),
            )
        return ReverseLookup(
            value,
            [],
            False,
            self.is_monotonic,
            "no control points, inverse callable, or domain available",
            self.non_monotonic_segments(),
        )

    def reverse_lookup(self, value: float) -> ReverseLookup:
        """
        Return all original positions mapping to *value* and whether the
        result is unique.
        """
        value = float(value)
        if self.is_identity:
            return ReverseLookup(value, [value], True, True, "", ())
        if not self._segments and len(self.control_points) == 1:
            (x0, y0) = self.control_points[0]
            if value == y0:
                return ReverseLookup(
                    value,
                    [x0],
                    False,
                    True,
                    "reverse lookup is not unique: a constant calibration "
                    "maps every original position to this value",
                    (),
                )
            return ReverseLookup(
                value,
                [],
                False,
                True,
                "no original position maps to this calibrated value",
                (),
            )
        if self._segments:
            candidates: list[float] = []
            flat_hit = False
            for seg in self._segments:
                if seg.flat and seg.contains(value):
                    flat_hit = True
                candidates.extend(seg.solve(value))
            if not candidates:
                # The forward mapping extrapolates with the first/last
                # segment, so reverse lookup does the same outside the
                # calibrated range (but never both directions at once).
                first_seg = self._segments[0]
                last_seg = self._segments[-1]
                first_y = first_seg.y0
                last_y = last_seg.y1
                seg = None
                if value < min(first_y, self._segments[0].y1):
                    seg = first_seg
                elif value > max(last_y, self._segments[-1].y0):
                    seg = last_seg
                if seg is not None and not seg.flat and seg.slope != 0:
                    x = seg.x0 + (value - seg.y0) / seg.slope
                    if np.isfinite(x):
                        candidates.append(float(x))
            candidates = _dedupe(candidates)
            if flat_hit:
                reason = (
                    "reverse lookup is not unique: a constant calibration "
                    "segment maps an interval onto this value"
                )
                unique = False
            elif len(candidates) == 1:
                reason = ""
                unique = True
            elif len(candidates) == 0:
                reason = "no original position maps to this calibrated value"
                unique = False
            else:
                reason = (
                    "reverse lookup is ambiguous: {0} original positions map "
                    "to this calibrated value".format(len(candidates))
                )
                unique = False
            bad = self.non_monotonic_segments()
            if bad:
                note = (
                    "non-monotonic calibration segment(s) at original m/z " +
                    ", ".join(
                        "[{0:.6g}, {1:.6g}]".format(seg.x0, seg.x1) for seg in bad
                    )
                )
                if not unique:
                    reason = (reason + "; " + note).strip("; ")
                else:
                    reason = (
                        "unique preimage in the requested window; " + note +
                        " does not affect this value but can produce other "
                        "ambiguous reverse lookups"
                    )
            return ReverseLookup(
                value,
                candidates,
                unique,
                self.is_monotonic,
                reason,
                self.non_monotonic_segments(),
            )
        return self._callable_inverse(value)

    def reverse(self, value: float) -> list[float]:
        """List of original positions mapping to *value*."""
        return self.reverse_lookup(value).preimages

    def inverse(self, value: float) -> float:
        """Return the unique original position, or raise."""
        result = self.reverse_lookup(value)
        if not result.unique:
            raise NonInvertibleError(result.explain())
        return result.preimages[0]

    def unique_inverse(self, value: float) -> float | None:
        result = self.reverse_lookup(value)
        return result.preimages[0] if result.unique else None

    def invert(self, value: float) -> float | None:
        """Non-raising inverse; returns ``None`` if not unique."""
        return self.unique_inverse(value)

    try_inverse = invert

    def is_invertible_at(self, value: float) -> bool:
        return self.can_reverse_uniquely(value)

    def inverse_is_unique(self, value: float) -> bool:
        return self.can_reverse_uniquely(value)

    def reverse_unique(self, value: float) -> bool:
        return self.can_reverse_uniquely(value)

    def find_inverse(self, value: float) -> float | None:
        return self.unique_inverse(value)

    def can_reverse_uniquely(self, value: float) -> bool:
        return self.reverse_lookup(value).unique

    def reverse_status(self, value: float) -> ReverseLookup:
        return self.reverse_lookup(value)

    inverse_lookup = reverse_lookup
    lookup_inverse = reverse_lookup
    can_invert = can_reverse_uniquely
    is_reversible_at = can_reverse_uniquely

    @property
    def has_unique_inverse(self) -> bool | None:
        return self.is_injective

    def explain_reverse(self, value: float) -> str:
        return self.reverse_lookup(value).explain()

    def uniqueness_report(self) -> dict[str, Any]:
        """
        Describe whether this calibration can be inverted uniquely.

        The report is value-independent and is intended for callers which
        need to decide up front whether reverse lookups are meaningful.
        """
        monotonic = self.is_monotonic
        injective = self.is_injective
        bad = self.non_monotonic_segments()
        if injective:
            explanation = "calibration is injective; reverse lookups are unique"
        elif injective is False and monotonic is False:
            explanation = (
                "calibration is non-monotonic; reverse lookups can have "
                "multiple preimages"
            )
        elif injective is False and monotonic in (True, 0):
            explanation = (
                "calibration has a constant segment; reverse lookups are "
                "not unique on that interval"
            )
        else:
            explanation = (
                "calibration injectivity could not be determined; provide "
                "control points, an affine mapping, or a domain"
            )
        return {
            "monotonic": monotonic,
            "injective": injective,
            "unique_inverse": injective,
            "non_monotonic_segments": tuple(seg.as_dict() for seg in bad),
            "non_monotonic_ranges": tuple((seg.x0, seg.x1) for seg in bad),
            "explanation": explanation,
        }

    def monotonicity_report(self) -> dict[str, Any]:
        return self.uniqueness_report()

    def explain(self) -> str:
        return self.uniqueness_report()["explanation"]

    def is_injective_approx(self) -> bool:
        if self.domain is None:
            return False
        # A sample-based check is deliberately conservative.
        ys = [self.apply(x) for x in np.linspace(*sorted(self.domain), 401)]
        diffs = np.diff(ys)
        return bool(np.all(diffs > 0) or np.all(diffs < 0))

    # ------------------------------------------------------------------
    # Identity / hashing helpers
    # ------------------------------------------------------------------
    def signature(self) -> tuple:
        # For affine and piecewise mappings the data is enough to identify the
        # calibration.  Only an unconstrained callable needs object identity.
        func_id = None
        if self._func is not None and self._affine is None and not self._segments:
            func_id = id(self._func)
        return (
            tuple(self._control_points or ()),
            self._affine,
            func_id,
            self.measured_precision,
        )

    def __repr__(self) -> str:
        label = self.name or "unnamed"
        return "<Calibration {0!r} monotonic={1!r} injective={2!r}>".format(
            label, self.is_monotonic, self.is_injective
        )


def _points_from_array(values: Any) -> list[tuple[float, float]]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise CalibrationError(
            "Calibration arrays must be two-dimensional (n, 2) or (2, n)"
        )
    if arr.shape[1] == 2:
        return [(float(x), float(y)) for x, y in arr]
    if arr.shape[0] == 2:
        return [(float(x), float(y)) for x, y in arr.T]
    raise CalibrationError(
        "Calibration arrays must have a control-point axis of length 2"
    )


def _dedupe(values: Iterable[float], tolerance: float = 1e-12) -> list[float]:
    out: list[float] = []
    for value in sorted(float(v) for v in values):
        if not out or abs(value - out[-1]) > tolerance * max(1.0, abs(value)):
            out.append(value)
    return out


@dataclass(frozen=True)
class PeakIdentity:
    """
    Stable identity of a peak, anchored to the original (uncalibrated) data.

    The identity does not contain the calibrated position, so recalibrating a
    peak never changes which peak it is.
    """

    source_index: int
    original_mz: float
    original_intensity: float
    spectrum_id: Any = None
    spectrum_index: Any = None

    @property
    def index(self) -> int:
        return self.source_index

    @property
    def mz(self) -> float:
        return self.original_mz

    @property
    def i(self) -> float:
        return self.original_intensity

    def as_tuple(self) -> tuple:
        return (self.source_index, self.original_mz, self.original_intensity)

    def as_dict(self) -> dict:
        return {
            "source_index": self.source_index,
            "original_mz": self.original_mz,
            "original_intensity": self.original_intensity,
            "spectrum_id": self.spectrum_id,
            "spectrum_index": self.spectrum_index,
        }


@dataclass(frozen=True)
class PeakRecord:
    """One calibrated peak together with its original peak identity."""

    mz: float
    i: float
    source_index: int
    original_mz: float
    original_i: float
    version: Any = None
    identity: PeakIdentity | None = None

    def __iter__(self) -> Iterator[float]:
        yield self.mz
        yield self.i

    def __len__(self) -> int:
        return 2

    def __getitem__(self, item):
        return (self.mz, self.i)[item]

    def as_tuple(self) -> tuple[float, float]:
        return (self.mz, self.i)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, PeakRecord):
            return self.as_tuple() == other.as_tuple()
        try:
            return tuple(other) == self.as_tuple()
        except TypeError:
            return NotImplemented

    def __hash__(self) -> int:
        return hash(self.as_tuple())

    @property
    def calibrated_mz(self) -> float:
        return self.mz

    def as_dict(self) -> dict:
        return {
            "mz": self.mz,
            "i": self.i,
            "source_index": self.source_index,
            "original_mz": self.original_mz,
            "original_i": self.original_i,
            "version": self.version,
        }

    @property
    def peak_identity(self) -> PeakIdentity | None:
        return self.identity


class _CallableArray(np.ndarray):
    """numpy array which can also be called like ``peaks(peak_type)``."""

    def __call__(self, *args: Any, **kwargs: Any) -> "_CallableArray":
        return self


def _callable_array(value: Any) -> _CallableArray:
    return np.asarray(value).view(_CallableArray)


class PeakQueryIndex:
    """
    A query index bound to exactly one set of peak positions and one measured
    precision.  All derived caches live here, which is what prevents queries
    against different calibration versions (or different precisions) from
    sharing stale index data.
    """

    def __init__(
        self,
        positions: Sequence[float],
        measured_precision: float = 5e-6,
        *,
        intensities: Sequence[float] | None = None,
        source_indices: Sequence[int] | None = None,
        version: Any = None,
    ) -> None:
        self.version = version
        self.positions = np.asarray(positions, dtype=float)
        self.intensities = (
            np.asarray(intensities, dtype=float)
            if intensities is not None
            else np.ones_like(self.positions)
        )
        self.source_indices = np.asarray(
            source_indices if source_indices is not None else np.arange(len(self.positions)),
            dtype=int,
        )
        self._measured_precision = float(measured_precision)
        self._internal_precision = _internal_precision(self._measured_precision)
        self._sorted_order = None
        self._sorted_positions = None
        self._t_mz_set = None
        self._transformed_mz_with_error = None

    @property
    def measured_precision(self) -> float:
        return self._measured_precision

    @property
    def peak_count(self) -> int:
        return len(self.positions)

    @measured_precision.setter
    def measured_precision(self, value: float) -> None:
        value = float(value)
        if value == self._measured_precision and self._transformed_mz_with_error is not None:
            return
        self._measured_precision = value
        self._internal_precision = _internal_precision(value)
        self.invalidate()

    @property
    def internal_precision(self) -> int:
        return self._internal_precision

    def invalidate(self) -> None:
        self._t_mz_set = None
        self._transformed_mz_with_error = None

    @property
    def sorted_order(self) -> np.ndarray:
        if self._sorted_order is None:
            self._sorted_order = np.argsort(self.positions, kind="stable")
            self._sorted_positions = self.positions[self._sorted_order]
        return self._sorted_order

    @property
    def sorted_positions(self) -> np.ndarray:
        if self._sorted_positions is None:
            _ = self.sorted_order
        return self._sorted_positions

    def transform_mz(self, value: float) -> int:
        return int(round(float(value) * self._internal_precision))

    @property
    def t_mz_set(self) -> set[int]:
        if self._t_mz_set is None:
            transformed: set[int] = set()
            for mz in self.positions:
                transformed.update(self._transformed_range(mz))
            self._t_mz_set = transformed
        return self._t_mz_set

    def _transformed_range(self, mz: float) -> range:
        low = int(round((mz - (mz * self._measured_precision)) * self._internal_precision))
        high = int(
            round((mz + (mz * self._measured_precision)) * self._internal_precision)
        )
        return range(low, high + 1)

    @property
    def transformed_mz_with_error(self) -> dict[int, list[int]]:
        if self._transformed_mz_with_error is None:
            mapping: dict[int, list[int]] = {}
            for index, mz in enumerate(self.positions):
                for transformed in self._transformed_range(float(mz)):
                    mapping.setdefault(transformed, []).append(index)
            self._transformed_mz_with_error = mapping
        return self._transformed_mz_with_error

    @property
    def transformed_peaks(self) -> list[tuple[int, float]]:
        return [
            (self.transform_mz(float(mz)), float(self.intensities[index]))
            for index, mz in enumerate(self.positions)
        ]

    def has_peak(self, mz: float) -> list[int]:
        """Indices of peaks within this index's measured precision."""
        value = self.transform_mz(mz)
        return list(self.transformed_mz_with_error.get(value, []))

    def query_window(self, mz: float, error_window: float) -> list[int]:
        """Indices of peaks within an absolute m/z window."""
        mz = float(mz)
        window = abs(float(error_window))
        # Nudge the bounds by one ULP so decimal boundary values such as
        # 100.2 == 100.1 + 0.1 remain inclusive despite binary rounding.
        lower = np.nextafter(mz - window, -np.inf)
        upper = np.nextafter(mz + window, np.inf)
        low = bisect.bisect_left(self.sorted_positions, lower)
        high = bisect.bisect_right(self.sorted_positions, upper)
        return [int(self.sorted_order[pos]) for pos in range(low, high)]

    def query_ppm(self, mz: float, ppm: float) -> list[int]:
        mz = float(mz)
        return self.query_window(mz, mz * abs(float(ppm)))

    def query(
        self,
        mz: float,
        error_window: float | None = None,
        ppm: float | None = None,
        *,
        window: float | None = None,
        mz_window: float | None = None,
        tolerance: float | None = None,
        error: float | None = None,
        mz_error: float | None = None,
        error_window_da: float | None = None,
        ppm_error: float | None = None,
        error_ppm: float | None = None,
    ) -> list[int]:
        for alias in (
            window,
            mz_window,
            tolerance,
            error,
            mz_error,
            error_window_da,
        ):
            if alias is not None:
                error_window = alias
        for alias in (ppm_error, error_ppm):
            if alias is not None:
                ppm = alias
        if ppm is not None:
            return self.query_ppm(mz, ppm)
        if error_window is not None:
            return self.query_window(mz, error_window)
        return self.has_peak(mz)

    find = query
    search = query
    lookup = query

    def __len__(self) -> int:
        return len(self.positions)


def _internal_precision(measured_precision: float) -> int:
    value = float(measured_precision)
    if value <= 0:
        raise CalibrationError("measured_precision must be positive")
    return int(round(50000.0 / (value * 1e6)))


class PeakView(Sequence):
    """
    Immutable calibrated view of one spectrum's peaks.

    Arguments:
        data: a spectrum, a :class:`PeakView`, or an ``(n, 2)`` peak array.
        calibration: calibration mapping/object to apply.  ``None`` means the
            identity mapping (used for the original view).
        version: name of this view.
        measured_precision/precision: ppm precision for this view.  If not
            given, the calibration's precision or the source spectrum's
            precision is used.
        source_view: view providing the original peak identities.
        peak_type: peak type used when *data* is a spectrum.

    The view stores copies of its arrays.  No operation on one view mutates
    the source spectrum or another view.
    """

    def __init__(
        self,
        data: Any = None,
        calibration: Any = None,
        version: Any = None,
        measured_precision: float | None = None,
        *,
        precision: float | None = None,
        source_view: "PeakView" | None = None,
        peak_type: str = "centroided",
        source_indices: Sequence[int] | None = None,
        original_peaks: Any = None,
        spectrum_id: Any = None,
        spectrum_index: Any = None,
        name: Any = None,
        **kwargs: Any,
    ) -> None:
        if version is None:
            version = name
        if precision is not None and measured_precision is None:
            measured_precision = precision
        self.version = version
        self.name = version
        self.label = version
        self.calibration_version = version
        self.spectrum = None
        self.source_view = source_view
        self.peak_type = peak_type
        if isinstance(data, PeakView) and source_view is None:
            source_view = data
        if calibration is None and source_view is not None:
            calibration = source_view.calibration
        self._calibration = _coerce_calibration(calibration)
        self._spectrum_id = spectrum_id
        self._spectrum_index = spectrum_index

        source_positions = None
        source_intensities = None

        if isinstance(data, PeakView):
            source_view = source_view or data
            data = None
        if source_view is not None:
            source_positions = source_view._base_positions
            source_intensities = source_view._base_intensities
            original_positions = source_view.original_mz
            original_intensities = source_view.original_i
            original_indices = source_view.source_indices
            if measured_precision is None:
                measured_precision = source_view.measured_precision
            if self._spectrum_id is None:
                self._spectrum_id = source_view.spectrum_id
            if self._spectrum_index is None:
                self._spectrum_index = source_view.spectrum_index
        elif original_peaks is not None:
            original_arr = np.asarray(original_peaks, dtype=float)
            if original_arr.ndim != 2 or original_arr.shape[1] < 2:
                raise CalibrationError("original_peaks must be an (n, 2) array")
            original_positions = original_arr[:, 0].copy()
            original_intensities = original_arr[:, 1].copy()
            original_indices = (
                np.asarray(source_indices, dtype=int)
                if source_indices is not None
                else np.arange(len(original_positions))
            )
            source_positions = original_positions
            source_intensities = original_intensities
        else:
            source_positions, source_intensities, original_indices = self._extract(data, peak_type)
            original_positions = source_positions
            original_intensities = source_intensities
            if _looks_like_spectrum(data):
                self.spectrum = data
                self.peak_type = peak_type
                self._spectrum_id = getattr(data, "ID", self._spectrum_id)
                self._spectrum_index = getattr(data, "index", self._spectrum_index)

        source_positions = np.asarray(source_positions, dtype=float)
        source_intensities = np.asarray(source_intensities, dtype=float)
        # Arrays and spectra both carry original (uncalibrated) positions, so
        # the calibration is always applied here.  Zero-copy aliasing is
        # avoided on purpose: views must not share writable peak buffers.
        positions = self.calibration.apply_array(source_positions)
        if measured_precision is None:
            measured_precision = self.calibration.measured_precision
        if measured_precision is None and self.spectrum is not None:
            measured_precision = getattr(self.spectrum, "measured_precision", 5e-6)
        if measured_precision is None:
            measured_precision = 5e-6

        self._base_positions = np.array(source_positions, dtype=float, copy=True)
        self._base_intensities = np.array(source_intensities, dtype=float, copy=True)
        self._positions = np.array(positions, dtype=float, copy=True)
        self._intensities = np.array(source_intensities, dtype=float, copy=True)
        self._original_mz = np.array(original_positions, dtype=float, copy=True)
        self._original_i = np.array(original_intensities, dtype=float, copy=True)
        self._source_indices = np.array(original_indices, dtype=int, copy=True)
        if len(self._source_indices) != len(self._positions):
            raise CalibrationError("source index count does not match peak count")

        self._measured_precision = float(measured_precision)
        self._query_index = PeakQueryIndex(
            self._positions,
            self._measured_precision,
            intensities=self._intensities,
            source_indices=self._source_indices,
            version=self.version,
        )

        for array in (
            self._base_positions,
            self._base_intensities,
            self._positions,
            self._intensities,
            self._original_mz,
            self._original_i,
            self._source_indices,
        ):
            array.flags.writeable = False

    @classmethod
    def from_spectrum(
        cls,
        spectrum: Any,
        calibration: Any = None,
        version: Any = None,
        **kwargs: Any,
    ) -> "PeakView":
        """Build a view directly from a spectrum (convenience constructor)."""
        return cls(spectrum, calibration, version, **kwargs)

    @classmethod
    def from_peaks(
        cls,
        peaks: Any,
        calibration: Any = None,
        version: Any = None,
        **kwargs: Any,
    ) -> "PeakView":
        """Build a view directly from an ``(n, 2)`` peak array."""
        return cls(peaks, calibration, version, **kwargs)

    @staticmethod
    def _extract(data: Any, peak_type: str):
        if data is None:
            return np.array([]), np.array([]), np.array([], dtype=int)
        if isinstance(data, PeakView):
            return data._positions.copy(), data._intensities.copy(), data.source_indices.copy()
        if _looks_like_spectrum(data):
            peaks = np.asarray(data.peaks(peak_type), dtype=float)
            if peaks.size == 0:
                return np.array([]), np.array([]), np.array([], dtype=int)
            return peaks[:, 0], peaks[:, 1], np.arange(len(peaks), dtype=int)
        arr = np.asarray(data, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise CalibrationError("Peak data must be an (n, 2) array")
        return arr[:, 0], arr[:, 1], np.arange(len(arr), dtype=int)

    # ------------------------------------------------------------------
    # Basic snapshots
    # ------------------------------------------------------------------
    @property
    def positions(self) -> np.ndarray:
        return self._positions

    @property
    def mz(self) -> np.ndarray:
        return self._positions

    @property
    def i(self) -> np.ndarray:
        return self._intensities

    @property
    def intensity(self) -> np.ndarray:
        return self._intensities

    @property
    def peaks(self) -> np.ndarray:
        return _callable_array(np.column_stack((self._positions, self._intensities)))

    @property
    def peak_count(self) -> int:
        return len(self._positions)

    def __len__(self) -> int:
        return len(self._positions)

    def __call__(self) -> "PeakView":
        # ``views.original`` is a property, but legacy code may still call it
        # as ``views.original()``.  Returning self keeps both forms working.
        return self

    @property
    def calibration(self) -> Calibration:
        return self._calibration

    @calibration.setter
    def calibration(self, value: Any) -> None:
        raise AttributeError(
            "PeakView calibration is read-only; create a new view with "
            "recalibrated() or VersionedPeakViews.add(replace=True)"
        )

    def __iter__(self) -> Iterator[tuple[float, float]]:
        return iter(self.peak_list)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return [
                (float(self._positions[pos]), float(self._intensities[pos]))
                for pos in range(len(self))[item]
            ]
        return (float(self._positions[item]), float(self._intensities[item]))

    @property
    def source_indices(self) -> np.ndarray:
        return self._source_indices

    @property
    def original_index(self) -> np.ndarray:
        return self._source_indices

    @property
    def original_peak_index(self) -> np.ndarray:
        return self._source_indices

    @property
    def original_peak_indices(self) -> np.ndarray:
        return self._source_indices

    @property
    def peak_source_index(self) -> np.ndarray:
        return self._source_indices

    @property
    def source_peak_indices(self) -> np.ndarray:
        return self._source_indices

    @property
    def peak_ids(self) -> tuple[PeakIdentity, ...]:
        return self.identities

    def peak_id(self, index: int) -> PeakIdentity:
        return self.identities[index]

    @property
    def original_index_map(self) -> dict[int, int]:
        return {int(src): pos for pos, src in enumerate(self._source_indices)}

    @property
    def source_index_map(self) -> dict[int, int]:
        return self.original_index_map

    @property
    def original_mz(self) -> np.ndarray:
        return self._original_mz

    @property
    def original_i(self) -> np.ndarray:
        return self._original_i

    @property
    def original_peaks(self) -> np.ndarray:
        return _callable_array(np.column_stack((self._original_mz, self._original_i)))

    @property
    def base_positions(self) -> np.ndarray:
        return self._base_positions

    @property
    def base_peaks(self) -> np.ndarray:
        return _callable_array(
            np.column_stack((self._base_positions, self._base_intensities))
        )

    @property
    def ID(self) -> Any:
        return self._spectrum_id

    @property
    def index(self) -> Any:
        return self._spectrum_index

    @property
    def peak_list(self) -> list[tuple[float, float]]:
        return [
            (float(self._positions[pos]), float(self._intensities[pos]))
            for pos in range(len(self._positions))
        ]

    @property
    def uid(self) -> tuple:
        """Stable identity of this snapshot (version plus calibration)."""
        return (
            self.version,
            self._spectrum_id,
            self._spectrum_index,
            self.calibration.signature(),
            len(self._positions),
        )

    @property
    def spectrum_id(self) -> Any:
        return self._spectrum_id

    @property
    def spectrum_index(self) -> Any:
        return self._spectrum_index

    @property
    def identities(self) -> tuple[PeakIdentity, ...]:
        return tuple(
            PeakIdentity(
                int(self._source_indices[pos]),
                float(self._original_mz[pos]),
                float(self._original_i[pos]),
                self._spectrum_id,
                self._spectrum_index,
            )
            for pos in range(len(self._positions))
        )

    @property
    def peak_identities(self) -> tuple[PeakIdentity, ...]:
        return self.identities

    @property
    def peak_records(self) -> tuple[PeakRecord, ...]:
        return tuple(
            PeakRecord(
                mz=float(self._positions[pos]),
                i=float(self._intensities[pos]),
                source_index=int(self._source_indices[pos]),
                original_mz=float(self._original_mz[pos]),
                original_i=float(self._original_i[pos]),
                version=self.version,
                identity=self.identities[pos],
            )
            for pos in range(len(self._positions))
        )

    @property
    def records(self) -> tuple[PeakRecord, ...]:
        return self.peak_records

    def record(self, index: int) -> PeakRecord:
        return self.peak_records[index]

    def peak_identity(self, index: int) -> PeakIdentity:
        return self.identities[index]

    def identity_of(self, index: int) -> PeakIdentity:
        return self.identities[index]

    def original_of(self, index: int) -> tuple[float, float]:
        return self.original_peak(index)

    def identity(self, index: int) -> PeakIdentity:
        return self.peak_identity(index)

    def original_peak(self, index: int) -> tuple[float, float]:
        """Original ``(mz, intensity)`` for the view row at *index*."""
        return (float(self._original_mz[index]), float(self._original_i[index]))

    get_original_peak = original_peak

    def source_peak(self, index: int) -> tuple[float, float]:
        return (float(self._base_positions[index]), float(self._base_intensities[index]))

    def original_index_for(self, index: int) -> int:
        return int(self._source_indices[index])

    def view_index_for_original(self, source_index: int) -> int | None:
        return self.original_index_map.get(int(source_index))

    def original_for_view_index(self, index: int) -> PeakRecord:
        """Original peak record corresponding to this view row."""
        source = int(self._source_indices[index])
        return PeakRecord(
            mz=float(self._original_mz[index]),
            i=float(self._original_i[index]),
            source_index=source,
            original_mz=float(self._original_mz[index]),
            original_i=float(self._original_i[index]),
            version=self.version,
            identity=self.identities[index],
        )

    def correspondence(self) -> list[tuple[int, float, float, float]]:
        """
        Row-by-row mapping ``(source_index, original_mz, calibrated_mz, i)``.
        """
        return [
            (
                int(self._source_indices[pos]),
                float(self._original_mz[pos]),
                float(self._positions[pos]),
                float(self._intensities[pos]),
            )
            for pos in range(len(self))
        ]

    @property
    def original_correspondence(self) -> list[tuple[int, float, float, float]]:
        return self.correspondence()

    # ------------------------------------------------------------------
    # Precision / query index
    # ------------------------------------------------------------------
    @property
    def is_monotonic(self) -> bool | None:
        return self.calibration.is_monotonic

    @property
    def monotonic(self) -> bool | None:
        return self.calibration.is_monotonic

    @property
    def is_injective(self) -> bool | None:
        return self.calibration.is_injective

    @property
    def non_monotonic_segments(self) -> tuple[CalibrationSegment, ...]:
        return self.calibration.non_monotonic_segments()

    @property
    def measured_precision(self) -> float:
        return self._measured_precision

    @measured_precision.setter
    def measured_precision(self, value: float) -> None:
        # A precision change creates a fresh index rather than mutating the
        # shared one in place.  Queries made before the change cannot then see
        # derived caches built for the old precision.  Existing references to
        # the old index intentionally keep their old precision.
        self._measured_precision = float(value)
        self._query_index = PeakQueryIndex(
            self._positions,
            self._measured_precision,
            intensities=self._intensities,
            source_indices=self._source_indices,
            version=self.version,
        )

    @property
    def precision(self) -> float:
        return self._measured_precision

    @precision.setter
    def precision(self, value: float) -> None:
        self.measured_precision = value

    @property
    def internal_precision(self) -> int:
        return self._query_index.internal_precision

    @property
    def query_index(self) -> PeakQueryIndex:
        return self._query_index

    def transform_mz(self, value: float) -> int:
        return self._query_index.transform_mz(value)

    @property
    def version_name(self) -> Any:
        return self.version

    @property
    def index_map(self) -> dict[int, int]:
        return self.original_index_map

    @property
    def peak_index_map(self) -> dict[int, int]:
        return self.original_index_map

    @property
    def correspondence_map(self) -> dict[int, int]:
        return self.original_index_map

    def ppm2abs(self, value: float, ppm_value: float, direction: int = 1, factor: int = 1):
        return value + (value * (ppm_value * factor)) * direction

    @property
    def t_mz_set(self) -> set[int]:
        return self._query_index.t_mz_set

    @property
    def transformed_mz_with_error(self) -> dict[int, list[tuple[float, float]]]:
        # The public shape mirrors Spectrum.transformed_mz_with_error.
        return {
            key: [
                (float(self._positions[pos]), float(self._intensities[pos]))
                for pos in indices
            ]
            for key, indices in self._query_index.transformed_mz_with_error.items()
        }

    @property
    def transformed_peaks(self) -> list[tuple[int, float]]:
        return self._query_index.transformed_peaks

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def _indices(
        self,
        mz: float,
        error_window: float | None = None,
        ppm: float | None = None,
        **kwargs: Any,
    ) -> list[int]:
        return self._query_index.query(
            mz, error_window=error_window, ppm=ppm, **kwargs
        )

    def find_peaks(
        self,
        mz: float,
        error_window: float | None = None,
        ppm: float | None = None,
        **kwargs: Any,
    ) -> list[PeakRecord]:
        """Return peak records whose calibrated m/z is in the error window."""
        indices = self._indices(mz, error_window=error_window, ppm=ppm, **kwargs)
        indices.sort(key=lambda index: float(self._positions[index]))
        records = self.peak_records
        return [records[index] for index in indices]

    def query(
        self,
        mz: float,
        error_window: float | None = None,
        ppm: float | None = None,
        **kwargs: Any,
    ):
        return self.find_peaks(mz, error_window=error_window, ppm=ppm, **kwargs)

    find_peak = find_peaks
    find_peaks_in_window = find_peaks
    lookup = find_peaks
    search = find_peaks
    match = find_peaks
    query_peaks = find_peaks
    find = find_peaks

    def has_peak(
        self,
        mz: float | None = None,
        error_window: float | None = None,
        ppm: float | None = None,
        **kwargs: Any,
    ) -> list[tuple[float, float]]:
        """Return ``(calibrated_mz, intensity)`` tuples, like Spectrum.has_peak."""
        if mz is None:
            for key in ("mz2find", "query_mz", "mass"):
                if kwargs.get(key) is not None:
                    mz = kwargs.pop(key)
                    break
        if mz is None:
            raise TypeError("has_peak() requires an m/z value")
        return [
            record.as_tuple()
            for record in self.find_peaks(mz, error_window, ppm, **kwargs)
        ]

    def has_overlapping_peak(
        self,
        mz: float,
        error_window: float | None = None,
        ppm: float | None = None,
        **kwargs: Any,
    ) -> bool:
        return len(self.find_peaks(mz, error_window, ppm, **kwargs)) > 1

    def find_original_peaks(
        self,
        mz: float,
        error_window: float | None = None,
        ppm: float | None = None,
        **kwargs: Any,
    ) -> list[PeakRecord]:
        """Alias for :meth:`find_peaks`, emphasising original correspondence."""
        return self.find_peaks(mz, error_window=error_window, ppm=ppm, **kwargs)

    def reverse_lookup(
        self,
        mz: float,
        error_window: float | None = None,
        ppm: float | None = None,
        **kwargs: Any,
    ) -> ReverseLookup:
        """
        Reverse lookup from a calibrated position to original peaks.

        The result's ``preimages`` are :class:`PeakRecord` objects.  Its
        ``unique`` flag tells whether exactly one original peak is in the
        requested window.  If the calibration itself is non-monotonic, the
        explanation names the offending calibration segment(s).
        """
        records = self.find_peaks(
            mz, error_window=error_window, ppm=ppm, **kwargs
        )
        calibration_result = self.calibration.reverse_lookup(mz)
        if len(records) == 0:
            peak_reason = "no original peak maps into this calibrated window"
        elif len(records) > 1:
            peak_reason = "ambiguous reverse lookup: {0} original peaks in window".format(
                len(records)
            )
        else:
            peak_reason = ""

        # A unique peak-level answer also requires a unique calibration
        # preimage.  Otherwise a peak outside this window could share the same
        # calibrated position and the lookup would be misleading.
        unique = len(records) == 1 and calibration_result.unique
        reasons = [reason for reason in (peak_reason,) if reason]
        if not calibration_result.unique:
            extra = calibration_result.explain()
            if extra:
                reasons.append(extra)
        elif calibration_result.non_monotonic_segments and unique:
            reasons.append(
                "unique peak-level preimage; note that calibration segment(s) "
                + ", ".join(
                    "[{0:.6g}, {1:.6g}]".format(seg.x0, seg.x1)
                    for seg in calibration_result.non_monotonic_segments
                )
                + " are non-monotonic and can yield ambiguous lookups elsewhere"
            )
        return ReverseLookup(
            float(mz),
            records,
            unique,
            calibration_result.monotonic,
            "; ".join(reasons),
            calibration_result.non_monotonic_segments,
        )

    def reverse(
        self,
        mz: float,
        error_window: float | None = None,
        ppm: float | None = None,
        **kwargs: Any,
    ):
        return self.reverse_lookup(
            mz, error_window=error_window, ppm=ppm, **kwargs
        ).preimages

    def can_reverse_uniquely(
        self,
        mz: float,
        error_window: float | None = None,
        ppm: float | None = None,
        **kwargs: Any,
    ) -> bool:
        return self.reverse_lookup(
            mz, error_window=error_window, ppm=ppm, **kwargs
        ).unique

    def unique_original_peak(
        self,
        mz: float,
        error_window: float | None = None,
        ppm: float | None = None,
        **kwargs: Any,
    ) -> PeakRecord | None:
        result = self.reverse_lookup(
            mz, error_window=error_window, ppm=ppm, **kwargs
        )
        return result.preimages[0] if result.unique else None

    def uniqueness_report(self) -> dict[str, Any]:
        report = self.calibration.uniqueness_report()
        report["version"] = self.version
        report["peak_count"] = len(self)
        return report

    def monotonicity_report(self) -> dict[str, Any]:
        return self.uniqueness_report()

    def find_peaks_with_precision(
        self,
        mz: float,
        measured_precision: float,
        **kwargs: Any,
    ) -> list[PeakRecord]:
        """
        Query with a temporary index at another precision.

        This never mutates the view or its own query index, so it is safe to
        interleave with queries at the view's stored precision.
        """
        index = PeakQueryIndex(
            self._positions,
            measured_precision,
            intensities=self._intensities,
            source_indices=self._source_indices,
            version=self.version,
        )
        indices = index.query(mz, **kwargs)
        indices.sort(key=lambda pos: float(self._positions[pos]))
        records = self.peak_records
        return [records[pos] for pos in indices]

    reverse_peaks = reverse_lookup
    map_to_original = reverse_lookup
    original_for = reverse_lookup

    def reverse_calibration(self, mz: float) -> ReverseLookup:
        """Mathematical inverse of this view's calibration (not peak lookup)."""
        return self.calibration.reverse_lookup(mz)

    def can_reverse_calibration_uniquely(self, mz: float) -> bool:
        return self.calibration.can_reverse_uniquely(mz)

    def is_invertible_at(self, mz: float, **kwargs: Any) -> bool:
        return self.can_reverse_uniquely(mz, **kwargs)

    def inverse_is_unique(self, mz: float, **kwargs: Any) -> bool:
        return self.can_reverse_uniquely(mz, **kwargs)

    def explain_reverse(self, mz: float, **kwargs: Any) -> str:
        return self.reverse_lookup(mz, **kwargs).explain()

    # ------------------------------------------------------------------
    # Derived views (never mutate self)
    # ------------------------------------------------------------------
    def with_precision(self, measured_precision: float, version: Any = None) -> "PeakView":
        return PeakView(
            None,
            self.calibration,
            version if version is not None else self.version,
            measured_precision,
            source_view=self,
        )

    def recalibrated(
        self,
        calibration: Any = None,
        version: Any = None,
        measured_precision: float | None = None,
        *,
        relative: bool = False,
        **kwargs: Any,
    ) -> "PeakView":
        """
        Build a new view from this view's original peak identities.

        The receiving view is untouched, so the original positions, results
        and identities remain valid.  With ``relative=True`` the supplied
        calibration is applied on top of this view's calibration (useful for
        chaining a small correction without re-specifying the first mapping).
        """
        if relative:
            parent = self.calibration
            def composed(mz: float, parent=parent, child=_coerce_calibration(calibration)):
                return child.apply(parent.apply(mz))
            kwargs.setdefault("description", "relative correction")
            calibration = Calibration(composed)
        return PeakView(
            None,
            calibration,
            version,
            measured_precision,
            source_view=self,
            spectrum_id=self.spectrum_id,
            spectrum_index=self.spectrum_index,
            **kwargs,
        )

    def copy(self, version: Any = None) -> "PeakView":
        return self.with_precision(self.measured_precision, version=version)

    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "measured_precision": self.measured_precision,
            "monotonic": self.is_monotonic,
            "injective": self.is_injective,
            "peaks": self.peak_list,
            "original_indices": self.source_indices.tolist(),
            "original_peaks": self.original_peaks.tolist(),
        }

    def __repr__(self) -> str:
        return "<PeakView {0!r} peaks={1} precision={2:g}>".format(
            self.version, len(self), self.measured_precision
        )


class VersionedPeakViews:
    """
    Mapping of version name -> :class:`PeakView`.

    The original view is created lazily from the supplied spectrum/peaks.  A
    newly added version snapshots the source peaks; it never changes an
    existing view, and adding a duplicate version raises unless ``replace`` is
    explicitly requested.
    """

    ORIGINAL_ALIASES = {"original", "raw", "base", "default", "uncalibrated", "none"}

    def __init__(
        self,
        spectrum: Any = None,
        peaks: Any = None,
        *,
        peak_type: str = "centroided",
        measured_precision: float | None = None,
        original_version: str = "original",
    ) -> None:
        self.spectrum = spectrum
        self._peaks = peaks
        self.peak_type = peak_type
        self.measured_precision = measured_precision
        self.original_version = original_version
        self._views: dict[Any, PeakView] = {}
        self._active_version: Any = None

    @classmethod
    def from_spectrum(cls, spectrum: Any, **kwargs: Any) -> "VersionedPeakViews":
        return cls(spectrum=spectrum, **kwargs)

    @classmethod
    def from_peaks(cls, peaks: Any, **kwargs: Any) -> "VersionedPeakViews":
        return cls(peaks=peaks, **kwargs)

    def _default_precision(self) -> float:
        if self.measured_precision is not None:
            return self.measured_precision
        if self.spectrum is not None:
            return getattr(self.spectrum, "measured_precision", 5e-6)
        return 5e-6

    def _source_peaks(self) -> np.ndarray:
        if self._peaks is not None:
            arr = np.asarray(self._peaks, dtype=float)
            if arr.ndim != 2 or arr.shape[1] < 2:
                raise CalibrationError("Peak data must be an (n, 2) array")
            return arr[:, :2].copy()
        if self.spectrum is not None:
            return np.asarray(self.spectrum.peaks(self.peak_type), dtype=float)[:, :2].copy()
        return np.empty((0, 2), dtype=float)

    @property
    def original(self) -> PeakView:
        for alias in (self.original_version, "original"):
            if alias in self._views:
                return self._views[alias]
        peaks = self._source_peaks()
        view = PeakView(
            peaks,
            Calibration(),
            self.original_version,
            self._default_precision(),
            original_peaks=peaks,
            spectrum_id=getattr(self.spectrum, "ID", None),
            spectrum_index=getattr(self.spectrum, "index", None),
        )
        self._views[self.original_version] = view
        if self._active_version is None:
            self._active_version = self.original_version
        return view

    def get_original(self) -> PeakView:
        return self.original

    @property
    def original_view(self) -> PeakView:
        return self.original

    @property
    def base(self) -> PeakView:
        return self.original

    @property
    def original_peaks(self) -> np.ndarray:
        return self.original.peaks

    @property
    def original_mz(self) -> np.ndarray:
        return self.original.mz

    @property
    def original_i(self) -> np.ndarray:
        return self.original.i

    @property
    def original_identities(self) -> tuple[PeakIdentity, ...]:
        return self.original.identities

    def add(
        self,
        version: Any = None,
        calibration: Any = None,
        *,
        mapping: Any = None,
        calibration_version: Any = None,
        measured_precision: float | None = None,
        precision: float | None = None,
        peaks: Any = None,
        peak_type: str | None = None,
        source_version: Any = None,
        source_view: PeakView | None = None,
        replace: bool = False,
        **kwargs: Any,
    ) -> PeakView:
        if calibration is None and _looks_like_calibration(version):
            calibration = version
            version = None
        if calibration is None:
            calibration = mapping
        if measured_precision is None:
            measured_precision = precision
        if version is None and calibration_version is not None:
            version = calibration_version
        if version is None:
            version = kwargs.get("name") or kwargs.get("label")
        if version is None:
            index = len(self._views)
            version = "version_{0}".format(index)
            while version in self._views:
                index += 1
                version = "version_{0}".format(index)
        original_requested = version in self.ORIGINAL_ALIASES
        if original_requested:
            version = self.original_version

        existing = self._views.get(version)
        if original_requested and existing is not None:
            candidate = _coerce_calibration(calibration)
            if candidate.is_identity and (
                measured_precision is None
                or float(measured_precision) == existing.measured_precision
            ):
                return existing
            raise VersionConflictError(
                "the original peak view cannot be replaced; add the revised "
                "calibration under a new version name"
            )
        if existing is not None and not replace:
            candidate = _coerce_calibration(calibration)
            same_precision = (
                measured_precision is None
                or float(measured_precision) == existing.measured_precision
            )
            if candidate.signature() == existing.calibration.signature() and same_precision:
                return existing
            raise VersionConflictError(
                "peak view version {0!r} already exists; use replace=True to "
                "create a new version explicitly".format(version)
            )

        if source_view is None and source_version is not None:
            source_view = self.view(source_version)
        if source_view is None:
            source_view = self.original

        if peaks is not None:
            base = np.asarray(peaks, dtype=float)
            if base.ndim != 2 or base.shape[1] < 2:
                raise CalibrationError("Peak data must be an (n, 2) array")
            if measured_precision is None:
                measured_precision = source_view.measured_precision
            view = PeakView(
                base[:, :2],
                calibration,
                version,
                measured_precision,
                spectrum_id=source_view.spectrum_id,
                spectrum_index=source_view.spectrum_index,
            )
        else:
            view = PeakView(
                None,
                calibration,
                version,
                measured_precision,
                source_view=source_view,
                peak_type=peak_type or self.peak_type,
            )
        self._views[version] = view
        if self._active_version is None:
            self._active_version = version
        return view

    add_view = add
    register = add
    add_version = add
    add_calibration = add
    add_calibration_version = add
    create_view = add
    new_view = add
    add_peak_view = add

    def view(self, version: Any = None) -> PeakView:
        if version is None:
            version = self._active_version
        if isinstance(version, int) and not isinstance(version, bool):
            versions = list(self._views)
            if -len(versions) <= version < len(versions):
                version = versions[version]
        if version is None:
            return self.original
        if version in self.ORIGINAL_ALIASES or version == self.original_version:
            return self.original
        try:
            return self._views[version]
        except KeyError:
            raise KeyError(
                "Unknown peak view version {0!r}; known versions: {1}".format(
                    version, ", ".join(repr(v) for v in self._views) or "<none>"
                )
            )

    def get(self, version: Any = None, default: Any = None) -> Any:
        try:
            return self.view(version)
        except KeyError:
            return default

    get_view = view

    def get_peak_view(self, version: Any = None) -> PeakView:
        return self.view(version)

    @property
    def active_view(self) -> PeakView:
        return self.view()

    @property
    def active_version(self) -> Any:
        return self._active_version

    def activate(self, version: Any) -> PeakView:
        view = self.view(version)
        # Activation changes only the default lookup target.  It does not
        # mutate any view's data or caches.
        self._active_version = view.version if view.version is not None else version
        return view

    select = activate
    set_active = activate

    @property
    def versions(self) -> tuple[Any, ...]:
        _ = self.original
        return tuple(self._views.keys())

    @property
    def version_names(self) -> tuple[Any, ...]:
        return self.versions

    def get_version(self, version: Any = None) -> PeakView:
        return self.view(version)

    def has_version(self, version: Any) -> bool:
        return version in self._views

    def compare(self, version_a: Any, version_b: Any) -> list[dict[str, Any]]:
        """
        Compare two versions peak by peak using original identities.

        Returns one entry per peak with the calibrated positions in both
        versions.  A missing peak gets ``None`` for its position.
        """
        view_a = self.view(version_a)
        view_b = self.view(version_b)
        map_a = view_a.original_index_map
        map_b = view_b.original_index_map
        rows = []
        for source_index in sorted(set(map_a) | set(map_b)):
            pos_a = map_a.get(source_index)
            pos_b = map_b.get(source_index)
            rows.append(
                {
                    "source_index": source_index,
                    "mz_a": None if pos_a is None else float(view_a.mz[pos_a]),
                    "mz_b": None if pos_b is None else float(view_b.mz[pos_b]),
                    "original_mz": (
                        float(view_a.original_mz[pos_a])
                        if pos_a is not None
                        else float(view_b.original_mz[pos_b])
                    ),
                }
            )
        return rows

    def keys(self):
        _ = self.original
        return self._views.keys()

    def values(self):
        _ = self.original
        return self._views.values()

    def items(self):
        _ = self.original
        return self._views.items()

    def remove(self, version: Any) -> PeakView:
        if version == self.original_version:
            raise VersionConflictError(
                "the original peak view cannot be removed"
            )
        view = self._views.pop(version)
        if self._active_version == version:
            self._active_version = self.original_version if self._views else None
        return view

    discard = remove

    def __getitem__(self, version: Any) -> PeakView:
        return self.view(version)

    def __setitem__(self, version: Any, calibration: Any) -> PeakView:
        return self.add(version, calibration, replace=True)

    def original_peak_identity(self, source_index: int) -> Any:
        """Identity of an original peak by its stable source index."""
        original = self.original()
        position = original.view_index_for_original(source_index)
        if position is None:
            return None
        return original.identities[position]

    def remap_original_index(self, source_index: int, version: Any = None) -> int:
        """Row index of an original peak in another calibration version."""
        position = self.view(version).view_index_for_original(source_index)
        if position is None:
            raise KeyError(
                "original peak {0!r} is not present in version {1!r}".format(
                    source_index, version
                )
            )
        return position

    def __contains__(self, version: Any) -> bool:
        if version in self.ORIGINAL_ALIASES or version == self.original_version:
            return True
        return version in self._views

    def __iter__(self):
        _ = self.original
        return iter(self._views)

    def __len__(self) -> int:
        _ = self.original
        return len(self._views)

    def __repr__(self) -> str:
        return "<VersionedPeakViews versions={0!r} active={1!r}>".format(
            list(self._views), self._active_version
        )


# Public aliases keep the API discoverable under the names used in different
# code bases (versioned peak view / peak view collection / calibration view).
VersionedPeakView = PeakView
PeakViews = VersionedPeakViews
PeakViewCollection = VersionedPeakViews
PeakViewVersions = VersionedPeakViews
VersionedPeakViewsCollection = VersionedPeakViews
VersionedPeakViewCollection = VersionedPeakViews
CalibratedPeakView = PeakView
CalibrationVersion = Calibration
CalibrationMapping = Calibration
PeakIndex = PeakQueryIndex
PeakQueryCache = PeakQueryIndex


def peak_view(
    spectrum: Any = None,
    calibration: Any = None,
    version: Any = None,
    **kwargs: Any,
) -> PeakView:
    """Module-level convenience constructor for one calibrated peak view."""
    if spectrum is None and kwargs.get("peaks") is not None:
        spectrum = kwargs.pop("peaks")
    return PeakView(spectrum, calibration, version, **kwargs)


def versioned_peak_views(spectrum: Any = None, **kwargs: Any) -> VersionedPeakViews:
    """Module-level convenience constructor for a version collection."""
    return VersionedPeakViews(spectrum=spectrum, **kwargs)


def _coerce_calibration(calibration: Any) -> Calibration:
    if calibration is None:
        return Calibration()
    if isinstance(calibration, Calibration):
        return calibration
    return Calibration(calibration)


def _looks_like_spectrum(data: Any) -> bool:
    return hasattr(data, "peaks") and hasattr(data, "measured_precision")


def _looks_like_calibration(value: Any) -> bool:
    if isinstance(value, Calibration):
        return True
    if isinstance(value, Mapping):
        return True
    if callable(value) and not isinstance(value, type):
        return True
    if isinstance(value, np.ndarray):
        return value.ndim == 2
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return bool(value) and all(
            isinstance(item, (Sequence, np.ndarray))
            and not isinstance(item, (str, bytes))
            for item in value
        )
    return False


# The real functools.lru_cache is intentionally not used above: per-view
# caches must be explicitly invalidated when precision changes.  ``field`` is
# imported for the dataclasses and re-exported here for symmetry with older
# versions of this module.
__all__ = [
    "CalibratedPeakView",
    "Calibration",
    "CalibrationError",
    "CalibrationMapping",
    "CalibrationSegment",
    "CalibrationVersion",
    "NonInvertibleError",
    "PeakIdentity",
    "PeakIndex",
    "PeakQueryCache",
    "PeakQueryIndex",
    "PeakRecord",
    "PeakView",
    "PeakViewCollection",
    "PeakViewVersions",
    "PeakViews",
    "ReverseLookup",
    "peak_view",
    "versioned_peak_views",
    "VersionConflictError",
    "VersionedPeakView",
    "VersionedPeakViewCollection",
    "VersionedPeakViews",
    "VersionedPeakViewsCollection",
]
