from __future__ import annotations

import itertools
import math
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:  # pragma: no cover
    from edfio.edf_signal import _BaseSignal

if sys.version_info < (3, 10):  # pragma: no cover
    from typing_extensions import TypeAlias
else:  # pragma: no cover
    from typing import TypeAlias


_IntArray: TypeAlias = npt.NDArray[np.int64]
_FloatArray: TypeAlias = npt.NDArray[np.float64]
_BoolArray: TypeAlias = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class QuantizationRange:
    """
    A declared digital-to-physical range.

    The declaration defines the affine mapping used by EDF/BDF headers::

        physical = (digital + offset) * gain

    with ``gain = (physical_max - physical_min) / (digital_max - digital_min)``
    and ``offset = physical_max / gain - digital_max``.

    Parameters
    ----------
    digital_min, digital_max : int
        The declared digital range.
    physical_min, physical_max : float
        The declared physical range.
    """

    digital_min: int
    digital_max: int
    physical_min: float
    physical_max: float

    def __post_init__(self) -> None:
        digital_min = int(self.digital_min)
        digital_max = int(self.digital_max)
        physical_min = float(self.physical_min)
        physical_max = float(self.physical_max)
        if digital_min >= digital_max:
            raise ValueError(
                f"Digital minimum ({digital_min}) must be smaller than digital maximum ({digital_max})."
            )
        if not (math.isfinite(physical_min) and math.isfinite(physical_max)):
            raise ValueError("Physical range must contain only finite values")
        if physical_min >= physical_max:
            raise ValueError(
                f"Physical minimum ({physical_min}) must be smaller than physical maximum ({physical_max})."
            )
        object.__setattr__(self, "digital_min", digital_min)
        object.__setattr__(self, "digital_max", digital_max)
        object.__setattr__(self, "physical_min", physical_min)
        object.__setattr__(self, "physical_max", physical_max)

    @classmethod
    def from_signal(cls, signal: _BaseSignal) -> QuantizationRange:
        """Create the declaration currently stored in a signal header."""
        return cls(
            digital_min=signal.digital_min,
            digital_max=signal.digital_max,
            physical_min=signal.physical_min,
            physical_max=signal.physical_max,
        )

    @property
    def gain(self) -> float:
        """Physical units per digital code."""
        return (self.physical_max - self.physical_min) / (
            self.digital_max - self.digital_min
        )

    @property
    def offset(self) -> float:
        """Digital offset used by the header mapping."""
        return self.physical_max / self.gain - self.digital_max

    @property
    def resolution(self) -> float:
        """Physical size of one code (alias for :attr:`gain`)."""
        return self.gain

    @property
    def num_codes(self) -> int:
        """Number of codes in the declared digital range."""
        return self.digital_max - self.digital_min + 1

    def codes_to_physical(self, codes: npt.ArrayLike) -> _FloatArray:
        """Map digital codes to the physical values declared by this range."""
        values = np.asarray(codes, dtype=np.float64)
        return (values + self.offset) * self.gain

    def physical_to_digital(self, values: npt.ArrayLike) -> _FloatArray:
        """Map physical values to fractional digital codes."""
        array = np.asarray(values, dtype=np.float64)
        return array / self.gain - self.offset

    def quantize(
        self, values: npt.ArrayLike
    ) -> tuple[npt.NDArray[np.int16 | np.int32], _BoolArray, _BoolArray]:
        """
        Quantize physical values to this range.

        Values outside the declared physical range are clipped to the nearest
        endpoint. The returned boolean arrays mark low and high saturation
        separately from the quantized values.
        """
        array = np.asarray(values, dtype=np.float64)
        saturated_low = array < self.physical_min
        saturated_high = array > self.physical_max
        digital = np.round(self.physical_to_digital(array))
        digital = np.clip(digital, self.digital_min, self.digital_max)
        dtype = (
            np.int16
            if self.digital_min >= -(2**15) and self.digital_max <= 2**15 - 1
            else np.int32
        )
        return digital.astype(dtype), saturated_low, saturated_high

    def __iter__(self) -> Iterator[float]:
        """Allow unpacking as ``(digital_min, digital_max, physical_min, physical_max)``."""
        return iter(
            (
                self.digital_min,
                self.digital_max,
                self.physical_min,
                self.physical_max,
            )
        )


@dataclass(frozen=True)
class SourceDeclaration:
    """
    Declaration of the source values used for code recovery.

    Parameters
    ----------
    range : QuantizationRange
        The range that was used to produce the source values.
    kind : {"digital", "physical"}, default: "digital"
        Whether the source contains raw digital codes or physical values.
    error_bound : float, default: 0
        Additional absolute physical uncertainty, for example floating point
        round-off introduced by an earlier conversion.
    quantized : bool, default: True for physical sources
        Whether the source values are already quantized. If True, half the
        source resolution is included in the previous error bound. Raw digital
        codes default to exact; physical views default to quantized.
    saturated_low, saturated_high : bool, default: False
        Whether values at the source endpoints should be treated as clipped
        rather than as true extrema. This only applies when the corresponding
        source value equals an endpoint.
    """

    range: QuantizationRange
    kind: Literal["digital", "physical"] = "digital"
    error_bound: float = 0.0
    quantized: bool | None = None
    saturated_low: bool = False
    saturated_high: bool = False

    def __post_init__(self) -> None:
        if self.kind not in ("digital", "physical"):
            raise ValueError(
                f"Source kind must be 'digital' or 'physical', got {self.kind!r}"
            )
        if not math.isfinite(self.error_bound) or self.error_bound < 0:
            raise ValueError(
                f"Error bound must be finite and non-negative, got {self.error_bound}"
            )
        if self.quantized is None:
            object.__setattr__(self, "quantized", self.kind == "physical")
        object.__setattr__(self, "error_bound", float(self.error_bound))

    @classmethod
    def from_signal(
        cls,
        signal: _BaseSignal,
        *,
        kind: Literal["digital", "physical"] = "digital",
        error_bound: float = 0.0,
        quantized: bool | None = None,
        saturated_low: bool = False,
        saturated_high: bool = False,
    ) -> SourceDeclaration:
        """
        Create a declaration from a signal header.

        Raw digital codes should use ``kind="digital"`` and are exact by
        default. If only a physical view is available (for example
        ``signal.data``), use ``kind="physical"`` and keep ``quantized=True``
        so the previous quantization error is retained.
        """
        if kind not in ("digital", "physical"):
            raise ValueError(
                f"Source kind must be 'digital' or 'physical', got {kind!r}"
            )
        if quantized is None:
            quantized = kind == "physical"
        return cls(
            range=QuantizationRange.from_signal(signal),
            kind=kind,
            error_bound=error_bound,
            quantized=quantized,
            saturated_low=saturated_low,
            saturated_high=saturated_high,
        )

    @property
    def previous_error_bound(self) -> float:
        """Total absolute physical uncertainty already present in the source."""
        quantization_error = self.range.resolution / 2 if self.quantized else 0.0
        return quantization_error + self.error_bound


@dataclass(frozen=True)
class DigitalRecovery:
    """
    Result of recovering digital codes from declared source values.

    Each sample has an inclusive candidate interval ``code_low..code_high``.
    If ``code_low == code_high``, the original code is unique for that sample.
    Otherwise the interval contains all codes compatible with the source value
    and its declared error bound.

    Saturation is reported separately from the recovered interval: a clipped
    endpoint code is not evidence that the true physical value equals that
    endpoint.
    """

    source: SourceDeclaration
    target: QuantizationRange
    code_low: _IntArray
    code_high: _IntArray
    physical_low: _FloatArray
    physical_high: _FloatArray
    total_physical_low: _FloatArray = field(repr=False)
    total_physical_high: _FloatArray = field(repr=False)
    nearest: _IntArray = field(repr=False)
    uncertainty: _FloatArray = field(repr=False)
    source_saturated_low: _BoolArray = field(repr=False)
    source_saturated_high: _BoolArray = field(repr=False)
    saturated_low: _BoolArray = field(repr=False)
    saturated_high: _BoolArray = field(repr=False)
    may_saturate_low: _BoolArray = field(repr=False)
    may_saturate_high: _BoolArray = field(repr=False)

    @property
    def unique(self) -> _BoolArray:
        """Boolean mask marking samples whose digital code is unique."""
        return self.code_low == self.code_high

    @property
    def ambiguous(self) -> _BoolArray:
        """Boolean mask marking samples with more than one possible code."""
        return ~self.unique

    @property
    def codes(self) -> _IntArray:
        """Nearest-code representative for each sample."""
        return self.nearest

    @property
    def low(self) -> _IntArray:
        """Alias for :attr:`code_low`."""
        return self.code_low

    @property
    def high(self) -> _IntArray:
        """Alias for :attr:`code_high`."""
        return self.code_high

    @property
    def widths(self) -> _IntArray:
        """Number of possible codes for each sample."""
        return self.code_high - self.code_low + 1

    @property
    def same_grid(self) -> bool:
        """Whether source and target use the same declared grid."""
        return self.source.range == self.target

    @property
    def num_unique(self) -> int:
        """Number of samples with a unique code."""
        return int(np.count_nonzero(self.unique))

    @property
    def num_ambiguous(self) -> int:
        """Number of samples with a non-unique code."""
        return int(np.count_nonzero(self.ambiguous))

    @property
    def num_saturated_low(self) -> int:
        """Number of samples clipped at the target minimum."""
        return int(np.count_nonzero(self.saturated_low))

    @property
    def num_saturated_high(self) -> int:
        """Number of samples clipped at the target maximum."""
        return int(np.count_nonzero(self.saturated_high))

    @property
    def num_may_saturate_low(self) -> int:
        """Number of samples whose uncertainty may reach the target minimum."""
        return int(np.count_nonzero(self.may_saturate_low))

    @property
    def num_may_saturate_high(self) -> int:
        """Number of samples whose uncertainty may reach the target maximum."""
        return int(np.count_nonzero(self.may_saturate_high))

    @property
    def num_source_saturated_low(self) -> int:
        """Number of samples marked as clipped at the source minimum."""
        return int(np.count_nonzero(self.source_saturated_low))

    @property
    def num_source_saturated_high(self) -> int:
        """Number of samples marked as clipped at the source maximum."""
        return int(np.count_nonzero(self.source_saturated_high))

    @property
    def quantization_error_bound(self) -> float:
        """Physical half-width of one target code."""
        return self.target.resolution / 2

    @property
    def new_quantization_error_bound(self) -> float:
        """Additional quantization error introduced by the target conversion."""
        return 0.0 if self.same_grid else self.quantization_error_bound

    @property
    def total_error_bound(self) -> float:
        """Worst-case previous plus new quantization error in physical units."""
        previous = float(np.max(self.uncertainty)) if self.uncertainty.size else 0.0
        return previous + self.new_quantization_error_bound

    @property
    def is_unique(self) -> bool:
        """Whether every sample has a unique recovered code."""
        return bool(self.code_low.size) and bool(np.all(self.unique))

    def ranges(self) -> list[tuple[int, int]]:
        """Return the candidate code interval for every sample."""
        return list(zip(self.code_low.tolist(), self.code_high.tolist()))

    def next_source(self) -> SourceDeclaration:
        """Return a declaration carrying total error into a further conversion."""
        return _declaration_with_total_error(
            self.target,
            self.total_error_bound,
            saturated_low=bool(self.num_saturated_low or self.num_source_saturated_low),
            saturated_high=bool(
                self.num_saturated_high or self.num_source_saturated_high
            ),
        )

    def __len__(self) -> int:
        return int(self.code_low.size)


@dataclass(frozen=True)
class RangePreview:
    """
    Aggregate result of previewing a new range.

    The preview processes the source in chunks, so a lazy signal does not have
    to be loaded into memory as a whole.

    ``observed_min`` and ``observed_max`` include values clipped at a source
    endpoint. ``true_min`` and ``true_max`` exclude such clipped samples, so
    saturation does not masquerade as a true extremum. ``min_bound`` and
    ``max_bound`` additionally include the total (previous and new) error and
    become infinite if a source endpoint was clipped.
    """

    source: SourceDeclaration
    target: QuantizationRange
    num_samples: int = 0
    num_unique_codes: int = 0
    num_ambiguous_codes: int = 0
    num_saturated_low: int = 0
    num_saturated_high: int = 0
    num_may_saturate_low: int = 0
    num_may_saturate_high: int = 0
    num_source_saturated_low: int = 0
    num_source_saturated_high: int = 0
    observed_min: float | None = None
    observed_max: float | None = None
    true_min: float | None = None
    true_max: float | None = None
    unsaturated_min: float | None = None
    unsaturated_max: float | None = None
    min_bound: float = math.inf
    max_bound: float = -math.inf
    max_code_span: int = 0
    previous_error_bound: float = 0.0
    num_chunks: int = 0

    @property
    def same_grid(self) -> bool:
        """Whether source and target use the same declared grid."""
        return self.source.range == self.target

    @property
    def quantization_error_bound(self) -> float:
        """Physical half-width of one target code."""
        return self.target.resolution / 2

    @property
    def new_quantization_error_bound(self) -> float:
        """Additional quantization error introduced by the target conversion."""
        return 0.0 if self.same_grid else self.quantization_error_bound

    @property
    def total_error_bound(self) -> float:
        """Worst-case previous plus new quantization error in physical units."""
        return self.previous_error_bound + self.new_quantization_error_bound

    @property
    def num_unique(self) -> int:
        """Alias for :attr:`num_unique_codes`."""
        return self.num_unique_codes

    @property
    def num_ambiguous(self) -> int:
        """Alias for :attr:`num_ambiguous_codes`."""
        return self.num_ambiguous_codes

    @property
    def saturated_low(self) -> int:
        """Number of target low-saturation samples."""
        return self.num_saturated_low

    @property
    def saturated_high(self) -> int:
        """Number of target high-saturation samples."""
        return self.num_saturated_high

    @property
    def source_saturated_low(self) -> int:
        """Number of source low-saturation samples."""
        return self.num_source_saturated_low

    @property
    def source_saturated_high(self) -> int:
        """Number of source high-saturation samples."""
        return self.num_source_saturated_high

    @property
    def data_min(self) -> float | None:
        """Alias for :attr:`observed_min`."""
        return self.observed_min

    @property
    def data_max(self) -> float | None:
        """Alias for :attr:`observed_max`."""
        return self.observed_max

    @property
    def is_unique(self) -> bool:
        """Whether every sample has a unique recovered code."""
        return self.num_samples > 0 and self.num_ambiguous_codes == 0

    @property
    def has_saturation(self) -> bool:
        """Whether any target or source sample is saturated."""
        return bool(
            self.num_saturated_low
            or self.num_saturated_high
            or self.num_source_saturated_low
            or self.num_source_saturated_high
        )

    def next_source(self) -> SourceDeclaration:
        """Return a declaration carrying total error into a further conversion."""
        return _declaration_with_total_error(
            self.target,
            self.total_error_bound,
            saturated_low=bool(self.num_saturated_low or self.num_source_saturated_low),
            saturated_high=bool(
                self.num_saturated_high or self.num_source_saturated_high
            ),
        )

    @property
    def suggested_physical_range(self) -> tuple[float, float] | None:
        """Smallest target range covering all non-clipped samples plus error."""
        if self.true_min is None or self.true_max is None:
            return None
        error = self.total_error_bound
        return (self.true_min - error, self.true_max + error)


@dataclass(frozen=True)
class QuantizedView:
    """
    A physical view of a signal together with its quantization declaration.

    The view stores physical values on the target grid and the previous error
    bound that was already present in the source. Passing :attr:`declaration`
    to a subsequent conversion therefore preserves both errors instead of
    treating the view as an exact original signal.
    """

    values: _FloatArray
    declaration: SourceDeclaration
    codes: _IntArray = field(repr=False)
    code_low: _IntArray = field(repr=False)
    code_high: _IntArray = field(repr=False)
    saturated_low: _BoolArray = field(repr=False)
    saturated_high: _BoolArray = field(repr=False)

    @property
    def unique(self) -> _BoolArray:
        """Boolean mask marking values whose recovered code was unique."""
        return self.code_low == self.code_high

    @property
    def ambiguous(self) -> _BoolArray:
        """Boolean mask marking values whose recovered code was not unique."""
        return ~self.unique

    @property
    def num_unique(self) -> int:
        """Number of values whose recovered code was unique."""
        return int(np.count_nonzero(self.unique))

    @property
    def num_ambiguous(self) -> int:
        """Number of values whose recovered code was not unique."""
        return int(np.count_nonzero(self.ambiguous))

    @property
    def num_saturated_low(self) -> int:
        """Number of values clipped at the target minimum."""
        return int(np.count_nonzero(self.saturated_low))

    @property
    def num_saturated_high(self) -> int:
        """Number of values clipped at the target maximum."""
        return int(np.count_nonzero(self.saturated_high))

    def next_source(self) -> SourceDeclaration:
        """Return the declaration to use when converting this view again."""
        return self.declaration


def _declaration_with_total_error(
    target: QuantizationRange,
    total_error: float,
    *,
    saturated_low: bool = False,
    saturated_high: bool = False,
) -> SourceDeclaration:
    quantization = target.resolution / 2
    if quantization > 0 and total_error >= quantization:
        return SourceDeclaration(
            target,
            kind="physical",
            quantized=True,
            error_bound=total_error - quantization,
            saturated_low=saturated_low,
            saturated_high=saturated_high,
        )
    return SourceDeclaration(
        target,
        kind="physical",
        quantized=False,
        error_bound=max(0.0, total_error),
        saturated_low=saturated_low,
        saturated_high=saturated_high,
    )


def _as_range(value: QuantizationRange | Sequence[float]) -> QuantizationRange:
    if isinstance(value, QuantizationRange):
        return value
    try:
        length = len(value)
    except TypeError:
        raise TypeError(f"Invalid quantization range: {value!r}") from None
    if length == 4:  # noqa: PLR2004
        digital_min, digital_max, physical_min, physical_max = value
        return QuantizationRange(
            digital_min=int(digital_min),
            digital_max=int(digital_max),
            physical_min=float(physical_min),
            physical_max=float(physical_max),
        )
    raise TypeError(
        "Expected a QuantizationRange or four values "
        "(digital_min, digital_max, physical_min, physical_max)"
    )


def _mask(
    *,
    value: bool | npt.ArrayLike | None,
    shape: tuple[int, ...],
    default: _BoolArray,
) -> _BoolArray:
    if value is None:
        return default
    array = np.asarray(value, dtype=bool)
    if array.ndim == 0:
        return np.full(shape, bool(array), dtype=bool)
    return np.broadcast_to(array, shape).copy()


def _source_saturation_masks(
    source: SourceDeclaration,
    codes: _IntArray | None,
    observed: _FloatArray,
    *,
    saturated_low: bool | npt.ArrayLike | None,
    saturated_high: bool | npt.ArrayLike | None,
) -> tuple[_BoolArray, _BoolArray]:
    if source.kind == "digital":
        assert codes is not None
        default_low = (
            codes == source.range.digital_min
            if source.saturated_low
            else np.zeros(observed.shape, dtype=bool)
        )
        default_high = (
            codes == source.range.digital_max
            if source.saturated_high
            else np.zeros(observed.shape, dtype=bool)
        )
    else:
        default_low = (
            observed <= source.range.physical_min
            if source.saturated_low
            else np.zeros(observed.shape, dtype=bool)
        )
        default_high = (
            observed >= source.range.physical_max
            if source.saturated_high
            else np.zeros(observed.shape, dtype=bool)
        )
    return (
        _mask(value=saturated_low, shape=observed.shape, default=default_low),
        _mask(value=saturated_high, shape=observed.shape, default=default_high),
    )


def _resolve_source(
    source: SourceDeclaration | QuantizationRange | None, array: npt.NDArray
) -> SourceDeclaration:
    if isinstance(source, SourceDeclaration):
        return source
    kind: Literal["digital", "physical"] = (
        "digital" if np.issubdtype(array.dtype, np.integer) else "physical"
    )
    if source is None:
        raise ValueError("Source declaration is required")
    return SourceDeclaration(source, kind=kind)


def recover_digital_codes(  # noqa: PLR0912, PLR0915
    values: npt.ArrayLike,
    target: QuantizationRange | Sequence[float],
    *,
    source: SourceDeclaration | QuantizationRange | None = None,
    error_bound: float | npt.ArrayLike = 0.0,
    source_saturated_low: bool | npt.ArrayLike | None = None,
    source_saturated_high: bool | npt.ArrayLike | None = None,
) -> DigitalRecovery:
    """
    Recover the digital codes compatible with declared source values.

    The source may contain raw digital codes (``source.kind == "digital"``) or
    physical values (``source.kind == "physical"``). A physical view is never
    treated as an exact original value: if ``source.quantized`` is true, half
    the source resolution is part of the previous error bound, so a coarser
    source yields an interval of possible target codes.

    Parameters
    ----------
    values : npt.ArrayLike
        Source codes or physical values.
    target : QuantizationRange | tuple
        Target declaration as a :class:`QuantizationRange` or as
        ``(digital_min, digital_max, physical_min, physical_max)``.
    source : SourceDeclaration | QuantizationRange | None, default: None
        Source declaration. If omitted, integer input is interpreted as raw
        codes in ``target``; floating point input is interpreted as a quantized
        physical view of ``target``.
    error_bound : float | npt.ArrayLike, default: 0
        Additional absolute physical uncertainty of the source values.
    source_saturated_low, source_saturated_high : bool | npt.ArrayLike | None
        Per-sample saturation masks. If omitted, the corresponding flags of
        ``source`` are applied to values at the source endpoints.

    Returns
    -------
    DigitalRecovery
        Candidate intervals, saturation masks, and physical bounds.
    """
    target_range = _as_range(target)
    array = np.asarray(values)
    if source is None:
        if np.issubdtype(array.dtype, np.integer):
            source_declaration = SourceDeclaration(
                target_range, kind="digital", quantized=False
            )
        else:
            source_declaration = SourceDeclaration(
                target_range, kind="physical", quantized=True
            )
    else:
        source_declaration = _resolve_source(source, array)

    codes: _IntArray | None
    if source_declaration.kind == "digital":
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError("Digital source values must be integers")
        codes = array.astype(np.int64)
        if np.any(codes < source_declaration.range.digital_min) or np.any(
            codes > source_declaration.range.digital_max
        ):
            raise ValueError(
                f"Signal range [{codes.min()}, {codes.max()}] out of source "
                f"digital range: [{source_declaration.range.digital_min}, "
                f"{source_declaration.range.digital_max}]"
            )
        observed = source_declaration.range.codes_to_physical(codes)
    else:
        codes = None
        if np.issubdtype(array.dtype, np.integer):
            array = array.astype(np.float64)
        observed = array.astype(np.float64)

    if not np.all(np.isfinite(observed)):
        raise ValueError("Signal data must contain only finite values")

    source_sat_low, source_sat_high = _source_saturation_masks(
        source_declaration,
        codes,
        observed,
        saturated_low=source_saturated_low,
        saturated_high=source_saturated_high,
    )

    extra_error = np.asarray(error_bound, dtype=np.float64)
    if np.any(~np.isfinite(extra_error)) or np.any(extra_error < 0):
        raise ValueError("Error bound must be finite and non-negative")
    extra_error = np.broadcast_to(extra_error, observed.shape).astype(np.float64)
    previous_error = np.broadcast_to(
        source_declaration.previous_error_bound, observed.shape
    ).astype(np.float64)
    uncertainty = previous_error + extra_error
    same_grid = source_declaration.range == target_range
    new_error = 0.0 if same_grid else target_range.resolution / 2

    if same_grid and source_declaration.quantized:
        # Values on the target grid are exact grid points, so the grid's own
        # quantization does not widen the candidate interval. Any explicit
        # prior error still does, because it is not implied by the grid.
        candidate_error = source_declaration.error_bound + extra_error
    else:
        candidate_error = uncertainty

    q = target_range.physical_to_digital(observed)
    nearest = np.clip(
        np.round(q), target_range.digital_min, target_range.digital_max
    ).astype(np.int64)
    radius = candidate_error / target_range.resolution

    if (
        source_declaration.kind == "digital"
        and same_grid
        and np.all(candidate_error == 0)
        and not np.any(source_sat_low | source_sat_high)
    ):
        assert codes is not None
        code_low = codes.copy()
        code_high = codes.copy()
    else:
        # A code c is compatible if |q - c| <= radius + 0.5. A value exactly
        # halfway between two codes is deliberately non-unique.
        q_low = q - radius
        q_high = q + radius
        q_low = np.where(np.isfinite(q_low), q_low, target_range.digital_min)
        q_high = np.where(np.isfinite(q_high), q_high, target_range.digital_max)
        code_low = np.ceil(q_low - 0.5).astype(np.int64)
        code_high = np.floor(q_high + 0.5).astype(np.int64)
        code_low = np.where(source_sat_low, target_range.digital_min, code_low)
        code_high = np.where(source_sat_high, target_range.digital_max, code_high)
        code_low = np.clip(code_low, target_range.digital_min, target_range.digital_max)
        code_high = np.clip(
            code_high, target_range.digital_min, target_range.digital_max
        )
        code_high = np.maximum(code_low, code_high)

    physical_low = np.where(source_sat_low, -np.inf, observed - uncertainty)
    physical_high = np.where(source_sat_high, np.inf, observed + uncertainty)
    total_error = uncertainty + new_error
    total_physical_low = np.where(source_sat_low, -np.inf, observed - total_error)
    total_physical_high = np.where(source_sat_high, np.inf, observed + total_error)

    saturated_low = observed < target_range.physical_min
    saturated_high = observed > target_range.physical_max
    may_saturate_low = (q - radius < target_range.digital_min) | source_sat_low
    may_saturate_high = (q + radius > target_range.digital_max) | source_sat_high

    return DigitalRecovery(
        source=source_declaration,
        target=target_range,
        code_low=code_low,
        code_high=code_high,
        physical_low=physical_low,
        physical_high=physical_high,
        total_physical_low=total_physical_low,
        total_physical_high=total_physical_high,
        nearest=nearest,
        uncertainty=uncertainty,
        source_saturated_low=source_sat_low,
        source_saturated_high=source_sat_high,
        saturated_low=saturated_low,
        saturated_high=saturated_high,
        may_saturate_low=may_saturate_low,
        may_saturate_high=may_saturate_high,
    )


class _PreviewAccumulator:
    def __init__(self, source: SourceDeclaration, target: QuantizationRange):
        self.source = source
        self.target = target
        self.num_samples = 0
        self.num_unique_codes = 0
        self.num_ambiguous_codes = 0
        self.num_saturated_low = 0
        self.num_saturated_high = 0
        self.num_may_saturate_low = 0
        self.num_may_saturate_high = 0
        self.num_source_saturated_low = 0
        self.num_source_saturated_high = 0
        self.observed_min: float | None = None
        self.observed_max: float | None = None
        self.true_min: float | None = None
        self.true_max: float | None = None
        self.unsaturated_min: float | None = None
        self.unsaturated_max: float | None = None
        self.min_bound = math.inf
        self.max_bound = -math.inf
        self.max_code_span = 0
        self.previous_error_bound = 0.0
        self.num_chunks = 0

    def update(self, recovery: DigitalRecovery, observed: _FloatArray) -> None:
        size = observed.size
        if size == 0:
            return
        self.num_chunks += 1
        self.num_samples += size
        self.num_unique_codes += recovery.num_unique
        self.num_ambiguous_codes += recovery.num_ambiguous
        self.num_saturated_low += recovery.num_saturated_low
        self.num_saturated_high += recovery.num_saturated_high
        self.num_may_saturate_low += recovery.num_may_saturate_low
        self.num_may_saturate_high += recovery.num_may_saturate_high
        self.num_source_saturated_low += recovery.num_source_saturated_low
        self.num_source_saturated_high += recovery.num_source_saturated_high
        self.max_code_span = max(
            self.max_code_span, int(np.max(recovery.code_high - recovery.code_low))
        )
        self.previous_error_bound = max(
            self.previous_error_bound, float(np.max(recovery.uncertainty))
        )

        observed_min = float(np.min(observed))
        observed_max = float(np.max(observed))
        self.observed_min = (
            observed_min
            if self.observed_min is None
            else min(self.observed_min, observed_min)
        )
        self.observed_max = (
            observed_max
            if self.observed_max is None
            else max(self.observed_max, observed_max)
        )

        unsaturated = ~(
            recovery.source_saturated_low
            | recovery.source_saturated_high
            | recovery.saturated_low
            | recovery.saturated_high
        )
        if np.any(unsaturated):
            unsat_min = float(np.min(observed[unsaturated]))
            unsat_max = float(np.max(observed[unsaturated]))
            self.unsaturated_min = (
                unsat_min
                if self.unsaturated_min is None
                else min(self.unsaturated_min, unsat_min)
            )
            self.unsaturated_max = (
                unsat_max
                if self.unsaturated_max is None
                else max(self.unsaturated_max, unsat_max)
            )

        true_mask = ~(recovery.source_saturated_low | recovery.source_saturated_high)
        if np.any(true_mask):
            true_min = float(np.min(observed[true_mask]))
            true_max = float(np.max(observed[true_mask]))
            self.true_min = (
                true_min if self.true_min is None else min(self.true_min, true_min)
            )
            self.true_max = (
                true_max if self.true_max is None else max(self.true_max, true_max)
            )

        if recovery.num_source_saturated_low:
            self.min_bound = -math.inf
        else:
            self.min_bound = min(
                self.min_bound, float(np.min(recovery.total_physical_low))
            )
        if recovery.num_source_saturated_high:
            self.max_bound = math.inf
        else:
            self.max_bound = max(
                self.max_bound, float(np.max(recovery.total_physical_high))
            )

    def result(self) -> RangePreview:
        return RangePreview(
            source=self.source,
            target=self.target,
            num_samples=self.num_samples,
            num_unique_codes=self.num_unique_codes,
            num_ambiguous_codes=self.num_ambiguous_codes,
            num_saturated_low=self.num_saturated_low,
            num_saturated_high=self.num_saturated_high,
            num_may_saturate_low=self.num_may_saturate_low,
            num_may_saturate_high=self.num_may_saturate_high,
            num_source_saturated_low=self.num_source_saturated_low,
            num_source_saturated_high=self.num_source_saturated_high,
            observed_min=self.observed_min,
            observed_max=self.observed_max,
            true_min=self.true_min,
            true_max=self.true_max,
            unsaturated_min=self.unsaturated_min,
            unsaturated_max=self.unsaturated_max,
            min_bound=self.min_bound,
            max_bound=self.max_bound,
            max_code_span=self.max_code_span,
            previous_error_bound=self.previous_error_bound,
            num_chunks=self.num_chunks,
        )


def _observed(source: SourceDeclaration, chunk: npt.ArrayLike) -> _FloatArray:
    if source.kind == "digital":
        return source.range.codes_to_physical(chunk)
    return np.asarray(chunk, dtype=np.float64)


def preview_requantization(
    chunks: Iterable[npt.ArrayLike],
    source: SourceDeclaration | QuantizationRange,
    target: QuantizationRange | Sequence[float],
    *,
    error_bound: float | npt.ArrayLike = 0.0,
) -> RangePreview:
    """
    Preview a new range from an iterable of source chunks.

    The iterable is consumed one chunk at a time, so callers can stream a
    recording without loading all samples at once. The previous quantization
    error declared by ``source`` is retained and added to the new target
    quantization error.

    Parameters
    ----------
    chunks : Iterable[npt.ArrayLike]
        Raw digital code chunks or physical value chunks, according to
        ``source.kind``.
    source : SourceDeclaration | QuantizationRange
        Source declaration. A plain range is interpreted as a quantized
        physical view.
    target : QuantizationRange | tuple
        Target declaration.
    error_bound : float | npt.ArrayLike, default: 0
        Additional absolute physical uncertainty of the source values.

    Returns
    -------
    RangePreview
        Aggregate statistics, including separate saturation counts and true
        extrema of non-saturated samples.
    """
    target_range = _as_range(target)
    iterator = iter(chunks)
    first_chunk: npt.NDArray | None = None
    for chunk in iterator:
        candidate = np.asarray(chunk)
        if candidate.size:
            first_chunk = candidate
            break
    if isinstance(source, SourceDeclaration):
        source_declaration = source
    else:
        kind: Literal["digital", "physical"] = (
            "digital"
            if first_chunk is not None and np.issubdtype(first_chunk.dtype, np.integer)
            else "physical"
        )
        source_declaration = SourceDeclaration(source, kind=kind)
    accumulator = _PreviewAccumulator(source_declaration, target_range)
    all_chunks: Iterable[npt.ArrayLike]
    if first_chunk is None:
        all_chunks = ()
    else:
        all_chunks = itertools.chain((first_chunk,), iterator)
    for chunk in all_chunks:
        array = np.asarray(chunk)
        if array.size == 0:
            continue
        recovery = recover_digital_codes(
            array,
            target_range,
            source=source_declaration,
            error_bound=error_bound,
        )
        accumulator.update(recovery, _observed(source_declaration, array))
    return accumulator.result()


def requantize(
    values: npt.ArrayLike,
    source: SourceDeclaration | QuantizationRange,
    target: QuantizationRange | Sequence[float],
    *,
    error_bound: float | npt.ArrayLike = 0.0,
) -> QuantizedView:
    """
    Convert values to a new physical range while retaining previous error.

    The returned :class:`QuantizedView` contains physical values rounded to the
    target grid and a declaration whose ``error_bound`` carries the source's
    previous error plus the new target quantization error. A further
    conversion using ``view.declaration`` therefore does not treat the view as
    an exact original signal.

    Parameters
    ----------
    values : npt.ArrayLike
        Source codes or physical values, according to ``source.kind``.
    source : SourceDeclaration | QuantizationRange
        Source declaration.
    target : QuantizationRange | tuple
        Target declaration.
    error_bound : float | npt.ArrayLike, default: 0
        Additional absolute physical uncertainty of the source values.

    Returns
    -------
    QuantizedView
        Target-grid values and the declaration to use for further conversions.
    """
    target_range = _as_range(target)
    if isinstance(source, SourceDeclaration):
        source_declaration = source
    else:
        source_declaration = SourceDeclaration(
            source,
            kind=(
                "digital"
                if np.issubdtype(np.asarray(values).dtype, np.integer)
                else "physical"
            ),
        )
    recovery = recover_digital_codes(
        values, target_range, source=source_declaration, error_bound=error_bound
    )
    codes = recovery.codes
    physical = target_range.codes_to_physical(codes)
    physical.setflags(write=False)
    extra_error = 0.0
    if np.asarray(error_bound).size:
        extra_error = float(np.max(np.asarray(error_bound, dtype=np.float64)))
    same_grid = source_declaration.range == target_range
    new_quantization = 0.0 if same_grid else target_range.resolution / 2
    total_error = (
        source_declaration.previous_error_bound + extra_error + new_quantization
    )
    declaration = _declaration_with_total_error(
        target_range,
        total_error,
        saturated_low=bool(
            source_declaration.saturated_low or recovery.num_saturated_low
        ),
        saturated_high=bool(
            source_declaration.saturated_high or recovery.num_saturated_high
        ),
    )
    return QuantizedView(
        values=physical,
        declaration=declaration,
        codes=codes,
        code_low=recovery.code_low,
        code_high=recovery.code_high,
        saturated_low=recovery.saturated_low,
        saturated_high=recovery.saturated_high,
    )


def _signal_target_range(
    signal: _BaseSignal,
    physical_range: tuple[float, float] | None,
    digital_range: tuple[int, int] | None,
) -> QuantizationRange:
    return QuantizationRange(
        digital_min=signal.digital_min if digital_range is None else digital_range[0],
        digital_max=signal.digital_max if digital_range is None else digital_range[1],
        physical_min=(
            signal.physical_min if physical_range is None else physical_range[0]
        ),
        physical_max=(
            signal.physical_max if physical_range is None else physical_range[1]
        ),
    )


def _iter_signal_chunks(
    signal: _BaseSignal,
    *,
    kind: Literal["digital", "physical"],
    chunk_size: int,
) -> Iterator[npt.ArrayLike]:
    if chunk_size <= 0:
        raise ValueError(f"Chunk size must be positive, got {chunk_size}")
    if signal._digital is not None:
        num_samples = signal._num_samples
        for start in range(0, num_samples, chunk_size):
            stop = min(start + chunk_size, num_samples)
            digital = signal.digital[start:stop]
            yield digital if kind == "digital" else signal._calibrate(digital)
        return
    lazy_loader = signal._lazy_loader
    if lazy_loader is None:
        raise ValueError("Signal data not set")
    samples_per_record = signal.samples_per_data_record
    batch_records = max(1, chunk_size // samples_per_record)
    num_records = lazy_loader.buffer.shape[0]
    for first_record in range(0, num_records, batch_records):
        last_record = min(first_record + batch_records, num_records)
        digital = lazy_loader.load(first_record, last_record)
        yield digital if kind == "digital" else signal._calibrate(digital)


def _preview_signal_range(
    signal: _BaseSignal,
    *,
    physical_range: tuple[float, float] | None = None,
    digital_range: tuple[int, int] | None = None,
    error_bound: float = 0.0,
    source_kind: Literal["digital", "physical"] = "digital",
    source_saturated_low: bool = False,
    source_saturated_high: bool = False,
    chunk_size: int = 1_000_000,
    source: SourceDeclaration | None = None,
) -> RangePreview:
    if source is None:
        source = SourceDeclaration.from_signal(
            signal,
            kind=source_kind,
            error_bound=error_bound,
            saturated_low=source_saturated_low,
            saturated_high=source_saturated_high,
        )
    target = _signal_target_range(signal, physical_range, digital_range)
    return preview_requantization(
        _iter_signal_chunks(signal, kind=source.kind, chunk_size=chunk_size),
        source,
        target,
    )


def _recover_signal_codes(
    signal: _BaseSignal,
    values: npt.ArrayLike | None = None,
    *,
    target: QuantizationRange | Sequence[float] | None = None,
    source: SourceDeclaration | QuantizationRange | None = None,
    error_bound: float | npt.ArrayLike = 0.0,
    source_saturated_low: bool | npt.ArrayLike | None = None,
    source_saturated_high: bool | npt.ArrayLike | None = None,
) -> DigitalRecovery:
    if target is None:
        target = QuantizationRange.from_signal(signal)
    if values is None:
        if source is None:
            source = SourceDeclaration.from_signal(signal, kind="digital")
        if isinstance(source, SourceDeclaration):
            values = signal.digital if source.kind == "digital" else signal.data
        else:
            values = signal.digital
    return recover_digital_codes(
        values,
        target,
        source=source,
        error_bound=error_bound,
        source_saturated_low=source_saturated_low,
        source_saturated_high=source_saturated_high,
    )
