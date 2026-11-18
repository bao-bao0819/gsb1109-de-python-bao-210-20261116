from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from numbers import Real
from typing import Any

import numpy as np

from edfio._header_field import encode_float
from edfio.edf import Bdf, Edf
from edfio.edf_signal import (
    _EDF_DEFAULT_RANGE,
    EdfSignal,
    _FloatRange,
    _IntRange,
    _round_float_to_8_characters,
)

__all__ = [
    "BdfConversionReport",
    "BdfToEdfReport",
    "BdfToEdfResult",
    "ChannelConversionReport",
    "ChannelReport",
    "ConversionReport",
    "ConversionResult",
    "SaturatedSample",
    "bdf_to_edf",
    "convert_bdf_to_edf",
    "convert_signal_to_edf",
    "to_edf",
]


@dataclass(frozen=True)
class SaturatedSample:
    """A sample clipped to the EDF digital range during conversion.

    Parameters
    ----------
    channel : str
        Label of the signal containing the sample.
    index : int
        Index of the sample within the signal.
    value : float
        Physical value before conversion.
    decoded_value : float
        Physical value after conversion and decoding.
    """

    channel: str
    index: int
    value: float
    decoded_value: float

    @property
    def error(self) -> float:
        """Absolute difference between the original and decoded value."""
        return abs(self.decoded_value - self.value)

    @property
    def absolute_error(self) -> float:
        """Alias of :attr:`error`."""
        return self.error

    @property
    def original_value(self) -> float:
        """Alias of :attr:`value`."""
        return self.value

    @property
    def encoded_value(self) -> float:
        """Alias of :attr:`decoded_value`."""
        return self.decoded_value


@dataclass(frozen=True)
class ChannelConversionReport:
    """Conversion details and error statistics for a single channel.

    Parameters
    ----------
    index : int
        Index of the signal in :attr:`Bdf.signals`.
    label : str
        Signal label.
    allowed_error : float
        Maximum absolute physical error allowed for this channel.
    physical_range : tuple[float, float]
        Physical range written to the EDF header.
    digital_range : tuple[int, int]
        Digital range written to the EDF header.
    quantization_step : float
        Physical difference represented by one digital step.
    max_abs_error : float
        Largest absolute error over all samples after conversion and decoding.
    saturated_samples : tuple[SaturatedSample, ...]
        Samples clipped to the digital range.
    num_samples : int
        Total number of samples in the channel.
    worst_sample_index : int
        Index of the sample with the largest absolute error.
    reason : str | None
        Why the allowed error could not be met, or `None` if it was met.
    """

    index: int
    label: str
    allowed_error: float
    physical_range: _FloatRange
    digital_range: _IntRange
    quantization_step: float
    max_abs_error: float
    saturated_samples: tuple[SaturatedSample, ...]
    num_samples: int
    worst_sample_index: int
    reason: str | None = None

    @property
    def satisfied(self) -> bool:
        """Whether the allowed error is met over the entire channel."""
        return self.reason is None and self.max_abs_error <= self.allowed_error

    @property
    def unsatisfiable(self) -> bool:
        """Whether the allowed error could not be met."""
        return not self.satisfied

    @property
    def error_bound(self) -> float:
        """Upper bound covering every sample of the segment.

        At least half a quantization step, and at least the largest realized
        error after decoding *all* samples back to physical values.  It is
        therefore not merely derived from the segment's extrema.
        """
        return max(self.quantization_error_bound, self.max_abs_error)

    @property
    def quantization_error_bound(self) -> float:
        """Theoretical bound of half a quantization step."""
        return self.quantization_step / 2

    @property
    def tolerance(self) -> float:
        """Alias of :attr:`allowed_error`."""
        return self.allowed_error

    @property
    def max_error(self) -> float:
        """Alias of :attr:`max_abs_error`."""
        return self.max_abs_error

    @property
    def num_saturated_samples(self) -> int:
        """Number of saturated samples."""
        return len(self.saturated_samples)

    @property
    def has_saturation(self) -> bool:
        """Whether at least one sample was clipped."""
        return bool(self.saturated_samples)

    @property
    def saturated_sample_indices(self) -> tuple[int, ...]:
        """Indices of all saturated samples."""
        return tuple(sample.index for sample in self.saturated_samples)

    @property
    def saturated_indices(self) -> tuple[int, ...]:
        """Alias of :attr:`saturated_sample_indices`."""
        return self.saturated_sample_indices

    @property
    def worst_saturated_sample(self) -> SaturatedSample | None:
        """Saturated sample with the largest error, if any."""
        return max(self.saturated_samples, key=lambda s: s.error, default=None)


class ChannelReportCollection(tuple[ChannelConversionReport, ...]):
    """Immutable sequence of channel reports that also matches labels."""

    def __new__(cls, reports: Iterable[ChannelConversionReport] = ()) -> Any:
        return super().__new__(cls, reports)

    @property
    def labels(self) -> tuple[str, ...]:
        """Labels of the contained channels."""
        return tuple(report.label for report in self)

    def __contains__(self, item: object) -> bool:
        if isinstance(item, str):
            return any(report.label == item for report in self)
        return super().__contains__(item)


@dataclass(frozen=True)
class BdfToEdfReport:
    """Report describing the result of a BDF to EDF conversion.

    Parameters
    ----------
    channels : tuple[ChannelConversionReport, ...]
        Conversion details for every non-annotation signal.
    """

    channels: ChannelReportCollection

    @property
    def unsatisfiable_channels(self) -> ChannelReportCollection:
        """Channels whose allowed error could not be met."""
        return ChannelReportCollection(
            report for report in self.channels if not report.satisfied
        )

    @property
    def unsatisfied_channels(self) -> ChannelReportCollection:
        """Alias of :attr:`unsatisfiable_channels`."""
        return self.unsatisfiable_channels

    @property
    def failed_channels(self) -> ChannelReportCollection:
        """Alias of :attr:`unsatisfiable_channels`."""
        return self.unsatisfiable_channels

    @property
    def unsatisfiable_channel_labels(self) -> tuple[str, ...]:
        """Labels of all channels whose allowed error could not be met."""
        return self.unsatisfiable_channels.labels

    @property
    def saturated_sample_indices_by_channel(self) -> dict[str, tuple[int, ...]]:
        """Saturated sample indices grouped by channel label."""
        return {
            report.label: report.saturated_sample_indices
            for report in self.channels
            if report.saturated_samples
        }

    @property
    def satisfied(self) -> bool:
        """Alias of :attr:`all_satisfied`."""
        return self.all_satisfied

    @property
    def ok(self) -> bool:
        """Alias of :attr:`all_satisfied`."""
        return self.all_satisfied

    @property
    def satisfied_channels(self) -> ChannelReportCollection:
        """Channels whose allowed error was met."""
        return ChannelReportCollection(
            report for report in self.channels if report.satisfied
        )

    @property
    def saturated_channels(self) -> ChannelReportCollection:
        """Channels containing at least one saturated sample."""
        return ChannelReportCollection(
            report for report in self.channels if report.saturated_samples
        )

    @property
    def saturated_samples(self) -> tuple[SaturatedSample, ...]:
        """All saturated samples of all channels, in channel order."""
        return tuple(
            sample for report in self.channels for sample in report.saturated_samples
        )

    @property
    def saturated_samples_by_channel(
        self,
    ) -> dict[str, tuple[SaturatedSample, ...]]:
        """Saturated samples grouped by channel label."""
        return {
            report.label: report.saturated_samples
            for report in self.channels
            if report.saturated_samples
        }

    @property
    def all_satisfied(self) -> bool:
        """Whether every channel meets its allowed error."""
        return all(report.satisfied for report in self.channels)

    @property
    def num_saturated_samples(self) -> int:
        """Total number of saturated samples."""
        return sum(report.num_saturated_samples for report in self.channels)

    @property
    def num_unsatisfiable_channels(self) -> int:
        """Number of channels whose allowed error could not be met."""
        return len(self.unsatisfiable_channels)

    @property
    def error_bound(self) -> float:
        """Largest per-channel error bound over the whole recording."""
        return max(
            (report.error_bound for report in self.channels),
            default=0.0,
        )

    def get_channel(self, key: int | str) -> ChannelConversionReport:
        """Get a channel report by index or label."""
        if isinstance(key, int):
            return self.channels[key]
        matches = [report for report in self.channels if report.label == key]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous channel label {key!r}")
        raise KeyError(f"No channel with label {key!r}")

    def __getitem__(self, key: int | str) -> ChannelConversionReport:
        return self.get_channel(key)

    def __len__(self) -> int:
        return len(self.channels)

    def __iter__(self) -> Any:
        return iter(self.channels)


class ConversionResult(Edf):
    """EDF produced by a BDF conversion, including its report.

    The object is a regular :class:`Edf` and can be used interchangeably.  It
    also supports unpacking as ``edf, report = result``.
    """

    report: BdfToEdfReport

    @property
    def edf(self) -> ConversionResult:
        """The converted EDF (i.e., this object)."""
        return self

    def __iter__(self) -> Any:
        yield self
        yield self.report


BdfToEdfResult = ConversionResult
BdfConversionReport = BdfToEdfReport
ConversionReport = BdfToEdfReport
ChannelReport = ChannelConversionReport


def _fits_header(value: float) -> bool:
    try:
        encode_float(value)
    except (OverflowError, ValueError):
        return False
    return True


def _is_fixed_point(value: float, rounding: Any) -> bool:
    return (
        _fits_header(value) and _round_float_to_8_characters(value, rounding) == value
    )


def _header_float(value: float, *, up: bool) -> float:
    """Nearest header-representable float in the given direction.

    `encode_float` uses ``str(value)``, so some values rounded by
    ``_round_float_to_8_characters`` do not fit into the eight header
    characters (e.g. ``-2.001e-05``).  This helper finds the nearest value in
    the requested direction that both fits and is a fixed point of the
    rounding applied again by `EdfSignal`.
    """
    rounding = math.ceil if up else math.floor
    if _is_fixed_point(value, rounding):
        return value

    decimal_value = Decimal(repr(value))
    exponent = decimal_value.adjusted()
    for digits in range(8, 0, -1):
        quantum = Decimal(1).scaleb(exponent - digits + 1)
        base = decimal_value.quantize(
            quantum,
            rounding=ROUND_CEILING if up else ROUND_FLOOR,
        )
        for step in range(201):
            candidate = float(base + step * quantum if up else base - step * quantum)
            if _is_fixed_point(candidate, rounding) and (
                (candidate >= value) if up else (candidate <= value)
            ):
                return candidate
    raise ValueError(  # pragma: no cover
        f"Value {value} cannot be represented in an EDF header field"
    )


def _header_range(
    desired_min: float,
    desired_max: float,
) -> tuple[float, float] | None:
    if not math.isfinite(desired_min) or not math.isfinite(desired_max):
        return None
    if desired_max <= desired_min:
        return None
    try:
        minimum = _header_float(desired_min, up=False)
        maximum = _header_float(desired_max, up=True)
    except (OverflowError, ValueError):
        return None
    if maximum <= minimum:  # pragma: no cover - rounding is monotonic
        return None
    return minimum, maximum


def _narrow_range(center: float) -> tuple[float, float]:
    width = max(abs(center) * 1e-6, 1e-6)
    for _ in range(200):
        for desired_min, desired_max in (
            (center - width / 2, center + width / 2),
            (center, center + width),
            (center - width, center),
        ):
            encoded = _header_range(desired_min, desired_max)
            if encoded is not None:
                return encoded
        width *= 2
        if not math.isfinite(width):
            break
    raise ValueError(
        f"Physical range around {center} cannot be represented in an EDF header"
    )


def _widest_centered_range(
    center: float,
    max_span: float,
) -> tuple[float, float] | None:
    """Widest representable range centered on `center` within `max_span`."""
    low = 0.0
    high = max_span
    best: tuple[float, float] | None = None
    for _ in range(200):
        width = (low + high) / 2
        encoded = _header_range(center - width / 2, center + width / 2)
        if encoded is not None and encoded[1] - encoded[0] <= max_span:
            best = encoded
            low = width
        else:
            high = width
    return best


def _choose_physical_range(
    data_min: float,
    data_max: float,
    max_span: float,
) -> tuple[tuple[float, float], bool]:
    """Return a header-representable physical range and whether it covers data."""
    data_span = data_max - data_min
    center = data_min + data_span / 2

    if data_span == 0:
        return _narrow_range(data_min), True

    # Flooring/ceiling the data limits to the eight header characters gives
    # the smallest representable range that contains all samples.
    tight = _header_range(data_min, data_max)
    if tight is not None and tight[1] - tight[0] <= max_span:
        return tight, True

    # Header resolution can make even the smallest covering range wider than
    # the tolerance.  Keep that range (no clipping) if the raw data span
    # itself was feasible; the channel will be reported as unsatisfiable.
    if tight is not None and data_span <= max_span:
        return tight, False

    if max_span > 0:
        encoded = _widest_centered_range(center, max_span)
        if encoded is not None:
            return encoded, False

    return _narrow_range(center), False


def _resolve_tolerances(
    bdf: Bdf,
    tolerances: Any,
    default_tolerance: float | None,
) -> list[float]:
    signals = bdf.signals
    if tolerances is None:
        if default_tolerance is None:
            raise ValueError(
                "No allowed error was specified; pass 'tolerances' "
                "(or 'default_tolerance')"
            )
        return [float(default_tolerance)] * len(signals)

    if isinstance(tolerances, Mapping):
        resolved: list[float] = []
        for index, signal in enumerate(signals):
            value = tolerances.get(index, tolerances.get(signal.label))
            if value is None:
                value = default_tolerance
            if value is None:
                raise ValueError(
                    f"No allowed error specified for channel {index} ({signal.label!r})"
                )
            resolved.append(float(value))
        return resolved

    if isinstance(tolerances, Real):
        return [float(tolerances)] * len(signals)

    values = [float(value) for value in tolerances]
    if len(values) != len(signals):
        raise ValueError(f"Expected {len(signals)} tolerances, got {len(values)}")
    return values


def _validate_tolerances(tolerances: Sequence[float]) -> None:
    for value in tolerances:
        if not math.isfinite(value):
            raise ValueError("Allowed errors must be finite")
        if value < 0:
            raise ValueError("Allowed errors must be non-negative")


def _validate_digital_range(
    digital_range: tuple[int, int] | None,
) -> tuple[int, int]:
    if digital_range is None:
        return _EDF_DEFAULT_RANGE
    digital_min, digital_max = digital_range
    if digital_min >= digital_max:
        raise ValueError(
            f"Digital minimum ({digital_min}) must be smaller than digital "
            f"maximum ({digital_max})"
        )
    if digital_min < _EDF_DEFAULT_RANGE[0] or digital_max > _EDF_DEFAULT_RANGE[1]:
        raise ValueError(
            f"Digital range {digital_range} out of supported range "
            f"{_EDF_DEFAULT_RANGE} for EDF"
        )
    return int(digital_min), int(digital_max)


def _convert_signal(
    signal: Any,
    index: int,
    allowed_error: float,
    digital_range: tuple[int, int],
) -> tuple[EdfSignal, ChannelConversionReport]:
    digital_min, digital_max = digital_range
    digital_spread = digital_max - digital_min
    data = np.asarray(signal.data, dtype=np.float64)
    if data.size == 0:
        raise ValueError(f"Signal {index} ({signal.label!r}) contains no samples")
    if not np.all(np.isfinite(data)):
        raise ValueError(
            f"Signal {index} ({signal.label!r}) contains non-finite values"
        )

    data_min = float(np.min(data))
    data_max = float(np.max(data))
    data_span = data_max - data_min

    max_span = 2 * allowed_error * digital_spread
    # Safety margin for header rounding and floating point arithmetic.
    max_span = math.nextafter(max_span, 0.0)

    physical_range, covers_all = _choose_physical_range(data_min, data_max, max_span)
    minimum, maximum = physical_range
    gain = (maximum - minimum) / digital_spread
    offset = maximum / gain - digital_max

    unclipped = np.round(data / gain - offset)
    clipped = np.clip(unclipped, digital_min, digital_max)
    saturated_mask = (unclipped < digital_min) | (unclipped > digital_max)
    digital = clipped.astype(np.int16)

    converted = EdfSignal.from_digital(
        digital,
        sampling_frequency=signal.sampling_frequency,
        label=signal.label,
        transducer_type=signal.transducer_type,
        physical_dimension=signal.physical_dimension,
        physical_range=(minimum, maximum),
        digital_range=digital_range,
        prefiltering=signal.prefiltering,
    )

    decoded = np.asarray(converted.data, dtype=np.float64)
    errors = np.abs(decoded - data)
    max_abs_error = float(np.max(errors))

    saturated_samples = tuple(
        SaturatedSample(
            channel=signal.label,
            index=int(sample_index),
            value=float(data[sample_index]),
            decoded_value=float(decoded[sample_index]),
        )
        for sample_index in np.flatnonzero(saturated_mask)
    )

    reason = None
    if max_abs_error > allowed_error:
        details = []
        if not covers_all:
            details.append(
                f"required physical span {data_span:g} exceeds the maximum "
                f"span {max_span:g} implied by allowed error {allowed_error:g}"
            )
        details.append(
            f"realized maximum absolute error is {max_abs_error:g} for "
            f"allowed error {allowed_error:g}"
        )
        reason = "; ".join(details)

    report = ChannelConversionReport(
        index=index,
        label=signal.label,
        allowed_error=allowed_error,
        physical_range=_FloatRange(minimum, maximum),
        digital_range=_IntRange(digital_min, digital_max),
        quantization_step=gain,
        max_abs_error=max_abs_error,
        saturated_samples=saturated_samples,
        num_samples=int(data.size),
        worst_sample_index=int(np.argmax(errors)),
        reason=reason,
    )
    return converted, report


def convert_signal_to_edf(
    signal: Any,
    allowed_error: Any,
    *,
    digital_range: tuple[int, int] | None = None,
) -> tuple[EdfSignal, ChannelConversionReport]:
    """Convert a single BDF signal to a 16-bit EDF signal.

    Parameters
    ----------
    signal : BdfSignal
        BDF signal to convert.
    allowed_error : float
        Maximum absolute physical error allowed for this signal.
    digital_range : tuple[int, int] | None, default: None
        Digital range used for the EDF signal.  Defaults to the full 16-bit
        range ``(-32768, 32767)``.

    Returns
    -------
    tuple[EdfSignal, ChannelConversionReport]
        Converted signal and its report.
    """
    if not isinstance(allowed_error, Real):
        raise TypeError("'allowed_error' must be a real number")
    value = float(allowed_error)
    _validate_tolerances([value])
    return _convert_signal(
        signal,
        0,
        value,
        _validate_digital_range(digital_range),
    )


def convert_bdf_to_edf(
    bdf: Bdf,
    tolerances: Any = None,
    *,
    tolerance: Any = None,
    allowed_error: Any = None,
    channel_tolerances: Any = None,
    default_tolerance: float | None = None,
    digital_range: tuple[int, int] | None = None,
    data_record_duration: float | None = None,
) -> ConversionResult:
    """Convert a BDF recording to a 16-bit EDF recording.

    The conversion works in the physical domain and never truncates the high
    bits of the 24-bit samples.  For each channel, the allowed absolute
    physical error determines the widest physical range that can be
    represented with the 16-bit digital range; that range is written to the
    header and all samples are requantized.

    The report covers the whole recording: `max_abs_error` is measured over
    every sample after decoding the EDF values back to physical values.
    Channels whose physical amplitude cannot be covered within the tolerance
    are listed with a reason, and every clipped sample is reported with its
    index and realized error.

    Parameters
    ----------
    bdf : Bdf
        BDF recording to convert.
    tolerances : float | Mapping[int | str, float] | Sequence[float] | None
        Allowed absolute physical error.  Either one value for all channels,
        one value per channel in signal order, or a mapping from signal index
        or label to a value.
    tolerance : float | Mapping[int | str, float] | Sequence[float] | None
        Alias of `tolerances`.
    allowed_error : float | Mapping[int | str, float] | Sequence[float] | None
        Alias of `tolerances`.
    channel_tolerances : float | Mapping[int | str, float] | Sequence[float] | None
        Alias of `tolerances`.
    default_tolerance : float | None, default: None
        Fallback for channels missing from a mapping.
    digital_range : tuple[int, int] | None, default: None
        Digital range used for the EDF signals.  Defaults to the full 16-bit
        range ``(-32768, 32767)``.
    data_record_duration : float | None, default: None
        Data record duration of the EDF.  Defaults to the BDF value.

    Returns
    -------
    ConversionResult
        Converted EDF (a subclass of :class:`Edf`) with a `report` attribute.
        It can also be unpacked as ``edf, report = result``.
    """
    specified = [
        value
        for value in (tolerances, tolerance, allowed_error, channel_tolerances)
        if value is not None
    ]
    if len(specified) > 1:
        raise ValueError(
            "Specify only one of 'tolerances', 'tolerance', 'allowed_error', "
            "or 'channel_tolerances'"
        )
    specification = specified[0] if specified else None
    digital_range = _validate_digital_range(digital_range)
    resolved = _resolve_tolerances(bdf, specification, default_tolerance)
    _validate_tolerances(resolved)

    signals: list[EdfSignal] = []
    reports: list[ChannelConversionReport] = []
    for index, signal in enumerate(bdf.signals):
        converted, report = _convert_signal(
            signal,
            index,
            resolved[index],
            digital_range,
        )
        signals.append(converted)
        reports.append(report)

    has_annotations = any(signal._is_annotation_signal for signal in bdf._signals)
    annotations = bdf.annotations if has_annotations else None
    duration = (
        bdf.data_record_duration
        if data_record_duration is None
        else data_record_duration
    )
    edf = ConversionResult(
        signals=signals,
        patient=bdf.patient,
        recording=bdf.recording,
        starttime=bdf.starttime,
        data_record_duration=None if not signals else duration,
        annotations=annotations,
    )
    edf.report = BdfToEdfReport(ChannelReportCollection(reports))
    return edf


def bdf_to_edf(
    bdf: Bdf,
    tolerances: Any = None,
    **kwargs: Any,
) -> ConversionResult:
    """Alias of :func:`convert_bdf_to_edf`."""
    return convert_bdf_to_edf(bdf, tolerances, **kwargs)


def to_edf(
    bdf: Bdf,
    tolerances: Any = None,
    **kwargs: Any,
) -> ConversionResult:
    """Alias of :func:`convert_bdf_to_edf`."""
    return convert_bdf_to_edf(bdf, tolerances, **kwargs)
