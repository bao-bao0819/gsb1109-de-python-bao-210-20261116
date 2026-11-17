"""Derivation of differential channels from source signals.

A differential channel is computed as the physical difference of two source signals
(``positive - negative``). Samples are only emitted at real time points at which *every*
source has a usable sample: gaps, saturated samples, and explicitly marked bad segments
make an output sample unavailable. A measured value of exactly zero is valid data and is
never treated as bad on its own.
"""

from __future__ import annotations

import math
import weakref
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Union, cast

import numpy as np
import numpy.typing as npt

from edfio.edf_signal import BadSegment, BdfSignal, EdfSignal, _BaseSignal

_Signal = Union[EdfSignal, BdfSignal]
_EPS = 1e-9
_PAIR_LENGTH = 2


@dataclass(frozen=True)
class DerivedChannelSpec:
    """Definition of a differential channel as a pair of source labels.

    Parameters
    ----------
    label : str
        Label of the derived channel.
    positive : str
        Label of the source signal used as the positive input.
    negative : str
        Label of the source signal used as the negative (reference) input.
    physical_dimension : str | None, default: None
        Physical dimension of the derived channel. If `None`, the dimension shared by the
        source signals is used.
    """

    label: str
    positive: str
    negative: str
    physical_dimension: str | None = None


@dataclass
class DerivedChannel:
    """A differential channel computed on a common time grid.

    Attributes
    ----------
    label : str
        Label of the channel.
    sampling_frequency : float
        Sampling frequency in Hz. It is the least common multiple of the source sampling
        frequencies, so every output time coincides with a source sample time.
    data : npt.NDArray[np.float64]
        Physical values (``positive - negative``) with `numpy.nan` at every time point at
        which at least one source is unavailable.
    available_mask : npt.NDArray[np.bool_]
        `True` for samples for which every source was usable.
    sources : tuple[str, str]
        Labels of the positive and negative source signals, in that order.
    physical_dimension : str
        Physical dimension of the channel.
    source_available_masks : dict[str, npt.NDArray[np.bool_]]
        Per-source availability masks (`True` means usable), keeping the origin of every
        output sample traceable.
    source_sampling_frequencies : dict[str, float]
        Sampling frequency of each source, keyed by source label.
    bad_segment_mask : npt.NDArray[np.bool_]
        `True` where at least one source is covered by a marked bad segment.
    saturation_mask : npt.NDArray[np.bool_]
        `True` where at least one source sits at a digital rail (saturation).
    zero_mask : npt.NDArray[np.bool_]
        `True` for available samples whose derived value is exactly zero, i.e. genuine
        measured zeros rather than missing data.
    """

    label: str
    sampling_frequency: float
    data: npt.NDArray[np.float64]
    available_mask: npt.NDArray[np.bool_]
    sources: tuple[str, str]
    physical_dimension: str
    source_available_masks: dict[str, npt.NDArray[np.bool_]] = field(default_factory=dict)
    source_sampling_frequencies: dict[str, float] = field(default_factory=dict)
    bad_segment_mask: npt.NDArray[np.bool_] = field(
        default_factory=lambda: np.zeros(0, dtype=np.bool_)
    )
    saturation_mask: npt.NDArray[np.bool_] = field(
        default_factory=lambda: np.zeros(0, dtype=np.bool_)
    )
    zero_mask: npt.NDArray[np.bool_] = field(
        default_factory=lambda: np.zeros(0, dtype=np.bool_)
    )
    _signal_class: type[_BaseSignal[Any]] = EdfSignal

    @property
    def num_samples(self) -> int:
        """Number of samples on the common output time grid."""
        return len(self.data)

    def unavailable_intervals(self) -> tuple[tuple[float, float], ...]:
        """Return the half-open time intervals ``[start, stop)`` without usable samples."""
        return _mask_to_intervals(~self.available_mask, self.sampling_frequency)

    def available_intervals(self) -> tuple[tuple[float, float], ...]:
        """Return the half-open time intervals ``[start, stop)`` with usable samples."""
        return _mask_to_intervals(self.available_mask, self.sampling_frequency)

    def source_sample_index(self, source: str, sample_index: int) -> int:
        """Index of the source sample underlying a derived sample.

        Parameters
        ----------
        source : str
            Label of one of the two source signals.
        sample_index : int
            Index of a sample in the derived channel.

        Returns
        -------
        int
            Index of the corresponding sample in the source signal.
        """
        if source not in self.sources:
            raise ValueError(
                f"{source!r} is not a source of derived channel {self.label!r}, "
                f"sources are {self.sources}"
            )
        step = _alignment_period(
            Fraction(str(self.sampling_frequency)),
            Fraction(str(self.source_sampling_frequencies[source])),
        )
        return sample_index // step

    def to_signal(
        self,
        *,
        fill_value: float = 0.0,
        physical_range: tuple[float, float] | None = None,
    ) -> _Signal:
        """Materialize the channel as a writable EDF/BDF signal.

        Unavailable samples (stored as `numpy.nan`) are replaced with `fill_value`. Note
        that EDF/BDF signals cannot represent missing samples, so the availability
        information should be retained separately (see :attr:`available_mask`).

        Parameters
        ----------
        fill_value : float, default: `0.0`
            Physical value written at unavailable sample positions.
        physical_range : tuple[float, float] | None, default: None
            Physical range of the new signal. If `None`, it is determined from the data.
        """
        data = self.data.copy()
        data[~self.available_mask] = fill_value
        return cast(
            _Signal,
            self._signal_class(
                data=data,
                sampling_frequency=self.sampling_frequency,
                label=self.label,
                physical_dimension=self.physical_dimension,
                physical_range=physical_range,
            ),
        )


@dataclass(frozen=True)
class ChannelDerivationReport:
    """Summary of a single derived channel."""

    label: str
    sources: tuple[str, str]
    sampling_frequency: float
    num_samples: int
    num_available: int
    num_unavailable: int
    num_bad_segment: int
    num_saturation: int
    num_zero: int
    unavailable_intervals: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class DerivationReport:
    """Summary of all channels produced in one derivation run."""

    channels: tuple[ChannelDerivationReport, ...]


@dataclass(frozen=True)
class DerivationResult:
    """Outcome of :func:`derive_differential_channels`."""

    channels: tuple[DerivedChannel, ...]
    report: DerivationReport


@dataclass(frozen=True)
class UpdateReport:
    """Outcome of :func:`set_bad_segments`.

    Intervals are given per derived channel label, in seconds from the recording start.
    """

    signal_label: str
    bad_segments: tuple[BadSegment, ...]
    withdrawn_intervals: dict[str, tuple[tuple[float, float], ...]]
    restored_intervals: dict[str, tuple[tuple[float, float], ...]]
    value_changed_intervals: dict[str, tuple[tuple[float, float], ...]]
    recomputed_intervals: dict[str, tuple[tuple[float, float], ...]]
    unaffected_channels: tuple[str, ...]


@dataclass
class _DerivedEntry:
    spec: DerivedChannelSpec
    channel: DerivedChannel
    source_signals: tuple[_Signal, _Signal]
    detect_saturation: bool


class DerivationRegistry:
    """Keeps the derived channels of one recording and their source provenance."""

    def __init__(self) -> None:
        self._entries: dict[str, _DerivedEntry] = {}

    @property
    def labels(self) -> tuple[str, ...]:
        """Labels of all registered derived channels."""
        return tuple(self._entries)

    def get_channel(self, label: str) -> DerivedChannel:
        """Return the channel registered under `label`."""
        try:
            return self._entries[label].channel
        except KeyError:
            raise ValueError(
                f"No derived channel with label {label!r}, available: {self.labels}"
            ) from None

    def entries_using(self, signal: _BaseSignal[Any]) -> tuple[_DerivedEntry, ...]:
        """Return all derivation entries whose sources include `signal`."""
        return tuple(
            entry for entry in self._entries.values() if signal in entry.source_signals
        )


_registries: weakref.WeakKeyDictionary[Any, DerivationRegistry] = (
    weakref.WeakKeyDictionary()
)


def _get_registry(edf: Any) -> DerivationRegistry | None:
    return _registries.get(edf)


def _common_sampling_frequency(frequencies: Sequence[float]) -> Fraction:
    fractions = [Fraction(str(fs)).limit_denominator(99_999_999) for fs in frequencies]
    numerator = math.lcm(*(fs.numerator for fs in fractions))
    denominator = math.gcd(*(fs.denominator for fs in fractions))
    return Fraction(numerator, denominator)


def _alignment_period(out_fs: Fraction, source_fs: Fraction) -> int:
    """Return the number of output samples between two consecutive source samples.

    The common output frequency is an integer multiple of every source frequency.
    """
    return (out_fs / source_fs).numerator


def _ceil_with_tolerance(value: float) -> int:
    rounded = round(value)
    if math.isclose(value, rounded, abs_tol=_EPS):
        return rounded
    return math.ceil(value)


def normalize_bad_segments(
    segments: Iterable[BadSegment | tuple[float, float] | tuple[float, float, str]],
) -> tuple[BadSegment, ...]:
    """Validate, sort, and merge bad segments, returning disjoint half-open intervals."""
    normalized: list[BadSegment] = []
    for segment in segments:
        if isinstance(segment, BadSegment):
            bad_segment = segment
        elif len(segment) == _PAIR_LENGTH:
            bad_segment = BadSegment(segment[0], segment[1])
        else:
            triple = cast(tuple[float, float, str], segment)
            bad_segment = BadSegment(triple[0], triple[1], triple[2])
        if bad_segment.start_time < 0:
            raise ValueError(
                f"Bad segment start_time must be non-negative, got {bad_segment.start_time}"
            )
        if bad_segment.start_time == bad_segment.stop_time:
            continue
        normalized.append(bad_segment)
    normalized.sort(key=lambda seg: (seg.start_time, seg.stop_time))
    merged: list[BadSegment] = []
    for segment in normalized:
        if merged and segment.start_time <= merged[-1].stop_time:
            previous = merged.pop()
            reason = _merge_reasons(previous.reason, segment.reason)
            merged.append(
                BadSegment(
                    previous.start_time,
                    max(previous.stop_time, segment.stop_time),
                    reason,
                )
            )
        else:
            merged.append(segment)
    return tuple(merged)


def _merge_reasons(left: str, right: str) -> str:
    if not left:
        return right
    if not right or left == right:
        return left
    return f"{left}; {right}"


def _bad_segment_mask(
    signal: _BaseSignal[Any],
    num_samples: int,
    out_fs: Fraction,
) -> npt.NDArray[np.bool_]:
    mask: npt.NDArray[np.bool_] = np.zeros(num_samples, dtype=np.bool_)
    for segment in signal.bad_segments:
        first_index = _ceil_with_tolerance(float(segment.start_time * out_fs))
        last_index_exclusive = _ceil_with_tolerance(float(segment.stop_time * out_fs))
        mask[max(0, first_index) : min(num_samples, last_index_exclusive)] = True
    return mask


def _saturation_mask(
    signal: _BaseSignal[Any],
    source_indices: npt.NDArray[np.int_],
) -> npt.NDArray[np.bool_]:
    digital = signal.digital[source_indices]
    return (digital == signal.digital_min) | (digital == signal.digital_max)


def _mask_to_intervals(
    mask: npt.NDArray[np.bool_], sampling_frequency: float
) -> tuple[tuple[float, float], ...]:
    if not mask.any():
        return ()
    padded = np.concatenate(([False], mask, [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    stops = np.flatnonzero(padded[:-1] & ~padded[1:])
    return tuple(
        (float(start / sampling_frequency), float(stop / sampling_frequency))
        for start, stop in zip(starts, stops)
    )


def _resolve_physical_dimension(
    positive: _BaseSignal[Any], negative: _BaseSignal[Any]
) -> str:
    if positive.physical_dimension != negative.physical_dimension:
        raise ValueError(
            "Cannot derive a differential channel from signals with different physical "
            f"dimensions: {positive.physical_dimension!r} vs {negative.physical_dimension!r}"
        )
    return positive.physical_dimension


def compute_differential_channel(
    positive: _Signal,
    negative: _Signal,
    *,
    label: str = "",
    physical_dimension: str | None = None,
    detect_saturation: bool = True,
) -> DerivedChannel:
    """Compute one differential channel (``positive - negative``).

    The output time grid is the least common multiple of the source sampling frequencies.
    A derived sample is only available when both sources have a sample at the same real
    time and neither source is bad (marked segment or, optionally, a saturated sample) at
    that time. Physical values are used, so each source's calibration (unit conversion)
    is applied before subtraction.

    Parameters
    ----------
    positive : EdfSignal | BdfSignal
        Source signal used as the positive input.
    negative : EdfSignal | BdfSignal
        Source signal used as the negative (reference) input.
    label : str, default: `""`
        Label of the derived channel.
    physical_dimension : str | None, default: None
        Physical dimension of the derived channel. If `None`, the dimension shared by the
        source signals is used.
    detect_saturation : bool, default: True
        If `True`, samples at either digital rail of a source are treated as
        unavailable.

    Returns
    -------
    DerivedChannel
        The derived channel, including availability masks and provenance.
    """
    if positive is negative:
        raise ValueError("Positive and negative source signals must be different signals")
    if type(positive) is not type(negative):
        raise ValueError(
            f"Source signals must have the same format, got {type(positive).__name__} "
            f"and {type(negative).__name__}"
        )
    dimension = _resolve_physical_dimension(positive, negative)
    if physical_dimension is not None:
        dimension = physical_dimension

    out_fs = _common_sampling_frequency(
        [positive.sampling_frequency, negative.sampling_frequency]
    )
    positive_period = _alignment_period(
        out_fs, Fraction(str(positive.sampling_frequency))
    )
    negative_period = _alignment_period(
        out_fs, Fraction(str(negative.sampling_frequency))
    )
    overlap_duration = min(
        len(positive.digital) / positive.sampling_frequency,
        len(negative.digital) / negative.sampling_frequency,
    )
    num_samples = _ceil_with_tolerance(float(overlap_duration * out_fs))
    indices = np.arange(num_samples)

    periods = {positive: positive_period, negative: negative_period}
    aligned = {
        positive: indices % positive_period == 0,
        negative: indices % negative_period == 0,
    }
    source_indices = {
        signal: indices // period for signal, period in periods.items()
    }
    source_masks: dict[str, npt.NDArray[np.bool_]] = {}
    bad_segment_union: npt.NDArray[np.bool_] = np.zeros(
        num_samples, dtype=np.bool_
    )
    saturation_union: npt.NDArray[np.bool_] = np.zeros(
        num_samples, dtype=np.bool_
    )
    for signal in (positive, negative):
        bad_mask = _bad_segment_mask(signal, num_samples, out_fs)
        bad_segment_union |= bad_mask
        if detect_saturation:
            saturation = aligned[signal] & _saturation_mask(
                signal, source_indices[signal]
            )
        else:
            saturation = np.zeros(num_samples, dtype=np.bool_)
        saturation_union |= saturation
        source_masks[signal.label] = (
            aligned[signal] & ~bad_mask & ~saturation
        )

    available = np.logical_and(
        source_masks[positive.label], source_masks[negative.label]
    )
    positive_data = positive.data[source_indices[positive]]
    negative_data = negative.data[source_indices[negative]]
    data: npt.NDArray[np.float64] = np.full(
        num_samples, np.nan, dtype=np.float64
    )
    data[available] = positive_data[available] - negative_data[available]
    zero_mask = available & (data == 0.0)

    return DerivedChannel(
        label=label,
        sampling_frequency=float(out_fs),
        data=data,
        available_mask=available,
        sources=(positive.label, negative.label),
        physical_dimension=dimension,
        source_available_masks=source_masks,
        source_sampling_frequencies={
            positive.label: positive.sampling_frequency,
            negative.label: negative.sampling_frequency,
        },
        bad_segment_mask=bad_segment_union,
        saturation_mask=saturation_union,
        zero_mask=zero_mask,
        _signal_class=type(positive),
    )


def _build_report(channel: DerivedChannel) -> ChannelDerivationReport:
    return ChannelDerivationReport(
        label=channel.label,
        sources=channel.sources,
        sampling_frequency=channel.sampling_frequency,
        num_samples=channel.num_samples,
        num_available=int(channel.available_mask.sum()),
        num_unavailable=int((~channel.available_mask).sum()),
        num_bad_segment=int(channel.bad_segment_mask.sum()),
        num_saturation=int(channel.saturation_mask.sum()),
        num_zero=int(channel.zero_mask.sum()),
        unavailable_intervals=channel.unavailable_intervals(),
    )


def derive_differential_channels(
    edf: Any,
    specs: Iterable[DerivedChannelSpec],
    *,
    detect_saturation: bool = True,
) -> DerivationResult:
    """Derive differential channels from the signals of a recording.

    The derived channels are registered with the recording, so later calls to
    :func:`set_bad_segments` automatically withdraw or recompute the affected intervals.
    A shared reference signal that is marked bad therefore invalidates every channel that
    uses it.

    Parameters
    ----------
    edf : Edf | Bdf
        The recording containing the source signals.
    specs : Iterable[DerivedChannelSpec]
        Definitions of the channels to derive.
    detect_saturation : bool, default: True
        Whether samples at the digital rails of a source are treated as unavailable.

    Returns
    -------
    DerivationResult
        The derived channels together with a summary report.
    """
    specs = tuple(specs)
    labels = edf.labels
    duplicate_labels = {label for label in labels if labels.count(label) > 1}
    if duplicate_labels:
        raise ValueError(
            "Source labels must be unique to derive differential channels, "
            f"ambiguous labels: {tuple(sorted(duplicate_labels))}"
        )
    signals_by_label = {signal.label: signal for signal in edf.signals}
    registry = _registries.setdefault(edf, DerivationRegistry())
    channels: list[DerivedChannel] = []
    reports: list[ChannelDerivationReport] = []
    for spec in specs:
        if spec.label in registry.labels:
            raise ValueError(f"Derived channel with label {spec.label!r} already exists")
        for source_label in (spec.positive, spec.negative):
            if source_label not in signals_by_label:
                raise ValueError(
                    f"Unknown source label {source_label!r} for derived channel "
                    f"{spec.label!r}, possible options: {labels}"
                )
        if spec.positive == spec.negative:
            raise ValueError(
                f"Derived channel {spec.label!r} must use two different source signals"
            )
        positive = signals_by_label[spec.positive]
        negative = signals_by_label[spec.negative]
        channel = compute_differential_channel(
            positive,
            negative,
            label=spec.label,
            physical_dimension=spec.physical_dimension,
            detect_saturation=detect_saturation,
        )
        registry._entries[spec.label] = _DerivedEntry(
            spec=spec,
            channel=channel,
            source_signals=(positive, negative),
            detect_saturation=detect_saturation,
        )
        channels.append(channel)
        reports.append(_build_report(channel))
    return DerivationResult(
        channels=tuple(channels), report=DerivationReport(tuple(reports))
    )


def get_derived_channel(edf: Any, label: str) -> DerivedChannel:
    """Retrieve a previously derived channel from a recording by label."""
    registry = _get_registry(edf)
    if registry is None:
        raise ValueError("No derived channels registered for this recording")
    return registry.get_channel(label)


def set_bad_segments(
    signal: _BaseSignal[Any],
    segments: Iterable[BadSegment | tuple[float, float] | tuple[float, float, str]],
    *,
    recording: Any | None = None,
) -> UpdateReport:
    """Replace the bad segments of a source signal and update derived channels.

    Every derived channel that uses the signal (including channels using it only as a
    shared reference) is recomputed. The returned report lists which intervals were
    withdrawn (previously usable, now unavailable), restored (previously unavailable, now
    usable), or otherwise recomputed, together with the unaffected channels.

    Parameters
    ----------
    signal : EdfSignal | BdfSignal
        The source signal to update.
    segments : Iterable[BadSegment | tuple]
        Bad segments as :class:`BadSegment` objects or ``(start, stop)`` /
        ``(start, stop, reason)`` tuples, in seconds from the recording start.
        Overlapping intervals are merged.
    recording : Edf | Bdf | None, default: None
        The recording whose derived channels should be updated. If `None`, every known
        recording containing the signal is updated.

    Returns
    -------
    UpdateReport
        The withdrawn, restored, and recomputed intervals per derived channel.
    """
    merged = normalize_bad_segments(segments)
    signal._bad_segments = merged

    withdrawn: dict[str, tuple[tuple[float, float], ...]] = {}
    restored: dict[str, tuple[tuple[float, float], ...]] = {}
    value_changed: dict[str, tuple[tuple[float, float], ...]] = {}
    recomputed: dict[str, tuple[tuple[float, float], ...]] = {}
    unaffected: list[str] = []
    for _edf, registry in _registries_for(signal, recording):
        affected = registry.entries_using(signal)
        unaffected.extend(
            entry.channel.label
            for entry in registry._entries.values()
            if entry not in affected
        )
        for entry in affected:
            old_channel = entry.channel
            new_channel = compute_differential_channel(
                entry.source_signals[0],
                entry.source_signals[1],
                label=entry.spec.label,
                physical_dimension=entry.spec.physical_dimension,
                detect_saturation=entry.detect_saturation,
            )
            entry.channel = new_channel
            withdrawn[entry.spec.label] = _mask_to_intervals(
                old_channel.available_mask & ~new_channel.available_mask,
                old_channel.sampling_frequency,
            )
            restored[entry.spec.label] = _mask_to_intervals(
                new_channel.available_mask & ~old_channel.available_mask,
                old_channel.sampling_frequency,
            )
            common_available = (
                old_channel.available_mask & new_channel.available_mask
            )
            changed = common_available & ~_values_equal(
                old_channel.data, new_channel.data
            )
            value_changed[entry.spec.label] = _mask_to_intervals(
                changed, old_channel.sampling_frequency
            )
            recomputed[entry.spec.label] = _mask_to_intervals(
                (old_channel.available_mask != new_channel.available_mask) | changed,
                old_channel.sampling_frequency,
            )
    return UpdateReport(
        signal_label=signal.label,
        bad_segments=merged,
        withdrawn_intervals=withdrawn,
        restored_intervals=restored,
        value_changed_intervals=value_changed,
        recomputed_intervals=recomputed,
        unaffected_channels=tuple(unaffected),
    )


def _registries_for(
    signal: _BaseSignal[Any], recording: Any | None
) -> tuple[tuple[Any, DerivationRegistry], ...]:
    if recording is not None:
        registry = _get_registry(recording)
        if registry is None:
            return ()
        return ((recording, registry),)
    return tuple(
        (edf, registry)
        for edf, registry in _registries.items()
        if any(signal is source for source in edf.signals)
    )


def _values_equal(
    left: npt.NDArray[np.float64], right: npt.NDArray[np.float64]
) -> npt.NDArray[np.bool_]:
    both_nan = np.isnan(left) & np.isnan(right)
    return np.isclose(left, right, rtol=0.0, atol=1e-12) | both_nan
