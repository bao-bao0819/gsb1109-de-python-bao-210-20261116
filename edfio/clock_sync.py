"""
Align recordings from multiple devices on a unified experiment timeline.

Each device is described by a :class:`ClockCorrectionTable`, a piecewise mapping
from the device clock to the experiment clock. Querying an experiment-time window
yields the corresponding raw sample ranges per signal, without resampling signals
of different sampling frequencies onto a common time axis.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from edfio.edf import Bdf, Edf
from edfio.edf_signal import BdfSignal, EdfSignal

_Certainty = Literal["definite", "possible"]


def _check_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")


def _ceil_to_int(value: float) -> int:
    nearest = round(value)
    if math.isclose(value, nearest, rel_tol=1e-9, abs_tol=1e-9):
        return nearest
    return math.ceil(value)


@dataclass(frozen=True)
class ClockCorrectionSegment:
    """
    One segment of a piecewise clock correction table.

    Within the segment, experiment time is obtained from device time as
    ``experiment_time = device_time + offset``, where the true offset may deviate
    from the nominal `offset` by at most `offset_error`.

    Parameters
    ----------
    device_start : float
        Start of the segment on the device clock in seconds since recording start
        (inclusive).
    device_stop : float
        End of the segment on the device clock in seconds since recording start
        (exclusive).
    offset : float
        Nominal offset in seconds added to the device clock to obtain experiment
        time.
    offset_error : float, default: 0.0
        Maximum absolute error of `offset` in seconds.
    """

    device_start: float
    device_stop: float
    offset: float
    offset_error: float = 0.0

    def __post_init__(self) -> None:
        for name in ("device_start", "device_stop", "offset", "offset_error"):
            _check_finite(getattr(self, name), name)
        if self.device_stop <= self.device_start:
            raise ValueError(
                f"device_stop ({self.device_stop}) must be after device_start ({self.device_start})"
            )
        if self.offset_error < 0:
            raise ValueError(
                f"offset_error must be non-negative, got {self.offset_error}"
            )

    @property
    def experiment_start(self) -> float:
        """Nominal experiment time corresponding to `device_start`."""
        return self.device_start + self.offset

    @property
    def experiment_stop(self) -> float:
        """Nominal experiment time corresponding to `device_stop`."""
        return self.device_stop + self.offset


@dataclass(frozen=True)
class DeviceTimeWindow:
    """
    A device-time interval corresponding to (part of) an experiment-time query.

    Attributes
    ----------
    segment : ClockCorrectionSegment
        The correction segment used to map this interval.
    device_start : float
        Start of the interval on the device clock in seconds (inclusive).
    device_stop : float
        End of the interval on the device clock in seconds (exclusive).
    experiment_start : float
        Earliest experiment time samples in this interval may correspond to,
        clipped to the query window.
    experiment_stop : float
        Latest experiment time samples in this interval may correspond to,
        clipped to the query window.
    certainty : {"definite", "possible"}
        `"definite"` if samples in this interval fall into the query window for
        every offset within the segment's error range, `"possible"` if they fall
        into the query window only for some offsets within the error range.
    """

    segment: ClockCorrectionSegment
    device_start: float
    device_stop: float
    experiment_start: float
    experiment_stop: float
    certainty: _Certainty


@dataclass(frozen=True)
class SampleRange:
    """
    A range of raw sample indices of one signal.

    Attributes
    ----------
    start : int
        Index of the first sample in the range (inclusive).
    stop : int
        Index of the last sample in the range (exclusive).
    window : DeviceTimeWindow
        The device-time window this sample range was derived from, including the
        correction segment used and the certainty classification.
    """

    start: int
    stop: int
    window: DeviceTimeWindow

    @property
    def certainty(self) -> _Certainty:
        """Certainty classification of the underlying device-time window."""
        return self.window.certainty

    def to_slice(self) -> slice:
        """Return a slice for indexing the signal's data array."""
        return slice(self.start, self.stop)


@dataclass(frozen=True)
class SignalSampleRanges:
    """
    Raw sample ranges of one signal for an experiment-time query.

    Attributes
    ----------
    label : str
        The signal's label.
    sampling_frequency : float
        The signal's own sampling frequency in Hz. Signals of different sampling
        frequencies are never resampled onto a common time axis.
    ranges : tuple[SampleRange, ...]
        The raw sample ranges, sorted by device time.
    """

    label: str
    sampling_frequency: float
    ranges: tuple[SampleRange, ...]


@dataclass(frozen=True)
class ExperimentWindowMapping:
    """
    Result of mapping an experiment-time query window to device-time intervals.

    Attributes
    ----------
    query_start : float
        Start of the query window on the experiment timeline in seconds.
    query_stop : float
        End of the query window on the experiment timeline in seconds.
    windows : tuple[DeviceTimeWindow, ...]
        All device-time intervals that definitely or possibly contain samples
        falling into the query window, sorted by device time. If several
        correction segments map into the query window (e.g. at overlapping
        segment switches), all of them are kept.
    uncovered : tuple[tuple[float, float], ...]
        Sub-intervals of the query window not covered by any correction segment,
        even accounting for offset errors. No zero offset is assumed for these
        gaps.
    """

    query_start: float
    query_stop: float
    windows: tuple[DeviceTimeWindow, ...]
    uncovered: tuple[tuple[float, float], ...]

    def get_sample_ranges(
        self, signal: EdfSignal | BdfSignal
    ) -> tuple[SampleRange, ...]:
        """
        Determine the raw sample index ranges of a single signal for this mapping.

        The signal is treated at its own sampling frequency; no resampling or
        interpolation is performed. A sample is included if its acquisition
        instant (sample index divided by the sampling frequency) lies within the
        corresponding device-time window. Ranges are clipped to the actual number
        of samples in the signal, and empty ranges are omitted.

        Parameters
        ----------
        signal : EdfSignal | BdfSignal
            The signal to determine sample ranges for.

        Returns
        -------
        tuple[SampleRange, ...]
            The raw sample ranges, sorted by device time.
        """
        frequency = signal.sampling_frequency
        num_samples = signal._num_samples
        ranges = []
        for window in self.windows:
            start_index = min(
                max(_ceil_to_int(window.device_start * frequency), 0), num_samples
            )
            stop_index = min(
                max(_ceil_to_int(window.device_stop * frequency), 0), num_samples
            )
            if stop_index > start_index:
                ranges.append(SampleRange(start_index, stop_index, window))
        return tuple(ranges)

    def get_recording_sample_ranges(
        self, recording: Edf | Bdf
    ) -> tuple[SignalSampleRanges, ...]:
        """
        Determine the raw sample ranges of all ordinary signals of a recording.

        Each signal keeps its own sampling frequency; results are not interpolated
        onto a common array.

        Parameters
        ----------
        recording : Edf | Bdf
            The recording to determine sample ranges for.

        Returns
        -------
        tuple[SignalSampleRanges, ...]
            The sample ranges of all ordinary (non-annotation) signals, in the
            order of the signals in the recording.
        """
        return tuple(
            SignalSampleRanges(
                label=signal.label,
                sampling_frequency=signal.sampling_frequency,
                ranges=self.get_sample_ranges(signal),
            )
            for signal in recording.signals
        )


def _make_window(
    segment: ClockCorrectionSegment,
    device_start: float,
    device_stop: float,
    certainty: _Certainty,
    query_start: float,
    query_stop: float,
) -> DeviceTimeWindow:
    error = segment.offset_error
    return DeviceTimeWindow(
        segment=segment,
        device_start=device_start,
        device_stop=device_stop,
        experiment_start=min(
            max(device_start + segment.offset - error, query_start), query_stop
        ),
        experiment_stop=min(
            max(device_stop + segment.offset + error, query_start), query_stop
        ),
        certainty=certainty,
    )


def _map_segment(
    segment: ClockCorrectionSegment, query_start: float, query_stop: float
) -> list[DeviceTimeWindow]:
    offset = segment.offset
    error = segment.offset_error
    poss_start = max(query_start - offset - error, segment.device_start)
    poss_stop = min(query_stop - offset + error, segment.device_stop)
    if poss_start >= poss_stop:
        return []
    def_start = min(max(query_start - offset + error, poss_start), poss_stop)
    def_stop = max(min(query_stop - offset - error, poss_stop), poss_start)
    if def_start >= def_stop:
        return [
            _make_window(
                segment, poss_start, poss_stop, "possible", query_start, query_stop
            )
        ]
    windows = []
    if poss_start < def_start:
        windows.append(
            _make_window(
                segment, poss_start, def_start, "possible", query_start, query_stop
            )
        )
    windows.append(
        _make_window(segment, def_start, def_stop, "definite", query_start, query_stop)
    )
    if def_stop < poss_stop:
        windows.append(
            _make_window(
                segment, def_stop, poss_stop, "possible", query_start, query_stop
            )
        )
    return windows


def _compute_uncovered(
    segments: tuple[ClockCorrectionSegment, ...], query_start: float, query_stop: float
) -> tuple[tuple[float, float], ...]:
    covered = []
    for segment in segments:
        start = max(segment.experiment_start - segment.offset_error, query_start)
        stop = min(segment.experiment_stop + segment.offset_error, query_stop)
        if start < stop:
            covered.append((start, stop))
    covered.sort()
    uncovered = []
    cursor = query_start
    for start, stop in covered:
        if start > cursor:
            uncovered.append((cursor, start))
        cursor = max(cursor, stop)
    if cursor < query_stop:
        uncovered.append((cursor, query_stop))
    return tuple(uncovered)


class ClockCorrectionTable:
    """
    Piecewise clock correction table mapping a device clock to experiment time.

    Segments may overlap or leave gaps, both on the device clock and on the
    experiment timeline (e.g. when a device clock is reset during recording).
    Device-time ranges not covered by any segment have no correction basis and
    are never mapped assuming zero offset.

    Parameters
    ----------
    segments : Iterable[ClockCorrectionSegment]
        The correction segments. They are sorted by device time on construction.
    """

    def __init__(self, segments: Iterable[ClockCorrectionSegment]) -> None:
        segments = tuple(segments)
        for segment in segments:
            if not isinstance(segment, ClockCorrectionSegment):
                raise TypeError(
                    f"segments must be ClockCorrectionSegment objects, got {type(segment).__name__}"
                )
        self._segments = tuple(
            sorted(segments, key=lambda s: (s.device_start, s.device_stop))
        )

    def __repr__(self) -> str:
        return f"<ClockCorrectionTable {len(self._segments)} segments>"

    @property
    def segments(self) -> tuple[ClockCorrectionSegment, ...]:
        """The correction segments, sorted by device time."""
        return self._segments

    def map_experiment_window(
        self, query_start: float, query_stop: float
    ) -> ExperimentWindowMapping:
        """
        Map an experiment-time window back to device-time intervals.

        Parameters
        ----------
        query_start : float
            Start of the query window on the unified experiment timeline in
            seconds.
        query_stop : float
            End of the query window on the unified experiment timeline in seconds.

        Returns
        -------
        ExperimentWindowMapping
            All device-time intervals that definitely or possibly contain samples
            falling into the query window, together with the experiment-time
            sub-intervals lacking correction coverage.
        """
        _check_finite(query_start, "query_start")
        _check_finite(query_stop, "query_stop")
        if query_stop <= query_start:
            raise ValueError(
                f"query_stop ({query_stop}) must be after query_start ({query_start})"
            )
        windows: list[DeviceTimeWindow] = []
        for segment in self._segments:
            windows.extend(_map_segment(segment, query_start, query_stop))
        windows.sort(key=lambda w: (w.device_start, w.device_stop))
        return ExperimentWindowMapping(
            query_start=query_start,
            query_stop=query_stop,
            windows=tuple(windows),
            uncovered=_compute_uncovered(self._segments, query_start, query_stop),
        )
