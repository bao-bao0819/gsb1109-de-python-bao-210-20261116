from __future__ import annotations

import decimal
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from edfio.edf_annotations import _get_data_record_onset


class EdfInterval(NamedTuple):
    """A time interval of an EDF recording, given in seconds from recording start.

    Intervals are half-open, i.e. they cover the times ``start <= t < stop``, as is
    customary for EDF sample and record times.

    Parameters
    ----------
    start : float
        Start of the interval in seconds from the recording start.
    stop : float
        End of the interval in seconds from the recording start.
    """

    start: float
    stop: float

    @property
    def duration(self) -> float:
        """Duration of the interval in seconds."""
        return self.stop - self.start


def _get_data_record_onsets(
    timekeeping_signal_digital: npt.NDArray[np.uint8],
    bytes_per_data_record: int,
) -> tuple[decimal.Decimal, ...]:
    data_records = timekeeping_signal_digital.reshape((-1, bytes_per_data_record))
    return tuple(_get_data_record_onset(data_record) for data_record in data_records)


def _sampled_intervals_from_onsets(
    onsets: tuple[decimal.Decimal, ...],
    data_record_duration: decimal.Decimal,
) -> tuple[EdfInterval, ...]:
    # TAL onsets include the subsecond starttime offset, so the first data record's
    # onset is used as the reference to express intervals relative to the recording start.
    reference = onsets[0]
    merged: list[tuple[decimal.Decimal, decimal.Decimal]] = []
    for onset in onsets:
        start = onset - reference
        stop = start + data_record_duration
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return tuple(EdfInterval(float(start), float(stop)) for start, stop in merged)


def _intervals_for_records(
    onsets: tuple[decimal.Decimal, ...],
    data_record_duration: decimal.Decimal,
    record_indices: tuple[int, ...],
) -> tuple[EdfInterval, ...]:
    """Map selected data records to half-open intervals on the relative time axis."""
    reference = onsets[0]
    intervals = tuple(
        EdfInterval(
            float(onsets[index] - reference),
            float(onsets[index] - reference + data_record_duration),
        )
        for index in record_indices
    )
    return _merge_intervals(intervals)


def _merge_intervals(intervals: tuple[EdfInterval, ...]) -> tuple[EdfInterval, ...]:
    merged: list[EdfInterval] = []
    for interval in intervals:
        if merged and interval.start <= merged[-1].stop:
            merged[-1] = EdfInterval(
                merged[-1].start, max(merged[-1].stop, interval.stop)
            )
        else:
            merged.append(interval)
    return tuple(merged)


def _interval_difference(
    minuend: tuple[EdfInterval, ...],
    subtrahend: tuple[EdfInterval, ...],
) -> tuple[EdfInterval, ...]:
    """Subtract disjoint, sorted, half-open intervals from another such sequence."""
    result: list[EdfInterval] = []
    j = 0
    for interval_start, interval_stop in minuend:
        start = interval_start
        stop = interval_stop
        for interval in subtrahend[j:]:
            if interval.stop <= start:
                j += 1
                continue
            if interval.start >= stop:
                break
            if interval.start > start:
                result.append(EdfInterval(start, interval.start))
            start = max(start, interval.stop)
            if start >= stop:
                break
        if start < stop:
            result.append(EdfInterval(start, stop))
    return tuple(result)
