from __future__ import annotations

import numpy as np
import pytest

from edfio import Edf, EdfAnnotation, EdfInterval, EdfSignal, read_edf
from edfio.edf_annotations import (
    _data_records_to_annotations_signal,
    _EdfAnnotationsDataRecord,
    _EdfTAL,
)
from edfio.edf_intervals import _interval_difference, _merge_intervals


def _edf_with_record_onsets(
    onsets: tuple[float, ...],
    data_record_duration: float,
    signal_data: np.ndarray | None = None,
) -> Edf:
    if signal_data is None:
        signal_data = np.arange(1, len(onsets) + 1, dtype=float)
    annotations_signal = _data_records_to_annotations_signal(
        [
            _EdfAnnotationsDataRecord([_EdfTAL(onset, None, [])]).to_bytes()
            for onset in onsets
        ],
        EdfSignal,
        data_record_duration,
    )
    return Edf(
        [EdfSignal(signal_data, 1 / data_record_duration), annotations_signal],
        data_record_duration=data_record_duration,
    )


def test_edf_interval_duration():
    interval = EdfInterval(2.0, 5.5)
    assert interval.start == 2.0
    assert interval.stop == 5.5
    assert interval.duration == 3.5


def test_sampled_intervals_continuous_recording():
    edf = _edf_with_record_onsets((0, 1, 2, 3), 1)
    assert edf.is_continuous
    assert edf.sampled_intervals == (EdfInterval(0.0, 4.0),)
    assert edf.gap_intervals == ()
    assert edf.sampling_duration == 4
    assert edf.gap_duration == 0
    assert edf.sampling_duration == edf.duration


def test_sampled_intervals_single_gap():
    edf = _edf_with_record_onsets((0, 1, 3, 4), 1)
    assert not edf.is_continuous
    assert edf.sampled_intervals == (
        EdfInterval(0.0, 2.0),
        EdfInterval(3.0, 5.0),
    )
    assert edf.gap_intervals == (EdfInterval(2.0, 3.0),)
    assert edf.gap_duration == 1
    assert edf.sampling_duration == 4
    assert edf.duration == 4


def test_sampled_intervals_multiple_gaps():
    edf = _edf_with_record_onsets((0, 1, 5, 6, 10, 11), 1)
    assert edf.gap_intervals == (
        EdfInterval(2.0, 5.0),
        EdfInterval(7.0, 10.0),
    )
    assert edf.gap_duration == 6
    assert edf.sampling_duration == 6
    assert edf.duration == 6


def test_ten_minute_gap_is_not_counted_as_silence(tmp_file):
    # Four 1-second data records, with a ten-minute (600 s) sampling gap in between.
    edf = _edf_with_record_onsets((0, 1, 602, 603), 1)
    edf.write(tmp_file)
    edf = read_edf(tmp_file)

    assert edf.gap_intervals == (EdfInterval(2.0, 602.0),)
    assert edf.gap_duration == 600
    # Nonzero data: nothing was recorded as zero signal.
    assert edf.zero_signal_intervals == ()
    assert edf.zero_signal_duration == 0
    # The gap must not enter the true sampling time.
    assert edf.sampling_duration == 4
    assert edf.sampling_duration + edf.gap_duration == 604


def test_zero_signal_interval_is_distinguished_from_gap():
    # Record 1 contains genuine zero samples, records 3-4 are separated by a gap.
    signal_data = np.array([1.0, 0.0, 1.0, 1.0, 1.0])
    edf = _edf_with_record_onsets((0, 1, 2, 12, 13), 1, signal_data=signal_data)

    assert edf.gap_intervals == (EdfInterval(3.0, 12.0),)
    assert edf.zero_signal_intervals == (EdfInterval(1.0, 2.0),)
    assert edf.zero_signal_duration == 1
    assert edf.sampling_duration == 5
    assert edf.gap_duration == 9


def test_zero_signal_requires_all_signals_to_be_zero():
    annotations_signal = _data_records_to_annotations_signal(
        [
            _EdfAnnotationsDataRecord([_EdfTAL(onset, None, [])]).to_bytes()
            for onset in (0, 1, 2)
        ],
        EdfSignal,
        1,
    )
    first = EdfSignal(np.array([0.0, 0.0, 1.0]), 1, label="EEG")
    second = EdfSignal(np.array([1.0, 0.0, 1.0]), 1, label="ECG")
    edf = Edf([first, second, annotations_signal], data_record_duration=1)

    # Only record 1 is zero in all signals.
    assert edf.zero_signal_intervals == (EdfInterval(1.0, 2.0),)


def test_adjacent_zero_records_are_merged():
    signal_data = np.array([1.0, 0.0, 0.0, 1.0])
    edf = _edf_with_record_onsets((0, 1, 2, 3), 1, signal_data=signal_data)
    assert edf.zero_signal_intervals == (EdfInterval(1.0, 3.0),)
    assert edf.zero_signal_duration == 2


def test_zero_signal_intervals_follow_true_time_axis_across_gap():
    # The zero record is the third data record, which starts at t=12 due to the gap.
    signal_data = np.array([1.0, 1.0, 0.0, 1.0])
    edf = _edf_with_record_onsets((0, 1, 12, 13), 1, signal_data=signal_data)
    assert edf.zero_signal_intervals == (EdfInterval(12.0, 13.0),)


def test_sampled_intervals_without_annotations_signal():
    edf = Edf([EdfSignal(np.arange(10), 1)])
    assert edf.is_continuous
    assert edf.sampled_intervals == (EdfInterval(0.0, 10.0),)
    assert edf.gap_intervals == ()
    assert edf.sampling_duration == edf.duration


def test_zero_signal_intervals_without_annotations_signal():
    edf = Edf([EdfSignal(np.array([1.0, 0.0, 0.0, 1.0]), 1)])
    assert edf.zero_signal_intervals == (EdfInterval(1.0, 3.0),)
    assert edf.zero_signal_duration == 2


def test_zero_signal_intervals_annotation_only_file():
    edf = Edf([], annotations=[EdfAnnotation(0, None, "note")])
    assert edf.zero_signal_intervals == ()
    assert edf.zero_signal_duration == 0


def test_overlapping_record_onsets_are_merged():
    edf = _edf_with_record_onsets((0, 0.5, 1.5), 1)
    assert edf.is_continuous is False
    assert edf.sampled_intervals == (EdfInterval(0.0, 2.5),)
    assert edf.gap_intervals == ()
    assert edf.sampling_duration == 2.5


def test_intervals_are_relative_to_first_record_onset():
    # The first TAL onset is nonzero (e.g. a subsecond starttime offset): intervals
    # must still be expressed relative to the recording start, not absolute TAL time.
    edf = _edf_with_record_onsets((0.5, 1.5, 3.5, 4.5), 1)
    assert edf.sampled_intervals == (
        EdfInterval(0.0, 2.0),
        EdfInterval(3.0, 5.0),
    )
    assert edf.gap_intervals == (EdfInterval(2.0, 3.0),)
    assert edf.zero_signal_intervals == ()


def test_continuous_header_declaration_with_gap_tals_keeps_gap_evidence():
    edf = _edf_with_record_onsets((0, 1, 3, 4), 1)
    # The header claims continuity (EDF+C), but the timekeeping annotations prove a gap.
    edf._set_reserved(f"{Edf._fmt}+C")
    assert edf.reserved == f"{Edf._fmt}+C"
    assert edf.is_continuous is False
    assert edf.gap_intervals == (EdfInterval(2.0, 3.0),)
    assert edf.sampled_intervals == (
        EdfInterval(0.0, 2.0),
        EdfInterval(3.0, 5.0),
    )


def test_discontinuous_recording_survives_roundtrip(tmp_file):
    edf = _edf_with_record_onsets((0, 1, 3, 4), 1)
    edf.write(tmp_file)
    edf = read_edf(tmp_file)
    assert not edf.is_continuous
    assert edf.gap_intervals == (EdfInterval(2.0, 3.0),)
    assert edf.sampling_duration == 4
    assert edf.gap_duration == 1


def test_interval_difference_helper():
    assert _interval_difference(
        (EdfInterval(0.0, 10.0), EdfInterval(20.0, 30.0)),
        (
            EdfInterval(0.0, 5.0),
            EdfInterval(2.0, 4.0),
            EdfInterval(6.0, 8.0),
            EdfInterval(22.0, 24.0),
        ),
    ) == (
        EdfInterval(5.0, 6.0),
        EdfInterval(8.0, 10.0),
        EdfInterval(20.0, 22.0),
        EdfInterval(24.0, 30.0),
    )
    # A subtrahend covering the whole minuend leaves nothing.
    assert (
        _interval_difference((EdfInterval(2.0, 4.0),), (EdfInterval(0.0, 10.0),)) == ()
    )
    # A subtrahend outside the minuend leaves the minuend unchanged.
    assert _interval_difference((EdfInterval(0.0, 2.0),), (EdfInterval(4.0, 6.0),)) == (
        EdfInterval(0.0, 2.0),
    )


def test_merge_intervals_helper():
    assert _merge_intervals(()) == ()
    assert _merge_intervals(
        (
            EdfInterval(0.0, 2.0),
            EdfInterval(2.0, 4.0),
            EdfInterval(3.0, 5.0),
            EdfInterval(8.0, 9.0),
        )
    ) == (EdfInterval(0.0, 5.0), EdfInterval(8.0, 9.0))


@pytest.mark.parametrize(
    ("onsets", "data_record_duration", "expected_gap"),
    [
        ((0, 1, 3, 4), 1, EdfInterval(2.0, 3.0)),
        ((0.0, 0.1, 0.3, 0.4), 0.1, EdfInterval(0.2, 0.3)),
    ],
)
def test_gap_detection_for_different_record_durations(
    onsets, data_record_duration, expected_gap
):
    edf = _edf_with_record_onsets(onsets, data_record_duration)
    assert edf.gap_intervals == (expected_gap,)
    assert not edf.is_continuous
