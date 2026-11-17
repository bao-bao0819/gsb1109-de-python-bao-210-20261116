import numpy as np
import pytest

from edfio import Edf, EdfSignal
from edfio.clock_sync import (
    ClockCorrectionSegment,
    ClockCorrectionTable,
)


def test_segment_fails_for_non_positive_device_range():
    with pytest.raises(ValueError, match="must be after device_start"):
        ClockCorrectionSegment(device_start=10, device_stop=10, offset=0)
    with pytest.raises(ValueError, match="must be after device_start"):
        ClockCorrectionSegment(device_start=10, device_stop=5, offset=0)


def test_segment_fails_for_negative_offset_error():
    with pytest.raises(ValueError, match="offset_error must be non-negative"):
        ClockCorrectionSegment(device_start=0, device_stop=10, offset=0, offset_error=-1)


@pytest.mark.parametrize(
    "field",
    ["device_start", "device_stop", "offset", "offset_error"],
)
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_segment_fails_for_non_finite_values(field: str, value: float):
    kwargs = {"device_start": 0, "device_stop": 10, "offset": 0, "offset_error": 0}
    kwargs[field] = value
    with pytest.raises(ValueError, match=f"{field} must be finite"):
        ClockCorrectionSegment(**kwargs)


def test_segment_experiment_time_properties():
    segment = ClockCorrectionSegment(device_start=10, device_stop=20, offset=5)
    assert segment.experiment_start == 15
    assert segment.experiment_stop == 25


def test_table_sorts_segments_by_device_time():
    later = ClockCorrectionSegment(device_start=20, device_stop=30, offset=0)
    earlier = ClockCorrectionSegment(device_start=0, device_stop=10, offset=0)
    table = ClockCorrectionTable([later, earlier])
    assert table.segments == (earlier, later)


def test_table_repr():
    table = ClockCorrectionTable(
        [ClockCorrectionSegment(0, 10, 0), ClockCorrectionSegment(10, 20, 0)]
    )
    assert repr(table) == "<ClockCorrectionTable 2 segments>"


def test_table_fails_for_non_segment_objects():
    with pytest.raises(TypeError, match="must be ClockCorrectionSegment"):
        ClockCorrectionTable([(0, 10, 0)])


def test_map_experiment_window_fails_for_invalid_window():
    table = ClockCorrectionTable([ClockCorrectionSegment(0, 10, 0)])
    with pytest.raises(ValueError, match="must be after query_start"):
        table.map_experiment_window(10, 10)
    with pytest.raises(ValueError, match="must be after query_start"):
        table.map_experiment_window(20, 10)
    with pytest.raises(ValueError, match="query_start must be finite"):
        table.map_experiment_window(np.nan, 10)
    with pytest.raises(ValueError, match="query_stop must be finite"):
        table.map_experiment_window(0, np.inf)


def test_exact_offset_maps_to_single_definite_window():
    segment = ClockCorrectionSegment(device_start=0, device_stop=100, offset=1000)
    table = ClockCorrectionTable([segment])
    mapping = table.map_experiment_window(1010, 1020)
    assert mapping.query_start == 1010
    assert mapping.query_stop == 1020
    assert len(mapping.windows) == 1
    window = mapping.windows[0]
    assert window.segment is segment
    assert window.device_start == 10
    assert window.device_stop == 20
    assert window.experiment_start == 1010
    assert window.experiment_stop == 1020
    assert window.certainty == "definite"
    assert mapping.uncovered == ()


def test_query_outside_coverage_returns_no_windows():
    table = ClockCorrectionTable([ClockCorrectionSegment(0, 100, offset=1000)])
    mapping = table.map_experiment_window(2000, 2010)
    assert mapping.windows == ()
    assert mapping.uncovered == ((2000, 2010),)


def test_empty_table_assumes_no_zero_offset():
    mapping = ClockCorrectionTable([]).map_experiment_window(10, 20)
    assert mapping.windows == ()
    assert mapping.uncovered == ((10, 20),)


def test_overlapping_segments_keep_all_valid_windows():
    segment_a = ClockCorrectionSegment(device_start=0, device_stop=100, offset=1000)
    segment_b = ClockCorrectionSegment(device_start=50, device_stop=150, offset=1000)
    table = ClockCorrectionTable([segment_a, segment_b])
    mapping = table.map_experiment_window(1060, 1070)
    assert len(mapping.windows) == 2
    first, second = mapping.windows
    assert first.segment is segment_a
    assert second.segment is segment_b
    assert (first.device_start, first.device_stop) == (60, 70)
    assert (second.device_start, second.device_stop) == (60, 70)
    assert first.certainty == second.certainty == "definite"
    assert mapping.uncovered == ()


def test_gap_between_segments_is_reported_uncovered():
    segment_a = ClockCorrectionSegment(device_start=0, device_stop=10, offset=0)
    segment_b = ClockCorrectionSegment(device_start=20, device_stop=30, offset=0)
    table = ClockCorrectionTable([segment_a, segment_b])
    mapping = table.map_experiment_window(5, 25)
    assert [(w.device_start, w.device_stop) for w in mapping.windows] == [
        (5, 10),
        (20, 25),
    ]
    assert all(w.certainty == "definite" for w in mapping.windows)
    assert mapping.uncovered == ((10, 20),)


def test_offset_error_distinguishes_definite_and_possible():
    segment = ClockCorrectionSegment(0, 1000, offset=0, offset_error=2)
    mapping = ClockCorrectionTable([segment]).map_experiment_window(10, 20)
    assert len(mapping.windows) == 3
    left, definite, right = mapping.windows
    assert (left.device_start, left.device_stop) == (8, 12)
    assert left.certainty == "possible"
    assert (left.experiment_start, left.experiment_stop) == (10, 14)
    assert (definite.device_start, definite.device_stop) == (12, 18)
    assert definite.certainty == "definite"
    assert (definite.experiment_start, definite.experiment_stop) == (10, 20)
    assert (right.device_start, right.device_stop) == (18, 22)
    assert right.certainty == "possible"
    assert (right.experiment_start, right.experiment_stop) == (16, 20)
    assert mapping.uncovered == ()


def test_offset_error_wider_than_window_yields_only_possible():
    segment = ClockCorrectionSegment(0, 1000, offset=0, offset_error=2)
    mapping = ClockCorrectionTable([segment]).map_experiment_window(10, 13)
    assert len(mapping.windows) == 1
    window = mapping.windows[0]
    assert window.certainty == "possible"
    assert (window.device_start, window.device_stop) == (8, 15)
    assert (window.experiment_start, window.experiment_stop) == (10, 13)


def test_uncovered_accounts_for_offset_error():
    segment = ClockCorrectionSegment(0, 10, offset=0, offset_error=1)
    mapping = ClockCorrectionTable([segment]).map_experiment_window(0, 20)
    assert mapping.uncovered == ((11, 20),)
    possible, definite = mapping.windows
    assert (possible.device_start, possible.device_stop) == (0, 1)
    assert possible.certainty == "possible"
    assert (definite.device_start, definite.device_stop) == (1, 10)
    assert definite.certainty == "definite"


def test_segment_domain_clips_possible_band_on_one_side():
    segment = ClockCorrectionSegment(95, 200, offset=0, offset_error=2)
    mapping = ClockCorrectionTable([segment]).map_experiment_window(10, 100)
    definite, right = mapping.windows
    assert (definite.device_start, definite.device_stop) == (95, 98)
    assert definite.certainty == "definite"
    assert (right.device_start, right.device_stop) == (98, 102)
    assert right.certainty == "possible"


def _make_edf(durations_and_frequencies=((100, 10), (100, 100))):
    signals = []
    for i, (duration, frequency) in enumerate(durations_and_frequencies):
        data = np.arange(duration * frequency, dtype=np.float64)
        signals.append(EdfSignal(data, frequency, label=f"s{i}"))
    return Edf(signals)


def test_sample_ranges_keep_individual_sampling_frequencies():
    edf = _make_edf()
    table = ClockCorrectionTable([ClockCorrectionSegment(0, 100, offset=0)])
    mapping = table.map_experiment_window(10, 20)
    result = mapping.get_recording_sample_ranges(edf)
    assert len(result) == 2
    slow, fast = result
    assert slow.label == "s0"
    assert slow.sampling_frequency == 10
    assert [(r.start, r.stop) for r in slow.ranges] == [(100, 200)]
    assert fast.label == "s1"
    assert fast.sampling_frequency == 100
    assert [(r.start, r.stop) for r in fast.ranges] == [(1000, 2000)]
    signal_data = edf.signals[0].data
    assert len(signal_data[slow.ranges[0].to_slice()]) == 100
    assert len(edf.signals[1].data[fast.ranges[0].to_slice()]) == 1000


def test_sample_ranges_are_clipped_to_recording_duration():
    edf = _make_edf(((100, 10),))
    table = ClockCorrectionTable([ClockCorrectionSegment(0, 200, offset=0)])
    mapping = table.map_experiment_window(90, 120)
    (result,) = mapping.get_recording_sample_ranges(edf)
    assert [(r.start, r.stop) for r in result.ranges] == [(900, 1000)]


def test_sample_ranges_beyond_recording_are_omitted():
    edf = _make_edf(((100, 10),))
    table = ClockCorrectionTable([ClockCorrectionSegment(0, 200, offset=0)])
    mapping = table.map_experiment_window(150, 160)
    (result,) = mapping.get_recording_sample_ranges(edf)
    assert result.ranges == ()


def test_sample_ranges_handle_fractional_boundaries():
    edf = _make_edf(((100, 10),))
    table = ClockCorrectionTable([ClockCorrectionSegment(0, 100, offset=0)])
    mapping = table.map_experiment_window(0.3, 0.5)
    (result,) = mapping.get_recording_sample_ranges(edf)
    assert [(r.start, r.stop) for r in result.ranges] == [(3, 5)]
    mapping = table.map_experiment_window(0.32, 0.57)
    (result,) = mapping.get_recording_sample_ranges(edf)
    assert [(r.start, r.stop) for r in result.ranges] == [(4, 6)]


def test_sample_range_exposes_window_and_slice():
    edf = _make_edf(((100, 10),))
    segment = ClockCorrectionSegment(0, 100, offset=0, offset_error=1)
    mapping = ClockCorrectionTable([segment]).map_experiment_window(10, 20)
    (result,) = mapping.get_recording_sample_ranges(edf)
    assert len(result.ranges) == 3
    possible = result.ranges[0]
    assert possible.certainty == "possible"
    assert possible.window.certainty == "possible"
    assert possible.to_slice() == slice(possible.start, possible.stop)
    data = edf.signals[0].data
    np.testing.assert_array_equal(
        data[possible.to_slice()], data[possible.start : possible.stop]
    )


def test_get_sample_ranges_without_windows_returns_empty():
    edf = _make_edf(((100, 10),))
    table = ClockCorrectionTable([ClockCorrectionSegment(0, 100, offset=1000)])
    mapping = table.map_experiment_window(2000, 2010)
    assert mapping.get_sample_ranges(edf.signals[0]) == ()


def test_multiple_valid_intervals_all_kept_per_signal():
    edf = _make_edf(((100, 10),))
    segment_a = ClockCorrectionSegment(0, 10, offset=0)
    segment_b = ClockCorrectionSegment(20, 30, offset=0)
    table = ClockCorrectionTable([segment_a, segment_b])
    mapping = table.map_experiment_window(5, 25)
    ranges = mapping.get_sample_ranges(edf.signals[0])
    assert [(r.start, r.stop) for r in ranges] == [(50, 100), (200, 250)]
    assert ranges[0].window.segment is segment_a
    assert ranges[1].window.segment is segment_b
