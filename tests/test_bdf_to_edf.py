from __future__ import annotations

import math

import numpy as np
import pytest

from edfio import (
    Bdf,
    BdfSignal,
    EdfAnnotation,
    SaturatedSample,
    convert_bdf_to_edf,
    read_bdf,
)
from edfio.bdf_to_edf import (
    ChannelConversionReport,
    _choose_physical_range,
    _header_range,
    _narrow_range,
    _validate_digital_range,
    _widest_centered_range,
    convert_signal_to_edf,
)
from edfio.edf import Edf as EdfClass
from edfio.edf import read_edf as read_edf_class
from edfio.edf_signal import EdfSignal as EdfSignalClass

pytestmark = pytest.mark.bdf


def ramp(n: int = 256, amplitude: float = 100.0) -> np.ndarray:
    return np.linspace(-amplitude, amplitude, n)


def make_bdf(*signals: BdfSignal) -> Bdf:
    return Bdf(list(signals))


def test_conversion_does_not_truncate_high_bits():
    # Naive truncation of the upper eight bits would destroy this signal,
    # because all 24-bit digital values are smaller than 2**16.
    data = np.array([-100.0, -1.0, 0.0, 1.0, 100.0])
    bdf = make_bdf(
        BdfSignal(
            data,
            1,
            label="tiny",
            physical_range=(-100.0, 100.0),
            digital_range=(-2048, 2047),
        )
    )
    edf, report = convert_bdf_to_edf(bdf, tolerance=0.01)

    assert isinstance(edf, EdfClass)
    np.testing.assert_allclose(edf.signals[0].data, bdf.signals[0].data, atol=0.01)
    assert report.all_satisfied
    assert report["tiny"].max_abs_error <= 0.01


def test_error_bound_covers_every_sample_not_only_extremes():
    # The extremes are exactly representable, while interior samples are not.
    # A bound computed only from the extrema would be zero here.
    data = np.array([-1.0, 0.3333333, 0.6666667, 1.0])
    bdf = make_bdf(
        BdfSignal(
            data,
            1,
            label="interior",
            physical_range=(-1.0, 1.0),
            digital_range=(-32768, 32767),
        )
    )
    edf, report = convert_bdf_to_edf(bdf, tolerance=1e-4)
    channel = report["interior"]

    decoded = edf.signals[0].data
    errors = np.abs(decoded - bdf.signals[0].data)
    # The reported maximum is the maximum over all samples.
    assert channel.max_abs_error == pytest.approx(float(errors.max()))
    assert channel.worst_sample_index == int(np.argmax(errors))
    assert channel.max_abs_error <= channel.error_bound
    for index, error in enumerate(errors):
        assert error <= channel.error_bound, index


def test_error_bound_covers_whole_segment_including_clipped_part():
    n = 10_000
    data = np.linspace(-1e6, 1e6, n)
    # A very tight tolerance cannot cover the full amplitude, so the extremes
    # saturate.  The report's bound must include those realized errors.
    bdf = make_bdf(
        BdfSignal(
            data,
            100,
            label="sat",
            physical_range=(-2e6, 2e6),
        )
    )
    edf, report = convert_bdf_to_edf(bdf, tolerance=0.5)
    channel = report["sat"]

    assert channel.satisfied is False
    assert channel.saturated_samples
    assert report.unsatisfiable_channels.labels == ("sat",)
    decoded = edf.signals[0].data
    errors = np.abs(decoded - bdf.signals[0].data)
    assert channel.max_abs_error == pytest.approx(float(errors.max()))
    assert channel.max_abs_error <= channel.error_bound
    for error in errors:
        assert error <= channel.error_bound


def test_unsatisfiable_channel_is_reported_but_other_channels_still_convert():
    n = 500
    small = np.sin(np.linspace(0, 4 * math.pi, n)) * 10
    huge = np.linspace(-1e6, 1e6, n)
    bdf = make_bdf(
        BdfSignal(small, 100, label="small", physical_range=(-10.0, 10.0)),
        BdfSignal(huge, 100, label="huge", physical_range=(-2e6, 2e6)),
    )
    result = convert_bdf_to_edf(
        bdf,
        tolerances={"small": 0.05, "huge": 1.0},
    )
    edf, report = result

    assert report["small"].satisfied
    assert not report["huge"].satisfied
    assert report.unsatisfiable_channels.labels == ("huge",)
    assert report["huge"].reason
    np.testing.assert_allclose(
        edf.get_signal("small").data,
        bdf.get_signal("small").data,
        atol=0.05,
    )


def test_saturated_samples_report_indices_values_and_errors():
    data = np.zeros(100)
    data[13] = 1e6
    data[71] = -1e6
    data[42] = 0.5
    bdf = make_bdf(BdfSignal(data, 10, label="spikes", physical_range=(-2e6, 2e6)))
    edf, report = convert_bdf_to_edf(bdf, tolerance=0.01)
    channel = report["spikes"]

    assert channel.saturated_sample_indices == (13, 71)
    assert channel.num_saturated_samples == 2
    assert all(isinstance(s, SaturatedSample) for s in channel.saturated_samples)
    decoded = edf.signals[0].data
    source = bdf.signals[0].data
    for sample in channel.saturated_samples:
        assert sample.channel == "spikes"
        assert sample.value == source[sample.index]
        assert sample.decoded_value == pytest.approx(decoded[sample.index])
        assert sample.error == pytest.approx(abs(decoded[sample.index] - sample.value))
    assert report.saturated_samples == channel.saturated_samples
    assert report.saturated_samples_by_channel == {"spikes": channel.saturated_samples}
    assert report.num_saturated_samples == 2


def test_saturation_is_split_between_both_extremes():
    data = np.array([-10.0, 10.0])
    bdf = make_bdf(BdfSignal(data, 1, label="symmetric", physical_range=(-20.0, 20.0)))
    _, report = convert_bdf_to_edf(bdf, tolerance=1e-5)
    channel = report["symmetric"]

    assert len(channel.saturated_samples) == 2
    lower = channel.saturated_samples[0].error
    upper = channel.saturated_samples[1].error
    assert lower == pytest.approx(upper, rel=1e-3)


def test_conversion_roundtrips_through_file(tmp_path):
    data = np.sin(np.linspace(0, 50, 5000)) * 250
    bdf = make_bdf(BdfSignal(data, 100, label="sine", physical_range=(-500.0, 500.0)))
    edf, report = convert_bdf_to_edf(bdf, tolerance=0.05)
    target = tmp_path / "converted.edf"
    edf.write(target)

    loaded = read_edf_class(target)
    np.testing.assert_allclose(loaded.signals[0].data, bdf.signals[0].data, atol=0.05)
    assert report["sine"].max_abs_error <= 0.05
    assert loaded.signals[0].physical_range == edf.signals[0].physical_range


def test_conversion_preserves_header_and_annotations(tmp_path):
    bdf = Bdf(
        [
            BdfSignal(
                np.arange(100, dtype=float),
                10,
                label="s",
                physical_range=(0.0, 99.0),
            )
        ],
        annotations=[EdfAnnotation(1.0, None, "marker")],
    )
    edf, report = convert_bdf_to_edf(bdf, tolerance=0.01)

    assert edf.annotations == bdf.annotations
    assert edf.local_patient_identification == bdf.local_patient_identification
    assert edf.local_recording_identification == bdf.local_recording_identification
    assert edf.starttime == bdf.starttime
    assert report.all_satisfied


def test_annotation_only_bdf_converts(tmp_path):
    bdf = read_bdf("tests/TEST_DATA/SC4001EC-Hypnogram.bdf")
    assert not bdf.signals
    edf, report = convert_bdf_to_edf(bdf, tolerance=0.1)

    assert isinstance(edf, EdfClass)
    assert not edf.signals
    assert edf.annotations == bdf.annotations
    assert edf.duration == bdf.duration == 0.0
    assert len(report.channels) == 0
    assert report.all_satisfied

    target = tmp_path / "hypnogram.edf"
    edf.write(target)
    loaded = read_edf_class(target)
    assert loaded.annotations == bdf.annotations


def test_constant_channel_is_represented_exactly():
    data = np.full(50, 3.25)
    bdf = make_bdf(BdfSignal(data, 10, label="dc", physical_range=(3.0, 3.5)))
    edf, report = convert_bdf_to_edf(bdf, tolerance=1e-6)

    np.testing.assert_allclose(edf.signals[0].data, bdf.signals[0].data, atol=1e-6)
    assert report["dc"].max_abs_error <= 1e-6


def test_per_channel_tolerances_sequence():
    data = ramp(128)
    bdf = make_bdf(
        BdfSignal(data, 128, label="a", physical_range=(-100.0, 100.0)),
        BdfSignal(data, 128, label="b", physical_range=(-100.0, 100.0)),
    )
    _, report = convert_bdf_to_edf(bdf, [0.1, 0.2])

    assert report[0].allowed_error == 0.1
    assert report[1].allowed_error == 0.2
    assert report["a"].satisfied
    assert report["b"].satisfied


def test_per_channel_tolerances_missing_channel_raises_error():
    bdf = make_bdf(
        BdfSignal(ramp(10), 10, label="a", physical_range=(-100.0, 100.0)),
    )
    with pytest.raises(ValueError, match="No allowed error specified for channel"):
        convert_bdf_to_edf(bdf, tolerances={"other": 1.0})


def test_invalid_tolerances_raise_error():
    bdf = make_bdf(
        BdfSignal(ramp(10), 10, label="a", physical_range=(-100.0, 100.0)),
    )
    with pytest.raises(ValueError, match="non-negative"):
        convert_bdf_to_edf(bdf, tolerance=-1.0)
    with pytest.raises(ValueError, match="finite"):
        convert_bdf_to_edf(bdf, tolerance=math.inf)
    with pytest.raises(ValueError, match="No allowed error"):
        convert_bdf_to_edf(bdf)


def test_multiple_tolerance_keywords_raise_error():
    bdf = make_bdf(
        BdfSignal(ramp(10), 10, label="a", physical_range=(-100.0, 100.0)),
    )
    with pytest.raises(ValueError, match="only one"):
        convert_bdf_to_edf(bdf, tolerance=1.0, allowed_error=1.0)


def test_invalid_digital_range_raises_error():
    bdf = make_bdf(
        BdfSignal(ramp(10), 10, label="a", physical_range=(-100.0, 100.0)),
    )
    with pytest.raises(ValueError, match="out of supported range"):
        convert_bdf_to_edf(bdf, tolerance=1.0, digital_range=(-40000, 40000))
    with pytest.raises(ValueError, match="smaller than digital maximum"):
        convert_bdf_to_edf(bdf, tolerance=1.0, digital_range=(1, 1))


def test_report_supports_label_and_index_access():
    bdf = make_bdf(
        BdfSignal(ramp(10), 10, label="a", physical_range=(-100.0, 100.0)),
        BdfSignal(ramp(10), 10, label="b", physical_range=(-100.0, 100.0)),
    )
    _, report = convert_bdf_to_edf(bdf, tolerance=1.0)

    assert report["a"] == report[0]
    assert report.channels.labels == ("a", "b")
    assert "a" in report.channels
    assert len(report) == 2
    assert list(report) == list(report.channels)
    with pytest.raises(KeyError, match="No channel"):
        report["missing"]


def test_bdf_and_bdf_signal_convenience_methods():
    data = ramp(100, 50.0)
    bdf = make_bdf(BdfSignal(data, 100, label="a", physical_range=(-100.0, 100.0)))
    edf, report = bdf.to_edf(tolerance=0.01)
    assert isinstance(edf, EdfClass)
    assert report["a"].max_abs_error <= 0.01

    signal, channel_report = bdf.get_signal("a").to_edf(0.01)
    assert isinstance(signal, EdfSignalClass)
    assert isinstance(channel_report, ChannelConversionReport)
    assert channel_report.max_abs_error <= 0.01
    np.testing.assert_allclose(signal.data, bdf.get_signal("a").data, atol=0.01)


def test_conversion_result_unpacks():
    bdf = make_bdf(
        BdfSignal(ramp(10), 10, label="a", physical_range=(-100.0, 100.0)),
    )
    result = convert_bdf_to_edf(bdf, tolerance=1.0)
    edf, report = result
    assert edf is result
    assert report is result.report
    assert result.edf is result


def test_conversion_uses_custom_data_record_duration():
    bdf = make_bdf(
        BdfSignal(ramp(100), 10, label="a", physical_range=(-100.0, 100.0)),
    )
    edf, _ = convert_bdf_to_edf(
        bdf,
        tolerance=1.0,
        data_record_duration=5.0,
    )
    assert edf.data_record_duration == 5.0


def test_saturated_sample_value_helpers():
    data = np.array([0.0, 1e6])
    bdf = make_bdf(BdfSignal(data, 1, label="s", physical_range=(-2e6, 2e6)))
    _, report = convert_bdf_to_edf(bdf, tolerance=1.0)
    sample = report["s"].saturated_samples[-1]
    assert sample.index == 1

    assert sample.original_value == sample.value
    assert sample.encoded_value == sample.decoded_value
    assert sample.absolute_error == sample.error
    assert sample == SaturatedSample(
        sample.channel, sample.index, sample.value, sample.decoded_value
    )


def test_channel_report_aliases():
    data = np.array([0.0, 1e6])
    bdf = make_bdf(BdfSignal(data, 1, label="s", physical_range=(-2e6, 2e6)))
    _, report = convert_bdf_to_edf(bdf, tolerance=1.0)
    channel = report["s"]

    assert channel.unsatisfiable
    assert channel.has_saturation
    assert channel.worst_saturated_sample == channel.saturated_samples[0]
    assert report.satisfied_channels.labels == ()
    assert report.saturated_channels.labels == ("s",)
    assert len(report.saturated_samples_by_channel["s"]) == 2


def test_report_collection_comparisons():
    bdf = make_bdf(
        BdfSignal(ramp(10), 10, label="a", physical_range=(-100.0, 100.0)),
    )
    _, report = convert_bdf_to_edf(bdf, tolerance=1.0)
    channels = report.channels

    assert channels == tuple(channels)
    assert channels != "a"
    assert channels != 5
    assert channels != (channels[0], channels[0])
    assert channels[0] in channels
    assert "a" in channels
    assert 0 not in channels


def test_get_channel_with_duplicate_labels_raises_error():
    bdf = make_bdf(
        BdfSignal(ramp(10), 10, label="same", physical_range=(-100.0, 100.0)),
        BdfSignal(ramp(10), 10, label="same", physical_range=(-100.0, 100.0)),
    )
    _, report = convert_bdf_to_edf(bdf, tolerance=1.0)
    with pytest.raises(ValueError, match="Ambiguous channel label"):
        report["same"]


def test_default_tolerance_and_aliases():
    bdf = make_bdf(
        BdfSignal(ramp(10), 10, label="a", physical_range=(-100.0, 100.0)),
    )
    _, report = convert_bdf_to_edf(bdf, {"a": None}, default_tolerance=0.5)
    assert report["a"].allowed_error == 0.5
    _, report = convert_bdf_to_edf(bdf, channel_tolerances=0.5)
    assert report["a"].allowed_error == 0.5
    _, report = convert_bdf_to_edf(bdf, default_tolerance=0.5)
    assert report["a"].allowed_error == 0.5


def test_conversion_error_for_empty_and_nonfinite_signals():
    empty = BdfSignal(np.array([1.0]), 1, label="x")
    empty._digital = np.array([], dtype=np.int32)
    with pytest.raises(ValueError, match="contains no samples"):
        convert_signal_to_edf(empty, 1.0)

    class FakeSignal:
        label = "x"
        sampling_frequency = 1
        transducer_type = ""
        physical_dimension = ""
        prefiltering = ""
        data = np.array([1.0, np.inf])

    with pytest.raises(ValueError, match="contains non-finite"):
        convert_signal_to_edf(FakeSignal(), 1.0)


def test_coverage_for_internal_range_helpers():
    assert _header_range(math.inf, 1.0) is None
    assert _header_range(1.0, 1.0) is None
    assert _header_range(0.0, 1e300) is None
    assert _widest_centered_range(0.0, 0.0) is None
    narrow = _narrow_range(0.5)
    assert narrow[0] <= 0.5 <= narrow[1]
    _, covered = _choose_physical_range(1e6, 1e6, 1.0)
    assert covered


def test_tolerance_sequence_length_mismatch():
    bdf = make_bdf(
        BdfSignal(ramp(10), 10, label="a", physical_range=(-100.0, 100.0)),
    )
    with pytest.raises(ValueError, match="Expected 1 tolerances"):
        convert_bdf_to_edf(bdf, [0.1, 0.2])


def test_signal_conversion_rejects_non_real_tolerance():
    bdf = make_bdf(
        BdfSignal(ramp(10), 10, label="a", physical_range=(-100.0, 100.0)),
    )
    with pytest.raises(TypeError, match="must be a real number"):
        convert_signal_to_edf(bdf.signals[0], "0.1")


def test_aliases_are_callable():
    from edfio.bdf_to_edf import bdf_to_edf, to_edf  # noqa: PLC0415

    bdf = make_bdf(
        BdfSignal(ramp(10), 10, label="a", physical_range=(-100.0, 100.0)),
    )
    assert bdf_to_edf(bdf, 0.1).report.all_satisfied
    assert to_edf(bdf, 0.1).report.all_satisfied


def test_custom_digital_range_returns_ints():
    assert _validate_digital_range(np.array([-1000, 1000])) == (-1000, 1000)


def test_narrow_range_raises_for_unrepresentable_center():
    with pytest.raises(ValueError, match="cannot be represented"):
        _narrow_range(1e300)


def test_zero_tolerance_is_unsatisfiable_without_saturation_error():
    # A zero allowed error can only be met by an exactly representable data
    # range; the fallback path is still required for ranges that are not.
    _, covered = _choose_physical_range(-1.0, 1.0, 0.0)
    assert not covered


def test_header_resolution_limit_is_reported_without_clipping():
    # Flooring -1.0000004 to eight header characters gives -1.00001, so even
    # the smallest representable enclosing range is wider than the requested
    # span.  No sample is clipped, but the channel is honestly reported as
    # unsatisfiable instead of pretending the tolerance was met.
    encoded, covered = _choose_physical_range(-1.0000004, 1.0000004, 2.000002)
    assert not covered
    assert encoded[0] <= -1.0000004
    assert encoded[1] >= 1.0000004
    assert encoded[1] - encoded[0] < 2.0001


def test_report_aggregate_views():
    data = np.zeros(10)
    data[2] = 1e6
    data[7] = -1e6
    bdf = make_bdf(
        BdfSignal(data, 1, label="spiky", physical_range=(-2e6, 2e6)),
    )
    _, report = convert_bdf_to_edf(bdf, tolerance=1.0)

    assert report.unsatisfiable_channel_labels == ("spiky",)
    assert report.unsatisfied_channels == report.unsatisfiable_channels
    assert report.failed_channels == report.unsatisfiable_channels
    assert report.saturated_sample_indices_by_channel == {"spiky": (2, 7)}
    assert report.ok is False
    assert report.satisfied is False
    channel = report["spiky"]
    assert channel.tolerance == channel.allowed_error
    assert channel.max_error == channel.max_abs_error
    assert channel.saturated_indices == channel.saturated_sample_indices
    assert channel.quantization_error_bound == channel.quantization_step / 2


def test_report_recording_level_aggregates():
    data = np.zeros(10)
    data[0] = -1e6
    data[5] = 1e6
    data[8] = 0.25
    bdf = make_bdf(
        BdfSignal(data, 1, label="noisy", physical_range=(-2e6, 2e6)),
    )
    _, report = convert_bdf_to_edf(bdf, tolerance=1.0)

    assert report.num_unsatisfiable_channels == 1
    assert report.error_bound == report["noisy"].error_bound
    _, empty = convert_bdf_to_edf(
        read_bdf("tests/TEST_DATA/SC4001EC-Hypnogram.bdf"),
        tolerance=1.0,
    )
    assert empty.error_bound == 0.0


def test_bdf_input_is_not_modified_by_conversion():
    data = np.sin(np.linspace(0, 6, 300)) * 100
    bdf = make_bdf(
        BdfSignal(data, 100, label="a", physical_range=(-200.0, 200.0)),
        BdfSignal(data, 100, label="b", physical_range=(-200.0, 200.0)),
    )
    digital_before = [signal.digital.copy() for signal in bdf.signals]
    range_before = [signal.physical_range for signal in bdf.signals]

    edf, report = convert_bdf_to_edf(bdf, tolerance=0.05)

    for signal, before, physical_range in zip(
        bdf.signals, digital_before, range_before
    ):
        np.testing.assert_array_equal(signal.digital, before)
        assert signal.physical_range == physical_range
    assert bdf.signals[0].digital.dtype == np.int32
    assert edf.signals[0].digital.dtype == np.int16
    assert all(channel.satisfied for channel in report)
