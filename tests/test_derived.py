from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from edfio import (
    BadSegment,
    DerivedChannelSpec,
    Edf,
    EdfSignal,
    compute_differential_channel,
    derive_differential_channels,
    get_derived_channel,
    normalize_bad_segments,
    read_edf,
    set_bad_segments,
)
from edfio.derived import _registries
from edfio.edf_signal import BdfSignal as _BdfSignal
from edfio.edf_signal import EdfSignal as _EdfSignal
from tests.conftest import _Context

EDF_FILE = Path(__file__).parent / "TEST_DATA" / "short_psg.edf"


def _signal(
    values: list[float] | np.ndarray,
    sampling_frequency: float,
    label: str,
    *,
    physical_range: tuple[float, float] = (-100.0, 100.0),
    physical_dimension: str = "uV",
) -> object:
    return EdfSignal(
        np.asarray(values, dtype=np.float64),
        sampling_frequency,
        label=label,
        physical_range=physical_range,
        physical_dimension=physical_dimension,
    )


def _recording(*signals: object) -> object:
    return Edf(list(signals))


def test_new_signal_has_no_bad_segments():
    signal = _signal([1.0, 2.0], 2, "A")
    assert signal.bad_segments == ()


def test_read_signal_has_no_bad_segments():
    recording = read_edf(EDF_FILE)
    assert all(signal.bad_segments == () for signal in recording.signals)


def test_bad_segment_rejects_negative_interval():
    with pytest.raises(ValueError, match="must not be before"):
        BadSegment(1.0, 0.5)


def test_bad_segment_overlaps():
    segment = BadSegment(1.0, 2.0)
    assert segment.overlaps(1.5, 1.6)
    assert segment.overlaps(0.0, 1.5)
    assert not segment.overlaps(2.0, 3.0)
    assert not segment.overlaps(0.0, 1.0)


def test_normalize_bad_segments_sorts_merges_and_skips_empty():
    segments = normalize_bad_segments(
        [
            (2.0, 3.0),
            BadSegment(0.0, 1.0, "gap"),
            (0.5, 1.5, "saturation"),
            (4.0, 4.0),
        ]
    )
    assert segments == (
        BadSegment(0.0, 1.5, "gap; saturation"),
        BadSegment(2.0, 3.0),
    )


def test_normalize_bad_segments_merges_touching_intervals_without_reason():
    assert normalize_bad_segments([(0.0, 1.0, "gap"), (1.0, 2.0)]) == (
        BadSegment(0.0, 2.0, "gap"),
    )


def test_normalize_bad_segments_keeps_reason_when_first_segment_has_none():
    assert normalize_bad_segments([(0.0, 1.0), (0.5, 2.0, "gap")]) == (
        BadSegment(0.0, 2.0, "gap"),
    )


def test_normalize_bad_segments_rejects_negative_start():
    with pytest.raises(ValueError, match="non-negative"):
        normalize_bad_segments([(-0.1, 1.0)])


def test_compute_differential_channel_uses_physical_values():
    positive = _signal([10.0, -4.0, 0.0, 7.0], 4, "A")
    negative = _signal([1.0, 4.0, 0.0, 20.0], 4, "REF")
    channel = compute_differential_channel(positive, negative, label="A-REF")
    assert channel.sampling_frequency == 4.0
    assert channel.sources == ("A", "REF")
    assert channel.physical_dimension == "uV"
    np.testing.assert_allclose(
        channel.data, [9.0, -8.0, 0.0, -13.0], atol=1e-2
    )
    assert channel.available_mask.all()
    assert channel.zero_mask.tolist() == [False, False, True, False]
    assert channel.unavailable_intervals() == ()
    assert channel.available_intervals() == ((0.0, 1.0),)


def test_compute_differential_channel_applies_unit_conversion():
    positive = _signal([20.0, 40.0], 2, "A", physical_range=(-100.0, 100.0))
    negative = _signal([5.0, 25.0], 2, "REF", physical_range=(-50.0, 50.0))
    channel = compute_differential_channel(positive, negative)
    np.testing.assert_allclose(channel.data, [15.0, 15.0], atol=1e-2)


def test_compute_differential_channel_rejects_dimension_mismatch():
    positive = _signal([1.0, 2.0], 2, "A", physical_dimension="uV")
    negative = _signal([1.0, 2.0], 2, "REF", physical_dimension="mV")
    with pytest.raises(ValueError, match="different physical dimensions"):
        compute_differential_channel(positive, negative)


def test_compute_differential_channel_rejects_same_signal():
    signal = _signal([1.0, 2.0], 2, "A")
    with pytest.raises(ValueError, match="different signals"):
        compute_differential_channel(signal, signal)


def test_compute_differential_channel_rejects_different_formats():
    positive = _EdfSignal(np.array([1.0, 2.0]), 2, label="A")
    negative = _BdfSignal(np.array([1.0, 2.0]), 2, label="REF")
    with pytest.raises(ValueError, match="same format"):
        compute_differential_channel(positive, negative)


def test_differential_grid_aligns_sources_on_lcm_time_grid():
    positive = _signal([10.0, 20.0], 2, "A")
    negative = _signal([1.0, 2.0, 3.0, 4.0], 4, "REF")
    channel = compute_differential_channel(positive, negative)
    assert channel.sampling_frequency == 4.0
    assert channel.available_mask.tolist() == [True, False, True, False]
    np.testing.assert_allclose(channel.data[channel.available_mask], [9.0, 17.0], atol=1e-2)
    assert channel.source_sample_index("A", 2) == 1
    assert channel.source_sample_index("REF", 2) == 2


def test_source_sample_index_rejects_unknown_source():
    positive = _signal([1.0, 2.0], 2, "A")
    negative = _signal([1.0, 2.0], 2, "REF")
    channel = compute_differential_channel(positive, negative)
    with pytest.raises(ValueError, match="not a source"):
        channel.source_sample_index("X", 0)


def test_bad_segment_makes_derived_samples_unavailable():
    positive = _signal([10.0, 10.0, 10.0, 10.0], 4, "A")
    negative = _signal([1.0, 2.0, 3.0, 4.0], 4, "REF")
    channel = compute_differential_channel(positive, negative)
    set_bad_segments(negative, [(0.25, 0.75)])
    channel = compute_differential_channel(positive, negative)
    assert channel.available_mask.tolist() == [True, False, False, True]
    assert np.isnan(channel.data[1:3]).all()
    assert channel.unavailable_intervals() == ((0.25, 0.75),)
    assert channel.source_available_masks["REF"].tolist() == [
        True,
        False,
        False,
        True,
    ]


def test_bad_segment_beyond_signal_end_is_clipped():
    positive = _signal([10.0, 10.0], 2, "A")
    negative = _signal([1.0, 2.0], 2, "REF")
    set_bad_segments(negative, [(5.0, 6.0)], recording=_recording(positive, negative))
    channel = compute_differential_channel(positive, negative)
    assert channel.available_mask.all()


def test_bad_segment_snaps_fractional_boundaries_outward_to_sample_grid():
    positive = _signal([10.0, 10.0, 10.0, 10.0, 10.0, 10.0], 3, "A")
    negative = _signal([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 3, "REF")
    set_bad_segments(negative, [(0.1, 0.4)])
    channel = compute_differential_channel(positive, negative)
    assert channel.available_mask.tolist() == [
        True,
        False,
        True,
        True,
        True,
        True,
    ]


def test_true_zero_is_kept_separate_from_missing_data():
    positive = _signal([0.0, 10.0], 2, "A")
    negative = _signal([0.0, 4.0], 2, "REF")
    channel = compute_differential_channel(positive, negative)
    assert channel.available_mask.all()
    assert channel.zero_mask.tolist() == [True, False]
    assert channel.data[0] == 0.0


def test_saturation_is_detected_at_digital_rail():
    rail = _Context.digital_range[1]
    dtype = EdfSignal._digital_dtype
    positive = EdfSignal.from_digital(
        np.array([rail, 0, 0, 0], dtype=dtype),
        4,
        label="A",
        physical_range=(-100.0, 100.0),
        physical_dimension="uV",
    )
    negative = _signal([0.0, 0.0, 0.0, 0.0], 4, "REF")
    channel = compute_differential_channel(positive, negative)
    assert channel.saturation_mask.tolist() == [True, False, False, False]
    assert channel.available_mask.tolist() == [False, True, True, True]
    channel_without_saturation = compute_differential_channel(
        positive, negative, detect_saturation=False
    )
    assert channel_without_saturation.available_mask.all()
    assert not channel_without_saturation.saturation_mask.any()


def test_derive_differential_channels_registers_channels_and_report():
    positive = _signal([10.0, 20.0, 30.0, 40.0], 4, "A")
    negative = _signal([1.0, 2.0, 3.0, 4.0], 4, "REF")
    recording = _recording(positive, negative)
    result = derive_differential_channels(
        recording,
        [DerivedChannelSpec("A-REF", "A", "REF", physical_dimension="uV")],
    )
    channel = result.channels[0]
    assert channel.label == "A-REF"
    assert get_derived_channel(recording, "A-REF") is channel
    report = result.report.channels[0]
    assert report.num_samples == 4
    assert report.num_available == 4
    assert report.num_unavailable == 0
    assert report.num_bad_segment == 0
    assert report.num_saturation == 0
    assert report.num_zero == 0
    assert isinstance(recording, Edf)


def test_derive_rejects_unknown_source_label():
    positive = _signal([1.0, 2.0], 2, "A")
    negative = _signal([1.0, 2.0], 2, "REF")
    recording = _recording(positive, negative)
    with pytest.raises(ValueError, match="Unknown source label"):
        derive_differential_channels(
            recording, [DerivedChannelSpec("d", "A", "MISSING")]
        )


def test_derive_rejects_identical_source_labels():
    positive = _signal([1.0, 2.0], 2, "A")
    negative = _signal([1.0, 2.0], 2, "REF")
    recording = _recording(positive, negative)
    with pytest.raises(ValueError, match="two different source signals"):
        derive_differential_channels(recording, [DerivedChannelSpec("d", "A", "A")])


def test_derive_rejects_duplicate_derived_label():
    positive = _signal([1.0, 2.0], 2, "A")
    negative = _signal([1.0, 2.0], 2, "REF")
    recording = _recording(positive, negative)
    spec = DerivedChannelSpec("d", "A", "REF")
    derive_differential_channels(recording, [spec])
    with pytest.raises(ValueError, match="already exists"):
        derive_differential_channels(recording, [spec])


def test_derive_rejects_ambiguous_source_labels():
    first = _signal([1.0, 2.0], 2, "X")
    second = _signal([3.0, 4.0], 2, "X")
    recording = _recording(first, second)
    with pytest.raises(ValueError, match="ambiguous labels"):
        derive_differential_channels(
            recording, [DerivedChannelSpec("d", "X", "X")]
        )


def test_get_derived_channel_without_registry_raises():
    recording = _recording(_signal([1.0, 2.0], 2, "A"))
    with pytest.raises(ValueError, match="No derived channels"):
        get_derived_channel(recording, "d")


def test_get_derived_channel_unknown_label_raises():
    positive = _signal([1.0, 2.0], 2, "A")
    negative = _signal([1.0, 2.0], 2, "REF")
    recording = _recording(positive, negative)
    derive_differential_channels(
        recording, [DerivedChannelSpec("d", "A", "REF")]
    )
    with pytest.raises(ValueError, match="No derived channel with label"):
        get_derived_channel(recording, "missing")


def test_set_bad_segments_withdraws_restores_and_lists_intervals():
    positive = _signal([10.0, 10.0, 10.0, 10.0], 4, "A")
    negative = _signal([1.0, 2.0, 3.0, 4.0], 4, "REF")
    recording = _recording(positive, negative)
    derive_differential_channels(
        recording, [DerivedChannelSpec("A-REF", "A", "REF")]
    )
    report = set_bad_segments(negative, [(0.25, 0.5)], recording=recording)
    assert report.signal_label == "REF"
    assert report.bad_segments == (BadSegment(0.25, 0.5),)
    assert report.withdrawn_intervals == {"A-REF": ((0.25, 0.5),)}
    assert report.restored_intervals == {"A-REF": ()}
    assert report.recomputed_intervals == {"A-REF": ((0.25, 0.5),)}
    assert report.unaffected_channels == ()
    assert not get_derived_channel(recording, "A-REF").available_mask[1]

    restored = set_bad_segments(negative, [], recording=recording)
    assert restored.restored_intervals == {"A-REF": ((0.25, 0.5),)}
    assert restored.withdrawn_intervals == {"A-REF": ()}
    assert get_derived_channel(recording, "A-REF").available_mask.all()


def test_shared_reference_bad_segment_affects_all_dependent_channels():
    reference = _signal([1.0, 1.0, 1.0, 1.0], 4, "REF")
    first = _signal([2.0, 2.0, 2.0, 2.0], 4, "A")
    second = _signal([3.0, 3.0, 3.0, 3.0], 4, "B")
    independent = _signal([9.0, 9.0, 9.0, 9.0], 4, "C")
    recording = _recording(first, second, reference, independent)
    derive_differential_channels(
        recording,
        [
            DerivedChannelSpec("A-REF", "A", "REF"),
            DerivedChannelSpec("B-REF", "B", "REF"),
            DerivedChannelSpec("C-only", "C", "A"),
        ],
    )
    report = set_bad_segments(reference, [(0.0, 0.25)], recording=recording)
    assert report.withdrawn_intervals == {
        "A-REF": ((0.0, 0.25),),
        "B-REF": ((0.0, 0.25),),
    }
    assert report.unaffected_channels == ("C-only",)
    independent_channel = get_derived_channel(recording, "C-only")
    assert independent_channel.available_mask.all()
    assert independent_channel.source_available_masks["C"].all()


def test_set_bad_segments_reports_value_changes_of_remaining_samples():
    positive = _signal([10.0, 10.0, 10.0, 10.0], 4, "A")
    negative = _signal([1.0, 2.0, 3.0, 4.0], 4, "REF")
    recording = _recording(positive, negative)
    derive_differential_channels(
        recording, [DerivedChannelSpec("A-REF", "A", "REF")]
    )
    positive.update_data(
        np.array([20.0, 20.0, 20.0, 20.0], dtype=np.float64),
        keep_physical_range=True,
    )
    report = set_bad_segments(negative, [], recording=recording)
    assert report.value_changed_intervals == {"A-REF": ((0.0, 1.0),)}
    assert report.recomputed_intervals == {"A-REF": ((0.0, 1.0),)}
    np.testing.assert_allclose(
        get_derived_channel(recording, "A-REF").data,
        [19.0, 18.0, 17.0, 16.0],
        atol=1e-2,
    )


def test_set_bad_segments_updates_all_registries_by_default():
    shared = _signal([1.0, 1.0, 1.0, 1.0], 4, "REF")
    partner_one = _signal([2.0, 2.0, 2.0, 2.0], 4, "A")
    partner_two = _signal([3.0, 3.0, 3.0, 3.0], 4, "B")
    recording_one = _recording(shared, partner_one)
    recording_two = _recording(shared, partner_two)
    derive_differential_channels(
        recording_one, [DerivedChannelSpec("A-REF", "REF", "A")]
    )
    derive_differential_channels(
        recording_two, [DerivedChannelSpec("B-REF", "REF", "B")]
    )
    report = set_bad_segments(shared, [(0.0, 0.25)])
    assert set(report.withdrawn_intervals) == {"A-REF", "B-REF"}
    assert not get_derived_channel(recording_one, "A-REF").available_mask[0]
    assert not get_derived_channel(recording_two, "B-REF").available_mask[0]


def test_set_bad_segments_with_unregistered_recording_is_noop_for_channels():
    signal = _signal([1.0, 2.0], 2, "A")
    recording = _recording(signal, _signal([0.0, 1.0], 2, "REF"))
    report = set_bad_segments(signal, [(0.0, 0.5)], recording=recording)
    assert report.withdrawn_intervals == {}
    assert report.unaffected_channels == ()
    assert signal.bad_segments == (BadSegment(0.0, 0.5),)


def test_derived_channel_to_signal_materializes_values():
    positive = _signal([10.0, 10.0, 10.0, 10.0], 4, "A")
    negative = _signal([1.0, 2.0, 3.0, 4.0], 4, "REF")
    set_bad_segments(negative, [(0.25, 0.5)], recording=_recording(positive, negative))
    channel = compute_differential_channel(positive, negative, label="A-REF")
    signal = channel.to_signal(fill_value=-99.0, physical_range=(-100.0, 100.0))
    assert isinstance(signal, EdfSignal)
    assert signal.label == "A-REF"
    assert signal.sampling_frequency == 4.0
    assert signal.physical_dimension == "uV"
    assert signal.data[1] == pytest.approx(-99.0, abs=1e-2)
    assert signal.data[0] == pytest.approx(9.0, abs=1e-2)


def test_derivation_registry_tracks_labels_and_provenance():
    positive = _signal([1.0, 2.0], 2, "A")
    negative = _signal([1.0, 2.0], 2, "REF")
    recording = _recording(positive, negative)
    derive_differential_channels(
        recording, [DerivedChannelSpec("d", "A", "REF")]
    )
    registry = _registries[recording]
    assert registry.labels == ("d",)
    assert registry.entries_using(negative)[0].spec.label == "d"
    assert registry.entries_using(positive)[0].channel.sources == ("A", "REF")


def test_non_integer_sampling_frequencies_align_on_lcm_grid():
    positive = _signal([10.0, 20.0, 30.0, 40.0, 50.0], 2.5, "A")
    negative = _signal([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 5, "REF")
    channel = compute_differential_channel(positive, negative)
    assert channel.sampling_frequency == 5.0
    assert channel.available_mask.tolist() == [
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]
