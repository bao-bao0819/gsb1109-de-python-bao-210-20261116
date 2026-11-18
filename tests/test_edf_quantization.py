from __future__ import annotations

import math

import numpy as np
import pytest

from edfio import Edf, EdfSignal
from edfio._lazy_loading import LazyLoader
from edfio.edf_quantization import (
    DigitalRecovery,
    QuantizationRange,
    RangePreview,
    SourceDeclaration,
    _as_range,
    _PreviewAccumulator,
    _resolve_source,
    preview_requantization,
    recover_digital_codes,
    requantize,
)

PHYSICAL = QuantizationRange(-32768, 32767, 0.0, 100.0)
COARSE = QuantizationRange(-2048, 2047, -500.0, 500.0)


def test_quantization_range_mapping_round_trip():
    codes = np.array([PHYSICAL.digital_min, -1, 0, 1, 1234, PHYSICAL.digital_max])
    physical = PHYSICAL.codes_to_physical(codes)
    np.testing.assert_allclose(PHYSICAL.physical_to_digital(physical), codes)
    assert PHYSICAL.num_codes == PHYSICAL.digital_max - PHYSICAL.digital_min + 1
    assert PHYSICAL.resolution == PHYSICAL.gain
    assert tuple(PHYSICAL) == (-32768, 32767, 0.0, 100.0)


@pytest.mark.parametrize(
    "ranges",
    [
        (0, 0, 0.0, 1.0),
        (0, 10, 1.0, 1.0),
        (0, 10, math.inf, 1.0),
    ],
)
def test_quantization_range_rejects_invalid_ranges(ranges):
    with pytest.raises(ValueError, match=r"must be|finite"):
        QuantizationRange(*ranges)


def test_quantization_range_quantize_marks_saturation_separately():
    codes, low, high = QuantizationRange(-100, 100, 0.0, 10.0).quantize(
        np.array([-1.0, 0.0, 5.0, 10.0, 11.0])
    )
    assert codes.tolist() == [-100, -100, 0, 100, 100]
    assert low.tolist() == [True, False, False, False, False]
    assert high.tolist() == [False, False, False, False, True]


def test_recover_digital_codes_from_raw_codes_is_unique():
    codes = np.array([-32768, -1, 0, 1, 1234, 32767], dtype=np.int16)
    source = SourceDeclaration(PHYSICAL, kind="digital", quantized=False)
    recovery = recover_digital_codes(codes, PHYSICAL, source=source)
    assert recovery.unique.all()
    assert recovery.ranges() == [(int(c), int(c)) for c in codes]
    assert recovery.num_unique == len(codes)
    assert recovery.num_ambiguous == 0
    assert np.array_equal(recovery.codes, codes)


def test_recover_digital_codes_from_exact_same_range_physical_view_is_unique():
    codes = np.array([-32768, -1, 0, 1, 1234, 32767], dtype=np.int16)
    physical = PHYSICAL.codes_to_physical(codes)
    source = SourceDeclaration(PHYSICAL, kind="physical", quantized=True)
    recovery = recover_digital_codes(physical, PHYSICAL, source=source)
    assert recovery.unique.all()
    assert recovery.ranges() == [(int(c), int(c)) for c in codes]


def test_recover_digital_codes_from_coarse_view_gives_ranges():
    coarse_range = QuantizationRange(-2048, 2047, 0.0, 100.0)
    codes = np.array([-2048, -1000, -1, 0, 1, 1000, 2047], dtype=np.int32)
    coarse = coarse_range.codes_to_physical(codes)
    source = SourceDeclaration(coarse_range, kind="physical", quantized=True)
    recovery = recover_digital_codes(coarse, PHYSICAL, source=source)
    assert recovery.num_ambiguous == len(codes)
    assert (recovery.widths > 1).all()
    lower = PHYSICAL.codes_to_physical(recovery.code_low)
    upper = PHYSICAL.codes_to_physical(recovery.code_high)
    assert np.all(lower <= coarse + recovery.uncertainty + 1e-9)
    assert np.all(upper >= coarse - recovery.uncertainty - 1e-9)


def test_recover_digital_codes_halfway_between_codes_is_not_unique():
    target = QuantizationRange(0, 10, 0.0, 10.0)
    source = SourceDeclaration(target, kind="physical", quantized=False)
    recovery = recover_digital_codes(np.array([0.5]), target, source=source)
    assert recovery.ranges() == [(0, 1)]


def test_recover_digital_codes_with_error_bound_widens_interval():
    target = QuantizationRange(0, 10, 0.0, 10.0)
    source = SourceDeclaration(target, kind="physical", quantized=False)
    recovery = recover_digital_codes(
        np.array([0.0]), target, source=source, error_bound=0.5
    )
    assert recovery.ranges() == [(0, 1)]
    assert recovery.total_error_bound == pytest.approx(0.5)


def test_recover_digital_codes_marks_source_and_target_saturation_separately():
    target = QuantizationRange(0, 100, 0.0, 10.0)
    source = SourceDeclaration(
        target, kind="physical", quantized=False, saturated_low=True
    )
    values = np.array([-1.0, 5.0, 11.0])
    recovery = recover_digital_codes(values, target, source=source)
    assert recovery.source_saturated_low.tolist() == [True, False, False]
    assert recovery.saturated_low.tolist() == [True, False, False]
    assert recovery.saturated_high.tolist() == [False, False, True]
    assert recovery.num_may_saturate_low == 1
    assert recovery.num_may_saturate_high == 1


@pytest.mark.parametrize("bound", [-1.0, math.inf, math.nan])
def test_recover_digital_codes_rejects_invalid_error_bounds(bound):
    with pytest.raises(ValueError, match="Error bound"):
        recover_digital_codes(
            np.array([1.0]), QuantizationRange(0, 10, 0.0, 10.0), error_bound=bound
        )


def test_recover_digital_codes_rejects_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        recover_digital_codes(np.array([np.nan]), QuantizationRange(0, 10, 0.0, 10.0))


def test_recover_digital_codes_rejects_out_of_range_codes():
    source = SourceDeclaration(QuantizationRange(0, 10, 0.0, 10.0), kind="digital")
    with pytest.raises(ValueError, match="out of source digital range"):
        recover_digital_codes(
            np.array([-1], dtype=np.int16), source.range, source=source
        )


def test_recover_digital_codes_rejects_non_integer_digital_source():
    source = SourceDeclaration(QuantizationRange(0, 10, 0.0, 10.0), kind="digital")
    with pytest.raises(ValueError, match="must be integers"):
        recover_digital_codes(np.array([1.0]), source.range, source=source)


def test_recover_digital_codes_accepts_tuple_range():
    recovery = recover_digital_codes(
        np.array([0], dtype=np.int16), (-100, 100, 0.0, 10.0)
    )
    assert recovery.target == QuantizationRange(-100, 100, 0.0, 10.0)


def test_source_declaration_defaults_and_bounds():
    assert SourceDeclaration(PHYSICAL).quantized is False
    assert SourceDeclaration(PHYSICAL, kind="physical").quantized is True
    assert SourceDeclaration(PHYSICAL, error_bound=0.5).previous_error_bound == 0.5
    with pytest.raises(ValueError, match="kind"):
        SourceDeclaration(PHYSICAL, kind="other")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative"):
        SourceDeclaration(PHYSICAL, error_bound=-1)


def test_source_declaration_from_signal():
    signal = EdfSignal(np.arange(10), 1, physical_range=(0, 100))
    source = SourceDeclaration.from_signal(signal)
    assert source.range == QuantizationRange.from_signal(signal)
    assert source.kind == "digital"
    assert source.previous_error_bound == 0


def test_requantize_keeps_previous_error_on_further_conversion():
    codes = np.array([0, 10, 20], dtype=np.int16)
    signal = EdfSignal(np.arange(3), 1, physical_range=(0, 10), digital_range=(0, 100))
    source = SourceDeclaration.from_signal(signal, kind="digital", quantized=False)
    target = QuantizationRange(-2048, 2047, -100, 100)
    view = requantize(codes, source, target)
    assert view.declaration.error_bound == pytest.approx(
        source.previous_error_bound, rel=1e-6
    )
    assert view.declaration.previous_error_bound == pytest.approx(
        source.previous_error_bound + target.resolution / 2, rel=1e-6
    )
    assert view.num_ambiguous == 1
    next_target = QuantizationRange(-1024, 1023, -100, 100)
    again = requantize(view.values, view.declaration, next_target)
    assert again.declaration.error_bound == pytest.approx(
        view.declaration.previous_error_bound, rel=1e-6
    )
    assert again.declaration.previous_error_bound == pytest.approx(
        view.declaration.previous_error_bound + next_target.resolution / 2,
        rel=1e-6,
    )


def test_requantize_same_grid_preserves_error_without_adding_again():
    target = QuantizationRange(-2048, 2047, -500.0, 500.0)
    source = SourceDeclaration(
        target, kind="physical", quantized=True, error_bound=0.25
    )
    view = requantize(np.array([0.0, 100.0]), source, target)
    assert view.declaration.error_bound == pytest.approx(0.25)
    assert view.declaration.previous_error_bound == pytest.approx(
        0.25 + target.resolution / 2
    )
    again = requantize(view.values, view.declaration, target)
    assert again.declaration.error_bound == pytest.approx(0.25)


def test_requantize_exact_codes_do_not_gain_phantom_error():
    target = QuantizationRange(0, 100, 0.0, 10.0)
    source = SourceDeclaration(target, kind="digital", quantized=False)
    view = requantize(np.array([0, 50, 100], dtype=np.int16), source, target)
    assert view.declaration.previous_error_bound == 0.0
    again = requantize(view.values, view.declaration, target)
    assert again.declaration.previous_error_bound == 0.0
    assert again.unique.all()


def test_next_source_from_preview_carries_total_error():
    target = QuantizationRange(0, 10, 0.0, 10.0)
    coarse = QuantizationRange(0, 5, 0.0, 10.0)
    source = SourceDeclaration(coarse, kind="physical", quantized=True)
    preview = preview_requantization([np.array([2.0, 8.0])], source, target)
    next_source = preview.next_source()
    assert next_source.previous_error_bound == pytest.approx(preview.total_error_bound)
    assert next_source.quantized is True


def test_preview_requantization_streams_chunks_and_aggregates():
    codes = np.array([0, 10, 20, 30], dtype=np.int16)
    source = SourceDeclaration(PHYSICAL, kind="digital", quantized=False)
    preview = preview_requantization([codes[:2], codes[2:]], source, COARSE)
    assert preview.num_samples == 4
    assert preview.num_chunks == 2
    assert preview.num_unique + preview.num_ambiguous == 4
    assert preview.observed_min == pytest.approx(
        PHYSICAL.codes_to_physical(codes).min()
    )


def test_preview_requantization_empty_chunks():
    source = SourceDeclaration(PHYSICAL, kind="digital")
    preview = preview_requantization([], source, COARSE)
    assert preview.num_samples == 0
    assert preview.min_bound == math.inf
    assert preview.max_bound == -math.inf
    assert not preview.is_unique


def test_preview_requantization_separates_saturation_from_true_extrema():
    signal = EdfSignal(
        np.array([0.0, 2.0, 5.0, 8.0, 10.0]),
        1,
        physical_range=(0, 10),
        digital_range=(0, 100),
    )
    preview = signal.preview_range(physical_range=(2, 8))
    assert preview.num_saturated_low == 1
    assert preview.num_saturated_high == 1
    assert preview.observed_min == 0.0
    assert preview.observed_max == 10.0
    assert preview.unsaturated_min == 2.0
    assert preview.unsaturated_max == 8.0
    assert preview.true_min == 0.0
    assert preview.true_max == 10.0


def test_preview_requantization_source_saturation_makes_bounds_infinite():
    target = QuantizationRange(0, 100, 0.0, 10.0)
    source = SourceDeclaration(target, kind="physical", saturated_high=True)
    preview = preview_requantization([np.array([0.0, 10.0])], source, target)
    assert preview.num_source_saturated_high == 1
    assert preview.max_bound == math.inf
    assert preview.true_max == 0.0


def test_preview_requantization_error_retained():
    target = QuantizationRange(0, 100, 0.0, 10.0)
    source = SourceDeclaration(target, kind="physical", quantized=True)
    preview = preview_requantization([np.array([5.0])], source, target)
    assert preview.previous_error_bound == pytest.approx(5 / 100)
    assert preview.new_quantization_error_bound == 0.0
    assert preview.total_error_bound == pytest.approx(5 / 100)


def test_signal_recover_digital_codes_default_uses_raw_codes():
    signal = EdfSignal(np.array([0.0, 5.0, 10.0]), 1, physical_range=(0, 10))
    recovery = signal.recover_digital_codes()
    assert recovery.unique.all()
    assert recovery.ranges() == [(int(code), int(code)) for code in signal.digital]


def test_signal_recover_digital_codes_from_physical_view():
    signal = EdfSignal(
        np.array([0.0, 5.0, 10.0]),
        1,
        physical_range=(0, 10),
        digital_range=(0, 100),
    )
    target = QuantizationRange(-100, 100, 0.0, 10.0)
    source = SourceDeclaration.from_signal(signal, kind="physical")
    recovery = signal.recover_digital_codes(signal.data, target=target, source=source)
    assert recovery.num_ambiguous == len(signal.digital)


def test_signal_preview_range_on_materialized_data():
    signal = EdfSignal(np.arange(10.0), 1, physical_range=(0, 9))
    preview = signal.preview_range(physical_range=(-1, 10), chunk_size=3)
    assert isinstance(preview, RangePreview)
    assert preview.num_samples == 10
    assert preview.num_chunks == 4


def test_signal_preview_range_without_data_raises():
    signal = EdfSignal._from_raw_header(
        1,
        label=b"",
        transducer_type=b"",
        physical_dimension=b"",
        physical_min=b"0",
        physical_max=b"10",
        digital_min=b"0",
        digital_max=b"100",
        prefiltering=b"",
        samples_per_data_record=b"1",
        reserved=b"",
    )
    with pytest.raises(ValueError, match="Signal data not set"):
        signal.preview_range()


def test_signal_preview_range_rejects_invalid_chunk_size():
    signal = EdfSignal(np.arange(10.0), 1)
    with pytest.raises(ValueError, match="Chunk size"):
        signal.preview_range(chunk_size=0)


@pytest.mark.edf
def test_edf_preview_ranges_returns_mapping():
    edf = Edf(
        [
            EdfSignal(np.arange(10.0), 1, label="A", physical_range=(0, 9)),
            EdfSignal(np.arange(10.0), 1, label="B", physical_range=(0, 9)),
        ]
    )
    previews = edf.preview_ranges(physical_range=(-1, 10))
    assert set(previews) == {"A", "B"}
    assert all(p.num_samples == 10 for p in previews.values())


@pytest.mark.edf
def test_edf_preview_ranges_by_index_and_label():
    edf = Edf([EdfSignal(np.arange(10.0), 1, label="A")])
    preview = edf.preview_ranges(physical_range={0: (-1, 10)}, error_bound={"A": 0.25})
    assert preview["A"].num_samples == 10
    assert preview["A"].previous_error_bound >= 0.25


@pytest.mark.edf
def test_edf_preview_ranges_missing_signal_raises():
    edf = Edf([EdfSignal(np.arange(10.0), 1, label="A")])
    with pytest.raises(ValueError, match="No physical_range"):
        edf.preview_ranges(physical_range={"B": (-1, 10)})


@pytest.mark.edf
def test_edf_preview_ranges_ambiguous_labels_get_indexed_keys():
    edf = Edf(
        [
            EdfSignal(np.arange(10.0), 1, label="A"),
            EdfSignal(np.arange(10.0), 1, label="A"),
        ]
    )
    previews = edf.preview_ranges()
    assert set(previews) == {"A#0", "A#1"}


def test_quantized_view_is_read_only():
    signal = EdfSignal(np.arange(10.0), 1, physical_range=(0, 9))
    view = signal.calibrated_view()
    with pytest.raises(ValueError, match="read-only"):
        view.values[0] = 1


def test_recovery_length_and_saturation_helpers():
    target = QuantizationRange(0, 100, 0.0, 10.0)
    source = SourceDeclaration(target, kind="physical", quantized=False)
    recovery = recover_digital_codes(np.array([-1.0, 5.0, 11.0]), target, source=source)
    assert len(recovery) == 3
    assert isinstance(recovery, DigitalRecovery)
    assert recovery.same_grid
    assert recovery.new_quantization_error_bound == 0.0
    assert recovery.quantization_error_bound == pytest.approx(0.05)


def test_source_declaration_from_signal_rejects_invalid_kind():
    signal = EdfSignal(np.arange(10.0), 1)
    with pytest.raises(ValueError, match="kind"):
        SourceDeclaration.from_signal(signal, kind="other")  # type: ignore[arg-type]


def test_recovery_alias_properties():
    target = QuantizationRange(0, 10, 0.0, 10.0)
    source = SourceDeclaration(target, kind="physical", quantized=False)
    recovery = recover_digital_codes(np.array([0.0, 5.0]), target, source=source)
    np.testing.assert_array_equal(recovery.low, recovery.code_low)
    np.testing.assert_array_equal(recovery.high, recovery.code_high)
    np.testing.assert_array_equal(
        recovery.widths, recovery.code_high - recovery.code_low + 1
    )


def test_recovery_next_source_carries_error():
    target = QuantizationRange(0, 10, 0.0, 10.0)
    source = SourceDeclaration(target, kind="physical", quantized=False)
    recovery = recover_digital_codes(
        np.array([0.0]), target, source=source, error_bound=1.0
    )
    next_source = recovery.next_source()
    assert next_source.range == target
    assert next_source.quantized is True
    assert next_source.error_bound == pytest.approx(
        recovery.total_error_bound - target.resolution / 2
    )


def test_preview_alias_properties_and_next_source():
    target = QuantizationRange(0, 10, 0.0, 10.0)
    source = SourceDeclaration(target, kind="physical", quantized=False)
    preview = preview_requantization([np.array([0.0, 10.0])], source, target)
    assert preview.num_unique == preview.num_unique_codes
    assert preview.num_ambiguous == preview.num_ambiguous_codes
    assert preview.saturated_low == preview.num_saturated_low
    assert preview.saturated_high == preview.num_saturated_high
    assert preview.source_saturated_low == preview.num_source_saturated_low
    assert preview.source_saturated_high == preview.num_source_saturated_high
    assert preview.data_min == preview.observed_min
    assert preview.data_max == preview.observed_max
    assert preview.has_saturation is False
    next_source = preview.next_source()
    assert next_source.range == target
    assert next_source.saturated_low is False


def test_preview_suggested_range_none_without_samples():
    target = QuantizationRange(0, 10, 0.0, 10.0)
    preview = preview_requantization([], target, target)
    assert preview.suggested_physical_range is None


def test_quantized_view_properties_and_next_source():
    target = QuantizationRange(0, 100, 0.0, 10.0)
    source = SourceDeclaration(target, kind="physical", quantized=False)
    view = requantize(np.array([-1.0, 5.0, 11.0]), source, target)
    assert view.num_saturated_low == 1
    assert view.num_saturated_high == 1
    assert view.unique.all()
    assert view.next_source() is view.declaration


@pytest.mark.parametrize(
    "value",
    [None, object(), (0.0, 10.0)],
)
def test_as_range_rejects_invalid_values(value):
    with pytest.raises(TypeError):
        _as_range(value)


def test_recover_digital_codes_with_integer_physical_source():
    target = QuantizationRange(0, 10, 0.0, 10.0)
    source = SourceDeclaration(target, kind="physical", quantized=False)
    recovery = recover_digital_codes(
        np.array([0, 5, 10], dtype=np.int64), target, source=source
    )
    assert recovery.unique.all()


def test_recover_digital_codes_with_mask_arrays():
    target = QuantizationRange(0, 10, 0.0, 10.0)
    source = SourceDeclaration(target, kind="physical", quantized=False)
    recovery = recover_digital_codes(
        np.array([0.0, 5.0, 10.0]),
        target,
        source=source,
        source_saturated_low=np.array([True, False, False]),
        source_saturated_high=np.array([False, False, True]),
    )
    assert recovery.source_saturated_low.tolist() == [True, False, False]
    assert recovery.source_saturated_high.tolist() == [False, False, True]


def test_preview_plain_range_infers_physical_source():
    target = QuantizationRange(0, 10, 0.0, 10.0)
    preview = preview_requantization([np.array([0.0, 5.0])], target, target)
    assert preview.source.kind == "physical"
    assert preview.source.quantized is True


def test_preview_plain_range_infers_digital_source():
    target = QuantizationRange(0, 10, 0.0, 10.0)
    preview = preview_requantization([np.array([0, 5], dtype=np.int16)], target, target)
    assert preview.source.kind == "digital"


def test_requantize_plain_range_infers_kind():
    target = QuantizationRange(0, 10, 0.0, 10.0)
    digital_view = requantize(np.array([0, 5], dtype=np.int16), target, target)
    physical_view = requantize(np.array([0.0, 5.0]), target, target)
    assert digital_view.declaration.kind == "physical"
    assert physical_view.declaration.kind == "physical"


def test_preview_source_saturation_bounds():
    target = QuantizationRange(0, 10, 0.0, 10.0)
    source = SourceDeclaration(
        target, kind="physical", saturated_low=True, saturated_high=True
    )
    preview = preview_requantization([np.array([0.0, 5.0, 10.0])], source, target)
    assert preview.min_bound == -math.inf
    assert preview.max_bound == math.inf


def test_preview_skips_empty_chunks():
    target = QuantizationRange(0, 10, 0.0, 10.0)
    source = SourceDeclaration(target, kind="physical", quantized=False)
    preview = preview_requantization(
        [np.array([]), np.array([5.0]), np.array([])], source, target
    )
    assert preview.num_samples == 1
    assert preview.num_chunks == 1


def test_signal_preview_range_physical_source_kind():
    signal = EdfSignal(
        np.array([0.0, 5.0, 10.0]), 1, physical_range=(0, 10), digital_range=(0, 100)
    )
    preview = signal.preview_range(source_kind="physical")
    assert preview.num_samples == 3


def test_signal_recover_with_plain_range_source_uses_raw_codes():
    signal = EdfSignal(np.array([0.0, 5.0]), 1, physical_range=(0, 10))
    recovery = signal.recover_digital_codes(
        source=QuantizationRange.from_signal(signal)
    )
    assert recovery.unique.all()


def test_signal_calibrated_view_uses_supplied_ranges():
    signal = EdfSignal(
        np.array([0.0, 5.0, 10.0]), 1, physical_range=(0, 10), digital_range=(0, 100)
    )
    view = signal.calibrated_view(physical_range=(-20, 20), digital_range=(-200, 200))
    assert view.declaration.range == QuantizationRange(-200, 200, -20, 20)


def test_signal_preview_range_uses_explicit_source():
    signal = EdfSignal(
        np.array([0.0, 5.0, 10.0]), 1, physical_range=(0, 10), digital_range=(0, 100)
    )
    source = SourceDeclaration.from_signal(signal, kind="physical")
    preview = signal.preview_range(source=source)
    assert preview.source == source


def test_lazy_signal_preview_reads_chunks(monkeypatch):
    signal = EdfSignal._from_raw_header(
        3,
        label=b"SpO2",
        transducer_type=b"",
        physical_dimension=b"",
        physical_min=b"0",
        physical_max=b"100",
        digital_min=b"0",
        digital_max=b"999",
        prefiltering=b"",
        samples_per_data_record=b"3",
        reserved=b"",
    )
    buffer = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=np.int16)
    signal._lazy_loader = LazyLoader(buffer, 0, 3)
    loaded: list[tuple[int, int]] = []
    original = signal._lazy_loader.load

    def tracking_load(first=None, last=None):
        loaded.append((first, last))
        return original(first, last)

    monkeypatch.setattr(signal._lazy_loader, "load", tracking_load)
    preview = signal.preview_range(digital_range=(0, 999), chunk_size=3)
    assert preview.num_samples == 12
    assert len(loaded) >= 2
    assert signal._digital is None


def test_edf_preview_ranges_rejects_missing_index():
    edf = Edf([EdfSignal(np.arange(10.0), 1, label="A")])
    with pytest.raises(ValueError, match="No error_bound"):
        edf.preview_ranges(error_bound={5: 0.1})


def test_preview_suggested_range_and_error_bounds():
    target = QuantizationRange(0, 10, 0.0, 10.0)
    coarse = QuantizationRange(0, 100, 0.0, 10.0)
    source = SourceDeclaration(coarse, kind="physical", quantized=True)
    preview = preview_requantization([np.array([2.0, 8.0])], source, target)
    error = preview.total_error_bound
    assert preview.quantization_error_bound == pytest.approx(0.5)
    assert preview.new_quantization_error_bound == pytest.approx(0.5)
    assert preview.suggested_physical_range == pytest.approx((2.0 - error, 8.0 + error))


def test_quantized_view_unique_count():
    target = QuantizationRange(0, 10, 0.0, 10.0)
    source = SourceDeclaration(target, kind="physical", quantized=False)
    view = requantize(np.array([0.0, 5.0]), source, target)
    assert view.num_unique == 2
    assert view.num_ambiguous == 0


def test_recover_digital_codes_with_scalar_saturation_mask():
    target = QuantizationRange(0, 10, 0.0, 10.0)
    source = SourceDeclaration(target, kind="physical", quantized=False)
    recovery = recover_digital_codes(
        np.array([1.0, 2.0]),
        target,
        source=source,
        source_saturated_low=True,
    )
    assert recovery.source_saturated_low.tolist() == [True, True]


def test_resolve_source_requires_source():
    with pytest.raises(ValueError, match="required"):
        _resolve_source(None, np.array([1]))


def test_preview_error_bound_boundaries_are_finite():
    target = QuantizationRange(0, 10, 0.0, 10.0)
    coarse = QuantizationRange(0, 100, 0.0, 10.0)
    source = SourceDeclaration(coarse, kind="physical", quantized=True)
    preview = preview_requantization([np.array([5.0]), np.array([5.0])], source, target)
    error = preview.total_error_bound
    assert preview.min_bound == pytest.approx(5.0 - error)
    assert preview.max_bound == pytest.approx(5.0 + error)


def test_preview_accumulator_ignores_empty_input():
    target = QuantizationRange(0, 10, 0.0, 10.0)
    source = SourceDeclaration(target, kind="physical", quantized=False)
    accumulator = _PreviewAccumulator(source, target)
    accumulator.update(
        recover_digital_codes(np.array([1.0]), target, source=source),
        np.array([], dtype=np.float64),
    )
    assert accumulator.num_samples == 0


def test_preview_with_negative_values_tracks_low_bound_across_chunks():
    target = QuantizationRange(-10, 10, -1.0, 1.0)
    source = SourceDeclaration(target, kind="physical", quantized=False)
    preview = preview_requantization(
        [np.array([-1.0]), np.array([0.5]), np.array([1.0])], source, target
    )
    assert preview.min_bound == pytest.approx(-1.0 - preview.total_error_bound)
    assert preview.max_bound == pytest.approx(1.0 + preview.total_error_bound)


def test_signal_quantization_range_property():
    signal = EdfSignal(
        np.arange(10.0), 1, physical_range=(0, 100), digital_range=(0, 1000)
    )
    assert signal.quantization_range == QuantizationRange(0, 1000, 0, 100)


def test_recovery_is_unique_scalar_property():
    target = QuantizationRange(0, 10, 0.0, 10.0)
    exact_source = SourceDeclaration(target, kind="digital", quantized=False)
    exact = recover_digital_codes(
        np.array([0, 5], dtype=np.int16), target, source=exact_source
    )
    assert exact.is_unique is True
    coarse_view = SourceDeclaration(
        QuantizationRange(0, 5, 0.0, 10.0), kind="physical", quantized=True
    )
    ambiguous = recover_digital_codes(np.array([0.0]), target, source=coarse_view)
    assert ambiguous.is_unique is False
    empty = recover_digital_codes(
        np.array([], dtype=np.int16), target, source=exact_source
    )
    assert empty.is_unique is False
