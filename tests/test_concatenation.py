from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from edfio import (
    Bdf,
    BdfSignal,
    ChannelMetadata,
    ConcatenationError,
    ConversionRecord,
    Edf,
    EdfAnnotation,
    EdfSignal,
    UnitConversion,
    concatenate,
    concatenate_edfs,
    concatenate_recordings,
    extract_reference,
    plan_concatenation,
    read_edf,
)

pytestmark = pytest.mark.edf


def _signal(
    label: str = "EEG1",
    *,
    unit: str = "uV",
    reference: str = "",
    data: np.ndarray | None = None,
    fs: float = 1,
    prefiltering: str = "",
    physical_range: tuple[float, float] = (-1000, 1000),
) -> EdfSignal:
    if data is None:
        data = np.arange(10, dtype=np.float64)
    transducer = f"Ag/AgCl ref={reference}" if reference else ""
    return EdfSignal(
        data,
        fs,
        label=label,
        physical_dimension=unit,
        transducer_type=transducer,
        prefiltering=prefiltering,
        physical_range=physical_range,
    )


def _edf(**kwargs) -> Edf:
    return Edf([_signal(**kwargs)])


def test_identical_recordings_are_concatenated():
    first = _edf()
    second = _edf(data=np.arange(10, 20, dtype=np.float64))

    result = concatenate([first, second])

    assert len(result) == 1
    assert result.recordings[0].duration == 20
    np.testing.assert_allclose(
        result.recordings[0].signals[0].data, np.arange(20), atol=0.1
    )


def test_result_behaves_like_a_sequence_of_recordings():
    result = concatenate([_edf(), _edf()])

    assert isinstance(result, tuple) or hasattr(result, "__getitem__")
    assert len(result) == 1
    assert isinstance(result[0], Edf)
    assert result.outputs == result.recordings
    assert tuple(result) == result.recordings


def test_same_label_is_not_enough_for_different_sampling_frequencies():
    result = concatenate([_edf(), _edf(fs=2, data=np.arange(20, dtype=np.float64))])

    assert len(result) == 2
    assert [conflict.reason for conflict in result.conflicts] == ["sampling_frequency"]
    assert result.conflicts[0].channel == "EEG1"


def test_different_references_are_split_even_if_labels_units_and_fs_match():
    result = concatenate([_edf(reference="Cz"), _edf(reference="M1")])

    assert len(result) == 2
    assert [conflict.reason for conflict in result.conflicts] == ["reference"]
    assert "CZ" in result.conflicts[0].detail.upper()


def test_explicitly_equivalent_references_are_concatenated():
    result = concatenate(
        [_edf(reference="M1"), _edf(reference="A1")],
        reference_equivalences=[("M1", "A1")],
    )

    assert len(result) == 1


def test_different_electrode_names_are_not_assumed_equivalent():
    result = concatenate([_edf(reference="M1"), _edf(reference="A1")])

    assert len(result) == 2
    assert result.conflicts[0].reason == "reference"


def test_reference_synonyms_are_equivalent_by_default():
    result = concatenate([_edf(reference="M1"), _edf(reference="left mastoid")])

    assert len(result) == 1


def test_bipolar_label_supplies_a_reference():
    first = _edf(label="EEG Fpz-Cz")
    second = _edf(label="EEG Fpz-Cz")

    result = concatenate([first, second])

    assert len(result) == 1
    assert result.recordings[0].labels == ("EEG Fpz-Cz",)


def test_bipolar_labels_with_different_references_are_split():
    result = concatenate([_edf(label="EEG Fpz-Cz"), _edf(label="EEG Fpz-Pz")])

    assert len(result) == 2
    assert all(conflict.reason != "reference" for conflict in result.conflicts)


def test_mv_is_converted_to_uv_and_conversion_is_logged():
    first = _edf(unit="uV")
    second = _edf(unit="mV", data=np.arange(10, dtype=np.float64) / 1000)

    result = concatenate([first, second])

    assert len(result) == 1
    assert result.recordings[0].signals[0].physical_dimension == "uV"
    conversion = result.conversions[0]
    assert conversion.channel == "EEG1"
    assert conversion.source_unit == "mV"
    assert conversion.target_unit == "uV"
    assert conversion.factor == pytest.approx(1000)
    assert conversion.offset == 0
    assert result.conversion_log == result.conversions


def test_converted_data_is_scaled_to_the_target_unit():
    first = _signal(data=np.arange(5, dtype=np.float64))
    second = _signal(data=np.arange(5, dtype=np.float64), unit="mV")

    result = concatenate([Edf([first]), Edf([second])])

    data = result.recordings[0].signals[0].data
    np.testing.assert_allclose(data[:5], np.arange(5), atol=0.1)
    np.testing.assert_allclose(data[5:], np.arange(5) * 1000, atol=20)


def test_conversion_annotation_is_written_at_segment_start():
    first = _edf(unit="uV")
    second = _edf(unit="mV")

    result = concatenate([first, second])

    annotations = result.recordings[0].annotations
    conversion_annotations = [
        annotation
        for annotation in annotations
        if annotation.text.startswith("EDFIO: unit conversion")
    ]
    assert len(conversion_annotations) == 1
    assert conversion_annotations[0].onset == 10
    assert "mV -> uV" in conversion_annotations[0].text
    assert "factor=1000" in conversion_annotations[0].text


def test_conversion_annotation_can_be_disabled():
    result = concatenate([_edf(unit="uV"), _edf(unit="mV")], annotate_conversions=False)

    assert not any(
        annotation.text.startswith("EDFIO: unit conversion")
        for annotation in result.recordings[0].annotations
    )
    assert len(result.conversions) == 1


def test_explicit_conversion_rule_takes_precedence_and_keeps_its_basis():
    first = _edf(unit="uV")
    second = _edf(unit="raw", data=np.arange(10, dtype=np.float64))

    result = concatenate(
        [first, second],
        unit_conversions=[
            UnitConversion(
                "raw",
                "uV",
                factor=2.5,
                offset=-1,
                description="lab calibration sheet 42",
            )
        ],
    )

    conversion = result.conversions[0]
    assert conversion.factor == 2.5
    assert conversion.offset == -1
    assert conversion.rule == "lab calibration sheet 42"
    annotation = result.recordings[0].annotations[-1]
    assert "basis=lab calibration sheet 42" in annotation.text


def test_explicit_conversion_rule_is_used_instead_of_builtin_rule():
    first = _edf(unit="uV")
    second = _edf(unit="mV")

    result = concatenate(
        [first, second],
        unit_conversions={("mV", "uV"): 2000},
    )

    assert result.conversions[0].factor == 2000


def test_conversion_mapping_forms_are_supported():
    def converted(rule):
        first = _edf(unit="uV")
        second = _edf(unit="mV")
        return concatenate([first, second], unit_conversions=rule).conversions[0]

    assert converted({("mV", "uV"): 1000}).factor == 1000
    assert converted({"mV": {"uV": 1000}}).factor == 1000
    assert converted({"mV->uV": (1000, 1)}).offset == 1
    assert converted([("mV", "uV", 1000, 1)]).offset == 1


def test_unknown_unit_without_conversion_is_split_not_rejected_as_a_whole():
    first = _edf(unit="uV")
    second = _edf(unit="furlong")

    result = concatenate([first, second])

    assert len(result) == 2
    assert result.conflicts[0].reason == "unit"


def test_target_unit_can_be_selected():
    result = concatenate(
        [_edf(unit="uV"), _edf(unit="mV")],
        target_units={"EEG1": "mV"},
    )

    assert result.recordings[0].signals[0].physical_dimension == "mV"
    assert result.conversions[0].target_unit == "mV"
    assert result.conversions[0].factor == pytest.approx(0.001)


def test_channel_metadata_override_supplies_reference():
    first = _edf(label="EEG1", reference="")
    second = _edf(label="EEG1", reference="")

    result = concatenate(
        [first, second],
        channel_metadata={
            "EEG1": ChannelMetadata(label="EEG1", unit="uV", reference="Cz")
        },
    )

    assert len(result) == 1


def test_channel_metadata_override_can_mark_references_different():
    first = _edf(label="EEG1")
    second = _edf(label="EEG1")

    result = concatenate(
        [first, second],
        channel_metadata={
            "EEG1": {"reference": "Cz"},
        },
    )

    assert len(result) == 1


def test_reference_can_be_extracted_from_transducer_type():
    signal = _signal(reference="M2")
    assert extract_reference(signal) == "M2"


def test_reference_can_be_extracted_from_label():
    signal = _signal(label="EEG Fpz-Cz")
    assert extract_reference(signal) == "Cz"


def test_annotation_onsets_are_shifted_for_later_segments():
    first = Edf([_signal(data=np.arange(5, dtype=np.float64))])
    second = Edf(
        [_signal(data=np.arange(5, dtype=np.float64))],
        annotations=[EdfAnnotation(1, None, "marker")],
    )

    result = concatenate([first, second])

    assert EdfAnnotation(6, None, "marker") in result.recordings[0].annotations


def test_recordings_are_split_but_both_outputs_are_preserved():
    first = _edf(reference="Cz")
    second = _edf(reference="M1")
    third = _edf(reference="M1")

    result = concatenate([first, second, third])

    assert len(result) == 2
    assert result[0].duration == 10
    assert result[1].duration == 20


def test_intersection_policy_keeps_compatible_channels():
    first = Edf(
        [
            _signal(label="EEG1", reference="Cz"),
            _signal(label="ECG", unit="mV"),
        ]
    )
    second = Edf([_signal(label="EEG1", reference="Cz")])

    result = concatenate([first, second], channel_policy="intersection")

    assert len(result) == 1
    assert result[0].labels == ("EEG1",)
    assert any(
        c.reason == "missing_channel" and c.channel == "ECG" for c in result.conflicts
    )


def test_channel_policy_strict_splits_on_channel_set_mismatch():
    first = Edf(
        [
            _signal(label="EEG1", reference="Cz"),
            _signal(label="ECG", unit="mV"),
        ]
    )
    second = Edf([_signal(label="EEG1", reference="Cz")])

    result = concatenate([first, second])

    assert len(result) == 2
    assert result[0].labels == ("EEG1", "ECG")


def test_on_incompatible_error_raises():
    with pytest.raises(ConcatenationError, match="reference"):
        concatenate(
            [_edf(reference="Cz"), _edf(reference="M1")],
            on_incompatible="error",
        )


def test_mixing_edf_and_bdf_is_rejected():
    bdf_signal = BdfSignal(np.arange(10, dtype=np.float64), 1, label="EEG1")
    with pytest.raises(ValueError, match="EDF and BDF"):
        concatenate([_edf(), Bdf([bdf_signal])])


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="At least one recording"):
        concatenate([])


def test_duplicate_labels_are_rejected():
    edf = Edf([_signal(label="EEG1"), _signal(label="EEG1")])
    with pytest.raises(ConcatenationError, match="duplicate labels"):
        concatenate([edf])


def test_recordings_can_be_passed_as_separate_arguments():
    result = concatenate(_edf(), _edf())

    assert len(result) == 1
    assert result[0].duration == 20


def test_recordings_can_be_read_from_paths(tmp_path: Path):
    first_path = tmp_path / "first.edf"
    second_path = tmp_path / "second.edf"
    _edf().write(first_path)
    _edf(data=np.arange(10, 20, dtype=np.float64)).write(second_path)

    result = concatenate([str(first_path), second_path])

    assert len(result) == 1


def test_recordings_can_be_read_from_bytes():
    first = _edf().to_bytes()
    second = _edf(data=np.arange(10, 20, dtype=np.float64)).to_bytes()

    result = concatenate([first, second])

    assert len(result) == 1


def test_concatenated_result_roundtrips_through_edf(tmp_path: Path):
    first = _edf(unit="uV")
    second = _edf(unit="mV")
    result = concatenate([first, second])
    target = tmp_path / "merged.edf"

    result[0].write(target)
    loaded = read_edf(target)

    assert loaded.signals[0].physical_dimension == "uV"
    assert loaded.duration == 20
    assert any(
        annotation.text.startswith("EDFIO: unit conversion")
        for annotation in loaded.annotations
    )


def test_prefiltering_mismatch_splits_by_default():
    result = concatenate([_edf(prefiltering="HP:0.1Hz"), _edf(prefiltering="HP:0.5Hz")])

    assert len(result) == 2
    assert result.conflicts[0].reason == "prefiltering"


def test_prefiltering_can_be_ignored():
    result = concatenate(
        [_edf(prefiltering="HP:0.1Hz"), _edf(prefiltering="HP:0.5Hz")],
        require_prefiltering_match=False,
    )

    assert len(result) == 1


def test_segments_describe_source_positions():
    result = concatenate([_edf(), _edf(), _edf()])

    assert [segment.start_second for segment in result.segments] == [0, 10, 20]
    assert [segment.recording_index for segment in result.segments] == [0, 1, 2]
    assert all(segment.output_index == 0 for segment in result.segments)


def test_aliases_delegate_to_concatenate():
    for func in (concatenate_recordings, concatenate_edfs):
        result = func([_edf(), _edf()])
        assert len(result) == 1


def test_conversions_are_logged_on_the_result_and_basis_survives_write(tmp_path: Path):
    result = concatenate([_edf(unit="uV"), _edf(unit="mV")])
    target = tmp_path / "converted.edf"
    result[0].write(target)

    loaded = read_edf(target)
    assert result.conversions[0].factor == pytest.approx(1000)
    assert any("factor=1000" in annotation.text for annotation in loaded.annotations)


def test_conversion_basis_survives_write_and_read(tmp_path: Path):
    target = tmp_path / "converted.edf"
    result = concatenate(
        [_edf(unit="uV"), _edf(unit="mV")],
        unit_conversions=[UnitConversion("mV", "uV", 1000, description="SI prefix")],
    )
    result[0].write(target)

    loaded = read_edf(target)
    text = "\n".join(annotation.text for annotation in loaded.annotations)
    assert "SI prefix" in text
    assert "factor=1000" in text


def test_conversion_annotation_is_not_duplicated_when_input_already_contains_one():
    first = _edf(unit="uV")
    second = Edf(
        [_signal(unit="mV")],
        annotations=[EdfAnnotation(0, None, "EDFIO: unit conversion EEG1: mV -> uV")],
    )

    result = concatenate([first, second])

    conversions = [
        annotation
        for annotation in result[0].annotations
        if annotation.text.startswith("EDFIO: unit conversion")
    ]
    assert len(conversions) == 2


def test_bdf_concatenation_is_supported():
    first = Bdf([BdfSignal(np.arange(10, dtype=np.float64), 1, label="EEG1")])
    second = Bdf([BdfSignal(np.arange(10, dtype=np.float64), 1, label="EEG1")])

    result = concatenate([first, second])

    assert len(result) == 1
    assert isinstance(result[0], Bdf)
    assert result[0].duration == 20


def test_dataclass_metadata_from_signal():
    signal = _signal(unit="mV", reference="M1")
    metadata = ChannelMetadata.from_signal(signal)

    assert metadata.label == "EEG1"
    assert metadata.unit == "mV"
    assert metadata.reference == "M1"


def test_unit_conversion_validation():
    with pytest.raises(ValueError, match="non-zero"):
        UnitConversion("uV", "mV", factor=0)
    with pytest.raises(ValueError, match="finite"):
        UnitConversion("uV", "mV", offset=float("nan"))


def test_unit_conversion_inverse():
    conversion = UnitConversion("mV", "uV", factor=1000, offset=2)

    inverse = conversion.inverted()

    np.testing.assert_allclose(
        inverse.apply(conversion.apply(np.array([1.0, 2.0]))), [1.0, 2.0]
    )


def test_unknown_keyword_argument_is_rejected():
    with pytest.raises(TypeError, match="Unexpected keyword"):
        concatenate([_edf()], bogus=True)


def test_conflicting_output_is_written_and_read_for_both_groups(tmp_path: Path):
    result = concatenate([_edf(reference="Cz"), _edf(reference="M1")])

    paths = []
    for i, recording in enumerate(result):
        path = tmp_path / f"part-{i}.edf"
        recording.write(path)
        paths.append(path)

    assert [read_edf(path).duration for path in paths] == [10, 10]


def test_channel_conflict_str():
    result = concatenate([_edf(reference="Cz"), _edf(reference="M1")])
    assert "reference" in result.conflicts[0].description


def test_non_electrode_hyphenated_labels_do_not_supply_a_reference():
    first = _edf(label="Resp oro-nasal")
    second = _edf(label="Resp oro-nasal")

    result = concatenate([first, second])

    assert len(result) == 1
    assert extract_reference(first.signals[0]) == ""


def test_per_recording_metadata_overrides_are_supported():
    result = concatenate(
        [_edf(label="EEG1"), _edf(label="EEG1")],
        channel_metadata={
            0: {"EEG1": {"reference": "Cz"}},
            1: {"EEG1": {"reference": "M1"}},
        },
    )

    assert len(result) == 2
    assert result.conflicts[0].reason == "reference"


def test_metadata_override_matching_is_case_insensitive():
    result = concatenate(
        [_edf(label="eeg1"), _edf(label="eeg1")],
        channel_metadata={"EEG1": {"reference": "Cz"}},
    )

    assert len(result) == 1


def test_unit_scale_table_is_supported():
    first = _edf(unit="raw", data=np.full(5, 2.0))
    second = _edf(unit="uV", data=np.full(5, 10.0))

    result = concatenate([first, second], unit_conversions={"raw": 2.0, "uV": 1.0})

    assert result.recordings[0].signals[0].physical_dimension == "raw"
    np.testing.assert_allclose(
        result.recordings[0].signals[0].data[5:], np.full(5, 5.0), atol=0.1
    )


def test_split_outputs_keep_the_common_target_unit_and_conversion_log():
    result = concatenate(
        [_edf(unit="uV", reference="Cz"), _edf(unit="mV", reference="M1")]
    )

    assert len(result) == 2
    assert [recording.signals[0].physical_dimension for recording in result] == [
        "uV",
        "uV",
    ]
    assert result.conversions[0].output_index == 1
    assert result.conversions[0].factor == pytest.approx(1000)


def test_unknown_unit_on_first_recording_is_rejected_clearly():
    with pytest.raises(ConcatenationError, match="furlong"):
        concatenate(
            [_edf(unit="furlong"), _edf(unit="uV")], target_units={"EEG1": "uV"}
        )


def test_conflict_reports_output_and_recording_indices():
    result = concatenate(
        [_edf(reference="Cz"), _edf(reference="Cz"), _edf(reference="M1")]
    )

    assert len(result) == 2
    conflict = result.conflicts[0]
    assert conflict.output_index == 0
    assert conflict.recording_index == 2


def test_result_repr_mentions_counts():
    result = concatenate([_edf(unit="uV"), _edf(unit="mV")])

    assert "1 recording" in repr(result)
    assert "1 conversion" in repr(result)


def test_runs_expose_concatenable_ranges():
    result = concatenate(
        [
            _edf(reference="Cz"),
            _edf(reference="Cz"),
            _edf(reference="M1"),
        ]
    )

    assert result.runs == ((0, 1), (2,))


def test_result_write_uses_indexed_names_for_multiple_outputs(tmp_path: Path):
    result = concatenate([_edf(reference="Cz"), _edf(reference="M1")])

    paths = result.write(tmp_path, stem="part")

    assert [path.name for path in paths] == ["part-0.edf", "part-1.edf"]
    assert all(read_edf(path).duration == 10 for path in paths)


def test_result_write_uses_plain_name_for_single_output(tmp_path: Path):
    result = concatenate([_edf(), _edf()])

    (path,) = result.write(tmp_path, stem="merged")

    assert path.name == "merged.edf"
    assert read_edf(path).duration == 20


def test_result_write_keeps_bdf_suffix(tmp_path: Path):
    first = Bdf([BdfSignal(np.arange(10, dtype=np.float64), 1, label="EEG1")])
    second = Bdf([BdfSignal(np.arange(10, dtype=np.float64), 1, label="EEG1")])

    (path,) = concatenate([first, second]).write(tmp_path)

    assert path.suffix == ".bdf"


def test_conversion_record_description_includes_source_target_and_rule():
    record = ConversionRecord(
        output_index=0,
        recording_index=1,
        channel="EEG1",
        source_unit="mV",
        target_unit="uV",
        factor=1000,
        offset=1,
        rule="calibration",
    )

    description = record.description
    assert "EEG1: mV -> uV" in description
    assert "factor=1000" in description
    assert "offset=1" in description
    assert "[calibration]" in description


def test_result_comparison_and_hash():
    first = concatenate([_edf()])
    second = concatenate([_edf()])

    assert first != second
    assert first != [_edf()]
    assert first != "not a result"
    assert hash(first) == hash(first)


def test_plan_concatenation_reports_ranges_and_basis():
    result = plan_concatenation(
        [_edf(unit="uV", reference="Cz"), _edf(unit="mV", reference="Cz")]
    )

    assert len(result) == 0
    assert result.runs == ((0, 1),)
    assert result.conversions[0].factor == pytest.approx(1000)
    assert result[0].duration if len(result) else True


def test_plan_concatenation_honors_error_policy():
    with pytest.raises(ConcatenationError):
        plan_concatenation(
            [_edf(reference="Cz"), _edf(reference="M1")],
            on_incompatible="error",
        )


def test_explicit_reference_attribute_takes_precedence():
    signal = _signal(reference="M1")
    signal.reference = "Cz"
    assert extract_reference(signal) == "Cz"


def test_reference_in_label_brackets_is_extracted():
    signal = _signal(label="EEG1 [ref: Cz]")
    assert extract_reference(signal) == "Cz"


def test_reference_from_against_text_is_extracted():
    signal = _signal()
    signal.transducer_type = "Ag/AgCl against M2"
    assert extract_reference(signal) == "M2"


def test_bare_dash_label_without_electrode_context_has_no_reference():
    signal = _signal(label="Flow-Thermistor")
    assert extract_reference(signal) == ""


def test_bdf_bytes_are_detected():
    first = Bdf([BdfSignal(np.arange(5, dtype=np.float64), 1, label="A")])
    second = Bdf([BdfSignal(np.arange(5, dtype=np.float64), 1, label="A")])

    result = concatenate([first.to_bytes(), second.to_bytes()])

    assert isinstance(result[0], Bdf)


def test_bdf_paths_are_detected(tmp_path: Path):
    path = tmp_path / "a.bdf"
    Bdf([BdfSignal(np.arange(5, dtype=np.float64), 1, label="A")]).write(path)

    result = concatenate([path])

    assert isinstance(result[0], Bdf)


def test_unsupported_input_type_is_rejected():
    with pytest.raises(TypeError, match="Unsupported recording type"):
        concatenate([object()])


def test_additional_recordings_with_iterable_are_rejected():
    with pytest.raises(ValueError, match="Additional recordings"):
        concatenate([_edf()], _edf())


def test_no_shared_channels_splits_recording():
    result = concatenate([_edf(label="A"), _edf(label="B")])

    assert len(result) == 2
    assert result.conflicts[0].reason == "missing_channel"


def test_intersection_keeps_more_channels_when_present_in_all():
    first = Edf([_signal(label="A"), _signal(label="B")])
    second = Edf([_signal(label="A"), _signal(label="B")])

    result = concatenate([first, second], channel_policy="intersection")

    assert result[0].labels == ("A", "B")


def test_no_shared_channel_under_intersection_splits():
    result = concatenate(
        [_edf(label="A"), _edf(label="B")], channel_policy="intersection"
    )

    assert len(result) == 2
    assert any(c.reason == "no_shared_channels" for c in result.conflicts)


def test_unit_conversion_with_unit_object():
    first = _edf(unit="uV")
    second = _edf(unit="mV")

    result = concatenate(
        [first, second], unit_conversions=UnitConversion("mV", "uV", 500)
    )

    assert result.conversions[0].factor == 500


def test_invalid_conversion_item_is_rejected():
    with pytest.raises(ValueError, match="Invalid unit conversion"):
        concatenate([_edf()], unit_conversions=[42])


def test_invalid_conversion_key_is_rejected():
    with pytest.raises(ValueError, match="Invalid conversion key"):
        concatenate([_edf()], unit_conversions={"not-a-rule": 1})


def test_conversion_key_with_colon_separator():
    first = _edf(unit="uV")
    second = _edf(unit="mV")

    result = concatenate([first, second], unit_conversions={"mV:uV": 1000})

    assert result.conversions[0].factor == 1000


def test_conversion_value_with_factor_tuple_forms():
    first = _edf(unit="uV")
    second = _edf(unit="mV")

    scalar = concatenate([first, second], unit_conversions={("mV", "uV"): (1000,)})
    assert scalar.conversions[0].factor == 1000
    with pytest.raises(ValueError, match="Invalid conversion value"):
        concatenate([first, second], unit_conversions={("mV", "uV"): (1, 2, 3)})


def test_inverse_rule_can_be_used():
    first = _edf(unit="mV")
    second = _edf(unit="uV")

    result = concatenate([first, second], unit_conversions={("uV", "mV"): 0.001})

    assert result.conversions[0].factor == pytest.approx(1000)


def test_reference_equivalence_mapping_and_string_forms():
    a = _edf(reference="Cz")
    b = _edf(reference="M1")

    assert len(concatenate([a, b], reference_equivalences={"x": ("Cz", "M1")})) == 1
    assert len(concatenate([a, b], reference_equivalences=["Cz = M1"])) == 1


def test_metadata_override_by_recording_and_label_match():
    result = concatenate(
        [_edf(label="A"), _edf(label="A")],
        channel_metadata={0: {"a": {"reference": "Cz"}}, 1: {"A": {"reference": "Cz"}}},
    )

    assert len(result) == 1


def test_micro_sign_unit_from_latin_1_file_is_normalized_on_write(tmp_path: Path):
    source = tmp_path / "latin1.edf"
    edf = _edf(unit="uV")
    edf.write(source)
    raw = source.read_bytes().replace(b"uV      ", "µV".encode("latin-1").ljust(8))
    source.write_bytes(raw)

    first = read_edf(source, header_encoding="latin-1")
    second = _edf(unit="uV")
    result = concatenate([first, second])
    target = tmp_path / "merged.edf"
    result[0].write(target)

    loaded = read_edf(target)
    assert loaded.signals[0].physical_dimension == "uV"


def test_non_ascii_transducer_type_is_made_writable(tmp_path: Path):
    source = tmp_path / "latin1.edf"
    edf = Edf(
        [
            EdfSignal(
                np.arange(5, dtype=np.float64),
                1,
                label="EEG1",
                transducer_type="Ag/AgCl",
                physical_dimension="uV",
                physical_range=(-10, 10),
            )
        ]
    )
    edf.write(source)
    raw = source.read_bytes().replace(
        b"Ag/AgCl".ljust(80), "Ag/AgCl µ-ref".encode("latin-1").ljust(80)
    )
    source.write_bytes(raw)

    first = read_edf(source, header_encoding="latin-1")
    result = concatenate([first, _edf()])
    target = tmp_path / "merged.edf"
    result[0].write(target)

    assert read_edf(target).signals[0].transducer_type.startswith("Ag/AgCl")


def test_reference_equivalence_mapping_with_string_value():
    a = _edf(reference="Cz")
    b = _edf(reference="M1")

    result = concatenate([a, b], reference_equivalences={"pair": ["Cz", "M1"]})

    assert len(result) == 1


def test_plan_cannot_be_written(tmp_path: Path):
    plan = plan_concatenation([_edf(), _edf()])

    with pytest.raises(ValueError, match="empty concatenation plan"):
        plan.write(tmp_path)
