from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

import clean_auxiliary_channels as cleanup_mod
from clean_auxiliary_channels import (
    PROVENANCE_TEMPLATE,
    _parse_provenance,
    build_demo_recording,
    clean_auxiliary_channels,
    main,
    snapshot_signals,
    stamp_provenance,
    verify_reread,
    write_and_verify,
)
from edfio import Edf, EdfAnnotation, read_bdf, read_edf
from tests import TEST_DATA_DIR

DROP_LABELS = ["AUX REF", "AUX HELPER"]
RETAINED_LABELS = ("EEG Fpz-Cz", "EEG Cz-Oz", "EEG Fpz-Cz d", "Temp rectal")
EXPECTED_ANNOTATIONS = (
    EdfAnnotation(0.0, None, "start"),
    EdfAnnotation(2.5, 0.5, "event A"),
    EdfAnnotation(9.0, None, "event B"),
)


def _demo():
    edf, provenance = build_demo_recording(Edf._fmt.lower())
    return edf, provenance


def test_clean_drops_exactly_the_two_auxiliary_channels():
    edf, provenance = _demo()
    assert edf.num_signals == 6

    manifest = clean_auxiliary_channels(edf, DROP_LABELS, provenance)

    assert manifest["dropped_labels"] == tuple(DROP_LABELS)
    assert manifest["retained_labels"] == RETAINED_LABELS
    assert edf.labels == RETAINED_LABELS
    assert edf.num_signals == 4
    # Annotation signal is retained, hence total number of signals is 5.
    assert edf._total_num_signals == 5
    assert edf.bytes_in_header_record == 256 * (5 + 1)


def test_clean_keeps_header_timing_and_per_signal_layout():
    edf, provenance = _demo()
    original_layout = {
        signal.label: (
            signal.sampling_frequency,
            signal._num_samples,
        )
        for signal in edf.signals
    }

    manifest = clean_auxiliary_channels(edf, DROP_LABELS, provenance)

    assert manifest["header"]["num_data_records"] == 10
    assert manifest["header"]["data_record_duration"] == 1.0
    for entry in manifest["signals"]:
        if entry["is_annotation_signal"]:
            continue
        fs, num_samples = original_layout[entry["label"]]
        assert entry["sampling_frequency"] == fs
        assert entry["num_samples"] == num_samples
        assert entry["samples_per_data_record"] == fs * 1.0
    assert edf.num_data_records == 10
    assert edf.data_record_duration == 1.0


def test_clean_preserves_digital_codes_of_retained_signals():
    edf, provenance = _demo()
    snapshots = snapshot_signals(edf)

    clean_auxiliary_channels(edf, DROP_LABELS, provenance)

    for label in RETAINED_LABELS:
        np.testing.assert_array_equal(edf.get_signal(label).digital, snapshots[label])


def test_clean_keeps_all_annotations():
    edf, provenance = _demo()
    clean_auxiliary_channels(edf, DROP_LABELS, provenance)
    assert edf.annotations == EXPECTED_ANNOTATIONS


def test_provenance_note_marks_derived_channel_as_materialized():
    edf, provenance = _demo()
    expected_note = PROVENANCE_TEMPLATE.format(reference="AUX REF")
    assert provenance == {"EEG Fpz-Cz d": expected_note}

    clean_auxiliary_channels(edf, DROP_LABELS, provenance)

    derived = edf.get_signal("EEG Fpz-Cz d")
    assert derived.prefiltering == expected_note
    assert "not recomputable" in derived.prefiltering


def test_stamp_provenance_appends_to_existing_note():
    edf, _ = _demo()
    edf.get_signal("EEG Fpz-Cz").prefiltering = "HP:0.1Hz"
    stamp_provenance(edf, {"EEG Fpz-Cz": "materialized"})
    assert edf.get_signal("EEG Fpz-Cz").prefiltering == "HP:0.1Hz materialized"


def test_stamp_provenance_rejects_unknown_label():
    edf, _ = _demo()
    with pytest.raises(ValueError, match="No signal with label"):
        stamp_provenance(edf, {"does not exist": "note"})


def test_clean_rejects_identifier_outside_the_minimal_set():
    edf, provenance = _demo()
    with pytest.raises(ValueError, match="No signal found with index/label"):
        clean_auxiliary_channels(edf, [*DROP_LABELS, "not a channel"], provenance)
    # Nothing was removed before the failure surfaced.
    assert edf.num_signals == 6


def test_parse_provenance_rejects_malformed_entry():
    with pytest.raises(argparse.ArgumentTypeError, match="LABEL=NOTE"):
        _parse_provenance(["missing-separator"])


def test_parse_provenance_parses_entries():
    assert _parse_provenance(["Status=materialized", "Other=note"]) == {
        "Status": "materialized",
        "Other": "note",
    }
    assert _parse_provenance(None) == {}


def test_verify_reread_reports_all_checks_for_valid_recording(tmp_path: Path):
    edf, provenance = _demo()
    snapshots = snapshot_signals(edf)
    clean_auxiliary_channels(edf, DROP_LABELS, provenance)
    target = tmp_path / f"cleaned.{Edf._fmt.lower()}"
    edf.write(target)

    report = verify_reread(read_edf(target), edf, snapshots)

    checks = report["checks"]
    assert checks["label_order"]
    assert checks["annotations"]
    assert checks["header_length"]
    assert checks["record_layout"]
    assert set(checks["signals"]) == set(RETAINED_LABELS)
    assert all(
        signal_checks["digital_codes_match"]
        and signal_checks["sample_count_consistent"]
        for signal_checks in checks["signals"].values()
    )


def test_verify_reread_flags_tampered_digital_codes(tmp_path: Path):
    edf, provenance = _demo()
    snapshots = snapshot_signals(edf)
    clean_auxiliary_channels(edf, DROP_LABELS, provenance)
    target = tmp_path / f"cleaned.{Edf._fmt.lower()}"
    edf.write(target)
    tampered = read_edf(target)
    tampered.get_signal("EEG Fpz-Cz").digital[0] += 1

    report = verify_reread(tampered, edf, snapshots)

    assert not report["checks"]["signals"]["EEG Fpz-Cz"]["digital_codes_match"]
    assert report["checks"]["signals"]["EEG Cz-Oz"]["digital_codes_match"]


def test_write_and_verify_raises_on_tampered_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    edf, provenance = _demo()
    snapshots = snapshot_signals(edf)
    clean_auxiliary_channels(edf, DROP_LABELS, provenance)
    target = tmp_path / f"cleaned.{Edf._fmt.lower()}"
    edf.write(target)

    reader_name = "read_bdf" if Edf._fmt == "BDF" else "read_edf"
    original_read = getattr(cleanup_mod, reader_name)

    def tampering_read(path: Path):
        recording = original_read(path)
        recording.get_signal("EEG Fpz-Cz").digital[0] += 1
        return recording

    monkeypatch.setattr(cleanup_mod, reader_name, tampering_read)
    with pytest.raises(AssertionError, match="digital_codes_match"):
        write_and_verify(edf, target, snapshots)


def test_write_and_reread_keeps_digital_codes_in_original_order(tmp_path: Path):
    edf, provenance = _demo()
    snapshots = snapshot_signals(edf)
    manifest = clean_auxiliary_channels(edf, DROP_LABELS, provenance)
    target = tmp_path / f"cleaned.{Edf._fmt.lower()}"

    report = write_and_verify(edf, target, snapshots)
    reread = read_edf(target)

    assert target.stat().st_size == manifest["file_layout"]["total_bytes"]
    assert tuple(report["checks"]["signals"]) == RETAINED_LABELS
    assert reread.labels == RETAINED_LABELS
    for label in RETAINED_LABELS:
        signal = reread.get_signal(label)
        np.testing.assert_array_equal(signal.digital, snapshots[label])
        assert (
            len(signal.digital)
            == signal.samples_per_data_record * reread.num_data_records
        )
    assert reread.annotations == EXPECTED_ANNOTATIONS
    assert reread.bytes_in_header_record == 256 * (reread._total_num_signals + 1)
    assert reread.num_data_records == 10
    assert reread.data_record_duration == 1.0


def test_provenance_survives_write_and_reread(tmp_path: Path):
    edf, provenance = _demo()
    snapshots = snapshot_signals(edf)
    clean_auxiliary_channels(edf, DROP_LABELS, provenance)
    target = tmp_path / f"cleaned.{Edf._fmt.lower()}"

    write_and_verify(edf, target, snapshots)

    derived = read_edf(target).get_signal("EEG Fpz-Cz d")
    assert derived.prefiltering == PROVENANCE_TEMPLATE.format(reference="AUX REF")
    # The materialized values stay readable as physical data.
    assert np.all(np.isfinite(derived.data))
    assert len(derived.data) == 2000


def test_main_demo_writes_verified_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    suffix = ".bdf" if Edf._fmt == "BDF" else ".edf"
    target = tmp_path / f"demo{suffix}"
    main(["--demo", *(["--bdf"] if Edf._fmt == "BDF" else []), str(target)])
    assert target.exists()
    out = capsys.readouterr().out
    assert "dropped channels: ('AUX REF', 'AUX HELPER')" in out
    assert "verified digital codes for" in out


@pytest.mark.edf
def test_main_on_real_annotated_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    source = TEST_DATA_DIR / "mne_test.edf"
    target = tmp_path / "mne_clean.edf"
    main([str(source), str(target), "--drop", "Ergo-Left", "Ergo-Right"])
    captured = capsys.readouterr().out
    assert (
        "retained annotations: ['start', 'type A', 'type A', 'type B', 'type A']"
        in captured
    )
    reread = read_edf(target)
    assert "Ergo-Left" not in reread.labels
    assert "Ergo-Right" not in reread.labels
    assert len(reread.labels) == 137
    assert len(reread.annotations) == 5


@pytest.mark.bdf
def test_main_on_real_bdf_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    source = TEST_DATA_DIR / "short_psg.bdf"
    target = tmp_path / "short_psg_clean.bdf"
    main([str(source), str(target), "--drop", "EMG submental", "Temp rectal"])
    captured = capsys.readouterr().out
    assert "dropped channels: ('EMG submental', 'Temp rectal')" in captured
    reread = read_bdf(target)
    assert "EMG submental" not in reread.labels
    assert "Temp rectal" not in reread.labels
    assert len(reread.labels) == 5
    assert reread.data_record_duration == 30.0
    assert reread.num_data_records == 10


def test_main_requires_args_without_demo():
    with pytest.raises(SystemExit):
        main([])
