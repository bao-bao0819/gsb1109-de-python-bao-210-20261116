"""Cleanup of auxiliary channels from an EDF/BDF recording.

The cleanup removes a minimal set of auxiliary channels (identified by label)
while keeping every other signal and all EDF+ annotations readable. Only the
`num_signals` and `bytes_in_header_record` header fields change; the data record
duration, number of data records, and per-signal sampling layout are untouched,
so the retained channels keep their original sample timing and digital codes.

Derived channels that share a dropped reference channel can not be recomputed
from their original formula once the reference is gone. They are therefore kept
as materialized signals: the physical/digital values already stored in the file
are preserved byte for byte, and a provenance note is written to the signal's
`prefiltering` header field stating that the values are materialized and can not
be recomputed. Nothing in the output claims that the original derivation is
still reproducible.
"""

from __future__ import annotations

import argparse
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from edfio import Bdf, Edf, EdfAnnotation, EdfSignal, read_bdf, read_edf
from edfio.edf_signal import BdfSignal

PROVENANCE_TEMPLATE = "materialized; ref {reference!r} dropped; not recomputable"


def _samples_per_data_record(signal: EdfSignal | BdfSignal, edf: Edf | Bdf) -> int:
    try:
        return signal.samples_per_data_record
    except AttributeError:
        # Newly instantiated signals only get this header field on write.
        return round(signal.sampling_frequency * edf.data_record_duration)


def _layout_manifest(edf: Edf | Bdf, dropped: Sequence[str]) -> dict[str, Any]:
    """Collect the output layout of an EDF/BDF recording as a plain dict."""
    data_record_bytes = sum(
        _samples_per_data_record(signal, edf) * signal._bytes_per_sample
        for signal in edf._signals
    )
    return {
        "dropped_labels": tuple(dropped),
        "retained_labels": edf.labels,
        "signals": [
            {
                "label": signal.label,
                "sampling_frequency": signal.sampling_frequency,
                "samples_per_data_record": _samples_per_data_record(signal, edf),
                "num_samples": signal._num_samples,
                "digital_dtype": signal.digital.dtype.name,
                "is_annotation_signal": signal._is_annotation_signal,
                "provenance": signal.prefiltering,
            }
            for signal in edf._signals
        ],
        "header": {
            "bytes_in_header_record": edf.bytes_in_header_record,
            "num_signals": edf.num_signals,
            "total_num_signals": edf._total_num_signals,
            "num_data_records": edf.num_data_records,
            "data_record_duration": edf.data_record_duration,
        },
        "file_layout": {
            "header_bytes": edf.bytes_in_header_record,
            "data_record_bytes": data_record_bytes,
            "num_data_records": edf.num_data_records,
            "total_bytes": edf.bytes_in_header_record
            + data_record_bytes * edf.num_data_records,
        },
        "annotations": [annotation.text for annotation in edf.annotations],
    }


def stamp_provenance(edf: Edf | Bdf, provenance: dict[str, str]) -> None:
    """Attach provenance notes to materialized derived channels.

    Each entry maps the label of a derived channel to the note that explains its
    materialized origin. The note is appended to the signal's `prefiltering`
    header field, so it travels with the written file.

    Parameters
    ----------
    edf : Edf | Bdf
        The recording containing the derived channels.
    provenance : dict[str, str]
        Mapping of channel label to provenance note.

    Raises
    ------
    ValueError
        If a label does not identify an ordinary signal in the recording.
    """
    for label, note in provenance.items():
        signal = edf.get_signal(label)
        existing = signal.prefiltering
        signal.prefiltering = f"{existing} {note}".strip() if existing else note


def snapshot_signals(edf: Edf | Bdf) -> dict[str, np.ndarray]:
    """Snapshot the digital codes of all ordinary signals, keyed by label."""
    return {signal.label: signal.digital.copy() for signal in edf.signals}


def clean_auxiliary_channels(
    edf: Edf | Bdf,
    drop_labels: Sequence[str],
    provenance: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Drop auxiliary channels and return the resulting output layout.

    Exactly the channels whose labels are given in `drop_labels` are removed.
    Annotation signals are never dropped, so all events remain readable. The
    digital values of every retained signal stay untouched and keep their
    original sampling order.

    Parameters
    ----------
    edf : Edf | Bdf
        The recording to clean in place.
    drop_labels : Sequence[str]
        Labels of the auxiliary channels to remove.
    provenance : dict[str, str] | None, default: None
        Optional provenance notes for derived channels that share a dropped
        reference channel. See :func:`stamp_provenance`.

    Returns
    -------
    dict[str, Any]
        Layout manifest of the cleaned recording, including the dropped labels,
        per-signal sampling layout, header fields, file size layout, and the
        retained annotation texts.
    """
    drop_labels = list(drop_labels)
    if provenance:
        stamp_provenance(edf, provenance)
    edf.drop_signals(drop_labels)
    return _layout_manifest(edf, drop_labels)


def verify_reread(
    reread: Edf | Bdf,
    cleaned: Edf | Bdf,
    snapshots: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Compare a reread recording against the cleaned recording.

    Parameters
    ----------
    reread : Edf | Bdf
        The recording read back from the written file.
    cleaned : Edf | Bdf
        The in-memory cleaned recording that was written.
    snapshots : dict[str, numpy.ndarray]
        Digital codes captured before the cleanup, keyed by label.

    Returns
    -------
    dict[str, Any]
        Report containing the reread layout and one boolean check per invariant:
        label order, annotation contents, header length, data record timing, and
        per-signal digital code identity plus sample-count consistency.
    """
    expected_labels = tuple(label for label in snapshots if label in cleaned.labels)
    signal_checks: dict[str, dict[str, bool]] = {}
    for label in expected_labels:
        signal = reread.get_signal(label)
        signal_checks[label] = {
            "digital_codes_match": np.array_equal(signal.digital, snapshots[label]),
            "sample_count_consistent": len(signal.digital)
            == signal.samples_per_data_record * reread.num_data_records,
        }
    checks = {
        "label_order": reread.labels == expected_labels,
        "annotations": reread.annotations == cleaned.annotations,
        "header_length": reread.bytes_in_header_record
        == 256 * (reread._total_num_signals + 1),
        "record_layout": reread.num_data_records == cleaned.num_data_records
        and reread.data_record_duration == cleaned.data_record_duration,
        "signals": signal_checks,
    }
    return {"reread_layout": _layout_manifest(reread, []), "checks": checks}


def write_and_verify(
    edf: Edf | Bdf,
    target: Path | str,
    snapshots: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Write the cleaned recording, read it back, and verify all invariants.

    Parameters
    ----------
    edf : Edf | Bdf
        The cleaned recording.
    target : Path | str
        Destination file path.
    snapshots : dict[str, numpy.ndarray]
        Digital codes captured before the cleanup, keyed by label.

    Returns
    -------
    dict[str, Any]
        Verification report from :func:`verify_reread`.

    Raises
    ------
    AssertionError
        If the reread recording deviates in label order, annotation contents,
        header layout, or the digital codes of any retained signal.
    """
    target = Path(target)
    edf.write(target)
    reread = read_bdf(target) if isinstance(edf, Bdf) else read_edf(target)
    report = verify_reread(reread, edf, snapshots)
    checks = report["checks"]
    failed = [
        name for name, passed in checks.items() if name != "signals" and not passed
    ]
    failed.extend(
        f"{label}:{name}"
        for label, signal_checks in checks["signals"].items()
        for name, passed in signal_checks.items()
        if not passed
    )
    if failed:
        raise AssertionError(f"Reread verification failed for: {failed}")
    return report


def _demo_annotations() -> list[EdfAnnotation]:
    return [
        EdfAnnotation(0.0, None, "start"),
        EdfAnnotation(2.5, 0.5, "event A"),
        EdfAnnotation(9.0, None, "event B"),
    ]


def build_demo_recording(fmt: str = "edf") -> tuple[Edf | Bdf, dict[str, str]]:
    """Build a recording with raw, auxiliary, and materialized derived channels."""
    rng = np.random.default_rng(0)
    num_samples = 10 * 200
    raw_0 = rng.standard_normal(num_samples)
    raw_1 = rng.standard_normal(num_samples)
    reference = rng.standard_normal(num_samples) * 0.1
    helper = rng.standard_normal(num_samples)
    # Materialized derived channel: raw_0 - reference. After the reference channel
    # is dropped, these stored values can not be recomputed.
    derived = raw_0 - reference
    provenance_note = PROVENANCE_TEMPLATE.format(reference="AUX REF")
    annotations = _demo_annotations()
    if fmt == "bdf":
        recording: Edf | Bdf = Bdf(
            [
                BdfSignal(raw_0, 200.0, label="EEG Fpz-Cz"),
                BdfSignal(raw_1, 200.0, label="EEG Cz-Oz"),
                BdfSignal(reference, 200.0, label="AUX REF"),
                BdfSignal(helper, 200.0, label="AUX HELPER"),
                BdfSignal(derived, 200.0, label="EEG Fpz-Cz d"),
                BdfSignal(np.arange(10, dtype=float), 1.0, label="Temp rectal"),
            ],
            annotations=annotations,
        )
    else:
        recording = Edf(
            [
                EdfSignal(raw_0, 200.0, label="EEG Fpz-Cz"),
                EdfSignal(raw_1, 200.0, label="EEG Cz-Oz"),
                EdfSignal(reference, 200.0, label="AUX REF"),
                EdfSignal(helper, 200.0, label="AUX HELPER"),
                EdfSignal(derived, 200.0, label="EEG Fpz-Cz d"),
                EdfSignal(np.arange(10, dtype=float), 1.0, label="Temp rectal"),
            ],
            annotations=annotations,
        )
    return recording, {"EEG Fpz-Cz d": provenance_note}


def _parse_provenance(values: list[str] | None) -> dict[str, str]:
    provenance: dict[str, str] = {}
    for value in values or []:
        label, separator, note = value.partition("=")
        if not separator or not label or not note:
            raise argparse.ArgumentTypeError(
                f"--provenance expects LABEL=NOTE, got {value!r}"
            )
        provenance[label] = note
    return provenance


def main(argv: list[str] | None = None) -> None:
    """Command line entry point for the auxiliary channel cleanup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, help="Input EDF/BDF file")
    parser.add_argument("output", nargs="?", type=Path, help="Output EDF/BDF file")
    parser.add_argument(
        "--drop",
        nargs="+",
        metavar="LABEL",
        help="Labels of the auxiliary channels to remove",
    )
    parser.add_argument(
        "--provenance",
        action="append",
        metavar="LABEL=NOTE",
        help="Provenance note for a materialized derived channel",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the built-in synthetic scenario instead of reading a file",
    )
    parser.add_argument("--bdf", action="store_true", help="Use BDF for --demo")
    args = parser.parse_args(argv)

    if args.demo:
        edf, provenance = build_demo_recording("bdf" if args.bdf else "edf")
        drop_labels = ["AUX REF", "AUX HELPER"]
        suffix = ".bdf" if args.bdf else ".edf"
        output = args.output or args.input or Path(tempfile.mkstemp(suffix=suffix)[1])
    else:
        if args.input is None or args.output is None or not args.drop:
            parser.error("INPUT, OUTPUT and --drop are required unless --demo is used")
        reader = read_bdf if args.input.suffix.lower() == ".bdf" else read_edf
        edf = reader(args.input)
        drop_labels = args.drop
        provenance = _parse_provenance(args.provenance)
        output = args.output

    snapshots = snapshot_signals(edf)
    manifest = clean_auxiliary_channels(edf, drop_labels, provenance)
    report = write_and_verify(edf, output, snapshots)
    print(f"cleaned recording written to {output}")
    print(f"dropped channels: {manifest['dropped_labels']}")
    print(f"retained channels: {manifest['retained_labels']}")
    print(f"header layout: {manifest['header']}")
    print(f"file layout: {manifest['file_layout']}")
    print(f"retained annotations: {manifest['annotations']}")
    print(f"verified digital codes for: {tuple(report['checks']['signals'])}")


if __name__ == "__main__":
    main()
