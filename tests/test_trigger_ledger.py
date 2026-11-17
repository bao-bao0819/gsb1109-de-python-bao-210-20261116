import csv
import io
import json

import numpy as np
import pytest

from edfio import Bdf, BdfSignal, Edf, EdfAnnotation, EdfSignal
from edfio.trigger_ledger import (
    TriggerEdge,
    detect_trigger_edges,
    format_ledger_csv,
    format_ledger_json,
    format_ledger_text,
    generate_sync_ledger,
    ledger_to_dict,
    main,
    reconcile,
)


def _annotations(*onsets: float) -> list[EdfAnnotation]:
    return [EdfAnnotation(onset, None, f"stim {i}") for i, onset in enumerate(onsets)]


def test_detect_trigger_edges_rising():
    data = np.array([0, 0, 5, 5, 0, 0, 5, 5, 5, 0], dtype=float)
    edges = detect_trigger_edges(data, 8, threshold=2.5)
    assert edges == (TriggerEdge(0, 0.125, 0.25), TriggerEdge(1, 0.625, 0.75))


def test_detect_trigger_edges_falling():
    data = np.array([5, 5, 0, 0, 5], dtype=float)
    edges = detect_trigger_edges(data, 10, threshold=2.5, edge="falling")
    assert edges == (TriggerEdge(0, 0.1, 0.2),)


def test_detect_trigger_edges_debounce_ignores_bounces():
    data = np.array([0, 5, 0, 5, 0, 0, 0, 0, 0, 5, 0], dtype=float)
    edges = detect_trigger_edges(data, 10, threshold=2.5, debounce=0.5)
    assert edges == (TriggerEdge(0, 0.0, 0.1), TriggerEdge(1, 0.8, 0.9))


def test_detect_trigger_edges_invalid_sampling_frequency():
    with pytest.raises(ValueError, match="Sampling frequency must be positive"):
        detect_trigger_edges(np.zeros(10), 0, threshold=1)


def test_detect_trigger_edges_negative_debounce():
    with pytest.raises(ValueError, match="Debounce time must be non-negative"):
        detect_trigger_edges(np.zeros(10), 10, threshold=1, debounce=-0.1)


def test_detect_trigger_edges_invalid_polarity():
    with pytest.raises(ValueError, match="Edge must be 'rising' or 'falling'"):
        detect_trigger_edges(np.zeros(10), 10, threshold=1, edge="both")  # type: ignore[arg-type]


@pytest.mark.parametrize("data", [np.array([]), np.array([1.0])])
def test_detect_trigger_edges_short_data_returns_empty(data: np.ndarray):
    assert detect_trigger_edges(data, 10, threshold=0.5) == ()


def test_reconcile_basic_match():
    edges = [TriggerEdge(0, 1.04, 1.06), TriggerEdge(1, 2.05, 2.07)]
    ledger = reconcile(_annotations(1.0, 2.0), edges, max_latency=0.2)
    assert len(ledger.entries) == 2
    entry = ledger.entries[0]
    assert entry.status == "confirmed"
    assert entry.user_confirmed is False
    assert entry.alternatives == ()
    assert entry.latency_low == pytest.approx(0.04)
    assert entry.latency_high == pytest.approx(0.06)
    assert ledger.dropped_annotations == ()
    assert ledger.extra_triggers == ()
    assert ledger.latency_window == (0.0, 0.2)


def test_reconcile_keeps_dropped_annotations_and_extra_triggers():
    edges = [TriggerEdge(0, 1.0, 1.1), TriggerEdge(1, 3.0, 3.1)]
    ledger = reconcile(_annotations(1.0, 5.0), edges, max_latency=0.5)
    assert len(ledger.entries) == 1
    assert [d.index for d in ledger.dropped_annotations] == [1]
    assert ledger.dropped_annotations[0].onset == 5.0
    assert [t.index for t in ledger.extra_triggers] == [1]


def test_reconcile_edge_claimed_at_most_once():
    edges = [TriggerEdge(0, 1.1, 1.2)]
    ledger = reconcile(_annotations(1.0, 1.05), edges, max_latency=0.5)
    assert len(ledger.entries) == 1
    assert ledger.entries[0].annotation_index == 0
    assert ledger.entries[0].edge_index == 0
    assert [d.index for d in ledger.dropped_annotations] == [1]


def test_reconcile_confirmed_pairs_calibrate_latency_window():
    edges = [
        TriggerEdge(0, 1.04, 1.06),
        TriggerEdge(1, 2.05, 2.07),
        TriggerEdge(2, 3.3, 3.32),
    ]
    ledger = reconcile(
        _annotations(1.0, 2.0, 3.0),
        edges,
        max_latency=0.5,
        confirmed_pairs=[(0, 0), (1, 1)],
    )
    assert ledger.latency_window == pytest.approx((0.05, 0.06))
    assert [d.index for d in ledger.dropped_annotations] == [2]
    assert [t.index for t in ledger.extra_triggers] == [2]
    assert all(entry.status == "confirmed" for entry in ledger.entries)
    user_entries = [e for e in ledger.entries if e.user_confirmed]
    assert len(user_entries) == 2
    for entry in user_entries:
        assert entry.latency_low == pytest.approx(0.05)
        assert entry.latency_high == pytest.approx(0.06)


def test_reconcile_inconsistent_confirmed_pairs_raise():
    edges = [TriggerEdge(0, 1.0, 1.1), TriggerEdge(1, 2.4, 2.5)]
    with pytest.raises(ValueError, match="Confirmed pairs are inconsistent"):
        reconcile(
            _annotations(1.0, 2.0),
            edges,
            max_latency=0.2,
            confirmed_pairs=[(0, 0), (1, 1)],
        )


@pytest.mark.parametrize("pair", [(0, 1), (1, 0)])
def test_reconcile_confirmed_pair_out_of_range(pair: tuple[int, int]):
    with pytest.raises(ValueError, match="Confirmed pair references"):
        reconcile(
            _annotations(1.0),
            [TriggerEdge(0, 1.0, 1.1)],
            max_latency=0.2,
            confirmed_pairs=[pair],
        )


def test_reconcile_duplicate_confirmed_pairs_raise():
    edges = [TriggerEdge(0, 1.0, 1.1), TriggerEdge(1, 2.0, 2.1)]
    with pytest.raises(ValueError, match="must not share"):
        reconcile(
            _annotations(1.0, 2.0),
            edges,
            max_latency=0.5,
            confirmed_pairs=[(0, 0), (1, 0)],
        )


def test_reconcile_invalid_latency_window():
    with pytest.raises(ValueError, match="min_latency"):
        reconcile([], [], min_latency=0.5, max_latency=0.2)


def test_reconcile_negative_annotation_uncertainty():
    with pytest.raises(ValueError, match="Annotation uncertainty"):
        reconcile([], [], max_latency=0.2, annotation_uncertainty=-0.1)


def test_reconcile_only_pairs_common_to_all_optimal_matchings_are_confirmed():
    annotations = _annotations(1.0, 1.25, 3.0)
    edges = [
        TriggerEdge(0, 1.5, 1.75),
        TriggerEdge(1, 1.75, 2.0),
        TriggerEdge(2, 3.5, 3.75),
    ]
    ledger = reconcile(annotations, edges, max_latency=2.0)
    by_annotation = {e.annotation_index: e for e in ledger.entries}
    assert by_annotation[0].status == "tentative"
    assert by_annotation[1].status == "tentative"
    assert by_annotation[2].status == "confirmed"
    assert by_annotation[2].edge_index == 2
    assert by_annotation[0].alternatives == (1 - by_annotation[0].edge_index,)
    assert by_annotation[1].alternatives == (1 - by_annotation[1].edge_index,)


def test_reconcile_annotation_uncertainty_widens_latency_interval():
    edges = [TriggerEdge(0, 1.1, 1.2)]
    ledger = reconcile(
        _annotations(1.0), edges, max_latency=0.5, annotation_uncertainty=0.05
    )
    entry = ledger.entries[0]
    assert entry.latency_low == pytest.approx(0.05)
    assert entry.latency_high == pytest.approx(0.25)


def test_reconcile_no_annotations_and_no_edges():
    ledger = reconcile([], [], max_latency=0.5)
    assert ledger.entries == ()
    assert ledger.dropped_annotations == ()
    assert ledger.extra_triggers == ()


def _trigger_recording(cls=Edf, signal_cls=EdfSignal):
    fs = 1000
    trigger = np.zeros(4000)
    for onset_sample in (1050, 2050):
        trigger[onset_sample : onset_sample + 50] = 5.0
    annotations = [
        EdfAnnotation(1.0, None, "stim A"),
        EdfAnnotation(2.0, None, "stim B"),
        EdfAnnotation(3.0, None, "other"),
    ]
    return cls(
        [signal_cls(trigger, fs, label="TRIG")],
        annotations=annotations,
    )


def test_generate_sync_ledger_end_to_end():
    ledger = generate_sync_ledger(
        _trigger_recording(),
        "TRIG",
        threshold=2.5,
        debounce=0.02,
        max_latency=0.2,
        annotation_pattern="stim",
    )
    assert len(ledger.entries) == 2
    first = ledger.entries[0]
    assert first.edge_start == pytest.approx(1.049)
    assert first.edge_end == pytest.approx(1.05)
    assert first.latency_low == pytest.approx(0.049)
    assert first.latency_high == pytest.approx(0.05)
    assert all(entry.status == "confirmed" for entry in ledger.entries)
    assert ledger.dropped_annotations == ()
    assert ledger.extra_triggers == ()


def test_generate_sync_ledger_unknown_channel():
    edf = Edf([EdfSignal(np.zeros(100), 100, label="EEG")])
    with pytest.raises(ValueError, match="No signal with label"):
        generate_sync_ledger(edf, "TRIG", threshold=1, max_latency=0.5)


@pytest.fixture
def ambiguous_ledger():
    annotations = _annotations(1.0, 1.25, 3.0, 9.0)
    edges = [
        TriggerEdge(0, 1.5, 1.75),
        TriggerEdge(1, 1.75, 2.0),
        TriggerEdge(2, 3.5, 3.75),
        TriggerEdge(3, 7.0, 7.25),
    ]
    return reconcile(annotations, edges, max_latency=2.0)


def test_format_ledger_text(ambiguous_ledger):
    text = format_ledger_text(ambiguous_ledger)
    assert "Calibrated latency window: [0.000000, 2.000000] s" in text
    assert "tentative" in text
    assert "confirmed" in text
    assert "Dropped annotations" in text
    assert "ann# 3" in text
    assert "Extra triggers" in text
    assert "edge# 3" in text


def test_format_ledger_text_empty():
    text = format_ledger_text(reconcile([], [], max_latency=0.5))
    assert text.count("(none)") == 3


def test_ledger_to_dict_and_json(ambiguous_ledger):
    data = ledger_to_dict(ambiguous_ledger)
    assert data["latency_window"] == [0.0, 2.0]
    assert len(data["entries"]) == 3
    assert data["entries"][0]["trial"] == 1
    assert len(data["dropped_annotations"]) == 1
    assert data["dropped_annotations"][0]["annotation_index"] == 3
    assert len(data["extra_triggers"]) == 1
    assert data["extra_triggers"][0]["edge_index"] == 3
    parsed = json.loads(format_ledger_json(ambiguous_ledger))
    assert parsed == data


def test_format_ledger_csv(ambiguous_ledger):
    rows = list(csv.reader(io.StringIO(format_ledger_csv(ambiguous_ledger))))
    assert rows[0][0] == "trial"
    statuses = [row[9] for row in rows[1:]]
    assert statuses.count("tentative") == 2
    assert statuses.count("confirmed") == 1
    assert "dropped_annotation" in statuses
    assert "extra_trigger" in statuses
    assert len(rows) == 6


def test_main_text_to_stdout(tmp_path, capsys):
    path = tmp_path / "rec.edf"
    _trigger_recording().write(path)
    result = main(
        [
            str(path),
            "--trigger-channel",
            "TRIG",
            "--threshold",
            "2.5",
            "--max-latency",
            "0.2",
            "--annotation-pattern",
            "stim",
        ]
    )
    assert result == 0
    out = capsys.readouterr().out
    assert "Stimulus-trigger synchronization ledger" in out
    assert "Matched trials:" in out


def test_main_json_to_file(tmp_path):
    path = tmp_path / "rec.edf"
    output = tmp_path / "ledger.json"
    _trigger_recording().write(path)
    result = main(
        [
            str(path),
            "--trigger-channel",
            "TRIG",
            "--threshold",
            "2.5",
            "--max-latency",
            "0.2",
            "--annotation-pattern",
            "stim",
            "--format",
            "json",
            "--output",
            str(output),
        ]
    )
    assert result == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert len(data["entries"]) == 2


def test_main_csv_to_file(tmp_path):
    path = tmp_path / "rec.edf"
    output = tmp_path / "ledger.csv"
    _trigger_recording().write(path)
    result = main(
        [
            str(path),
            "--trigger-channel",
            "TRIG",
            "--threshold",
            "2.5",
            "--max-latency",
            "0.2",
            "--format",
            "csv",
            "--output",
            str(output),
        ]
    )
    assert result == 0
    rows = list(csv.reader(io.StringIO(output.read_text(encoding="utf-8"))))
    assert rows[0][0] == "trial"


def test_main_bdf_recording(tmp_path, capsys):
    path = tmp_path / "rec.bdf"
    _trigger_recording(Bdf, BdfSignal).write(path)
    result = main(
        [
            str(path),
            "--trigger-channel",
            "TRIG",
            "--threshold",
            "2.5",
            "--max-latency",
            "0.2",
        ]
    )
    assert result == 0
    assert "synchronization ledger" in capsys.readouterr().out


def test_main_confirmed_pairs_narrow_latency_window(tmp_path, capsys):
    path = tmp_path / "rec.edf"
    _trigger_recording().write(path)
    result = main(
        [
            str(path),
            "--trigger-channel",
            "TRIG",
            "--threshold",
            "2.5",
            "--max-latency",
            "0.5",
            "--annotation-pattern",
            "stim",
            "--confirm",
            "0:0",
            "--confirm",
            "1:1",
        ]
    )
    assert result == 0
    out = capsys.readouterr().out
    assert "confirmed (user)" in out


def test_main_invalid_confirmed_pair(tmp_path):
    path = tmp_path / "rec.edf"
    _trigger_recording().write(path)
    with pytest.raises(SystemExit):
        main(
            [
                str(path),
                "--trigger-channel",
                "TRIG",
                "--threshold",
                "2.5",
                "--max-latency",
                "0.2",
                "--confirm",
                "bogus",
            ]
        )
