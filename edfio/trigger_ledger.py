"""Reconcile stimulus annotations with a digital trigger channel.

Stimulus presentation software logs annotations while a separate digital
trigger channel records hardware pulses. The two event streams are typically
offset by a small latency. This module detects trigger edges via a threshold
and debounce time, matches them one-to-one against stimulus annotations
within an allowed latency window (optionally calibrated by a few confirmed
annotation/edge pairs), and produces a per-trial synchronization ledger.

Each trigger edge may claim at most one annotation and vice versa. If
several optimal matchings exist, only pairs common to all of them are
marked as confirmed; all others remain tentative. Latencies are always
reported as intervals that include the sampling boundary of the trigger
channel (and an optional annotation onset uncertainty), never as falsely
precise point values.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt

from edfio.edf import read_bdf, read_edf

if TYPE_CHECKING:
    from edfio.edf import Bdf, Edf
    from edfio.edf_annotations import EdfAnnotation

_COST_SCALE = 1_000_000_000


@dataclass(frozen=True)
class TriggerEdge:
    """A threshold crossing detected on the digital trigger channel.

    The crossing is only known to lie within the sampling interval
    [`start`, `end`], whose width is the sampling period of the trigger
    channel.

    Parameters
    ----------
    index : int
        Ordinal of the edge among all detected edges.
    start : float
        Earliest possible crossing time in seconds from recording start.
    end : float
        Latest possible crossing time in seconds from recording start.
    """

    index: int
    start: float
    end: float


@dataclass(frozen=True)
class LedgerEntry:
    """A matched annotation/trigger pair in the synchronization ledger.

    Parameters
    ----------
    annotation_index : int
        Index of the annotation within the (filtered) annotation list.
    annotation_onset : float
        Annotation onset in seconds from recording start.
    annotation_text : str
        Annotation text.
    edge_index : int
        Index of the trigger edge within the detected edges.
    edge_start : float
        Earliest possible edge time in seconds from recording start.
    edge_end : float
        Latest possible edge time in seconds from recording start.
    latency_low : float
        Lower bound of the trigger latency in seconds.
    latency_high : float
        Upper bound of the trigger latency in seconds.
    status : {"confirmed", "tentative"}
        `"confirmed"` if the pair is user-confirmed or part of every optimal
        matching, `"tentative"` otherwise.
    user_confirmed : bool
        Whether the pair was provided as a confirmed input pair.
    alternatives : tuple[int, ...]
        Indices of other edges this annotation is paired with in some other
        optimal matching (empty for confirmed pairs).
    """

    annotation_index: int
    annotation_onset: float
    annotation_text: str
    edge_index: int
    edge_start: float
    edge_end: float
    latency_low: float
    latency_high: float
    status: Literal["confirmed", "tentative"]
    user_confirmed: bool
    alternatives: tuple[int, ...]


@dataclass(frozen=True)
class DroppedAnnotation:
    """A stimulus annotation that no trigger edge could be matched to.

    Parameters
    ----------
    index : int
        Index of the annotation within the (filtered) annotation list.
    onset : float
        Annotation onset in seconds from recording start.
    text : str
        Annotation text.
    """

    index: int
    onset: float
    text: str


@dataclass(frozen=True)
class SyncLedger:
    """Per-trial reconciliation of stimulus annotations and trigger edges.

    Parameters
    ----------
    entries : tuple[LedgerEntry, ...]
        Matched annotation/trigger pairs, sorted by annotation onset.
    dropped_annotations : tuple[DroppedAnnotation, ...]
        Annotations no trigger edge could be matched to.
    extra_triggers : tuple[TriggerEdge, ...]
        Trigger edges no annotation could be matched to.
    latency_window : tuple[float, float]
        The plausible latency window in seconds: the allowed latency
        intersected with the latency intervals of all confirmed pairs.
    """

    entries: tuple[LedgerEntry, ...]
    dropped_annotations: tuple[DroppedAnnotation, ...]
    extra_triggers: tuple[TriggerEdge, ...]
    latency_window: tuple[float, float]


def detect_trigger_edges(
    data: npt.ArrayLike,
    sampling_frequency: float,
    threshold: float,
    debounce: float = 0.0,
    edge: Literal["rising", "falling"] = "rising",
) -> tuple[TriggerEdge, ...]:
    """Detect threshold crossings on a digital trigger channel.

    Parameters
    ----------
    data : array-like
        The physical signal values of the trigger channel.
    sampling_frequency : float
        The sampling frequency of the trigger channel in Hz.
    threshold : float
        The trigger threshold in the channel's physical dimension.
    debounce : float, default: 0.0
        Crossings occurring within this many seconds after an accepted edge
        are ignored as bounces.
    edge : {"rising", "falling"}, default: "rising"
        The polarity of the trigger edges to detect.

    Returns
    -------
    tuple[TriggerEdge, ...]
        The detected edges in temporal order. Each edge time is only known
        up to the sampling period of the trigger channel.
    """
    if sampling_frequency <= 0:
        raise ValueError(
            f"Sampling frequency must be positive, got {sampling_frequency}"
        )
    if debounce < 0:
        raise ValueError(f"Debounce time must be non-negative, got {debounce}")
    if edge not in ("rising", "falling"):
        raise ValueError(f"Edge must be 'rising' or 'falling', got {edge!r}")
    samples = np.asarray(data, dtype=np.float64).ravel()
    if samples.size < 2:  # noqa: PLR2004
        return ()
    high = samples >= threshold
    if edge == "falling":
        high = ~high
    crossings = np.flatnonzero(~high[:-1] & high[1:]) + 1
    period = 1.0 / sampling_frequency
    edges: list[TriggerEdge] = []
    last_accepted = -math.inf
    for sample_index in crossings:
        time = float(sample_index) * period
        if time - last_accepted >= debounce:
            edges.append(
                TriggerEdge(len(edges), (int(sample_index) - 1) * period, time)
            )
            last_accepted = time
    return tuple(edges)


def reconcile(
    annotations: Sequence[EdfAnnotation],
    edges: Sequence[TriggerEdge],
    *,
    max_latency: float,
    min_latency: float = 0.0,
    confirmed_pairs: Sequence[tuple[int, int]] = (),
    annotation_uncertainty: float = 0.0,
) -> SyncLedger:
    """Match stimulus annotations to trigger edges within a latency window.

    Matching is one-to-one: each trigger edge claims at most one annotation
    and each annotation is matched to at most one edge. A matching is
    optimal if it matches as many pairs as possible and, among those,
    minimizes the total deviation from the calibrated latency window. If
    several optimal matchings exist, only pairs common to all of them are
    marked as confirmed.

    Parameters
    ----------
    annotations : Sequence[EdfAnnotation]
        The stimulus annotations to reconcile.
    edges : Sequence[TriggerEdge]
        The detected trigger edges.
    max_latency : float
        Maximum plausible trigger latency in seconds.
    min_latency : float, default: 0.0
        Minimum plausible trigger latency in seconds.
    confirmed_pairs : Sequence[tuple[int, int]], default: ()
        Confirmed `(annotation_index, edge_index)` pairs. They are trusted
        as ground truth, are always part of the ledger, and calibrate the
        plausible latency window.
    annotation_uncertainty : float, default: 0.0
        Half-width of the annotation onset uncertainty in seconds. The
        latency interval of every pair is widened by this amount on both
        sides.

    Returns
    -------
    SyncLedger
        The per-trial synchronization ledger, including dropped annotations
        and extra triggers.
    """
    if min_latency > max_latency:
        raise ValueError(
            f"min_latency ({min_latency}) must not exceed max_latency ({max_latency})"
        )
    if annotation_uncertainty < 0:
        raise ValueError(
            f"Annotation uncertainty must be non-negative, got {annotation_uncertainty}"
        )
    confirmed_pairs = tuple(confirmed_pairs)
    _validate_confirmed_pairs(confirmed_pairs, len(annotations), len(edges))
    edges = tuple(TriggerEdge(i, e.start, e.end) for i, e in enumerate(edges))
    intervals = [
        [
            (
                edge.start - annotation.onset - annotation_uncertainty,
                edge.end - annotation.onset + annotation_uncertainty,
            )
            for edge in edges
        ]
        for annotation in annotations
    ]
    window = _calibrate_latency_window(
        intervals, confirmed_pairs, min_latency, max_latency
    )
    matched, essential, alternatives = _match(
        intervals, window, confirmed_pairs, len(annotations), len(edges)
    )
    entries = [
        _make_entry(
            annotations,
            edges,
            a,
            e,
            window[0],
            window[1],
            "confirmed",
            alternatives=(),
            user_confirmed=True,
        )
        for a, e in confirmed_pairs
    ]
    for a, e in matched.items():
        low = max(intervals[a][e][0], window[0])
        high = min(intervals[a][e][1], window[1])
        status: Literal["confirmed", "tentative"] = (
            "confirmed" if (a, e) in essential else "tentative"
        )
        entries.append(
            _make_entry(
                annotations,
                edges,
                a,
                e,
                low,
                high,
                status,
                alternatives=alternatives[a],
                user_confirmed=False,
            )
        )
    entries.sort(key=lambda entry: (entry.annotation_onset, entry.annotation_index))
    matched_annotations = set(matched) | {a for a, _ in confirmed_pairs}
    matched_edges = set(matched.values()) | {e for _, e in confirmed_pairs}
    dropped = tuple(
        DroppedAnnotation(a, annotations[a].onset, annotations[a].text)
        for a in range(len(annotations))
        if a not in matched_annotations
    )
    extra = tuple(e for i, e in enumerate(edges) if i not in matched_edges)
    return SyncLedger(tuple(entries), dropped, extra, window)


def generate_sync_ledger(
    edf: Edf | Bdf,
    trigger_channel: str,
    *,
    threshold: float,
    max_latency: float,
    min_latency: float = 0.0,
    debounce: float = 0.0,
    confirmed_pairs: Sequence[tuple[int, int]] = (),
    annotation_pattern: str | None = None,
    annotation_uncertainty: float = 0.0,
    edge: Literal["rising", "falling"] = "rising",
) -> SyncLedger:
    """Generate a synchronization ledger from an EDF/BDF recording.

    Parameters
    ----------
    edf : Edf | Bdf
        The recording containing the stimulus annotations and the trigger
        channel.
    trigger_channel : str
        Label of the digital trigger channel.
    threshold : float
        The trigger threshold in the channel's physical dimension.
    max_latency : float
        Maximum plausible trigger latency in seconds.
    min_latency : float, default: 0.0
        Minimum plausible trigger latency in seconds.
    debounce : float, default: 0.0
        Trigger edges within this many seconds after an accepted edge are
        ignored as bounces.
    confirmed_pairs : Sequence[tuple[int, int]], default: ()
        Confirmed `(annotation_index, edge_index)` pairs, referring to the
        filtered annotation list and the detected edges.
    annotation_pattern : str | None, default: None
        Regular expression selecting stimulus annotations by their text. If
        `None`, all annotations are used.
    annotation_uncertainty : float, default: 0.0
        Half-width of the annotation onset uncertainty in seconds.
    edge : {"rising", "falling"}, default: "rising"
        The polarity of the trigger edges to detect.

    Returns
    -------
    SyncLedger
        The per-trial synchronization ledger.
    """
    signal = edf.get_signal(trigger_channel)
    edges = detect_trigger_edges(
        signal.data, signal.sampling_frequency, threshold, debounce, edge
    )
    annotations: Sequence[EdfAnnotation] = edf.annotations
    if annotation_pattern is not None:
        pattern = re.compile(annotation_pattern)
        annotations = [a for a in annotations if pattern.search(a.text)]
    return reconcile(
        annotations,
        edges,
        max_latency=max_latency,
        min_latency=min_latency,
        confirmed_pairs=confirmed_pairs,
        annotation_uncertainty=annotation_uncertainty,
    )


def format_ledger_text(ledger: SyncLedger) -> str:
    """Format a synchronization ledger as a human-readable text report.

    Parameters
    ----------
    ledger : SyncLedger
        The ledger to format.

    Returns
    -------
    str
        The formatted ledger.
    """
    lines = [
        "Stimulus-trigger synchronization ledger",
        "Calibrated latency window: "
        f"[{ledger.latency_window[0]:.6f}, {ledger.latency_window[1]:.6f}] s",
        "",
        "Matched trials:",
    ]
    if ledger.entries:
        lines.append(
            f"  {'trial':>5} {'ann#':>4} {'onset (s)':>12} {'text':<20.20}"
            f" {'edge#':>5} {'edge interval (s)':>25} {'latency (s)':>25}"
            f" {'status':<17} alternatives"
        )
        for trial, entry in enumerate(ledger.entries, start=1):
            status = entry.status + (" (user)" if entry.user_confirmed else "")
            alternatives = ";".join(map(str, entry.alternatives)) or "-"
            lines.append(
                f"  {trial:>5} {entry.annotation_index:>4}"
                f" {entry.annotation_onset:>12.6f} {entry.annotation_text:<20.20}"
                f" {entry.edge_index:>5}"
                f" [{entry.edge_start:.6f}, {entry.edge_end:.6f}]"
                f" [{entry.latency_low:.6f}, {entry.latency_high:.6f}]"
                f" {status:<17} {alternatives}"
            )
    else:
        lines.append("  (none)")
    lines += ["", "Dropped annotations (no trigger edge claimed them):"]
    if ledger.dropped_annotations:
        for dropped in ledger.dropped_annotations:
            lines.append(
                f"  ann# {dropped.index}  onset {dropped.onset:.6f} s"
                f"  text {dropped.text!r}"
            )
    else:
        lines.append("  (none)")
    lines += ["", "Extra triggers (no annotation claimed them):"]
    if ledger.extra_triggers:
        for trigger in ledger.extra_triggers:
            lines.append(
                f"  edge# {trigger.index}"
                f"  interval [{trigger.start:.6f}, {trigger.end:.6f}] s"
            )
    else:
        lines.append("  (none)")
    return "\n".join(lines)


def ledger_to_dict(ledger: SyncLedger) -> dict[str, Any]:
    """Convert a synchronization ledger to a JSON-serializable dictionary.

    Parameters
    ----------
    ledger : SyncLedger
        The ledger to convert.

    Returns
    -------
    dict[str, Any]
        The ledger as a dictionary.
    """
    return {
        "latency_window": list(ledger.latency_window),
        "entries": [
            {
                "trial": trial,
                "annotation_index": entry.annotation_index,
                "annotation_onset": entry.annotation_onset,
                "annotation_text": entry.annotation_text,
                "edge_index": entry.edge_index,
                "edge_interval": [entry.edge_start, entry.edge_end],
                "latency_interval": [entry.latency_low, entry.latency_high],
                "status": entry.status,
                "user_confirmed": entry.user_confirmed,
                "alternatives": list(entry.alternatives),
            }
            for trial, entry in enumerate(ledger.entries, start=1)
        ],
        "dropped_annotations": [
            {
                "annotation_index": dropped.index,
                "onset": dropped.onset,
                "text": dropped.text,
            }
            for dropped in ledger.dropped_annotations
        ],
        "extra_triggers": [
            {"edge_index": trigger.index, "interval": [trigger.start, trigger.end]}
            for trigger in ledger.extra_triggers
        ],
    }


def format_ledger_json(ledger: SyncLedger) -> str:
    """Format a synchronization ledger as JSON.

    Parameters
    ----------
    ledger : SyncLedger
        The ledger to format.

    Returns
    -------
    str
        The ledger as a JSON string.
    """
    return json.dumps(ledger_to_dict(ledger), indent=2)


_CSV_HEADER = [
    "trial",
    "annotation_index",
    "annotation_onset",
    "annotation_text",
    "edge_index",
    "edge_start",
    "edge_end",
    "latency_low",
    "latency_high",
    "status",
    "user_confirmed",
    "alternatives",
]


def format_ledger_csv(ledger: SyncLedger) -> str:
    """Format a synchronization ledger as CSV, one row per ledger item.

    Matched trials, dropped annotations, and extra triggers are all
    included and distinguished by the `status` column.

    Parameters
    ----------
    ledger : SyncLedger
        The ledger to format.

    Returns
    -------
    str
        The ledger as a CSV string.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(_CSV_HEADER)
    for trial, entry in enumerate(ledger.entries, start=1):
        writer.writerow(
            [
                trial,
                entry.annotation_index,
                entry.annotation_onset,
                entry.annotation_text,
                entry.edge_index,
                entry.edge_start,
                entry.edge_end,
                entry.latency_low,
                entry.latency_high,
                entry.status,
                entry.user_confirmed,
                ";".join(map(str, entry.alternatives)),
            ]
        )
    for dropped in ledger.dropped_annotations:
        writer.writerow(
            ["", dropped.index, dropped.onset, dropped.text]
            + [""] * 5
            + ["dropped_annotation", "", ""]
        )
    for trigger in ledger.extra_triggers:
        writer.writerow(
            ["", "", "", "", trigger.index, trigger.start, trigger.end]
            + [""] * 2
            + ["extra_trigger", "", ""]
        )
    return buffer.getvalue()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface for generating synchronization ledgers.

    Parameters
    ----------
    argv : Sequence[str] | None, default: None
        Command-line arguments. If `None`, `sys.argv` is used.

    Returns
    -------
    int
        The exit code.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate a per-trial synchronization ledger reconciling stimulus"
            " annotations with a digital trigger channel of an EDF/BDF recording."
        )
    )
    parser.add_argument("recording", type=Path, help="EDF/BDF file to analyze")
    parser.add_argument(
        "--trigger-channel",
        required=True,
        help="label of the digital trigger channel",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="trigger threshold in the channel's physical dimension",
    )
    parser.add_argument(
        "--debounce",
        type=float,
        default=0.0,
        help="ignore trigger edges within this many seconds after an accepted"
        " edge (default: 0)",
    )
    parser.add_argument(
        "--min-latency",
        type=float,
        default=0.0,
        help="minimum plausible trigger latency in seconds (default: 0)",
    )
    parser.add_argument(
        "--max-latency",
        type=float,
        required=True,
        help="maximum plausible trigger latency in seconds",
    )
    parser.add_argument(
        "--confirm",
        action="append",
        default=[],
        type=_parse_confirmed_pair,
        metavar="ANN:EDGE",
        help="confirmed annotation/edge pair as 0-based indices into the"
        " filtered annotations and the detected edges; may be given multiple"
        " times",
    )
    parser.add_argument(
        "--annotation-pattern",
        help="regular expression selecting stimulus annotations by their text"
        " (default: all annotations)",
    )
    parser.add_argument(
        "--annotation-uncertainty",
        type=float,
        default=0.0,
        help="half-width of the annotation onset uncertainty in seconds (default: 0)",
    )
    parser.add_argument(
        "--edge",
        choices=("rising", "falling"),
        default="rising",
        help="trigger edge polarity (default: rising)",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json", "csv"),
        default="text",
        help="ledger output format (default: text)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write the ledger to this file instead of stdout",
    )
    args = parser.parse_args(argv)
    ledger = generate_sync_ledger(
        _read_recording(args.recording),
        args.trigger_channel,
        threshold=args.threshold,
        max_latency=args.max_latency,
        min_latency=args.min_latency,
        debounce=args.debounce,
        confirmed_pairs=args.confirm,
        annotation_pattern=args.annotation_pattern,
        annotation_uncertainty=args.annotation_uncertainty,
        edge=args.edge,
    )
    if args.format == "json":
        output = format_ledger_json(ledger)
    elif args.format == "csv":
        output = format_ledger_csv(ledger)
    else:
        output = format_ledger_text(ledger)
    if args.output is None:
        print(output)
    else:
        args.output.write_text(output + "\n", encoding="utf-8")
    return 0


def _parse_confirmed_pair(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+):(\d+)", value)
    if match is None:
        raise argparse.ArgumentTypeError(
            f"invalid confirmed pair {value!r}, expected ANNOTATION_INDEX:EDGE_INDEX"
        )
    return int(match.group(1)), int(match.group(2))


def _read_recording(path: Path) -> Edf | Bdf:
    if path.suffix.lower() == ".bdf":
        return read_bdf(path)
    return read_edf(path)


def _validate_confirmed_pairs(
    confirmed_pairs: tuple[tuple[int, int], ...],
    num_annotations: int,
    num_edges: int,
) -> None:
    annotation_indices = [a for a, _ in confirmed_pairs]
    edge_indices = [e for _, e in confirmed_pairs]
    if len(set(annotation_indices)) != len(annotation_indices) or len(
        set(edge_indices)
    ) != len(edge_indices):
        raise ValueError("Confirmed pairs must not share annotations or edges")
    for a, e in confirmed_pairs:
        if not 0 <= a < num_annotations:
            raise ValueError(
                f"Confirmed pair references annotation index {a}, but only"
                f" {num_annotations} annotations were provided"
            )
        if not 0 <= e < num_edges:
            raise ValueError(
                f"Confirmed pair references edge index {e}, but only"
                f" {num_edges} trigger edges were detected"
            )


def _calibrate_latency_window(
    intervals: list[list[tuple[float, float]]],
    confirmed_pairs: tuple[tuple[int, int], ...],
    min_latency: float,
    max_latency: float,
) -> tuple[float, float]:
    low, high = min_latency, max_latency
    for a, e in confirmed_pairs:
        pair_low, pair_high = intervals[a][e]
        low = max(low, pair_low)
        high = min(high, pair_high)
    if low > high:
        raise ValueError(
            "Confirmed pairs are inconsistent: no single latency within the"
            " allowed window explains all of them"
        )
    return low, high


def _make_entry(
    annotations: Sequence[EdfAnnotation],
    edges: tuple[TriggerEdge, ...],
    annotation_index: int,
    edge_index: int,
    latency_low: float,
    latency_high: float,
    status: Literal["confirmed", "tentative"],
    *,
    user_confirmed: bool,
    alternatives: tuple[int, ...],
) -> LedgerEntry:
    annotation = annotations[annotation_index]
    edge = edges[edge_index]
    return LedgerEntry(
        annotation_index=annotation_index,
        annotation_onset=annotation.onset,
        annotation_text=annotation.text,
        edge_index=edge_index,
        edge_start=edge.start,
        edge_end=edge.end,
        latency_low=latency_low,
        latency_high=latency_high,
        status=status,
        user_confirmed=user_confirmed,
        alternatives=alternatives,
    )


def _match(
    intervals: list[list[tuple[float, float]]],
    window: tuple[float, float],
    confirmed_pairs: tuple[tuple[int, int], ...],
    num_annotations: int,
    num_edges: int,
) -> tuple[dict[int, int], set[tuple[int, int]], dict[int, tuple[int, ...]]]:
    confirmed_annotations = {a for a, _ in confirmed_pairs}
    confirmed_edges = {e for _, e in confirmed_pairs}
    remaining_annotations = [
        a for a in range(num_annotations) if a not in confirmed_annotations
    ]
    remaining_edges = [e for e in range(num_edges) if e not in confirmed_edges]
    edge_position = {e: i for i, e in enumerate(remaining_edges)}
    window_mid = (window[0] + window[1]) / 2

    def feasible(a: int, e: int) -> bool:
        low = max(intervals[a][e][0], window[0])
        high = min(intervals[a][e][1], window[1])
        return low <= high

    def pair_cost(a: int, e: int) -> int:
        low = max(intervals[a][e][0], window[0])
        high = min(intervals[a][e][1], window[1])
        return round(abs((low + high) / 2 - window_mid) * _COST_SCALE)

    def build_arcs(
        forbidden: tuple[int, int] | None = None,
        only: tuple[int, int] | None = None,
    ) -> list[list[tuple[int, int]]]:
        arcs = []
        for a in remaining_annotations:
            targets = []
            for e in remaining_edges:
                if (a, e) == forbidden:
                    continue
                if only is not None and a == only[0] and e != only[1]:
                    continue
                if feasible(a, e):
                    targets.append((edge_position[e], pair_cost(a, e)))
            arcs.append(targets)
        return arcs

    cardinality, total_cost, pairs = _optimal_matching(
        build_arcs(), len(remaining_edges)
    )
    matched = {remaining_annotations[a]: remaining_edges[e] for a, e in pairs}
    essential = set()
    alternatives = {}
    for a, e in matched.items():
        card_without, cost_without, _ = _optimal_matching(
            build_arcs(forbidden=(a, e)), len(remaining_edges)
        )
        if card_without < cardinality or (
            card_without == cardinality and cost_without > total_cost
        ):
            essential.add((a, e))
        alts = []
        for other_edge in remaining_edges:
            if other_edge == e or not feasible(a, other_edge):
                continue
            card_forced, cost_forced, _ = _optimal_matching(
                build_arcs(only=(a, other_edge)), len(remaining_edges)
            )
            if (card_forced, cost_forced) == (cardinality, total_cost):
                alts.append(other_edge)
        alternatives[a] = tuple(alts)
    return matched, essential, alternatives


def _build_flow_graph(
    arcs: list[list[tuple[int, int]]], num_edges: int
) -> tuple[list[list[list[int]]], int, int]:
    num_annotations = len(arcs)
    source = num_annotations + num_edges
    sink = source + 1
    graph: list[list[list[int]]] = [[] for _ in range(sink + 1)]

    def connect(node_from: int, node_to: int, capacity: int, cost: int) -> None:
        graph[node_from].append([node_to, capacity, cost, len(graph[node_to])])
        graph[node_to].append([node_from, 0, -cost, len(graph[node_from]) - 1])

    for annotation, targets in enumerate(arcs):
        connect(source, annotation, 1, 0)
        for edge, cost in targets:
            connect(annotation, num_annotations + edge, 1, cost)
    for edge in range(num_edges):
        connect(num_annotations + edge, sink, 1, 0)
    return graph, source, sink


def _shortest_augmenting_path(
    graph: list[list[list[int]]], source: int, sink: int
) -> tuple[int, list[int], list[int]] | None:
    num_nodes = len(graph)
    inf = 1 << 62
    dist = [inf] * num_nodes
    prev_node = [-1] * num_nodes
    prev_arc = [-1] * num_nodes
    dist[source] = 0
    for _ in range(num_nodes - 1):
        updated = False
        for node in range(num_nodes):
            if dist[node] == inf:
                continue
            for k, arc in enumerate(graph[node]):
                target, capacity, cost = arc[0], arc[1], arc[2]
                if capacity > 0 and dist[node] + cost < dist[target]:
                    dist[target] = dist[node] + cost
                    prev_node[target] = node
                    prev_arc[target] = k
                    updated = True
        if not updated:
            break
    if dist[sink] == inf:
        return None
    return dist[sink], prev_node, prev_arc


def _augment(
    graph: list[list[list[int]]],
    prev_node: list[int],
    prev_arc: list[int],
    source: int,
    sink: int,
) -> None:
    node = sink
    while node != source:
        arc = graph[prev_node[node]][prev_arc[node]]
        arc[1] -= 1
        graph[node][arc[3]][1] += 1
        node = prev_node[node]


def _optimal_matching(
    arcs: list[list[tuple[int, int]]], num_edges: int
) -> tuple[int, int, list[tuple[int, int]]]:
    """Find a maximum-cardinality, minimum-cost bipartite matching.

    `arcs[i]` holds `(edge_index, cost)` tuples for annotation `i`. Returns
    `(cardinality, total_cost, pairs)` with pairs as local
    `(annotation, edge)` indices.
    """
    graph, source, sink = _build_flow_graph(arcs, num_edges)
    cardinality = 0
    total_cost = 0
    while (path := _shortest_augmenting_path(graph, source, sink)) is not None:
        cost, prev_node, prev_arc = path
        cardinality += 1
        total_cost += cost
        _augment(graph, prev_node, prev_arc, source, sink)
    pairs = []
    for annotation in range(len(arcs)):
        for arc in graph[annotation]:
            if len(arcs) <= arc[0] < source and arc[1] == 0:
                pairs.append((annotation, arc[0] - len(arcs)))
    return cardinality, total_cost, pairs


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
