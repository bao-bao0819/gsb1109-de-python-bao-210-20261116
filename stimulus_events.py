"""Occurrence and event tables for stimuli recorded by two annotation channels.

Scenario
--------
Two EDF+ annotation channels both logged ``"stimulus"``.  A pair of
occurrences (one per channel) may be

* the *same* stimulus acquired twice (duplicate logging), or
* two *independent* stimuli that happened to fire at the same time.

Identical onsets are not proof of identity, so without an event-identity
convention from the data source, occurrences must not be merged.  This
module therefore keeps two separate tables:

* **Occurrence table** -- every annotation occurrence, one row each, with
  the original channel and onset preserved.  Nothing is merged or dropped.
* **Event table** -- distinct stimulus events, built only when an identity
  basis exists:

  1. The source provides an identity convention (e.g. an event id embedded
     in the annotation text): pass it as ``key`` to `build_event_table`.
  2. Pairing information arrives later (e.g. from acquisition logs): pass
     the matched occurrence rows as ``pairs``.  Only the listed rows are
     merged; every other row remains an event of its own, so independent
     simultaneous triggers are never swallowed by the pairing.
  3. No identity basis at all: do not merge anything; report the honest
     count range from `event_count_range` instead (largest per-channel
     count .. total number of occurrences).

Run ``python stimulus_events.py`` for a demonstration of all three regimes.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from edfio import Edf, EdfAnnotation, EdfSignal
from edfio.edf_annotations import _create_annotations_signal, _EdfAnnotationsDataRecord


@dataclass(frozen=True)
class AnnotationOccurrence:
    """A single annotation read from one specific annotation channel.

    Parameters
    ----------
    channel : str
        Label of the annotation signal the occurrence was read from.
    channel_index : int
        Ordinal of the annotation signal among all annotation signals.
        Needed because EDF+ names every annotation signal the same
        (``"EDF Annotations"``), so the label alone cannot tell the two
        channels apart.
    onset : float
        Onset in seconds from recording start.
    duration : float | None
        Duration in seconds, `None` if the annotation has none.
    text : str
        Annotation text.
    """

    channel: str
    channel_index: int
    onset: float
    duration: float | None
    text: str


def read_annotation_occurrences(edf: Edf) -> list[AnnotationOccurrence]:
    """Read all annotations channel by channel, preserving channel and onset.

    `Edf.annotations` merges all annotation signals into one flat sequence,
    which would lose the channel each occurrence came from, so the
    annotation signals are decoded individually here (mirroring
    `Edf.get_annotations`, including its timekeeping and subsecond-starttime
    handling).

    Parameters
    ----------
    edf : Edf
        The recording to read annotations from.

    Returns
    -------
    list[AnnotationOccurrence]
        All occurrences, sorted by onset, then channel, then text.
    """
    occurrences = []
    subsecond_offset = edf._subsecond_offset
    annotation_signals = [s for s in edf._signals if s._is_annotation_signal]
    for channel_index, signal in enumerate(annotation_signals):
        for data_record in signal.digital.reshape(-1, signal._bytes_per_data_record):
            record = _EdfAnnotationsDataRecord.from_bytes(data_record.tobytes())
            annotations = record.annotations
            if channel_index == 0:
                # EDF+ timekeeping: the first TAL of the first annotation
                # signal in each data record only carries the record start.
                annotations = annotations[1:]
            for ann in annotations:
                occurrences.append(
                    AnnotationOccurrence(
                        channel=signal.label,
                        channel_index=channel_index,
                        onset=round(ann.onset - subsecond_offset, 12),
                        duration=ann.duration,
                        text=ann.text,
                    )
                )
    return sorted(
        occurrences, key=lambda occ: (occ.onset, occ.channel_index, occ.text)
    )


def build_occurrence_table(
    occurrences: Iterable[AnnotationOccurrence], text: str | None = None
) -> list[AnnotationOccurrence]:
    """Build the occurrence table: one row per occurrence, nothing merged.

    Rows keep their original channel and onset.  Row numbers used by the
    event table are the positions in the returned list.

    Parameters
    ----------
    occurrences : Iterable[AnnotationOccurrence]
        The occurrences, usually from `read_annotation_occurrences`.
    text : str | None, default: None
        If given, keep only occurrences with exactly this annotation text
        (e.g. ``"stimulus"``).

    Returns
    -------
    list[AnnotationOccurrence]
        The occurrence table, sorted by onset, then channel, then text.
    """
    table = [occ for occ in occurrences if text is None or occ.text == text]
    return sorted(table, key=lambda occ: (occ.onset, occ.channel_index, occ.text))


def event_count_range(
    occurrence_table: Sequence[AnnotationOccurrence],
) -> tuple[int, int]:
    """Possible range of the distinct-event count without identity information.

    The minimum assumes every occurrence outside the most complete channel
    duplicates an occurrence in it; the maximum assumes all occurrences are
    independent events.

    Parameters
    ----------
    occurrence_table : Sequence[AnnotationOccurrence]
        The occurrence table.

    Returns
    -------
    tuple[int, int]
        ``(minimum, maximum)`` number of distinct events.
    """
    if not occurrence_table:
        return 0, 0
    per_channel = Counter(occ.channel_index for occ in occurrence_table)
    return max(per_channel.values()), len(occurrence_table)


@dataclass(frozen=True)
class StimulusEvent:
    """A distinct stimulus event composed of one or more occurrence rows.

    Parameters
    ----------
    event_id : str
        Identifier of the event (``E1``, ``E2``, ...).
    rows : tuple[int, ...]
        Row numbers in the occurrence table that form this event.  The
        original channel and onset of each member stay in the occurrence
        table and are not altered.
    """

    event_id: str
    rows: tuple[int, ...]


def build_event_table(
    occurrence_table: Sequence[AnnotationOccurrence],
    *,
    key: Callable[[AnnotationOccurrence], Hashable | None] | None = None,
    pairs: Iterable[Iterable[int]] | None = None,
) -> list[StimulusEvent]:
    """Build the event table from an occurrence table and an identity basis.

    Exactly one identity basis may be given:

    * ``key``: the source's identity convention.  Occurrences with an equal,
      non-`None` key belong to the same event.
    * ``pairs``: pairing information obtained later, as groups of occurrence
      table row numbers.  Only the listed rows are merged; every remaining
      row stays an event of its own, so independent events recorded at the
      same time are not swallowed.

    With neither, every occurrence becomes its own event; in that situation
    prefer reporting `event_count_range` instead of a single count.

    Parameters
    ----------
    occurrence_table : Sequence[AnnotationOccurrence]
        The occurrence table from `build_occurrence_table`.
    key : Callable[[AnnotationOccurrence], Hashable | None] | None, default: None
        Maps an occurrence to its event identity, or `None` if the
        occurrence has no identity tag.
    pairs : Iterable[Iterable[int]] | None, default: None
        Groups of occurrence table row numbers known to be the same event.

    Returns
    -------
    list[StimulusEvent]
        The event table, sorted by earliest member onset.
    """
    if key is not None and pairs is not None:
        raise ValueError("Pass either 'key' or 'pairs', not both")
    groups: list[tuple[int, ...]] = []
    assigned: set[int] = set()
    if key is not None:
        by_key: dict[Hashable, list[int]] = {}
        for row, occ in enumerate(occurrence_table):
            occ_key = key(occ)
            if occ_key is not None:
                by_key.setdefault(occ_key, []).append(row)
        for rows in by_key.values():
            groups.append(tuple(sorted(rows)))
            assigned.update(rows)
    elif pairs is not None:
        for group in pairs:
            rows = tuple(sorted(group))
            for row in rows:
                if not 0 <= row < len(occurrence_table):
                    raise ValueError(f"Pairing refers to unknown occurrence row {row}")
                if row in assigned:
                    raise ValueError(f"Occurrence row {row} is paired more than once")
            if rows:
                groups.append(rows)
                assigned.update(rows)
    groups.extend((row,) for row in range(len(occurrence_table)) if row not in assigned)
    groups.sort(
        key=lambda rows: (min(occurrence_table[row].onset for row in rows), rows)
    )
    return [StimulusEvent(f"E{i}", rows) for i, rows in enumerate(groups, start=1)]


def format_occurrence_table(occurrence_table: Sequence[AnnotationOccurrence]) -> str:
    """Render the occurrence table as aligned text.

    Parameters
    ----------
    occurrence_table : Sequence[AnnotationOccurrence]
        The occurrence table from `build_occurrence_table`.

    Returns
    -------
    str
        The formatted table.
    """
    header = (
        f"{'row':>3} | {'channel':<15} | {'ch#':>3} | {'onset [s]':>9} | "
        f"{'duration [s]':>12} | text"
    )
    lines = [header, "-" * len(header)]
    for row, occ in enumerate(occurrence_table):
        duration = "" if occ.duration is None else f"{occ.duration:g}"
        lines.append(
            f"{row:>3} | {occ.channel:<15} | {occ.channel_index:>3} | "
            f"{occ.onset:>9g} | {duration:>12} | {occ.text}"
        )
    return "\n".join(lines)


def format_event_table(
    events: Sequence[StimulusEvent],
    occurrence_table: Sequence[AnnotationOccurrence],
) -> str:
    """Render the event table, listing each event's member occurrences.

    Parameters
    ----------
    events : Sequence[StimulusEvent]
        The event table from `build_event_table`.
    occurrence_table : Sequence[AnnotationOccurrence]
        The occurrence table the events refer to.

    Returns
    -------
    str
        The formatted table.
    """
    lines = []
    for event in events:
        members = [occurrence_table[row] for row in event.rows]
        onset = min(occ.onset for occ in members)
        lines.append(
            f"{event.event_id}: onset ~{onset:g}s, {len(members)} occurrence(s)"
        )
        for row, occ in zip(event.rows, members):
            lines.append(
                f"    row {row}: channel {occ.channel!r} #{occ.channel_index}, "
                f"onset {occ.onset:g}s, text {occ.text!r}"
            )
    return "\n".join(lines)


def _demo_edf_two_channels() -> Edf:
    """Create an EDF where two annotation channels both logged 'stimulus'.

    Channel #0 (acquisition system A) logged stimuli at 10, 20, and 30 s.
    Channel #1 (acquisition system B) logged 10 s and 20 s as duplicate
    acquisitions of the same stimuli, plus a second 20 s entry that is an
    independent simultaneous trigger.
    """
    return Edf(
        [
            EdfSignal(np.zeros(3000), 100),
            _create_annotations_signal(
                [
                    EdfAnnotation(10, None, "stimulus"),
                    EdfAnnotation(20, None, "stimulus"),
                    EdfAnnotation(30, None, "stimulus"),
                ],
                num_data_records=30,
                data_record_duration=1,
                signal_class=EdfSignal,
            ),
            _create_annotations_signal(
                [
                    EdfAnnotation(10, None, "stimulus"),
                    EdfAnnotation(20, None, "stimulus"),
                    EdfAnnotation(20, None, "stimulus"),
                ],
                num_data_records=30,
                data_record_duration=1,
                with_timestamps=False,
                signal_class=EdfSignal,
            ),
        ]
    )


def _demo_edf_with_event_ids() -> Edf:
    """Create an EDF whose source tags every stimulus with an event id."""
    return Edf(
        [
            EdfSignal(np.zeros(3000), 100),
            _create_annotations_signal(
                [
                    EdfAnnotation(10, None, "stimulus S1"),
                    EdfAnnotation(20, None, "stimulus S2"),
                    EdfAnnotation(30, None, "stimulus S3"),
                ],
                num_data_records=30,
                data_record_duration=1,
                signal_class=EdfSignal,
            ),
            _create_annotations_signal(
                [
                    EdfAnnotation(10, None, "stimulus S1"),
                    EdfAnnotation(20, None, "stimulus S2"),
                    EdfAnnotation(30, None, "stimulus S3"),
                ],
                num_data_records=30,
                data_record_duration=1,
                with_timestamps=False,
                signal_class=EdfSignal,
            ),
        ]
    )


def _event_id_from_text(occ: AnnotationOccurrence) -> str | None:
    match = re.fullmatch(r"stimulus (S\d+)", occ.text)
    return match.group(1) if match else None


def main() -> None:
    """Demonstrate the occurrence table and the three event-table regimes."""
    edf = _demo_edf_two_channels()
    table = build_occurrence_table(read_annotation_occurrences(edf), text="stimulus")

    print("Occurrence table (one row per recorded occurrence, nothing merged):")
    print(format_occurrence_table(table))

    low, high = event_count_range(table)
    print(f"\nNo identity basis -> distinct stimuli: between {low} and {high}.")
    print("(Merging by equal onset would be unjustified: simultaneous entries")
    print("may be duplicate acquisitions or independent simultaneous triggers.)")

    print("\nPairing information arrived later: rows (0, 1) and (2, 3) are the")
    print("same stimuli acquired twice. Only those rows are merged:")
    events = build_event_table(table, pairs=[(0, 1), (2, 3)])
    print(format_event_table(events, table))
    print(
        f"-> {len(events)} distinct events; the independent simultaneous trigger"
        " (row 4) is kept as its own event, not swallowed."
    )

    print("\nSource provides an identity convention (event id in the text):")
    id_table = build_occurrence_table(read_annotation_occurrences(_demo_edf_with_event_ids()))
    print(format_occurrence_table(id_table))
    id_events = build_event_table(id_table, key=_event_id_from_text)
    print(format_event_table(id_events, id_table))
    print(f"-> {len(id_events)} distinct events, merged by event id.")


if __name__ == "__main__":
    main()
