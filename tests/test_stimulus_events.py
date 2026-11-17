import pytest
from stimulus_events import (
    _demo_edf_two_channels,
    _demo_edf_with_event_ids,
    _event_id_from_text,
    build_event_table,
    build_occurrence_table,
    event_count_range,
    read_annotation_occurrences,
)

# The demo files are built with Edf/EdfSignal inside stimulus_events, so the
# conftest's edf/bdf class injection does not apply; run these tests once.
pytestmark = pytest.mark.edf


@pytest.fixture()
def occurrence_table():
    edf = _demo_edf_two_channels()
    return build_occurrence_table(read_annotation_occurrences(edf), text="stimulus")


def test_occurrence_table_preserves_channel_and_onset(occurrence_table):
    assert len(occurrence_table) == 6
    assert [occ.onset for occ in occurrence_table] == [10, 10, 20, 20, 20, 30]
    assert [occ.channel_index for occ in occurrence_table] == [0, 1, 0, 1, 1, 0]
    assert all(occ.channel == "EDF Annotations" for occ in occurrence_table)


def test_timekeeping_annotations_are_skipped():
    occurrences = read_annotation_occurrences(_demo_edf_two_channels())
    assert all(occ.text for occ in occurrences)


def test_event_count_range_without_identity_basis(occurrence_table):
    assert event_count_range(occurrence_table) == (3, 6)


def test_event_count_range_empty():
    assert event_count_range([]) == (0, 0)


def test_no_identity_basis_keeps_all_occurrences_separate(occurrence_table):
    events = build_event_table(occurrence_table)
    assert len(events) == len(occurrence_table)
    assert all(len(event.rows) == 1 for event in events)


def test_pairing_merges_only_listed_rows(occurrence_table):
    events = build_event_table(occurrence_table, pairs=[(0, 1), (2, 3)])
    assert len(events) == 4
    assert [event.rows for event in events] == [(0, 1), (2, 3), (4,), (5,)]
    # the independent simultaneous trigger (row 4) is not swallowed
    independent = next(event for event in events if event.rows == (4,))
    assert occurrence_table[independent.rows[0]].onset == 20
    assert occurrence_table[independent.rows[0]].channel_index == 1


def test_identity_convention_from_source():
    table = build_occurrence_table(
        read_annotation_occurrences(_demo_edf_with_event_ids())
    )
    events = build_event_table(table, key=_event_id_from_text)
    assert len(events) == 3
    for event in events:
        assert len(event.rows) == 2
        channels = {table[row].channel_index for row in event.rows}
        assert channels == {0, 1}


def test_untagged_occurrences_stay_separate_with_key(occurrence_table):
    events = build_event_table(occurrence_table, key=_event_id_from_text)
    assert len(events) == len(occurrence_table)


def test_pairs_must_not_overlap(occurrence_table):
    with pytest.raises(ValueError, match="paired more than once"):
        build_event_table(occurrence_table, pairs=[(0, 1), (1, 2)])


def test_pairs_must_reference_existing_rows(occurrence_table):
    with pytest.raises(ValueError, match="unknown occurrence row"):
        build_event_table(occurrence_table, pairs=[(0, 99)])


def test_key_and_pairs_are_mutually_exclusive(occurrence_table):
    with pytest.raises(ValueError, match="either 'key' or 'pairs'"):
        build_event_table(
            occurrence_table, key=_event_id_from_text, pairs=[(0, 1)]
        )
