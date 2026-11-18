from __future__ import annotations

import numpy as np
import pytest

from edfio import (
    Edf,
    EdfSignal,
    LeadMatchingError,
    LeadPair,
    LeadRef,
    match_leads,
)


def signal(
    label: str,
    *,
    offset: float = 0.0,
    dimension: str = "uV",
    transducer: str = "AgAgCl electrode",
    prefiltering: str = "HP:0.1Hz LP:75Hz",
    frequency: float = 128,
    samples: int = 256,
) -> EdfSignal:
    data = np.sin(np.arange(samples) / 10) + offset
    return EdfSignal(
        data,
        frequency,
        label=label,
        physical_dimension=dimension,
        transducer_type=transducer,
        prefiltering=prefiltering,
    )


def session(*signals: EdfSignal) -> Edf:
    return Edf(list(signals))


def pairs_of(result: LeadPair | tuple[LeadRef, ...]) -> set[tuple[int, int]]:
    if isinstance(result, LeadPair):
        return {(result.left.index, result.right.index)}
    if isinstance(result, LeadRef):
        return {(result.index, result.index)}
    return {(pair.left.index, pair.right.index) for pair in result}


def test_unique_label_is_paired_without_using_order_of_same_label_leads():
    left = session(signal("EEG Fpz"), signal("EEG Fpz"), signal("ECG"))
    right = session(signal("ECG"), signal("EEG Fpz"), signal("EEG Fpz"))

    result = match_leads(left, right)

    assert result.pairs == (LeadPair(result.left_refs[2], result.right_refs[0]),)
    assert len(result.interchangeable) == 1
    group = result.interchangeable[0]
    assert group.label == "EEG Fpz"
    assert group.left_indices == (0, 1)
    assert group.right_indices == (1, 2)
    assert not result.is_unique


def test_no_pair_is_reported_for_same_label_leads_without_evidence():
    left = session(signal("EEG Fpz"), signal("EEG Fpz"))
    right = session(signal("EEG Fpz"), signal("EEG Fpz"))

    result = match_leads(left, right)

    assert result.pairs == ()
    assert len(result.interchangeable) == 1
    assert result.interchangeable[0].size == 2
    assert len(result.interchangeable[0].possible_pairs) == 4
    assert result.candidates_for(result.left_refs[0]) == result.right_refs


def test_metadata_distinguishes_same_label_leads_completely():
    left = session(
        signal("EEG Fpz", dimension="uV"),
        signal("EEG Fpz", dimension="mV"),
    )
    right = session(
        signal("EEG Fpz", dimension="mV"),
        signal("EEG Fpz", dimension="uV"),
    )

    result = match_leads(left, right)

    assert result.is_unique
    assert {(pair.left.index, pair.right.index) for pair in result.pairs} == {
        (0, 1),
        (1, 0),
    }


def test_metadata_weights_can_ignore_a_field():
    left = session(
        signal("EEG Fpz", dimension="uV", prefiltering="HP:0.1Hz"),
        signal("EEG Fpz", dimension="mV", prefiltering="HP:0.1Hz"),
    )
    right = session(
        signal("EEG Fpz", dimension="mV", prefiltering="HP:0.1Hz"),
        signal("EEG Fpz", dimension="uV", prefiltering="HP:0.1Hz"),
    )

    result = match_leads(left, right, metadata_weights={"physical_dimension": 0})

    assert not result.is_unique
    assert len(result.interchangeable) == 1


def test_anchor_leaves_remaining_candidates_interchangeable():
    left = session(signal("EEG Fpz"), signal("EEG Fpz"), signal("EEG Fpz"))
    right = session(signal("EEG Fpz"), signal("EEG Fpz"), signal("EEG Fpz"))

    result = match_leads(left, right, anchors=[(0, 2)])

    assert {(pair.left.index, pair.right.index) for pair in result.pairs} == {(0, 2)}
    assert len(result.interchangeable) == 1
    group = result.interchangeable[0]
    assert group.left_indices == (1, 2)
    assert group.right_indices == (0, 1)
    assert len(group.possible_pairs) == 4
    assert result.candidates_for(result.left_refs[0]) == (result.right_refs[2],)


def test_anchor_only_narrows_candidates_of_its_own_label():
    left = session(
        signal("EEG Fpz"),
        signal("EEG Fpz"),
        signal("EOG horizontal"),
        signal("EOG horizontal"),
    )
    right = session(
        signal("EEG Fpz"),
        signal("EEG Fpz"),
        signal("EOG horizontal"),
        signal("EOG horizontal"),
    )

    before = match_leads(left, right)
    after = before.add_anchor((0, 1))

    eeg_before = next(g for g in before.interchangeable if g.label == "EEG Fpz")
    eog_before = next(g for g in before.interchangeable if g.label == "EOG horizontal")
    eog_after = next(g for g in after.interchangeable if g.label == "EOG horizontal")

    assert eeg_before.size == 2
    assert all(group.label != "EEG Fpz" for group in after.interchangeable)
    assert pairs_of(eeg_before.possible_pairs) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert eog_after == eog_before


def test_anchors_accept_labels_occurrences_and_refs():
    left = session(signal("EEG Fpz"), signal("EEG Fpz"), signal("ECG"))
    right = session(signal("EEG Fpz"), signal("EEG Fpz"), signal("ECG"))

    by_occurrence = match_leads(left, right, anchors=[(("EEG Fpz", 1), ("EEG Fpz", 0))])
    assert {(p.left.index, p.right.index) for p in by_occurrence.pairs} == {
        (0, 1),
        (1, 0),
        (2, 2),
    }

    by_ref = match_leads(
        left,
        right,
        anchors=[
            LeadPair(
                left_refs := by_occurrence.left_refs[1],
                by_occurrence.right_refs[0],
            )
        ],
    )
    assert {(p.left.index, p.right.index) for p in by_ref.pairs} == {
        (0, 1),
        (1, 0),
        (2, 2),
    }
    assert left_refs.index == 1

    by_label = match_leads(left, right, anchors=[("ECG", "ECG")])
    assert {(p.left.index, p.right.index) for p in by_label.pairs} == {(2, 2)}


def test_matching_preserves_waveforms_and_metadata():
    left = session(
        signal("EEG Fpz", offset=1),
        signal("EEG Fpz", offset=-1),
    )
    right = session(
        signal("EEG Fpz", offset=10),
        signal("EEG Fpz", offset=-10),
    )
    left_data = [signal.data.copy() for signal in left.signals]
    right_data = [signal.data.copy() for signal in right.signals]
    left_labels = left.labels
    right_labels = right.labels

    match_leads(left, right, anchors=[(0, 1)])

    for original, signal_ in zip(left_data, left.signals):
        np.testing.assert_array_equal(original, signal_.data)
    for original, signal_ in zip(right_data, right.signals):
        np.testing.assert_array_equal(original, signal_.data)
    assert left.labels == left_labels
    assert right.labels == right_labels


def test_to_bytes_is_unchanged_by_matching():
    left = session(signal("EEG Fpz"), signal("EEG Fpz"))
    right = session(signal("EEG Fpz"), signal("EEG Fpz"))
    before = left.to_bytes()

    match_leads(left, right, anchors=[(0, 0)])

    assert left.to_bytes() == before


def test_different_label_anchor_is_rejected():
    left = session(signal("EEG Fpz"), signal("ECG"))
    right = session(signal("EEG Fpz"), signal("ECG"))

    with pytest.raises(LeadMatchingError, match="equal labels"):
        match_leads(left, right, anchors=[("EEG Fpz", "ECG")])


def test_conflicting_anchors_are_rejected():
    left = session(signal("EEG Fpz"), signal("EEG Fpz"))
    right = session(signal("EEG Fpz"), signal("EEG Fpz"))

    with pytest.raises(LeadMatchingError, match="Multiple anchors for left"):
        match_leads(left, right, anchors=[(0, 0), (0, 1)])

    with pytest.raises(LeadMatchingError, match="Multiple anchors for right"):
        match_leads(left, right, anchors=[(0, 0), (1, 0)])


def test_ambiguous_label_string_is_rejected():
    left = session(signal("EEG Fpz"), signal("EEG Fpz"))
    right = session(signal("EEG Fpz"), signal("EEG Fpz"))

    with pytest.raises(LeadMatchingError, match="Ambiguous label"):
        match_leads(left, right, anchors=[("EEG Fpz", 0)])


def test_different_number_of_same_label_leads_is_rejected():
    left = session(signal("EEG Fpz"), signal("EEG Fpz"))
    right = session(signal("EEG Fpz"))

    with pytest.raises(LeadMatchingError, match="Different number"):
        match_leads(left, right)


def test_unknown_metadata_weight_is_rejected():
    left = session(signal("EEG Fpz"))
    right = session(signal("EEG Fpz"))

    with pytest.raises(ValueError, match="Unknown metadata weight"):
        match_leads(left, right, metadata_weights={"label": 0})


def test_leads_without_counterpart_are_reported():
    left = session(signal("EEG Fpz"), signal("ECG"))
    right = session(signal("EEG Fpz"), signal("EOG horizontal"))

    result = match_leads(left, right)

    assert [(pair.left.label, pair.right.label) for pair in result.pairs] == [
        ("EEG Fpz", "EEG Fpz")
    ]
    assert {ref.label for ref in result.unmatched} == {"ECG", "EOG horizontal"}


def test_annotation_signals_are_ignored():
    left = Edf(
        [signal("EEG Fpz"), signal("EEG Fpz")],
        annotations=[],
    )
    right = Edf(
        [signal("EEG Fpz"), signal("EEG Fpz")],
        annotations=[],
    )

    result = match_leads(left, right)

    assert len(result.left) == 2
    assert len(result.right) == 2


def test_result_summary_and_dict_are_serializable():
    left = session(signal("EEG Fpz"), signal("EEG Fpz"))
    right = session(signal("EEG Fpz"), signal("EEG Fpz"))

    result = match_leads(left, right)
    summary = result.as_dict()

    assert summary["unique"] is False
    assert summary["pairs"] == []
    assert summary["interchangeable"] == [
        {
            "label": "EEG Fpz",
            "left_indices": [0, 1],
            "right_indices": [0, 1],
            "size": 2,
        }
    ]
    assert "Interchangeable" in str(result)


def test_anchor_is_reflected_in_result():
    left = session(signal("EEG Fpz"))
    right = session(signal("EEG Fpz"))

    result = match_leads(left, right, anchors=[(0, 0)])

    assert result.anchors == (LeadPair(result.left_refs[0], result.right_refs[0]),)
    assert result.is_unique
