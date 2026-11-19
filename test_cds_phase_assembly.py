"""Tests for cds_phase_assembly: strand-aware, phase-checked CDS assembly."""

import pytest

from cds_phase_assembly import (
    DEMO_CDS_TRANSCRIPTION,
    DEMO_PROTEIN,
    KNOWN_PREFIX,
    CDSSegment,
    assemble_cds,
    assemble_naive_ascending,
    build_demo_reference,
    reverse_complement,
)


@pytest.fixture()
def demo():
    reference, segments = build_demo_reference()
    return reference, segments


def test_minus_strand_assembly_roundtrip(demo):
    reference, segments = demo
    result = assemble_cds(reference, segments)
    assert result.cds == DEMO_CDS_TRANSCRIPTION
    assert result.translation == DEMO_PROTEIN + "*"
    assert result.five_prime_complete
    assert result.three_prime_complete
    assert result.leftover_3prime == ""
    assert result.contradictions == []


def test_naive_ascending_order_is_wrong(demo):
    reference, segments = demo
    naive = assemble_naive_ascending(reference, segments)
    assert naive != DEMO_PROTEIN + "*"
    assert not naive.startswith(KNOWN_PREFIX)


def test_segments_accepted_in_any_order(demo):
    reference, segments = demo
    shuffled = [segments[2], segments[0], segments[1]]
    assert assemble_cds(reference, shuffled).cds == DEMO_CDS_TRANSCRIPTION


def test_provenance_offsets_and_phases(demo):
    reference, segments = demo
    result = assemble_cds(reference, segments)
    contribs = result.contributions
    # Transcription order on the minus strand is descending coordinates.
    assert [c.segment.name for c in contribs] == ["cds1", "cds2", "cds3"]
    assert [c.cds_offset for c in contribs] == [0, 40, 71]
    assert [c.length for c in contribs] == [40, 31, 13]
    assert [c.expected_phase for c in contribs] == [None, 2, 1]
    assert all(c.phase_ok for c in contribs)
    # Per-segment sequence provenance reconstructs the full CDS.
    assert "".join(c.sequence for c in contribs) == result.cds
    # First contribution is the reverse complement of the genomic slice.
    genomic = reference[300 - 1 : 339]
    assert contribs[0].sequence == reverse_complement(genomic)


def test_phase_contradiction_names_segment_and_drops_nothing(demo):
    reference, segments = demo
    bad = [segments[0], CDSSegment("cds2", 200, 230, "-", 0), segments[2]]
    result = assemble_cds(reference, bad)
    assert len(result.contradictions) == 1
    message = result.contradictions[0]
    assert "cds2" in message and "cds1" in message
    assert "declared phase 0, expected 2" in message
    # No bases are silently discarded to force a multiple of three.
    assert len(result.cds) == 40 + 31 + 13
    assert result.contributions[1].phase_ok is False
    assert result.contributions[2].phase_ok is True


def test_three_prime_incomplete_keeps_leftover_and_verifies_prefix(demo):
    reference, segments = demo
    # Minus strand: truncate the 3' (transcription) end = low-coordinate end.
    truncated = [CDSSegment("cds3", 102, 112, "-", 1), segments[1], segments[2]]
    result = assemble_cds(reference, truncated)
    assert len(result.cds) == 82  # nothing dropped
    assert result.leftover_3prime == DEMO_CDS_TRANSCRIPTION[81:82]
    assert not result.three_prime_complete
    assert result.five_prime_complete
    # Translation stops before the incomplete codon; known prefix still checks out.
    assert result.translation == DEMO_PROTEIN  # 27 full codons, stop codon lost
    assert result.check_prefix(KNOWN_PREFIX)


def test_five_prime_partial_when_first_phase_nonzero(demo):
    reference, segments = demo
    partial = [segments[0], segments[1], CDSSegment("cds1", 300, 339, "-", 1)]
    result = assemble_cds(reference, partial)
    assert not result.five_prime_complete
    assert any("5' end is partial" in n for n in result.notes)


def test_plus_strand_still_works():
    # Same CDS placed on the plus strand of a 200 bp fragment.
    cds = DEMO_CDS_TRANSCRIPTION
    ref = ("GATTACA" * 30)[:200]
    ref = ref[:49] + cds[:40] + ref[89:]
    ref = ref[:99] + cds[40:71] + ref[130:]
    ref = ref[:149] + cds[71:] + ref[162:]
    segments = [
        CDSSegment("p1", 50, 89, "+", 0),
        CDSSegment("p2", 100, 130, "+", 2),
        CDSSegment("p3", 150, 162, "+", 1),
    ]
    result = assemble_cds(ref, segments)
    assert result.cds == cds
    assert result.translation == DEMO_PROTEIN + "*"
    assert [c.segment.name for c in result.contributions] == ["p1", "p2", "p3"]


def test_mixed_strands_rejected(demo):
    reference, segments = demo
    mixed = [segments[0], CDSSegment("cds2", 200, 230, "+", 2), segments[2]]
    with pytest.raises(ValueError, match="strand"):
        assemble_cds(reference, mixed)


def test_invalid_phase_rejected():
    with pytest.raises(ValueError, match="phase"):
        CDSSegment("x", 1, 10, "-", 3)
