# -*- coding: utf-8 -*-
"""Tests for :py:mod:`vcfpy.normalize`."""

import pytest

from vcfpy import normalize


# A non-repetitive left flank ("G") followed by five CAG units and one "T".
REPEAT = "G" + "CAG" * 5 + "T"
WINDOW = normalize.ReferenceWindow(REPEAT, start=0, reference="GRCh38", contig="chr1")


def key(pos, ref, alt, reference="GRCh38", contig="chr1"):
    return normalize.VariantKey(reference, contig, pos, ref, alt)


def test_equivalent_insertions_collapse_to_one_key():
    """The same inserted CAG unit has several valid representations."""
    expected = key(1, "G", "GCAG")
    results = [
        normalize.normalize_variant(WINDOW, 1, "G", "GCAG"),
        normalize.normalize_variant(WINDOW, 2, "C", "CAGC"),
        normalize.normalize_variant(WINDOW, 3, "A", "AGCA"),
        normalize.normalize_variant(WINDOW, 4, "G", "GCAG"),
        normalize.normalize_variant(WINDOW, 5, "C", "CAGC"),
        normalize.normalize_variant(WINDOW, 8, "C", "CAGC"),
    ]
    assert {result.key for result in results} == {expected}
    assert all(result.resolved for result in results)


def test_equivalent_deletions_collapse_to_one_key():
    expected = key(1, "GCAG", "G")
    for pos in (1, 4, 7, 10, 13):
        result = normalize.normalize_variant(WINDOW, pos, REPEAT[pos - 1 : pos + 3], REPEAT[pos - 1])
        assert result.key == expected


def test_single_base_repeat_insertion_collapses():
    window = normalize.ReferenceWindow("CAAAAAAT", reference="GRCh38", contig="chr1")
    keys = {
        normalize.normalize_variant(window, pos, "A", "AA").key
        for pos in range(2, 7)
    }
    assert keys == {key(1, "C", "CA")}


def test_unchanged_alleles_are_kept():
    result = normalize.normalize_variant(WINDOW, 2, "C", "T")
    assert result.key == key(2, "C", "T")
    assert result.status == "unchanged"


def test_reference_allele_with_dot_alt():
    result = normalize.normalize_variant(WINDOW, 2, "C", ".")
    assert result.normalized.alt == "."
    assert result.status == "reference"


def test_symbolic_and_spanning_deletion_alleles_are_not_shifted():
    for alt in ("<DEL>", "*", "N[chr2:10["):
        result = normalize.normalize_variant(WINDOW, 2, "C", alt)
        assert result.normalized.pos == 2
        assert result.normalized.alt == alt
        assert result.status == "not_normalizable"


def test_ref_mismatch_raises():
    with pytest.raises(normalize.ReferenceMismatchError):
        normalize.normalize_variant(WINDOW, 2, "A", "T")


def test_variant_outside_window_raises():
    window = normalize.ReferenceWindow("ACGT", start=10, reference="GRCh38")
    with pytest.raises(normalize.ReferenceWindowError):
        normalize.normalize_variant(window, 100, "A", "T")


def test_left_truncated_repeat_is_unresolved():
    """Without the preceding G the leftmost representation cannot be proven."""
    truncated = normalize.ReferenceWindow(REPEAT[3:10], start=3, reference="GRCh38", contig="chr1")
    result = normalize.normalize_variant(truncated, 7, "G", "GCAG")
    assert not result.resolved
    assert result.key is None
    assert result.unresolved_range is not None
    assert result.unresolved_range.open_left
    assert result.unresolved_range.end == 7
    assert result.unresolved_range.start == 1


def test_left_truncated_but_unambiguous_variant_is_resolved():
    """A SNV or an indel not touching the window edge does not need extra context."""
    window = normalize.ReferenceWindow("ACGT", start=10, reference="GRCh38", contig="chr1")
    assert normalize.normalize_variant(window, 11, "A", "T").key == key(11, "A", "T")
    assert normalize.normalize_variant(window, 12, "C", "CA").key == key(12, "C", "CA")
    assert normalize.normalize_variant(window, 12, "CG", "C").key == key(12, "CG", "C")


def test_complete_left_at_contig_start_is_resolved():
    window = normalize.ReferenceWindow(
        "CAGCAGCAGT",
        start=1,
        reference="GRCh38",
        contig="chr1",
        complete_left=True,
    )
    result = normalize.normalize_variant(window, 2, "C", "CAGC")
    assert result.resolved
    assert result.key == key(2, "C", "CAGC")


def test_identical_truncated_strings_do_not_merge_across_references():
    window38 = normalize.ReferenceWindow(REPEAT[3:10], start=3, reference="GRCh38", contig="chr1")
    window37 = normalize.ReferenceWindow(REPEAT[3:10], start=3, reference="GRCh37", contig="chr1")
    range38 = normalize.normalize_variant(window38, 7, "G", "GCAG").unresolved_range
    range37 = normalize.normalize_variant(window37, 7, "G", "GCAG").unresolved_range
    assert range38.ref == range37.ref and range38.alt == range37.alt
    assert range38.key != range37.key
    assert range38.reference == "GRCh38"
    assert range37.reference == "GRCh37"


def test_business_key_contains_reference_version():
    result = normalize.normalize_variant(WINDOW, 2, "C", "T")
    assert result.key.reference == "GRCh38"
    assert result.key.contig == "chr1"
    assert result.key == ("GRCh38", "chr1", 2, "C", "T")
    assert str(result.key) == "GRCh38:chr1:2:C:T"


def test_batch_keeps_mapping_from_every_original_alt():
    batch = normalize.normalize_alleles(WINDOW, 2, "C", ["CAGC", "T", "A"])
    assert batch.allele_map["CAGC"] == key(1, "G", "GCAG")
    assert batch.allele_map["T"] == key(2, "C", "T")
    assert batch.allele_map["A"] == key(2, "C", "A")
    assert set(batch.allele_map) == {"CAGC", "T", "A"}


def test_batch_groups_original_alleles_by_key():
    batch = normalize.normalize_variants(
        WINDOW,
        [
            (2, "C", "CAGC"),
            (4, "G", "GCAG"),
            (7, "G", "GCAG"),
        ],
    )
    groups = batch.groups
    assert list(groups) == [key(1, "G", "GCAG")]
    assert [allele.label for allele in groups[key(1, "G", "GCAG")]] == [
        "2:C>CAGC",
        "4:G>GCAG",
        "7:G>GCAG",
    ]


def test_batch_reports_unresolved_ranges_separately():
    truncated = normalize.ReferenceWindow(REPEAT[3:10], start=3, reference="GRCh38", contig="chr1")
    batch = normalize.normalize_alleles(truncated, 7, "G", ["GCAG", "A"])
    assert len(batch.resolved) == 1
    assert len(batch.unresolved) == 1
    assert batch.allele_map["A"] == key(7, "G", "A")
    assert batch.allele_map["GCAG"].end == 7
    assert batch.unresolved_groups


def test_normalize_variants_accepts_tuples_mappings_and_records():
    batch = normalize.normalize_variants(
        WINDOW,
        [
            (2, "C", "CAGC"),
            {"pos": 4, "ref": "G", "alt": "GCAG"},
        ],
    )
    # Both tuples describe the same insertion, so the original alleles map to
    # the same key; the ALT string alone is not unique across positions.
    assert len(batch.results) == 2
    assert {result.key for result in batch.results} == {key(1, "G", "GCAG")}

    class Alt:
        def __init__(self, value):
            self.value = value

    class Record:
        POS = 2
        REF = "C"
        ALT = [Alt("CAGC"), Alt("T")]

    record_batch = normalize.normalize_variants(WINDOW, Record())
    assert record_batch.allele_map["CAGC"] == key(1, "G", "GCAG")
    assert record_batch.allele_map["T"] == key(2, "C", "T")


def test_normalize_record_does_not_modify_input():
    class Alt:
        def __init__(self, value):
            self.value = value

    class Record:
        POS = 2
        REF = "C"
        ALT = [Alt("CCAG")]

    record = Record()
    normalize.normalize_record(record, WINDOW)
    assert record.POS == 2
    assert record.REF == "C"
    assert record.ALT[0].value == "CCAG"


def test_trim_alleles_keeps_anchor_base():
    assert normalize.trim_alleles(10, "GCAG", "G") == (10, "GCAG", "G")
    assert normalize.trim_alleles(10, "TTTT", "TTT") == (10, "TT", "T")
    assert normalize.trim_alleles(10, "ACGT", "AGGT") == (11, "C", "G")


def test_reference_window_bounds_and_aliases():
    window = normalize.ReferenceWindow.from_bounds("ACGT", 10, 14, reference="GRCh38", contig="chr1")
    assert (window.start, window.end) == (10, 14)
    assert (window.start_1based, window.end_1based) == (11, 14)
    assert window.version == "GRCh38"
    assert window.chrom == "chr1"
    assert window.slice(11, 13) == "CG"
    assert window.contains(12, 2)
    assert not window.contains(14, 2)


def test_reference_window_rejects_inconsistent_bounds():
    with pytest.raises(ValueError):
        normalize.ReferenceWindow.from_bounds("ACGT", 10, 13, reference="GRCh38")


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        normalize.normalize_variant(WINDOW, 0, "C", "T")
    with pytest.raises(ValueError):
        normalize.normalize_variant(WINDOW, 2, "", "T")
    with pytest.raises(ValueError):
        normalize.normalize_variants(WINDOW, ["just-a-string"])


def test_sequence_can_be_passed_without_window_object():
    result = normalize.normalize_variant(
        REPEAT,
        2,
        "C",
        "CAGC",
        reference="GRCh38",
        contig="chr1",
    )
    assert result.key == key(1, "G", "GCAG")
