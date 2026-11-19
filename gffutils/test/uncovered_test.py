"""
Tests for coverage-based "uncovered" (blank) intervals and boundary support.
"""

import random


import gffutils
from gffutils.feature import Feature
from gffutils.uncovered import SupportList, uncovered_intervals


def make_feature(feature_id, start, end, strand=".", seqid="chr1", featuretype="gene"):
    return Feature(
        seqid=seqid,
        source="test",
        featuretype=featuretype,
        start=start,
        end=end,
        strand=strand,
        attributes={"ID": [feature_id]},
        id=feature_id,
    )


def summarize(features):
    return [
        (
            feature.seqid,
            feature.start,
            feature.end,
            feature.left_support_ids,
            feature.right_support_ids,
        )
        for feature in features
    ]


def test_nested_and_overlapping_features_do_not_create_gaps():
    features = [
        make_feature("a", 1, 10),
        make_feature("b", 5, 20),
        make_feature("c", 15, 30),
        make_feature("d", 40, 50),
    ]
    result = list(uncovered_intervals(features))
    assert len(result) == 1
    gap = result[0]
    assert (gap.start, gap.end) == (31, 39)
    assert gap.left_support_ids == ("c",)
    assert gap.right_support_ids == ("d",)


def test_overlap_never_yields_negative_length():
    features = [make_feature("a", 10, 100), make_feature("b", 20, 30)]
    assert list(uncovered_intervals(features)) == []


def test_adjacent_features_do_not_create_gaps():
    features = [
        make_feature("a", 1, 10),
        make_feature("b", 11, 20),
        make_feature("c", 21, 21),
        make_feature("d", 22, 30),
    ]
    assert list(uncovered_intervals(features)) == []


def test_boundary_support_uses_rightmost_covering_feature():
    # 'a' spans much of the region but stops before 'b'; 'b' ends at exactly
    # the covered edge and is the feature actually supporting the left edge of
    # the gap.
    features = [
        make_feature("a", 1, 45),
        make_feature("b", 40, 50),
        make_feature("c", 60, 70),
    ]
    gaps = list(uncovered_intervals(features))
    assert len(gaps) == 1
    assert (gaps[0].start, gaps[0].end) == (51, 59)
    assert gaps[0].left_support_ids == ("b",)
    assert gaps[0].right_support_ids == ("c",)


def test_multiple_supporters_on_each_boundary():
    features = [
        make_feature("a", 1, 10),
        make_feature("b", 1, 10),
        make_feature("c", 20, 30),
        make_feature("d", 20, 30),
    ]
    gaps = list(uncovered_intervals(features))
    assert len(gaps) == 1
    assert gaps[0].left_support_ids == ("a", "b")
    assert gaps[0].right_support_ids == ("c", "d")


def test_gap_sources_independent_of_input_order():
    features = [
        make_feature("a", 1, 10),
        make_feature("b", 5, 20),
        make_feature("c", 15, 30),
        make_feature("d", 40, 50),
        make_feature("e", 60, 70),
    ]
    expected = summarize(uncovered_intervals(features))
    random.seed(0)
    for _ in range(20):
        shuffled = list(features)
        random.shuffle(shuffled)
        assert summarize(uncovered_intervals(shuffled)) == expected


def test_gap_featuretype_reflects_boundaries():
    features = [
        make_feature("a", 1, 10, featuretype="exon"),
        make_feature("b", 20, 30, featuretype="CDS"),
    ]
    gap = next(uncovered_intervals(features))
    assert gap.featuretype == "inter_exon_CDS"


def test_start_and_end_intervals_with_sequence_bounds():
    features = [
        make_feature("a", 10, 20),
        make_feature("b", 30, 40),
    ]
    gaps = list(uncovered_intervals(features, chromsizes={"chr1": 50}))
    assert [(gap.start, gap.end) for gap in gaps] == [(1, 9), (21, 29), (41, 50)]
    assert gaps[0].right_support_ids == ("a",)
    assert gaps[1].left_support_ids == ("a",)
    assert gaps[1].right_support_ids == ("b",)
    assert gaps[2].left_support_ids == ("b",)


def test_sequence_with_no_features_is_entirely_uncovered():
    gaps = list(uncovered_intervals([make_feature("a", 1, 10)], bounds={"chr2": 25}))
    assert len(gaps) == 1
    assert gaps[0].seqid == "chr2"
    assert (gaps[0].start, gaps[0].end) == (1, 25)
    assert gaps[0].left_support_ids == ()
    assert gaps[0].right_support_ids == ()


def test_strand_aware_mode():
    features = [
        make_feature("plus1", 1, 10, strand="+"),
        make_feature("minus1", 5, 20, strand="-"),
        make_feature("plus2", 15, 30, strand="+"),
    ]
    union = list(uncovered_intervals(features))
    assert union == []

    per_strand = list(uncovered_intervals(features, ignore_strand=False))
    assert [(gap.strand, gap.start, gap.end) for gap in per_strand] == [("+", 11, 14)]


def test_strand_filter():
    features = [
        make_feature("plus1", 1, 10, strand="+"),
        make_feature("minus1", 1, 20, strand="-"),
        make_feature("plus2", 15, 30, strand="+"),
    ]
    gaps = list(uncovered_intervals(features, strand="+"))
    assert [(gap.strand, gap.start, gap.end) for gap in gaps] == [("+", 11, 14)]


def test_merge_criteria_strand_callback_enables_per_strand():
    from gffutils import merge_criteria as mc

    features = [
        make_feature("plus1", 1, 10, strand="+"),
        make_feature("minus1", 5, 20, strand="-"),
        make_feature("plus2", 15, 30, strand="+"),
    ]
    gaps = list(uncovered_intervals(features, merge_criteria=(mc.seqid, mc.strand)))
    assert [(gap.strand, gap.start, gap.end) for gap in gaps] == [("+", 11, 14)]


def test_multiple_seqids():
    features = [
        make_feature("a", 1, 10, seqid="chr2"),
        make_feature("b", 20, 30, seqid="chr1"),
        make_feature("c", 40, 50, seqid="chr2"),
    ]
    gaps = list(uncovered_intervals(features))
    assert [(gap.seqid, gap.start, gap.end) for gap in gaps] == [
        ("chr2", 11, 39),
    ]
    gaps = list(uncovered_intervals(features, chromsizes={"chr1": 100, "chr2": 100}))
    assert [(gap.seqid, gap.start, gap.end) for gap in gaps] == [
        ("chr1", 1, 19),
        ("chr1", 31, 100),
        ("chr2", 11, 39),
        ("chr2", 51, 100),
    ]


def test_invalid_coordinates_are_ignored():
    features = [
        make_feature("empty", 10, 10),
        make_feature("backwards", 30, 20),
        make_feature("a", 1, 5),
        make_feature("b", 8, 12),
    ]
    gaps = list(uncovered_intervals(features))
    assert [(gap.start, gap.end) for gap in gaps] == [(6, 7)]


def test_missing_coordinates_are_ignored():
    feature = Feature(
        seqid="chr1",
        source="test",
        featuretype="gene",
        start=None,
        end=None,
        attributes={"ID": ["x"]},
        id="x",
    )
    assert list(uncovered_intervals([feature])) == []


def test_attributes_and_id_concatenation():
    features = [
        make_feature("a", 1, 10),
        make_feature("b", 20, 30),
    ]
    gap = next(uncovered_intervals(features))
    assert gap.attributes["ID"] == ["a-b"]


def test_support_attributes_are_opt_in():
    features = [
        make_feature("a", 1, 10),
        make_feature("b", 20, 30),
    ]
    gap = next(uncovered_intervals(features))
    assert "left_support" not in gap.attributes
    assert gap.left_support_ids == ("a",)

    gap = next(uncovered_intervals(features, include_support_attributes=True))
    assert gap.attributes["left_support"] == ["a"]
    assert gap.attributes["right_support"] == ["b"]


def test_support_list_comparison_helpers():
    features = [make_feature("a", 1, 10), make_feature("b", 20, 30)]
    gap = next(uncovered_intervals(features))
    assert isinstance(gap.left_support, SupportList)
    assert gap.left_support == ["a"]
    assert gap.left_support == ("a",)
    assert gap.left_support != ["b"]


def test_does_not_mutate_input_features():
    features = [make_feature("a", 1, 10), make_feature("b", 20, 30)]
    before = [str(feature) for feature in features]
    for _ in uncovered_intervals(features, include_support_attributes=True):
        pass
    assert [str(feature) for feature in features] == before


def test_database_method_matches_module_function():
    def generator():
        yield make_feature("a", 1, 10)
        yield make_feature("b", 20, 30)

    db = gffutils.create_db(generator(), ":memory:", merge_strategy="error")
    observed = [
        (gap.start, gap.end, gap.left_support_ids, gap.right_support_ids)
        for gap in db.uncovered_intervals()
    ]
    assert observed == [(11, 19, ("a",), ("b",))]


def test_issue_197_nested_overlaps():
    # The fixture documents that after merging overlapping genes, the next
    # uncovered region should be 50086-50089.
    db = gffutils.create_db(
        gffutils.example_filename("issue_197.gff"),
        ":memory:",
        merge_strategy="error",
    )
    genes = list(db.features_of_type("gene"))
    gaps = list(db.interfeatures(genes, new_featuretype="intergenic_space"))
    observed = [(gap.seqid, gap.start, gap.end) for gap in gaps]
    assert observed == [
        ("tig00000492", 47236, 47350),
        ("tig00000492", 48257, 49999),
        ("tig00000492", 50086, 50089),
    ]


def test_introns_still_work_for_non_overlapping_exons():
    db = gffutils.create_db(gffutils.example_filename("FBgn0031208.gff"), ":memory:")
    introns = list(db.create_introns())
    assert introns
    assert all(intron.start <= intron.end for intron in introns)


def test_alternate_aliases_agree():
    features = [
        make_feature("a", 1, 10),
        make_feature("b", 20, 30),
    ]
    expected = [(gap.start, gap.end) for gap in uncovered_intervals(features)]
    for alias in (gffutils.coverage_gaps, gffutils.blank_intervals):
        assert [(gap.start, gap.end) for gap in alias(features)] == expected


def test_database_aliases_agree():
    def generator():
        yield make_feature("a", 1, 10)
        yield make_feature("b", 20, 30)

    db = gffutils.create_db(generator(), ":memory:", merge_strategy="error")
    expected = [
        (gap.start, gap.end) for gap in db.uncovered_intervals(include_ends=False)
    ]
    for alias in (db.coverage_gaps, db.blank_intervals):
        assert [(gap.start, gap.end) for gap in alias(include_ends=False)] == expected
