import itertools

from gffutils.feature import feature_from_line
from gffutils.helpers import merge_segments


def make_segment(
    seqid="chr1",
    source="src",
    start=1,
    end=10,
    strand="+",
    featuretype="exon",
    attributes="ID=x",
):
    line = "\t".join(
        [seqid, source, featuretype, str(start), str(end), ".", strand, ".", attributes]
    )
    return feature_from_line(line)


class TestMergeSegments:
    def test_empty(self):
        merged, disputes = merge_segments([])
        assert merged == []
        assert disputes == []

    def test_single_segment(self):
        seg = make_segment()
        merged, disputes = merge_segments([seg])
        assert len(merged) == 1
        assert disputes == []
        assert merged[0].components == (seg,)
        assert (merged[0].start, merged[0].end) == (1, 10)

    def test_adjacent_compatible_segments_merge(self):
        a = make_segment(start=1, end=10, attributes="gene_id=g1")
        b = make_segment(start=11, end=20, attributes="gene_id=g1")
        merged, disputes = merge_segments([a, b])
        assert disputes == []
        assert len(merged) == 1
        assert (merged[0].start, merged[0].end) == (1, 20)
        assert merged[0].components == (a, b)

    def test_distance_limit_is_caller_supplied(self):
        a = make_segment(start=1, end=10, attributes="gene_id=g1")
        b = make_segment(start=16, end=20, attributes="gene_id=g1")
        # gap of 5 (positions 11..15): not merged with the default limit
        merged, disputes = merge_segments([a, b])
        assert len(merged) == 2
        assert disputes == []
        # merged when the caller allows a distance of 5
        merged, disputes = merge_segments([a, b], max_distance=5)
        assert len(merged) == 1
        assert (merged[0].start, merged[0].end) == (1, 20)
        assert disputes == []
        # but not when the limit is 4
        merged, disputes = merge_segments([a, b], max_distance=4)
        assert len(merged) == 2
        assert disputes == []

    def test_strand_must_match(self):
        a = make_segment(start=1, end=10, strand="+", attributes="gene_id=g1")
        b = make_segment(start=5, end=20, strand="-", attributes="gene_id=g1")
        merged, disputes = merge_segments([a, b])
        assert len(merged) == 2
        assert disputes == []

    def test_source_must_match(self):
        a = make_segment(start=1, end=10, source="src1", attributes="gene_id=g1")
        b = make_segment(start=5, end=20, source="src2", attributes="gene_id=g1")
        merged, disputes = merge_segments([a, b])
        assert len(merged) == 2
        assert disputes == []

    def test_seqid_must_match(self):
        a = make_segment(seqid="chr1", start=1, end=10, attributes="gene_id=g1")
        b = make_segment(seqid="chr2", start=5, end=20, attributes="gene_id=g1")
        merged, disputes = merge_segments([a, b])
        assert len(merged) == 2
        assert disputes == []

    def test_key_attributes_must_be_compatible(self):
        a = make_segment(start=1, end=10, attributes="ID=a;gene_id=g1")
        b = make_segment(start=5, end=20, attributes="ID=b;gene_id=g2")
        merged, disputes = merge_segments([a, b], key_attributes=["gene_id"])
        assert len(merged) == 2
        assert disputes == []

        c = make_segment(start=1, end=10, attributes="ID=a;gene_id=g1")
        d = make_segment(start=5, end=20, attributes="ID=b;gene_id=g1")
        merged, disputes = merge_segments([c, d], key_attributes=["gene_id"])
        assert len(merged) == 1
        assert disputes == []

    def test_missing_key_attribute_is_not_a_conflict(self):
        a = make_segment(start=1, end=10, attributes="ID=a;gene_id=g1")
        b = make_segment(start=5, end=20, attributes="ID=b")
        merged, disputes = merge_segments([a, b], key_attributes=["gene_id"])
        assert len(merged) == 1
        assert disputes == []

    def test_all_attributes_checked_by_default(self):
        a = make_segment(start=1, end=10, attributes="ID=a;gene_id=g1")
        b = make_segment(start=5, end=20, attributes="ID=b;gene_id=g1")
        # IDs differ, so by default these are not compatible
        merged, disputes = merge_segments([a, b])
        assert len(merged) == 2
        assert disputes == []

    def test_contributions_and_merged_attributes(self):
        a = make_segment(start=1, end=10, attributes="gene_id=g1")
        b = make_segment(start=11, end=20, attributes="gene_id=g1;note=x")
        merged, disputes = merge_segments([a, b], key_attributes=["gene_id"])
        assert len(merged) == 1
        assert disputes == []
        assert merged[0].components == (a, b)
        # contributions of all original segments are reflected in the
        # merged attributes
        assert merged[0].attributes["gene_id"] == ["g1"]
        assert merged[0].attributes["note"] == ["x"]

    def test_chaining_through_middle_segment(self):
        # A and C are distance-linked through B, and each is compatible with
        # B, but A and C are incompatible with each other: chaining does not
        # make all attributes mergeable.
        a = make_segment(start=1, end=10, attributes="gene_id=g1")
        b = make_segment(start=11, end=20, attributes="gene_id=g1;note=x")
        c = make_segment(start=21, end=30, attributes="gene_id=g2")
        merged, disputes = merge_segments(
            [a, b, c], max_distance=0, key_attributes=["gene_id"]
        )
        # A and B merge; C is incompatible with both so it stands alone, and
        # that grouping is unique so there is no dispute.
        assert disputes == []
        assert len(merged) == 2
        assert merged[0].components == (a, b)
        assert merged[1].components == (c,)

    def test_disputed_range_reported_for_ambiguous_grouping(self):
        # A-B and B-C are pairwise compatible and adjacent, but A and C are
        # incompatible: {A, B} + {C} and {A} + {B, C} are both legal
        # groupings, so the range must be reported as disputed rather than
        # resolved arbitrarily.
        a = make_segment(start=1, end=10, attributes="ID=a;grp=x")
        b = make_segment(start=11, end=20, attributes="ID=b")
        c = make_segment(start=21, end=30, attributes="ID=c;grp=y")
        merged, disputes = merge_segments([a, b, c], key_attributes=["grp"])
        assert merged == []
        assert len(disputes) == 1
        dispute = disputes[0]
        assert (dispute.seqid, dispute.source, dispute.strand) == ("chr1", "src", "+")
        assert (dispute.start, dispute.end) == (1, 30)
        assert dispute.segments == (a, b, c)

    def test_dispute_resolution_not_order_dependent(self):
        a = make_segment(start=1, end=10, attributes="ID=a;grp=x")
        b = make_segment(start=11, end=20, attributes="ID=b")
        c = make_segment(start=21, end=30, attributes="ID=c;grp=y")
        results = set()
        for perm in itertools.permutations([a, b, c]):
            merged, disputes = merge_segments(perm, key_attributes=["grp"])
            results.add(
                (
                    tuple((m.start, m.end) for m in merged),
                    tuple((d.start, d.end) for d in disputes),
                )
            )
        assert len(results) == 1

    def test_result_independent_of_input_order(self):
        segments = [
            make_segment(start=1, end=10, attributes="gene_id=g1"),
            make_segment(start=11, end=20, attributes="gene_id=g1"),
            make_segment(start=100, end=110, strand="-", attributes="gene_id=g2"),
            make_segment(start=105, end=120, strand="-", attributes="gene_id=g2"),
            make_segment(start=500, end=510, source="other", attributes="gene_id=g3"),
        ]
        expected = None
        for perm in itertools.permutations(segments):
            merged, disputes = merge_segments(perm, key_attributes=["gene_id"])
            signature = (
                tuple(
                    (m.seqid, m.strand, m.source, m.start, m.end, len(m.components))
                    for m in merged
                ),
                tuple((d.start, d.end) for d in disputes),
            )
            if expected is None:
                expected = signature
            assert signature == expected
        merged, disputes = merge_segments(segments, key_attributes=["gene_id"])
        assert disputes == []
        assert len(merged) == 3
        assert sorted((m.start, m.end) for m in merged) == [
            (1, 20),
            (100, 120),
            (500, 510),
        ]

    def test_singletons_for_incompatible_neighbors(self):
        # distance-linked but incompatible on the key attribute: the only
        # legal grouping is singletons, so no dispute is reported
        a = make_segment(start=1, end=10, attributes="gene_id=g1")
        b = make_segment(start=5, end=20, attributes="gene_id=g2")
        merged, disputes = merge_segments([a, b], key_attributes=["gene_id"])
        assert disputes == []
        assert len(merged) == 2
        assert merged[0].components == (a,)
        assert merged[1].components == (b,)

    def test_key_attributes_accepts_single_string(self):
        a = make_segment(start=1, end=10, attributes="gene_id=g1")
        b = make_segment(start=5, end=20, attributes="gene_id=g1")
        merged, disputes = merge_segments([a, b], key_attributes="gene_id")
        assert len(merged) == 1
        assert disputes == []
