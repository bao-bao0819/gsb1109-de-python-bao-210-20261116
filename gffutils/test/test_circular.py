"""
Tests for features that cross the origin of circular sequences.
"""

import os
from textwrap import dedent

import pytest

import gffutils
from gffutils import circular
from gffutils.exceptions import MissingSequenceLengthError


GFF = dedent(
    """\
    ##gff-version 3
    chr1\t.\tgene\t90\t10\t.\t+\t.\tID=g1;Name=wrapped_plus
    chr1\t.\tmRNA\t90\t10\t.\t+\t.\tID=m1;Parent=g1
    chr1\t.\texon\t90\t20\t.\t+\t.\tID=e1;Parent=m1
    chr1\t.\texon\t30\t50\t.\t+\t.\tID=e2;Parent=m1
    chr1\t.\tgene\t70\t5\t.\t-\t.\tID=g2;Name=wrapped_minus
    chr1\t.\tgene\t40\t60\t.\t+\t.\tID=g3;Name=linear
    chr1\t.\tgene\t95\t99\t.\t-\t.\tID=g4;Name=near_origin
    """
)

SEQUENCE_LENGTH = 100


@pytest.fixture
def gff_path(tmp_path):
    path = tmp_path / "circular.gff"
    path.write_text(GFF)
    return str(path)


@pytest.fixture
def db(gff_path):
    return gffutils.create_db(
        gff_path,
        ":memory:",
        keep_order=True,
        sequence_lengths={"chr1": SEQUENCE_LENGTH},
    )


@pytest.fixture
def db_no_length(gff_path):
    return gffutils.create_db(gff_path, ":memory:", keep_order=True)


# Coordinate helpers --------------------------------------------------------


def test_coverage_segments_linear():
    assert circular.coverage_segments(40, 60) == ((40, 60),)
    assert circular.covered_length(40, 60) == 21


def test_coverage_segments_wrapped():
    assert circular.coverage_segments(90, 10, 100) == ((1, 10), (90, 100))
    assert circular.covered_length(90, 10, 100) == 21
    assert circular.logical_span(90, 10, 100) == (90, 10, 21)


def test_coverage_requires_length():
    with pytest.raises(MissingSequenceLengthError):
        circular.coverage_segments(90, 10)
    with pytest.raises(MissingSequenceLengthError):
        circular.covered_length(90, 10)


def test_coverage_rejects_impossible_coordinates():
    # start is beyond the sequence, so this cannot be a wrap of that length
    with pytest.raises(ValueError):
        circular.coverage_segments(120, 10, 100)


def test_coverage_segments_strand_order():
    # ascending genomic order by default
    assert circular.coverage_segments(90, 10, 100) == ((1, 10), (90, 100))
    # annotation order follows the feature's start -> end traversal
    assert circular.coverage_segments(90, 10, 100, order="annotation") == (
        (90, 100),
        (1, 10),
    )


def test_merge_segments():
    assert circular.merge_segments([(1, 10), (5, 15), (90, 100)]) == [
        (1, 15),
        (90, 100),
    ]
    assert circular.merge_segments([(11, 20), (1, 10)]) == [(1, 20)]


def test_circular_extent_prefers_small_interval():
    # Gap across the origin (79 bp) is bigger than the inner gap (9 bp), so
    # the extent is the small, non-wrapping interval.
    assert circular.circular_extent([(10, 20), (30, 40)], 100) == (10, 40)
    # Here the gap across the origin (39 bp) is the biggest, so the smallest
    # containing arc is the non-wrapping 10..70.
    assert circular.circular_extent([(10, 20), (30, 70)], 100) == (10, 70)
    # The inner gap (19 bp) is now the biggest, so the extent wraps across
    # the origin, giving (40, 20).
    assert circular.circular_extent([(10, 20), (40, 100)], 100) == (40, 20)
    # At most one segment: unchanged.
    assert circular.circular_extent([(10, 20)], 100) == (10, 20)


def test_segment_predicates():
    assert circular.segments_overlap(((1, 10), (90, 100)), ((95, 100),))
    assert circular.segments_overlap(((1, 10), (90, 100)), ((1, 5),))
    assert not circular.segments_overlap(((1, 10), (90, 100)), ((30, 40),))
    assert circular.segments_within(
        ((1, 10), (90, 100)), ((90, 100), (1, 50))
    )
    assert not circular.segments_within(((1, 20),), ((1, 10),))


def test_load_sequence_lengths(tmp_path):
    chromsizes = tmp_path / "chrom.sizes"
    chromsizes.write_text("chr1 100\nchr2\t50\n")
    assert circular.load_sequence_lengths(str(chromsizes)) == {
        "chr1": 100,
        "chr2": 50,
    }
    assert circular.load_sequence_lengths({"chr1": "100"}) == {"chr1": 100}
    assert circular.load_sequence_lengths(100) == {None: 100}
    assert circular.load_sequence_lengths([("chr1", 100)]) == {"chr1": 100}
    assert circular.load_sequence_lengths(None) == {}


# Feature-level behavior ----------------------------------------------------


def test_feature_reports_logical_object(db):
    gene = db["g1"]
    assert gene.crosses_origin
    assert gene.segments == ((90, 100), (1, 10))
    assert gene.covered_segments == gene.segments
    assert gene.logical_span() == (90, 10, 21)
    assert len(gene) == 21
    logical = gene.logical()
    assert logical.start == 90
    assert logical.end == 10
    assert logical.length == 21
    assert logical.segments == ((90, 100), (1, 10))
    assert logical.feature is gene


def test_feature_strand_preserved(db):
    assert db["g1"].strand == "+"
    assert db["g2"].strand == "-"


def test_linear_feature_unaffected(db):
    gene = db["g3"]
    assert gene.crosses_origin is False
    assert gene.segments == ((40, 60),)
    assert len(gene) == 21


def test_missing_length_leaves_feature_undecided(db_no_length):
    gene = db_no_length["g1"]
    assert gene.crosses_origin is None
    with pytest.raises(MissingSequenceLengthError):
        len(gene)
    with pytest.raises(MissingSequenceLengthError):
        gene.segments
    # The coordinates themselves are preserved, not swapped.
    assert (gene.start, gene.end) == (90, 10)


def test_gff_round_trip_preserves_wrapped_coordinates(db):
    line = str(db["g1"])
    assert line.split("\t")[3:5] == ["90", "10"]


def test_parent_child_relations_unchanged(db):
    assert [c.id for c in db.children("g1")] == ["e1", "e2", "m1"]
    assert [p.id for p in db.parents("e1", level=1)] == ["m1"]
    assert [f.featuretype for f in db.children("g1")] == ["exon", "exon", "mRNA"]


def test_children_bp_uses_actual_coverage(db):
    # e1 is 90..20 -> 31 bp; e2 is 30..50 -> 21 bp
    assert len(db["e1"]) == 31
    assert db.children_bp("m1", child_featuretype="exon") == 31 + 21


# Region queries ------------------------------------------------------------


def test_wrapped_region_returns_each_feature_once(db):
    # The query crosses the origin and must not double-count.
    hits = list(db.region(region=("chr1", 95, 5)))
    ids = [f.id for f in hits]
    assert len(ids) == len(set(ids))
    assert set(ids) == {"g1", "m1", "e1", "g2", "g4"}


def test_region_matches_either_segment(db):
    assert {f.id for f in db.region(region=("chr1", 5, 15))} == {"g1", "m1", "e1", "g2"}
    assert {f.id for f in db.region(region=("chr1", 85, 100))} == {
        "g1",
        "m1",
        "e1",
        "g2",
        "g4",
    }


def test_region_does_not_select_linear_span(db):
    # 40-60 overlaps e2 (30-50) and g3 (40-60); the wrapped genes are
    # elsewhere on the circle and must not be pulled in by a linear query.
    assert {f.id for f in db.region(region=("chr1", 40, 60))} == {"e2", "g3"}


def test_region_completely_within_wrapped_query(db):
    # Everything covered by this query is 1..10 and 90..100; e1 (90..20)
    # extends to 20, so it is not completely within.
    hits = {f.id for f in db.region(region=("chr1", 90, 10), completely_within=True)}
    assert "g1" in hits
    assert "e1" not in hits


def test_region_wrapped_query_without_length_is_undecided(db_no_length):
    with pytest.raises(MissingSequenceLengthError):
        list(db_no_length.region(region=("chr1", 95, 5)))


def test_region_linear_query_does_not_guess_wrapped_features(db_no_length):
    # Without a sequence length a wrapped feature's coverage is undecided.  A
    # query that is provably outside every wrapped feature resolves normally,
    # and the wrapped features are not pulled in by swapping endpoints (which
    # would select most of the chromosome).
    assert {f.id for f in db_no_length.region(region=("chr1", 40, 60))} == {
        "e2",
        "g3",
    }
    # This query could touch the wrapped features' low tails, so it is reported
    # as undecided rather than silently wrong.
    with pytest.raises(MissingSequenceLengthError):
        list(db_no_length.region(region=("chr1", 5, 50)))


def test_region_without_wrapped_features_unaffected(tmp_path):
    # With no wrapped features on the sequence, a linear query behaves exactly
    # as before and does not raise.
    path = tmp_path / "linear.gff"
    path.write_text(
        dedent(
            """\
            ##gff-version 3
            chr2\t.\tgene\t40\t60\t.\t+\t.\tID=a
            chr2\t.\tgene\t10\t20\t.\t+\t.\tID=b
            """
        )
    )
    db = gffutils.create_db(str(path), ":memory:")
    assert {f.id for f in db.region(region=("chr2", 40, 60))} == {"a"}
    assert {f.id for f in db.region(region=("chr2", 1, 25))} == {"b"}


def test_region_skips_provably_disjoint_wrapped_features(gff_path):
    # g4 (95..99) is linear; the wrapped features are on another chromosome,
    # so a query on that chromosome must not raise because of them.
    path = gff_path
    db = gffutils.create_db(
        path,
        ":memory:",
        from_string=False,
    )
    # All features here are on chr1, but a query on a feature-free seqid must
    # still work.
    assert list(db.region(region=("chrX", 1, 100))) == []


def test_open_ended_queries(db):
    # start only: features at or after 85 (including the wrapped ones)
    assert {f.id for f in db.region(seqid="chr1", start=85)} == {
        "g1",
        "m1",
        "e1",
        "g2",
        "g4",
    }
    # end only: features at or before 15 (including wrapped ones)
    assert {f.id for f in db.region(seqid="chr1", end=15)} == {
        "g1",
        "m1",
        "e1",
        "g2",
    }


def test_open_ended_queries_without_length_are_undecided(db_no_length):
    # Without a length, an open-ended query cannot rule out a wrapped feature
    # (one of its segments always reaches the origin).
    with pytest.raises(MissingSequenceLengthError):
        list(db_no_length.region(seqid="chr1", start=85))
    with pytest.raises(MissingSequenceLengthError):
        list(db_no_length.region(seqid="chr1", end=15))


def test_strand_filter_still_applies(db):
    assert {f.id for f in db.region(region=("chr1", 95, 5), strand="-")} == {
        "g2",
        "g4",
    }


# Sequence extraction -------------------------------------------------------


def test_sequence_of_wrapped_feature(gff_path, tmp_path):
    fasta_path = tmp_path / "chr1.fa"
    sequence = "ACGT" * 25
    fasta_path.write_text(">chr1\n" + sequence + "\n")

    db = gffutils.create_db(
        gff_path,
        ":memory:",
        sequence_lengths={"chr1": 100},
    )
    gene = db["g1"]
    # plus strand: positions 90..100 then 1..10 in annotation order
    expected = sequence[89:100] + sequence[0:10]
    assert gene.sequence(str(fasta_path), use_strand=False) == expected
    assert gene.sequence(str(fasta_path), use_strand=True) == expected


def test_sequence_of_wrapped_minus_feature(gff_path, tmp_path):
    fasta_path = tmp_path / "chr1.fa"
    sequence = "ACGT" * 25
    fasta_path.write_text(">chr1\n" + sequence + "\n")

    db = gffutils.create_db(
        gff_path,
        ":memory:",
        sequence_lengths={"chr1": 100},
    )
    gene = db["g2"]
    # g2 is 70..5: 70..100 followed by 1..5 in annotation order
    forward = sequence[69:100] + sequence[0:5]
    assert gene.sequence(str(fasta_path), use_strand=False) == forward
    assert gene.sequence(str(fasta_path), use_strand=True) == (
        forward[::-1].translate(circular._COMPLEMENT)
    )


def test_sequence_length_can_come_from_fasta(db_no_length, tmp_path):
    # The database has no stored length, but the FASTA does: the feature is
    # resolved rather than left undecided.
    fasta_path = tmp_path / "chr1.fa"
    sequence = "ACGT" * 25
    fasta_path.write_text(">chr1\n" + sequence + "\n")
    assert db_no_length["g1"].sequence(str(fasta_path)) == (
        sequence[89:100] + sequence[0:10]
    )


# create_db / FeatureDB plumbing -------------------------------------------


def test_sequence_lengths_persisted(gff_path, tmp_path):
    dbfn = str(tmp_path / "circular.db")
    gffutils.create_db(
        gff_path,
        dbfn,
        sequence_lengths={"chr1": SEQUENCE_LENGTH},
    )
    db = gffutils.FeatureDB(dbfn)
    assert db.sequence_length("chr1") == SEQUENCE_LENGTH
    assert len(db["g1"]) == 21


def test_sequence_lengths_from_chromsizes(gff_path, tmp_path):
    chromsizes = tmp_path / "chrom.sizes"
    chromsizes.write_text("chr1 %s\n" % SEQUENCE_LENGTH)
    db = gffutils.create_db(
        gff_path, ":memory:", sequence_lengths=str(chromsizes)
    )
    assert db.sequence_length("chr1") == SEQUENCE_LENGTH
    assert len(db["g1"]) == 21


def test_sequence_lengths_from_fasta(gff_path, tmp_path):
    fasta_path = tmp_path / "chr1.fa"
    fasta_path.write_text(">chr1\n" + "ACGT" * 25 + "\n")
    db = gffutils.create_db(
        gff_path, ":memory:", sequence_lengths=str(fasta_path)
    )
    assert db.sequence_length("chr1") == SEQUENCE_LENGTH


def test_set_sequence_lengths_updates_existing_features(db_no_length):
    with pytest.raises(MissingSequenceLengthError):
        len(db_no_length["g1"])

    db_no_length.set_sequence_lengths({"chr1": SEQUENCE_LENGTH})

    assert len(db_no_length["g1"]) == 21
    assert db_no_length["g1"].crosses_origin is True
    assert len(list(db_no_length.region(region=("chr1", 95, 5)))) == 5


def test_unknown_sequence_length_is_none(db):
    assert db.sequence_length("chrX") is None


def test_feature_touching_origin_but_not_wrapping(db):
    # 95..99 does not wrap; it must not be reported as circular.
    gene = db["g4"]
    assert gene.crosses_origin is False
    assert gene.segments == ((95, 99),)
    assert len(gene) == 5


def test_wrapped_query_that_covers_whole_sequence(db):
    # 1..100 is the whole chromosome; every feature is covered.
    assert {f.id for f in db.region(region=("chr1", 1, 100))} == {
        "g1",
        "g2",
        "g3",
        "g4",
        "m1",
        "e1",
        "e2",
    }


def test_region_results_are_deduplicated_for_both_segments(db):
    # 95..5 hits g1 through both of its segments; it must still appear once.
    ids = [f.id for f in db.region(region=("chr1", 95, 5))]
    assert ids.count("g1") == 1
    assert len(ids) == len(set(ids))


def test_sequence_length_must_contain_wrapped_coordinates(db):
    with pytest.raises(ValueError):
        circular.coverage_segments(90, 10, 80)


def test_bin_of_wrapped_feature_is_whole_chromosome(db):
    # A feature that crosses the origin cannot fit in a small bin; using a
    # small bin would make region() miss it.
    assert db["g1"].bin == 1


def test_children_bp_without_length_raises(gff_path):
    db = gffutils.create_db(gff_path, ":memory:")
    with pytest.raises(MissingSequenceLengthError):
        db.children_bp("m1", child_featuretype="exon")


def test_sanitize_preserves_wrapped_feature_with_length(gff_path):
    from gffutils import helpers

    db = gffutils.create_db(
        gff_path, ":memory:", sequence_lengths={"chr1": SEQUENCE_LENGTH}
    )
    sanitized = helpers.sanitize_gff_db(db)
    gene = sanitized["g1"]
    assert (gene.start, gene.end) == (90, 10)
    assert sanitized.sequence_lengths["chr1"] == SEQUENCE_LENGTH


def test_sanitize_swaps_wrapped_feature_without_length(gff_path):
    from gffutils import helpers

    db = gffutils.create_db(gff_path, ":memory:")
    sanitized = helpers.sanitize_gff_db(db)
    gene = sanitized["g1"]
    assert (gene.start, gene.end) == (10, 90)


def test_logical_feature_preserves_relationship_and_strand(db):
    gene = db["g1"]
    logical = gene.logical()
    assert logical.strand == gene.strand
    assert logical.seqid == gene.seqid
    assert logical.featuretype == gene.featuretype
    assert logical.attributes is gene.attributes
    # Attribute-style access falls through to the wrapped Feature.
    assert logical.chrom == "chr1"
    assert repr(logical).startswith("<LogicalFeature gene (chr1:90-10[+]")


GTF = dedent(
    """\
    chr1\t.\texon\t9\t3\t.\t+\t.\tgene_id "g1"; transcript_id "t1";
    chr1\t.\texon\t8\t9\t.\t+\t.\tgene_id "g1"; transcript_id "t1";
    """
)


def test_gtf_inferred_extent_wraps_with_length(tmp_path):
    path = tmp_path / "circular.gtf"
    path.write_text(GTF)
    db = gffutils.create_db(
        str(path),
        ":memory:",
        sequence_lengths={"chr1": 10},
        disable_infer_genes=False,
        disable_infer_transcripts=False,
    )
    gene = db["g1"]
    # exons 9..3 and 8..9 cover 8,9,10,1,2,3
    assert (gene.start, gene.end) == (8, 3)
    assert gene.crosses_origin is True
    assert len(gene) == 6
    transcript = db["t1"]
    assert (transcript.start, transcript.end) == (8, 3)


def test_gtf_inferred_extent_without_length_is_undecided(tmp_path):
    path = tmp_path / "circular.gtf"
    path.write_text(GTF)
    with pytest.raises(MissingSequenceLengthError):
        gffutils.create_db(
            str(path),
            ":memory:",
            disable_infer_genes=False,
            disable_infer_transcripts=False,
        )


def test_gtf_single_wrapped_exon_is_preserved(tmp_path):
    path = tmp_path / "circular.gtf"
    path.write_text(
        'chr1\t.\texon\t9\t3\t.\t+\t.\tgene_id "g1"; transcript_id "t1";\n'
    )
    db = gffutils.create_db(
        str(path),
        ":memory:",
        disable_infer_genes=False,
        disable_infer_transcripts=False,
    )
    assert (db["g1"].start, db["g1"].end) == (9, 3)
