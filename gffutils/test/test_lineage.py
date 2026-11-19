"""
Tests for identity-preserving parent edges and the transcript/gene coverage
conventions implemented in gffutils.lineage and FeatureDB.

The motivating case is an exon shared by two transcripts: it is one source
line (one feature identity) with two parent edges. Expanding parent-child
rows must not turn it into two coverage contributions at the gene level,
and missing/conflicting parentage must stay unresolved rather than being
"repaired" by picking the first available parent.
"""

import warnings

import pytest

import gffutils
from gffutils import lineage as lin
from gffutils.exceptions import ParentResolutionError


SHARED_EXON_GFF = """##gff-version 3
chr1	test	gene	1	100	.	+	.	ID=gene1
chr1	test	mRNA	1	75	.	+	.	ID=tx1;Parent=gene1
chr1	test	mRNA	26	100	.	+	.	ID=tx2;Parent=gene1
chr1	test	exon	1	50	.	+	.	ID=exon_shared;Parent=tx1,tx2
chr1	test	exon	51	75	.	+	.	ID=exon_tx1;Parent=tx1
chr1	test	exon	76	100	.	+	.	ID=exon_tx2;Parent=tx2
"""


@pytest.fixture
def shared_db():
    return gffutils.create_db(
        SHARED_EXON_GFF,
        ":memory:",
        from_string=True,
        merge_strategy="create_unique",
    )


def test_shared_exon_keeps_single_identity_and_every_edge(shared_db):
    # One feature row for the shared exon...
    assert shared_db["exon_shared"].id == "exon_shared"
    # ...but one stored parent edge per transcript.
    edges = list(shared_db.parent_edges("exon_shared", level=1))
    assert [edge.parent for edge in edges] == ["tx1", "tx2"]
    assert all(edge.level == 1 for edge in edges)

    lineage = shared_db.lineage("exon_shared")
    assert isinstance(lineage, gffutils.FeatureLineage)
    assert lineage.feature_id == "exon_shared"
    assert lineage.parent_ids == ["tx1", "tx2"]
    assert lineage.status == gffutils.RESOLVED
    assert lineage.is_resolved


def test_transcript_scope_counts_each_parent_edge(shared_db):
    report = shared_db.coverage(scope="transcript")
    assert isinstance(report, gffutils.CoverageReport)
    assert report.scope == "transcript"
    assert report.unresolved == []

    # The shared exon contributes once to *each* transcript.
    assert report["tx1"].bp == 75
    assert report["tx2"].bp == 75
    assert report.total_bp == 150
    assert sorted(f.id for f in report["tx1"].features) == [
        "exon_shared",
        "exon_tx1",
    ]
    # Each result carries the underlying lineages (edges stay accessible).
    shared_lineage = next(
        item for item in report["tx1"].lineages if item.feature_id == "exon_shared"
    )
    assert shared_lineage.parent_ids == ["tx1", "tx2"]


def test_gene_scope_unions_feature_identity(shared_db):
    report = shared_db.coverage(scope="gene")
    assert report.unresolved == []

    # Three distinct exon identities; the shared one is counted once, not
    # doubled (50 + 25 + 25 = 100, versus 150 when edges are expanded).
    assert report["gene1"].bp == 100
    assert len(report["gene1"].features) == 3
    assert report.total_bp == 100


def test_both_scopes_reported_together(shared_db):
    reports = shared_db.coverage(scope="both")
    assert set(reports) == {"transcript", "gene"}
    assert reports["transcript"].total_bp == 150
    assert reports["gene"].total_bp == 100


def test_gene_scope_merge_dedupes_overlapping_coordinates():
    # Distinct exon identities with identical coordinates (typical GTF
    # representation of a shared genomic exon).
    gff = """##gff-version 3
chr1	test	gene	1	100	.	+	.	ID=g1
chr1	test	mRNA	1	100	.	+	.	ID=t1;Parent=g1
chr1	test	mRNA	1	100	.	+	.	ID=t2;Parent=g1
chr1	test	exon	1	50	.	+	.	ID=e1;Parent=t1
chr1	test	exon	1	50	.	+	.	ID=e2;Parent=t2
"""
    db = gffutils.create_db(
        gff, ":memory:", from_string=True, merge_strategy="create_unique"
    )
    identity_report = db.coverage(scope="gene")
    coordinate_report = db.coverage(scope="gene", merge=True)

    # Identity union keeps two source rows; coordinate union merges them.
    assert identity_report["g1"].bp == 100
    assert len(identity_report["g1"].features) == 2
    assert coordinate_report["g1"].bp == 50


def test_children_bp_supports_both_scopes_and_legacy_default(shared_db):
    assert shared_db.children_bp("tx1", scope="transcript") == 75
    assert shared_db.children_bp("gene1", scope="gene") == 100

    # Legacy behavior is unchanged when scope is not supplied: distinct
    # child rows under the gene are summed directly.
    assert shared_db.children_bp("gene1") == 100
    with pytest.raises(ValueError):
        shared_db.children_bp("gene1", scope="chromosome")


UNRESOLVED_GFF = """##gff-version 3
chr1	test	gene	1	100	.	+	.	ID=gA
chr1	test	gene	200	300	.	+	.	ID=gB
chr1	test	mRNA	1	100	.	+	.	ID=tA;Parent=gA
chr1	test	mRNA	200	300	.	+	.	ID=tB;Parent=gB
chr1	test	exon	1	20	.	+	.	ID=e_dangling;Parent=ghost_tx
chr1	test	exon	21	40	.	+	.	ID=e_orphan
chr1	test	exon	41	60	.	+	.	ID=e_conflict;Parent=tA,tB
chr1	test	exon	61	80	.	+	.	ID=e_ok;Parent=tA
"""


@pytest.fixture
def unresolved_db():
    return gffutils.create_db(
        UNRESOLVED_GFF,
        ":memory:",
        from_string=True,
        merge_strategy="create_unique",
    )


def test_lineage_detects_missing_and_conflicting_parents(unresolved_db):
    dangling = unresolved_db.lineage("e_dangling")
    assert dangling.status == gffutils.MISSING_PARENT
    assert dangling.missing_parent_ids == ["ghost_tx"]
    # The dangling edge itself is retained, not dropped.
    assert [edge.parent for edge in dangling.parent_edges] == ["ghost_tx"]

    orphan = unresolved_db.lineage("e_orphan")
    assert orphan.parent_edges == []

    conflict = unresolved_db.lineage("e_conflict")
    assert conflict.status == gffutils.CONFLICTING_PARENTS
    assert set(conflict.conflicting_gene_ids) == {"gA", "gB"}
    # Both edges are preserved; nothing picks tA just because it is first.
    assert conflict.parent_ids == ["tA", "tB"]

    assert unresolved_db.lineage("e_ok").is_resolved


def test_unresolved_lineages_are_pending_not_attached(unresolved_db):
    report = unresolved_db.coverage(scope="gene")
    pending = {item.feature_id: item for item in report.unresolved}

    assert set(pending) == {"e_dangling", "e_orphan", "e_conflict"}
    # Only the clean exon lands in a gene result; the conflicting exon is
    # never assigned to the first gene (gA) to force a tree.
    assert list(report.keys()) == ["gA"]
    assert report["gA"].bp == 20
    assert [f.id for f in report["gA"].features] == ["e_ok"]
    assert "gB" not in report

    transcript_report = unresolved_db.coverage(scope="transcript")
    assert set(item.feature_id for item in transcript_report.unresolved) == set(
        pending
    )
    assert transcript_report["tA"].bp == 20
    assert "tB" not in transcript_report


def test_unresolved_policies(unresolved_db):
    with pytest.warns(UserWarning, match="Unresolved parentage"):
        report = unresolved_db.coverage(scope="gene", on_unresolved="warn")
    assert len(report.unresolved) == 3

    with pytest.raises(ParentResolutionError) as exc_info:
        unresolved_db.coverage(scope="gene", on_unresolved="error")
    assert sorted(item.feature_id for item in exc_info.value.lineages) == [
        "e_conflict",
        "e_dangling",
        "e_orphan",
    ]

    with pytest.raises(ValueError):
        unresolved_db.coverage(scope="gene", on_unresolved="pick_first")


def test_gtf_shared_exon_identity_vs_coordinate_coverage():
    gtf = """chr1	test	exon	1	50	.	+	.	gene_id "g1"; transcript_id "t1";
chr1	test	exon	51	100	.	+	.	gene_id "g1"; transcript_id "t1";
chr1	test	exon	1	50	.	+	.	gene_id "g1"; transcript_id "t2";
chr1	test	exon	51	100	.	+	.	gene_id "g1"; transcript_id "t2";
"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        db = gffutils.create_db(gtf, ":memory:", from_string=True)

    transcript_report = db.coverage(
        featuretype="exon",
        scope="transcript",
        transcript_featuretype="transcript",
        gene_featuretype="gene",
    )
    assert transcript_report["t1"].bp == 100
    assert transcript_report["t2"].bp == 100
    assert transcript_report.unresolved == []

    identity_report = db.coverage(
        featuretype="exon",
        scope="gene",
        transcript_featuretype="transcript",
        gene_featuretype="gene",
    )
    # Four distinct GTF exon rows: identity union sums all of them.
    assert identity_report["g1"].bp == 200
    assert len(identity_report["g1"].features) == 4

    coordinate_report = db.coverage(
        featuretype="exon",
        scope="gene",
        merge=True,
        transcript_featuretype="transcript",
        gene_featuretype="gene",
    )
    # Coordinate union collapses the duplicated genomic region once.
    assert coordinate_report["g1"].bp == 100


def test_status_constants_are_distinct_strings():
    assert lin.RESOLVED != lin.MISSING_PARENT
    assert lin.MISSING_PARENT != lin.CONFLICTING_PARENTS


def test_exon_parented_directly_by_gene_is_anchored_to_that_gene():
    # No transcript intermediate: the gene is the direct parent and must
    # itself act as the gene anchor rather than being treated as missing.
    gff = """##gff-version 3
chr1	test	gene	1	100	.	+	.	ID=g1
chr1	test	exon	1	50	.	+	.	ID=e1;Parent=g1
chr1	test	exon	51	100	.	+	.	ID=e2;Parent=g1
"""
    db = gffutils.create_db(
        gff, ":memory:", from_string=True, merge_strategy="create_unique"
    )
    assert db.lineage("e1").is_resolved
    report = db.coverage(scope="gene")
    assert report.unresolved == []
    assert report["g1"].bp == 100
    assert sorted(f.id for f in report["g1"].features) == ["e1", "e2"]


def test_exon_shared_directly_by_two_genes_is_conflicting():
    # An exon whose direct parents are two different genes cannot be assigned
    # to either one without inventing information; it stays pending.
    gff = """##gff-version 3
chr1	test	gene	1	100	.	+	.	ID=gA
chr1	test	gene	200	300	.	+	.	ID=gB
chr1	test	exon	10	20	.	+	.	ID=e_shared;Parent=gA,gB
chr1	test	exon	30	40	.	+	.	ID=e_a;Parent=gA
"""
    db = gffutils.create_db(
        gff, ":memory:", from_string=True, merge_strategy="create_unique"
    )
    lineage = db.lineage("e_shared")
    assert lineage.status == gffutils.CONFLICTING_PARENTS
    assert set(lineage.conflicting_gene_ids) == {"gA", "gB"}

    report = db.coverage(scope="gene")
    assert [u.feature_id for u in report.unresolved] == ["e_shared"]
    assert report["gA"].bp == 11
    assert [f.id for f in report["gA"].features] == ["e_a"]
    assert "gB" not in report
