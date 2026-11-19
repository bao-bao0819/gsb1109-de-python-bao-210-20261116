"""
Tests for evidence boundaries between source-provided and gffutils-derived
features (inferred genes/transcripts, conflicts and reconciliation).
"""
import sqlite3
import tempfile
from textwrap import dedent

import pytest

import gffutils
from gffutils import constants
from gffutils.exceptions import FeatureNotFoundError


@pytest.fixture(autouse=True)
def _reset_tempdir():
    # Other test modules point tempfile.tempdir at a hard-coded path that
    # may be unwritable in restricted environments; GTF inference relies on
    # NamedTemporaryFile, so restore the platform default for these tests.
    tempfile.tempdir = None
    yield
    tempfile.tempdir = None


GTF_NO_PARENTS = dedent(
    """\
    chr1	test	exon	100	200	.	+	.	gene_id "G1"; transcript_id "T1";
    chr1	test	exon	300	400	.	+	.	gene_id "G1"; transcript_id "T1";
    """
)


def _db(data=GTF_NO_PARENTS, **kwargs):
    kwargs.setdefault("from_string", True)
    return gffutils.create_db(data, ":memory:", **kwargs)


def test_inferred_records_are_derived():
    db = _db()
    gene = db["G1"]
    transcript = db["T1"]
    assert gene.evidence_scope == constants.EVIDENCE_DERIVED
    assert transcript.evidence_scope == constants.EVIDENCE_DERIVED
    assert gene.is_derived and transcript.is_derived
    assert gene.source == constants.derived_source

    evidence = db.evidence_for(gene)
    assert evidence["scope"] == constants.EVIDENCE_DERIVED
    assert evidence["basis"] == constants.BASIS_INFERRED_GENE
    assert evidence["status"] == "active"
    assert evidence["conflicts_with"] is None
    assert evidence["derived_from"] == ["exon_1", "exon_2"]

    tx_evidence = db.evidence_for(transcript)
    assert tx_evidence["basis"] == constants.BASIS_INFERRED_TRANSCRIPT
    assert tx_evidence["derived_from"] == ["exon_1", "exon_2"]


def test_input_records_are_source():
    db = _db()
    exon = db["exon_1"]
    assert exon.evidence_scope == constants.EVIDENCE_SOURCE
    assert not exon.is_derived
    evidence = db.evidence_for(exon)
    assert evidence["scope"] == constants.EVIDENCE_SOURCE
    assert evidence["basis"] == constants.BASIS_INPUT_RECORD
    assert evidence["derived_from"] == []


def test_every_feature_has_exactly_one_evidence_row():
    db = _db()
    rows = list(
        db.execute("SELECT id, count() FROM feature_evidence GROUP BY id")
    )
    feature_ids = {f.id for f in db.all_features()}
    assert {row[0] for row in rows} == feature_ids
    assert all(row[1] == 1 for row in rows)


def test_count_and_query_filters():
    db = _db()
    # 2 source exons plus the inferred transcript T1 and gene G1
    assert db.count_features_of_type() == 4
    assert db.count_features_of_type(evidence="source") == 2
    assert db.count_features_of_type(evidence="derived") == 2
    assert db.count_features_of_type(evidence="all") == 4
    assert db.count_features_of_type("gene") == 1
    assert db.count_features_of_type("gene", evidence="source") == 0
    assert db.count_features_of_type("gene", evidence="derived") == 1

    assert {f.id for f in db.all_features(evidence="source")} == {
        "exon_1",
        "exon_2",
    }
    assert {
        f.id for f in db.all_features(evidence="derived")
    } == {"G1", "T1"}
    assert {
        f.id
        for f in db.features_of_type("exon", evidence="source")
    } == {"exon_1", "exon_2"}

    assert {
        f.id for f in db.region(seqid="chr1", evidence="derived")
    } == {"G1", "T1"}

    assert {
        f.id
        for f in db.children("T1", evidence="source")
    } == {"exon_1", "exon_2"}
    assert {f.id for f in db.children("T1", evidence="derived")} == set()
    assert {f.id for f in db.parents("exon_1", evidence="derived")} == {
        "G1",
        "T1",
    }


def test_invalid_evidence_value_raises():
    db = _db()
    with pytest.raises(ValueError):
        list(db.all_features(evidence="bogus"))
    with pytest.raises(ValueError):
        db.count_features_of_type(evidence="bogus")


def test_late_source_record_retains_both_pieces_of_evidence():
    db = _db()
    formal = gffutils.feature.feature_from_line(
        "chr1	RefSeq	gene	90	410	.	+	.	gene_id " '"G1";',
        strict=False,
    )
    db.update([formal])

    # Canonical id now belongs to the formal source record.
    canonical = db["G1"]
    assert canonical.evidence_scope == constants.EVIDENCE_SOURCE
    assert canonical.source == "RefSeq"

    # The derived placeholder was kept under a distinct id.
    conflicts = list(db.evidence_conflicts())
    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert conflict["canonical"] == "G1"
    assert conflict["status"] == "reconciled"

    derived = db[conflict["derived_id"]]
    assert derived.evidence_scope == constants.EVIDENCE_DERIVED
    assert derived.start == 100 and derived.end == 400

    source_evidence = db.evidence_for("G1")
    assert source_evidence["scope"] == constants.EVIDENCE_SOURCE
    assert source_evidence["conflicts_with"] == derived.id

    derived_evidence = db.evidence_for(derived.id)
    assert derived_evidence["scope"] == constants.EVIDENCE_DERIVED
    assert derived_evidence["conflicts_with"] == "G1"
    assert derived_evidence["status"] == "reconciled"
    assert derived_evidence["basis"] == constants.BASIS_INFERRED_GENE

    # Stats can select either boundary independently.
    assert db.count_features_of_type("gene") == 2
    assert db.count_features_of_type("gene", evidence="source") == 1
    assert db.count_features_of_type("gene", evidence="derived") == 1


def test_derived_record_yielding_to_existing_source():
    mixed = dedent(
        """\
        chr1	RefSeq	gene	90	410	.	+	.	gene_id "G1";
        chr1	test	exon	100	200	.	+	.	gene_id "G1"; transcript_id "T1";
        chr1	test	exon	300	400	.	+	.	gene_id "G1"; transcript_id "T1";
        """
    )
    db = _db(mixed)
    genes = list(db.all_features(featuretype="gene"))
    assert {f.evidence_scope for f in genes} == {
        constants.EVIDENCE_SOURCE,
        constants.EVIDENCE_DERIVED,
    }
    canonical = db["G1"]
    assert canonical.evidence_scope == constants.EVIDENCE_SOURCE

    conflict = next(db.evidence_conflicts())
    assert conflict["status"] == "active"
    assert db[conflict["derived_id"]].evidence_scope == "derived"


def test_same_scope_duplicate_still_raises():
    duplicate = dedent(
        """\
        chr1	a	gene	1	10	.	+	.	gene_id "G1";
        chr1	a	gene	2	20	.	+	.	gene_id "G1";
        """
    )
    with pytest.raises(ValueError):
        _db(duplicate)


def test_manual_reconcile_identity():
    db = _db()
    db.update([
        gffutils.feature.feature_from_line(
            'chr1	RefSeq	gene	90	410	.	+	.	gene_id "G1";',
            strict=False,
        )
    ])
    derived_id = next(db.evidence_conflicts())["derived_id"]
    result = db.reconcile_identity("G1")
    assert result == {
        "canonical": "G1",
        "derived_id": derived_id,
        "status": "reconciled",
    }
    assert db.reconcile_identity("missing") is None
    assert next(db.evidence_conflicts(status="reconciled"))["derived_id"] == derived_id
    assert list(db.evidence_conflicts(status="active")) == []


def test_evidence_for_missing_feature():
    db = _db()
    with pytest.raises(FeatureNotFoundError):
        db.evidence_for("does-not-exist")


def test_delete_cleans_evidence():
    db = _db()
    db.delete("exon_1")
    with pytest.raises(FeatureNotFoundError):
        db.evidence_for("exon_1")
    assert list(db.execute(
        "SELECT 1 FROM feature_evidence WHERE id = 'exon_1'"
    )) == []


def test_introns_are_marked_derived():
    db = _db()
    introns = list(db.create_introns())
    assert introns
    for intron in introns:
        assert intron.evidence_scope == constants.EVIDENCE_DERIVED
        assert intron.source == constants.derived_source

    db.update(
        introns,
        disable_infer_genes=True,
        disable_infer_transcripts=True,
    )
    intron_id = introns[0].id
    evidence = db.evidence_for(intron_id)
    assert evidence["scope"] == constants.EVIDENCE_DERIVED
    assert evidence["basis"] == "derived_feature"


def test_gff_input_records_all_source():
    gff = dedent(
        """\
        chr1	RefSeq	gene	1	100	.	+	.	ID=G1
        chr1	RefSeq	exon	10	50	.	+	.	ID=E1;Parent=G1
        """
    )
    db = gffutils.create_db(gff, ":memory:", from_string=True)
    assert all(
        f.evidence_scope == constants.EVIDENCE_SOURCE for f in db.all_features()
    )
    assert db.count_features_of_type(evidence="derived") == 0


def test_legacy_database_migration():
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE features (
            id text, seqid text, source text, featuretype text, start int,
            end int, score text, strand text, frame text, attributes text,
            extra text, bin int, primary key (id));
        CREATE TABLE relations (parent text, child text, level int,
            primary key (parent, child, level));
        CREATE TABLE meta (dialect text, version text);
        CREATE TABLE directives (directive text);
        CREATE TABLE autoincrements (base text, n int, primary key (base));
        CREATE TABLE duplicates (idspecid text, newid text,
            primary key (newid));
        INSERT INTO meta VALUES ('{}', '0.13');
        INSERT INTO features VALUES (
            'exon_1', 'chr1', 'test', 'exon', 100, 200, '.', '+', '.',
            '{"gene_id":["G1"]}', '[]', 0);
        INSERT INTO features VALUES (
            'G1', 'chr1', 'gffutils_derived', 'gene', 100, 200, '.', '+',
            '.', '{"gene_id":["G1"]}', '[]', 0);
        """
    )
    db = gffutils.FeatureDB(conn)
    assert db["exon_1"].evidence_scope == constants.EVIDENCE_SOURCE
    assert db["G1"].evidence_scope == constants.EVIDENCE_DERIVED
    assert db.evidence_for("G1")["basis"] == "legacy_derived_feature"
    assert db.count_features_of_type(evidence="source") == 1
    assert db.count_features_of_type(evidence="derived") == 1
