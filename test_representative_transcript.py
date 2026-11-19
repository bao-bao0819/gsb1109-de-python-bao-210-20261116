"""Tests for representative_transcript.py against the synthetic fixture
and the real FlyBase/FAKE example shipped with gffutils."""

import os

import pytest

import representative_transcript as rt

FIXTURE = os.path.join(os.path.dirname(__file__), "rep_tx_fixture.gff3")
REAL = os.path.join(
    os.path.dirname(__file__), "gffutils", "test", "data", "FBgn0031208.gff"
)
PRIORITY = ["ensembl", "refseq", "augustus"]
RANK_ORDER = ["source", "completeness", "length"]


@pytest.fixture(scope="module")
def fixture_result():
    db = rt.load_db(FIXTURE)
    return rt.analyze(db, PRIORITY, RANK_ORDER)


def reps_by_gene(gene_rows):
    out = {}
    for row in gene_rows:
        out.setdefault(row["gene_id"], []).append(row["representative_id"])
    return out


def excluded_by_gene(excluded_rows):
    out = {}
    for row in excluded_rows:
        out.setdefault(row["gene_id"], {})[row["transcript_id"]] = row["reasons"]
    return out


def test_longer_complete_transcript_wins(fixture_result):
    gene_rows, _ = fixture_result
    assert reps_by_gene(gene_rows)["gA"] == ["gA-t1"]
    row = [r for r in gene_rows if r["gene_id"] == "gA"][0]
    assert row["tx_len_bp"] == 600
    assert row["completeness_class"] == "complete_coding"
    assert row["source_rank"] == 0


def test_ties_are_kept(fixture_result):
    gene_rows, _ = fixture_result
    assert sorted(reps_by_gene(gene_rows)["gB"]) == ["gB-t1", "gB-t2"]
    for row in gene_rows:
        if row["gene_id"] == "gB":
            assert row["tied_representatives"] == 2
            assert row["tx_len_bp"] == 500


def test_shared_exons_counted_once(fixture_result):
    gene_rows, _ = fixture_result
    row = [r for r in gene_rows if r["gene_id"] == "gB"][0]
    # 500 + 500 bp of transcript exons, but only 650 bp actual coverage.
    assert row["all_tx_exon_sum_bp"] == 1000
    assert row["gene_exon_union_bp"] == 650


def test_source_rank_beats_abnormal_length(fixture_result):
    gene_rows, _ = fixture_result
    # augustus tx is 900 bp vs 600 bp ensembl record; source decides.
    assert reps_by_gene(gene_rows)["gC"] == ["gC-t1"]


def test_incomplete_boundary_excluded(fixture_result):
    gene_rows, excluded_rows = fixture_result
    assert reps_by_gene(gene_rows)["gD"] == ["gD-t1"]
    exc = excluded_by_gene(excluded_rows)
    assert exc["gD"]["gD-t2"] == "incomplete_boundary_3p"


def test_missing_phase_excluded(fixture_result):
    gene_rows, excluded_rows = fixture_result
    assert reps_by_gene(gene_rows)["gE"] == ["gE-t1"]
    exc = excluded_by_gene(excluded_rows)
    assert exc["gE"]["gE-t2"] == "missing_phase"


def test_gene_with_only_incomplete_candidates_has_no_rep(fixture_result):
    gene_rows, excluded_rows = fixture_result
    assert reps_by_gene(gene_rows)["gF"] == ["NONE"]
    exc = excluded_by_gene(excluded_rows)
    assert exc["gF"]["gF-t1"] == "missing_phase"


def test_noncoding_gene_still_gets_representative(fixture_result):
    gene_rows, _ = fixture_result
    assert reps_by_gene(gene_rows)["gG"] == ["gG-t1"]
    row = [r for r in gene_rows if r["gene_id"] == "gG"][0]
    assert row["completeness_class"] == "non_coding"


def test_real_data_missing_phase_listed_separately():
    db = rt.load_db(REAL)
    gene_rows, excluded_rows = rt.analyze(db, ["FlyBase"], RANK_ORDER)
    reps = reps_by_gene(gene_rows)
    # Fk_gene_1's transcript has a CDS with phase '.' (its Parent attribute
    # is malformed: "ID=...; Parent=...") and must not win.
    assert reps["Fk_gene_1"] == ["NONE"]
    exc = excluded_by_gene(excluded_rows)
    assert "missing_phase" in exc["Fk_gene_1"]["transcript_Fk_gene_1"]
    # Fk_gene_2's transcript has exons but no CDS: non-coding, eligible.
    assert reps["Fk_gene_2"] == ["transcript_Fk_gene_2"]
    row = [r for r in gene_rows if r["gene_id"] == "Fk_gene_2"][0]
    assert row["completeness_class"] == "non_coding"
    # The formal FlyBase gene still gets exactly one representative:
    # FBtr0300689 (1880 bp merged exons) over FBtr0300690 (1802 bp).
    assert reps["FBgn0031208"] == ["FBtr0300689"]
    row = [r for r in gene_rows if r["gene_id"] == "FBgn0031208"][0]
    assert row["source"] == "FlyBase"
    assert row["completeness_class"] == "complete_coding"
    assert row["tx_len_bp"] == 1880
