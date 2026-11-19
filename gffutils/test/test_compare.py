"""
Tests for gffutils.compare: mapping features across annotation versions
by coordinates, parentage and explicit identity -- never by exon_number.
"""

import pytest

from gffutils import compare


OLD_GFF3 = """##gff-version 3
chr1\ttest\tgene\t100\t800\t.\t+\t.\tID=gA
chr1\ttest\tmRNA\t100\t800\t.\t+\t.\tID=txA.1;Parent=gA
chr1\ttest\texon\t100\t200\t.\t+\t.\tID=eA1;Parent=txA.1;exon_number=1
chr1\ttest\texon\t300\t400\t.\t+\t.\tID=eA2;Parent=txA.1;exon_number=2
chr1\ttest\texon\t500\t600\t.\t+\t.\tID=eA3;Parent=txA.1;exon_number=3
chr1\ttest\texon\t700\t800\t.\t+\t.\tID=eA4;Parent=txA.1;exon_number=4
chr1\ttest\tgene\t1000\t3100\t.\t+\t.\tID=gB
chr1\ttest\tmRNA\t1000\t3100\t.\t+\t.\tID=txB.1;Parent=gB
chr1\ttest\texon\t1000\t1200\t.\t+\t.\tID=eB1;Parent=txB.1;exon_number=1
chr1\ttest\texon\t2000\t2100\t.\t+\t.\tID=eB2;Parent=txB.1;exon_number=2
chr1\ttest\texon\t2200\t2300\t.\t+\t.\tID=eB3;Parent=txB.1;exon_number=3
chr1\ttest\texon\t3000\t3100\t.\t+\t.\tID=eB4;Parent=txB.1;exon_number=4
chr1\ttest\tgene\t5000\t5100\t.\t+\t.\tID=gC
chr1\ttest\tmRNA\t5000\t5100\t.\t+\t.\tID=txC.1;Parent=gC
chr1\ttest\texon\t5000\t5100\t.\t+\t.\tID=eC1;Parent=txC.1;exon_number=1
chr1\ttest\tgene\t6000\t6100\t.\t+\t.\tID=gD
chr1\ttest\tmRNA\t6000\t6100\t.\t+\t.\tID=txD.1;Parent=gD
chr1\ttest\texon\t6000\t6100\t.\t+\t.\tID=eD1a;Parent=txD.1;exon_number=1
chr1\ttest\texon\t6000\t6100\t.\t+\t.\tID=eD1b;Parent=txD.1;exon_number=1
chr1\ttest\tgene\t9000\t9100\t.\t+\t.\tID=gGone
chr1\ttest\tmRNA\t9000\t9100\t.\t+\t.\tID=txGone.1;Parent=gGone
chr1\ttest\texon\t9000\t9100\t.\t+\t.\tID=eG1;Parent=txGone.1;exon_number=1
chr1\ttest\tgene\t11000\t11100\t.\t+\t.\tID=gE
chr1\ttest\tmRNA\t11000\t11100\t.\t+\t.\tID=txE.1;Parent=gE
chr1\ttest\texon\t11000\t11100\t.\t+\t.\tID=eE1;Parent=txE.1;exon_number=1
chr1\ttest\tgene\t12000\t12100\t.\t+\t.\tID=gF
chr1\ttest\tmRNA\t12000\t12100\t.\t+\t.\tID=txF.1;Parent=gF
chr1\ttest\texon\t12000\t12100\t.\t+\t.\tID=eF1;Parent=txF.1;exon_number=1
"""

NEW_GFF3 = """##gff-version 3
chr1\ttest\tgene\t100\t800\t.\t+\t.\tID=gA
chr1\ttest\tmRNA\t100\t800\t.\t+\t.\tID=txA.2;Parent=gA
chr1\ttest\texon\t100\t200\t.\t+\t.\tID=nA1;Parent=txA.2;exon_number=1
chr1\ttest\texon\t300\t400\t.\t+\t.\tID=nA2;Parent=txA.2;exon_number=2
chr1\ttest\texon\t450\t480\t.\t+\t.\tID=nA3;Parent=txA.2;exon_number=3
chr1\ttest\texon\t500\t600\t.\t+\t.\tID=nA4;Parent=txA.2;exon_number=4
chr1\ttest\texon\t700\t800\t.\t+\t.\tID=nA5;Parent=txA.2;exon_number=5
chr1\ttest\tgene\t1000\t3150\t.\t+\t.\tID=gB
chr1\ttest\tmRNA\t1000\t3150\t.\t+\t.\tID=txB-renamed;Parent=gB;previous_ID=txB.1
chr1\ttest\texon\t1000\t1090\t.\t+\t.\tID=nB1a;Parent=txB-renamed;exon_number=1
chr1\ttest\texon\t1110\t1200\t.\t+\t.\tID=nB1b;Parent=txB-renamed;exon_number=2
chr1\ttest\texon\t2000\t2300\t.\t+\t.\tID=nB2;Parent=txB-renamed;exon_number=3
chr1\ttest\texon\t3000\t3150\t.\t+\t.\tID=nB4;Parent=txB-renamed;exon_number=4
chr1\ttest\tgene\t5000\t5100\t.\t+\t.\tID=gC
chr1\ttest\tmRNA\t5000\t5100\t.\t+\t.\tID=txC.2;Parent=gC
chr1\ttest\texon\t5000\t5100\t.\t+\t.\tID=nC1a;Parent=txC.2;exon_number=1
chr1\ttest\texon\t5000\t5100\t.\t+\t.\tID=nC1b;Parent=txC.2;exon_number=1
chr1\ttest\tgene\t6000\t6100\t.\t+\t.\tID=gD
chr1\ttest\tmRNA\t6000\t6100\t.\t+\t.\tID=txD.1;Parent=gD
chr1\ttest\texon\t6000\t6100\t.\t+\t.\tID=nD1a;Parent=txD.1;exon_number=1
chr1\ttest\texon\t6000\t6100\t.\t+\t.\tID=nD1b;Parent=txD.1;exon_number=1
chr1\ttest\tgene\t9500\t9800\t.\t+\t.\tID=gNew
chr1\ttest\tmRNA\t9500\t9800\t.\t+\t.\tID=txNew.1;Parent=gNew
chr1\ttest\texon\t9500\t9600\t.\t+\t.\tID=nN1;Parent=txNew.1;exon_number=1
chr1\ttest\texon\t9700\t9800\t.\t+\t.\tID=nN2;Parent=txNew.1;exon_number=2
chr1\ttest\tgene\t11000\t12100\t.\t+\t.\tID=gF
chr1\ttest\tmRNA\t11000\t12100\t.\t+\t.\tID=txF.1;Parent=gF
chr1\ttest\texon\t11000\t11100\t.\t+\t.\tID=nF0;Parent=txF.1;exon_number=1
chr1\ttest\texon\t12000\t12100\t.\t+\t.\tID=nF1;Parent=txF.1;exon_number=2
"""


@pytest.fixture
def result(tmp_path):
    old = tmp_path / "old.gff3"
    new = tmp_path / "new.gff3"
    old.write_text(OLD_GFF3)
    new.write_text(NEW_GFF3)
    return compare.compare_annotations(str(old), str(new))


def relations_for(result, old_feature_id):
    return [
        r for r in result.relations if r.old and r.old.feature_id == old_feature_id
    ]


def novel_relations(result):
    return [r for r in result.relations if r.relation == compare.NOVEL]


def test_transcript_matching_methods(result):
    pairs = dict((old, (new, method)) for old, new, method in result.transcript_pairs)
    # exact ID kept across versions
    assert pairs["txD.1"] == ("txD.1", "id")
    assert pairs["txF.1"] == ("txF.1", "id")
    # version bump: txA.1 -> txA.2, txC.1 -> txC.2
    assert pairs["txA.1"] == ("txA.2", "id_base")
    assert pairs["txC.1"] == ("txC.2", "id_base")
    # explicit back-reference on the renamed transcript
    assert pairs["txB.1"] == ("txB-renamed", "previous_ID")
    # transcripts only in one version
    assert "txGone.1" in result.unmatched_old
    assert "txE.1" in result.unmatched_old
    assert "txNew.1" in result.unmatched_new


def test_inserted_exon_does_not_renumber_old_exons(result):
    # txA gained exon 450-480, so every downstream exon_number shifted.
    # Coordinate identity must win over the ordinal.
    (rel,) = relations_for(result, "eA3")
    assert rel.relation == compare.CONTINUATION
    assert rel.old.number == "3"
    (cand,) = rel.candidates
    assert (cand.start, cand.end) == (500, 600)
    # the true counterpart carries exon_number 4 in the new file
    assert cand.number == "4"
    # the inserted exon is reported as novel, not as "the new exon 3 of eA3"
    novel = novel_relations(result)
    assert any(
        c.feature_id == "nA3" and (c.start, c.end) == (450, 480)
        for r in novel
        for c in r.candidates
    )
    # eA3 must not be linked to the shifted ordinal slot (exon_number 3)
    assert all(c.number != "3" for r in relations_for(result, "eA3") for c in r.candidates)


def test_split_merge_and_boundary_shift(result):
    (split,) = relations_for(result, "eB1")
    assert split.relation == compare.SPLIT
    assert sorted((c.start, c.end) for c in split.candidates) == [
        (1000, 1090),
        (1110, 1200),
    ]

    (m2,) = relations_for(result, "eB2")
    (m3,) = relations_for(result, "eB3")
    assert m2.relation == compare.MERGED
    assert m3.relation == compare.MERGED
    # both old exons point at the same new merged exon
    assert [c.feature_id for c in m2.candidates] == ["nB2"]
    assert [c.feature_id for c in m3.candidates] == ["nB2"]

    (shift,) = relations_for(result, "eB4")
    assert shift.relation == compare.BOUNDARY_SHIFT
    assert [(c.start, c.end) for c in shift.candidates] == [(3000, 3150)]


def test_duplicate_segments_list_candidates(result):
    # one old exon, two identical new copies: no unique assignment
    (rel,) = relations_for(result, "eC1")
    assert rel.relation == compare.AMBIGUOUS
    assert sorted(c.feature_id for c in rel.candidates) == ["nC1a", "nC1b"]
    assert "1 time(s) in old and 2 in new" in rel.note

    # two identical old exons, two identical new copies
    for fid in ("eD1a", "eD1b"):
        (rel,) = relations_for(result, fid)
        assert rel.relation == compare.AMBIGUOUS
        assert sorted(c.feature_id for c in rel.candidates) == ["nD1a", "nD1b"]


def test_retired_transcript_keeps_original_position(result):
    # txGone.1 is gone entirely; its exon is retired with no counterpart
    (rel,) = relations_for(result, "eG1")
    assert rel.relation == compare.RETIRED
    assert rel.candidates == ()
    assert (rel.old.seqid, rel.old.start, rel.old.end) == ("chr1", 9000, 9100)

    # txE.1 is gone too, but its exact segment survives in txF.1: the old
    # hit stays traceable to its original position and lists the survivor
    # as a candidate instead of being renumbered into txF.1
    (rel,) = relations_for(result, "eE1")
    assert rel.relation == compare.RETIRED
    assert (rel.old.start, rel.old.end) == (11000, 11100)
    assert [c.feature_id for c in rel.candidates] == ["nF0"]
    assert "survives" in rel.note


def test_novel_transcript(result):
    novel_ids = {c.feature_id for r in novel_relations(result) for c in r.candidates}
    assert {"nN1", "nN2"} <= novel_ids


def test_tsv_traceability(result, tmp_path):
    out = tmp_path / "mapping.tsv"
    result.to_tsv(str(out))
    lines = out.read_text().splitlines()
    assert lines[0].split("\t") == compare.TSV_HEADER
    rows = [dict(zip(compare.TSV_HEADER, line.split("\t"))) for line in lines[1:]]

    # every old exon shows up with its own ID and original coordinates
    old_ids = {row["old_feature_id"] for row in rows if row["old_feature_id"]}
    assert {
        "eA1", "eA2", "eA3", "eA4",
        "eB1", "eB2", "eB3", "eB4",
        "eC1", "eD1a", "eD1b",
        "eE1", "eF1", "eG1",
    } <= old_ids

    # the renumbered exon keeps both ordinals visible side by side
    (row,) = [r for r in rows if r["old_feature_id"] == "eA3"]
    assert row["old_start"] == "500" and row["old_end"] == "600"
    assert row["old_exon_number"] == "3"
    assert row["new_exon_number"] == "4"

    # ambiguous duplicates produce one row per candidate, all anchored at
    # the old feature
    dup_rows = [r for r in rows if r["old_feature_id"] == "eC1"]
    assert len(dup_rows) == 2
    assert {r["new_feature_id"] for r in dup_rows} == {"nC1a", "nC1b"}
    assert all(r["old_start"] == "5000" for r in dup_rows)


def test_summary_counts(result):
    summary = result.summary()
    assert summary["transcript_pairs"] == 5
    assert summary["unmatched_old_transcripts"] == 2
    assert summary["unmatched_new_transcripts"] == 1
    counts = summary["relations"]
    assert counts[compare.CONTINUATION] == 5  # eA1, eA2, eA3, eA4, eF1
    assert counts[compare.SPLIT] == 1
    assert counts[compare.MERGED] == 2
    assert counts[compare.BOUNDARY_SHIFT] == 1
    assert counts[compare.AMBIGUOUS] == 3  # eC1, eD1a, eD1b
    assert counts[compare.RETIRED] == 2  # eG1, eE1
    assert counts[compare.NOVEL] == 4  # nA3, nF0, nN1, nN2
