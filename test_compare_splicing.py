"""Synthetic tests for compare_splicing.py."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import compare_splicing as cs


def gff_gene(gene_id, txs):
    """Build GFF3 text. txs: list of (tx_id, strand, [(s, e), ...], extra_attrs)."""
    lines = ["##gff-version 3"]
    for tx_id, strand, exons, extra in txs:
        lo = min(s for s, _ in exons)
        hi = max(e for _, e in exons)
        lines.append("\t".join(["chr1", "test", "gene", str(lo), str(hi),
                                ".", strand, ".", "ID=g_%s" % tx_id]))
        lines.append("\t".join(["chr1", "test", "mRNA", str(lo), str(hi),
                                ".", strand, ".",
                                "ID=%s;Parent=g_%s%s" % (tx_id, tx_id, extra)]))
        for i, (s, e) in enumerate(exons):
            lines.append("\t".join(["chr1", "test", "exon", str(s), str(e),
                                    ".", strand, ".",
                                    "ID=%s_e%d;Parent=%s" % (tx_id, i, tx_id)]))
    return "\n".join(lines) + "\n"


def run_pair(tmp_path, txs_a, txs_b, incomplete_a=None, incomplete_b=None):
    fa = tmp_path / "a.gff3"
    fb = tmp_path / "b.gff3"
    fa.write_text(gff_gene("g", txs_a))
    fb.write_text(gff_gene("g", txs_b))
    a = cs.build_transcripts(cs.load_db(str(fa)), "A", incomplete_a or {})
    b = cs.build_transcripts(cs.load_db(str(fb)), "B", incomplete_b or {})
    return a, b, cs.compare_annotations(a, b)


def verdicts(reports):
    out = {}
    for rep in reports:
        for x, y, _ in rep["different"]:
            out[(x, y)] = cs.DIFFERENT
        for x, y, _ in rep["insufficient"]:
            out[(x, y)] = cs.INSUFFICIENT
    return out


def test_identical_complete_transcripts(tmp_path):
    exons = [(1000, 1500), (2000, 2500), (8000, 9000)]
    _, _, reports = run_pair(tmp_path, [("txA", "+", exons, "")],
                             [("txB", "+", exons, "")])
    assert reports[0]["identical_groups"] == [["A:txA", "B:txB"]]
    assert not reports[0]["different"]
    assert not reports[0]["insufficient"]


def test_shared_exon_set_but_conflicting_path(tmp_path):
    # B skips the middle exon: exon *set* of B is a subset of A's, but B's
    # junction path crosses A's internal exon -> clearly different.
    a = [("txA", "+", [(1000, 1500), (2000, 2500), (8000, 9000)], "")]
    b = [("txB", "+", [(1000, 1500), (8000, 9000)], "")]
    _, _, reports = run_pair(tmp_path, a, b)
    v = verdicts(reports)
    assert v[("A:txA", "B:txB")] == cs.DIFFERENT
    assert "intron" in reports[0]["different"][0][2]


def test_partial_transcript_only_constrains_observed_part(tmp_path):
    # A is 3'-truncated (end_range): it shares the observed exons with the
    # longer B but says nothing about the rest -> insufficient evidence.
    a = [("txA", "+", [(1000, 1500), (2000, 2500)], ";end_range=2500,.")]
    b = [("txB", "+", [(1000, 1500), (2000, 2500), (8000, 9000)], "")]
    models_a, _, reports = run_pair(tmp_path, a, b)
    v = verdicts(reports)
    assert v[("A:txA", "B:txB")] == cs.INSUFFICIENT
    # Unknown endpoint must NOT be imputed from the longest transcript.
    assert models_a["txA"].hi == 2500
    assert models_a["txA"].hi_known is False
    assert not reports[0]["identical_groups"]


def test_complete_short_vs_complete_long_is_different(tmp_path):
    # Same as above but A's 3' end is known: A provably ends at 2500 while
    # B extends to 9000 -> clearly different isoforms.
    a = [("txA", "+", [(1000, 1500), (2000, 2500)], "")]
    b = [("txB", "+", [(1000, 1500), (2000, 2500), (8000, 9000)], "")]
    _, _, reports = run_pair(tmp_path, a, b)
    v = verdicts(reports)
    assert v[("A:txA", "B:txB")] == cs.DIFFERENT


def test_same_chain_unknown_endpoint_not_identical(tmp_path):
    # Identical observed exon chains, but B's 3' end is unknown: the unknown
    # end must not be filled in, so the pair is not necessarily identical.
    exons = [(1000, 1500), (2000, 2500)]
    a = [("txA", "+", exons, "")]
    b = [("txB", "+", exons, ";end_range=2500,.")]
    _, _, reports = run_pair(tmp_path, a, b)
    v = verdicts(reports)
    assert v[("A:txA", "B:txB")] == cs.INSUFFICIENT
    assert not reports[0]["identical_groups"]


def test_minus_strand_directionality(tmp_path):
    exons = [(1000, 1500), (2000, 2500)]
    models_a, models_b, reports = run_pair(
        tmp_path, [("txA", "-", exons, "")], [("txB", "-", exons, "")])
    assert reports[0]["identical_groups"] == [["A:txA", "B:txB"]]
    # Junction stored in transcription direction: donor > acceptor on '-'.
    (j,) = models_a["txA"].junctions
    assert (j.donor, j.acceptor) == (2000, 1500)
    plus = cs.TranscriptModel("A", "t", "g", "chr1", "+", tuple(exons),
                              True, True)
    (jp,) = plus.junctions
    assert (jp.donor, jp.acceptor) == (1500, 2000)
    assert models_a["txA"].junctions != plus.junctions


def test_incomplete_file_marks_unknown_ends(tmp_path):
    # A has no range attributes but is listed as 5'-incomplete; B extends
    # further upstream. Unknown 5' end -> insufficient, not different.
    a = [("txA", "+", [(1000, 1500), (2000, 2500)], "")]
    b = [("txB", "+", [(500, 700), (1000, 1500), (2000, 2500)], "")]
    _, _, reports = run_pair(tmp_path, a, b, incomplete_a={"txA": {"5"}})
    v = verdicts(reports)
    assert v[("A:txA", "B:txB")] == cs.INSUFFICIENT


def test_gene_only_in_one_annotation(tmp_path):
    exons = [(1000, 1500)]
    _, _, reports = run_pair(tmp_path, [("txA", "+", exons, "")],
                             [("txB", "+", [(100, 200)], "")])
    # non-overlapping, different gene keys -> both unmatched
    notes = sorted(r.get("note", "") for r in reports)
    assert notes == ["gene only in annotation A", "gene only in annotation B"]


def test_cli_end_to_end(tmp_path, capsys):
    exons = [(1000, 1500), (2000, 2500)]
    fa = tmp_path / "a.gff3"
    fb = tmp_path / "b.gff3"
    fa.write_text(gff_gene("g", [("txA", "+", exons, "")]))
    fb.write_text(gff_gene("g", [("txB", "+", exons, "")]))
    tsv = tmp_path / "out.tsv"
    rc = cs.main([str(fa), str(fb), "-o", str(tsv)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "IDENTICAL group: A:txA, B:txB" in out
    assert "IDENTICAL" in tsv.read_text()
