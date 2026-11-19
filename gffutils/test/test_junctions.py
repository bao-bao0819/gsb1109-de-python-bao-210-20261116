"""
Tests for molecule-level splice-junction resolution.
"""

import textwrap

import pytest

import gffutils
from gffutils.junctions import (
    DONOR_ACCEPTOR,
    INTRON,
    Junction,
    group_junctions,
    read_junctions,
)

ANNOTATION = textwrap.dedent(
    """\
    chr1\t.\tgene\t100\t1000\t.\t+\t.\tID=gplus
    chr1\t.\tmRNA\t100\t1000\t.\t+\t.\tID=t1;Parent=gplus
    chr1\t.\texon\t100\t200\t.\t+\t.\tID=t1e1;Parent=t1
    chr1\t.\texon\t301\t400\t.\t+\t.\tID=t1e2;Parent=t1
    chr1\t.\texon\t501\t600\t.\t+\t.\tID=t1e3;Parent=t1
    chr1\t.\texon\t701\t800\t.\t+\t.\tID=t1e4;Parent=t1
    chr1\t.\texon\t901\t1000\t.\t+\t.\tID=t1e5;Parent=t1
    chr1\t.\tmRNA\t100\t1000\t.\t+\t.\tID=t2;Parent=gplus
    chr1\t.\texon\t100\t200\t.\t+\t.\tID=t2e1;Parent=t2
    chr1\t.\texon\t301\t400\t.\t+\t.\tID=t2e2;Parent=t2
    chr1\t.\texon\t501\t600\t.\t+\t.\tID=t2e3;Parent=t2
    chr1\t.\texon\t651\t700\t.\t+\t.\tID=t2e4;Parent=t2
    chr1\t.\texon\t951\t1000\t.\t+\t.\tID=t2e5;Parent=t2
    chr1\t.\tmRNA\t100\t1000\t.\t+\t.\tID=t3;Parent=gplus
    chr1\t.\texon\t100\t200\t.\t+\t.\tID=t3e1;Parent=t3
    chr1\t.\texon\t951\t1000\t.\t+\t.\tID=t3e2;Parent=t3
    chr2\t.\tgene\t100\t1000\t.\t-\t.\tID=gminus
    chr2\t.\tmRNA\t100\t1000\t.\t-\t.\tID=mt1;Parent=gminus
    chr2\t.\texon\t100\t200\t.\t-\t.\tID=mt1e1;Parent=mt1
    chr2\t.\texon\t301\t400\t.\t-\t.\tID=mt1e2;Parent=mt1
    chr2\t.\texon\t501\t600\t.\t-\t.\tID=mt1e3;Parent=mt1
    chr2\t.\texon\t701\t800\t.\t-\t.\tID=mt1e4;Parent=mt1
    """
)


@pytest.fixture
def db():
    return gffutils.create_db(ANNOTATION, ":memory:", from_string=True)


def _table(rows):
    header = "molecule\tseqid\tstart\tend\tstrand\torder\n"
    return header + "".join(
        "\t".join(str(i) for i in row) + "\n" for row in rows
    )


# Parsing ---------------------------------------------------------------------

def test_read_junctions_columns_and_types():
    text = (
        "molecule\tseqid\tstart\tend\tstrand\torder\tid\tcount\n"
        "m1\tchr1\t201\t300\t+\t2\tj1\t7\n"
    )
    (j,) = read_junctions(text)
    assert j.molecule == "m1"
    assert j.seqid == "chr1"
    assert j.intron == (201, 300)
    assert j.strand == "+"
    assert j.order == 2
    assert j.id == "j1"
    assert j.count == 7


def test_read_junctions_aliases_and_blank_lines():
    text = (
        "read_id\tchrom\tintron_start\tintron_end\tjunction_order\n"
        "r1\tchr1\t201\t300\t1\n"
        "\n"
        "r2\tchr1\t401\t500\t2\n"
    )
    junctions = read_junctions(text)
    assert [j.molecule for j in junctions] == ["r1", "r2"]
    assert junctions[0].order == 1


def test_read_junctions_missing_columns():
    with pytest.raises(ValueError) as excinfo:
        read_junctions("molecule\tfoo\nm1\t1\n")
    assert "start" in str(excinfo.value)


def test_donor_acceptor_coordinates_are_normalized():
    (j,) = read_junctions(
        "molecule\tstart\tend\nm1\t200\t301\n",
        coordinate=DONOR_ACCEPTOR,
    )
    assert j.intron == (201, 300)
    assert j.donor == 200
    assert j.acceptor == 301
    assert j.coordinate == DONOR_ACCEPTOR


def test_donor_acceptor_inferred_from_header():
    (j,) = read_junctions("molecule\tdonor\tacceptor\nm1\t200\t301\n")
    assert j.coordinate == DONOR_ACCEPTOR
    assert j.intron == (201, 300)


def test_intron_coordinate_explicit():
    (j,) = read_junctions(
        "molecule\tstart\tend\nm1\t201\t300\n", coordinate=INTRON
    )
    assert j.intron == (201, 300)


def test_invalid_coordinate_mode():
    with pytest.raises(ValueError):
        read_junctions("molecule\tstart\tend\nm1\t1\t2\n", coordinate="nope")


def test_grouping_preserves_input_order_without_explicit_order():
    junctions = [
        Junction("m1", 601, 700),
        Junction("m1", 201, 300),
    ]
    grouped = group_junctions(junctions)
    assert [j.intron for j in grouped["m1"]] == [(601, 700), (201, 300)]


def test_grouping_sorts_by_explicit_order():
    junctions = [
        Junction("m1", 601, 700, order=2),
        Junction("m1", 201, 300, order=1),
    ]
    grouped = group_junctions(junctions)
    assert [j.intron for j in grouped["m1"]] == [(201, 300), (601, 700)]


def test_require_order_rejects_unordered_multi_junction_molecule():
    junctions = [Junction("m1", 201, 300), Junction("m1", 401, 500)]
    with pytest.raises(ValueError):
        group_junctions(junctions, require_order=True)


# Joint explanation -----------------------------------------------------------

def test_group_is_explained_by_intersection_not_per_junction_union(db):
    # 201-300 is in t1/t2, 401-500 in t1/t2, 601-700 only in t1.  The whole
    # molecule must resolve to t1; t2 must not be reported as explaining it.
    junctions = read_junctions(
        _table(
            [
                ("m1", "chr1", 201, 300, "+", 1),
                ("m1", "chr1", 401, 500, "+", 2),
                ("m1", "chr1", 601, 700, "+", 3),
            ]
        )
    )
    report = db.explain_junctions(junctions)
    group = report["m1"]
    assert group.status == "unique"
    assert group.candidates == ("t1",)
    assert group.transcript == "t1"
    assert group.unique_support
    assert not group.shared_support
    assert not group.no_compatible_path
    # t2 supports the first two junctions individually...
    assert group.evidence[0].transcripts == frozenset({"t1", "t2"})
    assert group.evidence[1].transcripts == frozenset({"t1", "t2"})
    # ...but not the third, so it is not a whole-group explanation.
    assert group.evidence[2].transcripts == frozenset({"t1"})
    assert "t2" not in group.candidates


def test_shared_support_keeps_all_joint_candidates(db):
    junctions = read_junctions(
        _table(
            [
                ("m1", "chr1", 201, 300, "+", 1),
                ("m1", "chr1", 401, 500, "+", 2),
            ]
        )
    )
    group = db.explain_junctions(junctions)["m1"]
    assert group.status == "shared"
    assert group.candidates == ("t1", "t2")
    assert group.shared_support
    assert group.transcript is None


def test_no_compatible_path_when_support_is_only_disjoint(db):
    # 601-650 is an intron only in t2; 801-900 only in t1; 201-300 is shared.
    # Any two can be matched, but no transcript matches all three.
    junctions = read_junctions(
        _table(
            [
                ("m1", "chr1", 601, 650, "+", 1),
                ("m1", "chr1", 801, 900, "+", 2),
                ("m1", "chr1", 201, 300, "+", 3),
            ]
        )
    )
    group = db.explain_junctions(junctions)["m1"]
    assert group.status == "none"
    assert group.candidates == ()
    assert group.no_compatible_path
    assert not group.supported
    assert group.reason == "no single transcript matches all junctions"
    # Partial evidence is preserved, but never promoted to a group answer.
    assert group.partial
    assert {t for t, _ in group.partial} == {"t1", "t2"}
    assert all(len(junctions) == 3 for _, matched in group.partial)
    assert all(len(matched) == 2 for _, matched in group.partial)


def test_missing_middle_junction_is_not_confirmed_exon_skipping(db):
    # t1 contains 201-300, 401-500, 601-700.  Observing only the outer two
    # does not prove the middle exon was skipped: without allow_exon_skipping,
    # the molecule is still explained by t1 exactly.
    junctions = read_junctions(
        _table(
            [
                ("m1", "chr1", 201, 300, "+", 1),
                ("m1", "chr1", 601, 700, "+", 2),
            ]
        )
    )
    group = db.explain_junctions(junctions)["m1"]
    assert group.status == "unique"
    assert group.candidates == ("t1",)
    # Neither observation is a spanning match.
    assert all(not e.spanning_transcripts for e in group.evidence)


def test_explicit_exon_skipping_rule(db):
    # 201-300 then 601-700 skips t1's 401-500 intron (exon 501-600).  It is
    # accepted only when the caller explicitly enables the rule.
    junctions = read_junctions(
        _table(
            [
                ("m1", "chr1", 201, 300, "+", 1),
                ("m1", "chr1", 601, 700, "+", 2),
            ]
        )
    )
    strict = db.explain_junctions(junctions)["m1"]
    assert strict.status == "unique"
    assert strict.candidates == ("t1",)
    assert strict.evidence[1].transcripts == frozenset({"t1"})
    # Both are exact intron matches, so nothing is marked as spanning.
    assert all(not e.spanning_transcripts for e in strict.evidence)

    # 201-300 then 401-900 is a spanning observation: donor 400 is the end of
    # t1's exon 301-400 and acceptor 901 is the start of exon 901-1000, so the
    # intervening exons are explicitly skipped by this observation.
    spanning_junctions = read_junctions(
        _table(
            [
                ("m1", "chr1", 201, 300, "+", 1),
                ("m1", "chr1", 401, 900, "+", 2),
            ]
        )
    )
    strict = db.explain_junctions(spanning_junctions)["m1"]
    assert strict.status == "none"

    skipping = db.explain_junctions(
        spanning_junctions, allow_exon_skipping=True
    )["m1"]
    assert skipping.status == "unique"
    assert skipping.candidates == ("t1",)
    spanning = skipping.evidence[1].spanning_transcripts
    assert spanning == frozenset({"t1"})


def test_spanning_match_requires_annotated_boundaries(db):
    # 211-300 shares no exon boundary in t1, so even the explicit skipping
    # rule must reject it.
    junctions = read_junctions("molecule\tseqid\tstart\tend\tstrand\nm1\tchr1\t211\t300\t+\n")
    group = db.explain_junctions(junctions, allow_exon_skipping=True)["m1"]
    assert group.status == "none"
    assert group.evidence[0].transcripts == frozenset()


def test_strand_mismatch_rejects_plus_junction(db):
    junctions = read_junctions(
        _table([("m1", "chr2", 201, 300, "+", 1)])
    )
    group = db.explain_junctions(junctions)["m1"]
    assert group.status == "none"
    assert group.evidence[0].transcripts == frozenset()
    assert "strand" in group.reason


def test_plus_strand_ordering_uses_biological_direction(db):
    wrong_order = read_junctions(
        _table(
            [
                ("m1", "chr1", 401, 500, "+", 1),
                ("m1", "chr1", 201, 300, "+", 2),
            ]
        )
    )
    group = db.explain_junctions(wrong_order, transcripts=["t1"])["m1"]
    assert group.status == "none"
    assert group.reason == (
        "matching junctions are not in a consistent biological order"
    )


def test_minus_strand_ordering_uses_biological_direction(db):
    # On chr2 (-), introns are 201-300, 401-500, 601-700.  Transcription runs
    # from right to left, so a biologically ordered molecule visits 601-700,
    # then 401-500, then 201-300.
    ordered = read_junctions(
        _table(
            [
                ("m1", "chr2", 601, 700, "-", 1),
                ("m1", "chr2", 401, 500, "-", 2),
                ("m1", "chr2", 201, 300, "-", 3),
            ]
        )
    )
    group = db.explain_junctions(ordered, transcripts=["mt1"])["m1"]
    assert group.status == "unique"
    assert group.candidates == ("mt1",)

    wrong_order = read_junctions(
        _table(
            [
                ("m1", "chr2", 401, 500, "-", 1),
                ("m1", "chr2", 601, 700, "-", 2),
            ]
        )
    )
    group = db.explain_junctions(wrong_order, transcripts=["mt1"])["m1"]
    assert group.status == "none"
    assert group.reason == (
        "matching junctions are not in a consistent biological order"
    )


def test_conflicting_strands_within_molecule_have_no_path(db):
    junctions = [
        Junction("m1", 201, 300, strand="+"),
        Junction("m1", 401, 500, strand="-"),
    ]
    group = db.explain_junctions(junctions)["m1"]
    assert group.status == "none"
    assert group.reason == "conflicting strands within molecule"


def test_seqid_constraint(db):
    junctions = read_junctions(
        _table([("m1", "chr3", 201, 300, "+", 1)])
    )
    group = db.explain_junctions(junctions)["m1"]
    assert group.status == "none"
    assert group.evidence[0].transcripts == frozenset()


def test_narrowing_candidates_preserves_joint_rule(db):
    junctions = read_junctions(
        _table(
            [
                ("m1", "chr1", 201, 300, "+", 1),
                ("m1", "chr1", 401, 500, "+", 2),
            ]
        )
    )
    narrowed = db.explain_junctions(junctions, transcripts=["t2"])["m1"]
    assert narrowed.status == "unique"
    assert narrowed.candidates == ("t2",)
    # Evidence is computed against the narrowed candidate set, so the joint
    # rule is unchanged even though t1 is not considered here.
    assert narrowed.evidence[0].transcripts == frozenset({"t2"})


def test_transcript_featuretype_filter(db):
    junctions = read_junctions(
        _table([("m1", "chr1", 201, 300, "+", 1)])
    )
    group = db.explain_junctions(
        junctions, transcript_featuretype="mRNA"
    )["m1"]
    assert group.status == "shared"
    assert group.candidates == ("t1", "t2")


def test_unknown_observation_strand_is_not_constrained(db):
    junctions = read_junctions(
        _table([("m1", "chr1", 201, 300, ".", 1)])
    )
    group = db.explain_junctions(junctions)["m1"]
    assert group.status == "shared"
    assert group.candidates == ("t1", "t2")


def test_known_strand_does_not_match_unstranded_transcript():
    annotation = textwrap.dedent(
        """\
        chr1\t.\tmRNA\t100\t400\t.\t.\t.\tID=u1
        chr1\t.\texon\t100\t200\t.\t.\t.\tID=u1e1;Parent=u1
        chr1\t.\texon\t301\t400\t.\t.\t.\tID=u1e2;Parent=u1
        """
    )
    udb = gffutils.create_db(annotation, ":memory:", from_string=True)
    junctions = read_junctions(
        _table([("m1", "chr1", 201, 300, "+", 1)])
    )
    group = udb.explain_junctions(junctions)["m1"]
    assert group.status == "none"
    assert group.evidence[0].transcripts == frozenset()

    # An unknown observation strand is unconstrained, so the same annotation
    # can still explain it.
    unknown = read_junctions(_table([("m1", "chr1", 201, 300, ".", 1)]))
    assert udb.explain_junctions(unknown)["m1"].status == "unique"


def test_conflicting_seqids_within_molecule_have_no_path(db):
    junctions = [
        Junction("m1", 201, 300, seqid="chr1", strand="+", order=1),
        Junction("m1", 401, 500, seqid="chr2", strand="+", order=2),
    ]
    group = db.explain_junctions(junctions)["m1"]
    assert group.status == "none"
    assert group.reason == "conflicting seqids within molecule"


def test_allow_exon_skipping_does_not_cross_strand(db):
    # The spanning junction is on chr1 (+); the only transcript with matching
    # boundaries there is t1.  A minus-strand transcript must not rescue it.
    junctions = read_junctions(
        _table(
            [
                ("m1", "chr1", 201, 300, "+", 1),
                ("m1", "chr1", 401, 900, "+", 2),
            ]
        )
    )
    group = db.explain_junctions(
        junctions, transcripts=["mt1"], allow_exon_skipping=True
    )["m1"]
    assert group.status == "none"
    assert all(not e.transcripts for e in group.evidence)


def test_report_helpers_and_dicts(db):
    junctions = read_junctions(
        _table(
            [
                ("unique", "chr1", 601, 700, "+", 1),
                ("shared", "chr1", 201, 300, "+", 1),
                ("none", "chr1", 211, 300, "+", 1),
            ]
        )
    )
    report = db.explain_junctions(junctions)
    assert report.counts == {"unique": 1, "shared": 1, "none": 1}
    assert [g.molecule for g in report.unique] == ["unique"]
    assert [g.molecule for g in report.shared] == ["shared"]
    assert [g.molecule for g in report.unresolved] == ["none"]

    d = report.as_dicts()
    by_molecule = {x["molecule"]: x for x in d}
    assert by_molecule["unique"]["transcript"] == "t1"
    assert by_molecule["shared"]["candidates"] == ["t1", "t2"]
    assert by_molecule["none"]["reason"] == "no transcript matches any junction"
    assert by_molecule["none"]["junctions"][0]["matches"] == {}


def test_empty_input(db):
    report = db.explain_junctions([])
    assert len(report) == 0
    assert report.counts == {"unique": 0, "shared": 0, "none": 0}


def test_format_report(db):
    junctions = read_junctions(
        _table(
            [
                ("shared", "chr1", 201, 300, "+", 1),
                ("none", "chr1", 211, 300, "+", 1),
            ]
        )
    )
    text = gffutils.junctions.format_report(db.explain_junctions(junctions))
    lines = text.splitlines()
    assert lines[0] == (
        "molecule\tstatus\ttranscript\tcandidate_count\tdetail"
    )
    assert "shared\tshared\t\t2\tt1,t2" in lines
    assert "none\tnone\t\t0\tno transcript matches any junction" in lines
