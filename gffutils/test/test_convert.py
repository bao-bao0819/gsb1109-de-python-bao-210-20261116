import os
import shutil
import tempfile

import pytest

import gffutils
from gffutils import convert
from gffutils.feature import feature_from_line


GFF = """##gff-version 3
chr1\t.\tgene\t100\t500\t.\t-\t.\tID=g1
chr1\t.\tmRNA\t100\t500\t.\t-\t.\tID=t1;Parent=g1
chr1\t.\texon\t100\t200\t.\t-\t.\tID=e1;Parent=t1
chr1\t.\texon\t350\t500\t.\t-\t.\tID=e2;Parent=t1
chr1\t.\tCDS\t120\t200\t.\t-\t0\tID=c1;Parent=t1
chr1\t.\tCDS\t350\t420\t.\t-\t0\tID=c2;Parent=t1
"""


@pytest.fixture(autouse=True)
def isolated_tempdir():
    original = tempfile.tempdir
    path = os.path.join(os.path.dirname(__file__), "_convert_tmp")
    os.makedirs(path, exist_ok=True)
    tempfile.tempdir = path
    try:
        yield
    finally:
        tempfile.tempdir = original
        shutil.rmtree(path, ignore_errors=True)


def create_db(*lines):
    features = [feature_from_line(line) for line in lines if not line.startswith("#")]
    return gffutils.create_db(
        features,
        ":memory:",
        verbose=False,
        disable_infer_genes=True,
        disable_infer_transcripts=True,
    )


@pytest.fixture
def db():
    return create_db(*GFF.splitlines())


def make_db(*lines):
    return create_db(*lines)


def test_bed12_coordinates_and_blocks(db):
    record = convert.to_bed12_record("t1", db, name_field="ID")
    assert record.chrom == "chr1"
    assert record.chrom_start == 99
    assert record.chrom_end == 500
    assert record.strand == "-"
    assert record.gff_blocks == ((100, 200), (350, 500))
    assert record.block_sizes == (101, 151)
    assert record.block_starts == (0, 250)
    assert (
        record.to_bed12()
        == "chr1\t99\t500\tt1\t0\t-\t99\t500\t0,0,0\t2\t101,151\t0,250"
    )


def test_blocks_keep_order_and_identity(db):
    record = convert.to_bed12_record("t1", db, name_field="ID")
    assert [f.id for f in record.source_blocks] == ["e1", "e2"]
    assert [block for block in record.block_records()] == [
        (0, 100, 200, 101, 0),
        (1, 350, 500, 151, 250),
    ]
    assert record.index_of(350, 500) == 1


def test_round_trip_restores_logical_coverage_not_just_length(db):
    record = convert.to_bed12_record("t1", db, name_field="ID")
    reparsed = convert.parse_bed12(record.to_bed12())
    assert reparsed.gff_blocks == record.gff_blocks
    assert sum(end - start + 1 for start, end in reparsed.gff_blocks) == sum(
        end - start + 1 for start, end in record.gff_blocks
    )

    parent, children = reparsed.to_features()
    assert (parent.start, parent.stop, parent.strand) == (100, 500, "-")
    assert [(child.start, child.stop) for child in children] == [(100, 200), (350, 500)]


def test_round_trip_does_not_only_restore_total_length():
    gff = "\n".join(
        [
            "##gff-version 3",
            "chr2\t.\tmRNA\t10\t120\t.\t+\t.\tID=t;",
            "chr2\t.\texon\t10\t20\t.\t+\t.\tID=a;Parent=t;",
            "chr2\t.\texon\t110\t120\t.\t+\t.\tID=b;Parent=t;",
        ]
    )
    db = gffutils.create_db(
        gff,
        ":memory:",
        from_string=True,
        verbose=False,
        disable_infer_genes=True,
        disable_infer_transcripts=True,
    )
    first = convert.to_bed12_record("t", db, name_field="ID")
    assert first.gff_blocks == ((10, 20), (110, 120))

    # Same total coverage (22 bases) but a different block layout.
    other = convert.parse_bed12("chr2\t9\t120\tt\t0\t+\t9\t120\t0,0,0\t2\t11,11\t0,100")
    assert other.gff_blocks == ((10, 20), (110, 120))
    assert sum(e - s + 1 for s, e in other.gff_blocks) == sum(
        e - s + 1 for s, e in first.gff_blocks
    )
    assert other.gff_blocks != ((10, 30), (111, 120))


def test_single_base_feature_is_a_one_base_block():
    db = make_db("chr1\t.\texon\t5\t5\t.\t+\t.\tID=b;Parent=t;")
    record = convert.to_bed12_record("b", db, name_field="ID")
    assert record.chrom_start == 4
    assert record.chrom_end == 5
    assert record.gff_blocks == ((5, 5),)
    assert record.block_sizes == (1,)


def test_start_equals_end_remains_one_base_regardless_of_featuretype():
    db = make_db("chr1\t.\ttransposable_element_insertion_site\t5\t5\t.\t+\t.\tID=p;")
    record = convert.to_bed12_record("p", db, name_field="ID")
    assert record.gff_blocks == ((5, 5),)
    assert record.block_sizes == (1,)


def test_zero_length_point_event_is_unrepresentable():
    db = make_db("chr1\t.\texon\t10\t9\t.\t+\t.\tID=p;Parent=t;")
    with pytest.raises(convert.UnrepresentableFeatureError) as err:
        convert.to_bed12("p", db)
    assert "zero-length point event" in str(err.value)
    assert "refusing to swap endpoints" in str(err.value)


def test_reversed_interval_is_not_repaired_by_swapping():
    db = make_db("chr1\t.\texon\t10\t5\t.\t+\t.\tID=r;Parent=t;")
    with pytest.raises(convert.UnrepresentableFeatureError) as err:
        convert.to_bed12("r", db)
    assert "reversed coordinates" in str(err.value)
    assert "refusing to swap endpoints" in str(err.value)


def test_unrepresentable_objects_are_listed_separately():
    db = make_db(
        "chr1\t.\texon\t5\t5\t.\t+\t.\tID=ok;Parent=t;",
        "chr1\t.\texon\t10\t9\t.\t+\t.\tID=point;Parent=t;",
        "chr1\t.\texon\t20\t15\t.\t+\t.\tID=reversed;Parent=t;",
    )
    lines, skipped = db.bed12_many(["ok", "point", "reversed"], name_field="ID")
    assert len(lines) == 1
    assert lines[0].startswith("chr1\t4\t5\tok\t")
    assert [feature.id for feature, _ in skipped] == ["point", "reversed"]
    assert all("refusing to swap endpoints" in reason for _, reason in skipped)


def test_overlapping_blocks_are_unrepresentable():
    db = make_db(
        "chr1\t.\tmRNA\t1\t100\t.\t+\t.\tID=t;",
        "chr1\t.\texon\t1\t50\t.\t+\t.\tID=a;Parent=t;",
        "chr1\t.\texon\t40\t100\t.\t+\t.\tID=b;Parent=t;",
    )
    with pytest.raises(convert.UnrepresentableFeatureError) as err:
        convert.to_bed12("t", db)
    assert "overlapping or duplicate blocks" in str(err.value)


def test_blocks_must_span_transcript_ends():
    db = make_db(
        "chr1\t.\tmRNA\t1\t100\t.\t+\t.\tID=t;",
        "chr1\t.\texon\t10\t90\t.\t+\t.\tID=a;Parent=t;",
    )
    with pytest.raises(convert.UnrepresentableFeatureError) as err:
        convert.to_bed12("t", db)
    assert "does not match start of feature" in str(err.value)


@pytest.mark.parametrize(
    "line,message",
    [
        (
            "chr1\t0\t100\tt\t0\t+\t0\t100\t0,0,0\t2\t10\t0,20",
            "does not match",
        ),
        (
            "chr1\t0\t30\tt\t0\t+\t0\t30\t0,0,0\t2\t10,10\t5,20",
            "first block must start",
        ),
        (
            "chr1\t0\t100\tt\t0\t+\t0\t100\t0,0,0\t2\t10,10\t0,20",
            "last block must end",
        ),
        (
            "chr1\t0\t100\tt\t0\t+\t0\t100\t0,0,0\t2\t10,10\t5,0",
            "before previous block",
        ),
        (
            "chr1\t10\t5\tt\t0\t+\t0\t5\t0,0,0\t1\t1\t0",
            "greater than chromStart",
        ),
        (
            "chr1\t0\t100\tt\t0\t+\t-1\t100\t0,0,0\t1\t100\t0",
            "within",
        ),
        (
            "chr1\t0\t100\tt\t0\t.\t0\t100\t0,0,0\t0\t\t",
            "at least 1",
        ),
    ],
)
def test_malformed_bed12_is_rejected(line, message):
    with pytest.raises(convert.MalformedBed12Error) as err:
        convert.parse_bed12(line)
    assert message in str(err.value)


def test_thin_utrs_infer_thick_bounds():
    db = make_db(
        "chr1\t.\tmRNA\t100\t500\t.\t+\t.\tID=t;",
        "chr1\t.\texon\t100\t500\t.\t+\t.\tID=e;Parent=t;",
        "chr1\t.\tfive_prime_UTR\t100\t119\t.\t+\t.\tID=u5;Parent=t;",
        "chr1\t.\tthree_prime_UTR\t421\t500\t.\t+\t.\tID=u3;Parent=t;",
    )
    record = convert.to_bed12_record(
        "t",
        db,
        name_field="ID",
        thick_featuretype=None,
        thin_featuretype=["five_prime_UTR", "three_prime_UTR"],
    )
    assert (record.thick_start, record.thick_end) == (119, 420)


def test_single_utr_does_not_guess_thick_bounds():
    db = make_db(
        "chr1\t.\tmRNA\t100\t500\t.\t+\t.\tID=t;",
        "chr1\t.\texon\t100\t500\t.\t+\t.\tID=e;Parent=t;",
        "chr1\t.\tfive_prime_UTR\t100\t119\t.\t+\t.\tID=u5;Parent=t;",
    )
    with pytest.raises(convert.UnrepresentableFeatureError) as err:
        convert.to_bed12(
            "t",
            db,
            name_field="ID",
            thick_featuretype=None,
            thin_featuretype="five_prime_UTR",
        )
    assert "single UTR" in str(err.value)


def test_gtf_name_field():
    gtf = "\n".join(
        [
            'chr1\t.\ttranscript\t10\t40\t.\t+\t.\tgene_id "g"; transcript_id "t";',
            'chr1\t.\texon\t10\t20\t.\t+\t.\tgene_id "g"; transcript_id "t";',
            'chr1\t.\texon\t30\t40\t.\t+\t.\tgene_id "g"; transcript_id "t";',
        ]
    )
    db = gffutils.create_db(
        gtf,
        ":memory:",
        from_string=True,
        verbose=False,
        disable_infer_genes=True,
        disable_infer_transcripts=True,
    )
    record = convert.to_bed12_record(
        db["t"], db, child_type="exon", name_field="transcript_id"
    )
    assert record.name == "t"
    assert record.gff_blocks == ((10, 20), (30, 40))
