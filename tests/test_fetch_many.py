# -*- coding: utf-8 -*-
"""Tests for merging multi-interval query hits by source occurrence."""

import gzip
import os
import struct

import pytest

from vcfpy import reader
from vcfpy.bgzf import BgzfWriter

HEADER = "##fileformat=VCFv4.3\n##contig=<ID=20,length=62435964>\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"

RECORDS = [
    "20\t100\t.\tA\tT\t.\t.\t.",
    "20\t200\t.\tA\tT\t.\t.\t.",
    "20\t200\t.\tA\tT\t.\t.\t.",
    "20\t300\t.\tA\tT\t.\t.\t.",
]


def _write_vcf(path, records, chunks):
    """Write a BGZF VCF and a hand-built tabix index.

    ``chunks`` is a list of ``(first_line, last_line)`` indices describing
    the index chunks.  The same line can legitimately appear in several
    chunks, mimicking overlapping bin entries.
    """
    writer = BgzfWriter(filename=path)
    writer.write(HEADER)
    offsets = []
    for line in records:
        offsets.append(writer.tell())
        writer.write(line + "\n")
    offsets.append(writer.tell())
    writer.close()

    names = b"20\x00"
    index = bytearray(b"TBI\x01")
    index += struct.pack("<i", 1)
    index += struct.pack("<7i", 2, 1, 2, 0, ord("#"), 0, len(names))
    index += names
    index += struct.pack("<i", len(chunks))
    for bin_number, (first, last) in enumerate(chunks, start=1):
        index += struct.pack("<Ii", bin_number, 1)
        index += struct.pack("<QQ", offsets[first], offsets[last + 1])
    index += struct.pack("<i", 1)
    index += struct.pack("<Q", offsets[0])
    with gzip.open(path + ".tbi", "wb") as f:
        f.write(bytes(index))


@pytest.fixture
def duplicate_vcf(tmp_path):
    """VCF with two identical lines and a chunk that repeats lines 1 and 2."""
    path = str(tmp_path / "duplicates.vcf.gz")
    # Chunks overlap: line 1 and 2 are reachable through both chunks.
    _write_vcf(path, RECORDS, [(0, 2), (1, 3)])
    return path


def _positions(hits):
    return [hit.POS for hit in hits]


def test_fetch_many_merges_overlapping_chunk_hits(duplicate_vcf):
    r = reader.Reader.from_path(duplicate_vcf)

    hits = r.fetch_many(["20:90-250"])

    # Lines 1 and 2 are in both chunks but represent one occurrence each.
    assert _positions(hits) == [100, 200, 200]
    assert len({hit.source_offset for hit in hits}) == 3


def test_fetch_many_keeps_identical_lines_as_separate_occurrences(duplicate_vcf):
    r = reader.Reader.from_path(duplicate_vcf)

    hits = r.fetch_many(["20:150-250"])

    assert _positions(hits) == [200, 200]
    assert len({hit.source_offset for hit in hits}) == 2
    assert hits[0]._core() == hits[1]._core()
    assert hits[0] != hits[1]
    assert hits[0].source_identity != hits[1].source_identity


def test_fetch_many_preserves_all_matching_intervals(duplicate_vcf):
    r = reader.Reader.from_path(duplicate_vcf)

    hits = r.fetch_many(["20:90-250", "20:150-250", "20:90-250"])

    by_position = {}
    for hit in hits:
        by_position.setdefault(hit.POS, []).append(hit.intervals)
    assert by_position[100] == [["20:90-250"]]
    assert by_position[200] == [["20:90-250", "20:150-250"], ["20:90-250", "20:150-250"]]


def test_fetch_many_query_order_does_not_determine_retained_hit(duplicate_vcf):
    r = reader.Reader.from_path(duplicate_vcf)

    forward = r.fetch_many(["20:90-250", "20:150-250"])
    reverse = r.fetch_many(["20:150-250", "20:90-250"])

    assert [h.source_offset for h in forward] == [h.source_offset for h in reverse]
    assert [h.POS for h in forward] == [h.POS for h in reverse]
    assert [sorted(h.intervals) for h in forward] == [sorted(h.intervals) for h in reverse]


def test_fetch_many_distinguishes_same_span_different_labels():
    path = os.path.join(os.path.dirname(__file__), "vcfs", "multi_contig.vcf.gz")
    r = reader.Reader.from_path(path)

    hits = r.fetch_many(
        [
            {"reference": "20", "start": 1230237, "end": 1234568, "gene": "A"},
            {"reference": "20", "start": 1230237, "end": 1234568, "gene": "B"},
        ]
    )

    assert _positions(hits) == [1230237, 1234567]
    assert hits[0].intervals == [
        {"reference": "20", "start": 1230237, "end": 1234568, "gene": "A"},
        {"reference": "20", "start": 1230237, "end": 1234568, "gene": "B"},
    ]


def test_fetch_many_rejects_unknown_reference(duplicate_vcf):
    r = reader.Reader.from_path(duplicate_vcf)

    with pytest.raises(ValueError, match="not found in index"):
        r.fetch_many(["21:1-100"])


def test_fetch_many_accepts_mapping_and_tuple_intervals():
    path = os.path.join(os.path.dirname(__file__), "vcfs", "multi_contig.vcf.gz")
    r = reader.Reader.from_path(path)

    hits = r.fetch_many(
        [
            {"reference": "20", "start": 1110695, "end": 1230236},
            ("20", 1230237, 1234568),
        ]
    )

    assert _positions(hits) == [1110696, 1230237, 1234567]
    assert hits[0].intervals == [{"reference": "20", "start": 1110695, "end": 1230236}]
    assert hits[1].intervals == [("20", 1230237, 1234568)]


def test_fetch_many_returns_records_with_source_identity():
    path = os.path.join(os.path.dirname(__file__), "vcfs", "multi_contig.vcf.gz")
    r = reader.Reader.from_path(path)

    hits = r.fetch_many(["20:1110000-1111000"])

    assert len(hits) == 1
    hit = hits[0]
    assert hit.CHROM == "20"
    assert hit.POS == 1110696
    assert hit.record.POS == 1110696
    assert hit.source_path == path
    assert hit.source_identity == (path, hit.source_offset)
    assert hit.intervals == ["20:1110000-1111000"]


def test_fetch_many_hits_iterate_calls_like_records():
    path = os.path.join(os.path.dirname(__file__), "vcfs", "multi_contig.vcf.gz")
    r = reader.Reader.from_path(path)

    hit = r.fetch_many(["20:1110000-1111000"])[0]

    assert [call.sample for call in hit] == ["NA00001", "NA00002", "NA00003"]
