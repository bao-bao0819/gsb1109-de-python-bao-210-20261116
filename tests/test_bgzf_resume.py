# -*- coding: utf-8 -*-
"""Tests for resuming VCF parsing at record boundaries over BGZF blocks.

The resume position is a BGZF virtual offset (block start in the high bits,
within-block offset in the low bits).  A line may span two compressed
blocks, so these tests deliberately build fixtures with tiny blocks where
nearly every record crosses a block boundary.
"""

import io
import pathlib
import struct
import zlib

import pytest

import vcfpy
from vcfpy.bgzf import BgzfReader, BgzfWriter, make_virtual_offset, split_virtual_offset
from vcfpy.exceptions import BgzfTruncatedError
from vcfpy.reader import Reader

HEADER = (
    "##fileformat=VCFv4.3\n"
    "##contig=<ID=1,length=1000000>\n"
    '##INFO=<ID=DP,Number=1,Type=Integer,Description="Depth">\n'
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
)


def vcf_text(num_records=200, padding=0):
    """Build deterministic VCF text with optional per-record padding."""
    records = []
    for pos in range(1, num_records + 1):
        records.append("1\t%d\t.\tA\tT\t.\t.\tDP=%d%s\n" % (pos, pos, "x" * padding))
    return HEADER + "".join(records)


def write_bgzf(path, text, block_size):
    writer = BgzfWriter(str(path), block_size=block_size)
    writer.write(text)
    writer.close()
    return path.read_bytes()


def compression_block(data):
    """Return a single BGZF block containing ``data`` (bytes)."""
    from vcfpy.bgzf import _bgzf_header

    compressor = zlib.compressobj(6, zlib.DEFLATED, -15)
    compressed = compressor.compress(data) + compressor.flush()
    block_size = len(compressed) + 25
    return (
        _bgzf_header
        + struct.pack("<H", block_size)
        + compressed
        + struct.pack("<I", zlib.crc32(data) & 0xFFFFFFFF)
        + struct.pack("<I", len(data))
    )


EOF_MARKER = (
    b"\x1f\x8b\x08\x04\x00\x00\x00\x00\x00\xff\x06\x00BC\x02\x00\x1b\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00"
)


def raw_bgzf(text, block_size):
    """Build BGZF bytes from text without going through the writer class."""
    payload = text.encode("latin-1")
    return b"".join(
        compression_block(payload[i : i + block_size]) for i in range(0, len(payload), block_size)
    ) + EOF_MARKER


def reader_for_bytes(data):
    """Create a VCF Reader over an in-memory BGZF stream."""
    return Reader.from_stream(BgzfReader(fileobj=io.BytesIO(data)))


def collect_positions(reader):
    return [(record.CHROM, record.POS) for record in reader]


@pytest.mark.parametrize("block_size", [16, 31, 64, 127, 1024, 65536])
def test_record_offsets_across_block_sizes(tmp_path, block_size):
    """Record boundaries are found regardless of the input block size."""
    data = write_bgzf(tmp_path / "in.vcf.gz", vcf_text(), block_size)
    reader = reader_for_bytes(data)

    offsets = [reader.tell()]
    records = []
    for record in reader:
        records.append(record)
        offsets.append(reader.tell())

    assert [(r.CHROM, r.POS) for r in records] == [("1", pos) for pos in range(1, 201)]
    assert len(offsets) == len(records) + 1
    assert len(set(offsets)) == len(offsets)
    # Every record must be addressable, including records split across blocks.
    assert sum(1 for offset in offsets if split_virtual_offset(offset)[1] > 0) > 0


def test_resume_does_not_repeat_previous_record(tmp_path):
    """Resuming at an offset yields the next record, never the previous one."""
    data = write_bgzf(tmp_path / "in.vcf.gz", vcf_text(50), 24)

    reference = reader_for_bytes(data)
    offsets = [reference.tell()]
    expected = []
    for record in reference:
        expected.append((record.CHROM, record.POS))
        offsets.append(reference.tell())

    for index, offset in enumerate(offsets):
        resumed = reader_for_bytes(data)
        resumed.seek(offset)
        assert collect_positions(resumed) == expected[index:]


def test_virtual_offset_is_not_a_byte_sum():
    """A within-block offset stays in the low 16 bits of a virtual offset."""
    data = raw_bgzf(vcf_text(5), 40)
    reader = BgzfReader(fileobj=io.BytesIO(data))
    reader.readline()

    offsets = []
    while True:
        offsets.append(reader.tell())
        if not reader.readline():
            break

    previous = None
    for offset in offsets:
        assert 0 <= split_virtual_offset(offset)[1] < 2**16
        block_start = split_virtual_offset(offset)[0]
        assert block_start % 1 == 0
        if previous is not None:
            # At least one record boundary lies strictly inside a block,
            # which cannot be represented by adding to raw byte offsets.
            assert block_start >= split_virtual_offset(previous)[0]
        previous = offset
        # Seek round trip must land exactly on the same virtual position.
        reader.seek(offset)
        assert reader.tell() == offset


def test_last_record_and_eof_boundaries(tmp_path):
    """The boundary after the last record can be resumed at (clean EOF)."""
    data = write_bgzf(tmp_path / "in.vcf.gz", vcf_text(3), 17)
    reader = reader_for_bytes(data)
    list(reader)
    eof_offset = reader.tell()

    resumed = reader_for_bytes(data)
    resumed.seek(eof_offset)
    assert list(resumed) == []


def test_seek_to_offset_makes_progress(tmp_path):
    """The parser must not treat the lookahead line as a new record."""
    data = write_bgzf(tmp_path / "in.vcf.gz", vcf_text(10), 19)
    reader = reader_for_bytes(data)
    first = next(reader)
    offset = reader.tell()

    reader.seek(offset)
    assert next(reader).POS == first.POS + 1


class ShortReadStream:
    """Binary stream wrapper that returns at most ``chunk`` bytes per read."""

    def __init__(self, data, chunk):
        self._stream = io.BytesIO(data)
        self._chunk = chunk

    def read(self, size=-1):
        if size is None or size < 0:
            size = self._chunk
        return self._stream.read(min(size, self._chunk))

    def seek(self, offset, whence=0):
        return self._stream.seek(offset, whence)

    def tell(self):
        return self._stream.tell()

    def close(self):
        self._stream.close()


@pytest.mark.parametrize("chunk", [1, 2, 3, 7])
def test_short_reads_are_not_truncation(chunk):
    """Short reads are retried and do not change the reconstructed records."""
    text = vcf_text(25, padding=3)
    data = raw_bgzf(text, 33)
    stream = ShortReadStream(data, chunk)
    reader = BgzfReader(fileobj=stream)

    lines = []
    while True:
        line = reader.readline()
        if not line:
            break
        lines.append(line)

    assert "".join(lines) == text


def test_truncated_block_is_distinguished_from_short_read():
    """A file ending inside a block raises BgzfTruncatedError."""
    data = raw_bgzf(vcf_text(5), 50)
    # Keep everything except the final EOF marker and the last few bytes of
    # the last data block, so the terminal state is incomplete.
    truncated = data[: -len(EOF_MARKER) - 7]
    reader = BgzfReader(fileobj=ShortReadStream(truncated, 3))
    with pytest.raises(BgzfTruncatedError):
        while reader.readline():
            pass


def test_terminal_record_without_newline(tmp_path):
    """A final record without a trailing newline is complete, not truncated."""
    text = HEADER + "".join("1\t%d\t.\tA\tT\t.\t.\tDP=%d\n" % (pos, pos) for pos in range(1, 11))
    text = text[:-1]  # drop the trailing newline of the last line
    data = write_bgzf(tmp_path / "in.vcf.gz", text, 23)

    reader = reader_for_bytes(data)
    offsets = [reader.tell()]
    records = []
    for record in reader:
        records.append(record.POS)
        offsets.append(reader.tell())
    assert records == list(range(1, 11))

    resumed = reader_for_bytes(data)
    resumed.seek(offsets[-2])
    assert [record.POS for record in resumed] == [10]
    resumed = reader_for_bytes(data)
    resumed.seek(offsets[-1])
    assert list(resumed) == []


def test_clean_eof_is_not_an_error():
    """Ending exactly at a block boundary is a normal end of input."""
    data = raw_bgzf(vcf_text(2), 50)
    reader = BgzfReader(fileobj=ShortReadStream(data, 2))
    assert "".join(list(reader)) == vcf_text(2)


def test_truncated_error_is_also_eof_and_value_error():
    """The truncation exception also satisfies legacy EOF/ValueError handlers."""
    assert issubclass(BgzfTruncatedError, EOFError)
    assert issubclass(BgzfTruncatedError, ValueError)
    assert issubclass(BgzfTruncatedError, vcfpy.VCFPyException)
    assert vcfpy.TruncatedBgzfError is BgzfTruncatedError


def test_plain_stream_tell_and_seek():
    """Plain text streams use their own tell() cookies for resumption."""
    text = vcf_text(6)
    reader = Reader.from_stream(io.StringIO(text))
    offset = reader.tell()
    first = next(reader)
    next_offset = reader.tell()

    resumed = Reader.from_stream(io.StringIO(text))
    resumed.seek(next_offset)
    assert next(resumed).POS == first.POS + 1
    assert offset != next_offset


def test_make_virtual_offset_roundtrip():
    """Virtual offset helpers agree with the bits stored by the reader."""
    block_start, within_block = 18239, 4
    offset = make_virtual_offset(block_start, within_block)
    assert split_virtual_offset(offset) == (block_start, within_block)
    assert (offset >> 16, offset & 0xFFFF) == (block_start, within_block)


def test_fixture_file_resume_matches_plain_parsing():
    """A real bgzip fixture can be resumed with the same header."""
    path = pathlib.Path(__file__).parent / "vcfs" / "multi_contig.vcf.gz"
    if not path.exists():  # pragma: no cover
        pytest.skip("Test VCF file not found")

    with path.open("rb") as handle:
        data = handle.read()

    reference = reader_for_bytes(data)
    offsets = [reference.tell()]
    expected = []
    for record in reference:
        expected.append((record.CHROM, record.POS))
        offsets.append(reference.tell())

    for index in range(0, len(expected), 3):
        resumed = reader_for_bytes(data)
        resumed.seek(offsets[index])
        assert collect_positions(resumed) == expected[index:]
