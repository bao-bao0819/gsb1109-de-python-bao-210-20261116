"""Tests for incremental point assembly from short-read streams."""

import io

import numpy as np
import pytest

import laspy
from laspy import errors
from laspy.streaming import PointStreamAssembler, StreamState
from tests.conftest import SIMPLE_LAS_FILE_PATH


class TrickleStream:
    """A file-like object that never returns more than `max_read` bytes."""

    def __init__(self, data: bytes, max_read: int) -> None:
        self._data = data
        self._pos = 0
        self.max_read = max_read

    def read(self, n: int) -> bytes:
        n = min(n, self.max_read)
        chunk = self._data[self._pos : self._pos + n]
        self._pos += len(chunk)
        return chunk

    def readinto(self, buffer) -> int:
        chunk = self.read(len(buffer))
        buffer[: len(chunk)] = chunk
        return len(chunk)

    def seek(self, pos: int, whence: int = io.SEEK_SET) -> int:
        self._pos = pos
        return pos

    def tell(self) -> int:
        return self._pos

    def seekable(self) -> bool:
        return True

    def close(self) -> None:
        pass


@pytest.fixture()
def simple_las():
    return laspy.read(SIMPLE_LAS_FILE_PATH)


@pytest.fixture()
def point_data(simple_las):
    return bytes(simple_las.points.memoryview().cast("B"))


def make_declared_points(count: int = 30):
    """Create high-entropy points with known return/flight-line controls."""
    header = laspy.LasHeader(version="1.2", point_format=0)
    record = laspy.ScaleAwarePointRecord.zeros(count, header=header)
    rng = np.random.default_rng(42)
    record.array["X"] = rng.integers(-(2**31), 2**31 - 1, count, dtype=np.int32)
    record.array["Y"] = rng.integers(-(2**31), 2**31 - 1, count, dtype=np.int32)
    record.array["Z"] = rng.integers(-(2**31), 2**31 - 1, count, dtype=np.int32)
    return_numbers = rng.integers(1, 4, count)
    record.array["bit_fields"] = (return_numbers & 0b111) | (
        (return_numbers & 0b111) << 3
    )
    record.array["point_source_id"] = rng.integers(10, 13, count, dtype=np.uint16)
    return header, record, return_numbers


def return_controls(return_numbers) -> dict:
    return {int(value): int((return_numbers == value).sum()) for value in (1, 2, 3)}


def test_short_reads_do_not_publish_a_smaller_file(point_data, simple_las):
    assembler = PointStreamAssembler(simple_las.header)
    for start in range(0, len(point_data), 3):
        result = assembler.feed(point_data[start : start + 3])
        assert result.state is not StreamState.COMPLETE
        assert not assembler.is_truncated

    assert assembler.finish() is StreamState.COMPLETE
    assert assembler.confirmed_point_count == simple_las.header.point_count
    assert np.array_equal(assembler.confirmed_prefix.array, simple_las.points.array)


def test_link_short_read_is_not_end_of_file(simple_las, point_data):
    assembler = PointStreamAssembler(simple_las.header)
    result = assembler.feed(point_data[: point_data.__len__() // 2], requested=10**9)

    assert result.short_read
    assert assembler.state is StreamState.SHORT_READ
    assert not assembler.at_eof

    # Bytes keep arriving, so this is not a final state.
    assembler.feed(point_data[len(point_data) // 2 : len(point_data) // 2 + 34])
    assert assembler.state is not StreamState.SHORT_READ
    assert not assembler.at_eof


def test_eof_before_declaration_is_truncation(simple_las, point_data):
    half = (len(point_data) // 2 // simple_las.header.point_format.size) * (
        simple_las.header.point_format.size
    )
    assembler = PointStreamAssembler(simple_las.header)
    assembler.feed(point_data[:half])

    assert assembler.finish() is StreamState.TRUNCATED
    assert assembler.point_count_residual == simple_las.header.point_count - (
        half // simple_las.header.point_format.size
    )
    with pytest.raises(errors.StreamDeclarationConflict):
        assembler.build()


def test_eof_with_partial_trailing_point_is_truncation(simple_las, point_data):
    assembler = PointStreamAssembler(simple_las.header)
    assembler.feed(point_data[:-1])

    assert assembler.pending_bytes == simple_las.header.point_format.size - 1
    assert assembler.finish() is StreamState.TRUNCATED


def test_more_points_than_declared_is_conflict(simple_las, point_data):
    assembler = PointStreamAssembler(
        simple_las.header, point_count=simple_las.header.point_count - 10
    )
    assembler.feed(point_data)

    assert assembler.state is StreamState.DECLARATION_CONFLICT
    assert assembler.declaration_conflict
    with pytest.raises(errors.StreamDeclarationConflict):
        assembler.finish(strict=True)


def test_overlapping_retransmission_is_not_counted_twice(simple_las, point_data):
    point_size = simple_las.header.point_format.size
    assembler = PointStreamAssembler(simple_las.header)

    chunks = []
    start = 0
    stride = 1000
    retransmit = 3 * point_size
    while start < len(point_data):
        end = min(len(point_data), start + stride)
        chunks.append(point_data[start:end])
        start = end - retransmit if end - retransmit > start else end

    duplicate_total = 0
    for chunk in chunks:
        duplicate_total += assembler.feed(chunk).duplicate_bytes

    assert duplicate_total >= len(chunks) - 1
    assert assembler.finish() is StreamState.COMPLETE
    assert assembler.confirmed_point_count == simple_las.header.point_count
    assert np.array_equal(assembler.confirmed_prefix.array, simple_las.points.array)


def test_overlap_by_offset_counts_existing_bytes_once(simple_las, point_data):
    point_size = simple_las.header.point_format.size
    assembler = PointStreamAssembler(simple_las.header)

    assembler.feed(point_data[: 3 * point_size], offset=0)
    result = assembler.feed(
        point_data[2 * point_size : 6 * point_size], offset=2 * point_size
    )

    assert result.duplicate_bytes == point_size
    assert assembler.confirmed_point_count == 6


def test_overlap_with_conflicting_evidence_is_an_error(simple_las, point_data):
    point_size = simple_las.header.point_format.size
    assembler = PointStreamAssembler(simple_las.header)
    assembler.feed(point_data[: 3 * point_size], offset=0)

    with pytest.raises(errors.StreamOverlapConflict):
        assembler.feed(b"\xff\xff\xff\xff", offset=2 * point_size)


def test_gap_at_offset_is_an_error(simple_las, point_data):
    assembler = PointStreamAssembler(simple_las.header)

    with pytest.raises(errors.StreamGapError):
        assembler.feed(point_data[100:120], offset=100)


def test_by_return_and_flightline_residuals():
    header, record, return_numbers = make_declared_points()
    raw = bytes(record.memoryview().cast("B"))
    flight_lines = record.array["point_source_id"]
    by_flightline = {
        int(value): int((flight_lines == value).sum()) for value in (10, 11, 12)
    }

    assembler = PointStreamAssembler(
        header,
        expected_by_return=return_controls(return_numbers),
        expected_by_flightline=by_flightline,
    )
    for start in range(0, len(raw), 5):
        assembler.feed(raw[start : start + 5])

    assert assembler.finish() is StreamState.COMPLETE
    assert assembler.control_residuals.is_zero


def test_control_count_conflict_is_reported_separately():
    header, record, return_numbers = make_declared_points()
    raw = bytes(record.memoryview().cast("B"))
    controls = return_controls(return_numbers)
    controls[1] += 1

    assembler = PointStreamAssembler(
        header, point_count=len(record), expected_by_return=controls
    )
    assembler.feed(raw)

    assert assembler.point_count_residual == 0
    assert assembler.finish() is StreamState.DECLARATION_CONFLICT
    assert assembler.residual_by_return[1] == 1


def test_header_return_counts_can_be_used_as_controls(simple_las, point_data):
    assembler = PointStreamAssembler(
        simple_las.header, use_header_control_counts=True
    )
    assembler.feed(point_data)

    assert assembler.finish() is StreamState.COMPLETE
    assert assembler.control_residuals.is_zero


@pytest.mark.parametrize("max_read", [1, 2, 3, 7, 17, 226, 227, 5000])
def test_las_reader_survives_short_reads(max_read, simple_las):
    with SIMPLE_LAS_FILE_PATH.open("rb") as raw_file:
        stream = TrickleStream(raw_file.read(), max_read=max_read)
        las = laspy.read(stream, closefd=False)

    assert len(las.points) == simple_las.header.point_count
    assert np.array_equal(las.points.array, simple_las.points.array)


def test_short_read_error_is_not_raised_by_default_reader(simple_las):
    assembler = PointStreamAssembler(simple_las.header)
    assembler.feed(b"")

    assert assembler.state is StreamState.SHORT_READ
    assert assembler.finish() is StreamState.TRUNCATED
