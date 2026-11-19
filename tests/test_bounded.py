from __future__ import annotations

from io import BytesIO

import pytest

from dissect.cstruct import IncompleteRead, cstruct, read_partial
from tests.util import NonSeekableStream


@pytest.fixture
def nested_cs(cs: cstruct, compiled: bool) -> cstruct:
    cs.load(
        """
        struct inner {
            uint32  count;
            uint32  values[count];
        };

        struct outer {
            uint32  count;
            inner   records[count];
        };
        """,
        compiled=compiled,
    )
    return cs


def _record_values(records: list) -> list[list[int]]:
    return [list(record.values) for record in records]


def _confirmed_records(error: IncompleteRead) -> list:
    """Walk an IncompleteRead chain and return the first list of confirmed records."""
    current = error
    while isinstance(current, IncompleteRead):
        if isinstance(current.partial, list) and current.partial and hasattr(current.partial[0], "values"):
            return current.partial
        current = current.error
    return []


def _well_formed() -> bytes:
    return (
        (2).to_bytes(4, "little")
        + (1).to_bytes(4, "little")
        + (10).to_bytes(4, "little")
        + (2).to_bytes(4, "little")
        + (20).to_bytes(4, "little")
        + (21).to_bytes(4, "little")
    )


def test_read_partial_matches_regular_read(nested_cs: cstruct) -> None:
    data = _well_formed()
    result = read_partial(nested_cs.outer, data)
    assert result.count == 2
    assert _record_values(result.records) == [[10], [20, 21]]
    assert result.dumps() == nested_cs.outer(data).dumps()


def test_corrupted_outer_count_keeps_confirmed_records(nested_cs: cstruct) -> None:
    # The outer header claims five records but the payload only holds two
    data = (5).to_bytes(4, "little") + _well_formed()[4:]

    with pytest.raises(IncompleteRead) as exc_info:
        read_partial(nested_cs.outer, data)

    error = exc_info.value
    assert error.path == ["records", 2, "count"]
    assert error.partial == {"count": 5}
    confirmed = _confirmed_records(error)
    assert _record_values(confirmed) == [[10], [20, 21]]


def test_corrupted_inner_count_cannot_read_through_parent_payload(nested_cs: cstruct) -> None:
    parent_size = 12
    data = (
        (1).to_bytes(4, "little")            # outer count
        + (100).to_bytes(4, "little")        # corrupted inner count
        + (10).to_bytes(4, "little")         # one real element
        + (0xDEADBEEF).to_bytes(4, "little")  # trailer beyond the parent payload
    )
    stream = BytesIO(data)

    with pytest.raises(IncompleteRead) as exc_info:
        read_partial(nested_cs.outer, stream, limit=parent_size)

    error = exc_info.value
    assert error.path == ["records", 0, "values", 1]
    innermost = error
    while isinstance(innermost.error, IncompleteRead):
        innermost = innermost.error
    assert innermost.partial == [10]

    # The reader stopped exactly at the parent boundary and never touched the trailer
    assert stream.tell() == parent_size
    assert nested_cs.uint32._read(stream, endian=nested_cs.endian) == 0xDEADBEEF


def test_prefix_consistency(nested_cs: cstruct) -> None:
    data = _well_formed()
    previous: list[list[int]] | None = None

    for length in range(4, len(data) + 1):
        prefix = data[:length]
        try:
            result = read_partial(nested_cs.outer, prefix)
            confirmed = _record_values(result.records)
        except IncompleteRead as error:
            confirmed = _record_values(_confirmed_records(error))

        # A shorter prefix may only confirm fewer elements, never a different count for the same bytes
        if previous is not None and confirmed:
            assert confirmed[: len(previous)] == previous
        if confirmed:
            previous = confirmed

    # The complete input is the only fully successful read
    result = read_partial(nested_cs.outer, data)
    assert _record_values(result.records) == [[10], [20, 21]]


def test_explicit_limit_bounds_a_longer_stream(nested_cs: cstruct) -> None:
    data = _well_formed()
    stream = BytesIO(data + b"\x99" * 8)

    result = read_partial(nested_cs.outer, stream, limit=len(data))
    assert result.count == 2
    assert stream.tell() == len(data)


def test_truncated_scalar_reports_incomplete_read(cs: cstruct) -> None:
    with pytest.raises(IncompleteRead) as exc_info:
        read_partial(cs.uint32, b"\x01\x02")

    error = exc_info.value
    assert error.partial is None
    assert error.offset == 2
    assert error.path == []


def test_short_fixed_array_keeps_confirmed_elements(cs: cstruct) -> None:
    # Two complete uint32 elements and one partial element
    data = b"\x01\x00\x00\x00\x02\x00\x00\x00\x03"
    with pytest.raises(IncompleteRead) as exc_info:
        read_partial(cs.uint32[4], data)

    error = exc_info.value
    assert error.partial == [1, 2]
    assert error.path == [2]


def test_short_char_array_keeps_confirmed_bytes(cs: cstruct) -> None:
    with pytest.raises(IncompleteRead) as exc_info:
        read_partial(cs.char[5], b"ab")

    assert exc_info.value.partial == b"ab"


def test_eof_arrays_on_non_seekable_streams(cs: cstruct) -> None:
    assert read_partial(cs.uint32["EOF"], NonSeekableStream(b"\x01\x00\x00\x00\x02\x00\x00\x00")) == [1, 2]
    assert read_partial(cs.char["EOF"], NonSeekableStream(b"abc")) == b"abc"
    assert read_partial(cs.wchar["EOF"], NonSeekableStream("ab".encode("utf-16-le"))) == "ab"

    cs.load("struct rec { uint32 a; uint32 b; };")
    records = read_partial(cs.rec["EOF"], NonSeekableStream(b"\x01\x00\x00\x00\x02\x00\x00\x00" * 2))
    assert [[record.a, record.b] for record in records] == [[1, 2], [1, 2]]


def test_bounded_eof_array_leaves_trailer_intact(cs: cstruct) -> None:
    body = b"\x01\x00\x00\x00\x02\x00\x00\x00"
    trailer = (0xDEADBEEF).to_bytes(4, "little")
    stream = BytesIO(body + trailer)

    assert read_partial(cs.uint32["EOF"], stream, limit=len(body)) == [1, 2]
    assert stream.tell() == len(body)
    assert cs.uint32._read(stream, endian=cs.endian) == 0xDEADBEEF


def test_null_terminated_arrays(cs: cstruct) -> None:
    cs.load(
        """
        struct strings {
            char    name[];
            uint32  marker;
        };
        struct wstrings {
            wchar   label[];
            uint32  marker;
        };
        """
    )

    data = b"abc\x00" + (0x11223344).to_bytes(4, "little")
    result = read_partial(cs.strings, NonSeekableStream(data))
    assert result.name == b"abc"
    assert result.marker == 0x11223344

    wdata = "ab".encode("utf-16-le") + b"\x00\x00" + (0x55667788).to_bytes(4, "little")
    result = read_partial(cs.wstrings, NonSeekableStream(wdata))
    assert result.label == "ab"
    assert result.marker == 0x55667788


def test_truncated_terminated_string_keeps_prefix(cs: cstruct) -> None:
    with pytest.raises(IncompleteRead) as exc_info:
        read_partial(cs.char[None], NonSeekableStream(b"abc"), limit=3)
    assert exc_info.value.partial == b"abc"

    with pytest.raises(IncompleteRead) as exc_info:
        read_partial(cs.wchar[None], NonSeekableStream("ab".encode("utf-16-le")), limit=4)
    assert exc_info.value.partial == "ab"


def test_null_terminated_numeric_array(cs: cstruct) -> None:
    cs.load("struct nums { uint32 values[]; uint32 marker; };")
    data = (
        (1).to_bytes(4, "little")
        + (2).to_bytes(4, "little")
        + (0).to_bytes(4, "little")
        + (9).to_bytes(4, "little")
    )
    result = read_partial(cs.nums, BytesIO(data))
    assert list(result.values) == [1, 2]
    assert result.marker == 9

    stream = BytesIO((1).to_bytes(4, "little") + (2).to_bytes(4, "little") + (9).to_bytes(4, "little"))
    with pytest.raises(IncompleteRead) as exc_info:
        read_partial(cs.uint32[None], stream, limit=8)
    assert list(exc_info.value.partial) == [1, 2]
    assert cs.uint32._read(stream, endian=cs.endian) == 9


def test_nested_records_on_non_seekable_stream(nested_cs: cstruct) -> None:
    data = _well_formed()
    result = read_partial(nested_cs.outer, NonSeekableStream(data))
    assert _record_values(result.records) == [[10], [20, 21]]

    with pytest.raises(IncompleteRead) as exc_info:
        read_partial(nested_cs.outer, NonSeekableStream(data[:16]))
    assert exc_info.value.path == ["records", 1, "values", 0]


def test_union_truncation_reports_incomplete_read(cs: cstruct) -> None:
    cs.load(
        """
        union choice {
            uint32  u;
            uint16  h[2];
        };
        """
    )
    result = read_partial(cs.choice, b"\x01\x00\x02\x00")
    assert list(result.h) == [1, 2]

    with pytest.raises(IncompleteRead):
        read_partial(cs.choice, b"\x01\x00")


def test_bitfield_partial_keeps_completed_group(cs: cstruct) -> None:
    cs.load(
        """
        struct bitf {
            uint16  a : 3;
            uint16  b : 5;
            uint32  c;
        };
        """
    )
    result = read_partial(cs.bitf, b"\xe0\x03" + b"\x01\x00\x00\x00")
    assert result.c == 1

    with pytest.raises(IncompleteRead) as exc_info:
        read_partial(cs.bitf, b"\xe0\x03\x01")
    assert "a" in exc_info.value.partial
    assert "b" in exc_info.value.partial
    assert "c" not in exc_info.value.partial
    assert exc_info.value.path == ["c"]


def test_aligned_struct_partial(cs: cstruct) -> None:
    cs.load("struct aligned { uint8 x; uint32 y; };", align=True)
    result = read_partial(cs.aligned, b"\x09" + b"\x00\x00\x00" + b"\x05\x00\x00\x00")
    assert (result.x, result.y) == (9, 5)

    with pytest.raises(IncompleteRead) as exc_info:
        read_partial(cs.aligned, b"\x09\x00\x00")
    assert exc_info.value.partial == {"x": 9}


def test_leb128_count_overclaim(cs: cstruct) -> None:
    cs.load("struct leb_holder { uleb128 count; uint8 vals[count]; };")
    result = read_partial(cs.leb_holder, b"\x03\x0a\x0b\x0c")
    assert result.count == 3
    assert list(result.vals) == [10, 11, 12]

    with pytest.raises(IncompleteRead) as exc_info:
        read_partial(cs.leb_holder, b"\x05\x0a\x0b")
    assert exc_info.value.partial["count"] == 5


def test_buffer_inputs(cs: cstruct) -> None:
    data = b"\x01\x00\x00\x00\x02\x00\x00\x00"
    assert read_partial(cs.uint32[2], bytearray(data)) == [1, 2]
    assert read_partial(cs.uint32[2], memoryview(data)) == [1, 2]

    with pytest.raises(IncompleteRead) as exc_info:
        read_partial(cs.uint32[3], memoryview(data[:6]))
    assert exc_info.value.partial == [1]


def test_endianness_argument_is_normalized(cs: cstruct) -> None:
    data = (0x01020304).to_bytes(4, "big")
    assert read_partial(cs.uint32, data, endian="big") == 0x01020304
    assert read_partial(cs.uint32, data, endian=">") == 0x01020304
    assert read_partial(cs.uint32, data) == 0x04030201


def test_pointer_dereference_through_bounded_reader(cs: cstruct) -> None:
    cs.load("struct ptr_holder { uint32* p; };")
    target_offset = 8
    buf = target_offset.to_bytes(8, "little") + (0x41424344).to_bytes(4, "little")

    result = read_partial(cs.ptr_holder, BytesIO(buf))
    assert int(result.p.dereference()) == 0x41424344


def test_multidimensional_array_keeps_per_layer_partials(cs: cstruct) -> None:
    cs.load("struct grid { uint8 cells[2][3]; };")
    result = read_partial(cs.grid, b"\x01\x02\x03\x04\x05\x06")
    assert [list(row) for row in result.cells] == [[1, 2, 3], [4, 5, 6]]

    with pytest.raises(IncompleteRead) as exc_info:
        read_partial(cs.grid, b"\x01\x02\x03\x04")

    error = exc_info.value
    assert error.path == ["cells", 1, 1]
    # Each array layer keeps its own confirmed elements
    assert [list(row) for row in error.error.partial] == [[1, 2, 3]]
    assert list(error.error.error.partial) == [4]


def test_corrupted_huge_count_does_not_overallocate_or_cross_boundary(cs: cstruct) -> None:
    cs.load("struct huge { uint32 count; uint32 vals[count]; };")
    body = (0xFFFFFFFF).to_bytes(4, "little") + b"\x01\x00\x00\x00\x02\x00\x00\x00"
    trailer = (0xCAFEBABE).to_bytes(4, "little")
    stream = BytesIO(body + trailer)

    with pytest.raises(IncompleteRead) as exc_info:
        read_partial(cs.huge, stream, limit=len(body))

    error = exc_info.value
    assert error.partial["count"] == 0xFFFFFFFF
    innermost = error
    while isinstance(innermost.error, IncompleteRead):
        innermost = innermost.error
    assert list(innermost.partial) == [1, 2]
    assert innermost.path == [2]

    # The corrupted count never drove reads past the bounded payload
    assert stream.tell() == len(body)
    assert cs.uint32._read(stream, endian=cs.endian) == 0xCAFEBABE
