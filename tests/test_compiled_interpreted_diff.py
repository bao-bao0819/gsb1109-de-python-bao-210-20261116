"""Differential verification between the compiled and the interpreted binary layout paths.

Two independent implementations can read the same bytes wrong in the same way, so simply
comparing their output against each other would let a shared mistake validate itself.  The
tests in this module therefore cross-check three things for every sample:

1. the *field values* produced by the compiled reader and the interpreted reader;
2. the *layout* of every field: the source byte range it was read from, the padding bytes
   around it, and the stream position at which the next record starts;
3. a hand-written byte oracle that derives every value, offset and padding byte directly
   from the raw buffer with ``struct``/bit shifts alone.  The oracle never imports a cstruct
   reader, so an error that both paths share still fails the manual assertions.

Nested arrays, unions, bit fields spanning byte boundaries and structure-relative alignment
are combined in the same samples, so their interaction is exercised rather than only the
final values of isolated fields.  Every mismatch is reported at the first diverging field
together with its byte position.
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass, field as dataclass_field
from typing import Any

import pytest

from dissect.cstruct.cstruct import cstruct
from dissect.cstruct.types import BaseArray
from dissect.cstruct.types.enum import EnumMetaType
from dissect.cstruct.types.structure import Structure, Union, UnionProxy

# All endiannesses that the compiler generates distinct code for.
ENDIANS = ("<", ">", "!", "@", "=")
INT_FORMATS = {1: "b", 2: "h", 4: "i", 8: "q"}
UINT_FORMATS = {1: "B", 2: "H", 4: "I", 8: "Q"}


# ---------------------------------------------------------------------------
# Manual byte oracle
# ---------------------------------------------------------------------------


@dataclass
class ManualField:
    """A single field value derived straight from the raw bytes."""

    path: tuple[str, ...]
    value: Any
    start: int
    end: int


@dataclass
class ManualTrace:
    """The independently computed layout of one record."""

    fields: list[ManualField] = dataclass_field(default_factory=list)
    # Byte ranges that no field covers: (start, end, value) for every padding gap.
    padding: list[tuple[int, int, int]] = dataclass_field(default_factory=list)
    next_offset: int = 0
    size: int = 0

    def value_map(self) -> dict[tuple[str, ...], Any]:
        return {field.path: field.value for field in self.fields}

    def spans(self) -> dict[tuple[str, ...], tuple[int, int]]:
        return {field.path: (field.start, field.end) for field in self.fields}


class ManualOracle:
    """Byte-level oracle that mirrors the cstruct layout without using any cstruct reader.

    Only the Python ``struct`` module and explicit bit shifts are used, so a bug shared by the
    compiled and interpreted readers cannot be introduced through this class.
    """

    def __init__(self, data: bytes, endian: str):
        self.data = data
        self.endian = endian
        self.struct_formats = {
            size: ("=" if endian == "@" else endian) + fmt for size, fmt in UINT_FORMATS.items()
        }

    def uint(self, offset: int, size: int) -> int:
        return struct.unpack_from(self.struct_formats[size], self.data, offset)[0]

    def int(self, offset: int, size: int) -> int:
        return struct.unpack_from(
            ("=" if self.endian == "@" else self.endian) + INT_FORMATS[size], self.data, offset
        )[0]

    def record(
        self,
        layout: list[tuple[str, str, int, int | None]],
        start: int = 0,
        path: tuple[str, ...] = (),
    ) -> ManualTrace:
        """Lay out a packed record.

        Each layout entry is ``(name, kind, size, bits)`` where ``kind`` is one of
        ``u``/``i`` (integer), ``bytes``, ``bits`` (start of a bit buffer) or ``pad``.
        Bit fields that share a buffer are given as consecutive ``bits`` entries; the first
        one carries the full buffer size in ``size`` and the remaining ones a size of 0.
        """
        trace = ManualTrace()
        offset = start
        bit_offset = 0
        bit_buffer_size = 0
        bit_buffer_value = 0

        for name, kind, size, bits in layout:
            field_path = (*path, name)

            if kind == "pad":
                end = offset + size
                for index in range(offset, end):
                    trace.padding.append((index, index + 1, self.data[index]))
                offset = end
                continue

            if kind == "bits":
                if size:
                    bit_buffer_size = size
                    bit_buffer_value = self.uint(offset, size)
                    bit_offset = 0
                    field_start = offset
                    offset += size
                else:
                    field_start = offset - bit_buffer_size

                if self.endian == "<":
                    value = (bit_buffer_value >> bit_offset) & ((1 << bits) - 1)
                else:
                    shift = bit_buffer_size * 8 - bit_offset - bits
                    value = (bit_buffer_value >> shift) & ((1 << bits) - 1)
                bit_offset += bits

                trace.fields.append(ManualField(field_path, value, field_start, field_start + bit_buffer_size))
                continue

            field_start = offset
            if kind == "bytes":
                value = self.data[offset : offset + size]
            elif kind == "u":
                value = self.uint(offset, size)
            else:
                value = self.int(offset, size)
            offset += size
            trace.fields.append(ManualField(field_path, value, field_start, offset))

        trace.next_offset = offset
        trace.size = offset - start
        return trace


# ---------------------------------------------------------------------------
# Tracing the two cstruct paths
# ---------------------------------------------------------------------------


@dataclass
class PathTrace:
    """What one cstruct path observed while reading a record."""

    values: dict[tuple[str, ...], Any]
    spans: dict[tuple[str, ...], tuple[int, int]]
    padding: list[tuple[int, int, int]]
    next_offset: int
    dumped: bytes


class TracingStream(io.BytesIO):
    """Stream that records every ``read``/``seek`` so the covered source bytes are known."""

    def __init__(self, data: bytes):
        super().__init__(data)
        self.events: list[tuple[str, int, int]] = []

    def read(self, size: int = -1) -> bytes:
        start = self.tell()
        result = super().read(size)
        self.events.append(("read", start, start + len(result)))
        return result

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        result = super().seek(offset, whence)
        self.events.append(("seek", result, result))
        return result


def _leaf(value: Any) -> Any:
    return getattr(value, "value", value)


def _is_scalar(field_type: Any, value: Any) -> bool:
    # Char/Wchar arrays expose bytes/str instead of a list, so they are leaf values.
    return isinstance(value, (bytes, str, bytearray)) or not (
        issubclass(field_type, BaseArray) or issubclass(field_type, Structure)
    )


def _walk_type(
    field_type: Any,
    value: Any,
    offset: int,
    path: tuple[str, ...],
    values: dict[tuple[str, ...], Any],
    spans: dict[tuple[str, ...], tuple[int, int]],
    dynamic_sizes: dict[str, int],
    field_name: str | None = None,
) -> int:
    """Walk a value of one type at a byte offset, returning the offset just after it."""
    if _is_scalar(field_type, value):
        size = field_type.__size__
        if size is None and field_name is not None:
            size = dynamic_sizes.get(field_name, 0)
        spans[path] = (offset, offset + size)
        values[path] = _leaf(value)
        return offset + size

    if issubclass(field_type, BaseArray):
        element_type = field_type.__type__
        cursor = offset
        for index, item in enumerate(value):
            cursor = _walk_type(
                element_type, item, cursor, (*path, str(index)), values, spans, dynamic_sizes
            )
        return cursor

    if issubclass(field_type, Structure):
        target = value.__target__ if isinstance(value, UnionProxy) else value
        member_sizes = getattr(target, "__dynamic_sizes__", {})
        is_union = issubclass(field_type, Union)

        if is_union:
            for member in field_type.__fields__:
                member_offset = offset + (member.offset or 0)
                _walk_field(
                    member, target, member_offset, (*path, member._name), values, spans, member_sizes
                )
            return offset + field_type.__size__

        cursor = offset
        bit_start = 0
        bit_size = 0
        for member in field_type.__fields__:
            if member.bits:
                if member.offset is not None:
                    cursor = offset + member.offset
                    bit_start = cursor
                    buffer_type = (
                        member.type.__type__ if isinstance(member.type, EnumMetaType) else member.type
                    )
                    bit_size = buffer_type.__size__
                    cursor += bit_size
                spans[(*path, member._name)] = (bit_start, bit_start + bit_size)
                values[(*path, member._name)] = _leaf(getattr(target, member._name))
                continue

            if member.offset is not None:
                cursor = offset + member.offset
            cursor = _walk_field(
                member, target, cursor, (*path, member._name), values, spans, member_sizes
            )
        return cursor

    size = field_type.__size__
    spans[path] = (offset, offset + size)
    values[path] = _leaf(value)
    return offset + size


def _walk_field(
    field: Any,
    obj: Any,
    offset: int,
    path: tuple[str, ...],
    values: dict[tuple[str, ...], Any],
    spans: dict[tuple[str, ...], tuple[int, int]],
    dynamic_sizes: dict[str, int],
) -> int:
    return _walk_type(
        field.type,
        getattr(obj, field._name),
        offset,
        path,
        values,
        spans,
        dynamic_sizes,
        field._name,
    )


def collect_layout(struct_type: type, obj: Any) -> tuple[dict, dict]:
    """Derive every leaf value and its source span from field metadata and the parsed object."""
    values: dict[tuple[str, ...], Any] = {}
    spans: dict[tuple[str, ...], tuple[int, int]] = {}
    _walk_type(
        struct_type, obj, 0, (), values, spans, getattr(obj, "__dynamic_sizes__", {})
    )
    return values, spans


def _merged_reads(stream: "TracingStream") -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for kind, s, e in sorted(stream.events):
        if kind != "read":
            continue
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def trace_path(struct_type: type, data: bytes, endian: str) -> PathTrace:
    """Read a record with one path and capture values, spans, padding and the next offset."""
    stream = TracingStream(data)
    start = stream.tell()
    obj = struct_type._read(stream, endian=endian)
    next_offset = stream.tell()

    values, spans = collect_layout(struct_type, obj)
    dumped = struct_type.dumps(obj, endian=endian)

    # Independent I/O evidence: every metadata-derived span must really be covered by a stream
    # read of this path, so a wrong seek/read cannot hide behind correct layout metadata.
    covered_ranges = _merged_reads(stream)
    for path, (span_start, span_end) in spans.items():
        if not any(read_start <= span_start and span_end <= read_end for read_start, read_end in covered_ranges):
            raise AssertionError(
                f"{struct_type.__name__}: span {'.'.join(path)} {span_start}:{span_end} "
                f"is not covered by any stream read {covered_ranges}"
            )

    # Padding is defined by the layout metadata: every position no field span covers.  The
    # compiler reads whole blocks at once, so stream events alone would hide internal padding.
    covered = {pos for s, e in spans.values() for pos in range(s, e)}
    padding = [
        (index, index + 1, data[index])
        for index in range(start, next_offset)
        if index not in covered
    ]

    return PathTrace(values, spans, padding, next_offset, dumped)


# ---------------------------------------------------------------------------
# First-divergence assertions
# ---------------------------------------------------------------------------


def assert_first_divergence(
    label: str,
    compiled: PathTrace,
    interpreted: PathTrace,
    manual: ManualTrace,
    data: bytes,
    forced_padding: frozenset[int] = frozenset(),
) -> None:
    """Compare one aspect of all three paths and fail at the first differing field.

    ``forced_padding`` lists positions that are structural padding even when a scalar union
    alternative's span covers them (e.g. padding internal to a structure stored in the union).
    """
    manual_values = manual.value_map()
    manual_spans = manual.spans()
    padding_positions = {start for start, _, _ in manual.padding} | set(forced_padding)

    # Iterate in the manual field order, which is the byte order of the layout, so the reported
    # failure is the first divergence rather than an arbitrary lexicographically early path.
    ordered_paths = [field.path for field in manual.fields]
    extra_paths = (set(compiled.values) | set(interpreted.values)) - set(ordered_paths)
    for path in [*ordered_paths, *sorted(extra_paths)]:
        key = ".".join(path)
        for name, trace in (("compiled", compiled), ("interpreted", interpreted)):
            if path not in trace.values:
                pytest.fail(f"{label}: field {key!r} missing from {name} path at first divergence")
            if trace.values[path] != manual_values[path]:
                pytest.fail(
                    f"{label}: first value divergence at {key!r}\n"
                    f"  manual={manual_values[path]!r} {name}={trace.values[path]!r}\n"
                    f"  bytes={data.hex()}"
                )
        if compiled.values[path] != interpreted.values[path]:
            pytest.fail(
                f"{label}: compiled/interpreted value divergence at {key!r}: "
                f"{compiled.values[path]!r} != {interpreted.values[path]!r}"
            )

        if path in manual_spans:
            expected_span = manual_spans[path]
            for name, trace in (("compiled", compiled), ("interpreted", interpreted)):
                actual = trace.spans.get(path)
                if actual != expected_span:
                    pytest.fail(
                        f"{label}: first source-span divergence at {key!r}\n"
                        f"  manual={expected_span} {name}={actual}"
                    )

    if compiled.next_offset != manual.next_offset or interpreted.next_offset != manual.next_offset:
        pytest.fail(
            f"{label}: next-record offset divergence: "
            f"manual={manual.next_offset} compiled={compiled.next_offset} "
            f"interpreted={interpreted.next_offset}"
        )

    manual_padding_positions = {start for start, _, _ in manual.padding}
    for name, trace in (("compiled", compiled), ("interpreted", interpreted)):
        observed_positions = {start for start, _, _ in trace.padding}
        # Positions the path leaves uncovered must all be known padding, and every oracle-derived
        # gap must be uncovered too.  Forced positions (gaps hidden under a scalar union member)
        # are validated through the zero-byte round-trip check below instead.
        if (
            not observed_positions <= padding_positions
            or not (manual_padding_positions - set(forced_padding)) <= observed_positions
        ):
            pytest.fail(
                f"{label}: padding divergence:\n"
                f"  manual={sorted(padding_positions)}\n  {name}={sorted(observed_positions)}"
            )

    if (
        compiled.dumped != data[: manual.next_offset]
        or interpreted.dumped != data[: manual.next_offset]
        or any(
            trace.dumped[index] != (0 if index in padding_positions else data[index])
            for trace in (compiled, interpreted)
            for index in range(manual.next_offset)
        )
    ):
        # Structural padding has no value to preserve: both writers emit zeroes there, even when
        # the source buffer happened to contain other bytes.  Compare field bytes exactly and
        # require zeroes at every padding position instead of a blind whole-buffer comparison.
        for name, trace in (("compiled", compiled), ("interpreted", interpreted)):
            for index in range(manual.next_offset):
                actual = trace.dumped[index]
                if index in padding_positions:
                    if actual != 0:
                        pytest.fail(f"{label}: padding byte at {index} not zeroed by {name}: {actual:#x}")
                elif actual != data[index]:
                    pytest.fail(
                        f"{label}: field byte divergence at {index} ({name}): "
                        f"source={data[index]:#x} dumped={actual:#x}"
                    )


def make_pair(cdef: str, *, align: bool = False) -> tuple[cstruct, cstruct]:
    """Return matching compiled and interpreted cstruct instances for one definition."""
    return cstruct().load(cdef, align=align), cstruct().load(cdef, compiled=False, align=align)


# ---------------------------------------------------------------------------
# Samples combining nested arrays, unions, cross-byte bit fields and alignment
# ---------------------------------------------------------------------------


COMPOSITE_DEF = """
enum Kind : uint8 {
    PLAIN = 1,
    RICH = 2,
};

flag Mode : uint16 {
    READ = 1,
    WRITE = 2,
};

struct point {
    int16   x;
    int16   y;
};

struct entry {
    uint8   tag;
    point   path[2];
    union {
        struct {
            uint32  length;
            uint16  code;
        } rich;
        struct {
            uint8   raw[6];
        } plain;
    } body;
    uint16  flags : 3;
    uint16  mode  : 9;
    Kind    kind  : 4;
    uint8   tail;
};

struct record {
    uint32  magic;
    uint16  count;
    entry   items[2];
    point   route[count];
    char    note[4];
};
"""


def _build_composite_sample(endian: str) -> bytes:
    """Construct a record by hand so every expected byte position is known."""
    e = "=" if endian == "@" else endian
    out = bytearray()
    out += struct.pack(e + "I", 0x44454647)
    out += struct.pack(e + "H", 2)

    for tag in (0x11, 0x22):
        out += struct.pack("B", tag)
        # point path[2]: four int16 values
        out += struct.pack(e + "hhhh", tag, tag + 1, tag + 2, tag + 3)
        # union body, 6 bytes: length (uint32) + code (uint16)
        out += struct.pack(e + "IH", 0x0A0B0C00 + tag, 0x1000 + tag)
        # first bit buffer (uint16): flags=3 bits, mode=9 bits
        if e == "<":
            bit_value = 0b101 | (0x0F0 << 3)
            kind_byte = 2  # kind lives in the low nibble for little endian
        else:
            # Big endian consumes the most significant bits first.
            bit_value = (0b101 << 13) | (0x0F0 << 4)
            kind_byte = 2 << 4
        out += struct.pack(e + "H", bit_value)
        # second bit buffer (uint8): the Kind enum occupies 4 bits of its own value
        out += struct.pack("B", kind_byte)
        out += struct.pack("B", 0x7F)

    # route[count=2]: four int16 values
    out += struct.pack(e + "hhhh", 0x0101, 0x0202, 0x0303, 0x0404)
    out += b"note"
    return bytes(out)


def _manual_composite(data: bytes, endian: str) -> ManualTrace:
    """Independent oracle for the composite sample, including nested and union members."""
    oracle = ManualOracle(data, endian)
    trace = oracle.record(
        [
            ("magic", "u", 4, None),
            ("count", "u", 2, None),
        ]
    )
    count = trace.value_map()[("count",)]

    for item_index in range(2):
        base = ("items", str(item_index))
        item_start = trace.next_offset
        item = oracle.record(
            [
                ("tag", "u", 1, None),
            ],
            start=item_start,
            path=base,
        )
        trace.fields.extend(item.fields)
        # Nested static array of structures, flattened per element.
        point_offset = item.next_offset
        for point_index in range(2):
            point_path = (*base, "path", str(point_index))
            point_trace = oracle.record(
                [("x", "i", 2, None), ("y", "i", 2, None)],
                start=point_offset,
                path=point_path,
            )
            trace.fields.extend(point_trace.fields)
            point_offset = point_trace.next_offset

        # The union is read once but both interpretations must expose every member.
        union_start = point_offset
        rich = oracle.record(
            [("length", "u", 4, None), ("code", "u", 2, None)],
            start=union_start,
            path=(*base, "body", "rich"),
        )
        plain_value = data[union_start : union_start + 6]
        trace.fields.extend(rich.fields)
        # The uint8[6] array is exposed as six indexed integer leaves on both cstruct paths.
        for raw_index, raw_byte in enumerate(plain_value):
            raw_path = (*base, "body", "plain", "raw", str(raw_index))
            trace.fields.append(
                ManualField(raw_path, raw_byte, union_start + raw_index, union_start + raw_index + 1)
            )
        point_offset = union_start + 6

        bits = oracle.record(
            [
                ("flags", "bits", 2, 3),
                ("mode", "bits", 0, 9),
                # The Kind enum bit field starts a separate uint8 bit buffer.
                ("kind", "bits", 1, 4),
            ],
            start=point_offset,
            path=base,
        )
        trace.fields.extend(bits.fields)
        point_offset = bits.next_offset

        tail = oracle.record([("tail", "u", 1, None)], start=point_offset, path=base)
        trace.fields.extend(tail.fields)
        trace.next_offset = tail.next_offset

    route_start = trace.next_offset
    for route_index in range(count):
        route_path = ("route", str(route_index))
        route_trace = oracle.record(
            [("x", "i", 2, None), ("y", "i", 2, None)], start=route_start, path=route_path
        )
        trace.fields.extend(route_trace.fields)
        route_start = route_trace.next_offset
    trace.next_offset = route_start

    note = oracle.record([("note", "bytes", 4, None)], start=trace.next_offset)
    trace.fields.extend(note.fields)
    trace.next_offset = note.next_offset
    trace.size = trace.next_offset
    return trace


@pytest.mark.parametrize("endian", ENDIANS)
def test_composite_interaction(endian: str) -> None:
    """Nested arrays, a union and cross-byte bit fields interact in one record on both paths."""
    compiled_cs, interpreted_cs = make_pair(COMPOSITE_DEF)
    data = _build_composite_sample(endian)

    manual = _manual_composite(data, endian)
    compiled = trace_path(compiled_cs.record, data, endian)
    interpreted = trace_path(interpreted_cs.record, data, endian)

    assert_first_divergence("composite", compiled, interpreted, manual, data)

    # The union members must overlap the same six source bytes (verified via spans, not dumps,
    # since a nested proxy re-serializes in the cstruct default endian).
    obj = compiled_cs.record(data, endian=endian)
    length_start = compiled.spans[("items", "0", "body", "rich", "length")][0]
    raw_spans = [
        compiled.spans[("items", "0", "body", "plain", "raw", str(i))] for i in range(6)
    ]
    assert raw_spans[0][0] == length_start
    assert raw_spans[-1][1] - length_start == 6
    assert bytes(obj.items[0].body.plain.raw) == data[length_start : length_start + 6]
    assert obj.items[0].mode == 0x0F0
    assert obj.items[0].flags == 0b101
    assert obj.items[0].kind == compiled_cs.Kind.RICH


ALIGNED_DEF = """
struct inner {
    uint8   a;
    uint32  b;
};

struct frame {
    uint8   lead;
    inner   slot;
    uint16  mid;
    uint64  wide;
    uint8   trail;
};
"""


def _manual_aligned(data: bytes, endian: str) -> ManualTrace:
    """Oracle for structure-relative alignment: padding is derived from alignment rules."""
    oracle = ManualOracle(data, endian)
    trace = ManualTrace()

    def emit(name: str, kind: str, size: int, offset: int, path: tuple[str, ...]) -> int:
        if kind == "u":
            value = oracle.uint(offset, size)
        else:
            value = data[offset : offset + size]
        trace.fields.append(ManualField((*path, name), value, offset, offset + size))
        return offset + size

    def pad(offset: int, alignment: int) -> int:
        aligned = offset + (-offset & (alignment - 1))
        for index in range(offset, aligned):
            trace.padding.append((index, index + 1, data[index]))
        return aligned

    offset = 0
    offset = emit("lead", "u", 1, offset, ())
    # inner starts aligned to 4 relative to the frame
    offset = pad(offset, 4)
    inner_start = offset
    offset = emit("a", "u", 1, offset, ("slot",))
    offset = pad(offset, 4)
    offset = emit("b", "u", 4, offset, ("slot",))
    # inner size is rounded up to its 4-byte alignment
    inner_end = inner_start + 8
    for index in range(offset, inner_end):
        trace.padding.append((index, index + 1, data[index]))
    offset = inner_end

    offset = pad(offset, 2)
    offset = emit("mid", "u", 2, offset, ())
    offset = pad(offset, 8)
    offset = emit("wide", "u", 8, offset, ())
    offset = emit("trail", "u", 1, offset, ())
    offset = pad(offset, 8)

    trace.next_offset = offset
    trace.size = offset
    return trace


@pytest.mark.parametrize("endian", ENDIANS)
def test_structure_relative_alignment(endian: str) -> None:
    """Alignment padding and field offsets are verified byte by byte, not just by final values."""
    compiled_cs, interpreted_cs = make_pair(ALIGNED_DEF, align=True)
    e = "=" if endian == "@" else endian

    data = bytearray(32)
    struct.pack_into("B", data, 0, 0xAA)
    struct.pack_into("B", data, 4, 0x01)  # slot.a at absolute offset 4
    struct.pack_into(e + "I", data, 8, 0x11223344)  # slot.b at offset 8
    struct.pack_into(e + "H", data, 12, 0x5566)  # mid at offset 12
    struct.pack_into(e + "Q", data, 16, 0x778899AABBCCDDEE)  # wide at offset 16
    struct.pack_into("B", data, 24, 0xBB)  # trail at offset 24
    data = bytes(data)

    manual = _manual_aligned(data, endian)
    compiled = trace_path(compiled_cs.frame, data, endian)
    interpreted = trace_path(interpreted_cs.frame, data, endian)

    assert_first_divergence("aligned", compiled, interpreted, manual, data)
    assert len(compiled_cs.frame) == len(interpreted_cs.frame) == manual.next_offset


CROSS_BYTE_DEF = """
enum Nibble : uint8 {
    N0 = 0,
};

struct bitset {
    uint8   low : 3;
    uint8   mid : 5;
    uint16  wide_hi : 7;
    uint16  wide_lo : 9;
    Nibble  enum_bit : 4;
    uint32  after;
    uint8   high_a : 4;
    uint8   high_b : 4;
};
"""


def _pack_bits(endian: str, size: int, fields: list[tuple[int, int]]) -> bytes:
    """Pack ``(value, bits)`` entries into one integer value using the cstruct bit order."""
    value = 0
    if endian == "<":
        used = 0
        for field_value, bits in fields:
            value |= field_value << used
            used += bits
    else:
        total = size * 8
        used = 0
        for field_value, bits in fields:
            value |= field_value << (total - used - bits)
            used += bits
    return struct.pack(("=" if endian == "@" else endian) + UINT_FORMATS[size], value)


def _manual_cross_byte(data: bytes, endian: str) -> ManualTrace:
    oracle = ManualOracle(data, endian)
    trace = oracle.record(
        [
            ("low", "bits", 1, 3),
            ("mid", "bits", 0, 5),
            ("wide_hi", "bits", 2, 7),
            ("wide_lo", "bits", 0, 9),
            ("enum_bit", "bits", 1, 4),
            ("after", "u", 4, None),
            ("high_a", "bits", 1, 4),
            ("high_b", "bits", 0, 4),
        ]
    )
    return trace


@pytest.mark.parametrize("endian", ENDIANS)
def test_cross_byte_bitfields(endian: str) -> None:
    """Bit fields sharing a buffer across byte boundaries keep identical spans and values."""
    compiled_cs, interpreted_cs = make_pair(CROSS_BYTE_DEF)
    e = "=" if endian == "@" else endian

    data = bytearray()
    data += _pack_bits(endian, 1, [(0b101, 3), (0b11101, 5)])
    data += _pack_bits(endian, 2, [(0b1100110, 7), (0b011001100, 9)])
    data += _pack_bits(endian, 1, [(0b1010, 4)])
    data += struct.pack(e + "I", 0xDEADBEEF)
    data += _pack_bits(endian, 1, [(0b1111, 4), (0b0000, 4)])
    data = bytes(data)

    manual = _manual_cross_byte(data, endian)
    compiled = trace_path(compiled_cs.bitset, data, endian)
    interpreted = trace_path(interpreted_cs.bitset, data, endian)

    assert_first_divergence("bits", compiled, interpreted, manual, data)

    # The two fields of one bit buffer must report the same source byte range.
    wide_span = compiled.spans[("wide_hi",)]
    assert compiled.spans[("wide_lo",)] == wide_span
    assert interpreted.spans[("wide_hi",)] == interpreted.spans[("wide_lo",)] == wide_span


DYNAMIC_DEF = """
struct node {
    uint16  header;
    uint8   width;
    uint32  cells[width];
    char    label[width];
};

struct chain {
    uint8   count;
    node    links[count];
    uint16  sentinel;
};
"""


COMBO_ALIGNED_DEF = """
struct cell {
    uint16  v;
    uint32  w;
};

union box {
    uint64  q;
    cell    c[2];
};

struct combo {
    uint8   head;
    cell    grid[2];
    box     u;
    uint16  a : 4;
    uint16  b : 12;
    uint32  last;
};
"""


def _build_aligned_combo(endian: str) -> bytes:
    e = "=" if endian == "@" else endian
    buf = bytearray(48)
    struct.pack_into("B", buf, 0, 0xAA)
    # grid: two 8-byte cells at offsets 4 and 12
    struct.pack_into(e + "H", buf, 4, 0x1111)
    struct.pack_into(e + "I", buf, 8, 0x22222222)
    struct.pack_into(e + "H", buf, 12, 0x3333)
    struct.pack_into(e + "I", buf, 16, 0x44444444)
    # union box at offset 24, 16 bytes; q and the cell array read the same storage
    struct.pack_into(e + "Q", buf, 24, 0x0123456789ABCDEF)
    # bit buffer at offset 40
    buf[40:42] = _pack_bits(endian, 2, [(0b1010, 4), (0b011001100110, 12)])
    struct.pack_into(e + "I", buf, 44, 0x55667788)
    return bytes(buf)


def _manual_aligned_combo(data: bytes, endian: str) -> ManualTrace:
    """Fully hand-derived layout: every offset follows the C alignment rules of the definition."""
    oracle = ManualOracle(data, endian)
    trace = ManualTrace()

    def add(path: tuple[str, ...], start: int, size: int, signed: bool = False) -> None:
        value = oracle.int(start, size) if signed else oracle.uint(start, size)
        trace.fields.append(ManualField(path, value, start, start + size))

    add(("head",), 0, 1)
    add(("grid", "0", "v"), 4, 2)
    add(("grid", "0", "w"), 8, 4)
    add(("grid", "1", "v"), 12, 2)
    add(("grid", "1", "w"), 16, 4)

    add(("u", "q"), 24, 8)
    add(("u", "c", "0", "v"), 24, 2)
    add(("u", "c", "0", "w"), 28, 4)
    add(("u", "c", "1", "v"), 32, 2)
    add(("u", "c", "1", "w"), 36, 4)

    bit = oracle.record(
        [("a", "bits", 2, 4), ("b", "bits", 0, 12)], start=40
    )
    trace.fields.extend(bit.fields)
    add(("last",), 44, 4)

    trace.next_offset = 48
    trace.size = 48

    # Padding follows the structural (cell array) union member; the overlapping scalar ``q`` is
    # excluded so its span doesn't hide the padding internal to the cells.
    structural_paths = {
        ("head",),
        ("grid", "0", "v"), ("grid", "0", "w"), ("grid", "1", "v"), ("grid", "1", "w"),
        ("u", "c", "0", "v"), ("u", "c", "0", "w"),
        ("u", "c", "1", "v"), ("u", "c", "1", "w"),
        ("a",), ("b",), ("last",),
    }
    covered = {
        pos
        for field in trace.fields
        if field.path in structural_paths
        for pos in range(field.start, field.end)
    }
    trace.padding = [(index, index + 1, data[index]) for index in range(48) if index not in covered]
    return trace


@pytest.mark.parametrize("endian", ENDIANS)
def test_aligned_combo_interaction(endian: str) -> None:
    """Alignment, nested arrays, a union and cross-byte bits all interact in one aligned record."""
    compiled_cs, interpreted_cs = make_pair(COMBO_ALIGNED_DEF, align=True)
    assert compiled_cs.combo.__compiled__ is True
    assert interpreted_cs.combo.__compiled__ is False

    data = _build_aligned_combo(endian)
    manual = _manual_aligned_combo(data, endian)
    compiled = trace_path(compiled_cs.combo, data, endian)
    interpreted = trace_path(interpreted_cs.combo, data, endian)

    # Positions 26-27 are padding internal to the first cell but covered by the scalar ``q``
    # span (24-32); both writers emit zeroes there.  The second cell's padding stays uncovered.
    forced = frozenset({26, 27})
    assert_first_divergence("aligned-combo", compiled, interpreted, manual, data, forced_padding=forced)

    # The alignment gaps must be real, otherwise the interaction would not actually be exercised.
    assert manual.padding, "expected structure-relative alignment padding in the combined sample"
    assert compiled.next_offset == interpreted.next_offset == 48


def _manual_dynamic(data: bytes, endian: str) -> ManualTrace:
    oracle = ManualOracle(data, endian)
    trace = ManualTrace()
    offset = 0

    count = oracle.uint(offset, 1)
    trace.fields.append(ManualField(("count",), count, offset, offset + 1))
    offset += 1

    for index in range(count):
        base = ("links", str(index))
        header = oracle.uint(offset, 2)
        trace.fields.append(ManualField((*base, "header"), header, offset, offset + 2))
        offset += 2

        width = oracle.uint(offset, 1)
        trace.fields.append(ManualField((*base, "width"), width, offset, offset + 1))
        offset += 1

        cells_start = offset
        for cell_index in range(width):
            value = oracle.uint(offset, 4)
            trace.fields.append(
                ManualField((*base, "cells", str(cell_index)), value, offset, offset + 4)
            )
            offset += 4
        assert offset - cells_start == width * 4

        label = data[offset : offset + width]
        trace.fields.append(ManualField((*base, "label"), label, offset, offset + width))
        offset += width

    sentinel = oracle.uint(offset, 2)
    trace.fields.append(ManualField(("sentinel",), sentinel, offset, offset + 2))
    offset += 2

    trace.next_offset = offset
    trace.size = offset
    return trace


@pytest.mark.parametrize("endian", ENDIANS)
def test_dynamic_nested_arrays(endian: str) -> None:
    """Dynamic arrays inside nested structures determine the next record's start on both paths."""
    compiled_cs, interpreted_cs = make_pair(DYNAMIC_DEF)
    e = "=" if endian == "@" else endian

    data = bytearray()
    data += struct.pack("B", 2)
    for width in (1, 3):
        data += struct.pack(e + "H", 0x4000 + width)
        data += struct.pack("B", width)
        data += struct.pack(e + f"{width}I", *[0x100 + i for i in range(width)])
        data += bytes(ord("a") + i for i in range(width))
    data += struct.pack(e + "H", 0xEDED)
    data = bytes(data)

    manual = _manual_dynamic(data, endian)
    compiled = trace_path(compiled_cs.chain, data, endian)
    interpreted = trace_path(interpreted_cs.chain, data, endian)

    assert_first_divergence("dynamic", compiled, interpreted, manual, data)
    assert compiled.next_offset == interpreted.next_offset == len(data)


def test_shared_reader_bug_is_not_endorsed() -> None:
    """Two readers that make the same mistake must still be rejected by the manual byte oracle.

    The compiled and interpreted trace are made identical but wrong for one field, simulating a
    layout bug copied into both paths.  Their agreement must not validate the error: the oracle
    derives the value from the raw bytes and reports the first diverging field.
    """
    compiled_cs, interpreted_cs = make_pair(CROSS_BYTE_DEF)
    data = bytearray()
    data += _pack_bits("<", 1, [(0b101, 3), (0b11101, 5)])
    data += _pack_bits("<", 2, [(0b1100110, 7), (0b011001100, 9)])
    data += _pack_bits("<", 1, [(0b1010, 4)])
    data += struct.pack("<I", 0xDEADBEEF)
    data += _pack_bits("<", 1, [(0b1111, 4), (0b0000, 4)])
    data = bytes(data)

    manual = _manual_cross_byte(data, "<")
    good = trace_path(compiled_cs.bitset, data, "<")

    # Both "readers" return the same corrupted value for ``low`` and the same wrong span.
    buggy_values = dict(good.values)
    buggy_spans = dict(good.spans)
    buggy_values[("low",)] = manual.value_map()[("low",)] ^ 0b111
    buggy_spans[("low",)] = (1, 2)
    buggy = PathTrace(buggy_values, buggy_spans, good.padding, good.next_offset, good.dumped)
    other_buggy = PathTrace(dict(buggy_values), dict(buggy_spans), good.padding, good.next_offset, good.dumped)

    # The two faulty paths agree with each other, so a reader-to-reader check would pass.
    assert buggy.values == other_buggy.values
    # The independent oracle catches the shared mistake at the first field.
    with pytest.raises(pytest.fail.Exception, match=r"first value divergence at 'low'"):
        assert_first_divergence("shared-bug", buggy, other_buggy, manual, data)

    # Sanity: the real readers do pass the same oracle.
    interpreted = trace_path(interpreted_cs.bitset, data, "<")
    assert_first_divergence("real", good, interpreted, manual, data)


def test_short_buffer_first_divergence() -> None:
    """Truncated input fails at the same first field on both paths rather than diverging silently.

    The two readers issue differently sized block reads, so the byte count in the error message
    need not match.  Instead both must raise ``EOFError`` and the failure must correspond to the
    first field whose manually derived source span no longer fits in the truncated buffer.
    """
    compiled_cs, interpreted_cs = make_pair(COMPOSITE_DEF)
    data = _build_composite_sample("<")
    full_manual = _manual_composite(data, "<")
    ordered_spans = sorted(full_manual.spans().items(), key=lambda item: item[1][0])

    for size in range(1, len(data) - 1):
        chunk = data[:size]
        first_missing = next(
            (".".join(path) for path, (start, end) in ordered_spans if end > size),
            None,
        )

        outcomes: list[str] = []
        for cs in (compiled_cs, interpreted_cs):
            try:
                cs.record(chunk)
            except EOFError:
                outcomes.append("EOF")
            except Exception as exc:  # noqa: BLE001
                outcomes.append(type(exc).__name__)
            else:
                outcomes.append("ok")

        assert outcomes == ["EOF", "EOF"], (
            f"size {size}: expected both paths to raise EOFError (first missing field {first_missing}), "
            f"got {outcomes}"
        )


def test_compiled_is_actually_compiled() -> None:
    """Guard against the compiler silently reverting both instances to the interpreted path."""
    compiled_cs, interpreted_cs = make_pair(COMPOSITE_DEF)
    assert compiled_cs.record.__compiled__ is True
    assert interpreted_cs.record.__compiled__ is False
    assert hasattr(compiled_cs.record._read, "__source__")
    assert not hasattr(interpreted_cs.record._read, "__source__")
