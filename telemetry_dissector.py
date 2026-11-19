"""Dissect telemetry packets into field values with exact source bit ranges.

A telemetry packet mixes cross-byte bit fields with integer fields that are each
interpreted in their own byte order. Given a structure declaration and the
packet bytes, this module exports every field's value together with the exact
bits it came from.

Declaration format (a dict, or JSON equivalent)::

    {
        "name": "telemetry",
        "endian": "big",  # default byte order for all fields
        "fields": [
            {"name": "sync", "type": "uint16"},  # big endian
            {"name": "sequence", "type": "uint16", "endian": "little"},
            {"name": "sensor_id", "type": "uint16", "bits": 12},  # cross-byte bit field
            {"name": "state", "type": "uint16", "bits": 4},  # same storage unit
            {"name": "temperature", "type": "int16", "endian": "little"},
            {"name": "payload", "type": "bytes", "length": 2},
        ],
    }

Type names are resolved through :class:`dissect.cstruct.cstruct`, so any
fixed-size integer type it knows (``uint8`` .. ``uint128``, ``int24``,
``uint48``, ...) can be used. ``bytes`` reads a raw byte string of ``length``
bytes. A field with ``bits`` is a bit field: consecutive bit fields of the
same type and byte order share one storage unit, exactly like C bit fields.

Exported bit positions use absolute packet coordinates: bit 0 is the most
significant bit of byte 0, bit 8 the most significant bit of byte 1, and so
on. Every field reports ``bit_segments`` as a list of ``[start, length]``
pairs in these coordinates, so cross-byte little-endian bit fields (whose
source bits are not contiguous) are still described exactly. ``bit_offset``
is the start bit when the range is contiguous, else ``None``.

Byte order is applied per field: each integer is interpreted in its own
declared endianness. Reversing the whole packet is *not* the same as swapping
a field's byte order, and this module never does the former to get the latter.

Two incomplete-data states are reported explicitly:

- Truncated packet (fewer bits than a field or bit group needs): the field is
  exported with ``status`` ``"insufficient-bits"``, a ``None`` value and the
  ``expected_bits``/``available_bits`` counts. ``complete`` is ``False`` and
  ``next_bit_offset`` is the bit position where the packet ran out.
- A bit group whose declared fields don't fill the storage unit: the leftover
  bits are exported as a ``padding`` record with ``status`` ``"unused"``.

Reading is chunk-size agnostic: feeding the same packet through a stream that
only returns a few bytes per ``read()`` call (see :class:`FragmentedReader`)
yields the same values and the same ``next_bit_offset`` as reading it in one
go.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

from dissect.cstruct import cstruct
from dissect.cstruct.types import Int, Packed

if TYPE_CHECKING:
    from collections.abc import Sequence

_CS = cstruct()

_ENDIAN_ALIASES = {
    "little": "little",
    "<": "little",
    "le": "little",
    "big": "big",
    ">": "big",
    "be": "big",
    "network": "big",
    "!": "big",
}

_SIGNED_FORMATS = frozenset("bhilq")


class DeclarationError(ValueError):
    """The structure declaration is invalid."""


class InsufficientBitsError(EOFError):
    """The packet ended before a field or bit group could be read.

    Attributes:
        position: Absolute bit position where the packet ran out.
        expected_bits: Bits the field or storage unit needed.
        available_bits: Bits that were actually left in the packet.
    """

    def __init__(self, *, position: int, expected_bits: int, available_bits: int) -> None:
        super().__init__(
            f"packet ended at bit {position}: expected {expected_bits} bits, only {available_bits} available"
        )
        self.position = position
        self.expected_bits = expected_bits
        self.available_bits = available_bits


class FragmentedReader(io.RawIOBase):
    """Read wrapper that returns at most ``chunk_size`` bytes per ``read()`` call.

    Simulates fragmented input, e.g. a packet arriving in segments. Parsing a
    packet through this wrapper yields the same values and bit positions as
    parsing the whole packet at once.
    """

    def __init__(self, stream: BinaryIO, chunk_size: int = 1) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        self.stream = stream
        self.chunk_size = chunk_size

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.chunk_size
        return self.stream.read(min(size, self.chunk_size))


class BitReader:
    """Byte-oriented reader over a stream that tracks the absolute bit position.

    Short reads (a stream returning fewer bytes than requested) are collected
    in a loop, so fragmented input behaves exactly like a contiguous buffer.
    """

    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.bytes_read = 0

    @property
    def bit_position(self) -> int:
        """Absolute bit position of the next unread bit."""
        return self.bytes_read * 8

    def read(self, size: int) -> bytes:
        """Read exactly ``size`` bytes, raising :class:`InsufficientBitsError` at end of packet."""
        chunks = []
        needed = size
        while needed > 0:
            chunk = self.stream.read(needed)
            if not chunk:
                self.bytes_read += size - needed
                raise InsufficientBitsError(
                    position=self.bit_position,
                    expected_bits=size * 8,
                    available_bits=(size - needed) * 8,
                )
            chunks.append(chunk)
            needed -= len(chunk)

        self.bytes_read += size
        return b"".join(chunks)


@dataclass(frozen=True)
class FieldSpec:
    """One field of a structure declaration."""

    name: str
    type_name: str
    endian: str
    size: int
    signed: bool = False
    bits: int | None = None
    is_bytes: bool = False


@dataclass(frozen=True)
class Declaration:
    """A parsed structure declaration."""

    name: str
    fields: tuple[FieldSpec, ...]


def _normalize_endian(endian: Any) -> str:
    try:
        return _ENDIAN_ALIASES[str(endian).lower()]
    except KeyError:
        raise DeclarationError(f"invalid endianness: {endian!r} (expected 'little' or 'big')") from None


def _resolve_integer_type(type_name: str) -> tuple[int, bool]:
    """Resolve a cstruct type name to its ``(size, signed)``, rejecting non-integer types."""
    try:
        type_ = _CS.resolve(type_name)
    except Exception:
        raise DeclarationError(f"unknown type: {type_name!r}") from None

    if type_.__size__ is None:
        raise DeclarationError(f"type {type_name!r} is dynamically sized")

    if issubclass(type_, Int):
        return type_.__size__, bool(type_.signed)

    if issubclass(type_, Packed) and type_.__fmt__.lower() in _SIGNED_FORMATS:
        return type_.__size__, type_.__fmt__ in _SIGNED_FORMATS

    raise DeclarationError(f"type {type_name!r} is not a fixed-size integer type")


def parse_declaration(declaration: dict[str, Any]) -> Declaration:
    """Validate a structure declaration and return it as a :class:`Declaration`.

    Raises:
        DeclarationError: If the declaration is invalid, e.g. an unknown type
            or a bit field that straddles a storage unit.
    """
    if not isinstance(declaration, dict):
        raise DeclarationError(f"declaration must be a dict, got {type(declaration).__name__}")

    name = declaration.get("name", "packet")
    default_endian = _normalize_endian(declaration.get("endian", "big"))

    entries = declaration.get("fields")
    if not isinstance(entries, list) or not entries:
        raise DeclarationError("declaration must contain a non-empty 'fields' list")

    fields = []
    group_key = None
    group_remaining = 0
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("name") or not entry.get("type"):
            raise DeclarationError(f"each field needs a 'name' and a 'type', got: {entry!r}")

        field_name = entry["name"]
        type_name = entry["type"]
        endian = _normalize_endian(entry.get("endian", default_endian))
        bits = entry.get("bits")

        if type_name == "bytes":
            if bits is not None:
                raise DeclarationError(f"bit field {field_name!r}: type 'bytes' cannot have 'bits'")
            length = entry.get("length")
            if not isinstance(length, int) or isinstance(length, bool) or length < 1:
                raise DeclarationError(f"field {field_name!r}: 'bytes' needs an integer 'length' of at least 1")
            spec = FieldSpec(field_name, type_name, endian, length, is_bytes=True)
        else:
            size, signed = _resolve_integer_type(type_name)
            if bits is not None and (not isinstance(bits, int) or isinstance(bits, bool) or not 0 < bits <= size * 8):
                raise DeclarationError(f"bit field {field_name!r}: 'bits' must be an integer between 1 and {size * 8}")
            spec = FieldSpec(field_name, type_name, endian, size, signed, bits)

        # Bit fields of the same type and byte order share one storage unit; validate it isn't straddled
        if spec.bits is None:
            group_key = None
        else:
            key = (spec.endian, spec.size)
            if key != group_key:
                group_key = key
                group_remaining = spec.size * 8
            if spec.bits > group_remaining:
                raise DeclarationError(
                    f"bit field {field_name!r} straddles a storage unit "
                    f"({spec.bits} bits requested, {group_remaining} left in the group)"
                )
            group_remaining -= spec.bits

        fields.append(spec)

    return Declaration(name=name, fields=tuple(fields))


def load_declaration(source: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Load a structure declaration from a JSON file path, a JSON string, or pass a dict through."""
    if isinstance(source, dict):
        return source

    text = None
    if isinstance(source, Path) or (isinstance(source, str) and not source.lstrip().startswith(("{", "["))):
        try:
            text = Path(source).read_text(encoding="utf-8")
        except OSError:
            if not isinstance(source, str):
                raise
    if text is None:
        text = source if isinstance(source, str) else source.read_text(encoding="utf-8")

    declaration = json.loads(text)
    if not isinstance(declaration, dict):
        raise DeclarationError("declaration JSON must be an object")
    return declaration


def _coalesce(positions: list[int]) -> list[list[int]]:
    """Coalesce sorted absolute bit positions into ``[start, length]`` segments."""
    segments = []
    for position in positions:
        if segments and position == segments[-1][0] + segments[-1][1]:
            segments[-1][1] += 1
        else:
            segments.append([position, 1])
    return segments


class _BitGroup:
    """State of one bit-field storage unit while dissecting.

    Mirrors the semantics of :class:`dissect.cstruct.bitbuffer.BitBuffer`: a
    little-endian unit hands out bits starting from its least significant bit,
    a big-endian unit from its most significant bit.
    """

    def __init__(self, *, endian: str, size: int, unit: int, byte_offset: int) -> None:
        self.endian = endian
        self.size = size
        self.unit = unit
        self.byte_offset = byte_offset
        self.consumed = 0

    @property
    def remaining(self) -> int:
        """Bits not yet consumed by a declared field."""
        return self.size * 8 - self.consumed

    def matches(self, spec: FieldSpec) -> bool:
        return self.endian == spec.endian and self.size == spec.size

    def _absolute_bit(self, unit_bit: int) -> int:
        """Map a unit bit index (LSB numbering) to an absolute packet bit position."""
        byte, bit = divmod(unit_bit, 8)
        if self.endian == "big":
            byte = self.size - 1 - byte
        return (self.byte_offset + byte) * 8 + (7 - bit)

    def segments(self, start: int, length: int) -> list[list[int]]:
        """Absolute ``[start, length]`` bit segments of the unit bit range ``[start, start + length)``."""
        return _coalesce(sorted(self._absolute_bit(i) for i in range(start, start + length)))

    def take(self, spec: FieldSpec) -> tuple[int, int]:
        """Consume ``spec.bits`` bits from the unit, returning ``(value, unit bit start)``."""
        start = self.consumed if self.endian == "little" else self.size * 8 - self.consumed - spec.bits
        value = (self.unit >> start) & ((1 << spec.bits) - 1)
        self.consumed += spec.bits
        return value, start

    def padding_record(self) -> dict[str, Any]:
        """Export the bits no declared field consumed, making the partial group state explicit."""
        start = self.consumed if self.endian == "little" else 0
        segments = self.segments(start, self.remaining)
        return {
            "name": None,
            "kind": "padding",
            "type": None,
            "value": None,
            "endian": self.endian,
            "byte_offset": self.byte_offset,
            "byte_length": self.size,
            "bits": self.remaining,
            "bit_offset": segments[0][0] if len(segments) == 1 else None,
            "bit_length": self.remaining,
            "bit_segments": segments,
            "status": "unused",
        }


def _field_record(
    spec: FieldSpec, value: Any, segments: list[list[int]], byte_offset: int, byte_length: int
) -> dict[str, Any]:
    record = {
        "name": spec.name,
        "kind": "field",
        "type": spec.type_name,
        "value": value,
        "endian": None if spec.is_bytes else spec.endian,
        "byte_offset": byte_offset,
        "byte_length": byte_length,
        "bit_offset": segments[0][0] if len(segments) == 1 else None,
        "bit_length": sum(length for _, length in segments),
        "bit_segments": segments,
        "status": "ok",
    }
    if spec.bits is not None:
        record["bits"] = spec.bits
    return record


def _insufficient_record(spec: FieldSpec, error: InsufficientBitsError, byte_offset: int) -> dict[str, Any]:
    record = {
        "name": spec.name,
        "kind": "field",
        "type": spec.type_name,
        "value": None,
        "endian": None if spec.is_bytes else spec.endian,
        "byte_offset": byte_offset,
        "byte_length": spec.size,
        "bit_offset": None,
        "bit_length": None,
        "bit_segments": [],
        "status": "insufficient-bits",
        "expected_bits": error.expected_bits,
        "available_bits": error.available_bits,
    }
    if spec.bits is not None:
        record["bits"] = spec.bits
    return record


def dissect_packet(
    declaration: dict[str, Any] | Declaration, packet: bytes | bytearray | memoryview | BinaryIO
) -> dict[str, Any]:
    """Dissect a packet according to a structure declaration.

    Args:
        declaration: Structure declaration, as a dict or a parsed :class:`Declaration`.
        packet: The packet bytes, or a binary stream to read them from. Streams
            may return short reads; the result is identical to reading the whole
            packet at once.

    Returns:
        A JSON-serializable dict with the field records, a ``complete`` flag,
        and ``next_bit_offset``: the absolute bit position where the next
        field would start.
    """
    decl = parse_declaration(declaration) if isinstance(declaration, dict) else declaration
    stream = io.BytesIO(bytes(packet)) if isinstance(packet, (bytes, bytearray, memoryview)) else packet
    reader = BitReader(stream)

    records: list[dict[str, Any]] = []
    group: _BitGroup | None = None
    complete = True

    def close_group() -> None:
        nonlocal group
        if group is not None and group.remaining:
            records.append(group.padding_record())
        group = None

    for spec in decl.fields:
        if spec.bits is not None:
            if group is None or not group.matches(spec):
                close_group()
                byte_offset = reader.bit_position // 8
                try:
                    unit_bytes = reader.read(spec.size)
                except InsufficientBitsError as e:
                    records.append(_insufficient_record(spec, e, byte_offset))
                    complete = False
                    break
                group = _BitGroup(
                    endian=spec.endian,
                    size=spec.size,
                    unit=int.from_bytes(unit_bytes, spec.endian),
                    byte_offset=byte_offset,
                )
            value, start = group.take(spec)
            records.append(_field_record(spec, value, group.segments(start, spec.bits), group.byte_offset, group.size))
        else:
            close_group()
            byte_offset = reader.bit_position // 8
            try:
                data = reader.read(spec.size)
            except InsufficientBitsError as e:
                records.append(_insufficient_record(spec, e, byte_offset))
                complete = False
                break
            value = data.hex() if spec.is_bytes else int.from_bytes(data, spec.endian, signed=spec.signed)
            records.append(_field_record(spec, value, [[byte_offset * 8, spec.size * 8]], byte_offset, spec.size))

    if complete:
        close_group()

    return {
        "name": decl.name,
        "complete": complete,
        "next_bit_offset": reader.bit_position,
        "fields": records,
    }


DEMO_DECLARATION: dict[str, Any] = {
    "name": "telemetry",
    "endian": "big",
    "fields": [
        {"name": "sync", "type": "uint16"},
        {"name": "sequence", "type": "uint16", "endian": "little"},
        {"name": "sensor_id", "type": "uint16", "bits": 12},
        {"name": "state", "type": "uint16", "bits": 4},
        {"name": "temperature", "type": "int16", "endian": "little"},
        {"name": "flags", "type": "uint8", "bits": 3, "endian": "little"},
        {"name": "payload", "type": "bytes", "length": 2},
    ],
}

DEMO_PACKET = bytes.fromhex("1acf1234abcd2efb05aa55")


def _read_packet(argument: str) -> bytes:
    if argument.startswith("@"):
        return Path(argument[1:]).read_bytes()
    return bytes.fromhex(argument)


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point: dissect a packet and print the export as JSON."""
    parser = argparse.ArgumentParser(
        prog="telemetry_dissector",
        description="Dissect a telemetry packet into field values with exact source bit ranges.",
    )
    parser.add_argument("declaration", nargs="?", help="JSON structure declaration file")
    parser.add_argument("packet", nargs="?", help="packet bytes as a hex string, or @file with raw bytes")
    parser.add_argument(
        "--chunk-size",
        type=int,
        metavar="N",
        help="feed the packet through a stream in fragments of at most N bytes",
    )
    parser.add_argument("--demo", action="store_true", help="dissect the built-in demo telemetry packet")
    args = parser.parse_args(argv)

    if args.demo:
        declaration, packet = DEMO_DECLARATION, DEMO_PACKET
    else:
        if not args.declaration or not args.packet:
            parser.error("declaration and packet are required unless --demo is given")
        declaration = load_declaration(args.declaration)
        packet = _read_packet(args.packet)

    source: bytes | BinaryIO = packet
    if args.chunk_size:
        source = FragmentedReader(io.BytesIO(packet), args.chunk_size)

    result = dissect_packet(declaration, source)
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
