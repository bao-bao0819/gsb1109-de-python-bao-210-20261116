from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, BinaryIO

from dissect.cstruct.exception import LEB128NormalizationError, SignedRangeError
from dissect.cstruct.types.base import BaseType

if TYPE_CHECKING:
    from collections.abc import Iterable

    from typing_extensions import Self

    from dissect.cstruct.cstruct import Endianness


def decode_leb128(data: bytes | bytearray | memoryview, *, signed: bool = False) -> int:
    """Decode a LEB128 encoded byte string into an integer.

    The whole byte string is consumed. An incomplete final byte is not allowed: it does not contain a value and
    should be rejected by the caller before calling this function.

    Args:
        data: The LEB128 encoded bytes.
        signed: Whether the value uses signed LEB128 semantics.

    Raises:
        ValueError: If the byte string ends before a final byte (a byte without the continuation bit) was read.
    """
    result = 0
    shift = 0
    final = 0

    for final in data:
        result |= (final & 0x7F) << shift
        shift += 7
        if (final & 0x80) == 0:
            break
    else:
        raise ValueError("Incomplete LEB128 encoding")

    if signed and final & 0x40 != 0:
        result |= ~0 << shift

    return result


def encode_leb128(value: int, *, signed: bool = False) -> bytes:
    """Encode an integer into its minimal (canonical) LEB128 representation.

    Args:
        value: The integer to encode.
        signed: Whether to use signed LEB128 semantics.

    Raises:
        ValueError: If ``signed`` is ``False`` and ``value`` is negative.
    """
    if value < 0 and not signed:
        raise ValueError("Attempt to encode a negative integer using unsigned LEB128 encoding")

    result = bytearray()
    data = int(value)

    while True:
        byte = data & 0x7F
        data >>= 7

        # The encoding stops as soon as the remaining bits are all sign bits and the sign of the current byte
        # matches. For unsigned values that simply means there are no remaining bits.
        if (signed and ((data == 0 and byte & 0x40 == 0) or (data == -1 and byte & 0x40 != 0))) or (
            not signed and data == 0
        ):
            result.append(byte)
            break

        result.append(0x80 | byte)

    return bytes(result)


class LEB128(int, BaseType):
    """Variable-length code compression to store an arbitrarily large integer in a small number of bytes.

    See https://en.wikipedia.org/wiki/LEB128 for more information and an explanation of the algorithm.

    ``LEB128`` values are integers, but values read from a stream also remember the exact bytes they were read from.
    This is important for telemetry archives where the integer value is used for aggregation, while the original
    encoding is covered by a signature. Reading never silently canonicalizes: the original encoding is kept in
    :attr:`raw`, and :attr:`canonical` describes whether that encoding is the shortest possible encoding for the
    value under the type's signedness.

    Values constructed from an integer (instead of read from a stream) have ``raw`` set to ``None`` and are written
    using the canonical encoding. Use :meth:`normalize` to explicitly create a canonicalized copy of a parsed value.
    """

    signed: bool

    raw: bytes | None
    """The exact bytes this value was read from, or ``None`` for values constructed from an integer."""

    def __new__(cls, value: int = 0, *, raw: bytes | None = None) -> Self:
        obj = int.__new__(cls, value)
        if raw is not None:
            raw = bytes(raw)
            if decode_leb128(raw, signed=getattr(cls, "signed", False)) != int(value):
                raise ValueError("Raw LEB128 encoding does not decode to the given value")
        obj.raw = raw
        return obj

    @property
    def value(self) -> int:
        """The decoded integer value, for value based aggregation."""
        return int(self)

    @property
    def signedness(self) -> str:
        """``"signed"`` or ``"unsigned"``, the interpretation this value was read with."""
        return "signed" if self.signed else "unsigned"

    @property
    def is_signed(self) -> bool:
        """Whether this value uses signed LEB128 semantics."""
        return self.signed

    @property
    def aggregation_key(self) -> int:
        """Key to aggregate records by their integer value, regardless of encoding."""
        return int(self)

    @property
    def encoding(self) -> bytes:
        """The original encoding if this value was read, otherwise the canonical encoding."""
        return self.raw if self.raw is not None else self.canonical_bytes

    @property
    def original(self) -> bytes:
        """Alias for :attr:`encoding`."""
        return self.encoding

    @property
    def original_encoding(self) -> bytes:
        """Alias for :attr:`encoding`."""
        return self.encoding

    @property
    def encoded_length(self) -> int:
        """The length of the original encoding in bytes."""
        return len(self.encoding)

    @property
    def canonical_bytes(self) -> bytes:
        """The minimal encoding of the value, using this type's signedness."""
        return encode_leb128(int(self), signed=self.signed)

    @property
    def canonical_encoding(self) -> bytes:
        """Alias for :attr:`canonical_bytes`."""
        return self.canonical_bytes

    @property
    def canonical_length(self) -> int:
        """The length of the canonical encoding in bytes."""
        return len(self.canonical_bytes)

    @property
    def is_canonical(self) -> bool:
        """Whether the original encoding is already the minimal, sign-correct encoding."""
        return self.raw is None or self.raw == self.canonical_bytes

    @property
    def canonical(self) -> bool:
        """Alias for :attr:`is_canonical`."""
        return self.is_canonical

    @property
    def is_overlong(self) -> bool:
        """Whether the original encoding uses more bytes than the canonical encoding."""
        return self.raw is not None and len(self.raw) > self.canonical_length

    @property
    def overlong(self) -> bool:
        """Alias for :attr:`is_overlong`."""
        return self.is_overlong

    @property
    def redundant_bytes(self) -> int:
        """The number of bytes that canonicalization would remove."""
        return max(0, self.encoded_length - self.canonical_length)

    @property
    def canonicality(self) -> str:
        """A human readable classification of the original encoding: ``"canonical"`` or ``"overlong"``.

        For a given value and signedness the minimal encoding is unique, so an encoding that is not canonical always
        uses more bytes than necessary. Signedness is a separate property: the same bytes can be a valid signed or
        unsigned encoding and mean different values.
        """
        return "canonical" if self.is_canonical else "overlong"

    @property
    def evidence_key(self) -> tuple[str, bool, int, bytes]:
        """Identity of this value as a piece of collected evidence.

        Numeric equality is not enough to identify evidence: two values may aggregate to the same integer while
        having different encodings (and therefore different signatures). The key includes the signedness and the
        exact original encoding. Values constructed from an integer use their canonical encoding, since that is the
        encoding that would be written.
        """
        return ("leb128", self.signed, int(self), self.encoding)

    @property
    def evidence_digest(self) -> str:
        """A stable digest of :attr:`evidence_key`."""
        payload = f"leb128-evidence-v1:{int(self.signed)}:{int(self)}:".encode() + self.encoding
        return hashlib.sha256(payload).hexdigest()

    def is_same_evidence(self, other: object) -> bool:
        """Return whether ``other`` is the exact same evidence (value, signedness and encoding)."""
        if not isinstance(other, LEB128):
            return False
        return self.evidence_key == other.evidence_key

    def normalize(self) -> Self:
        """Return a canonicalized copy of this value.

        The returned value has the same integer value and signedness, but its :attr:`raw` is set to the canonical
        encoding. The original value is left untouched, so callers can compare the evidence before and after.
        """
        return type(self).__new__(type(self), int(self), raw=self.canonical_bytes)

    def canonicalize(self) -> Self:
        """Alias for :meth:`normalize`."""
        return self.normalize()

    @classmethod
    def _read(cls, stream: BinaryIO, *, context: dict[str, Any] | None = None, endian: Endianness) -> Self:
        raw = bytearray()

        while True:
            b = stream.read(1)
            if b == b"":
                raise EOFError("EOF reached, while final LEB128 byte was not yet read")

            raw.append(b[0])
            if (b[0] & 0x80) == 0:
                break

        result = decode_leb128(raw, signed=cls.signed)
        return cls.__new__(cls, result, raw=bytes(raw))

    @classmethod
    def _read_0(cls, stream: BinaryIO, *, context: dict[str, Any] | None = None, endian: Endianness) -> list[Self]:
        result = []

        while True:
            if (value := cls._read(stream, context=context, endian=endian)) == 0:
                break

            result.append(value)

        return result

    @classmethod
    def _write(cls, stream: BinaryIO, data: int, *, endian: Endianness) -> int:
        # An integer read from a stream keeps its original encoding. This makes writing a parsed record byte-for-byte
        # stable, which is required to keep existing signatures valid.
        if (
            isinstance(data, LEB128)
            and data.signed == cls.signed
            and data.raw is not None
            and int(data) == decode_leb128(data.raw, signed=cls.signed)
        ):
            result = data.raw
        else:
            result = encode_leb128(int(data), signed=cls.signed)

        stream.write(result)
        return len(result)


@dataclass(frozen=True)
class SignedRange:
    """A byte range that is covered by an existing signature.

    Ranges are half-open: ``[start, end)`` and are relative to the start of a serialized record.
    """

    start: int
    end: int
    name: str | None = None

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"Invalid signed range: {self.start}:{self.end}")

    def overlaps(self, start: int, end: int) -> bool:
        """Return whether ``[start, end)`` overlaps this signed range."""
        return start < self.end and end > self.start


@dataclass(frozen=True)
class LengthUpdate:
    """A length field that has to change when a LEB128 field is normalized."""

    field: str
    original: int
    updated: int
    delta: int
    offset: int
    new_offset: int
    original_encoding: bytes
    updated_encoding: bytes
    region_start: int
    region_end: int | None
    automatic: bool = True

    @property
    def changed(self) -> bool:
        """Whether the value or its encoding changes."""
        return self.original != self.updated or self.original_encoding != self.updated_encoding


@dataclass(frozen=True)
class FieldShift:
    """A field that moves because an earlier field changed size."""

    field: str
    original_offset: int
    new_offset: int

    @property
    def delta(self) -> int:
        """The number of bytes this field moved."""
        return self.new_offset - self.original_offset


@dataclass
class LEB128NormalizationPlan:
    """A dry-run of canonicalizing a LEB128 field in a parsed structure.

    Creating a plan never modifies the structure. :meth:`apply` performs the write-back, but refuses to do so when
    any byte inside a supplied signed range would change: replacing signed evidence requires a new signature, it is
    not a formatting operation.

    The plan includes any length fields that depend on the encoded size of the normalized field, so the preview and
    the applied result contain the same bytes.
    """

    structure: Any
    field: str
    value: int
    signed: bool
    offset: int
    original: bytes
    normalized: bytes
    original_bytes: bytes
    normalized_bytes: bytes
    length_updates: tuple[LengthUpdate, ...] = ()
    shifts: tuple[FieldShift, ...] = ()
    dependent_fields: tuple[str, ...] = ()
    signed_ranges: tuple[SignedRange, ...] = ()
    changed_signed_ranges: tuple[SignedRange, ...] = ()
    old_evidence_key: tuple[str, bool, int, bytes] = dataclasses.field(default_factory=tuple)
    new_evidence_key: tuple[str, bool, int, bytes] = dataclasses.field(default_factory=tuple)
    old_evidence_digest: str = ""
    new_evidence_digest: str = ""
    blocked_reason: str | None = None

    @property
    def delta(self) -> int:
        """The size difference between the normalized and original encoding."""
        return len(self.normalized) - len(self.original)

    @property
    def length_delta(self) -> int:
        """The size difference of the complete record."""
        return len(self.normalized_bytes) - len(self.original_bytes)

    @property
    def allowed(self) -> bool:
        """Whether the write-back is allowed without creating a new signature."""
        return not self.changed_signed_ranges and self.blocked_reason is None

    @property
    def requires_resign(self) -> bool:
        """Whether applying this plan changes an existing signed range."""
        return bool(self.changed_signed_ranges)

    @property
    def same_evidence(self) -> bool:
        """Whether the normalized record is byte-for-byte the same evidence as the original."""
        return self.original_bytes == self.normalized_bytes

    def dumps(self) -> bytes:
        """Return the bytes of the previewed record."""
        return self.normalized_bytes

    def apply(self, *, force: bool = False) -> Any:
        """Apply the previewed write-back to the structure.

        Args:
            force: Apply even when a signed range would change. The caller is responsible for re-signing the result;
                the normalized record is still distinct evidence from the original.

        Raises:
            SignedRangeError: If a signed range would change and ``force`` is not set.
            LEB128NormalizationError: If the structure cannot be written back to exactly the previewed bytes.
        """
        if self.changed_signed_ranges and not force:
            names = ", ".join(range_.name or f"{range_.start}:{range_.end}" for range_ in self.changed_signed_ranges)
            raise SignedRangeError(f"Normalizing {self.field!r} changes signed range(s): {names}")

        if self.same_evidence:
            return self.structure

        if (current := self.structure.dumps()) != self.original_bytes:
            raise LEB128NormalizationError("The structure changed after the preview; create a new normalization plan")

        target = getattr(self.structure, self.field)
        old_values = {self.field: target}
        for update in self.length_updates:
            if update.field != self.field:
                old_values[update.field] = getattr(self.structure, update.field)

        try:
            if isinstance(target, LEB128):
                setattr(self.structure, self.field, target.normalize())
            else:
                setattr(self.structure, self.field, type(target).__new__(type(target), self.value))

            for update in self.length_updates:
                if update.field == self.field or not update.changed:
                    continue
                current = getattr(self.structure, update.field)
                if isinstance(current, LEB128):
                    setattr(self.structure, update.field, type(current)(update.updated))
                else:
                    setattr(self.structure, update.field, update.updated)

            result = self.structure.dumps()
        except Exception:
            for name, value in old_values.items():
                setattr(self.structure, name, value)
            raise

        if result != self.normalized_bytes:
            for name, value in old_values.items():
                setattr(self.structure, name, value)
            raise LEB128NormalizationError(
                f"Write-back for {self.field!r} produced {len(result)} bytes, expected {len(self.normalized_bytes)}"
            )

        return self.structure

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the plan."""
        return {
            "field": self.field,
            "value": self.value,
            "signed": self.signed,
            "offset": self.offset,
            "original": self.original.hex(),
            "normalized": self.normalized.hex(),
            "delta": self.delta,
            "length_delta": self.length_delta,
            "allowed": self.allowed,
            "requires_resign": self.requires_resign,
            "same_evidence": self.same_evidence,
            "old_evidence_digest": self.old_evidence_digest,
            "new_evidence_digest": self.new_evidence_digest,
            "dependent_fields": list(self.dependent_fields),
            "length_updates": [
                {
                    "field": update.field,
                    "original": update.original,
                    "updated": update.updated,
                    "delta": update.delta,
                    "offset": update.offset,
                    "new_offset": update.new_offset,
                    "original_encoding": update.original_encoding.hex(),
                    "updated_encoding": update.updated_encoding.hex(),
                    "region_start": update.region_start,
                    "region_end": update.region_end,
                    "automatic": update.automatic,
                }
                for update in self.length_updates
            ],
            "shifts": [
                {
                    "field": shift.field,
                    "original_offset": shift.original_offset,
                    "new_offset": shift.new_offset,
                    "delta": shift.delta,
                }
                for shift in self.shifts
            ],
            "signed_ranges": [
                {"start": range_.start, "end": range_.end, "name": range_.name} for range_ in self.signed_ranges
            ],
            "changed_signed_ranges": [
                {"start": range_.start, "end": range_.end, "name": range_.name} for range_ in self.changed_signed_ranges
            ],
            "blocked_reason": self.blocked_reason,
        }


@dataclass(frozen=True)
class _LayoutField:
    """Internal representation of a field in a serialized structure."""

    name: str
    type: Any
    offset: int
    size: int
    value: Any
    field: Any = None

    @property
    def end(self) -> int:
        return self.offset + self.size


def _normalize_ranges(ranges: Iterable[SignedRange | tuple[int, int] | dict[str, Any]]) -> tuple[SignedRange, ...]:
    result = []
    for range_ in ranges:
        if isinstance(range_, SignedRange):
            result.append(range_)
        elif isinstance(range_, dict):
            result.append(SignedRange(range_["start"], range_["end"], range_.get("name")))
        else:
            if len(range_) == 2:
                start, end = range_
                name = None
            else:
                start, end, name = range_
            result.append(SignedRange(start, end, name))
    return tuple(result)


def _field_size(obj: Any, field: Any) -> int:
    """Return the serialized size of a field value, using parsed dynamic sizes where available."""
    type_ = field.type

    if issubclass(type_, LEB128):
        value = getattr(obj, field._name)
        if isinstance(value, LEB128):
            return value.encoded_length
        return len(encode_leb128(int(value), signed=type_.signed))

    if type_.__dynamic__:
        if (size := obj.__sizes__.get(field._name)) is not None:
            return size
        value = getattr(obj, field._name)
        if hasattr(value, "dumps"):
            return len(value.dumps())
        return len(type_.dumps(value))

    return len(type_)


def _layout(obj: Any) -> tuple[list[_LayoutField], int]:
    """Build the serialized field layout of a parsed structure."""
    from dissect.cstruct.types.structure import Union

    cls = type(obj)
    if issubclass(cls, Union):
        raise LEB128NormalizationError("Normalization of unions is not supported")
    if any(field.bits for field in cls.__fields__):
        raise LEB128NormalizationError("Normalization of structures with bit fields is not supported")

    fields = []
    offset = 0

    for struct_field in cls.__fields__:
        if struct_field.offset is not None and offset < struct_field.offset:
            offset = struct_field.offset

        if cls.__align__ and struct_field.offset is None:
            offset += -offset & (struct_field.alignment - 1)

        size = _field_size(obj, struct_field)
        fields.append(
            _LayoutField(
                struct_field._name, struct_field.type, offset, size, getattr(obj, struct_field._name), struct_field
            )
        )
        offset += size

    if cls.__align__:
        offset += -offset & (cls.__alignment__ - 1)

    return fields, offset


def _expression_references(type_: Any, target: str) -> bool:
    """Return whether any array dimension of ``type_`` references an identifier called ``target``."""
    from dissect.cstruct.expression import Expression
    from dissect.cstruct.lexer import IDENTIFIER_TYPES

    while hasattr(type_, "__num_entries__"):
        entries = type_.__num_entries__
        if isinstance(entries, Expression) and any(
            token.value == target for token in entries._tokens if token.type in IDENTIFIER_TYPES
        ):
            return True
        type_ = getattr(type_, "__type__", None)
        if type_ is None:
            break

    return False


def _build_preview(
    obj: Any,
    layout: list[_LayoutField],
    target_name: str,
    target_bytes: bytes,
    length_updates: dict[str, int],
) -> tuple[bytes, list[FieldShift], dict[str, _LayoutField]]:
    """Serialize the structure with the target and length values replaced, preserving all other bytes."""
    cls = type(obj)
    original = obj.dumps()
    current = 0
    out = bytearray()
    shifts = []
    new_layout = {}

    for entry in layout:
        field = entry.field

        if field.offset is not None and current < field.offset:
            out.extend(b"\x00" * (field.offset - current))
            current = field.offset

        if cls.__align__ and field.offset is None:
            padding = -current & (field.alignment - 1)
            out.extend(b"\x00" * padding)
            current += padding

        if entry.name == target_name:
            blob = target_bytes
        elif entry.name in length_updates and (
            not isinstance(entry.value, int) or length_updates[entry.name] != int(entry.value)
        ):
            blob = field.type.dumps(length_updates[entry.name], endian=cls.__cs__.endian)
        else:
            blob = original[entry.offset : entry.end]

        new_entry = _LayoutField(entry.name, entry.type, current, len(blob), entry.value, field)
        new_layout[entry.name] = new_entry
        if current != entry.offset and entry.name != target_name:
            shifts.append(FieldShift(entry.name, entry.offset, current))

        out.extend(blob)
        current += len(blob)

    if cls.__align__:
        padding = -current & (cls.__alignment__ - 1)
        out.extend(b"\x00" * padding)
        current += padding

    return bytes(out), shifts, new_layout


@dataclass(frozen=True)
class _LengthRegion:
    """A length field and the contiguous region of the record it describes."""

    name: str
    value: int
    start_index: int
    end_index: int | None
    automatic: bool


_LENGTH_NAMES = {"len", "length", "size", "total", "total_len", "total_length", "body_len", "payload_len"}


def _looks_like_length_name(name: str) -> bool:
    """Return whether a field name looks like a length or size field."""
    lowered = name.lower()
    return lowered in _LENGTH_NAMES or lowered.endswith(("_len", "_length", "_size"))


def _detect_length_fields(
    layout: list[_LayoutField],
    target: _LayoutField,
    total: int,
    requested: Iterable[str] | None,
    auto: bool,
) -> list[_LengthRegion]:
    """Find integer fields whose value depends on the encoded size of the target field.

    A length field is associated with the target through one of two concrete signals:

    - The length field is named in an array expression of a field that directly follows it, e.g. ``char data[len]``
      where ``len`` is the target. The described region starts after that length field and ends where the array (and
      any trailing fields covered by it) ends.
    - The length field is the first field of the record, its name looks like a length, and its value is the complete
      record size, e.g. a total size.

    Coincidental numeric equality is not generalized on purpose: a payload length or a CRC must not be rewritten just
    because it happens to have the same value as a region size. When in doubt, callers can pass ``length_fields``
    explicitly, which also allows a differently named total size.
    """
    if not layout:
        return []

    requested_names = set(requested) if requested is not None else None
    result = []

    for index, entry in enumerate(layout):
        if entry.name == target.name:
            continue
        if isinstance(entry.value, bool) or not isinstance(entry.value, int):
            continue
        if requested_names is not None and entry.name not in requested_names:
            continue
        explicitly_requested = requested_names is not None and entry.name in requested_names

        value = int(entry.value)
        candidates: list[tuple[int, int | None]] = []

        # A length field placed before an array whose dimension references it describes a region starting after
        # that length field, ending at the end of the array (and the complete record if the array is last).
        for start_index in range(index + 1, len(layout)):
            start = layout[start_index].offset
            if start < entry.end or target.offset < start:
                continue

            for end_index in range(start_index, len(layout)):
                end = layout[end_index].end
                if target.offset >= end or target.offset < start:
                    continue
                if end - start != value:
                    continue
                if (
                    auto
                    and not explicitly_requested
                    and not any(
                        _expression_references(layout[position].type, entry.name)
                        for position in range(start_index, end_index + 1)
                    )
                ):
                    continue
                candidates.append((start_index, None if end == total else end_index + 1))

        # A total length at the start of the record can also include the length field itself. Only do this for a
        # field that is plausibly named as a length, so an unrelated first integer is not rewritten by accident.
        if entry.offset == 0 and value == total and (explicitly_requested or _looks_like_length_name(entry.name)):
            candidates.append((0, None))

        if not candidates:
            continue

        start_index, end_index = min(
            candidates,
            key=lambda candidate: (
                (total if candidate[1] is None else layout[candidate[1] - 1].end) - layout[candidate[0]].offset,
                layout[candidate[0]].offset,
            ),
        )
        result.append(_LengthRegion(entry.name, value, start_index, end_index, auto and not explicitly_requested))

    return result


def _region_size(
    layout: list[_LayoutField],
    new_layout: dict[str, _LayoutField],
    preview_length: int,
    start_index: int,
    end_index: int | None,
) -> int:
    """Return the size of a length region using original indexes and new field offsets."""
    start = new_layout[layout[start_index].name].offset
    end = preview_length if end_index is None else new_layout[layout[end_index - 1].name].end
    return end - start


def normalize_leb128(
    obj: Any,
    field: str,
    *,
    signed_ranges: Iterable[SignedRange | tuple[int, int] | dict[str, Any]] | None = None,
    signed_bytes: bytes | None = None,
    length_fields: Iterable[str] | None = None,
    auto_length_fields: bool = True,
) -> LEB128NormalizationPlan:
    """Preview the canonicalization of a LEB128 field in a parsed structure.

    The preview includes length fields that depend on the encoded size of the field, because shrinking or growing the
    LEB128 encoding changes the size of the region they describe. Length fields are detected automatically when an
    array dimension references the target (e.g. ``char data[len]``), or when the first field of the record is a total
    size. Length fields can also be selected explicitly with ``length_fields``.

    Length association deliberately does not use coincidental numeric equality: a checksum or an unrelated payload
    length that happens to have the same value is not rewritten.

    Args:
        obj: A parsed structure instance.
        field: The name of the LEB128 field to normalize.
        signed_ranges: Existing signature ranges. Applying a plan that changes bytes in one of these ranges requires
            a new signature and is refused by :meth:`LEB128NormalizationPlan.apply` unless forced.
        signed_bytes: The exact bytes covered by the existing signature, if they are not the same as the serialized
            structure. Ranges are evaluated against these bytes.
        length_fields: Explicit names of length fields that must be updated.
        auto_length_fields: Detect length fields whose value depends on the encoded size of the target.

    Raises:
        LEB128NormalizationError: If the field is not a LEB128 field, the structure layout cannot be reproduced, or
            an explicitly requested length field does not describe a region containing the target.
    """
    layout, total = _layout(obj)
    original = obj.dumps()
    signed_original = original if signed_bytes is None else bytes(signed_bytes)

    if total != len(original):
        raise LEB128NormalizationError(
            f"Calculated layout is {total} bytes, but the structure serializes to {len(original)} bytes"
        )
    if signed_bytes is not None and len(signed_original) != len(original):
        raise LEB128NormalizationError(
            f"Signed bytes are {len(signed_original)} bytes, but the structure serializes to {len(original)} bytes"
        )

    target = next((entry for entry in layout if entry.name == field), None)
    if target is None:
        raise LEB128NormalizationError(f"Unknown field: {field!r}")
    if not issubclass(target.type, LEB128):
        raise LEB128NormalizationError(f"Field {field!r} is not a LEB128 field")

    value = target.value
    if not isinstance(value, LEB128):
        # A value assigned as a plain integer has no original encoding; it is already canonical.
        value = target.type(value)

    original_encoding = value.raw if value.raw is not None else value.canonical_bytes
    normalized_encoding = value.canonical_bytes

    if original[target.offset : target.end] != original_encoding:
        raise LEB128NormalizationError(
            f"Layout for {field!r} does not match the serialized bytes at offset {target.offset}"
        )

    detected = []
    if auto_length_fields or length_fields is not None:
        detected = _detect_length_fields(layout, target, total, length_fields, auto_length_fields)

    if length_fields is not None:
        found = {item.name for item in detected}
        missing = set(length_fields) - found
        if missing:
            raise LEB128NormalizationError(
                f"Length field(s) {sorted(missing)!r} do not describe a region containing {field!r}"
            )

    length_values = {item.name: item.value for item in detected}
    target_bytes = normalized_encoding
    preview = original
    shifts: list[FieldShift] = []
    updates: list[LengthUpdate] = []

    if detected:
        # Iterate to a fixed point: updating a LEB128 length field can itself change the size of the record, which
        # in turn changes the value of an enclosing length field, and can even change the encoded size of that
        # enclosing length field. Every iteration shrinks or grows monotonically towards a stable layout, so the
        # number of iterations is bounded by the number of encoded length fields.
        for _ in range(len(layout) + len(detected) + 2):
            preview, shifts, new_layout = _build_preview(obj, layout, field, target_bytes, length_values)

            changed = False
            new_updates = []
            for region in detected:
                entry = next(item for item in layout if item.name == region.name)
                updated = _region_size(layout, new_layout, len(preview), region.start_index, region.end_index)
                if updated != length_values.get(region.name):
                    length_values[region.name] = updated
                    changed = True

                region_start = layout[region.start_index].offset
                region_end = None if region.end_index is None else layout[region.end_index - 1].end
                new_entry = new_layout[region.name]
                new_updates.append(
                    LengthUpdate(
                        region.name,
                        region.value,
                        updated,
                        updated - region.value,
                        entry.offset,
                        new_entry.offset,
                        original[entry.offset : entry.end],
                        preview[new_entry.offset : new_entry.end],
                        region_start,
                        region_end,
                        region.automatic,
                    )
                )

            updates = new_updates
            if not changed:
                break
    else:
        preview, shifts, _ = _build_preview(obj, layout, field, target_bytes, length_values)

    dependent_fields = tuple(
        field_.name
        for field_ in type(obj).__fields__
        if field_.name != field and _expression_references(field_.type, field)
    )

    signed = _normalize_ranges(signed_ranges) if signed_ranges is not None else ()
    changed_ranges = tuple(
        range_ for range_ in signed if signed_original[range_.start : range_.end] != preview[range_.start : range_.end]
    )

    blocked_reason = None
    if changed_ranges:
        names = ", ".join(range_.name or f"{range_.start}:{range_.end}" for range_ in changed_ranges)
        blocked_reason = f"normalization changes signed range(s): {names}"

    old_key = value.evidence_key
    new_key = value.normalize().evidence_key

    return LEB128NormalizationPlan(
        structure=obj,
        field=field,
        value=int(value),
        signed=value.signed,
        offset=target.offset,
        original=original_encoding,
        normalized=normalized_encoding,
        original_bytes=original,
        normalized_bytes=preview,
        length_updates=tuple(updates),
        shifts=tuple(shifts),
        dependent_fields=dependent_fields,
        signed_ranges=signed,
        changed_signed_ranges=changed_ranges,
        old_evidence_key=old_key,
        new_evidence_key=new_key,
        old_evidence_digest=hashlib.sha256(signed_original).hexdigest(),
        new_evidence_digest=hashlib.sha256(preview).hexdigest(),
        blocked_reason=blocked_reason,
    )


def iter_leb128_values(obj: Any, *, path: str = "") -> Any:
    """Yield ``(path, value)`` pairs for every LEB128 field in a parsed structure or array.

    The path uses dots for structure fields and ``[index]`` for array entries, e.g. ``entries[0].len``. This allows a
    telemetry archive to aggregate over all values in a record, while still being able to verify the encoding of each
    individual field.

    Args:
        obj: A parsed structure, an array of structures or a LEB128 value.
        path: The prefix for the yielded paths.
    """
    from dissect.cstruct.types.structure import Structure

    if isinstance(obj, LEB128):
        yield path, obj
        return

    if isinstance(obj, Structure):
        for field in obj.__class__.__fields__:
            name = f"{path}.{field._name}" if path else field._name
            yield from iter_leb128_values(getattr(obj, field._name), path=name)
        return

    if isinstance(obj, (list, tuple)):
        for index, entry in enumerate(obj):
            yield from iter_leb128_values(entry, path=f"{path}[{index}]")


def leb128_evidence(obj: Any) -> dict[str, tuple[str, bool, int, bytes]]:
    """Return the evidence key of every LEB128 field in a parsed record, keyed by path.

    The keys include the exact original encoding, so a signature over the record can be checked against the evidence
    that was actually collected. Numeric equality alone does not collapse entries.
    """
    return {path: value.evidence_key for path, value in iter_leb128_values(obj)}


def leb128_aggregation(obj: Any) -> dict[str, int]:
    """Return the integer value of every LEB128 field in a parsed record, keyed by path.

    Unlike :func:`leb128_evidence`, this collapses overlong and canonical encodings of the same value, which is what
    value based aggregation needs.
    """
    return {path: value.aggregation_key for path, value in iter_leb128_values(obj)}


preview_leb128_normalization = normalize_leb128
