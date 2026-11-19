"""Surgical in-place patch of a single flag inside a binary packet.

The packet contains unmodeled padding and a union branch we do not understand
(unknown selector). Re-serializing parsed cstruct objects would zero those
regions (``Structure._write`` fills gaps with ``\\x00`` and ``Union._write``
pads unmodeled branches with ``\\x00``), so instead we:

1. Prove the absolute byte range of the target field from static layout
   metadata only. If the offset/size cannot be proven, we refuse to modify.
2. Serialize only the new field value and splice it into the original bytes.
3. Verify every associated length field still matches reality; if the patch
   changed a covered length and its coverage cannot be proven, we refuse.
4. Verify that everything outside the approved ranges is byte-identical to
   the original packet (nothing unapproved gets zeroed or rewritten).
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Callable

from dissect.cstruct import cstruct


class RefusalError(Exception):
    """Raised when a boundary cannot be proven and modification is refused."""


PACKET_DEF = """
struct Header {
    uint16  msg_type;
    uint16  flags;          // <- the flag we want to fix
    uint32  total_len;      // associated length: covers the whole packet
    uint32  payload_len;    // associated length: covers the payload section
    uint32  selector;       // union branch selector
};

union Payload {
    uint32  a[3];           // known branch (selector == 1)
    uint16  b[6];           // known branch (selector == 2)
    // selector == 0x99 is an unknown branch: opaque bytes we must preserve
};
"""

cs = cstruct()
cs.load(PACKET_DEF)

BYTEORDER = {"<": "little", ">": "big", "!": "big"}[cs.endian]

HEADER_SIZE = len(cs.Header)
PAYLOAD_SIZE = len(cs.Payload)

# Packet layout (all offsets provable from the static definition above):
#   [0, 16)   Header
#   [16, 24)  unmodeled padding (8 bytes, not part of any struct)
#   [24, 36)  Payload union (selector 0x99 -> unknown branch, opaque bytes)
#   [36, 40)  unmodeled trailing bytes
PADDING_OFF, PADDING_SIZE = 16, 8
PAYLOAD_OFF = PADDING_OFF + PADDING_SIZE
TRAILING_OFF = PAYLOAD_OFF + PAYLOAD_SIZE
PACKET_SIZE = TRAILING_OFF + 4

ORIGINAL_PADDING = b"\xde\xad\xbe\xef\xca\xfe\xba\xbe"
ORIGINAL_TRAILING = b"\x13\x37\xc0\xde"
UNKNOWN_BRANCH_BYTES = b"\x99\x88\x77\x66\x55\x44\x33\x22\x11\x00\xff\xee"

ORIGINAL_FLAGS = 0x0001
NEW_FLAGS = 0x8001


def build_original_packet() -> bytes:
    """Build the original packet with an unknown union branch and padding."""
    header = cs.Header(
        msg_type=0x0042,
        flags=ORIGINAL_FLAGS,
        total_len=PACKET_SIZE,
        payload_len=PAYLOAD_SIZE,
        selector=0x99,  # unknown branch: Payload bytes below are opaque
    )
    return (
        bytes(header)
        + ORIGINAL_PADDING
        + UNKNOWN_BRANCH_BYTES
        + ORIGINAL_TRAILING
    )


@dataclass
class AssociatedLength:
    """A length field whose covered range must stay consistent after patching."""

    name: str
    struct_type: Any
    struct_offset: int  # absolute offset of the struct holding the field
    measure: Callable[[bytes], int]  # expected value computed from the packet

    def range(self) -> tuple[int, int]:
        offset, size = prove_static_range(self.struct_type, self.name)
        return self.struct_offset + offset, size


@dataclass
class PatchReport:
    approved_ranges: list[tuple[int, int, str]] = dc_field(default_factory=list)
    associated_lengths: list[tuple[str, tuple[int, int], int, bool]] = dc_field(default_factory=list)


def prove_static_range(struct_type: Any, field_name: str) -> tuple[int, int]:
    """Return (offset, size) of a field, or refuse if it cannot be proven."""
    field = struct_type.__lookup__.get(field_name)
    if field is None:
        raise RefusalError(f"no such field: {struct_type.__name__}.{field_name}")
    if field.bits is not None:
        raise RefusalError(
            f"{field_name} is a bitfield and shares its bytes with neighbours; "
            "cannot prove an exclusive byte range"
        )
    if field.offset is None:
        raise RefusalError(
            f"{field_name} has no static offset (a preceding field is dynamic); "
            "cannot prove where it lives in the packet"
        )
    if field.type.__dynamic__ or field.type.__size__ is None:
        raise RefusalError(f"{field_name} has a dynamic size; cannot prove its extent")
    return field.offset, field.type.__size__


def surgical_patch(
    packet: bytes,
    struct_type: Any,
    struct_offset: int,
    field_name: str,
    new_value: Any,
    associated_lengths: list[AssociatedLength],
) -> tuple[bytes, PatchReport]:
    """Patch one field in place. Refuses when any boundary is unprovable."""
    report = PatchReport()

    field_offset, field_size = prove_static_range(struct_type, field_name)
    abs_off = struct_offset + field_offset
    abs_end = abs_off + field_size
    if abs_end > len(packet):
        raise RefusalError(
            f"target range [{abs_off}, {abs_end}) exceeds packet size {len(packet)}"
        )

    new_bytes = struct_type.__lookup__[field_name].type.dumps(new_value)
    if len(new_bytes) != field_size:
        raise RefusalError(
            f"serialized value is {len(new_bytes)} bytes, field is {field_size}; "
            "size-changing patches are out of scope"
        )

    # Size is unchanged, so every associated length must keep its current
    # value. Prove each length's coverage and verify consistency; if a length
    # would have to change and we cannot prove its coverage, refuse.
    for length in associated_lengths:
        len_off, len_size = length.range()  # raises RefusalError if unprovable
        current = int.from_bytes(
            packet[len_off : len_off + len_size], BYTEORDER
        )
        expected = length.measure(packet)
        if current != expected:
            raise RefusalError(
                f"associated length {length.name}={current} does not match "
                f"measured {expected}; coverage assumption is wrong, refusing"
            )
        report.associated_lengths.append(
            (length.name, (len_off, len_size), current, False)
        )

    patched = bytearray(packet)
    patched[abs_off:abs_end] = new_bytes
    report.approved_ranges.append((abs_off, abs_end, field_name))

    result = bytes(patched)

    # Verification 1: reading back the new packet yields the target value.
    reread = struct_type(result[struct_offset : struct_offset + len(struct_type)])
    if getattr(reread, field_name) != new_value:
        raise AssertionError(f"read-back of {field_name} failed after patching")

    # Verification 2: nothing outside the approved ranges may differ.
    approved = sorted((start, end) for start, end, _ in report.approved_ranges)
    cursor = 0
    for start, end in approved:
        assert packet[cursor:start] == result[cursor:start], (
            f"unapproved range [{cursor}, {start}) was modified"
        )
        cursor = end
    assert packet[cursor:] == result[cursor:], (
        f"unapproved range [{cursor}, {len(packet)}) was modified"
    )

    return result, report


def demonstrate_reserialization_hazards() -> None:
    """Show why parse-and-reserialize is not an option for this packet."""
    demo = cstruct()
    # 1. Unmodeled padding inside an aligned struct is zero-filled on write.
    demo.load(
        """
        struct Aligned {
            uint8   tag;
            uint32  value;
        };
        """,
        align=True,
    )
    raw = b"\xaa" + b"\xde\xad\xbe" + b"\x01\x00\x00\x00"  # gap bytes are nonzero
    assert bytes(demo.Aligned(raw))[1:4] == b"\x00" * 3
    print("  aligned struct re-serialization zeroes the unmodeled gap:"
          f" {raw.hex()} -> {bytes(demo.Aligned(raw)).hex()}")

    # 2. A union model that only covers the known branches is smaller than
    #    the unknown branch on the wire: parsing consumes too few bytes and
    #    re-serialization truncates the unknown branch's tail.
    demo.load(
        """
        union UndersizedPayload {
            uint32  a[2];   // known branch 1: 8 bytes
            uint16  b[4];   // known branch 2: 8 bytes
            // the real selector 0x99 branch is 12 bytes: not modeled
        };
        """
    )
    raw = b"\x99\x88\x77\x66\x55\x44\x33\x22\x11\x00\xff\xee"  # 12 bytes on the wire
    model = demo.UndersizedPayload(raw)
    rewritten = bytes(model)
    assert len(rewritten) == 8 and rewritten != raw
    print("  undersized union model truncates the unknown branch:"
          f" {raw.hex()} -> {rewritten.hex()} ({len(raw)} -> {len(rewritten)} bytes)")

    # 3. Re-serializing only the modeled parts of our packet drops the
    #    unmodeled padding and trailing bytes entirely.
    packet = build_original_packet()
    header = cs.Header(packet[:HEADER_SIZE])
    payload = cs.Payload(packet[PAYLOAD_OFF : PAYLOAD_OFF + PAYLOAD_SIZE])
    rebuilt = bytes(header) + bytes(payload)
    assert len(rebuilt) == HEADER_SIZE + PAYLOAD_SIZE < len(packet)
    print(f"  modeled-parts-only rebuild shrinks the packet:"
          f" {len(packet)} -> {len(rebuilt)} bytes (padding and trailing lost)")


def demonstrate_refusal() -> None:
    """A dynamic field before the flag makes the offset unprovable -> refuse."""
    dynamic_cs = cstruct()
    dynamic_cs.load(
        """
        struct DynamicHeader {
            uint16  name_len;
            char    name[name_len];
            uint16  flags;
        };
        """
    )
    try:
        prove_static_range(dynamic_cs.DynamicHeader, "flags")
    except RefusalError as exc:
        print(f"  refused as required: {exc}")
    else:
        raise AssertionError("patch should have been refused")


def main() -> None:
    packet = build_original_packet()
    print(f"original packet ({len(packet)} bytes): {packet.hex()}")

    associated = [
        AssociatedLength(
            "total_len", cs.Header, 0, measure=lambda data: len(data)
        ),
        AssociatedLength(
            "payload_len", cs.Header, 0, measure=lambda data: PAYLOAD_SIZE
        ),
    ]

    patched, report = surgical_patch(
        packet, cs.Header, 0, "flags", NEW_FLAGS, associated
    )

    print("\nchange scope (approved ranges):")
    for start, end, name in report.approved_ranges:
        print(f"  [{start}, {end})  {name}  {packet[start:end].hex()} -> {patched[start:end].hex()}")

    print("\nassociated lengths (checked, none needed updating):")
    for name, (off, size), value, updated in report.associated_lengths:
        state = "updated" if updated else "unchanged"
        print(f"  {name} @ [{off}, {off + size}) = {value}  ({state})")

    print("\nread-back checks on the new packet:")
    header = cs.Header(patched[:HEADER_SIZE])
    print(f"  flags       = 0x{header.flags:04x}  (expected 0x{NEW_FLAGS:04x})")
    assert header.flags == NEW_FLAGS

    print("\nunapproved ranges preserved (no re-serialization zeroing):")
    print(f"  padding   [{PADDING_OFF}, {PADDING_OFF + PADDING_SIZE})  "
          f"{patched[PADDING_OFF:PADDING_OFF + PADDING_SIZE].hex()}")
    assert patched[PADDING_OFF:PADDING_OFF + PADDING_SIZE] == ORIGINAL_PADDING
    print(f"  union     [{PAYLOAD_OFF}, {PAYLOAD_OFF + PAYLOAD_SIZE})  "
          f"{patched[PAYLOAD_OFF:PAYLOAD_OFF + PAYLOAD_SIZE].hex()}  (unknown branch 0x99)")
    assert patched[PAYLOAD_OFF:PAYLOAD_OFF + PAYLOAD_SIZE] == UNKNOWN_BRANCH_BYTES
    print(f"  trailing  [{TRAILING_OFF}, {PACKET_SIZE})  "
          f"{patched[TRAILING_OFF:].hex()}")
    assert patched[TRAILING_OFF:] == ORIGINAL_TRAILING

    print("\nwhy re-serialization is not an option:")
    demonstrate_reserialization_hazards()

    print("\nrefusal path (unprovable boundary):")
    demonstrate_refusal()

    print("\nall checks passed")


if __name__ == "__main__":
    main()
