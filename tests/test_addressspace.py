from __future__ import annotations

from io import BytesIO

import pytest

from dissect.cstruct import (
    AddressMappingError,
    AddressSpace,
    AmbiguousMappingError,
    FileRange,
    MappedStream,
    Mapping,
    cstruct,
)
from dissect.cstruct.exception import NullPointerDereference


def make_chain_cs(compiled: bool = True) -> cstruct:
    cs = cstruct(pointer="uint16")
    cs.load(
        """
        struct node {
            uint32 value;
            node   *next;
        };
        struct root {
            node *head;
        };
        """,
        compiled=compiled,
    )
    return cs


def chain_image() -> tuple[BytesIO, BytesIO]:
    # segA: VA 0x0-0x10; root at VA 0 (head -> node n1 at VA 4), n1.next -> 0x1000
    seg_a = BytesIO()
    seg_a.write((4).to_bytes(2, "little"))  # root.head
    seg_a.write(b"\x00\x00")  # padding to VA 4
    seg_a.write((0x11111111).to_bytes(4, "little"))  # n1.value
    seg_a.write((0x1000).to_bytes(2, "little"))  # n1.next
    seg_a.write(b"\x00\x00")

    # segB: VA 0x1000-0x1010, backed at file offset 100; n2.next -> 4 (reference cycle)
    seg_b = BytesIO(b"\x00" * 100)
    seg_b.seek(100)
    seg_b.write((0x22222222).to_bytes(4, "little"))  # n2.value
    seg_b.write((4).to_bytes(2, "little"))  # n2.next -> n1
    seg_b.write(b"\x00\x00")
    return seg_a, seg_b


def test_mapping_validation() -> None:
    stream = BytesIO(b"abc")
    Mapping(0, 3, stream)
    Mapping(0, 3, stream, file_offset=5, name="data")

    with pytest.raises(ValueError, match="at least one"):
        Mapping(5, 5, stream)
    with pytest.raises(ValueError, match="at least one"):
        Mapping(5, 4, stream)
    with pytest.raises(ValueError, match="file offset"):
        Mapping(0, 3, stream, file_offset=-1)

    with pytest.raises(TypeError):
        AddressSpace().add_mapping(object())  # type: ignore[arg-type]


def test_address_space_version_and_mappings() -> None:
    mapping = Mapping(0, 4, BytesIO(b"test"))
    space = AddressSpace([mapping], version="v1.0")

    assert space.version == "v1.0"
    assert space.mappings == (mapping,)
    assert len(space) == 1
    assert list(space) == [mapping]

    other = Mapping(8, 12, BytesIO(b"more"))
    space.add_mapping(other)
    assert space.mappings == (mapping, other)

    space.remove_mapping(mapping)
    assert space.mappings == (other,)


def test_resolve_single_mapping() -> None:
    stream = BytesIO(b"abcdef")
    space = AddressSpace([Mapping(0x100, 0x106, stream, file_offset=0, name="seg")])

    chunks = space.resolve(0x101, 3)
    assert chunks == [[(space.mappings[0], 1, 3)]]


def test_resolve_spans_adjacent_mappings() -> None:
    first = Mapping(0x0, 0x2, BytesIO(b"\x01\x02"), 0, "lo")
    second = Mapping(0x2, 0x4, BytesIO(b"\x03\x04"), 0, "hi")
    space = AddressSpace([first, second])

    chunks = space.resolve(0x0, 4)
    assert [(mapping.name, offset, size) for ((mapping, offset, size),) in chunks] == [
        ("lo", 0, 2),
        ("hi", 0, 2),
    ]


def test_resolve_rejects_address_hole() -> None:
    space = AddressSpace(
        [
            Mapping(0x0, 0x2, BytesIO(b"\x01\x02"), 0, "lo"),
            Mapping(0x4, 0x6, BytesIO(b"\x05\x06"), 0, "hi"),
        ]
    )

    with pytest.raises(AddressMappingError, match="0x2"):
        space.resolve(0x0, 4)


def test_resolve_reports_version() -> None:
    space = AddressSpace(version="img-7")
    with pytest.raises(AddressMappingError, match="img-7"):
        space.resolve(0x10, 1)


def test_mapped_stream_basic_read_and_seek() -> None:
    stream = MappedStream(AddressSpace([Mapping(0x10, 0x14, BytesIO(b"\x01\x02\x03\x04"), 0)]))

    assert stream.seekable()
    assert stream.readable()
    assert not stream.writable()

    stream.seek(0x12)
    assert stream.tell() == 0x12
    assert stream.read(2) == b"\x03\x04"
    assert stream.tell() == 0x14

    stream.seek(-2, 1)
    assert stream.tell() == 0x12
    assert stream.read() == b"\x03\x04"


def test_mapped_stream_reads_across_mapping_boundary() -> None:
    space = AddressSpace(
        [
            Mapping(0x0, 0x2, BytesIO(b"\x01\x02"), 0, "lo"),
            Mapping(0x2, 0x4, BytesIO(b"\x03\x04"), 0, "hi"),
        ]
    )
    stream = MappedStream(space)

    assert stream.read(4) == b"\x01\x02\x03\x04"
    assert [(range_.file_offset, range_.size, range_.mapping.name) for range_ in stream.ranges] == [
        (0, 2, "lo"),
        (0, 2, "hi"),
    ]


def test_mapped_stream_does_not_zero_fill_holes() -> None:
    space = AddressSpace(
        [
            Mapping(0x0, 0x2, BytesIO(b"\x01\x02"), 0, "lo"),
            Mapping(0x4, 0x6, BytesIO(b"\x05\x06"), 0, "hi"),
        ]
    )
    stream = MappedStream(space)

    with pytest.raises(AddressMappingError):
        stream.read(4)
    # No range is recorded for the unbacked bytes
    assert stream.ranges == (FileRange(space.mappings[0].stream, 0, 2, space.mappings[0]),)


def test_mapped_stream_read_past_end_is_eof() -> None:
    stream = MappedStream(AddressSpace([Mapping(0x0, 0x2, BytesIO(b"\x01\x02"), 0)]))

    stream.seek(0x2)
    assert stream.read(10) == b""
    assert stream.read() == b""

    stream.seek(0x1)
    assert stream.read(10) == b"\x02"


def test_mapped_stream_consistent_overlap() -> None:
    backing = BytesIO(b"\x01\x02\x03\x04")
    duplicate = BytesIO(b"\x01\x02\x03\x04")
    space = AddressSpace(
        [
            Mapping(0x0, 0x4, backing, 0, "primary"),
            Mapping(0x2, 0x4, duplicate, 2, "overlay"),
        ]
    )

    assert MappedStream(space).read(4) == b"\x01\x02\x03\x04"


def test_mapped_stream_ambiguous_overlap() -> None:
    backing = BytesIO(b"\x01\x02\x03\x04")
    conflicting = BytesIO(b"\xff\xff")
    space = AddressSpace(
        [
            Mapping(0x0, 0x4, backing, 0, "primary"),
            Mapping(0x2, 0x4, conflicting, 0, "overlay"),
        ]
    )

    stream = MappedStream(space)
    with pytest.raises(AmbiguousMappingError, match="primary"):
        stream.read(4)


def test_mapped_stream_ambiguity_not_resolved_by_insertion_order() -> None:
    # The conflicting mapping is registered first; order must not pick a winner
    conflicting = BytesIO(b"\xff\xff")
    backing = BytesIO(b"\x01\x02\x03\x04")
    space = AddressSpace(
        [
            Mapping(0x2, 0x4, conflicting, 0, "overlay"),
            Mapping(0x0, 0x4, backing, 0, "primary"),
        ]
    )

    with pytest.raises(AmbiguousMappingError):
        MappedStream(space).read(4)


def test_pointer_dereference_through_mappings(compiled: bool) -> None:
    cs = make_chain_cs(compiled)
    seg_a, seg_b = chain_image()

    space = AddressSpace(version="v1.2.3")
    space.add_mapping(Mapping(0x0, 0x10, seg_a, 0, "segA"))
    space.add_mapping(Mapping(0x1000, 0x1010, seg_b, 100, "segB"))

    root = cs.root(MappedStream(space))
    head = root.head

    n1 = head.dereference()
    assert n1.value == 0x11111111

    n2 = n1.next.dereference()
    assert n2.value == 0x22222222


def test_pointer_ranges_single_and_spanning(compiled: bool) -> None:
    cs = make_chain_cs(compiled)
    seg_a, seg_b = chain_image()
    space = AddressSpace()
    space.add_mapping(Mapping(0x0, 0x10, seg_a, 0, "segA"))
    space.add_mapping(Mapping(0x1000, 0x1010, seg_b, 100, "segB"))

    stream = MappedStream(space)
    head = cs.root(stream).head

    ranges = head.ranges()
    assert len(ranges) == 1
    assert ranges[0].stream is seg_a
    assert ranges[0].file_offset == 4
    assert ranges[0].size == len(cs.node)
    assert ranges[0].mapping.name == "segA"

    assert [(r.file_offset, r.size, r.mapping.name) for r in head.dereference().next.ranges()] == [(100, 6, "segB")]


def test_pointer_ranges_object_spanning_mappings() -> None:
    cs = cstruct(pointer="uint16")
    space = AddressSpace(
        [
            Mapping(0x10, 0x12, BytesIO(b"\x01\x02"), 0, "lo"),
            Mapping(0x12, 0x14, BytesIO(b"\x03\x04"), 0, "hi"),
        ]
    )
    stream = MappedStream(space)
    pointer = cs._make_pointer(cs.uint32).__new__(
        cs._make_pointer(cs.uint32), 0x10, stream, context=None, endian=cs.endian
    )

    assert pointer.dereference() == 0x04030201
    assert [(r.file_offset, r.size, r.mapping.name) for r in pointer.ranges()] == [
        (0, 2, "lo"),
        (0, 2, "hi"),
    ]


def test_char_pointer_across_mapping_boundary() -> None:
    cs = cstruct(pointer="uint16")
    space = AddressSpace(
        [
            Mapping(0x100, 0x103, BytesIO(b"abc"), 0, "s1"),
            Mapping(0x103, 0x107, BytesIO(b"def\x00"), 0, "s2"),
        ]
    )
    stream = MappedStream(space)
    char_pointer = cs._make_pointer(cs.char).__new__(
        cs._make_pointer(cs.char), 0x100, stream, context=None, endian=cs.endian
    )

    assert char_pointer.dereference() == b"abcdef"
    assert [(r.file_offset, r.size, r.mapping.name) for r in char_pointer.ranges()] == [
        (0, 3, "s1"),
        (0, 4, "s2"),
    ]


def test_pointer_hole_raises_and_never_zero_fills(compiled: bool) -> None:
    cs = make_chain_cs(compiled)
    seg_a, _ = chain_image()
    space = AddressSpace([Mapping(0x0, 0x10, seg_a, 0, "segA")])
    stream = MappedStream(space)

    head = cs.root(stream).head
    # n1.next points to 0x1000, which has no mapping in this image
    with pytest.raises(AddressMappingError):
        head.dereference().next.dereference()


def test_pointer_ranges_require_mapped_stream(compiled: bool) -> None:
    cs = make_chain_cs(compiled)
    pointer = cs._make_pointer(cs.uint32)(b"\x02\x00\xff\xff\xff\xff")
    assert pointer.dereference() == 0xFFFFFFFF

    with pytest.raises(TypeError):
        pointer.ranges()


def test_pointer_ranges_null(compiled: bool) -> None:
    cs = make_chain_cs(compiled)
    stream = MappedStream(AddressSpace())
    null_pointer = cs._make_pointer(cs.uint32).__new__(
        cs._make_pointer(cs.uint32), 0, stream, context=None, endian=cs.endian
    )
    with pytest.raises(NullPointerDereference):
        null_pointer.ranges()


def test_shared_target_keeps_identity(compiled: bool) -> None:
    cs = make_chain_cs(compiled)
    seg_a, seg_b = chain_image()
    # Add a second root field that also points at n1 (VA 4)
    cs.load("struct shared { node *first; node *second; };")
    seg_a.seek(0)
    data = seg_a.getvalue()
    shared_data = (4).to_bytes(2, "little") + (4).to_bytes(2, "little") + data[4:]

    space = AddressSpace(
        [Mapping(0x0, 0x10, BytesIO(shared_data), 0, "segA"), Mapping(0x1000, 0x1010, seg_b, 100, "segB")]
    )
    obj = cs.shared(MappedStream(space))

    assert obj.first.dereference() is obj.second.dereference()


def test_cyclic_reference_keeps_identity(compiled: bool) -> None:
    cs = make_chain_cs(compiled)
    seg_a, seg_b = chain_image()
    space = AddressSpace(
        [
            Mapping(0x0, 0x10, seg_a, 0, "segA"),
            Mapping(0x1000, 0x1010, seg_b, 100, "segB"),
        ]
    )

    head = cs.root(MappedStream(space)).head
    n1 = head.dereference()
    n2 = n1.next.dereference()

    assert n2.next.dereference() is n1
    # Repeated dereferences keep returning the same object
    assert head.dereference() is n1


def test_same_vaddr_interpreted_per_image(compiled: bool) -> None:
    cs_a = make_chain_cs(compiled)
    cs_b = make_chain_cs(compiled)

    node_bytes = (0x11111111).to_bytes(4, "little") + (0).to_bytes(2, "little") + b"\x00\x00"
    other_bytes = (0x22222222).to_bytes(4, "little") + (0).to_bytes(2, "little") + b"\x00\x00"

    space_a = AddressSpace([Mapping(0x4, 0xA, BytesIO(node_bytes), 0, "imgA")], version="A")
    space_b = AddressSpace([Mapping(0x4, 0xA, BytesIO(other_bytes), 0, "imgB")], version="B")

    stream_a = MappedStream(space_a)
    stream_b = MappedStream(space_b)
    stream_a.seek(0x4)
    stream_b.seek(0x4)

    node_a = cs_a.node(stream_a)
    node_b = cs_b.node(stream_b)

    assert node_a.value == 0x11111111
    assert node_b.value == 0x22222222
    assert node_a is not node_b


def test_ambiguous_object_dereference_raises(compiled: bool) -> None:
    cs = make_chain_cs(compiled)
    seg_a, _ = chain_image()
    conflicting = BytesIO(b"\xff" * 6)

    space = AddressSpace(
        [
            Mapping(0x0, 0x10, BytesIO(seg_a.getvalue()), 0, "segA"),
            Mapping(0x4, 0xA, conflicting, 0, "conflict"),
        ]
    )

    head = cs.root(MappedStream(space)).head
    with pytest.raises(AmbiguousMappingError):
        head.dereference()


def test_truncated_object_raises_eof_not_zeros(compiled: bool) -> None:
    cs = make_chain_cs(compiled)
    space = AddressSpace([Mapping(0x10, 0x12, BytesIO(b"\x01\x02"), 0, "tiny")])
    stream = MappedStream(space)
    stream.seek(0x10)

    with pytest.raises(EOFError):
        cs.uint32(stream)


def test_mapping_with_file_offset() -> None:
    backing = BytesIO(b"\x00" * 8 + b"ABCD")
    space = AddressSpace([Mapping(0x100, 0x104, backing, file_offset=8, name="embedded")])
    stream = MappedStream(space)
    stream.seek(0x100)

    assert stream.read(4) == b"ABCD"
    assert stream.ranges[0].file_offset == 8


def test_overlap_starting_inside_object_conflicts() -> None:
    # A 4-byte object at VA 0x10; a second mapping only overlaps its last two bytes with different data
    space = AddressSpace(
        [
            Mapping(0x10, 0x14, BytesIO(b"\x01\x02\x03\x04"), 0, "primary"),
            Mapping(0x12, 0x14, BytesIO(b"\xff\xff"), 0, "overlay"),
        ]
    )
    stream = MappedStream(space)
    stream.seek(0x10)

    with pytest.raises(AmbiguousMappingError):
        stream.read(4)


def test_overlap_starting_inside_object_consistent() -> None:
    space = AddressSpace(
        [
            Mapping(0x10, 0x14, BytesIO(b"\x01\x02\x03\x04"), 0, "primary"),
            Mapping(0x12, 0x14, BytesIO(b"\x03\x04"), 0, "overlay"),
        ]
    )
    stream = MappedStream(space)
    stream.seek(0x10)

    assert stream.read(4) == b"\x01\x02\x03\x04"


def test_pointer_arithmetic_resolves_through_mappings() -> None:
    cs = cstruct(pointer="uint8")
    space = AddressSpace([Mapping(0x20, 0x22, BytesIO(b"\x07\x09"), 0, "pair")])
    stream = MappedStream(space)
    pointer_type = cs._make_pointer(cs.uint8)
    pointer = pointer_type.__new__(pointer_type, 0x20, stream, context=None, endian=cs.endian)

    assert pointer.dereference() == 7

    advanced = pointer + 1
    assert int(advanced) == 0x21
    assert advanced.dereference() == 9


def test_pointer_arithmetic_into_hole_raises() -> None:
    cs = cstruct(pointer="uint8")
    space = AddressSpace([Mapping(0x20, 0x21, BytesIO(b"\x07"), 0, "single")])
    stream = MappedStream(space)
    pointer_type = cs._make_pointer(cs.uint8)
    pointer = pointer_type.__new__(pointer_type, 0x20, stream, context=None, endian=cs.endian)

    with pytest.raises(AddressMappingError):
        (pointer + 1).dereference()
