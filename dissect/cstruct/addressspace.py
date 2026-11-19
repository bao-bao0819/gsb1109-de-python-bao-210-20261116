"""Virtual address spaces built from image mappings.

Pointers in a memory image hold *virtual* addresses, not file offsets. An :class:`AddressSpace` describes how ranges of
virtual addresses map back to ranges of one or more underlying file-like objects. A :class:`MappedStream` presents the
address space as a single seekable stream, so a structure can be read across several mappings and even several images.

The resolution rules are deliberately strict:

* A read may span multiple adjacent mappings, but stops with an :class:`~dissect.cstruct.exception.AddressMappingError`
  at the first address hole that has no backing mapping. Missing bytes are never silently zero-filled.
* When several mappings overlap, the bytes each mapping provides for the overlapping region are compared. Identical
  bytes are accepted, but differing bytes raise an :class:`~dissect.cstruct.exception.AmbiguousMappingError` instead of
  guessing a winner based on mapping order.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import SEEK_CUR, SEEK_END, SEEK_SET
from typing import TYPE_CHECKING, BinaryIO, NamedTuple

from dissect.cstruct.exception import AddressMappingError, AmbiguousMappingError

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True)
class Mapping:
    """A contiguous range of virtual addresses backed by a region of a file-like object.

    Attributes:
        vaddr_start: First virtual address of the mapping.
        vaddr_end: One-past-the-last virtual address of the mapping.
        stream: The file-like object holding the backing bytes.
        file_offset: Offset in ``stream`` corresponding to ``vaddr_start``.
        name: Human readable name, used in error messages.
    """

    vaddr_start: int
    vaddr_end: int
    stream: BinaryIO
    file_offset: int = 0
    name: str = ""

    def __post_init__(self) -> None:
        if self.vaddr_end <= self.vaddr_start:
            raise ValueError("A mapping must cover at least one virtual address")
        if self.file_offset < 0:
            raise ValueError("A file offset cannot be negative")

    def contains(self, vaddr: int) -> bool:
        return self.vaddr_start <= vaddr < self.vaddr_end

    def label(self) -> str:
        return self.name or f"mapping {self.vaddr_start:#x}-{self.vaddr_end:#x}"


class FileRange(NamedTuple):
    """The actual file region that satisfied a virtual read.

    Attributes:
        stream: The file-like object the bytes were read from.
        file_offset: The offset in ``stream`` where the read started.
        size: The number of bytes read.
        mapping: The mapping that translated the virtual address.
    """

    stream: BinaryIO
    file_offset: int
    size: int
    mapping: Mapping


class AddressSpace:
    """A collection of mappings describing one image's virtual address space.

    Each image gets its own :class:`AddressSpace`, so the same virtual address is interpreted independently in
    different images. Mappings may be added and removed at runtime, for example when image sections are discovered.
    """

    def __init__(self, mappings: list[Mapping] | None = None, *, version: str | None = None) -> None:
        self.version = version
        self._mappings: list[Mapping] = []
        for mapping in mappings or []:
            self.add_mapping(mapping)

    def __len__(self) -> int:
        return len(self._mappings)

    def __iter__(self) -> Iterator[Mapping]:
        return iter(self._mappings)

    @property
    def mappings(self) -> tuple[Mapping, ...]:
        """The current mappings, in insertion order."""
        return tuple(self._mappings)

    def add_mapping(self, mapping: Mapping) -> None:
        """Add a mapping. Overlapping mappings are allowed and checked for conflicting bytes on read."""
        if not isinstance(mapping, Mapping):
            raise TypeError("Expected a Mapping instance")
        self._mappings.append(mapping)

    def remove_mapping(self, mapping: Mapping) -> None:
        """Remove a previously added mapping."""
        self._mappings.remove(mapping)

    def contains(self, vaddr: int) -> bool:
        """Return whether at least one mapping covers the given virtual address."""
        return any(mapping.contains(vaddr) for mapping in self._mappings)

    def resolve(self, vaddr: int, size: int) -> list[list[tuple[Mapping, int, int]]]:
        """Resolve a virtual range to the backing regions of every mapping covering it.

        Returns a list of chunks, one per contiguous span with a fixed set of covering mappings. Each chunk lists
        ``(mapping, file_offset, size)`` candidates for the same virtual bytes; the first candidate is the primary
        source and the remaining candidates are overlapping mappings that must provide identical bytes.

        Raises:
            AddressMappingError: If part of the range has no backing mapping.
        """
        return list(self.iter_resolve(vaddr, size))

    def iter_resolve(self, vaddr: int, size: int) -> Iterator[list[tuple[Mapping, int, int]]]:
        """Lazily yield resolved chunks, raising :class:`AddressMappingError` at the first address hole."""
        if size < 0:
            raise ValueError("A resolved size cannot be negative")
        if size == 0:
            return

        end = vaddr + size
        position = vaddr

        while position < end:
            candidates = [mapping for mapping in self._mappings if mapping.contains(position)]
            if not candidates:
                raise AddressMappingError(
                    f"Virtual address {position:#x} is not covered by any mapping"
                    + (f" in image {self.version!r}" if self.version is not None else "")
                )

            # Split at the nearest mapping end *or* start, so a mapping that begins inside the chunk is compared
            # byte for byte instead of being silently skipped for the first part of the overlap
            boundaries = [mapping.vaddr_end for mapping in candidates]
            boundaries.extend(mapping.vaddr_start for mapping in self._mappings if position < mapping.vaddr_start < end)
            chunk_end = min([*boundaries, end])
            chunk_size = chunk_end - position

            chunk = []
            for mapping in candidates:
                file_offset = mapping.file_offset + (position - mapping.vaddr_start)
                chunk.append((mapping, file_offset, chunk_size))
            yield chunk
            position = chunk_end


class MappedStream:
    """A seekable stream over an :class:`AddressSpace`.

    The stream position is a virtual address. Reads are translated to the underlying files through the mappings and
    every consumed file region is recorded in :attr:`ranges`. The stream is safe to share while parsing one image,
    matching how a regular file-like object is shared and repositioned during structure parsing.
    """

    def __init__(self, address_space: AddressSpace) -> None:
        self.address_space = address_space
        self._vaddr = 0
        self._ranges: list[FileRange] = []
        self._object_cache: dict[tuple[type, int], object] = {}
        self._closed = False

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def tell(self) -> int:
        return self._vaddr

    def seek(self, offset: int, whence: int = SEEK_SET) -> int:
        if whence == SEEK_SET:
            target = offset
        elif whence == SEEK_CUR:
            target = self._vaddr + offset
        elif whence == SEEK_END:
            raise ValueError("A virtual address space has no defined end, seek to an absolute address instead")
        else:
            raise ValueError(f"Invalid whence value: {whence}")

        # Normalize e.g. a Pointer (an int subclass) to a plain virtual address
        target = int(target)
        if target < 0:
            raise ValueError("Cannot seek to a negative virtual address")
        self._vaddr = target
        return target

    def cached_object(self, type_: type, vaddr: int) -> tuple[bool, object]:
        """Return ``(found, object)`` for a previously parsed object at a virtual address."""
        key = (type_, int(vaddr))
        if key in self._object_cache:
            return True, self._object_cache[key]
        return False, None

    def cache_object(self, type_: type, vaddr: int, value: object) -> None:
        """Remember the object parsed at a virtual address, so shared and cyclic references keep identity."""
        self._object_cache[(type_, int(vaddr))] = value

    def mark(self) -> int:
        """Return a marker for use with :meth:`ranges_since`."""
        return len(self._ranges)

    def ranges_since(self, marker: int) -> tuple[FileRange, ...]:
        """Return the file ranges consumed since the given marker was taken.

        Adjacent ranges backed by the same mapping and contiguous file region are coalesced, so the result is the
        same regardless of whether a structure was read in one call or field by field.
        """
        return _coalesce_ranges(self._ranges[marker:])

    @property
    def ranges(self) -> tuple[FileRange, ...]:
        """All file ranges consumed by reads on this stream so far, coalesced per contiguous file region."""
        return _coalesce_ranges(self._ranges)

    def read(self, size: int = -1) -> bytes:
        if self._closed:
            raise ValueError("I/O operation on a closed stream")

        if not self.address_space.mappings:
            return b""

        # Reading past the highest mapped address is treated as the end of the stream, mirroring a regular file's
        # short read. An address hole *between* mapped regions still raises AddressMappingError below.
        end_of_space = max(mapping.vaddr_end for mapping in self.address_space.mappings)
        if self._vaddr >= end_of_space:
            return b""

        read_size = end_of_space - self._vaddr if size is None or size < 0 else min(size, end_of_space - self._vaddr)
        result = bytearray()

        # Resolve lazily so the bytes and ranges of earlier chunks are already recorded if a later chunk hits a hole
        for candidates in self.address_space.iter_resolve(self._vaddr, read_size):
            primary_mapping, primary_offset, chunk_size = candidates[0]
            primary_bytes = self._read_region(primary_mapping, primary_offset, chunk_size)

            for mapping, file_offset, _ in candidates[1:]:
                other_bytes = self._read_region(mapping, file_offset, chunk_size)
                if other_bytes != primary_bytes:
                    raise AmbiguousMappingError(
                        f"Overlapping mappings {primary_mapping.label()!r} and {mapping.label()!r} provide different "
                        f"bytes at virtual address {self._vaddr:#x}"
                    )

            result.extend(primary_bytes)
            self._ranges.append(FileRange(primary_mapping.stream, primary_offset, chunk_size, primary_mapping))
            self._vaddr += chunk_size

        return bytes(result)

    def readinto(self, buffer: bytearray) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def close(self) -> None:
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    @staticmethod
    def _read_region(mapping: Mapping, file_offset: int, size: int) -> bytes:
        stream = mapping.stream
        stream.seek(file_offset)
        data = stream.read(size)
        if len(data) != size:
            raise EOFError(
                f"Read {len(data)} bytes from {mapping.label()!r} at file offset {file_offset:#x}, expected {size}"
            )
        return data


def _coalesce_ranges(ranges: list[FileRange]) -> tuple[FileRange, ...]:
    """Merge consecutive ranges that are contiguous regions of the same mapping."""
    merged: list[FileRange] = []
    for range_ in ranges:
        if merged:
            previous = merged[-1]
            if (
                previous.mapping is range_.mapping
                and previous.stream is range_.stream
                and previous.file_offset + previous.size == range_.file_offset
            ):
                merged[-1] = FileRange(
                    previous.stream,
                    previous.file_offset,
                    previous.size + range_.size,
                    previous.mapping,
                )
                continue
        merged.append(range_)
    return tuple(merged)
