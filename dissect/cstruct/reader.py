from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any, BinaryIO

from dissect.cstruct.bitbuffer import BitBuffer
from dissect.cstruct.bounded import IncompleteRead, as_bounded
from dissect.cstruct.types.base import EOF, BaseArray
from dissect.cstruct.types.char import CharArray
from dissect.cstruct.types.enum import EnumMetaType
from dissect.cstruct.types.structure import StructureMetaType, UnionMetaType
from dissect.cstruct.types.void import VoidArray
from dissect.cstruct.types.wchar import Wchar, WcharArray
from dissect.cstruct.util import normalize_endianness

if TYPE_CHECKING:
    from dissect.cstruct.bounded import _BoundedReader
    from dissect.cstruct.cstruct import AllowedEndianness, Endianness
    from dissect.cstruct.types.base import BaseType
    from dissect.cstruct.types.structure import Structure


def read_partial(
    type_: type[BaseType],
    stream: BinaryIO | bytes | bytearray | memoryview,
    *,
    limit: int | None = None,
    endian: AllowedEndianness | None = None,
) -> Any:
    """Parse a value while guaranteeing that no read crosses the available payload boundary.

    ``stream`` is a file-like object or a bytes-like object. When it is a stream, ``limit`` may bound
    the readable region (e.g. to the size of one parent record); when omitted the whole stream up to
    its current end is available. Bytes-like input is bounded to its own length.

    On success the parsed value is returned, exactly like a regular read. If a (possibly corrupted)
    count or header cannot be satisfied within the boundary, an
    :class:`~dissect.cstruct.bounded.IncompleteRead` is raised carrying every value that was fully
    confirmed before the failure, the byte offset at which reading stopped, the original error and the
    field/index path of the failing element.

    Truncating the input only changes whether a read is complete or incomplete: the same prefix always
    yields the same confirmed values, because no layer ever reads past the boundary it was given.
    """
    if isinstance(stream, (bytes, bytearray, memoryview)):
        limit = len(stream) if limit is None else min(limit, len(stream))
        stream = io.BytesIO(stream)

    reader = as_bounded(stream, limit)
    eff_endian = normalize_endianness(endian) if endian is not None else type_.__cs__.endian
    try:
        return _read_value(type_, reader, None, eff_endian)
    except IncompleteRead:
        raise
    except EOFError as error:
        raise IncompleteRead(None, reader.tell(), error, []) from error


def _read_value(
    type_: type[BaseType], stream: _BoundedReader, context: dict[str, Any] | None, endian: Endianness
) -> Any:
    if isinstance(type_, StructureMetaType):
        return _read_structure(type_, stream, context, endian)
    if issubclass(type_, BaseArray):
        return _read_array(type_, stream, context, endian)

    # The shared bounded stream already clamps every read to the current payload; fixed-size values
    # additionally raise EOFError themselves on a short read, which the enclosing layer captures.
    return type_._read(stream, context=context, endian=endian)


def _read_array(
    array_type: type[BaseArray],
    stream: _BoundedReader,
    context: dict[str, Any] | None,
    endian: Endianness,
) -> Any:
    if issubclass(array_type, VoidArray):
        return array_type()

    if array_type.__null_terminated__:
        return _read_null_terminated(array_type, stream, context, endian)

    element_type = array_type.__type__
    entries: list[Any] = []

    if _is_composite_element(element_type):
        count = array_type._num_entries(context)
        if count == EOF:
            return _read_eof_composites(array_type, element_type, stream, context, endian)
        return _read_fixed_composites(array_type, element_type, count, stream, context, endian)

    if element_type.__size__ is None:
        # Variable-size leaf elements (e.g. LEB128) can only be consumed one at a time
        count = array_type._num_entries(context)
        if count == EOF:
            return _read_eof_composites(array_type, element_type, stream, context, endian)
        return _read_fixed_composites(array_type, element_type, count, stream, context, endian)

    return _read_fixed_sized_elements(array_type, element_type, stream, context, endian)


def _is_composite_element(element_type: type[BaseType]) -> bool:
    return isinstance(element_type, StructureMetaType) or issubclass(element_type, BaseArray)


def _read_fixed_composites(
    array_type: type[BaseArray],
    element_type: type[BaseType],
    count: int,
    stream: _BoundedReader,
    context: dict[str, Any] | None,
    endian: Endianness,
) -> Any:
    entries: list[Any] = []
    for index in range(count):
        try:
            entries.append(_read_value(element_type, stream, context, endian))
        except IncompleteRead as error:
            confirmed = _construct_array(array_type, entries)
            raise IncompleteRead(confirmed, stream.tell(), error, [index, *error.path]) from error
        except EOFError as error:
            confirmed = _construct_array(array_type, entries)
            raise IncompleteRead(confirmed, stream.tell(), error, [index]) from error

    return _construct_array(array_type, entries)


def _read_eof_composites(
    array_type: type[BaseArray],
    element_type: type[BaseType],
    stream: _BoundedReader,
    context: dict[str, Any] | None,
    endian: Endianness,
) -> Any:
    entries: list[Any] = []
    index = 0
    while True:
        start = stream.tell()
        try:
            entries.append(_read_value(element_type, stream, context, endian))
        except IncompleteRead as error:
            if stream.tell() == start:
                # The element could not even start: this is the clean end of the payload
                break
            confirmed = _construct_array(array_type, entries)
            raise IncompleteRead(confirmed, stream.tell(), error, [index, *error.path]) from error
        except EOFError as error:
            if stream.tell() == start:
                break
            # Some bytes of the element were consumed before the boundary, so it is only partial
            confirmed = _construct_array(array_type, entries)
            raise IncompleteRead(confirmed, stream.tell(), error, [index]) from error
        index += 1

    return _construct_array(array_type, entries)


def _read_fixed_sized_elements(
    array_type: type[BaseArray],
    element_type: type[BaseType],
    stream: _BoundedReader,
    context: dict[str, Any] | None,
    endian: Endianness,
) -> Any:
    """Read an array of fixed-size leaf elements in one clamped bulk read.

    The bounded reader never returns bytes past the payload limit. On a short read the elements that
    are fully contained in the returned bytes are still confirmed, instead of discarding the buffer.
    """
    elem_size = element_type.__size__
    count = array_type._num_entries(context)

    if count == EOF:
        data = stream.read(-1)
        if len(data) % elem_size == 0:
            # Clean boundary: keep the type's optimized bulk reader on the complete buffer
            entries = element_type._read_array(io.BytesIO(data), EOF, context=context, endian=endian)
            return _construct_array(array_type, entries)

        # A trailing partial element: confirm only the fully contained elements
        full = len(data) // elem_size
        entries = _parse_elements(element_type, data, full, context, endian)
        confirmed = _construct_array(array_type, entries)
        raise IncompleteRead(confirmed, stream.tell(), EOFError("Unexpected end of data"), [full])

    data = stream.read(elem_size * count)
    if len(data) == elem_size * count:
        # Fast path: hand the complete buffer to the type's optimized bulk reader
        entries = element_type._read_array(io.BytesIO(data), count, context=context, endian=endian)
        return _construct_array(array_type, entries)

    full = len(data) // elem_size
    entries = _parse_elements(element_type, data, full, context, endian)
    confirmed = _construct_array(array_type, entries)
    raise IncompleteRead(confirmed, stream.tell(), EOFError("Unexpected end of data"), [full])


def _parse_elements(
    element_type: type[BaseType],
    data: bytes,
    count: int,
    context: dict[str, Any] | None,
    endian: Endianness,
) -> list[Any]:
    elem_size = element_type.__size__
    entries: list[Any] = []
    for index in range(count):
        chunk = data[index * elem_size : (index + 1) * elem_size]
        entries.append(element_type._read(io.BytesIO(chunk), context=context, endian=endian))
    return entries


def _read_null_terminated(
    array_type: type[BaseArray],
    stream: _BoundedReader,
    context: dict[str, Any] | None,
    endian: Endianness,
) -> Any:
    if issubclass(array_type, CharArray):
        return _read_terminated_string(array_type, stream, 1, endian)
    if issubclass(array_type, WcharArray):
        return _read_terminated_string(array_type, stream, 2, endian)

    element_type = array_type.__type__
    entries: list[Any] = []
    index = 0
    while True:
        try:
            value = element_type._read(stream, context=context, endian=endian)
        except IncompleteRead as error:
            confirmed = _construct_array(array_type, entries)
            raise IncompleteRead(confirmed, stream.tell(), error, [index, *error.path]) from error
        except EOFError as error:
            # The boundary was reached without a null element: the array is missing its terminator
            confirmed = _construct_array(array_type, entries)
            raise IncompleteRead(confirmed, stream.tell(), error, [index]) from error

        if not value:
            break
        entries.append(value)
        index += 1

    return _construct_array(array_type, entries)


def _read_terminated_string(
    array_type: type[BaseArray],
    stream: _BoundedReader,
    width: int,
    endian: Endianness,
) -> Any:
    """Read a char/wchar null-terminated array while preserving the confirmed prefix on truncation.

    The scan never crosses the bounded region. When the boundary is reached without a terminator, the
    bytes confirmed so far are returned through an :class:`IncompleteRead`.
    """
    terminator = b"\x00" * width

    if stream.seekable():
        chunks: list[bytes] = []
        while True:
            chunk = stream.read(128 * width)
            if not chunk:
                raw = b"".join(chunks)
                raise IncompleteRead(
                    _decode_string(array_type, raw, width, endian),
                    stream.tell(),
                    EOFError("Missing null terminator"),
                    [len(raw) // width],
                )

            idx = chunk.find(terminator)
            while idx != -1 and idx % width:
                idx = chunk.find(terminator, idx + 1)

            if idx != -1:
                chunks.append(chunk[:idx])
                # Put back the bytes after the terminator so a following field can read them
                stream.seek(stream.tell() - (len(chunk) - idx - width))
                return _decode_string(array_type, b"".join(chunks), width, endian)

            chunks.append(chunk)

    # Forward-only path for non-seekable streams: consume exactly one item at a time
    items: list[bytes] = []
    while True:
        item = stream.read(width)
        if len(item) != width:
            raw = b"".join(items)
            raise IncompleteRead(
                _decode_string(array_type, raw, width, endian),
                stream.tell(),
                EOFError("Missing null terminator"),
                [len(raw) // width],
            )
        if item == terminator:
            return _decode_string(array_type, b"".join(items), width, endian)
        items.append(item)


def _decode_string(array_type: type[BaseArray], raw: bytes, width: int, endian: Endianness) -> Any:
    if width == 1:
        return type.__call__(array_type, raw)
    return type.__call__(array_type, str(raw, Wchar.__encoding_map__[endian], "surrogatepass"))


def _construct_array(array_type: type[BaseArray], entries: list[Any] | bytes | str) -> Any:
    if issubclass(array_type, CharArray):
        data = entries if isinstance(entries, (bytes, bytearray)) else b"".join(entries)
        return type.__call__(array_type, data)
    if issubclass(array_type, WcharArray):
        data = entries if isinstance(entries, str) else "".join(entries)
        return type.__call__(array_type, data)
    return type.__call__(array_type, entries)


def _read_structure(
    struct_type: type[Structure],
    stream: _BoundedReader,
    context: dict[str, Any] | None,
    endian: Endianness,
) -> Any:
    if isinstance(struct_type, UnionMetaType):
        # A union reads every field from the same region; the bounded stream enforces the boundary
        try:
            return struct_type._read(stream, context=context, endian=endian)
        except EOFError as error:
            raise IncompleteRead({}, stream.tell(), error, []) from error

    struct_start = stream.tell()
    result: dict[str, Any] = {}
    sizes: dict[str, int] = {}
    bit_buffer: BitBuffer | None = None

    for field in struct_type.__fields__:
        offset = stream.tell()

        if field.offset is not None and offset != struct_start + field.offset:
            offset = struct_start + field.offset
            stream.seek(offset)

        if struct_type.__align__ and field.offset is None:
            offset += -offset & (field.alignment - 1)
            stream.seek(offset)

        if field.bits:
            if bit_buffer is None:
                bit_buffer = BitBuffer(stream, endian=endian)

            try:
                if isinstance(field.type, EnumMetaType):
                    value = field.type(bit_buffer.read(field.type.__type__, field.bits))
                else:
                    value = bit_buffer.read(field.type, field.bits)
            except EOFError as error:
                raise IncompleteRead(result, stream.tell(), error, [field._name]) from error

            result[field._name] = value
            continue

        if bit_buffer is not None:
            bit_buffer.reset()
            bit_buffer = None

        try:
            value = _read_value(field.type, stream, result, endian)
        except IncompleteRead as error:
            raise IncompleteRead(result, stream.tell(), error, [field._name, *error.path]) from error
        except EOFError as error:
            raise IncompleteRead(result, stream.tell(), error, [field._name]) from error

        result[field._name] = value
        if field.type.__dynamic__:
            sizes[field._name] = stream.tell() - offset

    if struct_type.__align__:
        stream.seek(-stream.tell() & (struct_type.__alignment__ - 1), io.SEEK_CUR)

    obj = type.__call__(struct_type, **result)
    obj.__dynamic_sizes__ = sizes
    return obj
