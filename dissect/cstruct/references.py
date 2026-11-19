from __future__ import annotations

import io
import json
import pprint
from typing import TYPE_CHECKING, Any, Literal

from dissect.cstruct.exception import NullPointerDereference
from dissect.cstruct.types.char import Char
from dissect.cstruct.types.pointer import Pointer
from dissect.cstruct.types.structure import Structure, Union, UnionProxy
from dissect.cstruct.types.void import Void

if TYPE_CHECKING:
    from typing import BinaryIO

    from dissect.cstruct.types.base import BaseType

Output = Literal["dict", "json", "print", "string"]


def dumpreferences(
    obj: Structure | type[Structure],
    data: bytes | None = None,
    offset: int = 0,
    *,
    max_depth: int = 32,
    output: Output = "dict",
) -> dict[str, Any] | str | None:
    """Dump the objects and pointer references of a structure or parsed structure instance.

    Structures are exported as a flat list of ``objects`` and a list of ``references`` between them. The root object
    is the exported structure itself. Every pointer that is reachable from the root, including pointers in nested
    structures, unions and arrays, results in a reference. A reference is followed until its target was already
    exported (a shared reference or a cycle), its target can't be read, or ``max_depth`` pointer hops were taken.
    An object is never expanded again once it was exported, so recursive structures such as linked lists that point
    back to themselves terminate.

    Objects are identified by their address *and* resolved type. Pointers of different types to the same address are
    separate objects with their own type, value and bounds; they are not merged into one object just because they
    share an address. A pointer to an object that is currently being expanded results in a ``cycle`` reference,
    while a pointer to an object that was exported before results in a ``shared`` reference.

    A reference has a ``status``:

    - ``resolved``: the pointer was dereferenced and points to an object in ``objects``.
    - ``null``: the pointer value is zero.
    - ``out_of_bounds``: the pointer value is outside the stream, or the target extends past the end of the stream.
      ``available`` is the amount of bytes from the pointer address to the end of the stream and ``required`` is the
      static size of the target type, when known.
    - ``unresolved``: the target could not be obtained, e.g. a ``void`` pointer, a pointer without a stream, a
      reading error other than reaching the end of the stream, or a ``max_depth`` limit. The ``reason`` explains why.

    The ``dict`` output has the following keys:
        - bytes: hex string of the raw data.
        - address: address of the root object.
        - size: size of the root object in bytes, or ``None`` if unknown.
        - bounds: ``[address, address + size]`` of the root object, or ``None`` if unknown.
        - root: id of the root object in ``objects``.
        - type: type name of the root object.
        - kind: ``struct`` or ``union``.
        - truncated: whether a reference was not followed because of ``max_depth``.
        - objects: list of objects. Every object has an ``id``, ``address``, ``type``, ``kind``, ``size`` and
          ``bounds``. Structures and unions also have ``fields``, like ``dumpstruct(output="dict")``. Other values
          have a human-readable ``display``. A pointer target that is itself a pointer also has its ``value``.
          ``size`` and ``bounds`` are ``None`` when the object has no effective boundary.
        - references: list of references. Every reference has a ``source`` object id, the ``path`` of the pointer in
          that object, the pointer ``address`` and the declared target ``type``. Resolved references have a
          ``target`` object id and a ``relation`` of ``reference``, ``shared`` or ``cycle``. Unresolved and
          out-of-bounds references have a ``reason`` and, when known, an ``available`` amount of bytes.

    Args:
        obj: :class:`~dissect.cstruct.types.structure.Structure` to dump.
        data: Bytes to parse the :class:`~dissect.cstruct.types.structure.Structure` on,
            if obj is not a parsed :class:`~dissect.cstruct.types.structure.Structure` already.
        offset: Address of the root object. Pointer values are relative to the start of the stream.
        max_depth: Maximum amount of pointer hops to follow from the root object.
        output: Output format, can be ``dict``, ``json``, ``print`` or ``string``.

    Returns:
        A dict when ``output="dict"``, a JSON string when ``output="json"`` or ``output="string"``,
        or ``None`` when ``output="print"``.
    """
    if output not in ("dict", "json", "print", "string"):
        raise ValueError(f"Invalid output argument: {output!r} (should be 'dict', 'json', 'print' or 'string').")

    if max_depth < 0:
        raise ValueError("max_depth must be zero or greater")

    if isinstance(obj, Structure):
        if data is None:
            data = _stream_data(obj) or obj.dumps()
    elif isinstance(obj, type) and issubclass(obj, Structure) and data is not None:
        if offset and offset <= len(data):
            stream = io.BytesIO(data)
            stream.seek(offset)
            obj = obj(stream)
        else:
            obj = obj(data)
    else:
        raise ValueError("Invalid arguments: expected a Structure instance, or a Structure class with data.")

    result = _ReferenceGraph(data, offset, max_depth).build(obj)

    if output == "dict":
        return result
    if output in ("json", "string"):
        return json.dumps(result, indent=4)
    if output == "print":
        print(json.dumps(result, indent=4))
        return None
    raise ValueError(f"Invalid output argument: {output!r} (should be 'dict', 'json', 'print' or 'string').")


dump_references = dumpreferences
dump_graph = dumpreferences
dumpgraph = dumpreferences
dump_objects = dumpreferences
dumpobjects = dumpreferences


class _ReferenceGraph:
    """Build the object and pointer reference graph of a parsed structure."""

    def __init__(self, data: bytes, offset: int, max_depth: int):
        self.data = data
        self.offset = offset
        self.max_depth = max_depth

        self.objects: list[dict[str, Any]] = []
        self.references: list[dict[str, Any]] = []
        self._by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._stack: set[tuple[Any, ...]] = set()
        self.truncated = False

    def build(self, root: Structure) -> dict[str, Any]:
        key = (self.offset, type(root))
        size = _object_size(root, self.data, self.offset)
        root_node = self._add_node(key, self.offset, type(root), root, size)

        self._stack.add(key)
        try:
            self._walk(root, root_node["id"], None, 0)
        finally:
            self._stack.discard(key)

        return {
            "bytes": self.data.hex(),
            "address": self.offset,
            "size": root_node["size"],
            "bounds": root_node["bounds"],
            "root": root_node["id"],
            "type": root.__class__.__name__,
            "kind": "union" if isinstance(root, Union) else "struct",
            "truncated": self.truncated,
            "objects": self.objects,
            "references": self.references,
        }

    def _add_node(
        self,
        key: tuple[Any, ...],
        address: int,
        type_: type[BaseType],
        value: Any,
        size: int | None,
    ) -> dict[str, Any]:
        node: dict[str, Any] = {
            "id": len(self.objects),
            "address": address,
            "type": type_.__name__,
            "kind": _kind(value),
            "size": size,
            "bounds": None if size is None else [address, address + size],
        }

        if isinstance(value, Structure):
            # Reuse the field descriptions of dumpstruct, with addresses instead of dump-relative offsets
            from dissect.cstruct.dump import _describe_fields

            node["fields"] = _describe_fields(value, address)
        else:
            display = _display(value)
            if display is not None:
                node["display"] = display
            if isinstance(value, Pointer):
                node["value"] = int(value)

        self.objects.append(node)
        self._by_key[key] = node
        return node

    def _walk(self, value: Structure, source: int, path: str | None, depth: int) -> None:
        for field in value.__class__.__fields__:
            # Of fields with a repeated name ("_"), a structure only keeps the value of the last one
            if value.__class__.__lookup__[field._name] is not field:
                continue

            self._walk_value(getattr(value, field._name), source, _join(path, field.name), depth)

    def _walk_value(self, value: Any, source: int, path: str | None, depth: int) -> None:
        if isinstance(value, UnionProxy):
            # Nested Structure members inside a Union are proxied; walk the real Structure
            value = value.__target__

        if isinstance(value, Pointer):
            self._reference(value, source, path, depth)
        elif isinstance(value, Structure):
            self._walk(value, source, path, depth)
        elif isinstance(value, (list, tuple)):
            for i, entry in enumerate(value):
                self._walk_value(entry, source, _join(path, f"[{i}]"), depth)

    def _reference(self, pointer: Pointer, source: int, path: str | None, depth: int) -> None:
        address = int(pointer)
        target_type = pointer.__type__
        reference: dict[str, Any] = {
            "source": source,
            "path": path,
            "address": address,
            "type": target_type.__name__,
        }

        if address == 0:
            reference["status"] = "null"
            self.references.append(reference)
            return

        if issubclass(target_type, Void):
            reference["status"] = "unresolved"
            reference["reason"] = "void"
            self.references.append(reference)
            return

        stream = pointer._Pointer__stream
        if stream is None:
            reference["status"] = "unresolved"
            reference["reason"] = "no_stream"
            self.references.append(reference)
            return

        if not _is_seekable(stream):
            # Without seek we can't read targets at arbitrary addresses from the stream
            reference["status"] = "unresolved"
            reference["reason"] = "not_seekable"
            self.references.append(reference)
            return

        # The same address and resolved type is the same object. Including the type keeps pointers of a
        # different type to the same address as separate objects, each with their own value and bounds.
        key = (address, target_type)

        # An object that is currently being expanded is a cycle, one that was exported before is shared.
        # Checking this before reading the target also prevents endless recursion on self-referencing objects.
        if key in self._stack:
            node = self._by_key[key]
            reference.update({"status": "resolved", "target": node["id"], "relation": "cycle"})
            self.references.append(reference)
            return

        if key in self._by_key:
            node = self._by_key[key]
            reference.update({"status": "resolved", "target": node["id"], "relation": "shared"})
            self.references.append(reference)
            return

        stream_size = _stream_size(stream)
        required = target_type.__size__

        if address < 0 or (stream_size is not None and address > stream_size):
            reference["status"] = "out_of_bounds"
            if stream_size is not None:
                reference["available"] = max(0, stream_size - max(0, address))
            if required is not None:
                reference["required"] = required
            self.references.append(reference)
            return

        if depth >= self.max_depth:
            reference["status"] = "unresolved"
            reference["reason"] = "max_depth"
            self.truncated = True
            self.references.append(reference)
            return

        try:
            value, size = self._read_target(pointer, address, target_type)
        except NullPointerDereference:
            reference["status"] = "null"
            self.references.append(reference)
            return
        except EOFError:
            reference["status"] = "out_of_bounds"
            if stream_size is not None:
                reference["available"] = max(0, stream_size - address)
            if required is not None:
                reference["required"] = required
            self.references.append(reference)
            return
        except Exception as e:
            reference["status"] = "unresolved"
            reference["reason"] = type(e).__name__
            self.references.append(reference)
            return

        if value is None:
            reference["status"] = "unresolved"
            reference["reason"] = "no_value"
            self.references.append(reference)
            return

        node = self._add_node(key, address, target_type, value, size)
        reference.update({"status": "resolved", "target": node["id"], "relation": "reference"})
        self.references.append(reference)

        self._stack.add(key)
        try:
            self._walk_target(value, node["id"], depth + 1)
        finally:
            self._stack.discard(key)

    def _read_target(self, pointer: Pointer, address: int, target_type: type[BaseType]) -> tuple[Any, int | None]:
        """Read the target of ``pointer`` from the dumped data.

        A union parses its members from a copy of its own bytes, so a pointer in a union would only see the bytes of
        the union itself. Reading from the dumped data instead makes a pointer always see its address in the stream.
        """
        stream = io.BytesIO(self.data)
        stream.seek(address)

        if issubclass(target_type, Char):
            # This makes the assumption that a char pointer is a null-terminated string
            value = target_type._read_0(stream, context=pointer._Pointer__context, endian=pointer._Pointer__endian)
        else:
            value = target_type._read(stream, context=pointer._Pointer__context, endian=pointer._Pointer__endian)

        return value, stream.tell() - address

    def _walk_target(self, value: Any, source: int, depth: int) -> None:
        if isinstance(value, Structure):
            self._walk(value, source, None, depth)
        elif isinstance(value, Pointer):
            # A pointer to a pointer: the target is the pointer itself, follow its value too
            self._reference(value, source, "*", depth)
        elif isinstance(value, (list, tuple)):
            for i, entry in enumerate(value):
                self._walk_value(entry, source, f"[{i}]", depth)


def _kind(value: Any) -> str:
    if isinstance(value, Union):
        return "union"
    if isinstance(value, Structure):
        return "struct"
    if isinstance(value, Pointer):
        return "pointer"
    if isinstance(value, (list, tuple)):
        return "array"
    return "value"


def _display(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, Structure):
        return None
    if isinstance(value, (str, Pointer)):
        return repr(value)
    if isinstance(value, int):
        return hex(value)
    if isinstance(value, (list, tuple)):
        return pprint.pformat(list(value))
    return str(value)


def _stream_data(obj: Structure) -> bytes | None:
    """Return the data of the stream that ``obj`` was parsed from, if any.

    Pointers refer to addresses in the stream, which can be longer than the bytes of the root structure itself.
    Structures and unions that were read from a larger stream can have a local stream of only their own bytes, so
    pick the longest available stream.
    """
    best = None
    best_size = -1

    for stream in _find_streams(obj):
        try:
            position = stream.tell()
            stream.seek(0)
            data = stream.read()
            stream.seek(position)
        except Exception:
            continue

        if isinstance(data, bytes) and len(data) > best_size:
            best = data
            best_size = len(data)

    return best


def _find_streams(obj: Structure, seen: set[int] | None = None) -> list[BinaryIO]:
    seen = seen or set()
    if id(obj) in seen:
        return []
    seen.add(id(obj))

    streams = []
    for field in obj.__class__.__fields__:
        value = getattr(obj, field._name, None)

        if isinstance(value, UnionProxy):
            value = value.__target__

        if isinstance(value, Pointer):
            if (stream := value._Pointer__stream) is not None:
                streams.append(stream)
        elif isinstance(value, Structure):
            streams.extend(_find_streams(value, seen))
        elif isinstance(value, (list, tuple)):
            for entry in value:
                if isinstance(entry, UnionProxy):
                    entry = entry.__target__
                if isinstance(entry, Pointer) and (stream := entry._Pointer__stream) is not None:
                    streams.append(stream)
                elif isinstance(entry, Structure):
                    streams.extend(_find_streams(entry, seen))

    return streams


def _object_size(obj: Structure, data: bytes, offset: int) -> int | None:
    size = obj.__class__.__size__
    if size is not None:
        return size

    try:
        return len(obj.dumps())
    except Exception:
        return None if offset > len(data) else len(data) - offset


def _join(path: str | None, name: str | None) -> str | None:
    if name is None:
        return path
    if not path:
        return name
    if name.startswith("["):
        return f"{path}{name}"
    return f"{path}.{name}"


def _stream_size(stream: BinaryIO) -> int | None:
    try:
        position = stream.tell()
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(position)
    except Exception:
        return None
    else:
        return size


def _is_seekable(stream: BinaryIO) -> bool:
    try:
        return stream.seekable()
    except Exception:
        return False
