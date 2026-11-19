from __future__ import annotations

import io
import warnings
from collections import ChainMap
from collections.abc import MutableMapping
from contextlib import contextmanager
from enum import Enum
from functools import lru_cache
from itertools import chain, repeat
from operator import attrgetter
from textwrap import dedent
from types import FunctionType
from typing import TYPE_CHECKING, Any, BinaryIO

from dissect.cstruct.bitbuffer import BitBuffer
from dissect.cstruct.expression import Expression
from dissect.cstruct.lexer import IDENTIFIER_TYPES
from dissect.cstruct.types.base import (
    _BUFFER_TYPES,
    _IMMUTABLE_TYPES,
    BaseArray,
    BaseType,
    MetaType,
    _is_buffer_type,
)
from dissect.cstruct.types.enum import EnumMetaType
from dissect.cstruct.types.flag import Flag
from dissect.cstruct.types.pointer import Pointer
from dissect.cstruct.util import ENDIANNESS_TO_BYTEORDER_MAP

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping
    from types import FunctionType

    from typing_extensions import Self

    from dissect.cstruct.cstruct import Endianness


class Field:
    """Structure field."""

    def __init__(self, name: str | None, type_: type[BaseType], bits: int | None = None, offset: int | None = None):
        self.name = name  # The name of the field, or None if anonymous
        self._name = name or type_.__name__  # The name of the field, or the type name if anonymous
        self.type = type_
        self.bits = bits
        self.offset = offset
        self.alignment = type_.__alignment__ or 1

    def __repr__(self) -> str:
        bits_str = f" : {self.bits}" if self.bits else ""
        return f"<Field {self.name} {self.type.__name__}{bits_str}>"


class StructureMetaType(MetaType):
    """Base metaclass for cstruct structure type classes."""

    __members__: dict[str, Field]
    """Mapping of field names to :class:`Field` objects, including "folded" fields from anonymous structures."""
    __lookup__: dict[str, Field]
    """Mapping of "raw" field names to :class:`Field` objects. E.g. holds the anonymous struct and not its fields."""
    __fields__: list[Field]
    """List of :class:`Field` objects for this structure. This is the structures' Single Source Of Truth."""

    # Internal
    __align__: bool
    __anonymous__: bool
    __updating__ = False
    __compiled__ = False
    __static_sizes__: dict[str, int]  # Cache of static sizes by field name

    def __new__(metacls, name: str, bases: tuple[type, ...], classdict: dict[str, Any], **kwargs) -> Self:  # type: ignore
        if (fields := classdict.pop("__members__", None)) is not None:
            metacls._update_fields(metacls, fields, align=classdict.get("__align__", False), classdict=classdict)

        return super().__new__(metacls, name, bases, classdict, **kwargs)

    def __call__(cls, *args, **kwargs) -> Self:  # type: ignore
        if len(args) == 1 and not kwargs:
            # Fast paths for the most common cases
            obj = args[0]
            if type(obj) in _BUFFER_TYPES:
                return cls._read(io.BytesIO(obj), endian=cls.__cs__.endian)

            if hasattr(obj, "read") and not isinstance(obj, cls):
                return cls._read(obj, endian=cls.__cs__.endian)

        if not args and not kwargs:
            obj = type.__call__(cls)
            object.__setattr__(obj, "__dynamic_sizes__", {})
            return obj

        return super().__call__(*args, **kwargs)

    def _update_fields(
        cls, fields: list[Field], align: bool = False, classdict: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        classdict = classdict or {}

        lookup = {}
        raw_lookup = {}
        field_names = []
        static_sizes = {}
        for field in fields:
            if field._name in lookup and field._name != "_":
                raise ValueError(f"Duplicate field name: {field._name}")

            if not field.type.__dynamic__:
                static_sizes[field._name] = field.type.__size__

            if isinstance(field.type, StructureMetaType) and field.name is None:
                for anon_field in field.type.__members__.values():
                    attr = f"{field._name}.{anon_field.name}"
                    classdict[anon_field.name] = property(attrgetter(attr), attrsetter(attr))

                lookup.update(field.type.__members__)
            else:
                lookup[field._name] = field

            raw_lookup[field._name] = field

        field_names = lookup.keys()
        classdict["__members__"] = lookup
        classdict["__lookup__"] = raw_lookup
        classdict["__fields__"] = fields
        classdict["__static_sizes__"] = static_sizes
        classdict["__bool__"] = _generate__bool__(field_names)

        if issubclass(cls, UnionMetaType) or isinstance(cls, UnionMetaType):
            classdict["__init__"] = _generate_union__init__(raw_lookup.values())
            # Not a great way to do this but it works for now
            classdict["__eq__"] = Union.__eq__
        else:
            classdict["__init__"] = _generate_structure__init__(raw_lookup.values())
            classdict["__eq__"] = _generate__eq__(field_names)

        classdict["__hash__"] = _generate__hash__(field_names)

        # If we're calling this as a class method or a function on the metaclass
        if issubclass(cls, type):
            size, alignment = cls._calculate_size_and_offsets(cls, fields, align)
        else:
            size, alignment = cls._calculate_size_and_offsets(fields, align)

        # TODO: generate cached_property for lazy reading

        classdict["__size__"] = size
        classdict["__alignment__"] = alignment
        classdict["__dynamic__"] = size is None

        return classdict

    def _calculate_size_and_offsets(cls, fields: list[Field], align: bool = False) -> tuple[int | None, int]:
        """Iterate all fields in this structure to calculate the field offsets and total structure size.

        If a structure has a dynamic field, further field offsets will be set to None and ``self.__dynamic__``
        will be set to ``True``.
        """
        # The current offset, set to None if we become dynamic
        offset = 0
        # The current alignment for this structure
        alignment = 0

        # The current bit field type
        bits_type = None
        # The offset of the current bit field, set to None if we become dynamic
        bits_field_offset = 0
        # How many bits we have left in the current bit field
        bits_remaining = 0

        for field in fields:
            if field.offset is not None:
                # If a field already has an offset, it's leading
                offset = field.offset

            if align and offset is not None:
                # Round to next alignment
                offset += -offset & (field.alignment - 1)

            # The alignment of this struct is equal to its largest members' alignment
            alignment = max(alignment, field.alignment)

            if field.bits:
                field_type = field.type

                if isinstance(field_type, EnumMetaType):
                    field_type = field_type.__type__

                # Bit fields have special logic
                if (
                    # Exhausted a bit field
                    bits_remaining == 0
                    # Moved to a bit field of another type, e.g. uint16 f1 : 8, uint32 f2 : 8;
                    or field_type != bits_type
                    # Still processing a bit field, but it's at a different offset due to alignment or a manual offset
                    or (None not in (offset, bits_field_offset) and offset > bits_field_offset + bits_type.__size__)
                ):
                    # ... if any of this is true, we have to move to the next field
                    bits_type = field_type
                    bits_count = bits_type.__size__ * 8
                    bits_remaining = bits_count
                    bits_field_offset = offset

                    if offset is not None:
                        # We're not dynamic, update the structure size and current offset
                        offset += bits_type.__size__

                    field.offset = bits_field_offset

                bits_remaining -= field.bits

                if bits_remaining < 0:
                    raise ValueError("Straddled bit fields are unsupported")
            else:
                # Reset bits stuff
                bits_type = None
                bits_field_offset = bits_remaining = 0

                field.offset = offset

                if offset is not None:
                    # We're not dynamic, update the structure size and current offset
                    try:
                        field_len = len(field.type)
                    except TypeError:
                        # This field is dynamic
                        offset = None
                        continue

                    offset += field_len

        if align and offset is not None:
            # Add "tail padding" if we need to align
            # This bit magic rounds up to the next alignment boundary
            # E.g. offset = 3; alignment = 8; -offset & (alignment - 1) = 5
            offset += -offset & (alignment - 1)

        # The structure size is whatever the currently calculated offset is
        return offset, alignment

    def _read(cls, stream: BinaryIO, *, context: dict[str, Any] | None = None, endian: Endianness) -> Self:  # type: ignore
        bit_buffer = None
        struct_start = stream.tell()

        result = {}
        sizes = {}
        for field in cls.__fields__:
            offset = stream.tell()

            if field.offset is not None and offset != struct_start + field.offset:
                # Field is at a specific offset, either aligned or added that way
                offset = struct_start + field.offset
                stream.seek(offset)

            if cls.__align__ and field.offset is None:
                # Previous field was dynamically sized and we need to align
                offset += -offset & (field.alignment - 1)
                stream.seek(offset)

            if field.bits:
                if bit_buffer is None:
                    bit_buffer = BitBuffer(stream, endian=endian)

                if isinstance(field.type, EnumMetaType):
                    value = field.type(bit_buffer.read(field.type.__type__, field.bits))
                else:
                    value = bit_buffer.read(field.type, field.bits)

                result[field._name] = value
                continue

            if bit_buffer is not None:
                bit_buffer.reset()

            value = field.type._read(stream, context=result, endian=endian)

            result[field._name] = value
            if field.type.__dynamic__:
                sizes[field._name] = stream.tell() - offset

        if cls.__align__:
            # Align the stream
            stream.seek(-stream.tell() & (cls.__alignment__ - 1), io.SEEK_CUR)

        # Using type.__call__ directly calls the __init__ method of the class
        # This is faster than calling cls() and bypasses the metaclass __call__ method
        obj = type.__call__(cls, **result)
        obj.__dynamic_sizes__ = sizes
        return obj

    def _read_0(cls, stream: BinaryIO, *, context: dict[str, Any] | None = None, endian: Endianness) -> list[Self]:  # type: ignore
        result = []

        while obj := cls._read(stream, context=context, endian=endian):
            result.append(obj)

        return result

    def _write(cls, stream: BinaryIO, data: Structure, *, endian: Endianness) -> int:
        bit_buffer = None
        struct_start = stream.tell()

        for field in cls.__fields__:
            field_type = field.type

            if bit_buffer is not None and bit_buffer._type is not None:
                bit_field_type = (
                    (field_type.__type__ if isinstance(field_type, EnumMetaType) else field_type)
                    if field.bits
                    else None
                )
                # Current field is not a bit field, but previous was
                # Or, moved to a bit field of another type, e.g. uint16 f1 : 8, uint32 f2 : 8;
                if bit_buffer._type != bit_field_type:
                    # Flush the current bit buffer so we can process alignment properly
                    bit_buffer.flush()

            offset = stream.tell()

            if field.offset is not None and offset < struct_start + field.offset:
                # Field is at a specific offset, either aligned or added that way
                stream.write(b"\x00" * (struct_start + field.offset - offset))
                offset = struct_start + field.offset

            if cls.__align__ and field.offset is None:
                bit_type = bit_buffer._type if bit_buffer is not None else None
                if bit_type is None or bit_buffer._remaining == 0 or bit_type != field_type:
                    # Previous field was dynamically sized and we need to align
                    align_pad = -offset & (field.alignment - 1)
                    stream.write(b"\x00" * align_pad)
                    offset += align_pad

            value = getattr(data, field._name)
            if value is None:
                value = field_type.__default__()

            if field.bits:
                if bit_buffer is None:
                    bit_buffer = BitBuffer(stream, endian=endian)

                if isinstance(field_type, EnumMetaType):
                    bit_buffer.write(field_type.__type__, value.value, field.bits)
                else:
                    bit_buffer.write(field_type, value, field.bits)
            else:
                field_type._write(stream, value, endian=endian)

        if bit_buffer is not None and bit_buffer._type is not None:
            bit_buffer.flush()

        if cls.__align__:
            # Align the stream
            stream.write(b"\x00" * (-stream.tell() & (cls.__alignment__ - 1)))

        return stream.tell() - struct_start

    def add_field(cls, name: str, type_: type[BaseType], bits: int | None = None, offset: int | None = None) -> None:
        field = Field(name, type_, bits=bits, offset=offset)
        cls.__fields__.append(field)

        if not cls.__updating__:
            cls.commit()

    @contextmanager
    def start_update(cls) -> Iterator[None]:
        try:
            cls.__updating__ = True
            yield
        finally:
            cls.commit()
            cls.__updating__ = False

    def commit(cls) -> None:
        classdict = cls._update_fields(cls.__fields__, cls.__align__)

        for key, value in classdict.items():
            setattr(cls, key, value)

        if cls.__compiled__:
            # If the previous class was compiled try to compile this too
            from dissect.cstruct import compiler

            compiler.compile_structure(cls)

    def cdef(cls, *, recursive: bool = False) -> str:
        """Render this structure (or union) back to its C-style definition.

        When ``recursive=True``, any ``#define`` constants referenced in array sizes are emitted as well, so that the
        result can be parsed on its own. Note that the result is semantically equivalent to the original definition,
        but not necessarily identical.

        Args:
            recursive: Also emit the definitions of any types referenced by this structure.

        Returns:
            The C-style definition as a string.
        """

        def _decompose(type_: MetaType) -> tuple[MetaType, int, list[int | Expression | None]]:
            dimensions = []
            while issubclass(type_, BaseArray):
                dimensions.append(type_.__num_entries__)
                type_ = type_.__type__

            stars = 0
            while issubclass(type_, Pointer):
                stars += 1
                type_ = type_.__type__

            return type_, stars, dimensions

        def _deps(type_: StructureMetaType, seen: set[MetaType]) -> list[MetaType | str]:
            deps = []
            for field in type_.__fields__:
                base, _, dimensions = _decompose(field.type)

                for dim in dimensions:
                    if isinstance(dim, Expression):
                        for token in dim._tokens:
                            if (
                                token.type in IDENTIFIER_TYPES
                                and token.value in cls.__cs__.consts
                                and token.value not in seen
                            ):
                                deps.append(token.value)
                                seen.add(token.value)

                if isinstance(base, StructureMetaType) and base not in seen:
                    if base.__anonymous__:
                        # Anonymous structures are not emitted, but their dependencies are
                        deps.extend(_deps(base, seen))
                    else:
                        seen.add(base)
                        deps.extend(_deps(base, seen))
                        deps.append(base)

                elif isinstance(base, EnumMetaType) and base not in seen:
                    seen.add(base)
                    deps.append(base)

            return deps

        def _render_struct(type_: StructureMetaType, name: str) -> list[str]:
            keyword = "union" if isinstance(type_, UnionMetaType) else "struct"
            title = f"{keyword} {name}" if name else keyword

            lines = [f"{title} {{"]
            for field in type_.__fields__:
                base, stars, dimensions = _decompose(field.type)

                declarator = "*" * stars + (field.name or "") + "".join(f"[{dim or ''}]" for dim in dimensions)
                if field.bits is not None:
                    declarator = f"{declarator} : {field.bits}" if declarator else f": {field.bits}"

                if isinstance(base, StructureMetaType) and base.__anonymous__:
                    # Anonymous struct/union: inline the full definition
                    lines.extend(f"    {line}" for line in _render_struct(base, ""))
                    lines[-1] = f"{lines[-1]} {declarator};" if declarator else f"{lines[-1]};"
                else:
                    prefix = f"{base.__name__} " if declarator else base.__name__
                    lines.append(f"    {prefix}{declarator};")

            lines.append("}")
            return lines

        deps = _deps(cls, {cls}) if recursive else []
        deps.append(cls)

        consts = []
        blocks = []
        for type_ in deps:
            if isinstance(type_, str):
                consts.append(f"#define {type_} {cls.__cs__.consts[type_]!r}")
            elif isinstance(type_, EnumMetaType):
                blocks.append(type_.cdef())
            else:
                blocks.append("\n".join(_render_struct(type_, type_.__name__)) + ";")

        return ("\n".join(consts) + "\n\n" if consts else "") + ("\n\n".join(blocks) if blocks else "")

    @property
    def fields(cls) -> dict[str, Field]:
        """List of :class:`Field` objects for this structure."""
        warnings.warn(
            "The 'fields' property is deprecated, use '__members__' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.__members__

    @property
    def lookup(cls) -> dict[str, Field]:
        """Mapping of "raw" field names to :class:`Field` objects."""
        warnings.warn(
            "The 'lookup' property is deprecated, use '__lookup__' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.__lookup__


class Structure(BaseType, metaclass=StructureMetaType):
    """Base class for cstruct structure type classes.

    Note that setting attributes which do not correspond to a field in the structure results in undefined behavior.
    For performance reasons, the structure does not check if the field exists when writing to an attribute.
    """

    __dynamic_sizes__: dict[str, int]

    def __len__(self) -> int:
        return len(self.dumps())

    def __bytes__(self) -> bytes:
        return self.dumps()

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def __repr__(self) -> str:
        values = []
        for name, field in self.__class__.__members__.items():
            value = self[name]
            if issubclass(field.type, int) and not issubclass(field.type, (Pointer, Enum)):
                value = hex(value)
            else:
                value = repr(value)
            values.append(f"{name}={value}")

        return f"<{self.__class__.__name__} {' '.join(values)}>"

    @property
    def __values__(self) -> MutableMapping[str, Any]:
        return StructureValuesProxy(self)

    @property
    def __sizes__(self) -> Mapping[str, int | None]:
        # Structures created from values instead of parsed have no dynamic sizes
        return ChainMap(self.__class__.__static_sizes__, getattr(self, "__dynamic_sizes__", {}))


class StructureValuesProxy(MutableMapping):
    """A proxy for the values of fields of a Structure."""

    def __init__(self, struct: Structure):
        self._struct: Structure = struct

    def __getitem__(self, key: str) -> Any:
        if key in self:
            return getattr(self._struct, key)
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self:
            return setattr(self._struct, key, value)
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return key in self._struct.__class__.__members__

    def __iter__(self) -> Iterator[str]:
        return iter(self._struct.__class__.__members__)

    def __len__(self) -> int:
        return len(self._struct.__class__.__members__)

    def __repr__(self) -> str:
        return repr(dict(self))

    def __delitem__(self, _: str):
        # Is abstract in base, but deleting is not supported.
        raise NotImplementedError("Cannot delete fields from a Structure")


class UnionMetaType(StructureMetaType):
    """Base metaclass for cstruct union type classes."""

    def __call__(cls, *args, **kwargs) -> Self:  # type: ignore
        obj: Union = super().__call__(*args, **kwargs)

        if not hasattr(obj, "_buf"):
            # If we don't have a _buf attribute, we haven't read from a stream and are initializing with values
            # Set default internal attributes
            object.__setattr__(obj, "_buf", None)
            object.__setattr__(obj, "_endian", cls.__cs__.endian)

        # Calling with non-stream args or kwargs means we are initializing with values
        if (args and not (len(args) == 1 and (hasattr(args[0], "read") or _is_buffer_type(args[0])))) or kwargs:
            # We don't support user initialization of dynamic unions yet
            if cls.__dynamic__:
                raise NotImplementedError("Initializing a dynamic union is not yet supported")

            # User (partial) initialization, rebuild the union
            # First user-provided field is the one used to rebuild the union
            arg_fields = (field._name for _, field in zip(args, cls.__fields__, strict=False))
            kwarg_fields = (name for name in kwargs if name in cls.__lookup__)
            if (first_field := next(chain(arg_fields, kwarg_fields), None)) is not None:
                obj._rebuild(first_field)
        elif not args and not kwargs:
            # Initialized with default values
            # Note that we proxify here in case we have a default initialization (cls())
            # We don't proxify in case we read from a stream, as we do that later on in _read at a more appropriate time
            # Same with (partial) user initialization, we do that after rebuilding the union
            obj._proxify()

        return obj

    def _calculate_size_and_offsets(cls, fields: list[Field], align: bool = False) -> tuple[int | None, int]:
        size = 0
        alignment = 0

        for field in fields:
            if size is not None:
                try:
                    size = max(len(field.type), size)
                except TypeError:
                    size = None

            alignment = max(field.alignment, alignment)

        if align and size is not None:
            # Add "tail padding" if we need to align
            # This bit magic rounds up to the next alignment boundary
            # E.g. offset = 3; alignment = 8; -offset & (alignment - 1) = 5
            size += -size & (alignment - 1)

        return size, alignment

    def _read_fields(
        cls, stream: BinaryIO, *, context: dict[str, Any] | None = None, endian: Endianness
    ) -> tuple[dict[str, Any], dict[str, int]]:
        result = {}
        sizes = {}

        if cls.__size__ is None:
            offset = stream.tell()
            buf = stream
        else:
            offset = 0
            buf = io.BytesIO(stream.read(cls.__size__))

        for field in cls.__fields__:
            field_type = field.type

            start = 0
            if field.offset is not None:
                start = field.offset

            buf.seek(offset + start)
            value = field_type._read(buf, context=result, endian=endian)

            result[field._name] = value
            if field.type.__dynamic__:
                sizes[field._name] = buf.tell() - start

        return result, sizes

    def _read(cls, stream: BinaryIO, *, context: dict[str, Any] | None = None, endian: Endianness) -> Self:  # type: ignore
        if cls.__size__ is None:
            start = stream.tell()
            result, sizes = cls._read_fields(stream, context=context, endian=endian)
            size = stream.tell() - start
            stream.seek(start)
            buf = stream.read(size)
        elif "__init__" in cls.__dict__ and cls.__new__ is object.__new__:
            # Fast path for static unions
            buf = stream.read(cls.__size__)
            result, sizes = cls._read_fields(io.BytesIO(buf), endian=endian)

            obj: Union = object.__new__(cls)
            obj.__dict__.update(result)
            object.__setattr__(obj, "__dynamic_sizes__", sizes)
            object.__setattr__(obj, "_buf", buf)
            object.__setattr__(obj, "_endian", endian)
            obj._proxify()
            return obj
        else:
            result = {}
            sizes = {}
            buf = stream.read(cls.__size__)

        # Create the object and set the values
        # Using type.__call__ directly calls the __init__ method of the class
        # This is faster than calling cls() and bypasses the metaclass __call__ method
        # It also makes it easier to differentiate between user-initialization of the class
        # and initialization from a stream read
        obj: Union = type.__call__(cls, **result)
        object.__setattr__(obj, "__dynamic_sizes__", sizes)
        object.__setattr__(obj, "_buf", buf)
        object.__setattr__(obj, "_endian", endian)

        if cls.__size__ is not None:
            obj._update()

        # Proxify any nested structures
        obj._proxify()

        return obj

    def _write_field(cls) -> Field | None:
        """Return the field to write a union by, its largest field and preferably not an anonymous structure."""
        return max(cls.__fields__, key=lambda e: (e.type.__size__ or 0, e.name is not None), default=None)

    def _write(cls, stream: BinaryIO, data: Union, *, endian: Endianness) -> int:
        if cls.__dynamic__:
            raise NotImplementedError("Writing dynamic unions is not yet supported")

        original = getattr(data, "_buf", None)
        if original is None or endian != getattr(data, "_endian", endian):
            # Not parsed from bytes, or written in another endianness: write by the largest field
            offset = stream.tell()
            expected_offset = offset + len(cls)

            if (field := cls._write_field()) is not None:
                value = getattr(data, field._name)
                if value is None:
                    value = field.type.__default__()
                field.type._write(stream, value, endian=endian)

            # If we haven't filled the union size yet, pad it
            if remaining := expected_offset - stream.tell():
                stream.write(b"\x00" * remaining)

            return stream.tell() - offset

        # Start from the bytes the union was parsed from. For every member, only the bytes that differ from what the
        # original bytes would serialize to are written. This preserves the bytes of members that weren't changed,
        # including their padding, instead of clearing them with default values.
        buf = bytearray(original.ljust(cls.__size__, b"\x00")[: cls.__size__])

        # Remember the members that were changed, before the union is rebuilt from the new bytes
        for field in cls.__fields__:
            # Of fields with a repeated name ("_"), a union only keeps the value of the last one
            if cls.__lookup__[field._name] is not field or field.bits:
                # Bit fields patch their own bits into the parsed bytes when they change
                continue

            value = getattr(data, field._name)
            if value is None:
                continue

            start = field.offset or 0
            try:
                patch = field.type.dumps(value, endian=endian)
                baseline = field.type.dumps(
                    field.type._read(io.BytesIO(original[start : start + field.type.__size__]), endian=endian),
                    endian=endian,
                )
            except (AttributeError, TypeError, EOFError):
                # A member without a complete value can't be serialized, keep its parsed bytes
                continue

            end = min(start + len(patch), len(buf), start + len(baseline))
            for i in range(end - start):
                if patch[i] != baseline[i]:
                    buf[start + i] = patch[i]

        stream.write(bytes(buf))
        return cls.__size__


class Union(Structure, metaclass=UnionMetaType):
    """Base class for cstruct union type classes."""

    _buf: bytes
    _endian: Endianness

    def __eq__(self, other: object) -> bool:
        return self.__class__ is other.__class__ and bytes(self) == bytes(other)

    def __setattr__(self, attr: str, value: Any) -> None:
        if self.__class__.__dynamic__:
            raise NotImplementedError("Modifying a dynamic union is not yet supported")

        super().__setattr__(attr, value)
        # Fields of an anonymous structure are rebuilt through the proxy of that structure
        if attr in self.__class__.__lookup__:
            self._rebuild(attr)

    def select(self, key: str | None = None, *, value: Any = None) -> UnionView | UnionCandidates:
        """Select the union member named after a discriminant value.

        The discriminant is the member called ``key``, e.g. a kind or tag field. Its value is looked up in the
        union: a member name for enum and flag values, the string value for char values and the integer value for
        other integers. If the discriminant value does not match a member, the candidate member names are returned
        instead of reading all members.

        Args:
            key: The name of the discriminant member of the union. When omitted, the first member called ``kind``,
                ``type``, ``tag``, ``discriminator`` or ``selector`` is used.
            value: Optional discriminant value to select with, instead of the value of the member.
        """
        if key is None:
            key = next(
                (
                    candidate
                    for candidate in ("kind", "type", "tag", "discriminator", "selector")
                    if candidate in self.__class__.__lookup__
                ),
                None,
            )
            if key is None:
                candidates = [field._name for field in self.__class__.__fields__]
                return UnionCandidates(self, candidates)

        if key not in self.__class__.__lookup__:
            raise KeyError(f"{self.__class__.__name__!r} has no discriminant member {key!r}")

        discriminant = getattr(self, key) if value is None else value
        if isinstance(discriminant, (Enum, Flag)) and discriminant.name is not None:
            names = [discriminant.name]
        elif isinstance(discriminant, bytes):
            names = [discriminant.decode("utf-8", "surrogateescape").rstrip("\x00")]
        elif isinstance(discriminant, str):
            names = [discriminant.rstrip("\x00")]
        else:
            # Integer discriminants can name members by their decimal value, e.g. ``Kind`` values or ``case 1:``
            names = [str(int(discriminant)), hex(int(discriminant))]

        return self.resolve(key, names, value=discriminant)

    def view(self, key: str | None = None, *, value: Any = None) -> UnionView | UnionCandidates:
        """Return the view of the union member selected by the discriminant, see :meth:`select`."""
        return self.select(key, value=value)

    def resolve(self, key: str, names: str | list[str], *, value: Any = None) -> UnionView | UnionCandidates:
        """Select the union member called one of ``names``, or return the candidates if there is no such member.

        Args:
            key: The name of the discriminant member.
            names: The member name(s) the discriminant value points at.
            value: The discriminant value, defaults to the value of the discriminant member.
        """
        if value is None:
            value = getattr(self, key)

        if isinstance(names, str):
            names = [names]

        lookup = self.__class__.__lookup__
        for name in names:
            if name in lookup and name != key:
                return UnionView(self, name, discriminant=value, key=key)

        # The member that holds the discriminant itself is not a candidate
        candidates = [field._name for field in self.__class__.__fields__ if field._name != key]
        return UnionCandidates(self, candidates, discriminant=value, key=key)

    def export(self, key: str | None = None, *, value: Any = None) -> dict[str, Any]:
        """Export the union and its discriminant-selected member to a structured dictionary.

        The export always carries the original bytes of the union, so the members that weren't selected are
        preserved. If the discriminant value is unknown, the candidate members are listed instead.

        Args:
            key: The name of the discriminant member of the union.
            value: Optional discriminant value to select with, instead of the value of the member.
        """
        selected = self.select(key, value=value)
        node: dict[str, Any] = {
            "kind": "union",
            "type": self.__class__.__name__,
            "bytes": bytes(self).hex(),
        }

        if isinstance(selected, UnionCandidates):
            node["candidates"] = selected.candidates
            if selected.key is not None:
                node["discriminant"] = {"name": selected.key, "value": _export_value(selected.discriminant)}
            return node

        node["selected"] = selected.name
        node["field"] = selected.to_dict()
        if selected.key is not None:
            node["discriminant"] = {"name": selected.key, "value": _export_value(selected.discriminant)}
        return node

    def to_dict(self, key: str | None = None, *, value: Any = None) -> dict[str, Any]:
        """Alias of :meth:`export`."""
        return self.export(key, value=value)

    def candidates(self, key: str | None = None, *, value: Any = None) -> list[str]:
        """Return the names of the members that the discriminant could select, see :meth:`select`."""
        selected = self.select(key, value=value)
        if isinstance(selected, UnionCandidates):
            return selected.candidates
        return [field._name for field in self.__class__.__fields__ if field._name != selected.key]

    def _rebuild(self, attr: str) -> None:
        if (cur_buf := getattr(self, "_buf", None)) is None:
            cur_buf = b"\x00" * self.__class__.__size__

        buf = io.BytesIO(cur_buf)
        field = self.__class__.__lookup__[attr]
        if field.offset:
            buf.seek(field.offset)

        if (value := getattr(self, attr)) is None:
            value = field.type.__default__()

        if field.bits:
            # Only patch the bits of this field in its storage unit, so the other fields sharing the unit keep
            # their value instead of being overwritten by the zeroed default of the storage type.
            self._write_bits(buf, field, value)
        else:
            field.type._write(buf, value, endian=self._endian)

        object.__setattr__(self, "_buf", buf.getvalue())
        self._update()

        # (Re-)proxify all values
        self._proxify()

    def _write_bits(self, buf: BinaryIO, field: Field, value: Any) -> None:
        """Write the value of a bit field to its storage unit, keeping the bits of the other fields."""
        field_type = field.type
        if isinstance(field_type, EnumMetaType):
            field_type = field_type.__type__

        unit_size = field_type.__size__
        unit_bits = unit_size * 8
        unit_offset, bit_offset = _bit_field_offset(self.__class__, field)

        buf.seek(unit_offset)
        data = int.from_bytes(buf.read(unit_size), ENDIANNESS_TO_BYTEORDER_MAP[self._endian])

        mask = (1 << field.bits) - 1
        # Little endian fills the storage unit from the least significant bit, big endian from the most significant
        shift = bit_offset if self._endian == "<" else unit_bits - bit_offset - field.bits
        data = (data & ~(mask << shift)) | ((int(value) & mask) << shift)

        buf.seek(unit_offset)
        buf.write(data.to_bytes(unit_size, ENDIANNESS_TO_BYTEORDER_MAP[self._endian]))

    def _update(self) -> None:
        result, sizes = self.__class__._read_fields(io.BytesIO(self._buf), endian=self._endian)
        self.__dict__.update(result)
        object.__setattr__(self, "__dynamic_sizes__", sizes)

    def _proxify(self) -> None:
        def _proxy_structure(value: Structure, attr: str | None = None) -> None:
            for field in value.__class__.__fields__:
                if issubclass(field.type, Structure):
                    nested_value = getattr(value, field._name)
                    # Changing a nested field rebuilds the union from the top level field it's in
                    member = attr or field._name
                    object.__setattr__(value, field._name, UnionProxy(self, member, nested_value))
                    # The value of a nested union is already a proxy, so continue with the value behind it
                    if isinstance(nested_value, UnionProxy):
                        nested_value = nested_value.__target__
                    _proxy_structure(nested_value, member)

        _proxy_structure(self)


class UnionProxy:
    __union__: Union
    __attr__: str
    __target__: Structure

    def __init__(self, union: Union, attr: str, target: Structure):
        object.__setattr__(self, "__union__", union)
        object.__setattr__(self, "__attr__", attr)
        object.__setattr__(self, "__target__", target)

    def __len__(self) -> int:
        return len(self.__target__.dumps())

    def __bytes__(self) -> bytes:
        return self.__target__.dumps()

    def __getitem__(self, item: str) -> Any:
        return getattr(self.__target__, item)

    def __repr__(self) -> str:
        return repr(self.__target__)

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.__target__, attr)

    def __setattr__(self, attr: str, value: Any) -> None:
        setattr(self.__target__, attr, value)
        self.__union__._rebuild(self.__attr__)


class UnionView:
    """A view of one selected member of a :class:`Union`.

    A union stores all its members in the same bytes. Business logic usually only wants the member selected by a
    discriminant (e.g. an enum member name matching the member name), while the bytes of the other members must be
    preserved when writing back. This view keeps a reference to the union and its original bytes, and exposes only
    the selected member.

    Args:
        union: The union to view.
        field: The name of the selected union member.
        discriminant: The value of the discriminant that selected this view.
        key: The name of the discriminant member, if any.
    """

    __slots__ = ("__union__", "__field__", "__key__", "__discriminant__", "__target__")

    def __init__(
        self,
        union: Union,
        field: str,
        *,
        discriminant: Any = None,
        key: str | None = None,
    ):
        object.__setattr__(self, "__union__", union)
        object.__setattr__(self, "__field__", field)
        object.__setattr__(self, "__key__", key)
        object.__setattr__(self, "__discriminant__", discriminant)
        object.__setattr__(self, "__target__", getattr(union, field))

    @property
    def name(self) -> str:
        """The name of the selected union member."""
        return self.__field__

    @property
    def field(self) -> Field:
        """The :class:`Field` of the selected union member."""
        return self.__union__.__class__.__lookup__[self.__field__]

    @property
    def type(self) -> type[BaseType]:
        """The type of the selected union member."""
        return self.field.type

    @property
    def value(self) -> Any:
        """The value of the selected union member."""
        return getattr(self.__union__, self.__field__)

    @value.setter
    def value(self, value: Any) -> None:
        setattr(self.__union__, self.__field__, value)

    @property
    def offset(self) -> int:
        """The offset of the selected member relative to the start of the union."""
        return self.field.offset or 0

    @property
    def size(self) -> int | None:
        """The size of the selected member."""
        return self.field.type.__size__

    @property
    def discriminant(self) -> Any:
        """The discriminant value that selected this view."""
        return self.__discriminant__

    @property
    def key(self) -> str | None:
        """The name of the discriminant member, if any."""
        return self.__key__

    @property
    def raw(self) -> bytes:
        """The bytes of the whole union, including the members that weren't selected."""
        return bytes(self.__union__)

    @property
    def bytes(self) -> bytes:
        """The bytes of the whole union, including the members that weren't selected."""
        return self.raw

    @property
    def data(self) -> bytes:
        """The bytes of the whole union, including the members that weren't selected."""
        return self.raw

    def __len__(self) -> int:
        return self.size or 0

    def __bytes__(self) -> bytes:
        return self.raw

    def __getitem__(self, item: str) -> Any:
        if item in ("raw", "data"):
            return self.raw
        if item in ("name", "type", "value", "offset", "size", "discriminant", "key"):
            return getattr(self, item)
        if item in ("bytes",):
            return self.bytes
        return getattr(self.__target__, item)

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.__target__, attr)

    def __setattr__(self, attr: str, value: Any) -> None:
        if attr == "value":
            # A scalar member (e.g. an integer) can't hold attributes, assign it through the union
            setattr(self.__union__, self.__field__, value)
            return

        setattr(self.__target__, attr, value)
        # Only patch the selected member of the original bytes, leaving the other members intact
        self.__union__._rebuild(self.__field__)

    def __repr__(self) -> str:
        return repr(self.__target__)

    def to_dict(self) -> dict[str, Any]:
        """Export this view to a structured dictionary, including the original union bytes."""
        node: dict[str, Any] = {
            "name": self.__field__,
            "type": self.type.__name__,
            "offset": self.offset,
            "size": self.size,
            "bytes": self.raw.hex(),
            "value": _export_value(self.value),
        }
        if isinstance(self.__target__, Structure):
            node["fields"] = _export_fields(self.__target__)
        if self.__key__ is not None:
            node["discriminant"] = {"name": self.__key__, "value": _export_value(self.__discriminant__)}
        return node


class UnionCandidates(list):
    """The candidate members of a union whose discriminant value is unknown.

    This is returned instead of selecting a member, so that callers can inspect the possible views without parsing
    every branch of the union at once.
    """

    __slots__ = ("__union__", "__key__", "__discriminant__")

    def __init__(self, union: Union, candidates: list[str], *, discriminant: Any = None, key: str | None = None):
        super().__init__(candidates)
        object.__setattr__(self, "__union__", union)
        object.__setattr__(self, "__key__", key)
        object.__setattr__(self, "__discriminant__", discriminant)

    @property
    def candidates(self) -> list[str]:
        """The names of the union members that could be selected."""
        return list(self)

    @property
    def discriminant(self) -> Any:
        """The unknown discriminant value."""
        return self.__discriminant__

    @property
    def key(self) -> str | None:
        """The name of the discriminant member, if any."""
        return self.__key__

    @property
    def raw(self) -> bytes:
        """The bytes of the whole union."""
        return bytes(self.__union__)

    @property
    def bytes(self) -> bytes:
        """The bytes of the whole union."""
        return self.raw

    def to_dict(self) -> dict[str, Any]:
        """Export the candidates and the original union bytes."""
        node: dict[str, Any] = {"candidates": list(self), "bytes": self.raw.hex()}
        if self.__key__ is not None:
            node["discriminant"] = {"name": self.__key__, "value": _export_value(self.__discriminant__)}
        return node


def _export_fields(value: Structure) -> list[dict[str, Any]]:
    """Export the fields of a structure as a flat list of ``name``, ``type`` and ``value`` nodes."""
    return [
        {"name": field._name, "type": field.type.__name__, "value": _export_value(getattr(value, field._name))}
        for field in value.__class__.__fields__
        if value.__class__.__lookup__[field._name] is field
    ]


def _export_value(value: Any) -> Any:
    """Convert a value to a JSON-friendly representation for structured exports."""
    if isinstance(value, bytes):
        return {"type": "bytes", "value": value.hex()}
    if isinstance(value, Enum):
        return {"name": value.name, "value": value.value}
    if isinstance(value, Structure):
        return {name: _export_value(getattr(value, name)) for name in value.__class__.__members__}
    if isinstance(value, list):
        return [_export_value(item) for item in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return repr(value)


def _bit_field_offset(type_: type[Structure], field: Field) -> tuple[int, int]:
    """Return the offset of the storage unit of ``field`` and the position of ``field`` in its bits."""
    field_type = None
    bit_offset = 0
    for member in type_.__fields__:
        member_type = member.type.__type__ if isinstance(member.type, EnumMetaType) else member.type
        if bit_offset == 0 or member_type is not field_type or member.offset != field.offset:
            field_type = member_type
            bit_offset = 0

        if member is field:
            return field.offset or 0, bit_offset

        bit_offset += member.bits or 0

    raise ValueError(f"Field {field._name!r} is not a field of {type_.__name__}")


def attrsetter(path: str) -> Callable[[Any], Any]:
    path, _, attr = path.rpartition(".")
    path = path.split(".")

    def _func(obj: Any, value: Any) -> Any:
        for name in path:
            obj = getattr(obj, name)
        setattr(obj, attr, value)

    return _func


def _codegen(func: FunctionType) -> FunctionType:
    """Decorator that generates a template function with a specified number of fields.

    This code is a little complex but allows use to cache generated functions for a specific number of fields.
    For example, if we generate a structure with 10 fields, we can cache the generated code for that structure.
    We can then reuse that code and patch it with the correct field names when we create a new structure with 10 fields.

    The functions that are decorated with this decorator should take a list of field names and return a string of code.
    The decorated function is needs to be called with the number of fields, instead of the field names.
    The confusing part is that that the original function takes field names, but you then call it with
    the number of fields instead.

    Inspired by https://github.com/dabeaz/dataklasses.

    Args:
        func: The decorated function that takes a list of field names and returns a string of code.

    Returns:
        A cached function that generates the desired function code, to be called with the number of fields.
    """

    def make_func_code(num_fields: int) -> FunctionType:
        exec(func([f"_{n}" for n in range(num_fields)]), {}, d := {})
        return d.popitem()[1]

    make_func_code.__wrapped__ = func
    return lru_cache(make_func_code)


@_codegen
def _make_structure__init__(fields: list[str]) -> str:
    """Generates an ``__init__`` method for a structure with the specified fields.

    Args:
        fields: List of field names.
    """
    field_args = ", ".join(f"{field} = None" for field in fields)
    field_init = "\n".join(
        f" self.{name} = {name} if {name} is not None else _{i}_default()" for i, name in enumerate(fields)
    )

    code = f"def __init__(self{', ' + field_args or ''}):\n"
    return code + (field_init or " pass")


@_codegen
def _make_union__init__(fields: list[str]) -> str:
    """Generates an ``__init__`` method for a class with the specified fields using setattr.

    Args:
        fields: List of field names.
    """
    field_args = ", ".join(f"{field} = None" for field in fields)
    field_init = "\n".join(
        f" object.__setattr__(self, '{name}', {name} if {name} is not None else _{i}_default())"
        for i, name in enumerate(fields)
    )

    code = f"def __init__(self{', ' + field_args or ''}):\n"
    return code + (field_init or " pass")


@_codegen
def _make__eq__(fields: list[str]) -> str:
    """Generates an ``__eq__`` method for a class with the specified fields.

    Args:
        fields: List of field names.
    """
    self_vals = ",".join(f"self.{name}" for name in fields)
    other_vals = ",".join(f"other.{name}" for name in fields)

    if self_vals:
        self_vals += ","
    if other_vals:
        other_vals += ","

    # In the future this could be a looser check, e.g. an __eq__ on the classes, which compares the fields
    code = f"""
    def __eq__(self, other):
        if self.__class__ is other.__class__:
            return ({self_vals}) == ({other_vals})
        return False
    """

    return dedent(code)


@_codegen
def _make__bool__(fields: list[str]) -> str:
    """Generates a ``__bool__`` method for a class with the specified fields.

    Args:
        fields: List of field names.
    """
    vals = ", ".join(f"self.{name}" for name in fields)

    code = f"""
    def __bool__(self):
        return any([{vals}])
    """

    return dedent(code)


@_codegen
def _make__hash__(fields: list[str]) -> str:
    """Generates a ``__hash__`` method for a class with the specified fields.

    Args:
        fields: List of field names.
    """
    vals = ", ".join(f"self.{name}" for name in fields)

    code = f"""
    def __hash__(self):
        return hash(({vals}))
    """

    return dedent(code)


def _patch_attributes(func: FunctionType, fields: list[str], start: int = 0) -> FunctionType:
    """Patches a function's attributes.

    Args:
        func: The function to patch.
        fields: List of field names to add.
        start: The starting index for patching. Defaults to 0.
    """
    return type(func)(
        func.__code__.replace(co_names=(*func.__code__.co_names[:start], *fields)),
        func.__globals__,
    )


def _generate_structure__init__(fields: list[Field]) -> FunctionType:
    """Generates an ``__init__`` method for a structure with the specified fields.

    Args:
        fields: List of field names.
    """
    mapping = _generate_co_mapping(fields)

    template: FunctionType = _make_structure__init__(len(fields))
    return type(template)(
        template.__code__.replace(
            co_names=_remap_co_values(template.__code__.co_names, mapping),
            co_varnames=_remap_co_values(template.__code__.co_varnames, mapping),
        ),
        template.__globals__ | {f"__{field._name}_default__": _default_factory(field.type) for field in fields},
        argdefs=template.__defaults__,
    )


def _generate_union__init__(fields: list[Field]) -> FunctionType:
    """Generates an ``__init__`` method for a union with the specified fields.

    Args:
        fields: List of field names.
    """
    mapping = _generate_co_mapping(fields)

    template: FunctionType = _make_union__init__(len(fields))
    return type(template)(
        template.__code__.replace(
            co_consts=_remap_co_values(template.__code__.co_consts, mapping),
            co_names=_remap_co_values(template.__code__.co_names, mapping),
            co_varnames=_remap_co_values(template.__code__.co_varnames, mapping),
        ),
        template.__globals__ | {f"__{field._name}_default__": _default_factory(field.type) for field in fields},
        argdefs=template.__defaults__,
    )


def _default_factory(type_: type[BaseType]) -> Callable[[], Any]:
    """Return a function that returns default values of a type, which are only shared if they're immutable."""
    value = type_.__default__()
    return repeat(value).__next__ if isinstance(value, _IMMUTABLE_TYPES) else type_.__default__


def _generate_co_mapping(fields: list[Field]) -> dict[str, str]:
    """Generates a mapping of generated code object names to field names.

    The generated code uses names like ``_0``, ``_1``, etc. for fields, and ``_0_default``, ``_1_default``, etc.
    for default initializer values. Return a mapping of these names to the actual field names.

    Args:
        fields: List of field names.
    """
    return {
        key: value
        for i, field in enumerate(fields)
        for key, value in [(f"_{i}", field._name), (f"_{i}_default", f"__{field._name}_default__")]
    }


def _remap_co_values(value: tuple[Any, ...], mapping: dict[str, str]) -> tuple[Any, ...]:
    """Remaps code object values using a mapping.

    This is used to replace generated code object names with actual field names.

    Args:
        value: The original code object values.
        mapping: A mapping of generated code object names to field names.
    """
    # Only attempt to remap if the value is a string, otherwise return it as is
    # This is to avoid issues with trying to remap non-hashable types, and we only need to replace strings anyway
    return tuple(mapping.get(v, v) if isinstance(v, str) else v for v in value)


def _generate__eq__(fields: list[str]) -> FunctionType:
    """Generates an ``__eq__`` method for a class with the specified fields.

    Args:
        fields: List of field names.
    """
    return _patch_attributes(_make__eq__(len(fields)), fields, 1)


def _generate__bool__(fields: list[str]) -> FunctionType:
    """Generates a ``__bool__`` method for a class with the specified fields.

    Args:
        fields: List of field names.
    """
    return _patch_attributes(_make__bool__(len(fields)), fields, 1)


def _generate__hash__(fields: list[str]) -> FunctionType:
    """Generates a ``__hash__`` method for a class with the specified fields.

    Args:
        fields: List of field names.
    """
    return _patch_attributes(_make__hash__(len(fields)), fields, 1)
