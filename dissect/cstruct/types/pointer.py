from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any, BinaryIO, Generic, TypeVar

from dissect.cstruct.exception import AddressMappingError, NullPointerDereference
from dissect.cstruct.types.base import BaseType, normalize_endianness
from dissect.cstruct.types.char import Char
from dissect.cstruct.types.void import Void

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import Self

    from dissect.cstruct.addressspace import FileRange, MappedStream
    from dissect.cstruct.cstruct import AllowedEndianness, Endianness

T = TypeVar("T", bound=BaseType)


class Pointer(int, BaseType, Generic[T]):
    """Pointer to some other type."""

    __type__: type[T]
    __stream: BinaryIO | None
    __context: dict[str, Any] | None
    __endian: Endianness

    def __new__(
        cls, value: int, stream: BinaryIO | None, *, context: dict[str, Any] | None = None, endian: Endianness
    ) -> Self:
        obj = super().__new__(cls, value)
        obj.__stream = stream
        obj.__context = context
        obj.__endian = endian
        obj.dereference = cache(obj.dereference)
        return obj

    def __repr__(self) -> str:
        return f"<{self.__type__.__name__}* @ {self:#x}>"

    def __str__(self) -> str:
        return str(self.dereference())

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.dereference(), attr)

    @staticmethod
    def __op(op: Callable[[int, int], int]) -> Callable[[Self, int], Self]:
        def method(self: Self, other: int) -> Self:
            return type.__call__(
                self.__class__,
                op(self, other),
                self.__stream,
                context=self.__context,
                endian=self.__endian,
            )

        return method

    __add__ = __op(int.__add__)
    __sub__ = __op(int.__sub__)
    __mul__ = __op(int.__mul__)
    __floordiv__ = __op(int.__floordiv__)
    __mod__ = __op(int.__mod__)
    __pow__ = __op(int.__pow__)
    __lshift__ = __op(int.__lshift__)
    __rshift__ = __op(int.__rshift__)
    __and__ = __op(int.__and__)
    __xor__ = __op(int.__xor__)
    __or__ = __op(int.__or__)

    @classmethod
    def __default__(cls) -> Self:
        return cls.__new__(
            cls,
            cls.__cs__.pointer.__default__(),
            None,
            context=None,
            endian=cls.__cs__.endian,
        )

    @classmethod
    def _read(cls, stream: BinaryIO, *, context: dict[str, Any] | None = None, endian: Endianness) -> Self:
        return cls.__new__(
            cls,
            cls.__cs__.pointer._read(stream, context=context, endian=endian),
            stream,
            context=context,
            endian=endian,
        )

    @classmethod
    def _write(cls, stream: BinaryIO, data: int, *, endian: Endianness) -> int:
        return cls.__cs__.pointer._write(stream, data, endian=endian)

    def dereference(self, *, endian: AllowedEndianness | None = None) -> T | None:
        """Dereference the pointer and read the value it points to.

        When the pointer is backed by a :class:`~dissect.cstruct.addressspace.MappedStream`, virtual addresses are
        resolved through that image's mappings and objects already parsed at the same virtual address (shared
        targets and reference cycles) are returned as the same Python object.

        Args:
            endian: Optional endianness to use when reading the value.
                    If not provided, the endianness used when reading the pointer itself will be used.
        """
        if self == 0 or self.__stream is None:
            raise NullPointerDereference

        effective_endian = normalize_endianness(endian) if endian is not None else self.__endian
        if issubclass(self.__type__, Void):
            return None

        from dissect.cstruct.addressspace import MappedStream

        if isinstance(self.__stream, MappedStream):
            return self._dereference_mapped(self.__stream, effective_endian)
        return self._dereference_plain(self.__stream, effective_endian)

    def _dereference_plain(self, stream: BinaryIO, endian: Endianness) -> T:
        position = stream.tell()
        stream.seek(self)
        try:
            return self._read_target(stream, endian)
        finally:
            # Restore the stream position after reading the value
            stream.seek(position)

    def _dereference_mapped(self, stream: MappedStream, endian: Endianness) -> T:
        vaddr = int(self)
        if not stream.address_space.contains(vaddr):
            raise AddressMappingError(
                f"Pointer target {vaddr:#x} is not covered by any mapping"
                + (f" in image {stream.address_space.version!r}" if stream.address_space.version is not None else "")
            )

        found, cached = stream.cached_object(self.__type__, vaddr)
        if found:
            return cached

        position = stream.tell()
        stream.seek(vaddr)
        try:
            value = self._read_target(stream, endian)
        finally:
            stream.seek(position)

        stream.cache_object(self.__type__, vaddr, value)
        return value

    def _read_target(self, stream: BinaryIO, endian: Endianness) -> T:
        if issubclass(self.__type__, Char):
            # This makes the assumption that a char pointer is a null-terminated string
            return self.__type__._read_0(stream, context=self.__context, endian=endian)
        return self.__type__._read(stream, context=self.__context, endian=endian)

    def ranges(self, *, endian: AllowedEndianness | None = None) -> tuple[FileRange, ...]:
        """Dereference the pointer and return the actual file ranges the pointed-to object was read from.

        Each range identifies a backing stream, the file offset and size, and the mapping that supplied the bytes.
        An object that spans several mappings returns one range per mapping. Only works when the pointer was read from
        a :class:`~dissect.cstruct.addressspace.MappedStream`.

        Raises:
            NullPointerDereference: If the pointer is null or was not read from a stream.
            TypeError: If the pointer's stream has no address mappings.
        """
        from dissect.cstruct.addressspace import MappedStream

        if self == 0 or self.__stream is None:
            raise NullPointerDereference
        if not isinstance(self.__stream, MappedStream):
            raise TypeError("ranges() requires a pointer read from a MappedStream")
        if issubclass(self.__type__, Void):
            return ()

        stream: MappedStream = self.__stream
        effective_endian = normalize_endianness(endian) if endian is not None else self.__endian

        # Read the object again so the recorded ranges cover the full object, independent of the dereference cache
        marker = stream.mark()
        position = stream.tell()
        stream.seek(int(self))
        try:
            self._read_target(stream, effective_endian)
        finally:
            stream.seek(position)

        return stream.ranges_since(marker)
