from __future__ import annotations

from io import SEEK_CUR, SEEK_END, SEEK_SET
from typing import TYPE_CHECKING, Any, BinaryIO

if TYPE_CHECKING:
    from collections.abc import Iterator


class IncompleteRead(EOFError):
    """Raised when a record or array could only be parsed partially.

    The parser never reads past the boundary of the payload it was given. When a (possibly corrupted)
    header or element count cannot be satisfied within that boundary, the values that were fully
    confirmed before the failure are preserved instead of being discarded.

    Attributes:
        partial: The values that were fully read before the failure. A ``list`` for arrays and a
            ``dict`` of field names to values for structures.
        offset: The byte position in the outermost stream at which reading stopped.
        error: The original exception that terminated the read, possibly another ``IncompleteRead``
            from a nested layer.
        path: The field names and array indices leading from the outermost record to the failing
            element, e.g. ``["records", 1, "values"]``.
    """

    def __init__(
        self,
        partial: Any,
        offset: int,
        error: BaseException,
        path: list[str | int] | None = None,
    ) -> None:
        self.partial = partial
        self.offset = offset
        self.error = error
        self.path = path or []
        super().__init__(str(error))

    def __iter__(self) -> Iterator[Any]:
        return iter(self.partial or [])

    def __len__(self) -> int:
        return 0 if self.partial is None else len(self.partial)

    def __getitem__(self, key: Any) -> Any:
        if self.partial is None:
            raise KeyError(key)
        return self.partial[key]

    def __repr__(self) -> str:
        confirmed = 0 if self.partial is None else len(self.partial)
        return (
            f"<IncompleteRead path={self.path!r} offset={self.offset} "
            f"confirmed={confirmed} error={self.error!r}>"
        )


class _BoundedReader:
    """A file-like wrapper that guarantees reads never cross a payload boundary.

    Every :meth:`read` is clamped to ``limit``, an absolute position in the underlying stream. A read
    at the limit returns an empty ``bytes`` object instead of touching subsequent data, so a corrupted
    element count can never make an inner record read through its parent's payload.
    """

    def __init__(self, stream: BinaryIO, limit: int | None = None) -> None:
        self._stream = stream
        self._limit = limit
        self._seekable = _safe_seekable(stream)
        self._position = stream.tell() if self._seekable else 0

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            if self._limit is None:
                data = self._stream.read(-1)
            else:
                data = self._stream.read(max(0, self._limit - self._position))
        else:
            if self._limit is not None:
                size = min(size, max(0, self._limit - self._position))
            data = self._stream.read(size)

        self._position += len(data)
        return data

    def tell(self) -> int:
        return self._position

    def seekable(self) -> bool:
        return self._seekable

    def seek(self, offset: int, whence: int = SEEK_SET) -> int:
        if whence == SEEK_SET:
            target = offset
        elif whence == SEEK_CUR:
            target = self._position + offset
        elif whence == SEEK_END:
            if self._limit is None:
                target = self._stream.seek(offset, SEEK_END)
                self._position = target
                return target
            target = self._limit + offset
        else:
            raise ValueError(f"Invalid whence: {whence}")

        if self._limit is not None:
            target = min(target, self._limit)

        if self._seekable:
            self._stream.seek(target, SEEK_SET)
        elif target != self._position:
            raise OSError("The underlying stream is not seekable")

        self._position = target
        return target

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def _safe_seekable(stream: BinaryIO) -> bool:
    try:
        return stream.seekable()
    except Exception:
        return False


def as_bounded(stream: BinaryIO, limit: int | None = None) -> _BoundedReader:
    """Wrap ``stream`` in a :class:`_BoundedReader`, reusing an existing wrapper as-is."""
    if isinstance(stream, _BoundedReader):
        return stream
    return _BoundedReader(stream, limit)
