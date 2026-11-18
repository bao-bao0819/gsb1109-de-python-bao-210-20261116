"""Incremental assembly of LAS point records.

This module is meant for transports which do not guarantee that a single
``read`` returns all the requested bytes, and which may retransmit bytes that
were already delivered.

The assembler works on *point record boundaries*:

- bytes that form a complete point are moved to the confirmed prefix,
- the remaining bytes (always fewer than one point) are kept as the pending
  tail,
- control counts (per return number and per flight line / point source id)
  are tracked against the confirmed points only.

A short read is not an end of stream.  The caller must explicitly mark the
end of the stream with :meth:`PointStreamAssembler.finish` (or mark a
transport short read with :meth:`PointStreamAssembler.mark_short_read`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Optional, Sequence, Union

import numpy as np

from . import errors
from .header import LasHeader
from .lasdata import LasData
from .point import PointFormat
from .point.record import PackedPointRecord, ScaleAwarePointRecord

logger = logging.getLogger(__name__)

__all__ = [
    "ControlCounts",
    "ControlResiduals",
    "FeedResult",
    "PointStreamAssembler",
    "LasStreamAssembler",
    "LASStreamAssembler",
    "LasStreamReader",
    "StreamingLasReader",
    "IncrementalLasReader",
    "StreamState",
    "StreamStatus",
    "assemble_las_stream",
]


class StreamState(str, Enum):
    """State of an incremental point stream."""

    #: More bytes are needed.  This is the normal state while feeding.
    COLLECTING = "collecting"
    #: The last transport event was a short read.  It is not an end of file.
    SHORT_READ = "short_read"
    #: The declared point count and all control counts were satisfied.
    COMPLETE = "complete"
    #: The stream ended before the declared number of points was reached.
    TRUNCATED = "truncated"
    #: The stream contradicts its declaration (too many points or bad counts).
    DECLARATION_CONFLICT = "declaration_conflict"

    @property
    def is_final(self) -> bool:
        return self in (
            StreamState.COMPLETE,
            StreamState.TRUNCATED,
            StreamState.DECLARATION_CONFLICT,
        )


#: Backwards/forwards friendly alias.
StreamStatus = StreamState


@dataclass(frozen=True)
class ControlCounts:
    """Declared control counts for a stream.

    ``by_return`` maps a return number to its expected point count.
    ``by_flightline`` maps a flight line identifier (the LAS
    ``point_source_id`` dimension) to its expected point count.

    A sequence may be used for ``by_return``; its first element describes
    return number 1.
    """

    by_return: Mapping[int, int] = field(default_factory=dict)
    by_flightline: Mapping[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "by_return", _normalize_counts(self.by_return, 1))
        object.__setattr__(
            self, "by_flightline", _normalize_counts(self.by_flightline, 0)
        )


@dataclass(frozen=True)
class ControlResiduals:
    """Declared minus confirmed counts."""

    by_return: Mapping[int, int] = field(default_factory=dict)
    by_flightline: Mapping[int, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        """Sum of the absolute residuals."""
        return sum(abs(value) for value in self.by_return.values()) + sum(
            abs(value) for value in self.by_flightline.values()
        )

    @property
    def signed_total(self) -> int:
        return sum(self.by_return.values()) + sum(self.by_flightline.values())

    @property
    def is_zero(self) -> bool:
        return all(value == 0 for value in self.by_return.values()) and all(
            value == 0 for value in self.by_flightline.values()
        )

    def __bool__(self) -> bool:
        return not self.is_zero


@dataclass(frozen=True)
class FeedResult:
    """Outcome of a single :meth:`PointStreamAssembler.feed` call."""

    accepted_bytes: int
    duplicate_bytes: int
    confirmed_point_count: int
    pending_bytes: int
    point_count_residual: Optional[int]
    state: StreamState

    @property
    def status(self) -> StreamState:
        return self.state

    @property
    def short_read(self) -> bool:
        return self.state is StreamState.SHORT_READ

    @property
    def complete(self) -> bool:
        return self.state is StreamState.COMPLETE


def _normalize_counts(
    counts: Optional[Union[Mapping[int, int], Sequence[int]]],
    first_key: int,
) -> dict:
    if counts is None:
        return {}
    if isinstance(counts, Mapping):
        return {int(key): int(value) for key, value in counts.items()}
    return {first_key + index: int(value) for index, value in enumerate(counts)}


def _longest_prefix_that_is_suffix(text: bytes, suffix_source: bytes) -> int:
    """Longest prefix of ``text`` which is a suffix of ``suffix_source``.

    Uses the KMP failure function, so the cost is linear in the total size.
    """
    if not text or not suffix_source:
        return 0

    failure = [0] * len(text)
    matched = 0
    for index in range(1, len(text)):
        while matched and text[index] != text[matched]:
            matched = failure[matched - 1]
        if text[index] == text[matched]:
            matched += 1
        failure[index] = matched

    matched = 0
    for position, byte in enumerate(suffix_source):
        while matched and byte != text[matched]:
            matched = failure[matched - 1]
        if byte == text[matched]:
            matched += 1
        if matched == len(text) and position + 1 < len(suffix_source):
            # A shorter suffix may still match the end of the source.
            matched = failure[matched - 1]

    return matched


class PointStreamAssembler:
    """Assemble LAS point records from arbitrarily sized byte chunks.

    Parameters
    ----------
    source:
        A :class:`laspy.LasHeader`, a :class:`laspy.PointFormat`, or a point
        format id.  When a header is given, its declared point count and
        scales/offsets are used.
    point_count:
        Override the declared point count.  ``None`` means "no declaration".
    control_counts:
        Declared control counts.  A plain mapping is interpreted as
        ``by_return``.  The keyword arguments ``expected_by_return`` and
        ``expected_by_flightline`` are also accepted.
    overlap_window:
        Maximum number of already-seen bytes kept to detect an overlapping
        retransmission when :meth:`feed` is called without an offset.
    use_header_control_counts:
        When true and no explicit return controls were given, use the
        header's ``number_of_points_by_return`` as the declared counts.
    """

    def __init__(
        self,
        source: Union[LasHeader, PointFormat, int],
        *,
        point_count: Optional[int] = None,
        control_counts: Optional[
            Union[ControlCounts, Mapping[int, int], Sequence[int]]
        ] = None,
        expected_by_return: Optional[Union[Mapping[int, int], Sequence[int]]] = None,
        expected_by_flightline: Optional[Mapping[int, int]] = None,
        overlap_window: int = 1024 * 1024,
        min_overlap: Optional[int] = None,
        use_header_control_counts: bool = False,
    ) -> None:
        header: Optional[LasHeader] = None
        if isinstance(source, LasHeader):
            header = source
            point_format = source.point_format
            if point_count is None and source.point_count > 0:
                # A zero in a bare/new header means "not declared"; pass
                # ``point_count=0`` explicitly to declare an empty stream.
                point_count = source.point_count
        elif isinstance(source, PointFormat):
            point_format = source
        else:
            point_format = PointFormat(int(source))

        if point_count is not None and point_count < 0:
            raise ValueError("point_count must be >= 0 or None")

        self._header = header
        self._point_format = point_format
        self._point_size = point_format.size
        self._declared_point_count = point_count

        controls = self._coerce_controls(
            control_counts, expected_by_return, expected_by_flightline
        )
        if (
            use_header_control_counts
            and header is not None
            and not controls.by_return
        ):
            controls = ControlCounts(
                by_return={
                    index + 1: int(value)
                    for index, value in enumerate(header.number_of_points_by_return)
                    if int(value) != 0
                },
                by_flightline=controls.by_flightline,
            )

        self._has_return_controls = bool(controls.by_return)
        self._has_flightline_controls = bool(controls.by_flightline)
        self._declared_by_return = dict(controls.by_return)
        self._declared_by_flightline = dict(controls.by_flightline)

        self._confirmed = bytearray()
        self._pending = bytearray()
        self._history = bytearray()
        self._overlap_window = max(0, int(overlap_window))
        if min_overlap is None:
            # Offset-less deduplication needs real evidence. A couple of
            # matching bytes (e.g. repeated flags or zeros) are not enough.
            min_overlap = min(8, self._point_size)
        self._min_overlap = max(1, int(min_overlap))

        self._confirmed_by_return: dict[int, int] = {}
        self._confirmed_by_flightline: dict[int, int] = {}

        self._at_eof = False
        self._short_read = False
        self._short_read_count = 0
        self._conflict_reason: Optional[str] = None
        self._state = StreamState.COLLECTING

    @staticmethod
    def _coerce_controls(
        control_counts: Optional[
            Union[ControlCounts, Mapping[int, int], Sequence[int]]
        ],
        expected_by_return: Optional[Union[Mapping[int, int], Sequence[int]]],
        expected_by_flightline: Optional[Mapping[int, int]],
    ) -> ControlCounts:
        if isinstance(control_counts, ControlCounts):
            controls = control_counts
        elif control_counts is None:
            controls = ControlCounts()
        else:
            controls = ControlCounts(by_return=control_counts)

        by_return = controls.by_return
        by_flightline = controls.by_flightline
        if expected_by_return is not None:
            by_return = _normalize_counts(expected_by_return, 1)
        if expected_by_flightline is not None:
            by_flightline = _normalize_counts(expected_by_flightline, 0)
        return ControlCounts(by_return=by_return, by_flightline=by_flightline)

    # ------------------------------------------------------------------
    # Public state
    # ------------------------------------------------------------------
    @property
    def point_format(self) -> PointFormat:
        return self._point_format

    @property
    def point_size(self) -> int:
        return self._point_size

    @property
    def declared_point_count(self) -> Optional[int]:
        return self._declared_point_count

    @property
    def confirmed_point_count(self) -> int:
        return len(self._confirmed) // self._point_size

    @property
    def confirmed_count(self) -> int:
        return self.confirmed_point_count

    @property
    def pending_bytes(self) -> int:
        return len(self._pending)

    @property
    def pending_tail(self) -> bytes:
        return bytes(self._pending)

    @property
    def pending_tail_bytes(self) -> int:
        return len(self._pending)

    @property
    def trailing_bytes(self) -> int:
        return len(self._pending)

    @property
    def has_pending_tail(self) -> bool:
        return bool(self._pending)

    @property
    def received_bytes(self) -> int:
        return len(self._confirmed) + len(self._pending)

    @property
    def point_count_residual(self) -> Optional[int]:
        """Declared minus confirmed point count (negative means surplus)."""
        if self._declared_point_count is None:
            return None
        return self._declared_point_count - self.confirmed_point_count

    @property
    def declared_bytes(self) -> Optional[int]:
        if self._declared_point_count is None:
            return None
        return self._declared_point_count * self._point_size

    @property
    def confirmed_bytes(self) -> int:
        return len(self._confirmed)

    @property
    def byte_residual(self) -> Optional[int]:
        if self.declared_bytes is None:
            return None
        return self.declared_bytes - self.confirmed_bytes

    @property
    def declaration_satisfied(self) -> bool:
        """True when the confirmed points match the declared count and controls.

        This says nothing about EOF: extra bytes may still arrive.
        """
        if self.point_count_residual not in (None, 0):
            return False
        return not self.control_residuals

    @property
    def state(self) -> StreamState:
        return self._state

    @property
    def status(self) -> StreamState:
        return self._state

    @property
    def eof(self) -> bool:
        return self._at_eof

    @property
    def ended(self) -> bool:
        return self._at_eof

    @property
    def at_eof(self) -> bool:
        return self._at_eof

    @property
    def eof_reached(self) -> bool:
        return self._at_eof

    @property
    def end_of_file(self) -> bool:
        return self._at_eof

    @property
    def short_read(self) -> bool:
        return self._short_read

    @property
    def link_short_read(self) -> bool:
        return self._short_read

    @property
    def short_read_count(self) -> int:
        return self._short_read_count

    @property
    def declaration_conflict(self) -> bool:
        return self._state in (
            StreamState.DECLARATION_CONFLICT,
            StreamState.TRUNCATED,
        )

    @property
    def has_declaration_conflict(self) -> bool:
        return self.declaration_conflict

    @property
    def conflict(self) -> bool:
        return self.declaration_conflict

    @property
    def conflict_reason(self) -> Optional[str]:
        return self._conflict_reason

    @property
    def is_complete(self) -> bool:
        return self._state is StreamState.COMPLETE

    @property
    def complete(self) -> bool:
        return self.is_complete

    @property
    def is_truncated(self) -> bool:
        return self._state is StreamState.TRUNCATED

    @property
    def confirmed_prefix(self) -> PackedPointRecord:
        """The confirmed points, in their original packed representation."""
        return self._make_record()

    @property
    def confirmed_points(self) -> PackedPointRecord:
        return self.confirmed_prefix

    @property
    def points(self) -> PackedPointRecord:
        return self.confirmed_prefix

    def _make_record(self) -> PackedPointRecord:
        if self._header is not None:
            array = np.frombuffer(
                bytes(self._confirmed), dtype=self._header.point_format.dtype()
            )
            record = ScaleAwarePointRecord(
                array.copy(),
                self._header.point_format,
                self._header.scales,
                self._header.offsets,
            )
        else:
            array = np.frombuffer(
                bytes(self._confirmed), dtype=self._point_format.dtype()
            )
            record = PackedPointRecord(array.copy(), self._point_format)
        return record

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------
    @property
    def residual_by_return(self) -> dict:
        return self._residuals(
            self._declared_by_return,
            self._confirmed_by_return,
            self._has_return_controls,
        )

    @property
    def return_residual(self) -> dict:
        return self.residual_by_return

    @property
    def control_residual_by_return(self) -> dict:
        return self.residual_by_return

    @property
    def residual_by_flightline(self) -> dict:
        return self._residuals(
            self._declared_by_flightline,
            self._confirmed_by_flightline,
            self._has_flightline_controls,
        )

    @property
    def flightline_residual(self) -> dict:
        return self.residual_by_flightline

    @property
    def control_residual_by_flightline(self) -> dict:
        return self.residual_by_flightline

    @property
    def confirmed_by_return(self) -> dict:
        return dict(self._confirmed_by_return)

    @property
    def confirmed_by_flightline(self) -> dict:
        return dict(self._confirmed_by_flightline)

    @property
    def declared_by_return(self) -> dict:
        return dict(self._declared_by_return)

    @property
    def declared_by_flightline(self) -> dict:
        return dict(self._declared_by_flightline)

    @staticmethod
    def _residuals(
        declared: Mapping[int, int],
        confirmed: Mapping[int, int],
        has_controls: bool,
    ) -> dict:
        if not has_controls:
            return {}
        keys = sorted(set(declared) | set(confirmed))
        return {
            key: int(declared.get(key, 0)) - int(confirmed.get(key, 0)) for key in keys
        }

    @property
    def control_residuals(self) -> ControlResiduals:
        return ControlResiduals(
            by_return=self.residual_by_return,
            by_flightline=self.residual_by_flightline,
        )

    @property
    def control_residual(self) -> ControlResiduals:
        return self.control_residuals

    @property
    def control_counts_residual(self) -> ControlResiduals:
        return self.control_residuals

    @property
    def total_control_residual(self) -> int:
        return self.control_residuals.total

    # ------------------------------------------------------------------
    # Feeding
    # ------------------------------------------------------------------
    def feed(
        self,
        data: Union[bytes, bytearray, memoryview],
        *,
        offset: Optional[int] = None,
        requested: Optional[int] = None,
        eof: bool = False,
    ) -> FeedResult:
        """Feed a chunk of point-data bytes.

        Parameters
        ----------
        data:
            Bytes belonging to the point-data section (after the header and
            VLRs).
        offset:
            Absolute byte offset of ``data`` relative to the start of point
            data.  When provided, an overlap with previously seen bytes is
            resolved exactly (and a mismatch is an error).  When omitted, the
            assembler tries to detect an overlapping retransmission by
            matching the start of ``data`` against the tail of the already
            seen bytes.
        requested:
            How many bytes the transport was asked for.  When
            ``len(data) < requested`` the event is recorded as a link short
            read; it never marks the stream as finished.
        eof:
            When true, ``data`` is the last chunk and the stream is finished
            right after it is consumed.
        """
        chunk = bytes(data)
        if len(chunk) == 0:
            self.mark_short_read()
            if eof:
                self.finish()
            return self._result(0, 0)

        if offset is None:
            duplicate = self._detect_overlap(chunk)
            accepted = chunk[duplicate:]
        else:
            duplicate, accepted = self._apply_offset(chunk, int(offset))

        if accepted:
            self._pending.extend(accepted)
            self._remember(accepted)
            self._confirm_complete_points()

        short = requested is not None and len(chunk) < int(requested)
        if short:
            self.mark_short_read()
        elif self._state is StreamState.SHORT_READ:
            # A full read arrived after a short one: the link recovered, but
            # the short read remains visible in ``short_read_count``.
            self._short_read = False
        self._recompute_state()

        if eof:
            self.finish()

        return self._result(len(accepted), duplicate)

    # Friendly aliases.
    push = feed
    feed_bytes = feed
    add_bytes = feed
    append = feed

    def _result(self, accepted: int, duplicate: int) -> FeedResult:
        return FeedResult(
            accepted_bytes=accepted,
            duplicate_bytes=duplicate,
            confirmed_point_count=self.confirmed_point_count,
            pending_bytes=self.pending_bytes,
            point_count_residual=self.point_count_residual,
            state=self._state,
        )

    def _detect_overlap(self, chunk: bytes) -> int:
        """Return the length of ``chunk`` already present at the tail.

        A retransmission starts somewhere in the bytes already received and
        extends them, so its overlap is the longest suffix of the history
        which is a prefix of ``chunk``.

        Short matches need actual evidence: LAS coordinates frequently
        contain long runs of zero bytes, and a short all-zero match would
        otherwise look like a retransmission and cause genuine bytes to be
        dropped.  Pass an explicit ``offset`` to :meth:`feed` when the
        transport provides one; that resolves the overlap exactly.
        """
        history = bytes(self._history)
        max_overlap = min(len(history), len(chunk), self._overlap_window)
        if max_overlap < self._min_overlap:
            return 0

        tail = history[-max_overlap:]
        overlap = _longest_prefix_that_is_suffix(chunk, tail)
        if overlap < self._min_overlap:
            return 0
        if overlap < self._point_size and not (tail[-overlap:]).strip(b"\0"):
            return 0
        return overlap

    def _apply_offset(self, chunk: bytes, offset: int) -> tuple[int, bytes]:
        if offset < 0:
            raise ValueError("offset must be >= 0")

        known_len = len(self._confirmed) + len(self._pending)
        if offset > known_len:
            self._fail(
                "gap",
                f"received data at offset {offset} but only {known_len} bytes "
                "have been seen; missing bytes cannot be assembled",
            )
            raise errors.StreamGapError(self._conflict_reason)

        overlap = min(known_len - offset, len(chunk))
        if overlap > 0:
            known = self._known_bytes(offset, overlap)
            if known != chunk[:overlap]:
                self._fail(
                    "overlap_mismatch",
                    "retransmitted bytes overlap previously received bytes "
                    "but the evidence does not match",
                )
                raise errors.StreamOverlapConflict(self._conflict_reason)

        return overlap, chunk[overlap:]

    def _known_bytes(self, offset: int, length: int) -> bytes:
        end = offset + length
        confirmed_len = len(self._confirmed)
        if end <= confirmed_len:
            return bytes(self._confirmed[offset:end])
        first = bytes(self._confirmed[offset:confirmed_len])
        rest = bytes(self._pending[: end - confirmed_len])
        return first + rest

    def _remember(self, accepted: bytes) -> None:
        if self._overlap_window <= 0:
            return
        self._history.extend(accepted)
        excess = len(self._history) - self._overlap_window
        if excess > 0:
            del self._history[:excess]

    def _confirm_complete_points(self) -> None:
        complete_bytes = (len(self._pending) // self._point_size) * self._point_size
        if complete_bytes == 0:
            return

        new_points = bytes(self._pending[:complete_bytes])
        del self._pending[:complete_bytes]
        self._confirmed.extend(new_points)
        self._count_points(new_points)
        self._recompute_state()

    def _count_points(self, raw: bytes) -> None:
        record = PackedPointRecord.from_buffer(raw, self._point_format)
        return_numbers = np.asarray(record.return_number, dtype=np.int64)
        for value, count in zip(*np.unique(return_numbers, return_counts=True)):
            # A return number of 0 is invalid and is not part of the
            # per-return control counts.
            if int(value) == 0:
                continue
            self._confirmed_by_return[int(value)] = (
                self._confirmed_by_return.get(int(value), 0) + int(count)
            )

        flightlines = np.asarray(record.point_source_id, dtype=np.int64)
        for value, count in zip(*np.unique(flightlines, return_counts=True)):
            self._confirmed_by_flightline[int(value)] = (
                self._confirmed_by_flightline.get(int(value), 0) + int(count)
            )

    # ------------------------------------------------------------------
    # Stream events
    # ------------------------------------------------------------------
    def mark_short_read(self) -> None:
        """Record a transport short read.  This is not an end of stream."""
        if self._at_eof:
            return
        self._short_read = True
        self._short_read_count += 1
        self._state = StreamState.SHORT_READ

    feed_short_read = mark_short_read
    short_read_seen = mark_short_read
    note_short_read = mark_short_read

    def finish(self, *, strict: bool = False) -> StreamState:
        """Mark the end of the stream and evaluate the declarations."""
        self._at_eof = True
        # An explicit end of stream supersedes a previous transport short
        # read: the missing bytes will never arrive.
        self._short_read = False
        self._recompute_state(final=True)

        if strict and self._state in (
            StreamState.TRUNCATED,
            StreamState.DECLARATION_CONFLICT,
        ):
            self._raise_for_state()
        return self._state
    finish_eof = finish
    end_of_stream = finish
    mark_eof = finish

    def _recompute_state(self, final: bool = False) -> None:
        point_count = self.confirmed_point_count
        declared = self._declared_point_count

        if declared is not None and point_count > declared:
            self._fail(
                "too_many_points",
                f"stream contains {point_count} confirmed points but the "
                f"declaration is {declared}",
            )
            self._state = StreamState.DECLARATION_CONFLICT
            return

        if final:
            if self.pending_bytes:
                self._fail(
                    "incomplete_point",
                    f"stream ended with {self.pending_bytes} trailing bytes, "
                    f"which is less than one {self._point_size}-byte point",
                )
                self._state = StreamState.TRUNCATED
                return

            if declared is not None and point_count < declared:
                self._fail(
                    "missing_points",
                    f"stream ended after {point_count} points but the "
                    f"declaration is {declared}",
                )
                self._state = StreamState.TRUNCATED
                return

            if self.control_residuals:
                self._fail(
                    "control_count_mismatch",
                    "the confirmed points do not match the declared control "
                    f"counts: {self.control_residuals!r}",
                )
                self._state = StreamState.DECLARATION_CONFLICT
                return

            self._conflict_reason = None
            self._state = StreamState.COMPLETE
            return

        # Not at EOF yet.  Having all the declared points is not enough to
        # finish: extra bytes could still arrive, so completion is only
        # decided by ``finish``.
        self._state = (
            StreamState.SHORT_READ if self._short_read else StreamState.COLLECTING
        )

    def _fail(self, reason: str, message: str) -> None:
        self._conflict_reason = f"{reason}: {message}"

    def _raise_for_state(self) -> None:
        if self._state is StreamState.SHORT_READ:
            raise errors.StreamShortReadError(
                self._conflict_reason
                or "the stream is temporarily short, it is not an end of file"
            )
        if self._state is StreamState.COLLECTING:
            raise errors.StreamAssemblyError(
                self._conflict_reason
                or "the stream is not finished, call finish() before building"
            )
        if self._state is StreamState.TRUNCATED:
            message = self._conflict_reason or "stream ended before its declaration"
            if "incomplete_point" in message:
                raise errors.StreamIncompletePointError(message)
            raise errors.StreamTruncatedError(message)
        if self._state is StreamState.DECLARATION_CONFLICT:
            if "overlap" in (self._conflict_reason or ""):
                raise errors.StreamOverlapConflict(self._conflict_reason or "")
            raise errors.StreamDeclarationConflict(
                self._conflict_reason or "stream contradicts its declaration"
            )
        raise errors.StreamAssemblyError(
            self._conflict_reason or "stream is not publishable"
        )

    # ------------------------------------------------------------------
    # Final data
    # ------------------------------------------------------------------
    def build(
        self,
        *,
        require_complete: bool = True,
        strict: bool = False,
    ) -> Union[LasData, PackedPointRecord]:
        """Return the assembled data.

        With ``require_complete`` (the default), a stream that does not match
        its declaration raises instead of publishing a smaller cloud.
        ``require_complete=False`` returns the confirmed prefix only; the
        pending tail is never published as points.
        """
        if require_complete and self._state is not StreamState.COMPLETE:
            self._raise_for_state()
        if strict and self._state in (
            StreamState.TRUNCATED,
            StreamState.DECLARATION_CONFLICT,
        ):
            self._raise_for_state()

        if self._state is not StreamState.COMPLETE:
            logger.warning(
                "publishing %d confirmed points from an incomplete stream (%s)",
                self.confirmed_point_count,
                self._conflict_reason or self._state.value,
            )

        record = self.confirmed_prefix
        if self._header is None:
            return record

        header = self._header.copy()
        header.point_count = self.confirmed_point_count
        las = LasData(header=header, points=record)
        return las

    to_las_data = build
    to_points = build

    def publish(self, **kwargs) -> Union[LasData, PackedPointRecord]:
        return self.build(**kwargs)

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_chunks(
        cls,
        chunks: Iterable[Union[bytes, bytearray, memoryview]],
        source: Union[LasHeader, PointFormat, int],
        **kwargs,
    ) -> "PointStreamAssembler":
        assembler = cls(source, **kwargs)
        for chunk in chunks:
            assembler.feed(chunk)
        assembler.finish()
        return assembler

    @classmethod
    def from_stream(
        cls,
        stream,
        source: Union[LasHeader, PointFormat, int],
        *,
        chunk_size: int = 1024 * 1024,
        **kwargs,
    ) -> "PointStreamAssembler":
        assembler = cls(source, **kwargs)
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            assembler.feed(chunk, requested=chunk_size)
        assembler.finish()
        return assembler


# Common class-name aliases.
LasStreamAssembler = PointStreamAssembler
LASStreamAssembler = PointStreamAssembler
LasStreamReader = PointStreamAssembler
StreamingLasReader = PointStreamAssembler
IncrementalLasReader = PointStreamAssembler


def assemble_las_stream(
    source: Union[LasHeader, PointFormat, int],
    chunks: Iterable[Union[bytes, bytearray, memoryview]],
    **kwargs,
) -> PointStreamAssembler:
    """Feed ``chunks`` into a :class:`PointStreamAssembler` and finish it."""
    return PointStreamAssembler.from_chunks(chunks, source, **kwargs)
