"""Per-sample waveform extraction with spatial coordinates.

A LAS waveform packet is a block of raw digitizer samples.  The packet alone
does not say where each sample was measured; that comes from the point that
references the packet (its coordinates and direction vector) plus the packet
descriptor (sample count, temporal spacing, gain and offset).
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any

import numpy as np

from .descriptor import (
    WavePacketDescriptorRecordId,
    WaveformPacketDescriptor,
    WaveformPacketDescriptorRegistry,
)

#: Speed of light in a vacuum, metres per picosecond.
SPEED_OF_LIGHT_M_PER_PS = 0.000299792458

#: LAS stores waveform temporal sample spacing in picoseconds.
LAS_WAVEFORM_TIME_UNIT = "ps"


class WaveformSampleIssueKind(enum.StrEnum):
    """Why a point (or part of its packet) could not become a sample row."""

    MISSING_SOURCE = "missing_source"
    MISSING_PACKET = "missing_packet"
    MISSING_DESCRIPTOR = "missing_descriptor"
    UNKNOWN_COMPRESSION = "unknown_compression"
    INCOMPLETE_SAMPLES = "incomplete_samples"
    MISSING_GEOMETRY = "missing_geometry"


@dataclasses.dataclass(frozen=True, slots=True)
class WaveformSampleIssue:
    """A recoverable problem encountered while building the sample table."""

    kind: WaveformSampleIssueKind
    point_index: int
    message: str
    descriptor_index: int | None = None
    descriptor_record_id: int | None = None
    expected_samples: int | None = None
    available_samples: int | None = None
    byte_offset: int | None = None
    available_bytes: int | None = None

    def __str__(self) -> str:
        return f"{self.kind}: point {self.point_index}: {self.message}"


@dataclasses.dataclass(frozen=True)
class WaveformSampleTable:
    """Flat table with one row per waveform sample.

    Every row records the point that owns the packet, the descriptor that
    explains the packet, the sample index inside the packet, the raw amplitude
    byte range, the uncalibrated integer amplitude, the gain/offset adjusted
    amplitude, and the sample's spatial coordinates and time.

    Samples are in packet order.  The table only contains samples laspy could
    fully decode; problems are accumulated in ``issues`` instead of silently
    dropping points or inventing zeros.
    """

    point_index: np.ndarray
    descriptor_index: np.ndarray
    descriptor_record_id: np.ndarray
    sample_index: np.ndarray
    sample_offset_bytes: np.ndarray
    sample_size_bytes: np.ndarray
    raw_amplitude: np.ndarray
    amplitude: np.ndarray
    time: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    packet_byte_offset: np.ndarray
    packet_byte_length: np.ndarray
    source_packet_byte_offset: np.ndarray
    source_sample_byte_offset: np.ndarray
    time_unit: str = LAS_WAVEFORM_TIME_UNIT
    issues: tuple[WaveformSampleIssue, ...] = ()

    def __len__(self) -> int:
        return int(self.point_index.shape[0])

    @property
    def sample_count(self) -> int:
        return len(self)

    @property
    def point_count(self) -> int:
        return int(np.unique(self.point_index).shape[0])

    @property
    def xyz(self) -> np.ndarray:
        return np.column_stack((self.x, self.y, self.z))

    @property
    def time_ps(self) -> np.ndarray:
        if self.time_unit == "ps":
            return self.time
        factor = _parse_time_unit(self.time_unit)
        return self.time / factor

    @property
    def columns(self) -> dict[str, np.ndarray]:
        return {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name not in ("issues", "time_unit")
        }

    def column_names(self) -> tuple[str, ...]:
        return tuple(self.columns)

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str):
            return self.column(item)
        if isinstance(item, (int, np.integer)):
            index = np.array([int(item)], dtype=np.int64)
        elif isinstance(item, (slice, list, tuple, np.ndarray)):
            index = np.asarray(item)
        else:
            raise TypeError(f"unsupported waveform sample table index: {type(item)}")
        selected = self._take(index)
        if isinstance(item, (int, np.integer)):
            return selected._take(np.array([0], dtype=np.int64))
        return selected

    def _take(self, index: np.ndarray) -> "WaveformSampleTable":
        data = {}
        for field in dataclasses.fields(self):
            if field.name in ("issues", "time_unit"):
                continue
            data[field.name] = getattr(self, field.name)[index]
        return WaveformSampleTable(**data, time_unit=self.time_unit, issues=self.issues)

    def column(self, name: str) -> np.ndarray:
        aliases = {
            "point": "point_index",
            "descriptor": "descriptor_index",
            "record_id": "descriptor_record_id",
            "sample": "sample_index",
            "t": "time",
            "amplitude_raw": "raw_amplitude",
            "amplitude_scaled": "amplitude",
            "offset": "packet_byte_offset",
            "sample_byte_offset": "sample_offset_bytes",
            "byte_offset": "sample_offset_bytes",
            "byte_length": "packet_byte_length",
            "source_offset": "source_sample_byte_offset",
            "source_byte_offset": "source_sample_byte_offset",
            "source_packet_offset": "source_packet_byte_offset",
            "packet_id": "packet_byte_offset",
        }
        try:
            return getattr(self, aliases.get(name, name))
        except AttributeError:
            raise KeyError(name) from None

    def points(self) -> np.ndarray:
        return np.unique(self.point_index)

    def issues_by_kind(
        self,
    ) -> dict[WaveformSampleIssueKind, tuple[WaveformSampleIssue, ...]]:
        grouped: dict[WaveformSampleIssueKind, list[WaveformSampleIssue]] = {}
        for issue in self.issues:
            grouped.setdefault(issue.kind, []).append(issue)
        return {kind: tuple(items) for kind, items in grouped.items()}

    def to_records(self) -> np.ndarray:
        """Return the table as a structured NumPy array."""
        dtype = [
            ("point_index", np.int64),
            ("descriptor_index", np.uint8),
            ("descriptor_record_id", np.int64),
            ("sample_index", np.int64),
            ("sample_offset_bytes", np.int64),
            ("sample_size_bytes", np.int64),
            ("raw_amplitude", self.raw_amplitude.dtype),
            ("amplitude", np.float64),
            ("time", np.float64),
            ("x", np.float64),
            ("y", np.float64),
            ("z", np.float64),
            ("packet_byte_offset", np.int64),
            ("packet_byte_length", np.int64),
            ("source_packet_byte_offset", np.int64),
            ("source_sample_byte_offset", np.int64),
        ]
        out = np.empty(len(self), dtype=dtype)
        for name, _ in dtype:
            out[name] = getattr(self, name)
        return out


@dataclasses.dataclass(slots=True)
class _Accumulator:
    point_index: list[np.ndarray] = dataclasses.field(default_factory=list)
    descriptor_index: list[np.ndarray] = dataclasses.field(default_factory=list)
    descriptor_record_id: list[np.ndarray] = dataclasses.field(default_factory=list)
    sample_index: list[np.ndarray] = dataclasses.field(default_factory=list)
    sample_offset_bytes: list[np.ndarray] = dataclasses.field(default_factory=list)
    sample_size_bytes: list[np.ndarray] = dataclasses.field(default_factory=list)
    raw_amplitude: list[np.ndarray] = dataclasses.field(default_factory=list)
    amplitude: list[np.ndarray] = dataclasses.field(default_factory=list)
    time: list[np.ndarray] = dataclasses.field(default_factory=list)
    x: list[np.ndarray] = dataclasses.field(default_factory=list)
    y: list[np.ndarray] = dataclasses.field(default_factory=list)
    z: list[np.ndarray] = dataclasses.field(default_factory=list)
    packet_byte_offset: list[np.ndarray] = dataclasses.field(default_factory=list)
    packet_byte_length: list[np.ndarray] = dataclasses.field(default_factory=list)
    source_packet_byte_offset: list[np.ndarray] = dataclasses.field(default_factory=list)
    source_sample_byte_offset: list[np.ndarray] = dataclasses.field(default_factory=list)
    issues: list[WaveformSampleIssue] = dataclasses.field(default_factory=list)

    def add_block(
        self,
        *,
        point_index: int,
        descriptor_index: int,
        descriptor_record_id: int,
        packet_offset: int,
        packet_length: int,
        source_packet_offset: int,
        raw: np.ndarray,
        amplitude: np.ndarray,
        times: np.ndarray,
        xyz: np.ndarray,
        first_sample_offset: int,
        sample_size: int,
    ) -> None:
        count = raw.shape[0]
        sample_index = np.arange(count, dtype=np.int64)
        self.point_index.append(np.full(count, point_index, dtype=np.int64))
        self.descriptor_index.append(np.full(count, descriptor_index, dtype=np.uint8))
        self.descriptor_record_id.append(
            np.full(count, descriptor_record_id, dtype=np.int64)
        )
        self.sample_index.append(sample_index)
        self.sample_offset_bytes.append(
            np.int64(first_sample_offset) + sample_index * sample_size
        )
        self.sample_size_bytes.append(np.full(count, sample_size, dtype=np.int64))
        self.raw_amplitude.append(np.asarray(raw))
        self.amplitude.append(np.asarray(amplitude, dtype=np.float64))
        self.time.append(np.asarray(times, dtype=np.float64))
        self.x.append(np.asarray(xyz[:, 0], dtype=np.float64))
        self.y.append(np.asarray(xyz[:, 1], dtype=np.float64))
        self.z.append(np.asarray(xyz[:, 2], dtype=np.float64))
        self.packet_byte_offset.append(np.full(count, packet_offset, dtype=np.int64))
        self.packet_byte_length.append(np.full(count, packet_length, dtype=np.int64))
        self.source_packet_byte_offset.append(
            np.full(count, source_packet_offset, dtype=np.int64)
        )
        self.source_sample_byte_offset.append(
            np.int64(source_packet_offset) + sample_index * sample_size
        )

    def finish(self, time_unit: str) -> WaveformSampleTable:
        empty_f = np.empty(0, dtype=np.float64)
        empty_i = np.empty(0, dtype=np.int64)
        if not self.point_index:
            return WaveformSampleTable(
                point_index=empty_i,
                descriptor_index=np.empty(0, dtype=np.uint8),
                descriptor_record_id=empty_i,
                sample_index=empty_i,
                sample_offset_bytes=empty_i,
                sample_size_bytes=empty_i,
                raw_amplitude=np.empty(0, dtype=np.uint8),
                amplitude=empty_f,
                time=empty_f,
                x=empty_f,
                y=empty_f,
                z=empty_f,
                packet_byte_offset=empty_i,
                packet_byte_length=empty_i,
                source_packet_byte_offset=empty_i,
                source_sample_byte_offset=empty_i,
                time_unit=time_unit,
                issues=tuple(self.issues),
            )
        return WaveformSampleTable(
            point_index=np.concatenate(self.point_index),
            descriptor_index=np.concatenate(self.descriptor_index),
            descriptor_record_id=np.concatenate(self.descriptor_record_id),
            sample_index=np.concatenate(self.sample_index),
            sample_offset_bytes=np.concatenate(self.sample_offset_bytes),
            sample_size_bytes=np.concatenate(self.sample_size_bytes),
            raw_amplitude=np.concatenate(self.raw_amplitude),
            amplitude=np.concatenate(self.amplitude),
            time=np.concatenate(self.time),
            x=np.concatenate(self.x),
            y=np.concatenate(self.y),
            z=np.concatenate(self.z),
            packet_byte_offset=np.concatenate(self.packet_byte_offset),
            packet_byte_length=np.concatenate(self.packet_byte_length),
            source_packet_byte_offset=np.concatenate(self.source_packet_byte_offset),
            source_sample_byte_offset=np.concatenate(self.source_sample_byte_offset),
            time_unit=time_unit,
            issues=tuple(self.issues),
        )


def _sample_dtype(descriptor: WaveformPacketDescriptor) -> np.dtype:
    bits = descriptor.bits_per_sample
    if bits <= 0 or bits % 8 != 0:
        raise ValueError("Waveform sample size must be a multiple of 8 bits")
    return np.dtype(f"<u{bits // 8}")


def _parse_time_unit(unit: str) -> float:
    """Return how many ``unit`` fit in one picosecond."""
    table = {
        "ps": 1.0,
        "picosecond": 1.0,
        "picoseconds": 1.0,
        "ns": 1e-3,
        "nanosecond": 1e-3,
        "nanoseconds": 1e-3,
        "us": 1e-6,
        "microsecond": 1e-6,
        "microseconds": 1e-6,
        "ms": 1e-9,
        "millisecond": 1e-9,
        "milliseconds": 1e-9,
        "s": 1e-12,
        "second": 1e-12,
        "seconds": 1e-12,
    }
    try:
        return table[unit]
    except KeyError:
        raise ValueError(f"unknown waveform time unit: {unit!r}") from None


def _as_point_array(points: Any) -> np.ndarray:
    array = getattr(points, "array", None)
    return np.asarray(array if array is not None else points)


class WaveformPacketSource:
    """Random access to waveform packet bytes with an explicit payload base.

    LAS stores waveform packets either in a sibling ``.wdp`` file or in the
    LAS/LAZ file's internal waveform EVLR.  The packet byte offsets carried by
    points are relative to the start of the waveform data packet record (the
    EVLR header included).  This class keeps that base offset and exposes
    reads by the packet-relative offset; pass the EVLR header size explicitly
    when the file places its first packet there.
    """

    def __init__(self, path: Any, *, payload_offset: int = 0):
        self.path = path
        self.payload_offset = int(payload_offset)
        self._source: Any = None
        self._owned = False

    @classmethod
    def for_external_wdp(cls, las_path: Any) -> "WaveformPacketSource":
        # The .wdp file itself is the Waveform Data Packet Record, so point
        # offsets are relative to the start of that file.
        return cls(_path_with_suffix(las_path, ".wdp"), payload_offset=0)

    @classmethod
    def for_internal_las(
        cls, las_path: Any, start_of_waveform_record: int
    ) -> "WaveformPacketSource":
        # Point offsets are relative to the internal waveform EVLR, which
        # starts at this header offset.
        return cls(las_path, payload_offset=int(start_of_waveform_record))

    def read_packet(self, offset: int, length: int) -> tuple[bytes, int]:
        source = self._ensure_open()
        source.seek(self.payload_offset + int(offset))
        data = source.read(int(length))
        return bytes(data), len(data)

    def _ensure_open(self) -> Any:
        if self._source is None or getattr(self._source, "closed", False):
            self._source = open(self.path, "rb")
            self._owned = True
        return self._source

    def close(self) -> None:
        if self._owned and self._source is not None:
            self._source.close()
        self._source = None
        self._owned = False


def _path_with_suffix(path: Any, suffix: str) -> Any:
    from pathlib import Path

    if isinstance(path, (str, Path)):
        return Path(path).with_suffix(suffix)
    name = getattr(path, "name", None)
    if name:
        return Path(name).with_suffix(suffix)
    raise ValueError("cannot locate the external waveform .wdp file")


class _ByteSourceProvider:
    def __init__(self, source: WaveformPacketSource):
        self.source = source
        self.payload_offset = source.payload_offset

    def read(self, point_index: int, offset: int, length: int) -> tuple[bytes, int]:
        return self.source.read_packet(offset, length)

    def close(self) -> None:
        self.source.close()


class _StreamPacketProvider:
    """Reads packets by raw byte offset from an external waveform stream."""

    def __init__(self, source: Any, *, payload_offset: int | None = None):
        self._source = source
        self.payload_offset = 0 if payload_offset is None else payload_offset

    def read(self, point_index: int, offset: int, length: int) -> tuple[bytes, int]:
        source = self._source
        if not hasattr(source, "read") and hasattr(source, "reader"):
            source = source.reader
        ensure_open = getattr(source, "_ensure_open", None)
        if ensure_open is not None:
            ensure_open()
        if hasattr(source, "source") and hasattr(source.source, "read"):
            source = source.source
        if not hasattr(source, "seek") or not hasattr(source, "read"):
            return b"", 0
        source.seek(self.payload_offset + offset)
        data = source.read(length)
        return bytes(data), len(data)

    def close(self) -> None:
        return None


class _EagerPacketProvider:
    """Reads packets from an already materialised ``WaveformRecord``."""

    def __init__(self, waveforms: Any, points_waveform_index: np.ndarray):
        self._samples = np.ascontiguousarray(waveforms.samples)
        self._index = np.asarray(points_waveform_index, dtype=np.int64)
        self.wave_size = int(self._samples.dtype.itemsize)

    def read(self, point_index: int, offset: int, length: int) -> tuple[bytes, int]:
        if point_index >= self._index.shape[0]:
            return b"", 0
        packet_index = int(self._index[point_index])
        if packet_index < 0:
            return b"", 0
        start = packet_index * self.wave_size
        data = self._samples.tobytes()[start : start + length]
        return bytes(data), len(data)


def _make_provider(waveform_source: Any) -> tuple[Any, str | None]:
    """Return ``(provider, failure_reason)`` for a reader or eager state."""
    if waveform_source is None:
        return None, "no waveform source is available"
    if isinstance(waveform_source, WaveformPacketSource):
        return _ByteSourceProvider(waveform_source), None
    explicit_base = getattr(waveform_source, "payload_offset", None)
    waveforms = getattr(waveform_source, "waveforms", None)
    index = getattr(waveform_source, "points_waveform_index", None)
    if waveforms is not None and index is not None:
        return _EagerPacketProvider(waveforms, index), None
    return _StreamPacketProvider(waveform_source, payload_offset=explicit_base), None


def _point_geometry(
    points: Any,
    array: np.ndarray,
    names: set[str],
    scales: Any,
    offsets: Any,
    index: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return ``(anchor_xyz, direction_vector)`` for one point."""
    try:
        anchor = np.array(
            [
                float(np.asarray(points["x"])[index]),
                float(np.asarray(points["y"])[index]),
                float(np.asarray(points["z"])[index]),
            ],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError, IndexError):
        anchor_values = []
        for axis, name in enumerate(("X", "Y", "Z")):
            if name not in names:
                return None, None
            value = float(np.asarray(array[name])[index])
            if scales is not None and offsets is not None:
                value = value * float(scales[axis]) + float(offsets[axis])
            anchor_values.append(value)
        anchor = np.array(anchor_values, dtype=np.float64)

    if not np.all(np.isfinite(anchor)):
        return None, None

    components = []
    for name in ("x_t", "y_t", "z_t"):
        if name not in names:
            return anchor, None
        components.append(float(np.asarray(array[name])[index]))
    direction = np.array(components, dtype=np.float64)
    if not np.all(np.isfinite(direction)) or not direction.any():
        return anchor, None
    return anchor, direction


def build_waveform_sample_table(
    points: Any,
    waveform_source: Any,
    registry: WaveformPacketDescriptorRegistry,
    *,
    time_unit: str = LAS_WAVEFORM_TIME_UNIT,
) -> WaveformSampleTable:
    """Expand waveform packets referenced by ``points`` into a sample table.

    ``points`` is a point record exposing ``X/Y/Z`` (integer or scaled), or
    ``x/y/z`` floats plus the waveform location/vector dimensions.  The
    coordinates follow the LAS definition: for sample ``i`` of the packet
    attached to point ``p``,

    ``offset_ps = i * temporal_sample_spacing - return_point_wave_location``

    and ``sample_xyz = point_xyz + offset_ps * (x_t, y_t, z_t)``.  The vector is
    in metres per picosecond; its norm should be about ``c/2``.  Time values
    are therefore offsets in picoseconds unless another ``time_unit`` is
    requested.

    Packets are identified by the point's byte offset, not by the point index:
    several points may legitimately share one packet and must then reuse the
    same amplitude bytes while still getting their own spatial coordinates.

    ``waveform_source`` may be a lazy reader (anything with a readable
    ``source``), an ``EagerWaveformState``, or ``None``.  Problems do not abort
    the whole table; they are recorded in ``WaveformSampleTable.issues``.
    """
    unit_factor = _parse_time_unit(time_unit)
    provider, source_failure = _make_provider(waveform_source)

    array = _as_point_array(points)
    names = set(array.dtype.names or ())
    scales = getattr(points, "scales", None)
    offsets = getattr(points, "offsets", None)

    def attr(name: str, index: int) -> Any:
        if name not in names:
            return None
        return np.asarray(array[name])[index]

    acc = _Accumulator()
    if source_failure is not None:
        for point_index in range(len(points)):
            acc.issues.append(
                WaveformSampleIssue(
                    kind=WaveformSampleIssueKind.MISSING_SOURCE,
                    point_index=point_index,
                    message=source_failure,
                )
            )
        return acc.finish(time_unit)

    packet_cache: dict[tuple[int, int], tuple[bytes, int]] = {}

    for point_index in range(len(points)):
        descriptor_index_value = attr("wavepacket_index", point_index)
        descriptor_index = (
            0 if descriptor_index_value is None else int(descriptor_index_value)
        )

        if descriptor_index == 0:
            acc.issues.append(
                WaveformSampleIssue(
                    kind=WaveformSampleIssueKind.MISSING_PACKET,
                    point_index=point_index,
                    message="point has no waveform packet (wavepacket_index is 0)",
                )
            )
            continue

        descriptor = registry.descriptor_for_index(descriptor_index)
        record_id = int(WavePacketDescriptorRecordId.from_index(descriptor_index))
        if descriptor is None:
            acc.issues.append(
                WaveformSampleIssue(
                    kind=WaveformSampleIssueKind.MISSING_DESCRIPTOR,
                    point_index=point_index,
                    descriptor_index=descriptor_index,
                    descriptor_record_id=record_id,
                    message="no waveform packet descriptor matches this point",
                )
            )
            continue

        packet_size = descriptor.packet_size_bytes
        if descriptor.waveform_compression_type != 0 or packet_size is None:
            acc.issues.append(
                WaveformSampleIssue(
                    kind=WaveformSampleIssueKind.UNKNOWN_COMPRESSION,
                    point_index=point_index,
                    descriptor_index=descriptor_index,
                    descriptor_record_id=record_id,
                    expected_samples=descriptor.number_of_samples,
                    message=(
                        "cannot decode waveform packet: compression type "
                        f"{descriptor.waveform_compression_type}, "
                        f"{descriptor.bits_per_sample} bits per sample"
                    ),
                )
            )
            continue

        packet_offset_value = attr("wavepacket_offset", point_index)
        if packet_offset_value is None:
            acc.issues.append(
                WaveformSampleIssue(
                    kind=WaveformSampleIssueKind.MISSING_PACKET,
                    point_index=point_index,
                    descriptor_index=descriptor_index,
                    descriptor_record_id=record_id,
                    expected_samples=descriptor.number_of_samples,
                    message="point has no wavepacket_offset",
                )
            )
            continue
        packet_offset = int(packet_offset_value)

        declared_size_value = attr("wavepacket_size", point_index)
        declared_size = (
            packet_size
            if declared_size_value is None or int(declared_size_value) == 0
            else int(declared_size_value)
        )
        read_length = min(packet_size, declared_size)
        declared_short = declared_size < packet_size
        if declared_short:
            acc.issues.append(
                WaveformSampleIssue(
                    kind=WaveformSampleIssueKind.INCOMPLETE_SAMPLES,
                    point_index=point_index,
                    descriptor_index=descriptor_index,
                    descriptor_record_id=record_id,
                    expected_samples=descriptor.number_of_samples,
                    available_samples=(
                        declared_size // descriptor.sample_size_bytes
                        if descriptor.sample_size_bytes
                        else None
                    ),
                    byte_offset=packet_offset,
                    available_bytes=declared_size,
                    message=(
                        f"wavepacket_size is {declared_size} bytes, "
                        f"descriptor requires {packet_size}"
                    ),
                )
            )

        cache_key = (packet_offset, read_length)
        if cache_key not in packet_cache:
            packet_cache[cache_key] = provider.read(
                point_index, packet_offset, read_length
            )
        data, available = packet_cache[cache_key]

        if available == 0:
            acc.issues.append(
                WaveformSampleIssue(
                    kind=WaveformSampleIssueKind.MISSING_PACKET,
                    point_index=point_index,
                    descriptor_index=descriptor_index,
                    descriptor_record_id=record_id,
                    expected_samples=descriptor.number_of_samples,
                    byte_offset=packet_offset,
                    available_bytes=0,
                    message="waveform packet bytes are not available",
                )
            )
            continue

        sample_size = descriptor.sample_size_bytes or 0
        if available < packet_size:
            available_samples = available // sample_size if sample_size else 0
            if not declared_short:
                acc.issues.append(
                    WaveformSampleIssue(
                        kind=WaveformSampleIssueKind.INCOMPLETE_SAMPLES,
                        point_index=point_index,
                        descriptor_index=descriptor_index,
                        descriptor_record_id=record_id,
                        expected_samples=descriptor.number_of_samples,
                        available_samples=int(available_samples),
                        byte_offset=packet_offset,
                        available_bytes=available,
                        message=(
                            f"packet has {available} of {packet_size} bytes "
                            f"({available_samples} of "
                            f"{descriptor.number_of_samples} samples)"
                        ),
                    )
                )
            if available_samples == 0:
                continue
            data = data[: available_samples * sample_size]

        anchor, direction = _point_geometry(
            points, array, names, scales, offsets, point_index
        )
        if anchor is None:
            acc.issues.append(
                WaveformSampleIssue(
                    kind=WaveformSampleIssueKind.MISSING_GEOMETRY,
                    point_index=point_index,
                    descriptor_index=descriptor_index,
                    descriptor_record_id=record_id,
                    expected_samples=descriptor.number_of_samples,
                    byte_offset=packet_offset,
                    message="point has no usable coordinates",
                )
            )
            continue
        if direction is None:
            acc.issues.append(
                WaveformSampleIssue(
                    kind=WaveformSampleIssueKind.MISSING_GEOMETRY,
                    point_index=point_index,
                    descriptor_index=descriptor_index,
                    descriptor_record_id=record_id,
                    expected_samples=descriptor.number_of_samples,
                    byte_offset=packet_offset,
                    message="point has no usable waveform direction vector",
                )
            )
            continue

        anchor_ps_value = attr("return_point_wave_location", point_index)
        anchor_ps = 0.0 if anchor_ps_value is None else float(anchor_ps_value)

        dtype = _sample_dtype(descriptor)
        sample_count = min(descriptor.number_of_samples, len(data) // dtype.itemsize)
        raw = np.frombuffer(data, dtype=dtype, count=sample_count)
        times_ps = (
            np.arange(sample_count, dtype=np.float64)
            * float(descriptor.temporal_sample_spacing)
            - anchor_ps
        )
        xyz = anchor[None, :] + times_ps[:, None] * direction[None, :]
        amplitude = (
            raw.astype(np.float64) * float(descriptor.digitizer_gain)
            + float(descriptor.digitizer_offset)
        )
        acc.add_block(
            point_index=point_index,
            descriptor_index=descriptor_index,
            descriptor_record_id=record_id,
            packet_offset=packet_offset,
            packet_length=packet_size,
            source_packet_offset=(
                packet_offset + (getattr(provider, "payload_offset", 0) or 0)
            ),
            raw=raw,
            amplitude=amplitude,
            times=times_ps * unit_factor,
            xyz=xyz,
            first_sample_offset=packet_offset,
            sample_size=dtype.itemsize,
        )

    table = acc.finish(time_unit)
    close = getattr(provider, "close", None)
    if close is not None:
        close()
    return table


def waveform_source_for_header(header: Any, path: Any) -> Any:
    """Return a packet source for a header, or ``None`` if none is usable."""
    if path is None:
        return None
    try:
        global_encoding = header.global_encoding
        if (
            global_encoding.waveform_data_packets_internal
            and header.start_of_waveform_data_packet_record
        ):
            return WaveformPacketSource.for_internal_las(
                path, header.start_of_waveform_data_packet_record
            )
        if global_encoding.waveform_data_packets_external:
            return WaveformPacketSource.for_external_wdp(path)
    except (AttributeError, ValueError):
        return None
    return None


#: Backwards/forwards friendly alias.
extract_waveform_samples = build_waveform_sample_table


def read_waveform_sample_table(
    source: Any,
    *,
    time_unit: str = LAS_WAVEFORM_TIME_UNIT,
    laz_backend: Any = None,
) -> WaveformSampleTable:
    """Read a LAS/LAZ file and return its waveform sample table.

    The whole point set is read so that every referenced packet is resolved.
    The packet source is located from the header (external ``.wdp`` sibling or
    the internal waveform EVLR), so the returned table is fully materialised
    and the file can be closed immediately afterwards.
    """
    import laspy

    with laspy.open(source, fullwave="never", laz_backend=laz_backend) as reader:
        las = reader.read()
    return las.waveform_samples(time_unit=time_unit)
