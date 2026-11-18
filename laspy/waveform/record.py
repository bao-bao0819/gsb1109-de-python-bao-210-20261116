import abc
import logging
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from .utils import deduplicate_waveform_indices, iter_runs

logger = logging.getLogger(__name__)

#: Index used in a point -> waveform mapping when the referenced packet could
#: not be read from the external waveform source.
MISSING_WAVEFORM_INDEX = -1

if TYPE_CHECKING:
    from collections.abc import Buffer
else:
    try:
        from collections.abc import Buffer
    except ImportError:  # Python < 3.12

        class Buffer(Protocol):
            """Fallback typing stub for Python < 3.12."""

            ...


class IWaveformReader(abc.ABC):
    @property
    @abc.abstractmethod
    def temporal_sample_spacing(self) -> int: ...

    @property
    @abc.abstractmethod
    def wave_dtype(self) -> np.dtype: ...

    @property
    @abc.abstractmethod
    def wave_size_bytes(self) -> int: ...

    @abc.abstractmethod
    def read_n_waveforms(self, n: int) -> bytearray: ...

    def read_n_waveforms_or_none(self, n: int) -> bytearray | None:
        """Read ``n`` waveform packets, returning ``None`` if they are not
        fully available in the source.

        Readers that can distinguish a truncated packet from an I/O or
        programming error override this method. The default implementation
        keeps backward compatibility with simple readers.
        """
        try:
            data = self.read_n_waveforms(n)
        except (EOFError, OSError, ValueError):
            return None
        if len(data) != n * self.wave_size_bytes:
            return None
        return data

    @abc.abstractmethod
    def seek(self, waveform_index: int) -> None: ...

    @abc.abstractmethod
    def close(self) -> None: ...


def _read_n_waveforms_or_none(
    waveform_reader: IWaveformReader, count: int
) -> bytearray | None:
    """Read at most ``count`` packets, returning ``None`` on truncation."""
    read_or_none = getattr(waveform_reader, "read_n_waveforms_or_none", None)
    if read_or_none is not None:
        try:
            return read_or_none(count)
        except (EOFError, OSError, ValueError):
            return None
    try:
        return waveform_reader.read_n_waveforms(count)
    except (EOFError, OSError, ValueError):
        return None


class WaveformRecord:

    @property
    def wave_size(self) -> int:
        return self.samples.dtype.itemsize

    def __init__(self, data: np.ndarray, sample_spacing_ps: int):
        self.samples = data
        self.sample_spacing_ps = sample_spacing_ps

    @classmethod
    def from_buffer(
        cls,
        buffer: Buffer,
        wave_dtype: np.dtype,
        temporal_sample_spacing: int,
        count=-1,
        offset=0,
    ) -> "WaveformRecord":
        data = np.frombuffer(buffer, dtype=wave_dtype, offset=offset, count=count)
        return cls(data, temporal_sample_spacing)

    @classmethod
    def from_points(
        cls,
        points_array: np.ndarray,
        waveform_reader: IWaveformReader,
    ) -> tuple["WaveformRecord", np.ndarray[Any, np.dtype[np.int64]]]:
        wave_dtype = waveform_reader.wave_dtype
        temporal_sample_spacing = waveform_reader.temporal_sample_spacing
        waveform_size = waveform_reader.wave_size_bytes
        point_count = len(points_array)

        sizes = np.asarray(points_array["wavepacket_size"], dtype=np.uint64)
        if "wavepacket_index" in points_array.dtype.names:
            no_waveform_mask = np.asarray(
                points_array["wavepacket_index"] == 0, dtype=bool
            )
        else:
            no_waveform_mask = np.zeros(point_count, dtype=bool)
        has_waveform_mask = ~no_waveform_mask

        if point_count == 0:
            return (
                cls(np.empty((0,), dtype=wave_dtype), temporal_sample_spacing),
                np.empty((0,), dtype=np.int64),
            )

        if not has_waveform_mask.any():
            return (
                cls(np.zeros((1,), dtype=wave_dtype), temporal_sample_spacing),
                np.zeros(point_count, dtype=np.int64),
            )

        actual = set(sizes[has_waveform_mask])
        if actual - {waveform_size}:
            raise ValueError(
                f"Inconsistent waveform sizes in point data: {actual} but descriptor size is {waveform_size}"
            )

        offsets = np.asarray(points_array["wavepacket_offset"], dtype=np.uint64)
        unique_indices, inverse_indices = deduplicate_waveform_indices(
            offsets, has_waveform_mask, waveform_size
        )

        waveform_data = bytearray()
        # Maps a position in `unique_indices` to its position in the packed
        # waveform record, or -1 when that packet could not be read.
        unique_packet_indices = np.full(
            len(unique_indices), MISSING_WAVEFORM_INDEX, dtype=np.int64
        )
        next_waveform_index = 0
        missing_packet_count = 0

        for start, end in iter_runs(unique_indices):
            count = int(end - start + 1)
            positions_start, positions_end = np.searchsorted(
                unique_indices, np.asarray([start, end], dtype=unique_indices.dtype)
            )
            positions_end += 1
            waveform_reader.seek(int(start))
            data = _read_n_waveforms_or_none(waveform_reader, count)
            if data is not None and len(data) == count * waveform_size:
                waveform_data.extend(data)
                unique_packet_indices[positions_start:positions_end] = np.arange(
                    next_waveform_index,
                    next_waveform_index + count,
                    dtype=np.int64,
                )
                next_waveform_index += count
                continue

            # The external waveform source does not contain the whole run.
            # Retry packet by packet so that a missing segment only affects
            # the points referencing that segment, not the other points.
            for position, packet_index in zip(
                range(int(positions_start), int(positions_end)),
                range(int(start), int(end) + 1),
            ):
                waveform_reader.seek(packet_index)
                packet = _read_n_waveforms_or_none(waveform_reader, 1)
                if packet is None or len(packet) != waveform_size:
                    missing_packet_count += 1
                    continue
                waveform_data.extend(packet)
                unique_packet_indices[position] = next_waveform_index
                next_waveform_index += 1

        if missing_packet_count:
            logger.warning(
                "Could not read %d waveform packet(s) from the external waveform "
                "source; the referencing points are marked as having no waveform",
                missing_packet_count,
            )

        waveforms = cls.from_buffer(
            waveform_data,
            wave_dtype,
            temporal_sample_spacing,
            count=next_waveform_index,
        )

        points_waveform_index = np.full(
            point_count, MISSING_WAVEFORM_INDEX, dtype=np.int64
        )
        points_waveform_index[has_waveform_mask] = unique_packet_indices[
            inverse_indices
        ]

        if not np.all(has_waveform_mask):
            missing_wave_index = len(waveforms)
            new_samples = np.empty((missing_wave_index + 1,), dtype=wave_dtype)
            if len(waveforms):
                new_samples[:-1] = waveforms.samples
            new_samples["waveform"][-1] = 0
            waveforms = cls(new_samples, temporal_sample_spacing)
            points_waveform_index[~has_waveform_mask] = missing_wave_index

        return waveforms, np.asarray(points_waveform_index, dtype=np.int64)

    def __len__(self):
        if self.samples.ndim == 0:
            return 1
        return self.samples.shape[0]

    def samples_at(
        self, indices: np.ndarray[Any, np.dtype[np.int64]]
    ) -> np.ndarray:
        """Return waveform sample arrays for point-to-waveform indices.

        Points whose index is :data:`MISSING_WAVEFORM_INDEX` get an all-zero
        sample array, which keeps point geometry usable even when the external
        waveform file is truncated.
        """
        samples = self.samples["waveform"]
        indices = np.asarray(indices, dtype=np.int64)
        result = np.zeros((indices.shape[0], *samples.shape[1:]), dtype=samples.dtype)
        valid_mask = indices != MISSING_WAVEFORM_INDEX
        if np.any(valid_mask):
            result[valid_mask] = samples[indices[valid_mask]]
        return result

    def __repr__(self):
        return f"<{self.__class__.__name__}(len: {len(self)}, num samples: {self.samples.dtype['waveform'].shape[0]}, type: {self.samples.dtype['waveform'].base}, size: {self.wave_size} bytes)>"

    def __getitem__(self, item):
        """Gives access to the underlying numpy array
        Unpack the dimension if item is the name a sub-field
        """
        if isinstance(item, (int, slice, np.ndarray, list, tuple)):
            return WaveformRecord(self.samples[item], self.sample_spacing_ps)

        return self.samples[item]
