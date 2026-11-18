from __future__ import annotations

import logging

import numpy as np
import pytest

from laspy.waveform.record import MISSING_WAVEFORM_INDEX, WaveformRecord


class TruncatableWaveformReader:
    def __init__(
        self,
        waveforms: np.ndarray,
        wave_dtype: np.dtype,
        spacing: int,
        *,
        available_packets: int | None = None,
    ):
        self._waveforms = waveforms
        self.wave_dtype = wave_dtype
        self.number_of_samples = wave_dtype["waveform"].shape[0]
        self.temporal_sample_spacing = spacing
        self.wave_size_bytes = wave_dtype.itemsize
        self.available_packets = (
            len(waveforms) if available_packets is None else available_packets
        )
        self._pos = 0

    def seek(self, waveform_index: int) -> None:
        self._pos = waveform_index

    def read_n_waveforms(self, n: int) -> bytearray:
        available = max(0, self.available_packets - self._pos)
        actual = min(n, available)
        data = self._waveforms[self._pos : self._pos + actual].tobytes()
        self._pos += actual
        if actual != n:
            raise ValueError("short waveform read")
        return bytearray(data)


def make_waveforms(values: list[list[int]]) -> np.ndarray:
    sample_count = len(values[0])
    dtype = np.dtype([("waveform", np.uint8, (sample_count,))])
    waveforms = np.zeros(len(values), dtype=dtype)
    waveforms["waveform"] = values
    return waveforms


def make_points(
    waveform_size: int, offsets: list[int], indices: list[int]
) -> np.ndarray:
    dtype = np.dtype(
        [
            ("wavepacket_size", np.uint32),
            ("wavepacket_offset", np.uint64),
            ("wavepacket_index", np.uint8),
        ]
    )
    points = np.zeros(len(offsets), dtype=dtype)
    points["wavepacket_size"] = waveform_size
    points["wavepacket_offset"] = np.asarray(offsets, dtype=np.uint64)
    points["wavepacket_index"] = np.asarray(indices, dtype=np.uint8)
    return points


def test_from_points_deduplicates_by_offset_not_by_content() -> None:
    # Two offsets share the same sample content, but they are different
    # waveform objects and must not be merged based on their bytes.
    waveforms = make_waveforms([[1, 2], [1, 2], [3, 4]])
    wave_dtype = waveforms.dtype
    size = wave_dtype.itemsize
    reader = TruncatableWaveformReader(waveforms, wave_dtype, spacing=5)
    points = make_points(
        size,
        offsets=[0, size, size * 2],
        indices=[1, 1, 1],
    )

    record, point_to_waveform = WaveformRecord.from_points(points, reader)

    assert len(record) == 3
    assert np.array_equal(point_to_waveform, [0, 1, 2])
    assert np.array_equal(
        record.samples["waveform"], [[1, 2], [1, 2], [3, 4]]
    )


def test_from_points_shared_offset_is_one_waveform_object() -> None:
    waveforms = make_waveforms([[1, 2], [3, 4], [5, 6]])
    wave_dtype = waveforms.dtype
    size = wave_dtype.itemsize
    reader = TruncatableWaveformReader(waveforms, wave_dtype, spacing=1)
    points = make_points(
        size,
        offsets=[size, size, size * 2],
        indices=[1, 1, 1],
    )

    record, point_to_waveform = WaveformRecord.from_points(points, reader)

    assert len(record) == 2
    assert np.array_equal(point_to_waveform, [0, 0, 1])
    assert np.array_equal(record.samples["waveform"], [[3, 4], [5, 6]])


def test_from_points_truncated_source_only_invalidates_referencing_points(
    caplog: pytest.LogCaptureFixture,
) -> None:
    waveforms = make_waveforms([[10, 11], [20, 21], [30, 31], [40, 41]])
    wave_dtype = waveforms.dtype
    size = wave_dtype.itemsize
    # Packet index 2 is the last one available, indexes 3+ are truncated.
    reader = TruncatableWaveformReader(
        waveforms, wave_dtype, spacing=1, available_packets=3
    )
    points = make_points(
        size,
        offsets=[0, size, size * 2, size * 3],
        indices=[1, 1, 1, 1],
    )

    caplog.set_level(logging.WARNING, logger="laspy.waveform.record")
    record, point_to_waveform = WaveformRecord.from_points(points, reader)

    assert np.array_equal(point_to_waveform, [0, 1, 2, MISSING_WAVEFORM_INDEX])
    assert len(record) == 3
    assert np.array_equal(
        record.samples["waveform"], [[10, 11], [20, 21], [30, 31]]
    )
    assert np.array_equal(
        record.samples_at(point_to_waveform)[-1], np.zeros(2, dtype=np.uint8)
    )
    assert any("Could not read 1 waveform packet" in r.message for r in caplog.records)


def test_from_points_gap_inside_run_keeps_later_packets() -> None:
    class HoleReader(TruncatableWaveformReader):
        def read_n_waveforms(self, n: int) -> bytearray:
            if self._pos <= 1 < self._pos + n:
                raise ValueError("hole")
            return super().read_n_waveforms(n)

    waveforms = make_waveforms([[1], [2], [3]])
    wave_dtype = waveforms.dtype
    size = wave_dtype.itemsize
    reader = HoleReader(waveforms, wave_dtype, spacing=1)
    points = make_points(size, offsets=[0, size, size * 2], indices=[1, 1, 1])

    record, point_to_waveform = WaveformRecord.from_points(points, reader)

    assert len(record) == 2
    assert np.array_equal(point_to_waveform, [0, MISSING_WAVEFORM_INDEX, 1])
    assert np.array_equal(record.samples["waveform"], [[1], [3]])


def test_samples_at_handles_empty_mapping() -> None:
    waveforms = make_waveforms([[1, 2]])
    record = WaveformRecord(waveforms, sample_spacing_ps=1)

    samples = record.samples_at(np.array([], dtype=np.int64))

    assert samples.shape == (0, 2)
    assert samples.dtype == np.uint8
