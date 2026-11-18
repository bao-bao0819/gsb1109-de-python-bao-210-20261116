from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import laspy
from laspy.point.record import EagerWaveformState
from laspy.vlrs.known import WaveformPacketStruct, WaveformPacketVlr
from laspy.waveform import WaveformRecord
from laspy.waveform.descriptor import WaveformPacketDescriptorRegistry
from laspy.waveform.samples import (
    WaveformSampleIssueKind,
    WaveformSampleTable,
    WaveformPacketSource,
    build_waveform_sample_table,
    read_waveform_sample_table,
)


def add_descriptor(
    las,
    *,
    record_id: int = 100,
    bits: int = 8,
    samples: int = 4,
    spacing: int = 500,
    gain: float = 2.0,
    offset: float = 1.0,
    compression: int = 0,
) -> None:
    vlr = WaveformPacketVlr(record_id)
    vlr.parsed_record = WaveformPacketStruct(
        bits_per_sample=bits,
        waveform_compression_type=compression,
        number_of_samples=samples,
        temporal_sample_spacing=spacing,
        digitizer_gain=gain,
        digitizer_offset=offset,
    )
    las.header.vlrs.append(vlr)


def make_las(
    *,
    descriptors: list[dict] | None = None,
    offsets: list[int] | None = None,
    sizes: list[int] | None = None,
    indices: list[int] | None = None,
    locations: list[float] | None = None,
    vectors: list[tuple[float, float, float]] | None = None,
    points_xyz: list[tuple[float, float, float]] | None = None,
    samples: list[list[int]] | None = None,
    with_state: bool = True,
    point_waveform_indices: list[int] | None = None,
    sample_bits: int = 8,
):
    las = laspy.create(point_format=4, file_version="1.3")
    las.header.scales = [0.01, 0.01, 0.01]
    las.header.offsets = [0.0, 0.0, 0.0]
    for descriptor in descriptors or [{}]:
        add_descriptor(las, **descriptor)
    xyz = points_xyz or [(10.0, 11.0, 12.0), (20.0, 21.0, 22.0), (30.0, 31.0, 32.0)]
    count = len(xyz)
    offsets = offsets or [0] * count
    sizes = sizes or [4] * count
    indices = indices or [1] * count
    locations = locations or [0.0] * count
    las.points = laspy.ScaleAwarePointRecord.zeros(count, header=las.header)
    las.x = np.array([p[0] for p in xyz])
    las.y = np.array([p[1] for p in xyz])
    las.z = np.array([p[2] for p in xyz])
    las.points.array["wavepacket_index"] = np.array(indices, dtype=np.uint8)
    las.points.array["wavepacket_offset"] = np.array(offsets, dtype=np.uint64)
    las.points.array["wavepacket_size"] = np.array(sizes, dtype=np.uint32)
    las.points.array["return_point_wave_location"] = np.array(
        locations, dtype=np.float32
    )
    vectors = vectors or [(0.1, 0.0, 0.0)] * count
    las.points.array["x_t"] = np.array([v[0] for v in vectors], dtype=np.float32)
    las.points.array["y_t"] = np.array([v[1] for v in vectors], dtype=np.float32)
    las.points.array["z_t"] = np.array([v[2] for v in vectors], dtype=np.float32)

    if with_state:
        wave_samples = samples or [[1, 2, 3, 4]]
        width = max(len(row) for row in wave_samples)
        base_dtype = np.dtype(f"<u{sample_bits // 8}")
        dtype = np.dtype([("waveform", base_dtype, (width,))])
        raw = np.zeros(len(wave_samples), dtype=dtype)
        for row_index, row in enumerate(wave_samples):
            raw["waveform"][row_index, : len(row)] = row
        indices_for_points = point_waveform_indices or [
            min(int(o) // dtype.itemsize, len(wave_samples) - 1) for o in offsets
        ]
        las.points._set_waveform_state(
            EagerWaveformState(
                WaveformRecord(raw, sample_spacing_ps=500),
                np.array(indices_for_points, dtype=np.int64),
            )
        )
    return las


def test_table_from_eager_packet_reports_geometry_gain_and_provenance() -> None:
    las = make_las(
        offsets=[0, 0, 4],
        points_xyz=[(10.0, 11.0, 12.0), (20.0, 21.0, 22.0), (30.0, 31.0, 32.0)],
        locations=[1000.0, 2000.0, 0.0],
        vectors=[(0.1, 0.0, 0.0), (0.2, 0.0, 0.0), (0.3, 0.0, 0.0)],
        samples=[[1, 2, 3, 4], [5, 6, 7, 8]],
    )

    table = las.waveform_samples()

    assert isinstance(table, WaveformSampleTable)
    assert table.issues == ()
    assert len(table) == 12
    assert table.point_count == 3
    assert table.time_unit == "ps"

    first = table[table.point_index == 0]
    assert first["sample_index"].tolist() == [0, 1, 2, 3]
    assert first["raw_amplitude"].tolist() == [1, 2, 3, 4]
    assert first["time"].tolist() == [-1000.0, -500.0, 0.0, 500.0]
    np.testing.assert_allclose(first["amplitude"], [3.0, 5.0, 7.0, 9.0])
    np.testing.assert_allclose(first["x"], 10.0 + np.array([-1000, -500, 0, 500]) * 0.1)
    np.testing.assert_allclose(first["y"], 11.0)
    np.testing.assert_allclose(first["z"], 12.0)

    second = table[table.point_index == 1]
    assert second["raw_amplitude"].tolist() == [1, 2, 3, 4]
    np.testing.assert_allclose(second["x"], 20.0 + np.array([-2000, -1500, -1000, -500]) * 0.2)

    third = table[table.point_index == 2]
    assert third["raw_amplitude"].tolist() == [5, 6, 7, 8]

    assert table["sample_offset_bytes"][:4].tolist() == [0, 1, 2, 3]
    assert table["packet_byte_offset"][:4].tolist() == [0, 0, 0, 0]
    assert table["source_packet_byte_offset"][:4].tolist() == [0, 0, 0, 0]
    assert table["source_sample_byte_offset"][-1] == 7
    assert table["descriptor_record_id"][0] == 100


def test_shared_packet_reuses_amplitudes_but_not_coordinates() -> None:
    las = make_las(
        offsets=[0, 0],
        points_xyz=[(1.0, 2.0, 3.0), (101.0, 202.0, 303.0)],
        locations=[0.0, 0.0],
        vectors=[(0.1, 0.0, 0.0), (0.1, 0.0, 0.0)],
        samples=[[9, 8, 7, 6]],
    )
    table = las.waveform_samples()
    first = table[table.point_index == 0]
    second = table[table.point_index == 1]
    assert first["raw_amplitude"].tolist() == second["raw_amplitude"].tolist()
    assert not np.allclose(first["x"], second["x"])
    assert first["packet_byte_offset"].tolist() == second["packet_byte_offset"].tolist()


def test_missing_packet_and_unknown_compression_are_reported_separately() -> None:
    las = make_las(
        descriptors=[{}, {"record_id": 101, "compression": 1}],
        offsets=[0, 0, 0, 0],
        indices=[0, 1, 2, 3],
        samples=[[1, 2, 3, 4]],
        points_xyz=[
            (10.0, 11.0, 12.0),
            (20.0, 21.0, 22.0),
            (30.0, 31.0, 32.0),
            (40.0, 41.0, 42.0),
        ],
    )
    table = las.waveform_samples()
    kinds = [issue.kind for issue in table.issues]
    assert WaveformSampleIssueKind.MISSING_PACKET in kinds
    assert WaveformSampleIssueKind.UNKNOWN_COMPRESSION in kinds
    assert WaveformSampleIssueKind.MISSING_DESCRIPTOR in kinds
    assert table.point_index.tolist() == [1, 1, 1, 1]


def test_incomplete_packet_reports_and_keeps_complete_samples() -> None:
    las = make_las(
        offsets=[0],
        sizes=[2],
        locations=[0.0],
        samples=[[1, 2, 3, 4]],
        points_xyz=[(10.0, 11.0, 12.0)],
    )
    table = las.waveform_samples()
    assert len(table.issues) == 1
    issue = table.issues[0]
    assert issue.kind == WaveformSampleIssueKind.INCOMPLETE_SAMPLES
    assert issue.expected_samples == 4
    assert issue.available_samples == 2
    assert len(table) == 2
    assert table["raw_amplitude"].tolist() == [1, 2]
    assert table["sample_offset_bytes"].tolist() == [0, 1]


def test_mixed_descriptors_use_their_own_spacing_and_sample_count() -> None:
    las = make_las(
        descriptors=[
            {"record_id": 100, "samples": 4, "spacing": 500},
            {"record_id": 101, "samples": 2, "spacing": 250},
        ],
        offsets=[0, 4],
        sizes=[4, 2],
        indices=[1, 2],
        locations=[0.0, 0.0],
        points_xyz=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)],
        vectors=[(0.1, 0.0, 0.0), (0.1, 0.0, 0.0)],
        samples=[[1, 2, 3, 4], [5, 6]],
    )
    table = las.waveform_samples()
    first = table[table.point_index == 0]
    second = table[table.point_index == 1]
    assert first["time"].tolist() == [0.0, 500.0, 1000.0, 1500.0]
    assert second["time"].tolist() == [0.0, 250.0]
    assert second["raw_amplitude"].tolist() == [5, 6]


def test_time_unit_translation() -> None:
    las = make_las(
        offsets=[0],
        locations=[0.0],
        samples=[[1, 2, 3, 4]],
        points_xyz=[(10.0, 11.0, 12.0)],
    )
    table = las.waveform_samples(time_unit="ns")
    assert table.time_unit == "ns"
    np.testing.assert_allclose(table["time"], [0.0, 0.5, 1.0, 1.5])


def test_external_wdp_round_trip(tmp_path: Path) -> None:
    las = make_las(
        offsets=[0, 0, 4],
        locations=[0.0, 0.0, 0.0],
        samples=[[1, 2, 3, 4], [5, 6, 7, 8]],
        points_xyz=[(10.0, 11.0, 12.0), (20.0, 21.0, 22.0), (30.0, 31.0, 32.0)],
        vectors=[(0.1, 0.0, 0.0)] * 3,
    )
    path = tmp_path / "samples.las"
    las.write(str(path))
    assert path.with_suffix(".wdp").exists()

    table = read_waveform_sample_table(path)
    assert table.issues == ()
    assert len(table) == 12
    assert table["raw_amplitude"][:4].tolist() == [1, 2, 3, 4]

    loaded = laspy.read(path)
    from_data = loaded.waveform_samples(source=WaveformPacketSource(
        path.with_suffix(".wdp"), payload_offset=0
    ))
    np.testing.assert_array_equal(from_data.raw_amplitude, table.raw_amplitude)
    np.testing.assert_allclose(from_data.xyz, table.xyz)
