from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

import laspy
from laspy.lasdata import LasData
from laspy.point.record import EagerWaveformState, LazyWaveformState
from laspy.vlrs.known import WaveformPacketStruct, WaveformPacketVlr
from laspy.waveform.descriptor import WaveformPacketDescriptorRegistry
from laspy.waveform.record import WaveformRecord

FULLWAVE_LAZ_PATH = Path(__file__).parent / "data" / "fullwave.laz"
FULLWAVE_WDP_PATH = FULLWAVE_LAZ_PATH.with_suffix(".wdp")


@pytest.fixture()
def fullwave_path() -> Path:
    if not FULLWAVE_LAZ_PATH.exists() or not FULLWAVE_WDP_PATH.exists():
        pytest.skip("Missing fullwave test data")
    if len(laspy.LazBackend.detect_available()) == 0:
        pytest.skip("No Laz Backend")
    return FULLWAVE_LAZ_PATH


def assert_waveform_write_flags_updated(las: LasData) -> None:
    assert las.header.global_encoding.waveform_data_packets_external is True
    assert las.header.global_encoding.waveform_data_packets_internal is False
    if las.header.version.minor >= 3:
        assert las.header.start_of_waveform_data_packet_record == 0


def assert_waveform_roundtrip_matches(expected: LasData, actual: LasData) -> None:
    assert np.array_equal(actual.points.array, expected.points.array)
    assert np.array_equal(actual.points["waveform"], expected.points["waveform"])


def test_lasdata_write_without_waveforms_does_not_create_wdp(
    simple_las_path: Path, tmp_path: Path
) -> None:
    las = laspy.read(simple_las_path)
    wf_las = LasData(las.header, las.points)

    assert wf_las["X"].shape[0] == len(wf_las)

    out_path = tmp_path / "no_waveforms.las"
    wf_las.write(str(out_path))
    assert out_path.exists()
    assert not out_path.with_suffix(".wdp").exists()


def test_waveform_lasdata_write_eager_roundtrip(
    fullwave_path: Path, tmp_path: Path
) -> None:
    with laspy.open(fullwave_path, fullwave="eager") as reader:
        las = reader.read()

    out_path = tmp_path / "fullwave_eager_roundtrip.laz"
    las.header.global_encoding.waveform_data_packets_internal = True
    las.write(str(out_path))

    out_wdp = out_path.with_suffix(".wdp")
    assert out_path.exists()
    assert out_wdp.exists()

    with laspy.open(out_path, fullwave="eager") as reader:
        roundtrip = reader.read()

    assert_waveform_roundtrip_matches(las, roundtrip)
    assert_waveform_write_flags_updated(las)


def test_waveform_lasdata_write_lazy_roundtrip(
    fullwave_path: Path, tmp_path: Path
) -> None:
    with laspy.open(fullwave_path, fullwave="lazy") as reader:
        las = reader.read()

    with laspy.open(fullwave_path, fullwave="eager") as reader:
        expected = reader.read()

    out_path = tmp_path / "fullwave_lazy_roundtrip.laz"
    las.header.global_encoding.waveform_data_packets_internal = True
    las.write(str(out_path))

    out_wdp = out_path.with_suffix(".wdp")
    assert out_path.exists()
    assert out_wdp.exists()

    with laspy.open(out_path, fullwave="eager") as reader:
        roundtrip = reader.read()

    assert np.array_equal(roundtrip.points.array, las.points.array)
    assert np.array_equal(roundtrip.points["waveform"], expected.points["waveform"])
    assert_waveform_write_flags_updated(las)


def test_waveform_lasdata_write_lazy_subset_deduplicates_waveforms(
    fullwave_path: Path, tmp_path: Path
) -> None:
    with laspy.open(fullwave_path, fullwave="lazy") as reader:
        las = reader.read()
        indices = np.arange(0, len(las.points), 7)
        subset = las[indices]
        registry = WaveformPacketDescriptorRegistry.from_vlrs(subset.header.vlrs)
        wave_dtype = registry.dtype()
        assert wave_dtype is not None
        expected_wdp_size = (
            np.unique(subset.points.array["wavepacket_offset"]).size
            * wave_dtype.itemsize
        )

        subset.header.global_encoding.waveform_data_packets_internal = True
        out_path = tmp_path / "subset_fullwave.laz"
        subset.write(str(out_path))

        out_wdp = out_path.with_suffix(".wdp")
        assert out_path.exists()
        assert out_wdp.exists()
        assert out_wdp.stat().st_size == expected_wdp_size

        with laspy.open(out_path, fullwave="eager") as roundtrip_reader:
            roundtrip = roundtrip_reader.read()

    with laspy.open(fullwave_path, fullwave="eager") as expected_reader:
        expected = expected_reader.read()

    assert np.array_equal(roundtrip.points.array, subset.points.array)
    assert np.array_equal(
        roundtrip.points["waveform"],
        expected.points["waveform"][indices],
    )
    assert_waveform_write_flags_updated(subset)


def test_waveform_lasdata_write_preserves_points_without_waveforms(
    fullwave_path: Path, tmp_path: Path
) -> None:
    with laspy.open(fullwave_path, fullwave="eager") as reader:
        las = reader.read()

    las.header.global_encoding.waveform_data_packets_internal = True

    las.points.array["wavepacket_index"][0] = 0
    las.points.array["wavepacket_size"][0] = 0
    las.points.array["wavepacket_offset"][0] = 0

    out_path = tmp_path / "missing_first_waveform.laz"
    las.write(str(out_path))

    with laspy.open(out_path, fullwave="eager") as reader:
        points = reader.read_points(4)

    assert points.array["wavepacket_index"][0] == 0
    assert points.array["wavepacket_size"][0] == 0
    assert points.array["wavepacket_offset"][0] == 0
    assert np.all(points["waveform"][0] == 0)
    out_wdp = out_path.with_suffix(".wdp")
    assert out_wdp.exists()
    assert_waveform_write_flags_updated(las)


def test_waveform_lasdata_write_without_valid_waveforms_clears_metadata(
    fullwave_path: Path, tmp_path: Path
) -> None:
    with laspy.open(fullwave_path, fullwave="eager") as reader:
        las = reader.read()

    las.header.global_encoding.waveform_data_packets_external = True
    las.header.global_encoding.waveform_data_packets_internal = True

    las.points.array["wavepacket_index"][:] = 0
    las.points.array["wavepacket_size"][:] = 123
    las.points.array["wavepacket_offset"][:] = 456

    out_path = tmp_path / "eager_all_no_waveforms.laz"
    las.write(out_path)

    assert out_path.exists()
    assert not out_path.with_suffix(".wdp").exists()
    assert np.all(las.points.array["wavepacket_size"] == 0)
    assert np.all(las.points.array["wavepacket_offset"] == 0)
    assert las.header.global_encoding.waveform_data_packets_external is False
    assert las.header.global_encoding.waveform_data_packets_internal is False
    if las.header.version.minor >= 3:
        assert las.header.start_of_waveform_data_packet_record == 0

    roundtrip = laspy.read(out_path)
    assert np.all(roundtrip.points.array["wavepacket_index"] == 0)
    assert np.all(roundtrip.points.array["wavepacket_size"] == 0)
    assert np.all(roundtrip.points.array["wavepacket_offset"] == 0)


def test_waveform_lasdata_write_rejects_binaryio(fullwave_path: Path) -> None:
    with laspy.open(fullwave_path, fullwave="eager") as reader:
        las = reader.read()

    with pytest.raises(NotImplementedError):
        las.write(io.BytesIO())


def test_waveform_lasdata_write_rejects_unaligned_lazy_offsets(
    fullwave_path: Path, tmp_path: Path
) -> None:
    with laspy.open(fullwave_path, fullwave="lazy") as reader:
        las = reader.read()

    las.points.array["wavepacket_offset"][0] += 1

    with pytest.raises(NotImplementedError, match="byte offset"):
        las.write(tmp_path / "unaligned_offset.laz")


### Private API tests ###


def test_lazy_write_dedup_all_no_waveforms(fullwave_path: Path, tmp_path: Path) -> None:
    with laspy.open(fullwave_path, fullwave="lazy") as reader:
        las = reader.read()
        lazy_state = las.points._waveform_state
        assert isinstance(lazy_state, LazyWaveformState)
        wave_size = lazy_state.reader.wave_size_bytes
        las.points.array["wavepacket_index"][:] = 0
        las.points.array["wavepacket_size"][:] = 0
        las.points.array["wavepacket_offset"][:] = 123
        has_waveform_mask = np.asarray(
            las.points.array["wavepacket_index"] != 0, dtype=bool
        )

        out_path = tmp_path / "dedup_all_no_waveforms.laz"
        las._write_wdp_lazy(
            destination=out_path,
            lazy_state=lazy_state,
            has_waveform_mask=has_waveform_mask,
            waveform_size=wave_size,
            chunksize=1024,
        )

    out_wdp = out_path.with_suffix(".wdp")
    assert out_wdp.exists()
    assert out_wdp.stat().st_size == 0
    assert np.all(las.points.array["wavepacket_offset"] == 0)
    assert np.all(las.points.array["wavepacket_size"] == 0)
    assert las.header.global_encoding.waveform_data_packets_external is False


def test_write_waveforms_without_wavepacket_index_dimension_raises(
    simple_las_path: Path, tmp_path: Path
) -> None:
    las = laspy.read(simple_las_path)
    out_path = tmp_path / "no_wavepacket_index.las"
    with pytest.raises(ValueError, match="wavepacket_index"):
        las._write_waveforms(out_path, waveform_chunksize=1024)
    assert not out_path.with_suffix(".wdp").exists()


def test_write_wdp_no_waveforms_noop(tmp_path: Path) -> None:
    path = tmp_path / "noop.wdp"
    LasData._write_wdp(path, None)
    assert not path.exists()


def test_write_wdp_lazy_errors(fullwave_path: Path, tmp_path: Path) -> None:
    with laspy.open(fullwave_path, fullwave="lazy") as reader:
        las = reader.read()
        lazy_state = las.points._waveform_state
        assert isinstance(lazy_state, LazyWaveformState)
        wave_size = lazy_state.reader.wave_size_bytes

        with pytest.raises(ValueError):
            las._write_wdp_lazy(
                destination=tmp_path / "bad_chunk.laz",
                lazy_state=lazy_state,
                has_waveform_mask=np.asarray(
                    las.points.array["wavepacket_index"] != 0, dtype=bool
                ),
                waveform_size=wave_size,
                chunksize=0,
            )


def test_write_wdp_lazy_inconsistent_sizes(fullwave_path: Path, tmp_path: Path) -> None:
    with laspy.open(fullwave_path, fullwave="lazy") as reader:
        las = reader.read()
        lazy_state = las.points._waveform_state
        assert isinstance(lazy_state, LazyWaveformState)
        wave_size = lazy_state.reader.wave_size_bytes
        las.points.array["wavepacket_size"][0] = wave_size + 1
        with pytest.raises(ValueError):
            las._write_wdp_lazy(
                destination=tmp_path / "bad_size.laz",
                lazy_state=lazy_state,
                has_waveform_mask=np.asarray(
                    las.points.array["wavepacket_index"] != 0, dtype=bool
                ),
                waveform_size=wave_size,
                chunksize=1024,
            )


def test_write_wdp_lazy_empty_points(fullwave_path: Path, tmp_path: Path) -> None:
    with laspy.open(fullwave_path, fullwave="lazy") as reader:
        las = reader.read()
        empty = las[:0]
        lazy_state = empty.points._waveform_state
        assert isinstance(lazy_state, LazyWaveformState)
        wave_size = lazy_state.reader.wave_size_bytes

        out_path = tmp_path / "empty.laz"
        empty._write_wdp_lazy(
            destination=out_path,
            lazy_state=lazy_state,
            has_waveform_mask=np.asarray(
                empty.points.array["wavepacket_index"] != 0, dtype=bool
            ),
            waveform_size=wave_size,
            chunksize=1024,
        )

    out_wdp = out_path.with_suffix(".wdp")
    assert out_wdp.exists()
    assert out_wdp.stat().st_size == 0


def test_eager_write_deduplicates_shared_waveform_objects_by_reference(
    tmp_path: Path,
) -> None:
    las = _make_waveform_las()
    # Points 0 and 1 share one packet object; points 2 and 3 reference two
    # distinct objects whose sample bytes are identical.
    eager_state = las.points._waveform_state
    assert isinstance(eager_state, EagerWaveformState)
    eager_state.waveforms.samples["waveform"][:] = [
        [1, 2],
        [1, 2],
        [3, 4],
        [9, 9],
    ]
    eager_state.points_waveform_index[:] = np.array([0, 0, 1, 2], dtype=np.int64)

    out_path = tmp_path / "shared_by_reference.las"
    las.write(out_path)

    assert out_path.with_suffix(".wdp").stat().st_size == 3 * 2
    assert las.points.array["wavepacket_offset"].tolist() == [0, 0, 2, 4]
    assert las.points.array["wavepacket_size"].tolist() == [2, 2, 2, 2]


def test_eager_write_clears_points_with_missing_packet_reference(
    tmp_path: Path,
) -> None:
    las = _make_waveform_las()
    eager_state = las.points._waveform_state
    assert isinstance(eager_state, EagerWaveformState)
    eager_state.points_waveform_index[1] = -1

    out_path = tmp_path / "missing_ref.las"
    las.write(out_path)

    assert las.points.array["wavepacket_index"].tolist() == [1, 0, 1, 1]
    assert las.points.array["wavepacket_offset"].tolist() == [0, 0, 2, 4]
    assert las.points.array["wavepacket_size"].tolist() == [2, 0, 2, 2]
    assert out_path.with_suffix(".wdp").stat().st_size == 3 * 2

    with laspy.open(out_path, fullwave="eager") as reader:
        points = reader.read_points(-1)
    assert np.array_equal(points.array["X"], las.points.array["X"])
    assert np.all(points["waveform"][1] == 0)


def _make_waveform_las() -> LasData:
    las = laspy.create(point_format=10, file_version="1.4")
    las.header.global_encoding.waveform_data_packets_external = True
    las.header.global_encoding.waveform_data_packets_internal = False

    record = WaveformPacketStruct(
        bits_per_sample=8,
        waveform_compression_type=0,
        number_of_samples=2,
        temporal_sample_spacing=1,
        digitizer_gain=1.0,
        digitizer_offset=0.0,
    )
    vlr = WaveformPacketVlr(100)
    vlr.parsed_record = record
    las.header.vlrs.append(vlr)

    point_count = 4
    las.points.resize(point_count)
    las.points.array["X"] = np.arange(point_count)
    las.points.array["wavepacket_index"] = 1
    las.points.array["wavepacket_size"] = 2
    las.points.array["wavepacket_offset"] = np.array([0, 0, 2, 4], dtype=np.uint64)

    samples = np.zeros(
        point_count, dtype=np.dtype([("waveform", np.uint8, (2,))])
    )
    samples["waveform"] = np.arange(point_count * 2, dtype=np.uint8).reshape(
        point_count, 2
    )
    las.points._set_waveform_state(
        EagerWaveformState(
            WaveformRecord(samples, sample_spacing_ps=1),
            np.array([0, 0, 1, 2], dtype=np.int64),
        )
    )
    return las


def test_lazy_write_after_truncated_source_only_drops_missing_references(
    tmp_path: Path,
) -> None:
    las = _make_waveform_las()
    source_path = tmp_path / "source.las"
    las.write(source_path)
    # Truncate the generated waveform file to its first two packets.
    source_wdp = source_path.with_suffix(".wdp")
    source_wdp.write_bytes(source_wdp.read_bytes()[:4])

    with laspy.open(source_path, fullwave="lazy") as reader:
        lazy_las = reader.read()

    out_path = tmp_path / "lazy_truncated_copy.las"
    lazy_las.write(out_path)

    assert lazy_las.points.array["wavepacket_index"].tolist() == [1, 1, 1, 0]
    assert lazy_las.points.array["wavepacket_size"].tolist() == [2, 2, 2, 0]
    assert lazy_las.points.array["wavepacket_offset"].tolist() == [0, 0, 2, 0]
    assert out_path.with_suffix(".wdp").stat().st_size == 4

    with laspy.open(out_path, fullwave="eager") as reader:
        points = reader.read_points(-1)
    assert np.array_equal(points.array["X"], np.arange(4))
    assert np.all(points["waveform"][3] == 0)


def test_lazy_write_rewinds_partially_copied_run_when_source_truncates(
    tmp_path: Path,
) -> None:
    las = _make_waveform_las()
    # Five packets in one contiguous run. Chunk size covers two packets, so
    # the first two chunks succeed before the third one hits the truncation.
    las.points.resize(5)
    las.points.array["X"] = np.arange(5)
    las.points.array["wavepacket_index"] = 1
    las.points.array["wavepacket_size"] = 2
    las.points.array["wavepacket_offset"] = np.array([0, 2, 4, 6, 8], dtype=np.uint64)
    state = las.points._waveform_state
    assert isinstance(state, EagerWaveformState)
    state.points_waveform_index[:] = np.arange(5, dtype=np.int64)

    source_path = tmp_path / "chunked_source.las"
    las.write(source_path)
    source_wdp = source_path.with_suffix(".wdp")
    # Keep 4 packets out of 5; chunked reads will succeed twice then fail.
    source_wdp.write_bytes(source_wdp.read_bytes()[:8])

    with laspy.open(source_path, fullwave="lazy") as reader:
        lazy_las = reader.read()

    out_path = tmp_path / "chunked_copy.las"
    lazy_las.write(out_path, waveform_chunksize=4)

    assert out_path.with_suffix(".wdp").stat().st_size == 8
    assert np.array_equal(
        lazy_las.points.array["wavepacket_offset"], [0, 2, 4, 6, 0]
    )
    assert np.array_equal(lazy_las.points.array["wavepacket_size"], [2, 2, 2, 2, 0])
