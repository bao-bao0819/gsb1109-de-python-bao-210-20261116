"""Tests for crash-safe, idempotent point-cloud publication."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import laspy
from laspy.lasreader import WaveReader
from laspy.point.record import EagerWaveformState, LazyWaveformState
from laspy.publish import (
    AmbiguousAppendError,
    PublicationError,
    PublicationVerificationError,
    _PublicationStore,
    publish_append,
    recover_publishing,
    verify_publication,
)
from laspy.vlrs.known import WaveformPacketStruct, WaveformPacketVlr
from laspy.waveform.record import WaveformRecord

WAVE_SAMPLES = 4
WAVE_SIZE = WAVE_SAMPLES * 2
WAVE_DTYPE = np.dtype([("waveform", (np.uint16, (WAVE_SAMPLES,)))])


def make_points(header, n, base):
    points = laspy.ScaleAwarePointRecord.zeros(n, header=header)
    points.X = np.arange(base, base + n, dtype=np.int64)
    points.Y = np.arange(n, dtype=np.int64)
    points.Z = np.arange(n, dtype=np.int64)
    points.classification = np.full(n, 2, dtype=np.uint8)
    return points


def write_plain_cloud(path: Path) -> laspy.LasData:
    las = laspy.create(point_format=3, file_version="1.2")
    cloud = laspy.LasData(las.header, make_points(las.header, 10, 0))
    cloud.write(str(path))
    return cloud


def descriptor_vlr() -> WaveformPacketVlr:
    vlr = WaveformPacketVlr(100)
    vlr.parsed_record = WaveformPacketStruct(
        bits_per_sample=16,
        waveform_compression_type=0,
        number_of_samples=WAVE_SAMPLES,
        temporal_sample_spacing=10,
        digitizer_gain=1.0,
        digitizer_offset=0.0,
    )
    return vlr


def waveform_points(header, n, base, values, *, lazy_source=None):
    points = laspy.ScaleAwarePointRecord.zeros(n, header=header)
    points.X = np.arange(base, base + n, dtype=np.int64)
    points.array["wavepacket_index"] = 1
    points.array["wavepacket_size"] = WAVE_SIZE
    points.array["wavepacket_offset"] = np.arange(n, dtype=np.uint64) * WAVE_SIZE
    if lazy_source is None:
        samples = np.zeros(n, dtype=WAVE_DTYPE)
        samples["waveform"] = values
        points._set_waveform_state(
            EagerWaveformState(
                WaveformRecord(samples, 10), np.arange(n, dtype=np.int64)
            )
        )
    else:
        reader = WaveReader(
            lazy_source.open("rb"),
            bits_per_sample=16,
            number_of_samples=WAVE_SAMPLES,
            temporal_sample_spacing=10,
            wave_dtype=WAVE_DTYPE,
            closefd=True,
            source_path=lazy_source,
        )
        points._set_waveform_state(LazyWaveformState(reader))
    return points


def write_waveform_cloud(path: Path):
    header = laspy.LasHeader(version="1.4", point_format=laspy.PointFormat(4))
    header.global_encoding.waveform_data_packets_external = True
    header.vlrs.append(descriptor_vlr())
    n = 6
    values = np.arange(n * WAVE_SAMPLES, dtype=np.uint16).reshape(n, WAVE_SAMPLES)
    points = waveform_points(header, n, 0, values)
    cloud = laspy.LasData(header, points)
    cloud.write(str(path), waveforms=False)
    path.with_suffix(".wdp").write_bytes(values.tobytes())
    return values


# --------------------------------------------------------------------- basics
def test_publish_creates_revision_and_is_pointwise_correct(tmp_path):
    path = tmp_path / "cloud.las"
    write_plain_cloud(path)

    header = laspy.read(path).header
    result = publish_append(path, make_points(header, 5, 10), append_id="b1")

    assert result.revision == 1
    assert result.deduped is False
    assert result.point_count == 15
    assert result.appended_count == 5
    assert len(result.las_digest) == 64 and result.wdp_digest is None

    cloud = laspy.read(path)
    expected_x = np.concatenate([np.arange(10), np.arange(10, 15)]).astype(float)
    assert np.array_equal(cloud.X, expected_x)

    store = _PublicationStore(path)
    manifest = store.read_manifest()
    assert manifest["current_revision"] == 1
    assert [record["revision"] for record in manifest["revisions"]] == [0, 1]
    assert manifest["revisions"][0]["point_count"] == 10
    assert verify_publication(path).ok


def test_retry_with_same_append_id_is_idempotent(tmp_path):
    path = tmp_path / "cloud.las"
    write_plain_cloud(path)
    header = laspy.read(path).header
    batch = make_points(header, 4, 30)

    first = publish_append(path, batch, append_id="dup")
    second = publish_append(path, batch, append_id="dup")

    assert first.revision == second.revision == 1
    assert second.deduped is True
    assert laspy.read(path).header.point_count == 14
    # The appended points occur exactly once.
    assert np.sum(laspy.read(path).X >= 30) == 4


def test_generated_append_id_still_publishes(tmp_path):
    path = tmp_path / "cloud.las"
    write_plain_cloud(path)
    result = publish_append(path, make_points(laspy.read(path).header, 2, 7))
    assert result.append_id is not None and result.revision == 1


def test_rejects_empty_batch_and_format_mismatch(tmp_path):
    path = tmp_path / "cloud.las"
    write_plain_cloud(path)
    header = laspy.read(path).header
    empty = laspy.ScaleAwarePointRecord.empty(header=header)
    with pytest.raises(PublicationError):
        publish_append(path, empty, append_id="empty")

    other_header = laspy.LasHeader(version="1.4", point_format=laspy.PointFormat(6))
    foreign = laspy.ScaleAwarePointRecord.zeros(1, header=other_header)
    with pytest.raises(Exception):
        publish_append(path, foreign, append_id="foreign")


def test_concurrent_publication_is_blocked_by_lock(tmp_path):
    path = tmp_path / "cloud.las"
    write_plain_cloud(path)
    store = _PublicationStore(path)
    store.ensure_initialized()
    store.lock_path.parent.mkdir(parents=True, exist_ok=True)
    store.lock_path.write_text("pid=999\n")
    try:
        with pytest.raises(PublicationError):
            publish_append(
                path, make_points(laspy.read(path).header, 1, 1), append_id="lock"
            )
    finally:
        store.lock_path.unlink()


# ------------------------------------------------------------- idempotency /
def test_unknown_outcome_blocks_blind_retry(tmp_path):
    path = tmp_path / "cloud.las"
    write_plain_cloud(path)
    original = path.read_bytes()
    header = laspy.read(path).header

    def boom():
        raise RuntimeError("crash after durable STAGED")

    with pytest.raises(RuntimeError):
        publish_append(
            path, make_points(header, 3, 40), append_id="inflight", on_staged=boom
        )

    # The previous revision is untouched and still readable.
    assert path.read_bytes() == original
    assert laspy.read(path).header.point_count == 10

    with pytest.raises(AmbiguousAppendError):
        publish_append(path, make_points(header, 3, 40), append_id="inflight")
    with pytest.raises(PublicationError):
        publish_append(path, make_points(header, 3, 40), append_id="someone-else")


# ----------------------------------------------------------------- recovery /
def test_recover_promotes_durable_stage(tmp_path):
    path = tmp_path / "cloud.las"
    write_plain_cloud(path)
    header = laspy.read(path).header

    def boom():
        raise RuntimeError("crash")

    with pytest.raises(RuntimeError):
        publish_append(
            path, make_points(header, 4, 50), append_id="crash-a", on_staged=boom
        )

    recovered = recover_publishing(path)
    assert [result.revision for result in recovered] == [1]

    cloud = laspy.read(path)
    assert cloud.header.point_count == 14
    assert np.array_equal(cloud.X[-4:], [50, 51, 52, 53])
    assert verify_publication(path).ok

    # After recovery the same append_id dedupes instead of double counting.
    again = publish_append(path, make_points(header, 4, 50), append_id="crash-a")
    assert again.deduped is True
    assert laspy.read(path).header.point_count == 14


def test_recover_discards_corrupt_stage_and_keeps_old_revision(tmp_path):
    path = tmp_path / "cloud.las"
    write_plain_cloud(path)
    header = laspy.read(path).header

    def boom():
        raise RuntimeError("crash")

    with pytest.raises(RuntimeError):
        publish_append(
            path, make_points(header, 4, 50), append_id="good", on_staged=boom
        )
    recover_publishing(path)
    cloud = laspy.read(path)
    # Snapshot the committed revision that the corrupt attempt must preserve.
    original = path.read_bytes()

    def corrupt_stage():
        for staged in _PublicationStore(path).tmp_dir.glob("rev-000002-*.las"):
            staged.write_bytes(b"half-written point data")
        raise RuntimeError("crash mid write")

    with pytest.raises(RuntimeError):
        publish_append(
            path,
            make_points(cloud.header, 2, 900),
            append_id="corrupt",
            on_staged=corrupt_stage,
        )

    assert recover_publishing(path) == []
    # Old revision stays byte-for-byte readable.
    assert path.read_bytes() == original
    assert laspy.read(path).header.point_count == 14
    assert [event["event"] for event in _PublicationStore(path).read_journal(2)][
        -1
    ] == "ABORTED"

    # A fresh append gets a new, never reused revision number.
    result = publish_append(
        path, make_points(cloud.header, 2, 900), append_id="corrupt-retry"
    )
    assert result.revision == 3
    assert verify_publication(path).ok


def test_recover_after_promotion_before_manifest_commit(tmp_path):
    path = tmp_path / "cloud.las"
    write_plain_cloud(path)
    store = _PublicationStore(path)
    manifest = store.ensure_initialized()
    revision = int(manifest["current_revision"]) + 1
    parent = manifest["revisions"][-1]
    source = laspy.read(path)

    staged = laspy.publish._build_candidate(
        store,
        source,
        make_points(source.header, 1, 5000),
        revision,
        "promoted-only",
        parent,
    )
    staged["revision"] = revision
    laspy.publish._promote(store, staged, revision)
    # Manifest deliberately not updated and no DONE event recorded.

    recovered = recover_publishing(path)
    assert [result.revision for result in recovered] == [revision]
    assert laspy.read(path).X[-1] == 5000
    assert verify_publication(path).ok
    assert publish_append(
        path, make_points(source.header, 1, 5000), append_id="promoted-only"
    ).deduped


def test_recover_discards_attempt_without_stage(tmp_path):
    path = tmp_path / "cloud.las"
    write_plain_cloud(path)
    store = _PublicationStore(path)
    manifest = store.ensure_initialized()
    revision = int(manifest["current_revision"]) + 1
    store.append_journal(
        revision,
        "BEGIN",
        {"append_id": "early-crash", "parent_revision": 0, "requested_count": 1},
    )

    before = laspy.read(path).header.point_count
    assert recover_publishing(path) == []
    assert laspy.read(path).header.point_count == before
    assert [event["event"] for event in store.read_journal(revision)][-1] == (
        "ABORTED"
    )


def test_revision_numbers_are_monotonic_after_aborts(tmp_path):
    path = tmp_path / "cloud.las"
    write_plain_cloud(path)
    header = laspy.read(path).header

    def boom():
        raise RuntimeError("crash")

    with pytest.raises(RuntimeError):
        publish_append(path, make_points(header, 1, 1), append_id="x", on_staged=boom)
    recover_publishing(path)

    result = publish_append(path, make_points(header, 1, 2), append_id="y")
    assert result.revision == 2
    journals = sorted(p.name for p in _PublicationStore(path).journal_dir.glob("*"))
    assert journals == ["000001.log", "000002.log"]


# ------------------------------------------------------------------ waveforms
def test_waveform_append_eager_and_lazy_is_pointwise_consistent(tmp_path):
    path = tmp_path / "cloud.las"
    base_values = write_waveform_cloud(path)

    new_values = (
        np.arange(3 * WAVE_SAMPLES, dtype=np.uint16) + 1000
    ).reshape(3, WAVE_SAMPLES)
    eager = publish_append(
        path,
        waveform_points(laspy.read(path).header, 3, 50, new_values),
        append_id="wf-eager",
    )
    assert eager.wdp_digest is not None

    wdp_path = path.with_suffix(".wdp")
    assert wdp_path.read_bytes() == base_values.tobytes() + new_values.tobytes()

    extra_values = (
        np.arange(2 * WAVE_SAMPLES, dtype=np.uint16) + 5000
    ).reshape(2, WAVE_SAMPLES)
    source_wdp = tmp_path / "extra.wdp"
    source_wdp.write_bytes(extra_values.tobytes())
    lazy_points = waveform_points(
        laspy.read(path).header,
        2,
        80,
        extra_values,
        lazy_source=source_wdp,
    )
    publish_append(path, lazy_points, append_id="wf-lazy")

    cloud = laspy.read(path, fullwave="eager")
    expected = np.concatenate([base_values, new_values, extra_values])
    assert np.array_equal(cloud.points["waveform"], expected)
    # Every point references a packet fully contained in the frozen wdp.
    offsets = cloud.points.array["wavepacket_offset"]
    sizes = cloud.points.array["wavepacket_size"]
    assert np.all(offsets + sizes <= wdp_path.stat().st_size)
    assert verify_publication(path).ok


def test_waveform_crash_keeps_both_files_on_old_revision(tmp_path):
    path = tmp_path / "cloud.las"
    base_values = write_waveform_cloud(path)
    old_las = path.read_bytes()
    old_wdp = path.with_suffix(".wdp").read_bytes()

    more_values = (
        np.arange(WAVE_SAMPLES, dtype=np.uint16) + 9000
    ).reshape(1, WAVE_SAMPLES)
    crash_points = waveform_points(
        laspy.read(path).header, 1, 99, more_values
    )

    def boom():
        raise RuntimeError("crash")

    with pytest.raises(RuntimeError):
        publish_append(path, crash_points, append_id="wf-crash", on_staged=boom)

    # Half-finished publication: both files remain the old revision.
    assert path.read_bytes() == old_las
    assert path.with_suffix(".wdp").read_bytes() == old_wdp
    old_cloud = laspy.read(path, fullwave="eager")
    assert np.array_equal(old_cloud.points["waveform"], base_values)

    recovered = recover_publishing(path)
    assert len(recovered) == 1
    cloud = laspy.read(path, fullwave="eager")
    assert np.array_equal(
        cloud.points["waveform"], np.concatenate([base_values, more_values])
    )
    assert verify_publication(path).ok


def test_first_waveform_append_creates_wdp_and_sets_flags(tmp_path):
    path = tmp_path / "no_waveforms_yet.las"
    header = laspy.LasHeader(version="1.4", point_format=laspy.PointFormat(4))
    header.vlrs.append(descriptor_vlr())
    base = laspy.ScaleAwarePointRecord.zeros(3, header=header)
    base.X = np.arange(3, dtype=np.int64)
    laspy.LasData(header, base).write(str(path), waveforms=False)
    assert not path.with_suffix(".wdp").exists()

    values = np.arange(2 * WAVE_SAMPLES, dtype=np.uint16).reshape(2, WAVE_SAMPLES)
    appended = waveform_points(laspy.read(path).header, 2, 40, values)
    publish_append(path, appended, append_id="first-waveform")

    wdp_path = path.with_suffix(".wdp")
    assert wdp_path.exists() and wdp_path.read_bytes() == values.tobytes()
    cloud = laspy.read(path, fullwave="eager")
    assert cloud.header.global_encoding.waveform_data_packets_external is True
    assert np.all(cloud.points["waveform"][:3] == 0)
    assert np.array_equal(cloud.points["waveform"][3:], values)
    assert verify_publication(path).ok


def test_waveform_points_without_state_are_rejected(tmp_path):
    path = tmp_path / "cloud.las"
    write_waveform_cloud(path)
    header = laspy.read(path).header
    points = laspy.ScaleAwarePointRecord.zeros(1, header=header)
    points.array["wavepacket_index"] = 1
    points.array["wavepacket_size"] = WAVE_SIZE
    with pytest.raises(PublicationError, match="waveform data"):
        publish_append(path, points, append_id="no-state")


def test_recover_rejects_stale_parent_that_would_overwrite_head(tmp_path):
    path = tmp_path / "cloud.las"
    write_plain_cloud(path)
    store = _PublicationStore(path)
    manifest = store.ensure_initialized()

    # First revision commits normally and becomes the chain head.
    publish_append(
        path,
        make_points(laspy.read(path).header, 2, 5),
        append_id="head",
    )
    head = store.read_manifest()["revisions"][-1]
    assert head["revision"] == 1

    # Forge an uncommitted journal whose parent is revision 0, i.e. a stale
    # branch that predates the committed revision 1.
    stale_revision = 2
    source = laspy.read(path)
    stale_staged = laspy.publish._build_candidate(
        store,
        source,
        make_points(source.header, 1, 700),
        stale_revision,
        "stale",
        manifest["revisions"][0],
    )
    stale_staged["revision"] = stale_revision
    # _build_candidate already appended STAGED; no PROMOTED/DONE, so recover
    # must detect the stale parent instead of overwriting revision 1.
    committed_bytes = path.read_bytes()

    assert recover_publishing(path) == []
    assert path.read_bytes() == committed_bytes
    events = [event["event"] for event in store.read_journal(stale_revision)]
    assert events[-1] == "ABORTED"
    assert verify_publication(path).ok


def test_append_without_waveforms_keeps_frozen_wdp(tmp_path):
    path = tmp_path / "cloud.las"
    base_values = write_waveform_cloud(path)
    wdp_path = path.with_suffix(".wdp")
    frozen_wdp = wdp_path.read_bytes()

    header = laspy.read(path).header
    plain = laspy.ScaleAwarePointRecord.zeros(2, header=header)
    plain.X = np.arange(60, 62, dtype=np.int64)
    # wavepacket_index stays 0: these points carry no waveform.
    result = publish_append(path, plain, append_id="no-wf")

    assert wdp_path.read_bytes() == frozen_wdp
    assert result.wdp_digest == _PublicationStore(path).digest_file(wdp_path)
    cloud = laspy.read(path, fullwave="eager")
    assert cloud.header.point_count == 8
    assert np.array_equal(cloud.points["waveform"][:6], base_values)
    assert np.all(cloud.points["waveform"][6:] == 0)
    assert verify_publication(path).ok


def test_recover_reaps_stale_lock_but_refuses_a_live_one(tmp_path):
    path = tmp_path / "cloud.las"
    write_plain_cloud(path)
    store = _PublicationStore(path)
    store.ensure_initialized()
    header = laspy.read(path).header

    def boom():
        raise RuntimeError("crash")

    with pytest.raises(RuntimeError):
        publish_append(
            path,
            make_points(header, 2, 5),
            append_id="locked-crash",
            on_staged=boom,
        )
    # The normal release removed the lock; emulate a crashed process leaving it.
    store.lock_path.write_text("pid=99999999\n")
    recovered = recover_publishing(path)
    assert [result.revision for result in recovered] == [1]
    assert not store.lock_path.exists()
    assert verify_publication(path).ok

    # A lock owned by a live process must not be stolen.
    publish_append(
        path, make_points(laspy.read(path).header, 1, 9), append_id="more"
    )
    store.lock_path.write_text(f"pid={os.getpid()}\n")
    try:
        with pytest.raises(PublicationError, match="live process"):
            recover_publishing(path)
    finally:
        store.lock_path.unlink()


# --------------------------------------------------------------- verification
def test_verify_detects_tampered_las(tmp_path):
    path = tmp_path / "cloud.las"
    write_plain_cloud(path)
    publish_append(
        path, make_points(laspy.read(path).header, 2, 5), append_id="v"
    )
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 0xFF
    path.write_bytes(bytes(raw))
    with pytest.raises(PublicationVerificationError):
        verify_publication(path)


def test_verify_without_manifest_raises(tmp_path):
    path = tmp_path / "cloud.las"
    write_plain_cloud(path)
    with pytest.raises(PublicationError):
        verify_publication(path)


def test_recover_is_idempotent(tmp_path):
    path = tmp_path / "cloud.las"
    write_plain_cloud(path)
    header = laspy.read(path).header

    def boom():
        raise RuntimeError("crash")

    with pytest.raises(RuntimeError):
        publish_append(
            path, make_points(header, 2, 5), append_id="once", on_staged=boom
        )
    first = recover_publishing(path)
    second = recover_publishing(path)
    assert len(first) == 1 and second == []
    assert verify_publication(path).ok
