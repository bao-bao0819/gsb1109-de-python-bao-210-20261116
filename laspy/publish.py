"""Crash-safe publication of point-cloud append operations.

This module turns an append into an *atomic publication*:

* the new revision is fully built in temporary files (point records, waveform
  packets and the recomputed header statistics all come from the same merged
  point set),
* the previous revision stays byte-for-byte readable until the new one is
  promoted with atomic renames,
* every step is recorded in a durable journal and a manifest, so a crash is
  either rolled forward (the staged bytes are digest-checked first) or cleaned
  up,
* the caller supplied ``append_id`` makes retries idempotent: an already
  published revision is returned as-is instead of being counted twice, while a
  revision whose outcome is unknown is never blindly re-appended.

The published bytes are content-addressed by their SHA-256 digests in the
manifest, so every revision recovered locally is byte identical to the frozen
revision that was originally committed and can be verified point by point.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Optional

import numpy as np

from . import errors
from .header import LasHeader
from .lasdata import LasData
from .point.record import (
    EagerWaveformState,
    LazyWaveformState,
    PackedPointRecord,
    ScaleAwarePointRecord,
)

MANIFEST_NAME = "manifest.json"
JOURNAL_DIR_NAME = "journal"
TMP_DIR_NAME = "tmp"
LOCK_NAME = "lock"
MANIFEST_SCHEMA = 1

_DIGEST_CHUNK = 1024 * 1024


class PublicationError(errors.LaspyException):
    """Base class for publication / recovery failures."""


class AmbiguousAppendError(PublicationError):
    """Raised when an append_id already has an in-flight, unresolved journal.

    The outcome of that append is unknown: callers must run
    :func:`recover_publishing` instead of retrying, otherwise points could be
    counted twice.
    """


class PublicationVerificationError(PublicationError):
    """Raised when a published revision does not match its frozen record."""


@dataclass(frozen=True)
class PublicationResult:
    """Summary of a published (or already known) revision."""

    revision: int
    append_id: str
    point_count: int
    appended_count: int
    las_path: Path
    wdp_path: Optional[Path]
    las_digest: str
    wdp_digest: Optional[str]
    deduped: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "append_id": self.append_id,
            "point_count": self.point_count,
            "appended_count": self.appended_count,
            "las_path": str(self.las_path),
            "wdp_path": str(self.wdp_path) if self.wdp_path is not None else None,
            "las_digest": self.las_digest,
            "wdp_digest": self.wdp_digest,
            "deduped": self.deduped,
        }


@dataclass
class VerificationReport:
    """Result of :func:`verify_publication`."""

    revision: int
    las_digest: str
    wdp_digest: Optional[str]
    point_count: int
    checks: dict[str, bool] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues and bool(self.checks) and all(self.checks.values())


class _PublicationStore:
    """On-disk layout and durable bookkeeping for one LAS/LAZ file."""

    def __init__(self, las_path: str | Path) -> None:
        self.las_path = Path(las_path)
        if not self.las_path.is_file():
            raise PublicationError(f"LAS file does not exist: {self.las_path}")
        self.wdp_path = self.las_path.with_suffix(".wdp")
        if not self.wdp_path.is_file():
            self.wdp_path = None
        self.root = self.las_path.with_name(self.las_path.name + ".publish")
        self.journal_dir = self.root / JOURNAL_DIR_NAME
        self.tmp_dir = self.root / TMP_DIR_NAME
        self.manifest_path = self.root / MANIFEST_NAME
        self.lock_path = self.root / LOCK_NAME

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _fsync_file(handle: BinaryIO) -> None:
        handle.flush()
        os.fsync(handle.fileno())

    @staticmethod
    def _fsync_dir(path: Path) -> None:
        flags = os.O_RDONLY
        fd = os.open(path, flags)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def digest_file(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as src:
            for chunk in iter(lambda: src.read(_DIGEST_CHUNK), b""):
                digest.update(chunk)
        return digest.hexdigest()

    # --------------------------------------------------------------- manifest
    def read_manifest(self) -> dict[str, Any]:
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        tmp_path = self.manifest_path.with_name(
            f"{MANIFEST_NAME}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        )
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            self._fsync_file(handle)
        os.replace(tmp_path, self.manifest_path)
        self._fsync_dir(self.root)

    def ensure_initialized(self) -> dict[str, Any]:
        """Create the store and record the pre-existing file as revision 0."""
        if self.manifest_path.is_file():
            return self.read_manifest()

        self.root.mkdir(parents=True, exist_ok=True)
        self.journal_dir.mkdir(exist_ok=True)
        self.tmp_dir.mkdir(exist_ok=True)

        from . import read as read_las

        initial = read_las(self.las_path)
        point_count = int(initial.header.point_count)
        las_digest = self.digest_file(self.las_path)
        # A stray .wdp next to a non-waveform point format is not part of the
        # LAS data model and must not be tracked as part of any revision.
        tracks_wdp = (
            initial.header.point_format.has_waveform_packet
            and self.wdp_path is not None
        )
        wdp_digest = (
            self.digest_file(self.wdp_path) if tracks_wdp else None
        )

        manifest = {
            "schema": MANIFEST_SCHEMA,
            "las_path": str(self.las_path),
            "current_revision": 0,
            "revisions": [
                {
                    "revision": 0,
                    "append_id": None,
                    "parent_revision": None,
                    "point_count": point_count,
                    "appended_count": 0,
                    "las_digest": las_digest,
                    "wdp_digest": wdp_digest,
                    "appended_points_digest": None,
                    "parent_las_digest": None,
                    "parent_wdp_digest": None,
                    "created_at": self._utc_now(),
                    "status": "committed",
                }
            ],
        }
        self.write_manifest(manifest)
        self._fsync_dir(self.las_path.parent)
        return manifest

    # ---------------------------------------------------------------- journal
    def journal_path(self, revision: int) -> Path:
        return self.journal_dir / f"{revision:06d}.log"

    def append_journal(
        self, revision: int, event: str, payload: dict[str, Any]
    ) -> None:
        record = {
            "ts": self._utc_now(),
            "event": event,
            "revision": revision,
            **payload,
        }
        path = self.journal_path(revision)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            self._fsync_file(handle)
        self._fsync_dir(self.journal_dir)

    def read_journal(self, revision: int) -> list[dict[str, Any]]:
        path = self.journal_path(revision)
        if not path.is_file():
            return []
        events = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events

    # ------------------------------------------------------------------- lock
    def _lock_owner_pid(self) -> Optional[int]:
        try:
            text = self.lock_path.read_text(encoding="utf-8")
        except OSError:
            return None
        for line in text.splitlines():
            if line.startswith("pid="):
                try:
                    return int(line.split("=", 1)[1].strip())
                except ValueError:
                    return None
        return None

    def reap_stale_lock(self) -> None:
        """Remove a lock left by a dead process; refuse to touch a live lock.

        Only an explicit recovery call reaches this path, so a lock owned by a
        running process is never stolen.
        """
        if not self.lock_path.exists():
            return
        pid = self._lock_owner_pid()
        if pid is None:
            raise PublicationError(
                f"Lock {self.lock_path} has no parseable owner pid; "
                "inspect and remove it manually"
            )
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            with contextlib.suppress(FileNotFoundError):
                self.lock_path.unlink()
            self._fsync_dir(self.root)
        except PermissionError as error:
            raise PublicationError(
                f"Lock {self.lock_path} is owned by process {pid} which is "
                "alive but cannot be signalled"
            ) from error
        except OSError as error:
            # Platforms where signal 0 cannot probe liveness: never steal.
            raise PublicationError(
                f"Cannot determine whether lock owner pid {pid} is alive; "
                "inspect and remove the lock manually"
            ) from error
        else:
            raise PublicationError(
                f"Lock {self.lock_path} is held by the live process {pid}; "
                "recovery would interfere with an in-progress publication"
            )

    @contextlib.contextmanager
    def locked(self) -> Any:
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(
                self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY
            )
        except FileExistsError as error:
            raise PublicationError(
                f"Another publication is in progress ({self.lock_path}). "
                "Run recover_publishing() if the previous process crashed."
            ) from error
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(f"pid={os.getpid()}\n")
                handle.flush()
                os.fsync(handle.fileno())
            yield
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(self.lock_path)
            self._fsync_dir(self.root)


# --------------------------------------------------------------------- helpers
def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalize_points(
    points: PackedPointRecord, header: LasHeader
) -> PackedPointRecord:
    """Return a record whose raw integer coordinates match the file header.

    The record type (including its scales/offsets and waveform state) is
    preserved; only the raw X/Y/Z are re-scaled onto the target header.
    """
    if not isinstance(points, ScaleAwarePointRecord):
        return points
    if np.all(points.scales == header.scales) and np.all(
        points.offsets == header.offsets
    ):
        return points

    rescaled = ScaleAwarePointRecord(
        points.array.copy(),
        points.point_format,
        scales=points.scales,
        offsets=points.offsets,
        waveform_state=points._waveform_state,
    )
    rescaled.change_scaling(scales=header.scales, offsets=header.offsets)
    return rescaled


def _waveform_packet_bytes(
    points: PackedPointRecord,
    wave_size: int,
) -> list[bytes]:
    """Collect the waveform packets referenced by the points, in order."""
    array = points.array
    has_mask = np.asarray(array["wavepacket_index"] != 0, dtype=bool)
    if not has_mask.any():
        return []

    sizes = np.asarray(array["wavepacket_size"], dtype=np.uint64)[has_mask]
    unexpected = set(int(size) for size in np.unique(sizes)) - {wave_size}
    if unexpected:
        raise PublicationError(
            "Appended points reference waveform packets with inconsistent sizes: "
            f"{sorted(unexpected)} but the descriptor size is {wave_size}"
        )

    state = points._waveform_state
    packets: list[bytes] = []

    if isinstance(state, EagerWaveformState):
        samples = state.waveforms.samples
        indices = state.points_waveform_index
        for point_index in np.flatnonzero(has_mask):
            packet = bytes(samples[int(indices[int(point_index)])])
            if len(packet) != wave_size:
                raise PublicationError(
                    f"Eager waveform packet is {len(packet)} bytes, "
                    f"expected {wave_size}"
                )
            packets.append(packet)
        return packets

    if isinstance(state, LazyWaveformState):
        reader = state.reader
        if reader.wave_size_bytes != wave_size:
            raise PublicationError(
                "Lazy waveform reader packet size "
                f"{reader.wave_size_bytes} does not match descriptor size {wave_size}"
            )
        offsets = np.asarray(array["wavepacket_offset"], dtype=np.uint64)[has_mask]
        remainders = np.remainder(offsets, wave_size)
        if np.any(remainders != remainders[0]):
            raise PublicationError(
                "Unsupported waveform byte offset layout; laspy requires a single "
                "fixed byte-offset remainder for all waveform packets"
            )
        base_remainder = int(remainders[0])
        for point_index in np.flatnonzero(has_mask):
            offset = int(array["wavepacket_offset"][int(point_index)])
            packet_index = (offset - base_remainder) // wave_size
            reader.seek(packet_index)
            packet = bytes(reader.read_n_waveforms(1))
            if len(packet) != wave_size:
                raise PublicationError(
                    f"Short waveform read for point {int(point_index)}"
                )
            packets.append(packet)
        return packets

    raise PublicationError(
        "The appended points reference waveforms but carry no waveform data "
        "(read them with fullwave='lazy' or 'eager')"
    )


def _stream_copy(
    src: Path, dst: BinaryIO, digest: "hashlib._Hash"
) -> None:
    with src.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_DIGEST_CHUNK), b""):
            dst.write(chunk)
            digest.update(chunk)


def _build_candidate(
    store: _PublicationStore,
    source: LasData,
    points: PackedPointRecord,
    revision: int,
    append_id: str,
    parent_record: dict[str, Any],
) -> dict[str, Any]:
    """Build the full new revision in staging files and journal STAGED."""
    header = source.header.copy()
    points = _normalize_points(points, header)

    if points.point_format != header.point_format:
        raise errors.LaspyException("Incompatible point formats")
    if len(points) == 0:
        raise PublicationError("Cannot publish an empty append batch")
    if header.max_point_count() - header.point_count < len(points):
        raise PublicationError(
            "Appending would exceed the maximum point count supported "
            "by the file version"
        )

    source_array = source.points.array
    appended_array = np.array(points.array, copy=True)
    if appended_array.dtype != source_array.dtype:
        raise errors.LaspyException("Incompatible point formats")

    wave_size: Optional[int] = None
    appended_packets: list[bytes] = []
    if header.point_format.has_waveform_packet:
        from .waveform.descriptor import WaveformPacketDescriptorRegistry

        descriptor_registry = WaveformPacketDescriptorRegistry.from_vlrs(
            header.vlrs
        )
        wave_dtype = descriptor_registry.dtype()
        has_appended_waveform = bool(
            np.any(np.asarray(points.array["wavepacket_index"] != 0, dtype=bool))
        )
        if has_appended_waveform and wave_dtype is None:
            raise PublicationError(
                "Appended points reference waveforms but no waveform packet "
                "descriptor VLR is present in the source file"
            )
        if wave_dtype is not None:
            wave_size = int(wave_dtype.itemsize)
            if has_appended_waveform:
                appended_packets = _waveform_packet_bytes(points, wave_size)

    store.tmp_dir.mkdir(parents=True, exist_ok=True)
    suffix = store.las_path.suffix
    tmp_las = store.tmp_dir / f"rev-{revision:06d}-{uuid.uuid4().hex}{suffix}"
    tmp_wdp: Optional[Path] = None

    las_digest = hashlib.sha256()
    wdp_digest: Optional[str] = None

    if wave_size is not None and appended_packets:
        tmp_wdp = store.tmp_dir / f"rev-{revision:06d}-{uuid.uuid4().hex}.wdp"
        wdp_hash = hashlib.sha256()
        base_wdp_length = 0
        with tmp_wdp.open("wb") as dst:
            if store.wdp_path is not None:
                _stream_copy(store.wdp_path, dst, wdp_hash)
                base_wdp_length = store.wdp_path.stat().st_size
            packet_cursor = base_wdp_length
            packet_offsets: list[int] = []
            for packet in appended_packets:
                packet_offsets.append(packet_cursor)
                dst.write(packet)
                wdp_hash.update(packet)
                packet_cursor += len(packet)
            store._fsync_file(dst)
        wdp_digest = wdp_hash.hexdigest()

        waveform_points = np.flatnonzero(
            np.asarray(appended_array["wavepacket_index"] != 0, dtype=bool)
        )
        if len(waveform_points) != len(packet_offsets):
            raise PublicationError("Internal waveform bookkeeping mismatch")
        appended_array["wavepacket_offset"][waveform_points] = np.asarray(
            packet_offsets, dtype=appended_array["wavepacket_offset"].dtype
        )
        appended_array["wavepacket_size"][waveform_points] = wave_size
    elif store.wdp_path is not None:
        # No new packets: keep the existing waveform payload in place and
        # record its frozen digest; there is nothing staged to promote.
        wdp_digest = store.digest_file(store.wdp_path)

    merged_array = np.concatenate([source_array, appended_array])
    if header.point_format.has_waveform_packet:
        # Writing via waveforms=False bypasses LasData's storage-flag update,
        # so mirror its semantics here based on the merged point set.
        has_any_waveform = bool(
            np.any(np.asarray(merged_array["wavepacket_index"] != 0, dtype=bool))
        )
        header.global_encoding.waveform_data_packets_external = has_any_waveform
        if has_any_waveform:
            header.global_encoding.waveform_data_packets_internal = False
        if header.version.minor >= 3:
            header.start_of_waveform_data_packet_record = 0
    merged_record = PackedPointRecord(merged_array, header.point_format)
    candidate = LasData(header, merged_record)
    candidate.update_header()
    candidate.write(tmp_las, waveforms=False)

    with tmp_las.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_DIGEST_CHUNK), b""):
            las_digest.update(chunk)
    las_digest_hex = las_digest.hexdigest()

    staged = {
        "tmp_las": str(tmp_las),
        "tmp_wdp": str(tmp_wdp) if tmp_wdp is not None else None,
        "las_digest": las_digest_hex,
        "wdp_digest": wdp_digest,
        "point_count": int(header.point_count),
        "appended_count": int(len(points)),
        "appended_points_digest": _hash_bytes(appended_array.tobytes()),
        "parent_revision": int(parent_record["revision"]),
        "parent_las_digest": parent_record["las_digest"],
        "parent_wdp_digest": parent_record["wdp_digest"],
        "append_id": append_id,
    }
    store.append_journal(revision, "STAGED", staged)
    return staged


def _promote(store: _PublicationStore, staged: dict[str, Any], revision: int) -> None:
    """Atomically move staged files into place (wdp first, then the LAS/LAZ).

    Files that no longer exist in staging are assumed to have already been
    promoted in a previous attempt whose final journal event was lost; the
    digest check performed by the caller guarantees the in-place file is the
    frozen revision.
    """
    tmp_wdp = staged["tmp_wdp"]
    if tmp_wdp is not None:
        tmp_wdp_path = Path(tmp_wdp)
        wdp_final = store.las_path.with_suffix(".wdp")
        if tmp_wdp_path.is_file():
            # Extending the wdp first keeps the previous LAS valid: its
            # references only ever address the prefix shared with the new
            # waveform payload.
            os.replace(tmp_wdp_path, wdp_final)
            store._fsync_dir(store.las_path.parent)
        elif not wdp_final.is_file():
            raise PublicationVerificationError(
                f"neither staged nor published .wdp exists for revision {revision}"
            )

    tmp_las_path = Path(staged["tmp_las"])
    if tmp_las_path.is_file():
        os.replace(tmp_las_path, store.las_path)
        store._fsync_dir(store.las_path.parent)
    elif not store.las_path.is_file():
        raise PublicationVerificationError(
            f"neither staged nor published LAS exists for revision {revision}"
        )
    store.append_journal(revision, "PROMOTED", {})


def _commit_manifest(
    store: _PublicationStore,
    manifest: dict[str, Any],
    staged: dict[str, Any],
    revision: int,
) -> None:
    manifest["revisions"].append(
        {
            "revision": revision,
            "append_id": staged["append_id"],
            "parent_revision": staged["parent_revision"],
            "point_count": staged["point_count"],
            "appended_count": staged["appended_count"],
            "las_digest": staged["las_digest"],
            "wdp_digest": staged["wdp_digest"],
            "appended_points_digest": staged["appended_points_digest"],
            "parent_las_digest": staged["parent_las_digest"],
            "parent_wdp_digest": staged["parent_wdp_digest"],
            "created_at": store._utc_now(),
            "status": "committed",
        }
    )
    manifest["current_revision"] = revision
    store.write_manifest(manifest)


# ---------------------------------------------------------------- public API
def publish_append(
    las_path: str | Path,
    points: PackedPointRecord,
    append_id: Optional[str] = None,
    *,
    on_staged: Optional[Any] = None,
) -> PublicationResult:
    """Publish a new revision that appends ``points`` to the cloud.

    The revision is built out of place and only promoted once the point data,
    the waveform packets and the header statistics have all been written and
    digest-checked together. A crash leaves the previous revision readable;
    :func:`recover_publishing` finishes or discards the in-flight revision.

    Parameters
    ----------
    las_path:
        Path to the published LAS/LAZ file (its ``.wdp`` companion, when
        present, is handled automatically).
    points:
        The points to append. Points carrying waveforms must be read with
        ``fullwave='lazy'`` or ``'eager'`` so the packets are available.
    append_id:
        Caller chosen identity of the append. Retrying with the same id after a
        successful (but possibly unacknowledged) publication returns the
        existing revision instead of appending twice. When ``None`` an id is
        generated and de-duplication is not possible.
    on_staged:
        Optional zero-argument callable invoked after the durable STAGED event
        and before promotion. It is intended for crash injection in tests.
    """
    if append_id is not None and not isinstance(append_id, str):
        raise TypeError("append_id must be a string or None")
    own_append_id = append_id or uuid.uuid4().hex

    store = _PublicationStore(las_path)
    with store.locked():
        manifest = store.ensure_initialized()

        existing = _find_revision_by_append_id(manifest, own_append_id)
        if existing is not None:
            return _result_from_record(store, existing, deduped=True)

        unresolved = _unresolved_journals(store, manifest)
        if unresolved:
            revision, begin_append_id = unresolved[0]
            if begin_append_id == own_append_id:
                raise AmbiguousAppendError(
                    f"append_id {own_append_id!r} has an unresolved journal "
                    f"(revision {revision}). Run recover_publishing() before retrying; "
                    "re-appending blindly could count the same points twice."
                )
            raise PublicationError(
                f"Another publication (revision {revision}, append_id "
                f"{begin_append_id!r}) is unresolved. Run recover_publishing() first; "
                "publishing concurrently would corrupt the revision chain."
            )

        revision = _next_revision(store, manifest)
        parent_record = manifest["revisions"][-1]

        store.append_journal(
            revision,
            "BEGIN",
            {
                "append_id": own_append_id,
                "parent_revision": int(parent_record["revision"]),
                "requested_count": int(len(points)),
            },
        )

        from . import read as read_las

        source = read_las(store.las_path)
        try:
            staged = _build_candidate(
                store, source, points, revision, own_append_id, parent_record
            )
        except BaseException:
            _cleanup_staging(store, revision)
            store.append_journal(revision, "ABORTED", {"reason": "build_failed"})
            raise
        staged["revision"] = revision

        if on_staged is not None:
            # A failure here is deliberately treated like a crash: the durable
            # STAGED event remains and recover_publishing() rolls the revision
            # forward or discards it after digest verification.
            on_staged()

        _promote(store, staged, revision)
        _commit_manifest(store, manifest, staged, revision)
        store.append_journal(revision, "DONE", {})
        _cleanup_staging(store, revision)

        final_record = store.read_manifest()["revisions"][-1]
        return _result_from_record(store, final_record, deduped=False)


def recover_publishing(las_path: str | Path) -> list[PublicationResult]:
    """Recover all interrupted publications for ``las_path``.

    Every journal lacking a DONE event is inspected. A fully staged revision
    whose digests still match is promoted and recorded in the manifest; a
    partially written revision is discarded. Already committed revisions are
    left untouched. The recovered bytes are digest verified, so the resulting
    files are identical to the frozen revision that was originally staged.
    """
    store = _PublicationStore(las_path)
    store.root.mkdir(parents=True, exist_ok=True)
    store.journal_dir.mkdir(exist_ok=True)
    store.tmp_dir.mkdir(exist_ok=True)

    # Take over only a lock whose owner process is gone.
    store.reap_stale_lock()

    with store.locked():
        manifest = store.ensure_initialized()
        known_revisions = {
            int(record["revision"]) for record in manifest["revisions"]
        }
        recovered: list[PublicationResult] = []

        journal_files = sorted(store.journal_dir.glob("*.log"))
        for journal_file in journal_files:
            revision = int(journal_file.stem)
            events = store.read_journal(revision)
            if not events or any(event["event"] == "DONE" for event in events):
                continue
            if any(event["event"] == "ABORTED" for event in events):
                _cleanup_staging(store, revision)
                continue

            staged_event = next(
                (event for event in events if event["event"] == "STAGED"), None
            )
            if staged_event is None:
                # Crashed before any durable staging output existed.
                _cleanup_staging(store, revision)
                store.append_journal(revision, "ABORTED", {"reason": "no_stage"})
                continue

            staged = dict(staged_event)
            staged["revision"] = revision
            try:
                _verify_staged(store, staged)
            except PublicationVerificationError:
                # The frozen staged bytes cannot be trusted. A matching
                # promotion cannot have happened, so the published file is
                # still the previous valid revision: discard and move on.
                _cleanup_staging(store, revision)
                store.append_journal(
                    revision, "ABORTED", {"reason": "corrupt_stage"}
                )
                continue

            if revision in known_revisions:
                # Promotion and the manifest update already happened, only the
                # terminal journal event was lost. It is already part of the
                # chain, so the stale-parent check below must not apply.
                store.append_journal(revision, "DONE", {"recovered": True})
                record = _find_revision(store.read_manifest(), revision)
                recovered.append(_result_from_record(store, record, deduped=False))
                continue

            head = manifest["revisions"][-1]
            if not _staged_descends_from_head(staged, head):
                # The staged revision was built on a parent that is no longer
                # the chain head (a concurrent publication committed first).
                # Promoting it would overwrite the current revision, so it is
                # discarded as a stale branch.
                _cleanup_staging(store, revision)
                store.append_journal(
                    revision, "ABORTED", {"reason": "stale_parent"}
                )
                continue

            _promote(store, staged, revision)
            _commit_manifest(store, manifest, staged, revision)
            store.append_journal(revision, "DONE", {"recovered": True})
            _cleanup_staging(store, revision)
            manifest = store.read_manifest()
            known_revisions = {
                int(record["revision"]) for record in manifest["revisions"]
            }
            record = _find_revision(manifest, revision)
            recovered.append(_result_from_record(store, record, deduped=False))

        # Garbage collect anything left in the staging directory without a
        # matching live journal (e.g. process killed mid temp-file creation).
        for leftover in store.tmp_dir.glob("*"):
            with contextlib.suppress(FileNotFoundError):
                leftover.unlink()

        return recovered


def verify_publication(las_path: str | Path) -> VerificationReport:
    """Verify the current revision against its frozen manifest record.

    Checks file digests, header statistics against a point-by-point
    recomputation, the revision chain and, for waveform clouds, that every
    point's waveform reference resolves inside the frozen ``.wdp`` payload.
    """
    store = _PublicationStore(las_path)
    if not store.manifest_path.is_file():
        raise PublicationError(
            f"No publication manifest found next to {store.las_path}"
        )

    manifest = store.read_manifest()
    revision = int(manifest["current_revision"])
    record = _find_revision(manifest, revision)

    report = VerificationReport(
        revision=revision,
        las_digest=record["las_digest"],
        wdp_digest=record["wdp_digest"],
        point_count=int(record["point_count"]),
    )

    actual_las_digest = store.digest_file(store.las_path)
    report.checks["las_digest"] = actual_las_digest == record["las_digest"]
    if not report.checks["las_digest"]:
        report.issues.append(
            "LAS/LAZ digest differs from the frozen manifest record"
        )

    if record["wdp_digest"] is not None:
        wdp_path = store.las_path.with_suffix(".wdp")
        if not wdp_path.is_file():
            report.checks["wdp_digest"] = False
            report.issues.append("waveform .wdp companion is missing")
        else:
            actual_wdp_digest = store.digest_file(wdp_path)
            report.checks["wdp_digest"] = actual_wdp_digest == record["wdp_digest"]
            if not report.checks["wdp_digest"]:
                report.issues.append(
                    ".wdp digest differs from the frozen manifest record"
                )

    _verify_chain(manifest, report)

    from . import read as read_las

    las = read_las(store.las_path)
    _verify_header_stats(las, report)
    if record["wdp_digest"] is not None:
        _verify_waveform_references(store, las, report)

    if not report.ok:
        raise PublicationVerificationError(
            "Publication verification failed: " + "; ".join(report.issues)
        )
    return report


# ------------------------------------------------------------- verification
def _verify_chain(manifest: dict[str, Any], report: VerificationReport) -> None:
    revisions = manifest["revisions"]
    by_revision = {int(record["revision"]): record for record in revisions}
    chain_ok = True
    for record in revisions[1:]:
        parent = by_revision.get(int(record["parent_revision"]))
        if parent is None:
            chain_ok = False
            report.issues.append(
                f"revision {record['revision']} references missing parent "
                f"revision {record['parent_revision']}"
            )
            continue
        if record["parent_las_digest"] != parent["las_digest"]:
            chain_ok = False
            report.issues.append(
                f"revision {record['revision']} parent LAS digest breaks the chain"
            )
        if record["parent_wdp_digest"] != parent["wdp_digest"]:
            chain_ok = False
            report.issues.append(
                f"revision {record['revision']} parent .wdp digest breaks the chain"
            )
    report.checks["revision_chain"] = chain_ok


def _verify_header_stats(las: LasData, report: VerificationReport) -> None:
    header = las.header
    array = las.points.array
    report.checks["point_count"] = int(header.point_count) == len(array)
    if header.point_count != len(array):
        report.issues.append(
            f"header point_count {header.point_count} != {len(array)} points on disk"
        )
        return

    scaled = np.stack(
        [
            array["X"] * header.scales[0] + header.offsets[0],
            array["Y"] * header.scales[1] + header.offsets[1],
            array["Z"] * header.scales[2] + header.offsets[2],
        ],
        axis=1,
    )
    mins_ok = np.allclose(scaled.min(axis=0), header.mins, atol=0.0, rtol=0.0)
    maxs_ok = np.allclose(scaled.max(axis=0), header.maxs, atol=0.0, rtol=0.0)
    report.checks["header_bounds"] = bool(mins_ok and maxs_ok)
    if not (mins_ok and maxs_ok):
        report.issues.append("header mins/maxs do not match a pointwise recompute")

    expected_returns = np.zeros(15, dtype=np.uint64)
    values, counts = np.unique(las.points.return_number, return_counts=True)
    for value, count in zip(values, counts):
        if value == 0 or value > 15:
            continue
        expected_returns[int(value) - 1] = count
    returns_ok = np.array_equal(
        expected_returns, np.asarray(header.number_of_points_by_return, dtype=np.uint64)
    )
    report.checks["return_counts"] = bool(returns_ok)
    if not returns_ok:
        report.issues.append(
            "header number_of_points_by_return does not match the point records"
        )

    if "gps_time" in array.dtype.names and header.version.minor >= 5:
        gps_ok = header.min_gps_time == float(array["gps_time"].min()) and (
            header.max_gps_time == float(array["gps_time"].max())
        )
        report.checks["gps_bounds"] = bool(gps_ok)
        if not gps_ok:
            report.issues.append("header gps time bounds do not match the points")


def _verify_waveform_references(
    store: _PublicationStore, las: LasData, report: VerificationReport
) -> None:
    wdp_path = store.las_path.with_suffix(".wdp")
    wdp_size = wdp_path.stat().st_size
    array = las.points.array
    has_mask = np.asarray(array["wavepacket_index"] != 0, dtype=bool)
    if not has_mask.any():
        report.checks["waveform_references"] = True
        return

    offsets = np.asarray(array["wavepacket_offset"], dtype=np.uint64)[has_mask]
    sizes = np.asarray(array["wavepacket_size"], dtype=np.uint64)[has_mask]
    ends = offsets + sizes
    valid = bool(np.all(sizes > 0) and np.all(ends <= wdp_size))
    report.checks["waveform_references"] = valid
    if not valid:
        report.issues.append(
            "at least one point references waveform bytes outside the frozen .wdp"
        )


def _staged_descends_from_head(
    staged: dict[str, Any], head_record: dict[str, Any]
) -> bool:
    """True if the staged revision was built on the current chain head."""
    return (
        int(staged["parent_revision"]) == int(head_record["revision"])
        and staged["parent_las_digest"] == head_record["las_digest"]
        and staged["parent_wdp_digest"] == head_record["wdp_digest"]
    )


def _verify_staged(
    store: _PublicationStore, staged: dict[str, Any]
) -> None:
    tmp_las = Path(staged["tmp_las"])
    final_las = store.las_path
    las_ok = False
    if tmp_las.is_file() and store.digest_file(tmp_las) == staged["las_digest"]:
        las_ok = True
    elif final_las.is_file() and store.digest_file(final_las) == staged["las_digest"]:
        las_ok = True
    if not las_ok:
        raise PublicationVerificationError(
            f"staged LAS for revision {staged['revision']} is missing or corrupt"
        )

    if staged["tmp_wdp"] is not None:
        wdp_final = store.las_path.with_suffix(".wdp")
        tmp_wdp = Path(staged["tmp_wdp"])
        wdp_ok = False
        if tmp_wdp.is_file() and store.digest_file(tmp_wdp) == staged["wdp_digest"]:
            wdp_ok = True
        elif wdp_final.is_file() and (
            store.digest_file(wdp_final) == staged["wdp_digest"]
        ):
            wdp_ok = True
        if not wdp_ok:
            raise PublicationVerificationError(
                f"staged .wdp for revision {staged['revision']} is missing or corrupt"
            )


# --------------------------------------------------------------------- lookup
def _find_revision(manifest: dict[str, Any], revision: int) -> dict[str, Any]:
    for record in manifest["revisions"]:
        if int(record["revision"]) == revision:
            return record
    raise PublicationError(f"revision {revision} is not present in the manifest")


def _find_revision_by_append_id(
    manifest: dict[str, Any], append_id: str
) -> Optional[dict[str, Any]]:
    for record in manifest["revisions"]:
        if record["append_id"] == append_id:
            return record
    return None


def _next_revision(
    store: _PublicationStore, manifest: dict[str, Any]
) -> int:
    """Monotonic revision number across committed revisions and all attempts.

    Aborted or crashed attempts keep their journal forever, so their revision
    number must never be reused: a journal file documents exactly one attempt.
    """
    highest = int(manifest["current_revision"])
    for journal_file in store.journal_dir.glob("*.log"):
        highest = max(highest, int(journal_file.stem))
    return highest + 1


def _unresolved_journals(
    store: _PublicationStore, manifest: dict[str, Any]
) -> list[tuple[int, str]]:
    """Return ``(revision, append_id)`` for journals with no terminal event."""
    committed = {
        int(record["revision"]) for record in manifest["revisions"]
    }
    unresolved: list[tuple[int, str]] = []
    for journal_file in sorted(store.journal_dir.glob("*.log")):
        revision = int(journal_file.stem)
        events = store.read_journal(revision)
        if not events:
            continue
        if any(event["event"] in ("DONE", "ABORTED") for event in events):
            continue
        if revision in committed:
            continue
        begin = next(
            (event for event in events if event["event"] == "BEGIN"), None
        )
        append_id = begin.get("append_id") if begin is not None else None
        unresolved.append((revision, append_id))
    return unresolved


def _result_from_record(
    store: _PublicationStore, record: dict[str, Any], *, deduped: bool
) -> PublicationResult:
    wdp_digest = record["wdp_digest"]
    wdp_path = store.las_path.with_suffix(".wdp")
    return PublicationResult(
        revision=int(record["revision"]),
        append_id=record["append_id"],
        point_count=int(record["point_count"]),
        appended_count=int(record["appended_count"]),
        las_path=store.las_path,
        wdp_path=wdp_path if wdp_digest is not None else None,
        las_digest=record["las_digest"],
        wdp_digest=wdp_digest,
        deduped=deduped,
    )


def _cleanup_staging(store: _PublicationStore, revision: int) -> None:
    for leftover in store.tmp_dir.glob(f"rev-{revision:06d}-*"):
        with contextlib.suppress(FileNotFoundError):
            if leftover.is_file():
                leftover.unlink()
