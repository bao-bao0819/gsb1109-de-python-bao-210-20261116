import hashlib
import importlib.util
import io
import json
import os
import struct

import pytest

from Registry import Registry, RegistryLog, RegistryParse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(REPO_ROOT, "testing", "reg_samples")

spec = importlib.util.spec_from_file_location(
    "recover_hive", os.path.join(REPO_ROOT, "samples", "recover_hive.py"))
recover_hive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recover_hive)


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def case_paths(case):
    directory = os.path.join(SAMPLES, case)
    return (os.path.join(directory, "SYSTEM"),
            [os.path.join(directory, "SYSTEM.LOG1"),
             os.path.join(directory, "SYSTEM.LOG2")])


def reference_recover_bytes(primary_path, log1_path, log2_path):
    """Recover with the RegistryLog reference implementation."""
    primary = io.BytesIO(open(primary_path, "rb").read())
    primary.seek(0)
    log1 = RegistryLog.RegistryLog(primary, log1_path)
    primary.seek(0)
    log2 = RegistryLog.RegistryLog(primary, log2_path)
    primary.seek(0)
    regf = RegistryParse.REGFBlock(primary.read(512), 0, False)
    status = regf.recovery_required()
    eligible = [log for log in (log1, log2) if log.is_eligible_log()]
    if len(eligible) == 1:
        eligible[0].recover_hive()
    elif len(eligible) == 2:
        if log1.is_starting_log(log2):
            first, second = log1, log2
        else:
            first, second = log2, log1
        if not status.recover_header:
            seqnum = first.recover_hive()
            second.recover_hive_continue(seqnum + 1)
        else:
            second.recover_hive()
    primary.seek(0)
    return primary.read()


def test_full_chain_across_two_logs(tmp_path):
    primary, logs = case_paths("new_log_1")
    report = recover_hive.recover_hive(primary, logs, out_dir=str(tmp_path))

    recovery = report["recovery"]
    assert recovery["result"] == "recovered"
    assert recovery["base_generation"] == 178
    assert recovery["recovered_range"] == [178, 216]
    assert recovery["gap"] is None

    applied = recovery["applied_entries"]
    assert [entry["generation"] for entry in applied] == list(range(178, 217))
    # generations 178..185 come from LOG1, 186..216 from LOG2
    for entry in applied:
        expected = "SYSTEM.LOG1" if entry["generation"] <= 185 else "SYSTEM.LOG2"
        assert os.path.basename(entry["source_log"]) == expected

    verification = report["output"]["verification"]
    assert verification["regf_checksum_valid"]
    assert verification["sequence_numbers_equal"]
    assert not verification["recovery_required"]
    assert verification["final_generation"] == 216
    assert verification["hive_parses"]

    # every dirty page of every applied entry has a provenance record
    total_pages = sum(entry["dirty_pages"] for entry in applied)
    assert len(report["page_writes"]) == total_pages
    for write in report["page_writes"]:
        assert write["source_log"]
        assert len(write["sha256"]) == 64
        assert write["target_offset"] >= 0x1000

    assert report["inputs_unmodified"]


def test_reversed_logs_recover_identically(tmp_path):
    primary, logs = case_paths("new_log_1_reversed")
    report = recover_hive.recover_hive(primary, logs, out_dir=str(tmp_path))

    recovery = report["recovery"]
    assert recovery["recovered_range"] == [178, 216]
    # the order is determined by the generations, not by the file names:
    # here LOG2 holds the earlier generations
    for entry in recovery["applied_entries"]:
        expected = "SYSTEM.LOG2" if entry["generation"] <= 185 else "SYSTEM.LOG1"
        assert os.path.basename(entry["source_log"]) == expected

    clean_primary, clean_logs = case_paths("new_log_1")
    clean_report = recover_hive.recover_hive(
        clean_primary, clean_logs, out_dir=str(tmp_path / "clean"))
    assert report["output"]["sha256"] == clean_report["output"]["sha256"]


def test_corrupt_primary_header_restored_from_earliest_log(tmp_path):
    primary, logs = case_paths("new_log_1_reversed_and_invalid_checksum")
    before = sha256_file(primary)
    report = recover_hive.recover_hive(primary, logs, out_dir=str(tmp_path))

    assert report["primary_assessment"]["header_checksum_valid"] is False
    assert report["identity_reference"] == "transaction_logs"
    recovery = report["recovery"]
    # the header is restored from the log with the oldest first sequence,
    # so the full chain 178..216 stays provable
    assert os.path.basename(recovery["header_restored_from"]) == "SYSTEM.LOG2"
    assert recovery["recovered_range"] == [178, 216]
    assert recovery["result"] == "recovered"
    assert report["output"]["verification"]["final_generation"] == 216
    # the corrupt input was only read, never written
    assert sha256_file(primary) == before
    assert report["inputs_unmodified"]

    clean_primary, clean_logs = case_paths("new_log_1")
    clean_report = recover_hive.recover_hive(
        clean_primary, clean_logs, out_dir=str(tmp_path / "clean"))
    assert report["output"]["sha256"] == clean_report["output"]["sha256"]


def test_stale_log_entries_are_not_applied(tmp_path):
    primary, logs = case_paths("new_log_2")
    report = recover_hive.recover_hive(primary, logs, out_dir=str(tmp_path))

    recovery = report["recovery"]
    assert recovery["recovered_range"] == [206, 237]
    assert recovery["gap"] is None
    # LOG2 holds a single, already flushed generation
    assert [entry["generation"] for entry in recovery["stale_entries"]] == [196]
    assert all(os.path.basename(entry["source_log"]) == "SYSTEM.LOG1"
               for entry in recovery["applied_entries"])
    assert report["output"]["verification"]["final_generation"] == 237


@pytest.mark.parametrize("case", ["new_log_1", "new_log_1_reversed",
                                  "new_log_2", "new_log_2_reversed"])
def test_matches_reference_implementation(tmp_path, case):
    primary, logs = case_paths(case)
    report = recover_hive.recover_hive(primary, logs, out_dir=str(tmp_path))
    reference = reference_recover_bytes(primary, logs[0], logs[1])
    with open(report["output"]["path"], "rb") as f:
        assert f.read() == reference


def make_gapped_log(source_path, keep_ranges, destination_path):
    """Build a log file keeping only the generations in keep_ranges."""
    with open(source_path, "rb") as f:
        buf = f.read()
    entries, _ = recover_hive.walk_log_entries(buf)
    out = bytearray(buf[:0x200])
    for entry in entries:
        if any(low <= entry.sequence <= high for low, high in keep_ranges):
            out += entry.raw
    with open(destination_path, "wb") as f:
        f.write(bytes(out))
    return destination_path


def test_missing_generations_stop_at_provable_range(tmp_path):
    primary, logs = case_paths("new_log_1")
    # generations 186..189 and 195..216 are present, 190..194 are missing
    gapped = make_gapped_log(logs[1], [(186, 189), (195, 216)],
                             str(tmp_path / "SYSTEM.LOG2"))
    report = recover_hive.recover_hive(primary, [logs[0], gapped],
                                       out_dir=str(tmp_path / "out"))

    recovery = report["recovery"]
    assert recovery["recovered_range"] == [178, 189]
    assert recovery["gap"] is not None
    assert recovery["gap"]["first_unprovable_generation"] == 190
    assert recovery["gap"]["unapplied_generations"] == list(range(195, 217))
    # the recovery stops at the gap instead of jumping to the highest
    # sequence number available (216)
    assert report["output"]["verification"]["final_generation"] == 189
    assert report["output"]["verification"]["hive_parses"]
    assert report["inputs_unmodified"]


def test_foreign_log_generations_are_not_mixed_in(tmp_path):
    primary, logs = case_paths("new_log_1")
    foreign_log = os.path.join(SAMPLES, "new_log_2", "SYSTEM.LOG1")
    report = recover_hive.recover_hive(primary, [logs[0], foreign_log],
                                       out_dir=str(tmp_path))

    recovery = report["recovery"]
    # the foreign log carries generations 206..237; the chain must stop at
    # the last provable generation instead of taking the maximum (237)
    assert recovery["recovered_range"] == [178, 185]
    assert recovery["gap"]["first_unprovable_generation"] == 186
    assert recovery["gap"]["unapplied_generations"] == list(range(206, 238))
    assert report["output"]["verification"]["final_generation"] == 185


def test_log_with_mismatched_identity_is_excluded(tmp_path):
    primary, logs = case_paths("new_log_1")
    # forge a log whose hive name does not match the primary
    with open(logs[0], "rb") as f:
        buf = bytearray(f.read())
    buf[0x30:0x30 + 12] = "SYSTEN".encode("utf-16le")
    regf = RegistryParse.REGFBlock(bytes(buf[:0x200]), 0, False)
    struct.pack_into("<I", buf, 0x1FC, regf.calculate_checksum())
    forged = str(tmp_path / "SYSTEM.LOG1")
    with open(forged, "wb") as f:
        f.write(bytes(buf))

    report = recover_hive.recover_hive(primary, [forged, logs[1]],
                                       out_dir=str(tmp_path / "out"))

    forged_report, good_report = report["logs"]
    assert forged_report["eligible"] is False
    assert forged_report["identity_checks"]["hive_name_matches_hive"] is False
    assert any("hive_name" in reason
               for reason in forged_report["exclusion_reasons"])
    assert good_report["eligible"] is True

    # generations 186..216 are available but 179..185 are missing, so no
    # consistent hive can be produced and none is written
    recovery = report["recovery"]
    assert recovery["result"] == "no_provable_recovery"
    assert recovery["recovered_range"] == [178, 178]
    assert recovery["gap"]["first_unprovable_generation"] == 179
    assert report["output"] is None
    assert not os.path.exists(str(tmp_path / "out" / "SYSTEM.recovered"))
    assert report["inputs_unmodified"]


def test_report_file_is_written_and_loadable(tmp_path):
    primary, logs = case_paths("new_log_1")
    report = recover_hive.recover_hive(primary, logs, out_dir=str(tmp_path))

    report_path = tmp_path / "SYSTEM.recovery.json"
    assert report["report_path"] == str(report_path)
    with open(str(report_path)) as f:
        loaded = json.load(f)
    assert loaded["recovery"]["recovered_range"] == [178, 216]
    assert loaded["inputs"]["primary"]["sha256"] == sha256_file(primary)
    assert len(loaded["page_writes"]) == len(report["page_writes"])

    # the recovered hive is usable
    registry = Registry.Registry(str(tmp_path / "SYSTEM.recovered"))
    assert registry.root().name() == loaded["output"]["verification"]["root_key_name"]
