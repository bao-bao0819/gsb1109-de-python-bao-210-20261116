#!/usr/bin/env python
"""
Auditable recovery of a Windows registry hive from its transaction logs.

Given a primary hive file whose sequence numbers disagree (a mid-update
state) and its transaction logs, this tool produces a recovered hive plus
a JSON report that makes the recovery independently verifiable:

  * the original files are never modified (they are hashed before and
    after the run and the hashes are recorded in the report);
  * a log is only used after its identity has been verified against the
    hive (file type, format version, clustering factor, root cell offset
    and hive name must match), so pages from a foreign hive's log can
    never be mixed in;
  * log entries are applied as a consecutive chain of generations
    starting at the hive's flushed generation -- the final generation is
    wherever the chain ends, never the highest sequence number seen;
  * when an intermediate generation is missing, recovery stops at the
    last provable generation and the gap is reported;
  * every dirty page written to the output is recorded with its source
    (log file, log entry, generation) and a SHA-256 hash.
"""
from __future__ import absolute_import, print_function

import argparse
import hashlib
import io
import json
import os
import struct
import sys
import time
from collections import OrderedDict
from ctypes import c_uint32

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Registry import Registry, RegistryParse

TOOL_NAME = "recover_hive"
TOOL_VERSION = "1.0"

REGF_HEADER_SIZE = 0x200
FIRST_HBIN_OFFSET = 0x1000
UINT32_HALF = 0x80000000

LOG_ENTRY_ALIGNMENT = RegistryParse.LOG_ENTRY_SIZE_ALIGNMENT

# REGF header fields that identify which hive a transaction log belongs to.
IDENTITY_FIELDS = ("major_version", "minor_version", "clustering_factor",
                   "file_format", "root_cell_offset", "hive_name")


def sha256_bytes(buf):
    return hashlib.sha256(buf).hexdigest()


def read_file(path):
    with open(path, "rb") as f:
        return f.read()


def seq_next(seq):
    """The generation following seq, in circular uint32 arithmetic."""
    return c_uint32(seq + 1).value


def seq_is_older(seq, base):
    """Check if seq is strictly older than base, in circular uint32 arithmetic."""
    return seq != base and c_uint32(base - seq).value < UINT32_HALF


class LogEntry(object):
    """A single HvLE log entry walked from a transaction log file."""

    def __init__(self, hvle, offset, raw):
        self.hvle = hvle
        self.offset = offset
        self.raw = raw
        self.valid = hvle.validate_log_entry()
        self.sequence = hvle.sequence()
        self.size = hvle.size()
        self.dirty_pages_count = hvle.dirty_pages_count()
        self.hbins_size = hvle.hbins_size()
        self.hive_flags = hvle.hive_flags()


def walk_log_entries(buf):
    """
    Walk all HvLE log entries of a transaction log file, valid or not.
    Returns (entries, stop_reason). Walking stops at the first entry whose
    size is not sane, because the offset of the following entry cannot be
    determined anymore.
    """
    entries = []
    stop_reason = "end of file"
    offset = REGF_HEADER_SIZE
    while offset + LOG_ENTRY_ALIGNMENT <= len(buf):
        try:
            hvle = RegistryParse.HvLEBlock(buf, offset, None)
        except RegistryParse.ParseException:
            stop_reason = "invalid HvLE signature at 0x%x" % offset
            break
        size = hvle.size()
        if size < LOG_ENTRY_ALIGNMENT or size % LOG_ENTRY_ALIGNMENT != 0:
            stop_reason = "invalid log entry size at 0x%x" % offset
            break
        if offset + size > len(buf):
            stop_reason = "log entry extends past end of file at 0x%x" % offset
            break
        entries.append(LogEntry(hvle, offset, buf[offset:offset + size]))
        offset += size
    return entries, stop_reason


class LogFile(object):
    """A parsed transaction log file."""

    def __init__(self, path):
        self.path = path
        self.buf = read_file(path)
        self.parse_error = None
        self.regf = None
        try:
            self.regf = RegistryParse.REGFBlock(self.buf, 0, False)
        except Exception as e:
            self.parse_error = str(e)
        self.entries = []
        self.walk_stop_reason = None
        if self.regf is not None:
            self.entries, self.walk_stop_reason = walk_log_entries(self.buf)


def identity_fields(regf):
    """The REGF header fields that identify which hive a file belongs to."""
    fields = OrderedDict()
    fields["major_version"] = regf.major_version()
    fields["minor_version"] = regf.minor_version()
    fields["clustering_factor"] = regf.clustering_factor()
    fields["file_format"] = regf.file_format()
    fields["root_cell_offset"] = regf.unpack_dword(0x24)
    try:
        fields["hive_name"] = regf.hive_name()
    except Exception:
        fields["hive_name"] = None
    return fields


def evaluate_log(log, reference_fields):
    """
    Verify a transaction log's identity against the hive.
    Returns (checks, reasons): every check must pass for the log to be eligible.
    """
    checks = OrderedDict()
    reasons = []

    if log.parse_error is not None:
        checks["regf_header_parses"] = False
        reasons.append("unparseable REGF header: %s" % log.parse_error)
        return checks, reasons
    checks["regf_header_parses"] = True

    regf = log.regf
    try:
        file_type = regf.file_type()
    except ValueError:
        file_type = None
    is_new_log = file_type == RegistryParse.FileType.FILE_TYPE_LOG_NEW
    checks["file_type_is_new_transaction_log"] = is_new_log
    if not is_new_log:
        if file_type == RegistryParse.FileType.FILE_TYPE_PRIMARY:
            reasons.append("file is a primary hive, not a transaction log")
        elif file_type in (RegistryParse.FileType.FILE_TYPE_LOG_OLD_1,
                           RegistryParse.FileType.FILE_TYPE_LOG_OLD_2):
            reasons.append("old transaction log format is not supported")
        else:
            reasons.append("unknown REGF file type")

    ok = regf.validate_checksum()
    checks["log_header_checksum_valid"] = ok
    if not ok:
        reasons.append("log REGF header checksum is invalid")

    ok = regf.validate_sequence_numbers()
    checks["log_sequence_numbers_consistent"] = ok
    if not ok:
        reasons.append("log file is itself in a mid-update state")

    ok = regf.clustering_factor() == 1
    checks["clustering_factor_supported"] = ok
    if not ok:
        reasons.append("clustering factor not equal to 1 is not supported")

    fields = identity_fields(regf)
    for name in IDENTITY_FIELDS:
        ok = fields[name] == reference_fields[name]
        checks[name + "_matches_hive"] = ok
        if not ok:
            reasons.append("%s differs from the hive (%r != %r)"
                           % (name, fields[name], reference_fields[name]))

    return checks, reasons


def find_header_source(eligible_logs):
    """
    Pick the log to restore a corrupt primary REGF header from: the eligible
    log with the oldest first sequence number, which yields the longest
    provable chain of generations.
    """
    best = None
    for log in eligible_logs:
        if best is None or seq_is_older(log.regf.hive_sequence2(),
                                        best.regf.hive_sequence2()):
            best = log
    return best


def recover_hive(primary_path, log_paths, out_dir=None):
    """
    Recover a hive from its transaction logs and build the audit report.
    Never modifies the input files. Writes the recovered hive and the JSON
    report into out_dir when given. Returns the report as an OrderedDict.
    """
    report = OrderedDict()
    report["tool"] = "%s %s" % (TOOL_NAME, TOOL_VERSION)
    report["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    primary_bytes = read_file(primary_path)
    report["inputs"] = OrderedDict([
        ("primary", OrderedDict([("path", primary_path),
                                 ("size", len(primary_bytes)),
                                 ("sha256", sha256_bytes(primary_bytes))])),
        ("logs", [OrderedDict([("path", p),
                               ("size", os.path.getsize(p)),
                               ("sha256", sha256_bytes(read_file(p)))])
                  for p in log_paths]),
    ])

    primary_regf = RegistryParse.REGFBlock(primary_bytes, 0, False)
    status = primary_regf.recovery_required()
    checksum_valid = primary_regf.validate_checksum()
    report["primary_assessment"] = OrderedDict([
        ("sequence1", primary_regf.hive_sequence1()),
        ("sequence2", primary_regf.hive_sequence2()),
        ("sequence_numbers_equal", primary_regf.validate_sequence_numbers()),
        ("header_checksum_valid", checksum_valid),
        ("recover_header", status.recover_header),
        ("recover_data", status.recover_data),
        ("hbins_size", primary_regf.hbins_size()),
        ("hive_name", identity_fields(primary_regf)["hive_name"]),
    ])

    logs = [LogFile(p) for p in log_paths]

    # Establish the hive identity reference: the primary header when it is
    # intact, otherwise the consensus of the trustworthy log headers.
    reference_fields = None
    reference_source = None
    if checksum_valid:
        reference_fields = identity_fields(primary_regf)
        reference_source = "primary"
    else:
        trustworthy = []
        for log in logs:
            if log.regf is None:
                continue
            try:
                file_type = log.regf.file_type()
            except ValueError:
                continue
            if file_type != RegistryParse.FileType.FILE_TYPE_LOG_NEW:
                continue
            if not log.regf.validate_checksum():
                continue
            if not log.regf.validate_sequence_numbers():
                continue
            if log.regf.clustering_factor() != 1:
                continue
            trustworthy.append(log)
        identities = set(tuple(identity_fields(log.regf)[name]
                               for name in IDENTITY_FIELDS)
                         for log in trustworthy)
        if len(identities) == 1:
            reference_fields = identity_fields(trustworthy[0].regf)
            reference_source = "transaction_logs"

    report["identity_reference"] = reference_source
    report["logs"] = []
    report["recovery"] = None
    report["page_writes"] = []
    report["output"] = None

    if reference_fields is None:
        if not checksum_valid and not logs:
            report["error"] = ("primary REGF header is corrupt and no "
                               "transaction logs were given")
        elif not checksum_valid:
            report["error"] = ("primary REGF header is corrupt and the "
                               "transaction logs cannot establish a single "
                               "hive identity")
        return finish_report(report, primary_path, out_dir)

    # Verify every log against the hive identity before touching its pages.
    eligible_logs = []
    for log in logs:
        checks, reasons = evaluate_log(log, reference_fields)
        eligible = all(checks.values())
        valid_entries = [e for e in log.entries if e.valid]
        invalid_entries = [e for e in log.entries if not e.valid]
        report["logs"].append(OrderedDict([
            ("path", log.path),
            ("sha256", sha256_bytes(log.buf)),
            ("first_sequence", log.regf.hive_sequence2() if log.regf else None),
            ("entries_found", len(log.entries)),
            ("entries_valid", len(valid_entries)),
            ("invalid_entries", [OrderedDict([("log_entry_offset", e.offset),
                                              ("sequence", e.sequence)])
                                 for e in invalid_entries]),
            ("walk_stop_reason", log.walk_stop_reason),
            ("identity_checks", checks),
            ("eligible", eligible),
            ("exclusion_reasons", reasons),
        ]))
        if eligible:
            eligible_logs.append(log)

    # The generation the primary file is flushed up to; recovery starts here.
    header_source = None
    if checksum_valid:
        base = primary_regf.hive_sequence2()
    else:
        header_source = find_header_source(eligible_logs)
        base = (header_source.regf.hive_sequence2()
                if header_source is not None else None)

    recovery = OrderedDict()
    report["recovery"] = recovery
    recovery["base_generation"] = base
    recovery["header_restored_from"] = (header_source.path
                                        if header_source is not None else None)

    # Collect the candidate generations from all eligible logs.
    candidates = {}
    stale = []
    duplicates = []
    conflicts = []
    poisoned = set()
    if base is not None:
        for log in eligible_logs:
            for entry in log.entries:
                if not entry.valid:
                    continue
                seq = entry.sequence
                if seq != base and seq_is_older(seq, base):
                    stale.append((log, entry))
                    continue
                if seq in candidates:
                    other_log, other_entry = candidates[seq]
                    if other_entry.raw == entry.raw:
                        duplicates.append((other_log, log, entry))
                    else:
                        conflicts.append(seq)
                        poisoned.add(seq)
                    continue
                candidates[seq] = (log, entry)

    # Apply generations as a consecutive chain starting at the base
    # generation. The chain ends at the first missing or conflicting
    # generation -- never at the highest sequence number seen.
    chain = []
    current = base
    if base is not None:
        if base in candidates and base not in poisoned:
            # Already flushed to the primary; applied for completeness so the
            # result matches the reference implementation byte for byte.
            chain.append(candidates[base])
        while True:
            following = seq_next(current)
            if following in poisoned or following not in candidates:
                break
            chain.append(candidates[following])
            current = following
    final_generation = current

    applied_seqs = set(entry.sequence for _, entry in chain)
    orphaned = sorted(seq for seq in candidates
                      if seq not in applied_seqs and seq not in poisoned)

    gap = None
    if base is not None:
        following = seq_next(final_generation)
        if following in poisoned or orphaned:
            if following in poisoned:
                cause = "conflicting log entries for this generation"
            else:
                cause = "generation missing from all eligible logs"
            gap = OrderedDict([
                ("first_unprovable_generation", following),
                ("cause", cause),
                ("unapplied_generations", orphaned),
            ])

    recovery["applied_entries"] = [
        OrderedDict([("generation", entry.sequence),
                     ("source_log", log.path),
                     ("log_entry_offset", entry.offset),
                     ("dirty_pages", entry.dirty_pages_count),
                     ("hbins_size", entry.hbins_size),
                     ("hive_flags", entry.hive_flags)])
        for log, entry in chain
    ]
    recovery["recovered_range"] = [base, final_generation]
    recovery["gap"] = gap
    recovery["stale_entries"] = [
        OrderedDict([("generation", entry.sequence),
                     ("source_log", log.path),
                     ("log_entry_offset", entry.offset),
                     ("reason", "generation already flushed to the primary file")])
        for log, entry in stale
    ]
    recovery["duplicate_entries"] = [
        OrderedDict([("generation", entry.sequence),
                     ("kept_source_log", kept.path),
                     ("duplicate_source_log", log.path),
                     ("note", "identical content, applied once")])
        for kept, log, entry in duplicates
    ]
    recovery["conflicting_generations"] = sorted(conflicts)

    # Decide whether a provably consistent hive can be produced.
    needs_recovery = status.recover_header or status.recover_data
    advancing = [entry for _, entry in chain if entry.sequence != base]
    out_bytes = None
    if base is None:
        recovery["result"] = "no_provable_recovery"
        recovery["result_note"] = ("no eligible transaction log to restore "
                                   "the corrupt primary header from")
    elif needs_recovery and not advancing:
        recovery["result"] = "no_provable_recovery"
        recovery["result_note"] = (
            "no consecutive generations available past the flushed "
            "generation %d; the in-flight update cannot be proven" % base)
    else:
        out = bytearray(primary_bytes)
        if status.recover_header:
            out[0:REGF_HEADER_SIZE] = header_source.buf[0:REGF_HEADER_SIZE]
            # The restored header is a log file header; the recovered file
            # is a primary hive again.
            struct.pack_into("<I", out, 0x1C,
                             RegistryParse.FileType.FILE_TYPE_PRIMARY.value)
            recovery["header_file_type_corrected"] = True

        page_writes = []
        for log, entry in chain:
            pages = entry.hvle.dirty_pages_with_references()
            for page_index, (reference, page) in enumerate(pages):
                target = FIRST_HBIN_OFFSET + reference.offset()
                data = page.data()
                if target + len(data) > len(out):
                    out.extend(b"\x00" * (target + len(data) - len(out)))
                out[target:target + len(data)] = data
                page_writes.append(OrderedDict([
                    ("generation", entry.sequence),
                    ("source_log", log.path),
                    ("log_entry_offset", entry.offset),
                    ("page_index", page_index),
                    ("target_offset", target),
                    ("size", len(data)),
                    ("sha256", sha256_bytes(data)),
                ]))
        report["page_writes"] = page_writes
        report["page_provenance_note"] = (
            "For pages written more than once, the last write in this list "
            "wins; pages not listed come from the primary file unchanged.")

        if chain:
            final_hbins_size = chain[-1][1].hbins_size
            final_hive_flags = chain[-1][1].hive_flags
        else:
            final_hbins_size = primary_regf.hbins_size()
            final_hive_flags = primary_regf.hive_flags()
        struct.pack_into("<II", out, 0x4, final_generation, final_generation)
        struct.pack_into("<I", out, 0x28, final_hbins_size)
        struct.pack_into("<I", out, 0x90, final_hive_flags)
        header_regf = RegistryParse.REGFBlock(bytes(out[0:REGF_HEADER_SIZE]),
                                              0, False)
        struct.pack_into("<I", out, 0x1FC, header_regf.calculate_checksum())
        out_bytes = bytes(out)

        verification = OrderedDict()
        regf2 = RegistryParse.REGFBlock(out_bytes, 0, False)
        verification["regf_checksum_valid"] = regf2.validate_checksum()
        verification["sequence_numbers_equal"] = \
            regf2.validate_sequence_numbers()
        recovered_status = regf2.recovery_required()
        verification["recovery_required"] = (recovered_status.recover_header
                                             or recovered_status.recover_data)
        verification["final_generation"] = regf2.hive_sequence1()
        verification["final_generation_matches_chain"] = \
            regf2.hive_sequence1() == final_generation
        verification["hbins_size"] = regf2.hbins_size()
        verification["hbins_walked"] = sum(1 for _ in regf2.hbins())
        try:
            reg = Registry.Registry(io.BytesIO(out_bytes))
            verification["hive_parses"] = True
            verification["root_key_name"] = reg.root().name()
        except Exception as e:
            verification["hive_parses"] = False
            verification["parse_error"] = str(e)

        recovery["result"] = ("already_consistent"
                              if not needs_recovery and not chain
                              else "recovered")
        report["output"] = OrderedDict([
            ("path", None),
            ("size", len(out_bytes)),
            ("sha256", sha256_bytes(out_bytes)),
            ("verification", verification),
        ])

    return finish_report(report, primary_path, out_dir, out_bytes)


def finish_report(report, primary_path, out_dir, out_bytes=None):
    """Write the output files and prove the inputs were not modified."""
    inputs = [report["inputs"]["primary"]] + report["inputs"]["logs"]
    report["inputs_unmodified"] = all(
        sha256_bytes(read_file(entry["path"])) == entry["sha256"]
        for entry in inputs)

    if out_dir is not None:
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        base_name = os.path.basename(primary_path.rstrip(os.sep))
        input_paths = set(os.path.abspath(entry["path"]) for entry in inputs)
        if out_bytes is not None:
            hive_path = os.path.join(out_dir, base_name + ".recovered")
            if os.path.abspath(hive_path) in input_paths:
                raise ValueError("output path would overwrite an input file")
            with open(hive_path, "wb") as f:
                f.write(out_bytes)
            if report["output"] is not None:
                report["output"]["path"] = hive_path
        report_path = os.path.join(out_dir, base_name + ".recovery.json")
        if os.path.abspath(report_path) in input_paths:
            raise ValueError("report path would overwrite an input file")
        report["report_path"] = report_path
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
            f.write("\n")

    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Recover a registry hive from its transaction logs and "
                    "produce an auditable recovery report.")
    parser.add_argument("primary", help="path to the primary hive file")
    parser.add_argument("logs", nargs="+",
                        help="paths to the transaction log files")
    parser.add_argument("--out-dir", default=None,
                        help="directory for the recovered hive and the JSON "
                             "report (default: no files are written)")
    args = parser.parse_args(argv)

    report = recover_hive(args.primary, args.logs, out_dir=args.out_dir)

    assessment = report["primary_assessment"]
    print("Primary: sequence1=%d sequence2=%d header_checksum=%s "
          "recover_header=%s recover_data=%s"
          % (assessment["sequence1"], assessment["sequence2"],
             "valid" if assessment["header_checksum_valid"] else "INVALID",
             assessment["recover_header"], assessment["recover_data"]))
    print("Identity reference: %s" % report["identity_reference"])
    for log_report in report["logs"]:
        if log_report["eligible"]:
            print("Log %s: eligible (first sequence %s, %d valid entries)"
                  % (log_report["path"], log_report["first_sequence"],
                     log_report["entries_valid"]))
        else:
            print("Log %s: EXCLUDED (%s)"
                  % (log_report["path"],
                     "; ".join(log_report["exclusion_reasons"])))

    if "error" in report:
        print("Error: %s" % report["error"])
        return 2

    recovery = report["recovery"]
    low, high = recovery["recovered_range"]
    print("Applied %d consecutive generations: %d..%d"
          % (len(recovery["applied_entries"]), low, high))
    if recovery["gap"] is not None:
        print("Gap: generation %d is not provable (%s); %d later "
              "generation(s) left unapplied"
              % (recovery["gap"]["first_unprovable_generation"],
                 recovery["gap"]["cause"],
                 len(recovery["gap"]["unapplied_generations"])))
    if recovery["stale_entries"]:
        print("Skipped %d stale log entr(ies) already flushed to the primary"
              % len(recovery["stale_entries"]))
    print("Result: %s" % recovery["result"])

    if report["output"] is not None:
        verification = report["output"]["verification"]
        print("Recovered hive: generation %d, checksum %s, parses %s, "
              "sha256=%s"
              % (verification["final_generation"],
                 "valid" if verification["regf_checksum_valid"] else "INVALID",
                 "ok" if verification["hive_parses"] else "FAILED",
                 report["output"]["sha256"]))
        if report["output"]["path"]:
            print("Written: %s" % report["output"]["path"])
    if "report_path" in report:
        print("Report: %s" % report["report_path"])
    print("Inputs unmodified: %s" % report["inputs_unmodified"])

    if recovery["result"] == "no_provable_recovery":
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
