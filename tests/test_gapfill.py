from __future__ import annotations

import pytest

from dissect.cstruct.cstruct import cstruct

from gapfill import (
    Candidate,
    ConstraintSpec,
    Gap,
    format_report,
    solve,
)

MAGIC = 0xBEEF
VERSION = 1
COUNT = 2
CHECKSUM = 0x03  # sum(payload) & 0xFF must equal this
TOTAL_LEN = 4 + COUNT + 3


def make_parser():
    cs = cstruct()
    cs.load(
        """
        struct header {
            uint16  magic;
            uint8   version;
            uint8   count;
        };
        struct trailer {
            uint16  total_len;
            uint8   checksum;
        };
        """
    )

    def parse(buf: bytes) -> dict:
        header = cs.header(buf)
        payload = bytes(buf[4 : 4 + header.count])
        if len(payload) != header.count:
            raise EOFError("payload truncated")
        trailer = cs.trailer(buf[4 + header.count :])
        return {
            "magic": header.magic,
            "version": header.version,
            "count": header.count,
            "payload": payload,
            "total_len": trailer.total_len,
            "checksum": trailer.checksum,
            "buffer_len": len(buf),
        }

    return parse


def make_sample() -> bytes:
    cs = cstruct()
    cs.load(
        """
        struct header {
            uint16  magic;
            uint8   version;
            uint8   count;
        };
        struct trailer {
            uint16  total_len;
            uint8   checksum;
        };
        """
    )
    header = cs.header(magic=MAGIC, version=VERSION, count=COUNT)
    trailer = cs.trailer(total_len=TOTAL_LEN, checksum=CHECKSUM)
    # gap [4, 4): the payload segment is missing entirely
    return bytes(header) + bytes(trailer)


def base_conditions() -> list[ConstraintSpec]:
    return [
        ConstraintSpec("magic_ok", lambda f: f["magic"] == MAGIC),
        ConstraintSpec("version_ok", lambda f: f["version"] == VERSION),
        ConstraintSpec(
            "length_consistent",
            lambda f: f["total_len"] == f["buffer_len"] == 4 + f["count"] + 3,
        ),
        ConstraintSpec(
            "checksum_ok",
            lambda f: (sum(f["payload"]) & 0xFF) == f["checksum"],
        ),
    ]


def candidates() -> list[Candidate]:
    return [
        Candidate("good-a", b"\x01\x02"),      # sum 3 -> legal
        Candidate("good-b", b"\x02\x01"),      # sum 3 -> legal
        Candidate("good-c", b"\x03\x00"),      # sum 3 -> legal
        Candidate("too-long", b"\x01\x02\x03"),  # shifts trailer -> bad length/checksum
        Candidate("too-short", b"\x01"),       # payload truncated -> parse error
        Candidate("bad-sum", b"\x05\x05"),     # sum 10 -> checksum FAIL
    ]


def test_all_legal_completions_are_found():
    result = solve(make_sample(), Gap(4, 4), candidates(), make_parser(), base_conditions())
    assert result.solved
    assert {ev.candidate.label for ev in result.solutions} == {"good-a", "good-b", "good-c"}


def test_common_fields_and_domains():
    result = solve(make_sample(), Gap(4, 4), candidates(), make_parser(), base_conditions())
    assert result.common == {
        "magic": MAGIC,
        "version": VERSION,
        "count": COUNT,
        "total_len": TOTAL_LEN,
        "checksum": CHECKSUM,
        "buffer_len": TOTAL_LEN,
    }
    payload_domain = result.domains["payload"]
    assert not payload_domain.is_fixed
    assert sorted(payload_domain.values) == [b"\x01\x02", b"\x02\x01", b"\x03\x00"]
    # fixed numeric fields expose their (singleton) range
    assert result.domains["total_len"].minimum == result.domains["total_len"].maximum == TOTAL_LEN


def test_length_change_shifts_trailing_fields():
    # "too-long" splices 3 bytes into an empty gap; the trailer then sits one
    # byte earlier than the parser expects, so the re-parsed total_len is garbage.
    result = solve(make_sample(), Gap(4, 4), candidates(), make_parser(), base_conditions())
    evidence = {ev.candidate.label: ev for ev in result.evidence}
    shifted = evidence["too-long"]
    assert shifted.parsed  # it parses, but is not legal
    assert shifted.fields["total_len"] != TOTAL_LEN
    assert "length_consistent" in shifted.failed_conditions
    assert not shifted.valid


def test_evidence_is_kept_for_every_candidate():
    result = solve(make_sample(), Gap(4, 4), candidates(), make_parser(), base_conditions())
    evidence = {ev.candidate.label: ev for ev in result.evidence}
    assert len(result.evidence) == 6
    assert evidence["too-short"].parse_error is not None
    assert evidence["bad-sum"].failed_conditions == ["checksum_ok"]
    assert evidence["good-a"].valid


def test_placeholder_gap_with_different_candidate_length():
    # gap [4, 7) holds 3 placeholder bytes; a 2-byte candidate shifts the
    # trailer left by one byte, and the whole buffer is re-parsed accordingly.
    sample = make_sample()[:4] + b"\xaa\xbb\xcc" + make_sample()[4:]
    result = solve(
        sample,
        Gap(4, 7),
        [Candidate("fits", b"\x01\x02"), Candidate("overflow", b"\x01\x02\x03\x04")],
        make_parser(),
        base_conditions(),
    )
    assert [ev.candidate.label for ev in result.solutions] == ["fits"]
    assert len(result.solutions[0].buffer) == TOTAL_LEN


def test_conflicting_conditions_are_reported():
    # both conditions are individually satisfiable (good-* pass checksum_ok,
    # bad-sum passes first_byte_05) but no parseable candidate passes both
    conditions = base_conditions() + [
        ConstraintSpec("first_byte_05", lambda f: f["payload"][0] == 0x05),
    ]
    result = solve(make_sample(), Gap(4, 4), candidates(), make_parser(), conditions)
    assert not result.solved
    cores = [set(c.conditions) for c in result.conflicts]
    assert {"checksum_ok", "first_byte_05"} in cores
    core = next(c for c in result.conflicts if "first_byte_05" in c.conditions)
    # evidence: which parseable candidates satisfy each side of the conflict
    assert core.satisfiers["first_byte_05"] == ["bad-sum"]
    # "too-long" shifts the trailer, so its parsed checksum is 0x00 and it
    # fails checksum_ok along with the good candidates' complement
    assert core.satisfiers["checksum_ok"] == ["good-a", "good-b", "good-c"]


def test_individually_unsatisfiable_condition_is_smaller_core():
    # version==2 holds for no candidate, so it alone is the minimal core even
    # though version==1 and version==2 are also mutually exclusive
    conditions = base_conditions() + [
        ConstraintSpec("version_is_one", lambda f: f["version"] == 1),
        ConstraintSpec("version_is_two", lambda f: f["version"] == 2),
    ]
    result = solve(make_sample(), Gap(4, 4), candidates(), make_parser(), conditions)
    assert not result.solved
    cores = [set(c.conditions) for c in result.conflicts]
    assert {"version_is_two"} in cores
    assert {"version_is_one", "version_is_two"} not in cores


def test_unsatisfiable_single_condition_is_a_conflict():
    conditions = base_conditions() + [
        ConstraintSpec("checksum_impossible", lambda f: f["checksum"] == 0xFF),
    ]
    result = solve(make_sample(), Gap(4, 4), candidates(), make_parser(), conditions)
    assert not result.solved
    cores = [set(c.conditions) for c in result.conflicts]
    assert {"checksum_impossible"} in cores


def test_no_parseable_candidate():
    result = solve(
        make_sample(),
        Gap(4, 4),
        [Candidate("short-1", b"\x01"), Candidate("empty", b"")],
        make_parser(),
        base_conditions(),
    )
    assert not result.solved
    assert len(result.unparseable) == 2
    assert result.conflicts == []


def test_solver_does_not_stop_at_first_parseable():
    # "too-long" parses fine and comes first; a first-hit solver would wrongly
    # accept it.  The legal completions are the later, fully valid ones.
    ordered = [candidates()[3], *candidates()[:3]]
    result = solve(make_sample(), Gap(4, 4), ordered, make_parser(), base_conditions())
    assert result.evidence[0].candidate.label == "too-long"
    assert result.evidence[0].parsed and not result.evidence[0].valid
    assert {ev.candidate.label for ev in result.solutions} == {"good-a", "good-b", "good-c"}


def test_format_report_contains_evidence_and_domains():
    result = solve(make_sample(), Gap(4, 4), candidates(), make_parser(), base_conditions())
    report = format_report(result)
    assert "candidate 'too-short'" in report
    assert "parse: FAILED" in report
    assert "checksum_ok: FAIL" in report
    assert "3 legal completion(s)" in report
    assert "magic = 48879" in report
    assert "payload:" in report


def test_format_report_for_conflicts():
    conditions = base_conditions() + [
        ConstraintSpec("version_is_two", lambda f: f["version"] == 2),
    ]
    result = solve(make_sample(), Gap(4, 4), candidates(), make_parser(), conditions)
    report = format_report(result)
    assert "no legal completion" in report
    assert "version_is_two" in report


def test_invalid_gap_rejected():
    with pytest.raises(ValueError, match="invalid gap"):
        Gap(5, 2)
    with pytest.raises(ValueError, match="exceeds sample"):
        solve(b"\x00" * 4, Gap(2, 10), [Candidate("x", b"")], make_parser(), [])
