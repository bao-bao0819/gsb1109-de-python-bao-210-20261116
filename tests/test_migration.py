from __future__ import annotations

from io import BytesIO

import pytest

from dissect.cstruct import cstruct
from dissect.cstruct.expression import Expression
from dissect.cstruct.migration import (
    IMPACT_ENUM,
    IMPACT_LAYOUT,
    IMPACT_POINTER,
    ReplayResult,
    compare_values,
    impact_report,
    impacts_by_reason,
    migrate,
    replay_records,
    schema_signature,
    verify_replay,
)

ENUM_OLD = """
enum Status : uint8 { OFF = 0, ON = 1, UNKNOWN = 2 };
"""

ENUM_NEW = """
enum Status : uint8 { OFF = 0, ENABLED = 1, UNKNOWN = 2, DEGRADED = 3 };
"""


def _load(definition: str, *, compiled: bool = True) -> cstruct:
    return cstruct().load(definition, compiled=compiled)


def test_schema_signature_stable_for_identical_schemas() -> None:
    old_cs = _load(f"{ENUM_OLD} struct r {{ Status a; uint8 b; }};")
    new_cs = _load(f"{ENUM_OLD} struct r {{ Status a; uint8 b; }};")

    assert schema_signature(old_cs, "r") == schema_signature(new_cs, "r")

    changed_cs = _load(f"{ENUM_NEW} struct r {{ Status a; uint8 b; }};")
    assert schema_signature(old_cs, "r") != schema_signature(changed_cs, "r")


def test_enum_rename_is_detected_without_size_or_offset_change() -> None:
    old_cs = _load(f"{ENUM_OLD} struct r {{ Status state; uint8 x; }};")
    new_cs = _load(f"{ENUM_NEW} struct r {{ Status state; uint8 x; }};")

    report = impact_report(old_cs, new_cs, "r")
    assert report.affected
    assert report.size_unchanged
    assert report.affected_paths == frozenset({"state"})
    assert impacts_by_reason(report, IMPACT_ENUM)
    assert not impacts_by_reason(report, IMPACT_LAYOUT)

    raw = b"\x01\x09"
    old_obj = old_cs.r(raw)
    new_obj = new_cs.r(raw)

    # The raw integer and byte span are identical, but the business value changed and must not be reused.
    assert int(old_obj.state) == int(new_obj.state) == 1
    differences = compare_values(old_obj, new_obj)
    assert differences[0].path == "state"
    assert differences[0].old == ("enum", "Status", 1, "ON")
    assert differences[0].new == ("enum", "Status", 1, "ENABLED")


def test_primitive_to_enum_reinterpretation_is_both_layout_and_enum_impact() -> None:
    old_cs = _load("enum Code : uint32 { NONE = 0 }; struct r { uint32 code; };")
    new_cs = _load("enum Code : uint32 { NONE = 0, RUNNING = 1 }; struct r { Code code; };")

    report = impact_report(old_cs, new_cs, "r")
    field_impact = next(impact for impact in report.impacts if impact.path == "code")
    assert IMPACT_LAYOUT in field_impact.reasons
    assert IMPACT_ENUM in field_impact.reasons
    assert report.size_unchanged


def test_layout_impact_for_bitfield_change_with_unchanged_total_size() -> None:
    old_cs = _load("struct r { uint16 a:4; uint16 b:12; uint8 tail; };")
    new_cs = _load("struct r { uint16 a:6; uint16 b:10; uint8 tail; };")

    report = impact_report(old_cs, new_cs, "r")
    assert report.size_unchanged
    assert report.unconditional_paths == frozenset({"a", "b"})
    assert all(IMPACT_LAYOUT in impact.reasons for impact in report.impacts)

    raw = (1 | (200 << 4)).to_bytes(2, "little") + b"\x07"
    result = migrate(old_cs, new_cs, "r", [raw])
    assert result.verification.ok
    assert {diff.path for diff in result.replays[0].differences} == {"b"}


def test_layout_impact_reports_changed_total_size() -> None:
    old_cs = _load("struct r { uint8 a; uint8 b; };")
    new_cs = _load("struct r { uint8 a; uint16 b; };")

    report = impact_report(old_cs, new_cs, "r")
    assert not report.size_unchanged
    assert report.old_size == 2
    assert report.new_size == 3


def test_length_expression_change_is_evaluated_per_record(compiled: bool) -> None:
    old_cs = _load("struct r { uint8 n; char data[n + 1]; };", compiled=compiled)
    new_cs = _load("struct r { uint8 n; char data[(n & 1) + 1]; };", compiled=compiled)

    report = impact_report(old_cs, new_cs, "r")
    assert report.length_paths == frozenset({"data"})
    assert not report.unconditional_paths

    # n=1: old length 2, new length 2 -> expressions coincide, record is not affected.
    coinciding = old_cs.r(n=1, data=b"ab").dumps()
    # n=2: old length 3, new length 1 -> the records diverge and must be replayed.
    diverging = old_cs.r(n=2, data=b"abc").dumps()

    replays = replay_records(old_cs, new_cs, "r", [coinciding, diverging])
    assert [replay.index for replay in replays] == [1]
    assert replays[0].changed
    assert replays[0].differences[0].path == "data"
    assert replays[0].differences[0].old == b"abc"
    assert replays[0].differences[0].new == b"a"

    verification = verify_replay(old_cs, new_cs, "r", [coinciding, diverging], replays)
    assert verification.ok
    assert verification.replayed == 1
    assert verification.skipped == 1


def test_constant_change_behind_identical_expression_tokens_is_detected_per_record() -> None:
    old_cs = _load("#define N 1\nstruct r { uint8 n; char buf[n + N]; };")
    new_cs = _load("#define N 2\nstruct r { uint8 n; char buf[n + N]; };")

    report = impact_report(old_cs, new_cs, "r")
    assert report.length_paths == frozenset({"buf"})

    raw = old_cs.r(n=0, buf=b"q").dumps() + b"zz"
    result = migrate(old_cs, new_cs, "r", [raw])
    assert result.verification.ok
    assert result.replays[0].differences[0].old == b"q"
    assert result.replays[0].differences[0].new == b"qz"


def test_pointer_target_change_is_a_pointer_impact_and_reparses_dereferenced_value() -> None:
    old_cs = _load(
        """
        enum Code : uint32 { NONE = 0, RUNNING = 1 };
        struct entry { uint32 id; uint32 raw; };
        struct r { uint8 key; entry* ref; };
        """
    )
    new_cs = _load(
        """
        enum Code : uint32 { NONE = 0, RUNNING = 1, IDLE = 2 };
        struct entry { uint32 id; Code raw; };
        struct r { uint8 key; entry* ref; };
        """
    )

    report = impact_report(old_cs, new_cs, "r")
    assert report.size_unchanged
    pointer_impacts = impacts_by_reason(report, IMPACT_POINTER)
    assert {impact.path for impact in pointer_impacts} == {"ref"}
    assert any(impact.path == "ref.raw" and IMPACT_ENUM in impact.reasons for impact in report.impacts)

    target = old_cs.entry(id=5, raw=1).dumps()
    body = old_cs.r(key=1, ref=0).dumps()
    raw = body[:1] + len(body).to_bytes(8, "little") + target

    result = migrate(old_cs, new_cs, "r", [raw])
    assert result.verification.ok
    pointer_diff = next(diff for diff in result.replays[0].differences if diff.path == "ref")
    assert pointer_diff.old[2]["raw"] == 1
    assert pointer_diff.new[2]["raw"] == ("enum", "Code", 1, "RUNNING")


def test_null_pointer_normalizes_without_dereference() -> None:
    from dissect.cstruct.migration import _normalize

    old_cs = _load("struct entry { uint32 id; }; struct r { entry* ref; };")
    new_cs = _load("struct entry { uint32 id; uint32 tag; }; struct r { entry* ref; };")
    raw = (0).to_bytes(8, "little")

    report = impact_report(old_cs, new_cs, "r")
    assert IMPACT_POINTER in report.impacts[0].reasons
    result = migrate(old_cs, new_cs, "r", [raw])
    assert result.verification.ok

    # A null pointer is normalized without dereferencing and carries no target.
    assert _normalize(old_cs.r(raw)) == {"ref": ("pointer", 0, None)}
    assert compare_values(old_cs.r(raw), new_cs.r(raw)) == ()


def test_verify_replay_flags_a_missed_record_as_mismatch() -> None:
    old_cs = _load("enum E : uint8 { A = 0, B = 1 }; struct r { E e; uint8 x; };")
    new_cs = _load("enum E : uint8 { A = 0, C = 1 }; struct r { E e; uint8 x; };")
    raw = old_cs.r(e=old_cs.E.B, x=9).dumps()

    full_replays = replay_records(old_cs, new_cs, "r", [raw])
    assert full_replays

    # Simulate a buggy incremental run that replays nothing: verification must catch the stale business value.
    verification = verify_replay(old_cs, new_cs, "r", [raw], ())
    assert not verification.ok
    assert verification.mismatches[0].path == "[0].e"
    assert verification.mismatches[0].old == ("enum", "E", 1, "B")
    assert verification.mismatches[0].new == ("enum", "E", 1, "C")


def test_verify_replay_flags_replay_that_differs_from_full_parse() -> None:
    old_cs = _load("struct r { uint8 a; };")
    new_cs = _load("struct r { uint8 a; };")
    raw = b"\x01"

    stale = ReplayResult(0, raw, old_cs.r(raw), old_cs.r(b"\x02"), ())
    verification = verify_replay(old_cs, new_cs, "r", [raw], [stale])
    assert not verification.ok
    assert "full parse" in verification.mismatches[0].old


def test_unchanged_schema_replays_nothing() -> None:
    old_cs = _load(f"{ENUM_OLD} struct r {{ Status a; }};")
    new_cs = _load(f"{ENUM_OLD} struct r {{ Status a; }};")

    report = impact_report(old_cs, new_cs, "r")
    assert not report.affected
    result = migrate(old_cs, new_cs, "r", [b"\x01", b"\x02"])
    assert result.replays == ()
    assert result.verification.ok
    assert result.verification.skipped == 2


def test_anonymous_struct_fields_are_folded_into_field_paths() -> None:
    old_cs = _load("struct r { struct { uint16 a; uint16 b; }; uint8 tail; };")
    new_cs = _load("struct r { struct { uint32 a; uint16 b; }; uint8 tail; };")

    report = impact_report(old_cs, new_cs, "r")
    assert "a" in report.affected_paths
    assert "b" in report.affected_paths


def test_union_reinterpretation_keeps_byte_span_and_changes_business_value() -> None:
    old_cs = _load("union u { uint32 n; char s[4]; }; struct box { uint8 k; u val; };")
    new_cs = _load("enum E : uint32 { X = 0, Y = 1 }; union u { E n; char s[4]; }; struct box { uint8 k; u val; };")

    report = impact_report(old_cs, new_cs, "box")
    assert report.size_unchanged
    assert any(IMPACT_ENUM in impact.reasons for impact in report.impacts)

    raw = old_cs.box(k=1, val=old_cs.u(n=1)).dumps()
    result = migrate(old_cs, new_cs, "box", [raw])
    assert result.verification.ok
    n_diff = next(diff for diff in result.replays[0].differences if diff.path == "val.n")
    assert n_diff.old == 1
    assert n_diff.new == ("enum", "E", 1, "Y")


def test_replay_accepts_bytes_like_inputs() -> None:
    old_cs = _load("enum E : uint8 { A = 0, B = 1 }; struct r { E e; };")
    new_cs = _load("enum E : uint8 { A = 0, C = 1 }; struct r { E e; };")

    replays = replay_records(old_cs, new_cs, "r", [bytearray(b"\x01"), memoryview(b"\x01")])
    assert [replay.index for replay in replays] == [0, 1]
    assert all(isinstance(replay.raw, bytes) for replay in replays)


def test_replay_uses_fresh_parse_and_not_old_values() -> None:
    old_cs = _load("enum E : uint8 { A = 0, B = 1 }; struct r { E e; };")
    new_cs = _load("enum E : uint8 { A = 0, C = 1 }; struct r { E e; };")
    raw = b"\x01"

    replays = replay_records(old_cs, new_cs, "r", [raw])
    new_obj = new_cs.r(BytesIO(raw))
    assert replays[0].new_parsed == new_obj
    assert replays[0].new_parsed is not old_cs.r(raw)


def test_impacts_by_reason_rejects_unknown_reason() -> None:
    old_cs = _load("struct r { uint8 a; };")
    report = impact_report(old_cs, old_cs, "r")
    with pytest.raises(ValueError, match="Unknown impact reason"):
        impacts_by_reason(report, "bogus")


def test_self_referencing_structure_does_not_recurse_forever() -> None:
    old_cs = _load("struct node { uint32 value; node* next; char tag[value & 1]; };")
    new_cs = _load("struct node { uint32 value; node* next; char tag[(value & 1) + 1]; };")

    report = impact_report(old_cs, new_cs, "node")
    assert report.length_paths == frozenset({"tag"})

    # value=0: old length 0, new length 1 -> affected; the self pointer does not trigger infinite recursion.
    raw = (0).to_bytes(4, "little") + (0).to_bytes(8, "little") + b"z"
    result = migrate(old_cs, new_cs, "node", [raw])
    assert result.verification.ok
    assert [replay.index for replay in result.replays] == [0]


def test_dangling_pointer_dereference_error_is_surfaced_not_raised() -> None:
    from dissect.cstruct.migration import _normalize

    old_cs = _load("struct entry { uint32 id; }; struct r { entry* ref; };")
    new_cs = _load("struct entry { uint32 id; }; struct r { entry* ref; };")

    # The pointer claims a target past the end of the record stream.
    raw = (99).to_bytes(8, "little")
    normalized = _normalize(old_cs.r(raw))
    assert normalized["ref"][1] == 99
    assert normalized["ref"][2][0] == "<deref error>"

    # Comparison against the new schema still completes and reports the same error on both sides as no change.
    assert compare_values(old_cs.r(raw), new_cs.r(raw)) == ()


def test_unevaluable_length_expression_fails_safe_by_replaying(monkeypatch: pytest.MonkeyPatch) -> None:
    old_cs = _load("struct r { uint8 n; char data[n + 1]; };")
    new_cs = _load("struct r { uint8 n; char data[(n & 1) + 1]; };")
    raw = old_cs.r(n=2, data=b"abc").dumps()

    # A length-only change where evaluating the new expression for a record fails must default to "affected".
    new_expr = new_cs.r.__members__["data"].type.__num_entries__
    assert isinstance(new_expr, Expression)

    def _boom(cs: object, context: object) -> None:
        raise RuntimeError("cannot evaluate")

    monkeypatch.setattr(new_expr, "evaluate", _boom)
    replays = replay_records(old_cs, new_cs, "r", [raw])
    assert [replay.index for replay in replays] == [0]

