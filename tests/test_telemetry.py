from __future__ import annotations

import pytest

from dissect.cstruct import (
    FieldKind,
    FieldSpec,
    FieldState,
    InsufficientBudgetError,
    MissingOriginalError,
    OffsetInvarianceError,
    TelemetryView,
    build_view,
    cstruct,
    plan_sidecar,
    reimport_telemetry,
)
from dissect.cstruct.exception import TelemetryError
from tests.util import verify_compiled

CDEF = """
struct header {
    char  magic[4];
    char  name[];
};

struct entry {
    uint8  version;
    wchar  label[4];
    char   note[];
};

struct telemetry {
    header hdr;
    entry  entries[2];
    wchar  title[];
    char   trailer[];
};
"""


def _telemetry_data() -> bytes:
    return (
        b"PF\x00\x00"  # magic: fixed bytes with padding
        b"probe\x00"  # name: zero terminated
        b"\x01"
        + "AB".encode("utf-16-le")
        + b"\x00\x00\x00\x00"  # label: padded wide characters
        + b"first\x00"  # note: zero terminated
        + b"\x02"
        + "C".encode("utf-16-le")
        + b"\x00\x00\x00\x00\x00\x00"
        + b"second\x00"
        + "go".encode("utf-16-le")
        + b"\x00\x00"  # title: zero terminated wide
        + b"tail\x00"  # trailer: zero terminated
    )


def test_view_kinds_and_boundaries(cs: cstruct, compiled: bool) -> None:
    """Every declaration kind is recognized with its exact offset and length, for compiled and interpreted reads."""
    cs.load(CDEF, compiled=compiled)
    assert verify_compiled(cs.telemetry, compiled)

    data = _telemetry_data()
    view = build_view(cs.telemetry(data))

    assert [(field.name, field.kind, field.offset, field.length) for field in view] == [
        ("hdr.magic", FieldKind.FIXED_BYTES, 0, 4),
        ("hdr.name", FieldKind.ZERO_TERMINATED, 4, 6),
        ("entries[0].label", FieldKind.WIDE_PADDED, 11, 8),
        ("entries[0].note", FieldKind.ZERO_TERMINATED, 19, 6),
        ("entries[1].label", FieldKind.WIDE_PADDED, 26, 8),
        ("entries[1].note", FieldKind.ZERO_TERMINATED, 34, 7),
        ("title", FieldKind.ZERO_TERMINATED, 41, 6),
        ("trailer", FieldKind.ZERO_TERMINATED, 47, 5),
    ]
    assert view.texts == {
        "hdr.magic": "PF",
        "hdr.name": "probe",
        "entries[0].label": "AB",
        "entries[0].note": "first",
        "entries[1].label": "C",
        "entries[1].note": "second",
        "title": "go",
        "trailer": "tail",
    }
    assert all(field.state is FieldState.TEXT for field in view)
    assert len(view) == 8


def test_empty_unterminated_and_truncated_are_distinct() -> None:
    """A genuine empty string, a missing terminator and a truncated code unit are separate states."""
    data = b"\x00" + b"abc" + b"a\x00b\x00\x00"
    view = TelemetryView.from_parts(
        [
            FieldSpec("empty", FieldKind.ZERO_TERMINATED, 0, 1),
            FieldSpec("missing", FieldKind.ZERO_TERMINATED, 1, 3),
            FieldSpec("truncated", FieldKind.ZERO_TERMINATED, 4, 5, unit=2),
        ],
        data,
    )

    empty, missing, truncated = view

    assert empty.state is FieldState.EMPTY
    assert empty.text == ""
    assert empty.lossless

    assert missing.state is FieldState.UNTERMINATED
    assert missing.text == "abc"
    assert missing.is_unterminated
    assert not missing.lossless

    assert truncated.state is FieldState.TRUNCATED
    assert truncated.is_truncated
    assert truncated.truncated == b"\x00"
    assert not truncated.lossless

    assert [field.name for field in view.problems] == ["missing", "truncated"]
    assert {field.name for field in view.lossy} == {"missing", "truncated"}


def test_collision_after_transcoding() -> None:
    """Byte and wide encodings that decode to the same text, but from different bytes, are a collision."""
    utf8 = "é".encode()
    wide = "é".encode("utf-16-le") + b"\x00\x00"
    data = b"\xe9\x00" + utf8 + b"\x00" + wide

    view = TelemetryView.from_parts(
        [
            FieldSpec("latin1", FieldKind.ZERO_TERMINATED, 0, 2),
            FieldSpec("utf8", FieldKind.ZERO_TERMINATED, 2, 3),
            FieldSpec("wide", FieldKind.ZERO_TERMINATED, 5, 4, unit=2),
        ],
        data,
    )

    assert view.texts["latin1"] == "�"
    assert view.texts["utf8"] == view.texts["wide"] == "é"

    collisions = view.collisions
    assert len(collisions) == 1
    assert collisions[0].text == "é"
    assert collisions[0].names == ("utf8", "wide")
    assert collisions[0].payloads == (utf8, "é".encode("utf-16-le"))

    # The invalid byte string does not collide with anything, but it cannot be reproduced from text
    assert view.lossy == (view[0],)


def test_collision_includes_terminator_removal() -> None:
    """A terminated and an unterminated value with the same payload collide in value but keep their own state."""
    data = b"abc\x00abc"
    view = TelemetryView.from_parts(
        [
            FieldSpec("terminated", FieldKind.ZERO_TERMINATED, 0, 4),
            FieldSpec("unterminated", FieldKind.ZERO_TERMINATED, 4, 3),
        ],
        data,
    )

    assert [field.text for field in view] == ["abc", "abc"]
    assert [field.state for field in view] == [FieldState.TEXT, FieldState.UNTERMINATED]

    collision = view.collisions[0]
    assert collision.names == ("terminated", "unterminated")
    assert collision.payloads == (b"abc",)
    assert collision.raws == (b"abc\x00", b"abc")


def test_lone_surrogate_collision() -> None:
    """A lone surrogate is unified into valid text and collides with an invalid byte string."""
    data = b"\xff\x00" + b"\x3d\xd8"
    view = TelemetryView.from_parts(
        [
            FieldSpec("bytes", FieldKind.ZERO_TERMINATED, 0, 2),
            FieldSpec("wide", FieldKind.WIDE_PADDED, 2, 2),
        ],
        data,
    )

    assert view.texts == {"bytes": "�", "wide": "�"}
    assert view.collisions[0].names == ("bytes", "wide")
    assert all(field.needs_original for field in view)


def test_duplicate_values() -> None:
    """Fields with identical text and identical bytes are duplicates, not collisions."""
    data = b"same\x00same\x00"
    view = TelemetryView.from_parts(
        [
            FieldSpec("a", FieldKind.ZERO_TERMINATED, 0, 5),
            FieldSpec("b", FieldKind.ZERO_TERMINATED, 5, 5),
        ],
        data,
    )

    assert view.collisions == ()
    assert view.duplicates[0].names == ("a", "b")


def test_budget_selects_required_originals() -> None:
    """The budget must hold every lossy original; more budget cannot drop any of them."""
    data = b"ok\x00" + b"missing" + b"a\x00b\x00\x00"
    view = TelemetryView.from_parts(
        [
            FieldSpec("text", FieldKind.ZERO_TERMINATED, 0, 3),
            FieldSpec("missing", FieldKind.ZERO_TERMINATED, 3, 7),
            FieldSpec("truncated", FieldKind.ZERO_TERMINATED, 10, 5, unit=2),
        ],
        data,
    )

    required = [field.name for field in view.plan().selection]
    assert required == ["missing", "truncated"]
    assert view.plan().total_size == 7 + 5

    tight = plan_sidecar(view, budget=12)
    assert tight.feasible
    assert tight.deficit == 0
    assert [field.name for field in tight.selection] == ["missing", "truncated"]

    short = plan_sidecar(view, budget=11)
    assert not short.feasible
    assert short.deficit == 1

    with pytest.raises(InsufficientBudgetError, match="1 bytes short"):
        view.export(budget=11)

    with pytest.raises(InsufficientBudgetError) as exc_info:
        plan_sidecar(view, budget=5).raise_if_infeasible()
    assert exc_info.value.plan.deficit == 7

    # An unlimited budget keeps exactly the same originals
    assert view.export().sidecar.keys() == view.export(budget=1000).sidecar.keys()


def test_budget_overhead() -> None:
    """Fixed per-original overhead counts towards the budget."""
    data = b"missing"
    view = TelemetryView.from_parts([FieldSpec("a", FieldKind.ZERO_TERMINATED, 0, 7)], data)

    assert view.plan(budget=7).feasible
    assert not view.plan(budget=7, overhead=2).feasible
    assert view.plan(budget=9, overhead=2).feasible


def test_reimport_patches_in_place(cs: cstruct, compiled: bool) -> None:
    """An export rebuilds the exact original bytes and never changes any field offset."""
    cs.load(CDEF, compiled=compiled)
    data = _telemetry_data()

    view = build_view(cs.telemetry(data))
    export = view.export()

    restored = view.reimport(export)
    assert restored == data
    assert len(restored) == len(data)

    # The template is zeroed at every field, but has the same size
    assert len(export.template) == len(data)
    assert all(
        export.template[spec.offset : spec.offset + spec.length] == b"\x00" * spec.length for spec in export.layout
    )

    # The dictionary form survives a JSON-like round trip
    assert reimport_telemetry(export.to_dict()) == data


def test_reimport_requires_originals() -> None:
    """A lossy field cannot be rebuilt from text alone, and a missing original is reported."""
    data = b"abc\x00" + b"missing"
    view = TelemetryView.from_parts(
        [
            FieldSpec("terminated", FieldKind.ZERO_TERMINATED, 0, 4),
            FieldSpec("unterminated", FieldKind.ZERO_TERMINATED, 4, 7),
        ],
        data,
    )

    export = view.export()
    without_sidecar = export.to_dict()
    without_sidecar["sidecar"] = {}

    with pytest.raises(MissingOriginalError, match="unterminated"):
        reimport_telemetry(without_sidecar)

    with pytest.raises(MissingOriginalError):
        view[1].rebuild()

    # The lossless field still rebuilds from its text
    assert view[0].rebuild() == b"abc\x00"


def test_offset_invariance() -> None:
    """Text that does not fit its declared boundary is rejected instead of shifting later fields."""
    view = TelemetryView.from_parts(
        [
            FieldSpec("name", FieldKind.ZERO_TERMINATED, 0, 4),
            FieldSpec("after", FieldKind.ZERO_TERMINATED, 4, 2),
        ],
        b"abc\x00x\x00",
    )

    with pytest.raises(OffsetInvarianceError, match="needs a terminator inside its 4 byte boundary"):
        view[0].rebuild("toolong")

    # A text that leaves exactly one byte for the terminator fits the boundary
    assert view[0].rebuild("abc") == b"abc\x00"

    with pytest.raises(OffsetInvarianceError, match="needs a terminator inside its 4 byte boundary"):
        view[0].rebuild("abcd")


def test_wide_padded_rebuild() -> None:
    """Padded wide fields rebuild to their exact boundary, including the padding."""
    view = TelemetryView.from_parts(
        [FieldSpec("label", FieldKind.WIDE_PADDED, 0, 8, unit=2)],
        "AB".encode("utf-16-le") + b"\x00\x00\x00\x00",
    )

    field = view[0]
    assert field.text == "AB"
    assert field.rebuild() == "AB".encode("utf-16-le") + b"\x00\x00\x00\x00"
    assert field.rebuild("A") == b"A\x00" + b"\x00" * 6


def test_big_endian_wide() -> None:
    """Wide fields decode and rebuild according to the view endianness."""
    data = b"\x00A\x00B" + b"\x00\x00"
    view = TelemetryView.from_parts(
        [FieldSpec("label", FieldKind.WIDE_PADDED, 0, 6, unit=2)],
        data,
        endian=">",
    )

    assert view[0].text == "AB"
    assert view[0].rebuild() == data


def test_explicit_specs_validation() -> None:
    """Invalid boundaries are rejected up front."""
    with pytest.raises(ValueError, match="exceeds the data size"):
        TelemetryView.from_parts([FieldSpec("a", FieldKind.ZERO_TERMINATED, 0, 5)], b"abcd")

    with pytest.raises(ValueError, match="Duplicate telemetry field name"):
        TelemetryView.from_parts(
            [
                FieldSpec("a", FieldKind.ZERO_TERMINATED, 0, 1),
                FieldSpec("a", FieldKind.ZERO_TERMINATED, 1, 1),
            ],
            b"ab",
        )

    with pytest.raises(TelemetryError, match="overlaps"):
        TelemetryView.from_parts(
            [
                FieldSpec("a", FieldKind.FIXED_BYTES, 0, 4),
                FieldSpec("b", FieldKind.FIXED_BYTES, 2, 4),
            ],
            b"abcdef",
        )

    numeric = cstruct(load="struct t { uint32 a; };")
    with pytest.raises(TelemetryError, match="no telemetry string fields"):
        build_view(numeric.t(b"\x00" * 4))


def test_union_boundaries(cs: cstruct, compiled: bool) -> None:
    """Union string members are opt-in, because their overlapping boundaries need care."""
    cdef = """
    struct test {
        uint8 before;
        union {
            char  as_text[4];
            uint32 as_int;
        } u;
        char after;
    };
    """
    cs.load(cdef, compiled=compiled)

    data = b"\x01WXYZ\x02"
    obj = cs.test(data)

    view = build_view(obj, include_unions=True)
    assert [(field.name, field.offset, field.length) for field in view] == [
        ("u.as_text", 1, 4),
    ]

    # A union member is a real member of the view; leaving it in the template keeps it intact
    assert view.texts == {"u.as_text": "WXYZ"}
    assert view.export().rebuild() == data

    with pytest.raises(TelemetryError, match="no telemetry string fields"):
        build_view(obj)
