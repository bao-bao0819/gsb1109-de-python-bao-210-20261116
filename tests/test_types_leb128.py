from __future__ import annotations

import io
from typing import TYPE_CHECKING

import pytest

from dissect.cstruct.exception import LEB128NormalizationError, SignedRangeError
from dissect.cstruct.types.leb128 import (
    SignedRange,
    decode_leb128,
    encode_leb128,
    iter_leb128_values,
    leb128_aggregation,
    leb128_evidence,
    normalize_leb128,
)

if TYPE_CHECKING:
    from dissect.cstruct.cstruct import cstruct


def test_leb128_unsigned_read_EOF(cs: cstruct) -> None:
    with pytest.raises(EOFError, match="EOF reached, while final LEB128 byte was not yet read"):
        cs.uleb128(b"\x8b")


def test_leb128_unsigned_read(cs: cstruct) -> None:
    assert cs.uleb128(b"\x02") == 2
    assert cs.uleb128(b"\x8b\x25") == 4747
    assert cs.uleb128(b"\xc9\x8f\xb0\x06") == 13371337
    assert cs.uleb128(b"\x7e") == 126
    assert cs.uleb128(b"\xf5\x5a") == 11637
    assert cs.uleb128(b"\xde\xd6\xcf\x7c") == 261352286


def test_leb128_signed_read(cs: cstruct) -> None:
    assert cs.ileb128(b"\x02") == 2
    assert cs.ileb128(b"\x8b\x25") == 4747
    assert cs.ileb128(b"\xc9\x8f\xb0\x06") == 13371337
    assert cs.ileb128(b"\x7e") == -2
    assert cs.ileb128(b"\xf5\x5a") == -4747
    assert cs.ileb128(b"\xde\xd6\xcf\x7c") == -7083170


def test_leb128_struct_unsigned(cs: cstruct) -> None:
    cdef = """
    struct test {
        uleb128 len;
        char    data[len];
    };
    """
    cs.load(cdef)

    buf = b"\xaf\x18"
    buf += b"\x41" * 3119
    obj = cs.test(buf)

    assert obj.len == 3119
    assert obj.data == (b"\x41" * 3119)
    assert len(obj.data) == 3119
    assert len(buf) == 3119 + 2

    assert obj.dumps() == buf


def test_leb128_struct_unsigned_zero(cs: cstruct) -> None:
    cdef = """
    struct test {
        uleb128 numbers[];
    };
    """
    cs.load(cdef)

    buf = b"\xaf\x18\x8b\x25\xc9\x8f\xb0\x06\x00"
    obj = cs.test(buf)

    assert len(obj.numbers) == 3
    assert obj.numbers[0] == 3119
    assert obj.numbers[1] == 4747
    assert obj.numbers[2] == 13371337

    assert obj.dumps() == buf


def test_leb128_struct_signed_zero(cs: cstruct) -> None:
    cdef = """
    struct test {
        ileb128 numbers[];
    };
    """
    cs.load(cdef)

    buf = b"\xaf\x18\xf5\x5a\xde\xd6\xcf\x7c\x00"
    obj = cs.test(buf)

    assert len(obj.numbers) == 3
    assert obj.numbers[0] == 3119
    assert obj.numbers[1] == -4747
    assert obj.numbers[2] == -7083170

    assert obj.dumps() == buf


def test_leb128_nested_struct_unsigned(cs: cstruct) -> None:
    cdef = """
    struct entry {
        uleb128 len;
        char    data[len];
        uint32  crc;
    };
    struct nested {
        uleb128 name_len;
        char    name[name_len];
        uleb128 n_entries;
        entry   entries[n_entries];
    };
    """
    cs.load(cdef)

    # Dummy file format specifying 300 entries
    buf = b"\x08\x54\x65\x73\x74\x66\x69\x6c\x65\xac\x02"

    # Each entry has 4 byte data + 4 byte CRC
    buf += b"\x04\x41\x41\x41\x41\x42\x42\x42\x42" * 300

    obj = cs.nested(buf)

    assert obj.name_len == 8
    assert obj.name == b"\x54\x65\x73\x74\x66\x69\x6c\x65"
    assert obj.n_entries == 300

    assert obj.dumps() == buf


def test_leb128_nested_struct_signed(cs: cstruct) -> None:
    cdef = """
    struct entry {
        ileb128 len;
        char    data[len];
        uint32  crc;
    };
    struct nested {
        ileb128 name_len;
        char    name[name_len];
        ileb128 n_entries;
        entry   entries[n_entries];
    };
    """
    cs.load(cdef)

    # Dummy file format specifying 300 entries
    buf = b"\x08\x54\x65\x73\x74\x66\x69\x6c\x65\xac\x02"

    # Each entry has 4 byte data + 4 byte CRC
    buf += b"\x04\x41\x41\x41\x41\x42\x42\x42\x42" * 300

    obj = cs.nested(buf)

    assert obj.name_len == 8
    assert obj.name == b"\x54\x65\x73\x74\x66\x69\x6c\x65"
    assert obj.n_entries == 300

    assert obj.dumps() == buf


def test_leb128_unsigned_write(cs: cstruct) -> None:
    assert cs.uleb128(2).dumps() == b"\x02"
    assert cs.uleb128(4747).dumps() == b"\x8b\x25"
    assert cs.uleb128(13371337).dumps() == b"\xc9\x8f\xb0\x06"
    assert cs.uleb128(126).dumps() == b"\x7e"
    assert cs.uleb128(11637).dumps() == b"\xf5\x5a"
    assert cs.uleb128(261352286).dumps() == b"\xde\xd6\xcf\x7c"

    assert cs.uleb128(b"\xde\xd6\xcf\x7c").dumps() == b"\xde\xd6\xcf\x7c"


def test_leb128_signed_write(cs: cstruct) -> None:
    assert cs.ileb128(2).dumps() == b"\x02"
    assert cs.ileb128(4747).dumps() == b"\x8b\x25"
    assert cs.ileb128(13371337).dumps() == b"\xc9\x8f\xb0\x06"
    assert cs.ileb128(-2).dumps() == b"\x7e"
    assert cs.ileb128(-4747).dumps() == b"\xf5\x5a"
    assert cs.ileb128(-7083170).dumps() == b"\xde\xd6\xcf\x7c"

    assert cs.ileb128(b"\xde\xd6\xcf\x7c").dumps() == b"\xde\xd6\xcf\x7c"


def test_leb128_write_negatives(cs: cstruct) -> None:
    with pytest.raises(ValueError, match="Attempt to encode a negative integer using unsigned LEB128 encoding"):
        cs.uleb128(-2).dumps()
    assert cs.ileb128(-2).dumps() == b"\x7e"


def test_leb128_unsigned_write_amount_written(cs: cstruct) -> None:
    out1 = io.BytesIO()
    bytes_written1 = cs.uleb128(2).write(out1)
    assert bytes_written1 == out1.tell()

    out2 = io.BytesIO()
    bytes_written2 = cs.uleb128(4747).write(out2)
    assert bytes_written2 == out2.tell()

    out3 = io.BytesIO()
    bytes_written3 = cs.uleb128(13371337).write(out3)
    assert bytes_written3 == out3.tell()


def test_leb128_default(cs: cstruct) -> None:
    assert cs.uleb128.__default__() == 0
    assert cs.ileb128.__default__() == 0

    assert cs.uleb128[1].__default__() == [0]
    assert cs.uleb128[None].__default__() == []


def test_leb128_decode_encode_helpers() -> None:
    assert decode_leb128(b"\x8b\x25") == 4747
    assert decode_leb128(b"\x7e", signed=True) == -2
    assert decode_leb128(b"\xf5\x5a", signed=True) == -4747

    assert encode_leb128(4747) == b"\x8b\x25"
    assert encode_leb128(-2, signed=True) == b"\x7e"
    assert encode_leb128(-4747, signed=True) == b"\xf5\x5a"

    with pytest.raises(ValueError, match="Incomplete LEB128 encoding"):
        decode_leb128(b"\x8b")
    with pytest.raises(ValueError, match="Attempt to encode a negative integer"):
        encode_leb128(-1)


def test_leb128_aggregation_by_value(cs: cstruct) -> None:
    """Values aggregate by integer value, regardless of their original encoding."""
    canonical = cs.uleb128(b"\x02")
    overlong = cs.uleb128(b"\x82\x00")

    assert canonical == overlong == 2
    assert canonical.aggregation_key == overlong.aggregation_key == 2
    assert canonical.value == overlong.value == 2


def test_leb128_original_encoding_preserved(cs: cstruct) -> None:
    value = cs.uleb128(b"\xc9\x8f\xb0\x06")

    assert value == 13371337
    assert value.raw == b"\xc9\x8f\xb0\x06"
    assert value.encoding == b"\xc9\x8f\xb0\x06"
    assert value.original_encoding == b"\xc9\x8f\xb0\x06"
    assert value.encoded_length == 4
    assert value.canonical_bytes == b"\xc9\x8f\xb0\x06"
    assert value.is_canonical
    assert value.canonicality == "canonical"
    assert value.redundant_bytes == 0


def test_leb128_does_not_canonicalize_on_read_or_write(cs: cstruct) -> None:
    value = cs.uleb128(b"\x82\x00")

    assert value == 2
    assert value.raw == b"\x82\x00"
    assert not value.is_canonical
    assert value.canonical_bytes == b"\x02"

    # Writing a parsed value writes the original encoding, so existing signatures stay valid.
    assert value.dumps() == b"\x82\x00"


def test_leb128_overlong_unsigned(cs: cstruct) -> None:
    value = cs.uleb128(b"\x80\x00")

    assert value == 0
    assert value.is_overlong
    assert value.overlong
    assert value.canonicality == "overlong"
    assert value.encoded_length == 2
    assert value.canonical_length == 1
    assert value.redundant_bytes == 1


def test_leb128_overlong_signed(cs: cstruct) -> None:
    value = cs.ileb128(b"\xff\x7f")

    assert value == -1
    assert value.signedness == "signed"
    assert value.is_signed
    assert value.raw == b"\xff\x7f"
    assert value.canonical_bytes == b"\x7f"
    assert value.is_overlong


def test_leb128_signedness_is_semantic(cs: cstruct) -> None:
    """The same bytes can be valid evidence for both encodings, but mean different values."""
    unsigned = cs.uleb128(b"\x7f")
    signed = cs.ileb128(b"\x7f")

    assert unsigned == 127
    assert signed == -1
    assert unsigned.signedness == "unsigned"
    assert signed.signedness == "signed"
    assert unsigned.evidence_key != signed.evidence_key


def test_leb128_negative_unsigned_is_not_accepted(cs: cstruct) -> None:
    with pytest.raises(ValueError, match="Attempt to encode a negative integer using unsigned LEB128 encoding"):
        cs.uleb128(-2).dumps()
    assert cs.ileb128(-2).dumps() == b"\x7e"


def test_leb128_values_constructed_from_int_are_canonical(cs: cstruct) -> None:
    value = cs.uleb128(13371337)

    assert value.raw is None
    assert value.encoding == b"\xc9\x8f\xb0\x06"
    assert value.is_canonical
    assert value.dumps() == b"\xc9\x8f\xb0\x06"


def test_leb128_evidence_identity(cs: cstruct) -> None:
    """Equal values with different encodings are not the same evidence."""
    canonical = cs.uleb128(b"\x02")
    overlong = cs.uleb128(b"\x82\x00")

    assert canonical == overlong
    assert canonical.evidence_key != overlong.evidence_key
    assert canonical.evidence_digest != overlong.evidence_digest
    assert not canonical.is_same_evidence(overlong)
    assert canonical.is_same_evidence(cs.uleb128(b"\x02"))


def test_leb128_normalize_returns_copy(cs: cstruct) -> None:
    value = cs.uleb128(b"\x82\x00")
    normalized = value.normalize()

    assert normalized == value == 2
    assert normalized.raw == b"\x02"
    assert normalized.is_canonical
    assert normalized.dumps() == b"\x02"

    # The original value is untouched.
    assert value.raw == b"\x82\x00"
    assert value.dumps() == b"\x82\x00"
    assert value.evidence_key != normalized.evidence_key


def test_leb128_raw_must_match_value(cs: cstruct) -> None:
    with pytest.raises(ValueError, match="Raw LEB128 encoding does not decode"):
        cs.uleb128.__new__(cs.uleb128, 3, raw=b"\x02")


def test_leb128_preview_updates_length_without_writing(cs: cstruct) -> None:
    cdef = """
    struct test {
        uleb128 len;
        uleb128 value;
        char    data[len];
    };
    """
    cs.load(cdef, compiled=False)

    # ``len`` describes the following data, not the LEB128 field itself.
    buf = b"\x02\x82\x00AB"
    obj = cs.test(buf)

    plan = normalize_leb128(obj, "value")

    assert plan.delta == -1
    assert plan.length_delta == -1
    assert plan.normalized_bytes == b"\x02\x02AB"
    assert plan.original_bytes == buf
    assert plan.allowed
    assert not plan.same_evidence
    assert obj.dumps() == buf

    # ``len`` describes ``data``, which does not change size when ``value`` is normalized.
    assert plan.length_updates == ()
    plan.apply()
    assert obj.dumps() == b"\x02\x02AB"


def test_leb128_preview_updates_total_length(cs: cstruct) -> None:
    cdef = """
    struct test {
        uleb128 total;
        uleb128 value;
        char    data[2];
    };
    """
    cs.load(cdef, compiled=False)

    buf = b"\x05\x82\x00AB"
    obj = cs.test(buf)

    plan = normalize_leb128(obj, "value")

    assert plan.normalized_bytes == b"\x04\x02AB"
    assert [(update.field, update.original, update.updated) for update in plan.length_updates] == [("total", 5, 4)]
    assert plan.apply().dumps() == b"\x04\x02AB"
    assert obj.total == 4


def test_leb128_preview_cascading_length_encoding(cs: cstruct) -> None:
    """A total length that crosses an encoding boundary must be re-encoded too."""
    cdef = """
    struct test {
        uleb128 total;
        uleb128 value;
        char    data[124];
    };
    """
    cs.load(cdef, compiled=False)

    # ``total`` is 128 in two bytes; normalizing ``value`` makes the record 126 bytes, so ``total`` fits in one byte.
    buf = b"\x80\x01\x82\x00" + b"A" * 124
    obj = cs.test(buf)

    plan = normalize_leb128(obj, "value")

    assert plan.length_delta == -2
    assert plan.normalized_bytes == b"\x7e\x02" + b"A" * 124
    assert plan.apply().dumps() == plan.normalized_bytes
    assert obj.total == 126


def test_leb128_preview_does_not_rewrite_coincidental_length(cs: cstruct) -> None:
    """Numeric equality alone is not enough to identify a length field."""
    cdef = """
    struct test {
        uleb128 len;
        uleb128 value;
        uint8   crc;
    };
    """
    cs.load(cdef, compiled=False)

    buf = b"\x02\x82\x00\xff"
    obj = cs.test(buf)

    plan = normalize_leb128(obj, "value")

    assert plan.length_updates == ()
    assert plan.normalized_bytes == b"\x02\x02\xff"


def test_leb128_preview_explicit_length_field(cs: cstruct) -> None:
    cdef = """
    struct test {
        uleb128 body_len;
        uleb128 value;
        char    data[2];
    };
    """
    cs.load(cdef, compiled=False)

    buf = b"\x04\x82\x00AB"
    obj = cs.test(buf)

    plan = normalize_leb128(obj, "value", length_fields=["body_len"])

    assert plan.normalized_bytes == b"\x03\x02AB"
    assert [(update.field, update.original, update.updated) for update in plan.length_updates] == [("body_len", 4, 3)]

    with pytest.raises(LEB128NormalizationError, match="do not describe a region"):
        normalize_leb128(obj, "value", length_fields=["crc"])


def test_leb128_preview_rejects_signed_range_change(cs: cstruct) -> None:
    cdef = """
    struct test {
        uleb128 len;
        uleb128 value;
        char    data[len];
    };
    """
    cs.load(cdef, compiled=False)

    buf = b"\x02\x82\x00AB"
    obj = cs.test(buf)

    plan = normalize_leb128(obj, "value", signed_ranges=[SignedRange(0, len(buf), "record")])

    assert not plan.allowed
    assert plan.requires_resign
    assert plan.changed_signed_ranges == (SignedRange(0, len(buf), "record"),)
    assert plan.blocked_reason is not None

    with pytest.raises(SignedRangeError, match="changes signed range"):
        plan.apply()

    # A refused apply does not modify the record.
    assert obj.dumps() == buf

    # The caller can explicitly re-sign by forcing the apply.
    plan.apply(force=True)
    assert obj.dumps() == b"\x02\x02AB"


def test_leb128_preview_signed_range_outside_target_is_allowed(cs: cstruct) -> None:
    cdef = """
    struct test {
        uleb128 len;
        uleb128 value;
        char    data[len];
        uint8   signature;
    };
    """
    cs.load(cdef, compiled=False)

    buf = b"\x02\x82\x00AB\xff"
    obj = cs.test(buf)

    # The length field itself does not change when the LEB128 value is normalized.
    plan = normalize_leb128(obj, "value", signed_ranges=[(0, 1, "length")])

    assert plan.allowed
    plan.apply()
    assert obj.dumps() == b"\x02\x02AB\xff"


def test_leb128_preview_shifted_signed_range_requires_resign(cs: cstruct) -> None:
    """A fixed signature range after a shrinking field is different evidence, even if the moved bytes are identical."""
    cdef = """
    struct test {
        uleb128 len;
        uleb128 value;
        char    data[len];
        uint8   signature;
    };
    """
    cs.load(cdef, compiled=False)

    obj = cs.test(b"\x02\x82\x00AB\xff")
    plan = normalize_leb128(obj, "value", signed_ranges=[(5, 6, "footer")])

    assert not plan.allowed
    assert plan.changed_signed_ranges == (SignedRange(5, 6, "footer"),)


def test_leb128_preview_signed_bytes(cs: cstruct) -> None:
    cdef = """
    struct test {
        uleb128 len;
        uleb128 value;
        char    data[len];
    };
    """
    cs.load(cdef, compiled=False)

    obj = cs.test(b"\x02\x82\x00AB")

    # The signature covers the record with a different, unrelated byte at the position of the target. The preview
    # compares against the supplied signed bytes, so it refuses a change in the signed range.
    plan = normalize_leb128(obj, "value", signed_bytes=b"\x02\xff\x00AB", signed_ranges=[(0, 5)])

    assert not plan.allowed
    assert plan.old_evidence_digest != plan.new_evidence_digest


def test_leb128_preview_unknown_and_non_leb128_fields(cs: cstruct) -> None:
    cdef = """
    struct test {
        uleb128 len;
        uint8   value;
        char    data[len];
    };
    """
    cs.load(cdef, compiled=False)

    obj = cs.test(b"\x01\x02A")

    with pytest.raises(LEB128NormalizationError, match="Unknown field"):
        normalize_leb128(obj, "nope")
    with pytest.raises(LEB128NormalizationError, match="not a LEB128 field"):
        normalize_leb128(obj, "value")


def test_leb128_preview_plan_as_dict(cs: cstruct) -> None:
    cdef = """
    struct test {
        uleb128 total;
        uleb128 value;
        char    data[2];
    };
    """
    cs.load(cdef, compiled=False)

    plan = normalize_leb128(cs.test(b"\x05\x82\x00AB"), "value")
    result = plan.as_dict()

    assert result["field"] == "value"
    assert result["original"] == "8200"
    assert result["normalized"] == "02"
    assert result["delta"] == -1
    assert result["requires_resign"] is False
    assert result["length_updates"][0]["field"] == "total"
    assert result["length_updates"][0]["original_encoding"] == "05"


def test_leb128_plan_apply_rolls_back_on_mismatch(cs: cstruct, monkeypatch: pytest.MonkeyPatch) -> None:
    cdef = """
    struct test {
        uleb128 total;
        uleb128 value;
        char    data[2];
    };
    """
    cs.load(cdef, compiled=False)

    obj = cs.test(b"\x05\x82\x00AB")
    plan = normalize_leb128(obj, "value")

    calls = []
    original_dumps = type(obj).dumps

    def dumps(self: object) -> bytes:
        calls.append(None)
        # The first call verifies that the structure did not change after the preview. The second call is the
        # write-back itself and returns bytes that do not match the preview.
        return b"wrong" if len(calls) > 1 else original_dumps(self)

    monkeypatch.setattr(type(obj), "dumps", dumps, raising=True)
    with pytest.raises(LEB128NormalizationError, match="Write-back for 'value' produced"):
        plan.apply()
    monkeypatch.undo()

    assert obj.dumps() == b"\x05\x82\x00AB"
    assert obj.value.raw == b"\x82\x00"


def test_leb128_plan_apply_rejects_stale_preview(cs: cstruct) -> None:
    cdef = """
    struct test {
        uleb128 total;
        uleb128 value;
        char    data[2];
    };
    """
    cs.load(cdef, compiled=False)

    obj = cs.test(b"\x05\x82\x00AB")
    plan = normalize_leb128(obj, "value")

    obj.data = b"XY"

    with pytest.raises(LEB128NormalizationError, match="changed after the preview"):
        plan.apply()

    # A defeated plan does not modify anything.
    assert obj.dumps() == b"\x05\x82\x00XY"
    assert obj.value.raw == b"\x82\x00"


def test_leb128_plan_apply_noop_for_canonical(cs: cstruct) -> None:
    cdef = """
    struct test {
        uleb128 total;
        uleb128 value;
        char    data[2];
    };
    """
    cs.load(cdef, compiled=False)

    obj = cs.test(b"\x04\x02AB")
    plan = normalize_leb128(obj, "value")

    assert plan.same_evidence
    assert plan.dumps() == obj.dumps()
    assert plan.apply() is obj


def test_leb128_iter_values_and_aggregation(cs: cstruct) -> None:
    cdef = """
    struct entry {
        uleb128 id;
        uleb128 score;
    };
    struct test {
        uleb128 n;
        entry   entries[n];
    };
    """
    cs.load(cdef, compiled=False)

    # n=2, first entry id=1 and score=2 both overlong, second entry id=2 and score=5 canonical.
    buf = b"\x02\x81\x00\x82\x00\x02\x05"
    obj = cs.test(buf)

    assert obj.dumps() == buf
    assert [(path, int(value)) for path, value in iter_leb128_values(obj)] == [
        ("n", 2),
        ("entries[0].id", 1),
        ("entries[0].score", 2),
        ("entries[1].id", 2),
        ("entries[1].score", 5),
    ]

    assert leb128_aggregation(obj) == {
        "n": 2,
        "entries[0].id": 1,
        "entries[0].score": 2,
        "entries[1].id": 2,
        "entries[1].score": 5,
    }

    evidence = leb128_evidence(obj)
    assert evidence["entries[0].id"] != evidence["entries[1].id"]
    assert evidence["entries[0].score"] != evidence["entries[1].id"]
    assert evidence["entries[0].id"] == ("leb128", False, 1, b"\x81\x00")


def test_leb128_iter_values_yields_single_value(cs: cstruct) -> None:
    value = cs.uleb128(b"\x82\x00")
    assert list(iter_leb128_values(value)) == [("", value)]


def test_leb128_preview_rejects_union_and_bit_fields(cs: cstruct) -> None:
    cs.load(
        """
        union test_union {
            uleb128 value;
            uint32  other;
        };
        struct test_bits {
            uint8   a : 4;
            uleb128 value;
        };
        """,
        compiled=False,
    )

    with pytest.raises(LEB128NormalizationError, match="unions"):
        normalize_leb128(cs.test_union(b"\x02\x00\x00\x00"), "value")

    with pytest.raises(LEB128NormalizationError, match="bit fields"):
        normalize_leb128(cs.test_bits(b"\x01\x02"), "value")
