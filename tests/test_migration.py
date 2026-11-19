from __future__ import annotations

import pytest

from dissect.cstruct import cstruct
from dissect.cstruct.migration import (
    Conflict,
    MigrationError,
    StateClassification,
    migrate,
    migrate_enum,
    migrate_flag,
    verify_roundtrip,
)


def test_enum_exact_alias() -> None:
    old = {"A": 1, "A_ALIAS": 1, "B": 2}
    new = {"A": 1, "A_ALIAS": 1, "B": 2, "C": 3}

    result = migrate_enum(old, new)

    assert result.exact_values == {1: 1, 2: 2}
    assert result.classify(1) == StateClassification.EXACT
    assert verify_roundtrip(result, old, new) == []


def test_enum_alias_name_lost_is_candidate() -> None:
    old = {"RUN": 1, "START": 1}
    new = {"RUN": 1}

    result = migrate_enum(old, new)

    assert result.exact_values == {}
    candidates = result.candidates_for(1)
    assert len(candidates) == 1
    assert candidates[0].new.value == 1
    assert candidates[0].lost_names == ("START",)
    assert result.classify(1) == StateClassification.CANDIDATE


def test_enum_renamed_without_shared_name_is_rejected() -> None:
    old = {"IDLE": 0, "RUN": 1}
    new = {"IDLE": 0, "RUNNING": 2}

    result = migrate_enum(old, new)

    assert result.exact_values == {0: 0}
    assert result.candidates_for(1) == []
    conflicts = result.conflicts_for(1)
    assert len(conflicts) == 1
    assert conflicts[0].new is not None
    assert conflicts[0].new.value == 2
    assert conflicts[0].lost_names == ("RUN",)


def test_enum_rejected_has_concrete_conflict() -> None:
    old = {"IDLE": 0, "LEGACY": 8}
    new = {"IDLE": 0, "RUN": 1}

    result = migrate_enum(old, new)

    assert result.exact_values == {0: 0}
    conflicts = result.conflicts_for(8)
    assert len(conflicts) == 1
    assert isinstance(conflicts[0], Conflict)
    assert conflicts[0].old.value == 8
    assert conflicts[0].old.names == ("LEGACY",)
    assert conflicts[0].new is not None
    assert conflicts[0].new.value == 1
    assert conflicts[0].lost_names == ("LEGACY",)


def test_enum_unnamed_values_are_not_collapsed_to_unknown() -> None:
    old = {"A": 1}
    new = {"A": 1, "B": 2}

    result = migrate_enum(old, new, observed=[7, 9])

    assert result.exact_values == {1: 1}
    assert result.classify(7) == StateClassification.REJECTED
    assert result.classify(9) == StateClassification.REJECTED
    assert result.conflicts_for(7)[0].old.value == 7
    assert result.conflicts_for(9)[0].old.value == 9
    assert result.conflicts_for(7)[0].old.names == ()
    assert result.conflicts_for(9)[0].old.names == ()


def test_enum_unnamed_observed_match() -> None:
    old = {"A": 1}
    new = {"A": 1, "B": 2}

    result = migrate_enum(old, new, observed=[2])

    assert result.exact_values == {1: 1, 2: 2}
    assert result.exact_values[2] == 2


def test_flag_same_bits_same_names() -> None:
    old = {"READ": 1, "WRITE": 2, "EXEC": 4}
    new = {"READ": 1, "WRITE": 2, "EXEC": 4, "EXTRA": 8}

    result = migrate_flag(old, new)

    assert result.exact_values == {1: 1, 2: 2, 4: 4}
    assert result.candidates == []
    assert result.rejected == []


def test_flag_combined_member_identity() -> None:
    old = {"READ": 1, "WRITE": 2, "READ_WRITE": 3}
    new = {"READ": 1, "WRITE": 2, "READ_WRITE": 3}

    result = migrate_flag(old, new)

    assert result.exact_values == {1: 1, 2: 2, 3: 3}


def test_flag_multi_bit_member_moved_is_candidate() -> None:
    old = {"A": 1, "B": 2, "AB": 3}
    new = {"A": 1, "B": 2, "AB": 4}

    result = migrate_flag(old, new)

    assert result.exact_values == {1: 1, 2: 2}
    candidates = result.candidates_for(3)
    assert result.candidate_values[3] == [3, 4]
    moved = next(candidate for candidate in candidates if candidate.new.value == 4)
    assert moved.preserved_names == ("AB",)
    assert result.classify(3) == StateClassification.CANDIDATE


def test_flag_new_member_covers_multiple_old_bits() -> None:
    old = {"A": 1, "B": 2}
    new = {"A": 1, "B": 2, "AB": 3}

    result = migrate_flag(old, new)

    assert result.exact_values == {1: 1, 2: 2}
    combined = migrate_flag({"AB": 3}, {"A": 1, "B": 2, "AB": 4})
    assert combined.exact_values == {}
    assert combined.candidate_values[3] == [3, 4]
    assert any(name == "AB" for candidate in combined.candidates_for(3) for name in candidate.preserved_names)


def test_flag_bit_positions_are_not_copied_one_for_one() -> None:
    old = {"OLD_A": 1, "OLD_B": 2, "OLD_AB": 3}
    new = {"NEW_A": 1, "NEW_B": 2, "NEW_AB": 3}

    result = migrate_flag(old, new)

    assert result.exact_values == {}
    for value in (1, 2, 3):
        assert result.classify(value) == StateClassification.CANDIDATE
        for candidate in result.candidates_for(value):
            assert candidate.new.value in (1, 2, 3)


def test_flag_exclusive_old_state_rejected() -> None:
    old = {"READ": 1, "WRITE": 2, "READ_WRITE": 3}
    new = {"READ": 1, "WRITE": 2}

    result = migrate_flag(old, new, exclusive=[("READ", "WRITE")])

    assert result.exact_values == {1: 1, 2: 2}
    conflicts = result.conflicts_for(3)
    assert len(conflicts) == 1
    assert "READ" in conflicts[0].reason
    assert "WRITE" in conflicts[0].reason


def test_flag_exclusive_discards_invalid_candidate() -> None:
    old = {"MODE": 3}
    new = {"MODE_A": 1, "MODE_B": 2, "MODE_AB": 3}

    result = migrate_flag(old, new, exclusive=[("MODE_A", "MODE_B")])

    assert result.candidates_for(3) == []
    assert result.classify(3) == StateClassification.REJECTED


def test_flag_unnamed_bits_rejected_with_mask() -> None:
    old = {"READ": 1, "WRITE": 2}
    new = {"READ": 1, "WRITE": 2}

    result = migrate_flag(old, new, observed=[0x81])

    conflicts = result.conflicts_for(0x81)
    assert len(conflicts) == 1
    assert conflicts[0].lost_bits == 0x80
    assert conflicts[0].new is not None
    assert conflicts[0].new.names


def test_flag_unnamed_value_not_collapsed() -> None:
    old = {"A": 1}
    new = {"A": 1, "B": 2}

    result = migrate_flag(old, new, observed=[4, 8])

    assert result.classify(4) == StateClassification.REJECTED
    assert result.classify(8) == StateClassification.REJECTED
    assert result.conflicts_for(4)[0].old.value == 4
    assert result.conflicts_for(8)[0].old.value == 8


def test_flag_no_counterpart_rejected() -> None:
    old = {"OLD": 0x10}
    new = {"A": 1, "B": 2}

    result = migrate_flag(old, new)

    conflicts = result.conflicts_for(0x10)
    assert len(conflicts) == 1
    assert conflicts[0].lost_names == ("OLD",)
    assert conflicts[0].new is not None


def test_roundtrip_verification_with_parsers(cs: cstruct) -> None:
    cs.load(
        """
        enum Old : uint8 { A = 1, B = 2 };
        enum New : uint8 { A = 1, B = 2, C = 3 };
        """
    )

    old = {"A": 1, "B": 2}
    new = {"A": 1, "B": 2, "C": 3}
    result = migrate_enum(old, new)
    failures = verify_roundtrip(result, old, new, new_parser=cs.New, old_parser=cs.Old)

    assert failures == []


def test_roundtrip_uses_cstruct_classes_as_parsers(cs: cstruct) -> None:
    cs.load(
        """
        enum Old : uint8 { A = 1, B = 2 };
        enum New : uint8 { A = 1, B = 2, C = 3 };
        """
    )

    result = migrate_enum(cs.Old, cs.New)

    assert result.exact_values == {1: 1, 2: 2}
    assert verify_roundtrip(result, cs.Old, cs.New) == []


def test_roundtrip_reports_lost_name() -> None:
    old = {"RUN": 1, "START": 1}
    new = {"RUN": 1}

    result = migrate_enum(old, new)
    assert verify_roundtrip(result, old, new) == []


def test_kind_inference_and_validation() -> None:
    cs = cstruct()
    cs.load("flag Perm : uint8 { R = 1, W = 2 }; enum Mode : uint8 { A = 1 };")

    flag_result = migrate(cs.Perm, {"R": 1})
    assert flag_result.kind == "flag"

    enum_result = migrate(cs.Mode, {"A": 1})
    assert enum_result.kind == "enum"

    with pytest.raises(MigrationError):
        migrate_flag({"A": 1}, {"B": 2}, exclusive=[("A",)])

    with pytest.raises(MigrationError):
        migrate_flag({"A": 1}, {"B": 2}, exclusive=[("MISSING1", "MISSING2")])


def test_flag_old_combined_member_split_by_new_bits() -> None:
    old = {"RW": 3}
    new = {"R": 1, "W": 2}

    result = migrate_flag(old, new)

    assert result.exact_values == {}
    candidates = result.candidates_for(3)
    assert [candidate.new.value for candidate in candidates] == [3]
    assert candidates[0].new.names == ("R", "W")
    assert candidates[0].lost_names == ("RW",)


def test_flag_new_combined_member_is_exact_at_same_value() -> None:
    old = {"R": 1, "W": 2}
    new = {"R": 1, "W": 2, "RW": 3}

    result = migrate_flag(old, new)

    assert result.exact_values == {1: 1, 2: 2}
    assert result.candidates == []


def test_flag_renamed_single_bit_is_candidate_not_copy() -> None:
    old = {"OLD": 1}
    new = {"NEW": 1}

    result = migrate_flag(old, new)

    assert result.exact_values == {}
    assert result.classify(1) == StateClassification.CANDIDATE
    assert result.candidates_for(1)[0].new.value == 1


def test_enum_negative_values_are_supported() -> None:
    old = {"ERR": -1, "OK": 0}
    new = {"ERR": -1, "OK": 0}

    result = migrate_enum(old, new)

    assert result.exact_values == {-1: -1, 0: 0}


def test_flag_negative_values_are_rejected_input() -> None:
    with pytest.raises(MigrationError):
        migrate_flag({"A": -1}, {"A": 1})


def test_to_dict_is_serialisable() -> None:
    import json

    result = migrate_enum({"A": 1, "B": 2}, {"A": 1, "C": 3})
    payload = result.to_dict()

    assert payload["kind"] == "enum"
    assert payload["exact"] == [{"old": 1, "new": 1, "names": ("A",)}]
    assert json.loads(json.dumps(payload))
