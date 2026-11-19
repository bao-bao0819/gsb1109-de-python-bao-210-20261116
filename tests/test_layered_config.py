#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import json

import pytest

from Registry.LayeredConfig import (
    DELETE,
    EMPTY,
    DeleteTree,
    LayeredConfig,
    State,
    compute_effective_config,
)


def test_system_value_is_effective_with_full_chain():
    config = LayeredConfig()
    config.set("system", "app.theme", "dark", "REG_SZ", origin="SYSTEM\\app")

    item = config.effective()["app.theme"]

    assert item.state is State.PRESENT
    assert item.value == "dark"
    assert item.value_type == "REG_SZ"
    assert [source.layer for source in item.source_chain] == ["system"]
    assert item.source_chain[0].outcome == "winner"
    assert item.source_chain[0].origin == "SYSTEM\\app"


def test_user_overrides_system_and_chain_marks_masked_source():
    config = LayeredConfig()
    config.set("system", "app.theme", "dark", "REG_SZ")
    config.set("user", "app.theme", "light", "REG_SZ")

    item = config.effective()["app.theme"]

    assert item.value == "light"
    assert item.layers == ("user", "system")
    assert item.source_chain[0].outcome == "winner"
    assert item.source_chain[1].outcome == "masked"
    assert item.source_chain[1].value == "dark"


def test_policy_overrides_user_and_system():
    config = LayeredConfig.from_mappings(
        system={"app.theme": ("dark", "REG_SZ")},
        user={"app.theme": ("light", "REG_SZ")},
        policy={"app.theme": ("high-contrast", "REG_SZ")},
    )

    item = config.effective()["app.theme"]

    assert item.value == "high-contrast"
    assert item.layers == ("policy", "user", "system")
    assert [source.outcome for source in item.source_chain] == [
        "winner",
        "masked",
        "masked",
    ]


def test_delete_marker_masks_all_lower_layers():
    config = LayeredConfig()
    config.set("system", "feature.enabled", True, "REG_DWORD")
    config.set("user", "feature.enabled", True, "REG_DWORD")
    config.delete("policy", "feature.enabled", origin="GPO")

    item = config.effective()["feature.enabled"]

    assert item.state is State.DELETED
    assert item.value is None
    assert item.value_type is None
    assert item.is_deleted
    assert not item.is_present
    assert item.layers == ("policy", "user", "system")
    assert item.source_chain[0].outcome == "tombstone"
    assert item.source_chain[0].origin == "GPO"
    assert item.source_chain[1].outcome == "masked"
    assert item.source_chain[2].outcome == "masked"


def test_policy_withdrawal_reveals_user_value_without_stale_overlay():
    config = LayeredConfig()
    config.set("system", "app.theme", "dark", "REG_SZ")
    config.set("user", "app.theme", "light", "REG_SZ")
    config.delete("policy", "app.theme")

    before = config.effective()
    assert before["app.theme"].state is State.DELETED
    assert before["app.theme"].value is None

    config.withdraw("policy")
    after = config.effective()

    assert after["app.theme"].state is State.PRESENT
    assert after["app.theme"].value == "light"
    assert after["app.theme"].layers == ("user", "system")
    # The old snapshot is a snapshot: it must not change retroactively.
    assert before["app.theme"].state is State.DELETED
    assert before["app.theme"].value is None


def test_replacing_policy_recomputes_from_scratch():
    config = LayeredConfig.from_mappings(
        system={"app.mode": ("default", "REG_SZ")},
        user={"app.mode": ("user", "REG_SZ")},
        policy={"app.mode": DELETE},
    )
    assert config.effective()["app.mode"].state is State.DELETED

    config.set_layer("policy", {"app.mode": "enforced"})
    assert config.effective()["app.mode"].value == "enforced"

    config.withdraw("policy")
    assert config.effective()["app.mode"].value == "user"

    config.withdraw("user")
    assert config.effective()["app.mode"].value == "default"


def test_absent_empty_and_deleted_are_distinct():
    config = LayeredConfig()
    config.set("system", "empty.string", "", "REG_SZ")
    config.set("system", "empty.binary", b"", "REG_BINARY")
    config.set("system", "empty.list", [], "REG_MULTI_SZ")

    effective = config.effective()

    absent = effective.resolve("never.set")
    assert absent.state is State.ABSENT
    assert absent.value is None
    assert absent.source_chain == ()

    for name in ("empty.string", "empty.binary", "empty.list"):
        item = effective[name]
        assert item.state is State.EMPTY
        assert item.is_empty
        assert item.is_present
        assert item.winner.kind.value == "set"


def test_explicit_empty_masks_lower_value():
    config = LayeredConfig()
    config.set("system", "app.theme", "dark", "REG_SZ")
    config.set("policy", "app.theme", "", "REG_SZ")

    item = config.effective()["app.theme"]

    assert item.state is State.EMPTY
    assert item.value == ""
    assert item.layers == ("policy", "system")
    assert item.source_chain[0].outcome == "winner"
    assert item.source_chain[1].outcome == "masked"


def test_empty_marker_is_typed_empty():
    config = LayeredConfig()
    config.set("user", "app.theme", EMPTY)

    item = config.effective()["app.theme"]

    assert item.state is State.EMPTY
    assert item.value is None
    assert item.value_type == "none"


def test_delete_marker_without_lower_value_is_still_deleted_not_absent():
    config = LayeredConfig()
    config.delete("policy", "app.theme")

    item = config.effective()["app.theme"]

    assert item.state is State.DELETED
    assert item.layers == ("policy",)
    assert config.effective().state("other") is State.ABSENT


def test_same_name_different_types_is_listed_as_conflict():
    config = LayeredConfig()
    config.set("system", "app.theme", "dark", "REG_SZ")
    config.set("policy", "app.theme", 1, "REG_DWORD")

    effective = config.effective()

    assert effective["app.theme"].value == 1
    assert effective["app.theme"].value_type == "REG_DWORD"
    assert len(effective.conflicts) == 1
    conflict = effective.conflicts[0]
    assert conflict.name == "app.theme"
    assert set(conflict.types) == {"REG_SZ", "REG_DWORD"}
    assert conflict.layers == ("policy", "system")


def test_same_type_aliases_are_not_a_conflict():
    config = LayeredConfig()
    config.set("system", "app.theme", "dark", "string")
    config.set("user", "app.theme", "light", "REG_SZ")

    effective = config.effective()

    assert effective["app.theme"].value == "light"
    assert effective.conflicts == ()


def test_inferred_types_participate_in_conflicts():
    config = LayeredConfig.from_mappings(
        system={"app.count": 1},
        user={"app.count": "one"},
    )

    conflict = config.effective().conflicts[0]

    assert conflict.name == "app.count"
    assert set(conflict.types) == {"integer", "string"}


def test_conflicts_are_only_reported_for_set_values():
    config = LayeredConfig()
    config.set("system", "app.theme", "dark", "REG_SZ")
    config.delete("policy", "app.theme")

    effective = config.effective()

    assert effective["app.theme"].state is State.DELETED
    assert effective.conflicts == ()


def test_within_layer_last_write_wins():
    config = LayeredConfig()
    config.set("system", "app.theme", "first", "REG_SZ")
    config.set("system", "app.theme", "second", "REG_SZ")

    item = config.effective()["app.theme"]

    assert item.value == "second"
    assert item.layers == ("system", "system")
    assert [source.outcome for source in item.source_chain] == ["winner", "masked"]


def test_set_after_delete_in_same_layer_reveals_value_again():
    config = LayeredConfig()
    config.set("system", "app.theme", "dark", "REG_SZ")
    config.delete("policy", "app.theme")
    config.set("policy", "app.theme", "enforced", "REG_SZ")

    item = config.effective()["app.theme"]

    assert item.state is State.PRESENT
    assert item.value == "enforced"
    assert item.layers == ("policy", "policy", "system")
    assert [source.kind.value for source in item.source_chain] == [
        "set",
        "delete",
        "set",
    ]
    assert [source.outcome for source in item.source_chain] == [
        "winner",
        "masked",
        "masked",
    ]


def test_subtree_delete_masks_nested_names():
    config = LayeredConfig()
    config.set("system", "app.theme", "dark", "REG_SZ")
    config.set("system", "app.locale", "en-US", "REG_SZ")
    config.set("system", "other.theme", "light", "REG_SZ")
    config.delete_tree("policy", "app")

    effective = config.effective()

    assert effective["app.theme"].state is State.DELETED
    assert effective["app.locale"].state is State.DELETED
    assert effective["other.theme"].state is State.PRESENT
    assert effective["app.theme"].source_chain[0].scope == "subtree"


def test_subtree_delete_with_no_lower_values_still_reports_deleted():
    config = LayeredConfig()
    config.delete_tree("policy", "app")

    item = config.effective()["app"]

    assert item.state is State.DELETED
    assert item.source_chain[0].scope == "subtree"
    assert config.effective().state("other") is State.ABSENT


def test_subtree_delete_can_be_overridden_by_higher_layer():
    config = LayeredConfig()
    config.set("system", "app.theme", "dark", "REG_SZ")
    config.delete_tree("user", "app")
    config.set("policy", "app.theme", "enforced", "REG_SZ")

    effective = config.effective()

    assert effective["app.theme"].state is State.PRESENT
    assert effective["app.theme"].value == "enforced"
    assert [source.kind.value for source in effective["app.theme"].source_chain] == [
        "set",
        "delete_tree",
        "set",
    ]


def test_names_are_case_insensitive_by_default():
    config = LayeredConfig()
    config.set("system", "App.Theme", "dark", "REG_SZ")
    config.set("user", "app.theme", "light", "REG_SZ")

    effective = config.effective()

    assert len(effective) == 1
    assert effective["APP.THEME"].value == "light"


def test_case_sensitive_mode_keeps_names_distinct():
    config = LayeredConfig(case_sensitive=True)
    config.set("system", "App.Theme", "dark", "REG_SZ")
    config.set("system", "app.theme", "light", "REG_SZ")

    effective = config.effective()

    assert len(effective) == 2
    assert effective["App.Theme"].value == "dark"
    assert effective["app.theme"].value == "light"


def test_custom_priority_order():
    config = LayeredConfig(order=("system", "policy", "user"))
    config.set("system", "app.theme", "dark", "REG_SZ")
    config.set("policy", "app.theme", "enforced", "REG_SZ")
    config.set("user", "app.theme", "light", "REG_SZ")

    item = config.effective()["app.theme"]

    assert item.value == "light"
    assert item.layers == ("user", "policy", "system")


def test_unknown_layer_is_rejected():
    config = LayeredConfig()
    with pytest.raises(ValueError):
        config.set("machine", "app.theme", "dark")


def test_duplicate_layers_are_rejected():
    with pytest.raises(ValueError):
        LayeredConfig(order=("system", "user", "system"))


def test_set_many_accepts_mapping_and_delete_markers():
    config = LayeredConfig()
    config.set_many("system", {"a": ("1", "REG_SZ"), "b": ("2", "REG_SZ")})
    config.set_many("policy", {"a": DELETE})

    effective = config.effective()

    assert effective["a"].state is State.DELETED
    assert effective["b"].value == "2"


def test_compute_effective_config_convenience_function():
    effective = compute_effective_config(
        system={"app.theme": "dark"},
        policy={"app.theme": "enforced"},
    )

    assert effective["app.theme"].value == "enforced"
    assert effective["app.theme"].layers == ("policy", "system")


def test_withdrawing_single_policy_item_reveals_lower_value():
    config = LayeredConfig()
    config.set("system", "app.theme", "dark", "REG_SZ")
    config.set("user", "app.theme", "light", "REG_SZ")
    config.set("policy", "app.theme", "enforced", "REG_SZ")
    config.set("policy", "app.locale", "de-DE", "REG_SZ")

    assert config.effective()["app.theme"].value == "enforced"
    assert config.remove("policy", "app.theme") == 1

    effective = config.effective()
    assert effective["app.theme"].value == "light"
    assert effective["app.locale"].value == "de-DE"
    assert effective["app.theme"].layers == ("user", "system")


def test_remove_differs_from_delete():
    config = LayeredConfig()
    config.set("system", "app.theme", "dark", "REG_SZ")
    config.set("user", "app.theme", "light", "REG_SZ")

    config.delete("policy", "app.theme")
    assert config.effective()["app.theme"].state is State.DELETED

    config.remove("policy", "app.theme")
    assert config.effective()["app.theme"].state is State.PRESENT
    assert config.effective()["app.theme"].value == "light"


def test_same_layer_different_types_is_a_conflict():
    config = LayeredConfig()
    config.set("system", "app.theme", "dark", "REG_SZ")
    config.set("system", "app.theme", b"\x01", "REG_BINARY")

    effective = config.effective()

    assert effective["app.theme"].value == b"\x01"
    assert effective["app.theme"].value_type == "REG_BINARY"
    assert set(effective.conflicts[0].types) == {"REG_BINARY", "REG_SZ"}


def test_conflict_entries_are_ordered_from_high_to_low_priority():
    config = LayeredConfig()
    config.set("system", "app.theme", "dark", "REG_SZ")
    config.set("user", "app.theme", 2, "REG_DWORD")
    config.set("policy", "app.theme", ["a"], "REG_MULTI_SZ")

    conflict = config.effective().conflicts[0]

    assert conflict.layers == ("policy", "user", "system")
    assert conflict.types == ("REG_MULTI_SZ", "REG_DWORD", "REG_SZ")


def test_case_insensitive_delete_matches_different_case():
    config = LayeredConfig()
    config.set("system", "App.Theme", "dark", "REG_SZ")
    config.delete("policy", "app.theme")

    assert config.effective()["APP.THEME"].state is State.DELETED


def test_typed_int_constants_are_recognized():
    config = LayeredConfig()
    config.set("system", "app.count", 1, 0x0004)
    config.set("user", "app.count", 2, 0x0004)

    effective = config.effective()

    assert effective["app.count"].value == 2
    assert effective["app.count"].value_type == "reg_dword"
    assert effective.conflicts == ()


def test_untyped_values_do_not_create_false_conflicts():
    config = LayeredConfig()
    config.set("system", "empty.value", "")
    config.delete("policy", "empty.value")

    assert config.effective().conflicts == ()


def test_to_dict_is_json_serializable_and_contains_sources():
    config = LayeredConfig()
    config.set("system", "app.data", b"\x00\x01", "REG_BINARY")
    config.set("policy", "app.data", "override", "REG_SZ")

    payload = config.to_dict()

    assert payload["order"] == ["system", "user", "policy"]
    item = payload["effective"]["app.data"]
    assert item["state"] == "present"
    assert item["value"] == "override"
    assert [source["outcome"] for source in item["sources"]] == ["winner", "masked"]
    assert [source["value"] for source in item["sources"]] == ["override", "0001"]
    assert payload["conflicts"][0]["types"] == ["REG_SZ", "REG_BINARY"]
    json.dumps(payload)


def test_compatibility_module_exports_same_resolver():
    from Registry.Config import Config, resolve_effective_config

    assert Config is LayeredConfig
    effective = resolve_effective_config(policy={"app.theme": "enforced"})
    assert effective["app.theme"].value == "enforced"


def test_snapshot_revision_matches_raw_entries():
    config = LayeredConfig()
    assert config.effective().revision == 0

    config.set("system", "a", "1")
    assert config.effective().revision == 1

    config.delete("policy", "a")
    assert config.effective().revision == 2

    config.withdraw("policy")
    assert config.effective().revision == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
