#!/usr/bin/env python
#
#    This file is part of python-registry.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import os

from Registry import Registry
from Registry.install_audit import (CANDIDATE, CONFLICT, NONE, PROVABLE,
                                    RESIDUE, InMemoryHiveView, InstallAuditor,
                                    RuleSet, normalize_guid, parse_version,
                                    RegistryHiveView, version_satisfies)

PRODUCT_GUID = "{11111111-2222-3333-4444-555555555555}"
COMPONENT_GUID = "{aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee}"
UNINSTALL_KEY = ("SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\"
                 + PRODUCT_GUID)
UNINSTALL_ROOT = "SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall"
COMPONENT_ROOT = ("SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Installer\\"
                  "Components")
COMPONENT_KEY = COMPONENT_ROOT + "\\" + COMPONENT_GUID
STALE_PATH_KEY = "SOFTWARE\\ExampleSuite\\Settings"


def make_rule(**overrides):
    rule = {
        "name": "ExampleSuite",
        "product_codes": [PRODUCT_GUID],
        "uninstall_keys": [UNINSTALL_KEY],
        "uninstall_roots": [UNINSTALL_ROOT],
        "component_keys": [COMPONENT_GUID],
        "component_roots": [COMPONENT_ROOT],
        "version_values": ["DisplayVersion"],
        "path_values": ["InstallLocation"],
        "path_keys": [STALE_PATH_KEY],
        "requires": [],
        "excludes": [],
        "versions": [
            {"version": "1.x", "constraint": ">=1.0,<2.0",
             "requires": ["component:" + COMPONENT_GUID],
             "excludes": []},
            {"version": "2.x", "constraint": ">=2.0,<3.0"},
        ],
    }
    rule.update(overrides)
    return {"products": [rule]}


def audit(layout_a, layout_b=None, rule=None):
    rule_set = RuleSet.from_dict(rule or make_rule())
    view_a = InMemoryHiveView(layout_a, "hive_a")
    view_b = InMemoryHiveView(layout_b or {}, "hive_b")
    return InstallAuditor(rule_set, view_a, view_b).run()


def installed_layout(version="1.5"):
    return {
        UNINSTALL_ROOT.lower(): {"timestamp": "2020-01-01 00:00:00", "values": {}},
        UNINSTALL_KEY.lower(): {
            "timestamp": "2020-01-02 00:00:00",
            "values": {
                "DisplayVersion": (1, version),
                "InstallLocation": (1, "C:\\Program Files\\ExampleSuite"),
            },
        },
        COMPONENT_ROOT.lower(): {"timestamp": "2020-01-01 00:00:00", "values": {}},
        COMPONENT_KEY.lower(): {"timestamp": "2020-01-02 00:00:00", "values": {}},
    }


def stale_path_layout():
    return {
        STALE_PATH_KEY.lower(): {
            "timestamp": "2019-01-01 00:00:00",
            "values": {"InstallLocation": (1, "D:\\old\\ExampleSuite")},
        }
    }


def verdict_for(report, hive_label):
    return [v for v in report.verdicts if v.hive_label == hive_label][0]


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_normalize_guid():
    assert normalize_guid("11111111-2222-3333-4444-555555555555") == PRODUCT_GUID.lower()
    assert normalize_guid(PRODUCT_GUID) == PRODUCT_GUID.lower()


def test_parse_version():
    assert parse_version("Version 10.0.19041 beta") == (10, 0, 19041)
    assert parse_version("no digits") is None


def test_version_satisfies():
    assert version_satisfies("1.5", ">=1.0,<2.0")
    assert not version_satisfies("2.0", ">=1.0,<2.0")
    assert version_satisfies("2.0", "=2.0")
    assert version_satisfies("3.1", ">2.0")
    assert not version_satisfies("garbage", ">=1.0")


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def test_provable_when_uninstall_key_product_code_version_and_component_align():
    report = audit(installed_layout("1.5"))
    verdict = verdict_for(report, "hive_a")
    assert verdict.status == PROVABLE
    assert verdict.versions == ["1.5"]
    kinds = {e.kind for e in verdict.evidence}
    assert {"uninstall_key", "product_code", "component_key", "version"} <= kinds
    assert verdict.conflicts == []


def test_stale_path_alone_is_residue_never_an_install():
    report = audit(stale_path_layout())
    verdict = verdict_for(report, "hive_a")
    assert verdict.status == RESIDUE
    path_evidence = [e for e in verdict.evidence if e.kind == "path"]
    assert path_evidence and path_evidence[0].note is not None
    # The two competing interpretations and how to separate them.
    joined = " ".join(verdict.discriminators)
    assert UNINSTALL_KEY in joined
    assert PRODUCT_GUID in joined


def test_product_code_without_uninstall_key_is_only_candidate():
    rule = make_rule(product_code_keys=["SOFTWARE\\ExampleSuite"])
    layout = {
        "SOFTWARE\\ExampleSuite".lower(): {
            "timestamp": "2020-01-01 00:00:00",
            "values": {"ProductCode": (1, PRODUCT_GUID)},
        }
    }
    report = audit(layout, rule=rule)
    verdict = verdict_for(report, "hive_a")
    assert verdict.status == CANDIDATE
    assert any("uninstall key absent" in gap for gap in verdict.missing)
    assert any(UNINSTALL_KEY in d for d in verdict.discriminators)


def test_unsupported_version_is_candidate_with_conflict_basis():
    report = audit(installed_layout("9.9"))
    verdict = verdict_for(report, "hive_a")
    assert verdict.status == CANDIDATE
    assert any("satisfy no supported version rule" in gap
               for gap in verdict.missing)


def test_exclusion_rule_raises_conflict():
    rule = make_rule(excludes=[
        "value:SOFTWARE\\Policies\\ExampleSuite|Blocked=1"])
    layout = installed_layout("1.5")
    layout["SOFTWARE\\Policies\\ExampleSuite".lower()] = {
        "timestamp": "2020-01-03 00:00:00",
        "values": {"Blocked": (4, 1)},
    }
    report = audit(layout, rule=rule)
    verdict = verdict_for(report, "hive_a")
    assert verdict.status == CONFLICT
    assert verdict.conflicts
    assert "exclusion triggered" in verdict.conflicts[0].reason


def test_version_level_exclusion_conflicts():
    rule = make_rule(versions=[
        {"version": "1.x", "constraint": ">=1.0,<2.0",
         "excludes": ["key:SOFTWARE\\ExampleSuite\\V3Only"]},
        {"version": "2.x", "constraint": ">=2.0,<3.0"},
    ])
    layout = installed_layout("1.5")
    layout["SOFTWARE\\ExampleSuite\\V3Only".lower()] = {
        "timestamp": "2020-01-03 00:00:00", "values": {}}
    report = audit(layout, rule=rule)
    verdict = verdict_for(report, "hive_a")
    assert verdict.status == CONFLICT
    assert any("exclusion" in c.reason for c in verdict.conflicts)


def test_missing_component_downgrades_provable_to_candidate():
    layout = installed_layout("1.5")
    del layout[COMPONENT_KEY.lower()]
    report = audit(layout)
    verdict = verdict_for(report, "hive_a")
    assert verdict.status == CANDIDATE
    assert any("component" in fail for fail in verdict.missing)


def test_no_evidence_none_with_acquisition_hint():
    report = audit({})
    verdict = verdict_for(report, "hive_a")
    assert verdict.status == NONE
    assert verdict.discriminators


def test_global_exclusion_reported_per_hive():
    rule = make_rule()
    rule["exclusions"] = [{
        "reason": "locked-down image blocks the product",
        "paths": ["SOFTWARE\\Policies\\ExampleSuite\\Blocked"],
    }]
    layout = installed_layout("1.5")
    layout["SOFTWARE\\Policies\\ExampleSuite\\Blocked".lower()] = {
        "timestamp": "2020-01-03 00:00:00", "values": {}}
    report = audit(layout, rule=rule)
    assert "hive_a" in report.global_conflicts
    assert report.global_conflicts["hive_a"][0].reason == \
        "locked-down image blocks the product"


def test_cross_hive_installed_then_removed_timeline():
    report = audit(installed_layout("1.5"), stale_path_layout())
    assert any("installed then removed" in item for item in report.timeline)
    later = verdict_for(report, "hive_b")
    assert later.status == RESIDUE


def test_report_dict_and_text_include_all_sections():
    report = audit(installed_layout("1.5"), stale_path_layout())
    payload = report.to_dict()
    assert payload["hives"] == ["hive_a", "hive_b"]
    assert len(payload["verdicts"]) == 2
    text = report.to_text()
    for heading in ("可证", "候选", "冲突", "空闲残留", "补证项"):
        assert heading in text


# --------------------------------------------------------------------------- #
# Real hive smoke test: the bundled SYSTEM hives hold no MSI data, so an
# uninstall-based rule must resolve to NONE rather than crashing.
# --------------------------------------------------------------------------- #
def test_real_system_hives_smoke():
    base = os.path.join(os.path.dirname(__file__), "..", "testing",
                        "reg_samples")
    hive_a = Registry.Registry(os.path.join(base, "new_log_1", "SYSTEM"))
    hive_b = Registry.Registry(os.path.join(base, "new_log_2", "SYSTEM"))
    report = InstallAuditor(
        RuleSet.from_dict(make_rule()),
        RegistryHiveView(hive_a, "new_log_1"),
        RegistryHiveView(hive_b, "new_log_2")).run()
    assert all(v.status in (NONE, RESIDUE) for v in report.verdicts)
    assert report.to_text()
