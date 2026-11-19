"""清洗候选影响分析的回归测试。

这些断言直接对应任务要求：
- 一删消三违但保留共同依赖的完整约束视图；
- 必填缺失只能通过真实重验证暴露；
- 组合效应不可由单候选结果简单相加；
- 报告可确定性重建并与 expected_diff.json 字节级一致。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analyzer import (
    build_report,
    canonical_json,
    graph_fingerprint,
    violation_fingerprint,
)

HERE = Path(__file__).parent
SCENARIO = HERE / "scenario"


def _components(result: dict, focus_suffix: str) -> dict[str, dict[str, str]]:
    for entry in result["constraintImpact"].values():
        if entry["focusNode"].endswith(focus_suffix):
            return entry["components"]
    raise AssertionError(f"未找到焦点 {focus_suffix} 的约束影响")


def _component_names(violations: list[dict]) -> set[str]:
    return {v["sourceConstraintComponent"].split("#")[-1] for v in violations}


@pytest.fixture(scope="module")
def report() -> dict:
    return build_report(SCENARIO)


def test_baseline_has_four_violations(report: dict) -> None:
    baseline = report["baseline"]
    assert baseline["conforms"] is False
    assert baseline["violationCount"] == 4
    names = _component_names(baseline["violations"])
    assert names == {
        "MaxCountConstraintComponent",
        "DatatypeConstraintComponent",
        "MinInclusiveConstraintComponent",
        "MinCountConstraintComponent",
    }


def test_age_path_carries_four_shared_constraints(report: dict) -> None:
    age_dep = next(d for d in report["sharedDependencies"] if d["path"].endswith("#age"))
    assert set(age_dep["constraints"]) == {
        "MinCountConstraintComponent",
        "MaxCountConstraintComponent",
        "DatatypeConstraintComponent",
        "MinInclusiveConstraintComponent",
    }


def test_c1_removes_three_violations_without_harming_min_count(report: dict) -> None:
    c1 = next(c for c in report["candidates"] if c["id"] == "C1")
    assert c1["counts"] == {
        "removed": 3,
        "introduced": 0,
        "retained": 1,
        "netViolationChange": -3,
    }
    assert _component_names(c1["removedViolations"]) == {
        "MaxCountConstraintComponent",
        "DatatypeConstraintComponent",
        "MinInclusiveConstraintComponent",
    }
    components = _components(c1, "#alice")
    assert components["MaxCountConstraintComponent"]["baseline"] == "FAIL"
    assert components["MaxCountConstraintComponent"]["after"] == "PASS"
    assert components["DatatypeConstraintComponent"]["after"] == "PASS"
    assert components["MinInclusiveConstraintComponent"]["after"] == "PASS"
    assert components["MinCountConstraintComponent"] == {"baseline": "PASS", "after": "PASS", "parameter": "1"}


def test_c2_only_clears_max_count_and_keeps_value_violations(report: dict) -> None:
    c2 = next(c for c in report["candidates"] if c["id"] == "C2")
    assert c2["counts"]["removed"] == 1
    assert c2["counts"]["introduced"] == 0
    assert _component_names(c2["removedViolations"]) == {"MaxCountConstraintComponent"}
    components = _components(c2, "#alice")
    datatype_state = components["DatatypeConstraintComponent"]
    assert datatype_state["baseline"] == "FAIL"
    assert datatype_state["after"] == "FAIL"
    assert components["MinInclusiveConstraintComponent"]["after"] == "FAIL"
    assert components["MinCountConstraintComponent"]["after"] == "PASS"


def test_c3_introduces_required_min_count_violation(report: dict) -> None:
    c3 = next(c for c in report["candidates"] if c["id"] == "C3")
    assert c3["counts"]["removed"] == 0
    assert c3["counts"]["introduced"] == 1
    assert c3["counts"]["netViolationChange"] == 1
    introduced = c3["introducedViolations"][0]
    assert introduced["focusNode"].endswith("#bob")
    assert introduced["sourceConstraintComponent"].endswith("MinCountConstraintComponent")
    components = _components(c3, "#bob")
    assert components["MinCountConstraintComponent"] == {"baseline": "PASS", "after": "FAIL", "parameter": "1"}


def test_combo_is_not_additive(report: dict) -> None:
    combo = report["combinations"][0]
    assert combo["additive"] is False
    assert len(combo["naiveSum"]["introduced"]) == 0
    actual_introduced = combo["actual"]["introduced"]
    assert len(actual_introduced) == 1
    assert actual_introduced[0]["sourceConstraintComponent"].endswith("MinCountConstraintComponent")
    assert actual_introduced[0]["focusNode"].endswith("#alice")
    missed = combo["nonAdditivity"]["introducedActualButMissedByNaiveSum"]
    assert len(missed) == 1
    assert missed[0]["sourceConstraintComponent"].endswith("MinCountConstraintComponent")
    assert combo["actual"]["removed"] and combo["naiveSum"]["removed"]


def test_unrelated_conformant_node_never_appears(report: dict) -> None:
    all_violations = list(report["baseline"]["violations"])
    for candidate in report["candidates"]:
        for bucket in ("removedViolations", "introducedViolations", "retainedViolations"):
            all_violations.extend(candidate[bucket])
    for combo in report["combinations"]:
        all_violations.extend(combo["actual"]["introduced"])
    assert all("#carol" not in v["focusNode"] for v in all_violations)


def test_report_is_deterministic(report: dict) -> None:
    rebuilt = build_report(SCENARIO)
    assert canonical_json(rebuilt) == canonical_json(report)
    assert graph_fingerprint(rebuilt["baseline"]["violations"]) == report["baseline"]["fingerprint"]


def test_report_rebuilds_expected_diff(report: dict) -> None:
    expected = json.loads((HERE / "expected_diff.json").read_text(encoding="utf-8"))
    assert canonical_json(report) == canonical_json(expected)


def test_violation_fingerprint_ignores_message_text() -> None:
    base = {
        "focusNode": "http://example.org/hr#alice",
        "resultPath": "http://example.org/hr#age",
        "sourceConstraintComponent": "http://www.w3.org/ns/shacl#MaxCountConstraintComponent",
        "sourceShape": "http://example.org/hr#AgeShape",
        "value": "",
        "resultMessage": "消息文案 A",
    }
    reworded = dict(base, resultMessage="消息文案 B")
    assert violation_fingerprint(base) == violation_fingerprint(reworded)
