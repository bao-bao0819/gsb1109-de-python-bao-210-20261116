"""清洗候选影响分析器。

方法（对应任务要求）：
1. 对原始数据图做一次基线 SHACL 验证，得到结构化违规集合与确定性指纹。
2. 每个候选都在数据图的全新副本上真实执行删除后重新验证——
   绝不从"哪些违规消失了"做静态推断。
3. 差异按基线 vs 重验证的违规指纹集合计算，分为消失/新增/保留三类，
   并列出候选路径上挂载的全部相关约束（含验证通过的约束）的前后状态，
   以暴露共同依赖。
4. 候选组合同样整体执行、整体重验证，再与单候选结果的朴素集合求和对比，
   量化"不可简单相加"。
5. 报告带确定性指纹，可用 --check 重建同一份预期验证差异。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import SH

from pyshacl import validate

SCENARIO_DIR = Path(__file__).parent / "scenario"


def load_prefixes(config: dict[str, Any]) -> dict[str, str]:
    return config["prefixes"]


def resolve_curie(token: str, prefixes: dict[str, str]) -> str:
    if token.startswith("http://") or token.startswith("https://"):
        return token
    prefix, _, local = token.partition(":")
    if prefix not in prefixes:
        raise ValueError(f"未知前缀: {token!r}")
    return prefixes[prefix] + local


def build_object(spec: dict[str, Any], prefixes: dict[str, str]):
    if spec.get("type") == "uri":
        return URIRef(resolve_curie(spec["value"], prefixes))
    value = spec["value"]
    datatype = spec.get("datatype")
    if datatype:
        return Literal(value, datatype=URIRef(resolve_curie(datatype, prefixes)))
    return Literal(value)


def build_triple(spec: dict[str, Any], prefixes: dict[str, str]):
    return (
        URIRef(resolve_curie(spec["s"], prefixes)),
        URIRef(resolve_curie(spec["p"], prefixes)),
        build_object(spec["o"], prefixes),
    )


def term_key(term) -> str:
    """结果项中 RDF 术语的稳定字符串表示。"""
    if term is None:
        return ""
    return str(term)


def extract_violations(results_graph: Graph) -> list[dict[str, str]]:
    """从验证结果图提取结构化违规，按指纹排序保证确定性。"""
    violations: list[dict[str, str]] = []
    for result in results_graph.objects(None, SH.result):
        focus = results_graph.value(result, SH.focusNode)
        path = results_graph.value(result, SH.resultPath)
        component = results_graph.value(result, SH.sourceConstraintComponent)
        shape = results_graph.value(result, SH.sourceShape)
        value = results_graph.value(result, SH.value)
        message = results_graph.value(result, SH.resultMessage)
        violations.append(
            {
                "focusNode": term_key(focus),
                "resultPath": term_key(path),
                "sourceConstraintComponent": term_key(component),
                "sourceShape": term_key(shape),
                "value": term_key(value),
                "resultMessage": term_key(message),
            }
        )
    violations.sort(key=lambda v: violation_fingerprint(v))
    return violations


def violation_fingerprint(v: dict[str, str]) -> str:
    """指纹不含 resultMessage：消息文案变化不应改变违规身份。"""
    material = "|".join(
        [
            v["focusNode"],
            v["resultPath"],
            v["sourceConstraintComponent"],
            v["sourceShape"],
            v["value"],
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def fingerprint_set(violations: list[dict[str, str]]) -> set[str]:
    return {violation_fingerprint(v) for v in violations}


def graph_fingerprint(violations: list[dict[str, str]]) -> str:
    joined = "\n".join(sorted(violation_fingerprint(v) for v in violations))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def run_validation(data_graph: Graph, shapes_graph: Graph) -> tuple[bool, list[dict[str, str]]]:
    conforms, results_graph, _ = validate(
        data_graph,
        shacl_graph=shapes_graph,
        inference="none",
        debug=False,
    )
    return conforms, extract_violations(results_graph)


def load_graph(path: Path) -> Graph:
    graph = Graph()
    graph.parse(path, format="turtle")
    return graph


def relevant_constraints(shapes_graph: Graph) -> dict[str, dict[str, Any]]:
    """读取每个 PropertyShape 挂载的完整相关约束（含参数），按路径索引。"""
    component_predicates = {
        SH.minCount: "MinCountConstraintComponent",
        SH.maxCount: "MaxCountConstraintComponent",
        SH.datatype: "DatatypeConstraintComponent",
        SH.minInclusive: "MinInclusiveConstraintComponent",
        SH.maxInclusive: "MaxInclusiveConstraintComponent",
        SH.minExclusive: "MinExclusiveConstraintComponent",
        SH.maxExclusive: "MaxExclusiveConstraintComponent",
        SH["class"]: "ClassConstraintComponent",
    }
    indexed: dict[str, dict[str, Any]] = {}
    for shape in shapes_graph.subjects(SH.path, None):
        path = shapes_graph.value(shape, SH.path)
        entry = {"propertyShape": str(shape), "path": str(path), "constraints": {}}
        for predicate, component in component_predicates.items():
            param = shapes_graph.value(shape, predicate)
            if param is not None:
                entry["constraints"][component] = str(param)
        indexed[str(path)] = entry
    return indexed


def constraint_status(
    violations: list[dict[str, str]], focus: str, path: str, component: str
) -> str:
    return "FAIL" if any(
        v["focusNode"] == focus
        and v["resultPath"] == path
        and v["sourceConstraintComponent"].endswith(component)
        for v in violations
    ) else "PASS"


def analyze_candidate(
    candidate: dict[str, Any],
    data_text: str,
    shapes_graph: Graph,
    prefixes: dict[str, str],
    baseline: list[dict[str, str]],
    constraints_by_path: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    graph = Graph()
    graph.parse(data=data_text, format="turtle")
    deleted = []
    for spec in candidate["delete"]:
        triple = build_triple(spec, prefixes)
        before = len(graph)
        graph.remove(triple)
        if len(graph) == before:
            raise ValueError(f"候选 {candidate['id']} 试图删除不存在的三元组: {triple}")
        deleted.append([str(t) for t in triple])

    conforms, after = run_validation(graph, shapes_graph)

    base_fp = fingerprint_set(baseline)
    after_fp = fingerprint_set(after)
    after_by_fp = {violation_fingerprint(v): v for v in after}
    base_by_fp = {violation_fingerprint(v): v for v in baseline}

    removed = [base_by_fp[fp] for fp in sorted(base_fp - after_fp)]
    introduced = [after_by_fp[fp] for fp in sorted(after_fp - base_fp)]
    retained = [after_by_fp[fp] for fp in sorted(base_fp & after_fp)]

    touched = sorted(
        {
            (str(build_triple(spec, prefixes)[0]), str(build_triple(spec, prefixes)[1]))
            for spec in candidate["delete"]
        }
    )
    constraint_impact = {}
    for focus, path in touched:
        info = constraints_by_path.get(path)
        if not info:
            continue
        constraint_impact[f"{focus} {path}"] = {
            "focusNode": focus,
            "path": path,
            "propertyShape": info["propertyShape"],
            "components": {
                component: {
                    "parameter": param,
                    "baseline": constraint_status(baseline, focus, path, component),
                    "after": constraint_status(after, focus, path, component),
                }
                for component, param in sorted(info["constraints"].items())
            },
        }

    return {
        "id": candidate["id"],
        "description": candidate["description"],
        "deletedTriples": deleted,
        "conforms": conforms,
        "fingerprint": graph_fingerprint(after),
        "removedViolations": removed,
        "introducedViolations": introduced,
        "retainedViolations": retained,
        "constraintImpact": constraint_impact,
        "counts": {
            "removed": len(removed),
            "introduced": len(introduced),
            "retained": len(retained),
            "netViolationChange": len(introduced) - len(removed),
        },
    }


def analyze_combination(
    combo: dict[str, Any],
    candidate_results: dict[str, dict[str, Any]],
    data_text: str,
    shapes_graph: Graph,
    prefixes: dict[str, str],
    baseline: list[dict[str, str]],
    constraints_by_path: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    merged_candidate = {
        "id": combo["id"],
        "description": combo["description"],
        "delete": [],
    }
    for cid in combo["candidates"]:
        merged_candidate["delete"].extend(
            next(c for c in candidate_results["_config"]["candidates"] if c["id"] == cid)["delete"]
        )

    actual = analyze_candidate(
        merged_candidate, data_text, shapes_graph, prefixes, baseline, constraints_by_path
    )

    naive_removed: set[str] = set()
    naive_introduced: set[str] = set()
    for cid in combo["candidates"]:
        single = candidate_results[cid]
        naive_removed.update(violation_fingerprint(v) for v in single["removedViolations"])
        naive_introduced.update(violation_fingerprint(v) for v in single["introducedViolations"])

    actual_removed = fingerprint_set(actual["removedViolations"])
    actual_introduced = fingerprint_set(actual["introducedViolations"])

    base_by_fp = {violation_fingerprint(v): v for v in baseline}

    all_single_lookup: dict[str, dict[str, str]] = dict(base_by_fp)
    for cid in combo["candidates"]:
        single = candidate_results[cid]
        for bucket in ("removedViolations", "introducedViolations", "retainedViolations"):
            for violation in single[bucket]:
                all_single_lookup.setdefault(violation_fingerprint(violation), violation)

    def materialize(fp_set: set[str], lookup: dict[str, dict[str, str]]) -> list[dict[str, str]]:
        return [lookup[fp] for fp in sorted(fp_set) if fp in lookup]

    after_lookup = {
        violation_fingerprint(v): v
        for v in actual["removedViolations"]
        + actual["introducedViolations"]
        + actual["retainedViolations"]
    }
    after_lookup.update(all_single_lookup)

    return {
        "id": combo["id"],
        "description": combo["description"],
        "memberCandidates": combo["candidates"],
        "deletedTriples": actual["deletedTriples"],
        "conforms": actual["conforms"],
        "fingerprint": actual["fingerprint"],
        "actual": {
            "removed": actual["removedViolations"],
            "introduced": actual["introducedViolations"],
            "retained": actual["retainedViolations"],
        },
        "naiveSum": {
            "removed": materialize(naive_removed, base_by_fp),
            "introduced": materialize(naive_introduced, after_lookup),
        },
        "nonAdditivity": {
            "removedClaimedByNaiveSumButNotActual": materialize(naive_removed - actual_removed, base_by_fp),
            "removedActualButMissedByNaiveSum": materialize(actual_removed - naive_removed, base_by_fp),
            "introducedClaimedByNaiveSumButNotActual": materialize(
                naive_introduced - actual_introduced, after_lookup
            ),
            "introducedActualButMissedByNaiveSum": materialize(
                actual_introduced - naive_introduced, after_lookup
            ),
        },
        "additive": (
            naive_removed == actual_removed and naive_introduced == actual_introduced
        ),
    }


def build_report(scenario_dir: Path) -> dict[str, Any]:
    config = json.loads((scenario_dir / "candidates.json").read_text(encoding="utf-8"))
    prefixes = load_prefixes(config)
    data_text = (scenario_dir / "data.ttl").read_text(encoding="utf-8")
    data_graph = Graph()
    data_graph.parse(data=data_text, format="turtle")
    shapes_graph = load_graph(scenario_dir / "shapes.ttl")

    conforms, baseline = run_validation(data_graph, shapes_graph)
    constraints_by_path = relevant_constraints(shapes_graph)

    candidate_results: dict[str, Any] = {"_config": config}
    for candidate in config["candidates"]:
        result = analyze_candidate(
            candidate, data_text, shapes_graph, prefixes, baseline, constraints_by_path
        )
        candidate_results[candidate["id"]] = result

    combinations = [
        analyze_combination(
            combo,
            candidate_results,
            data_text,
            shapes_graph,
            prefixes,
            baseline,
            constraints_by_path,
        )
        for combo in config.get("combinations", [])
    ]

    return {
        "scenario": {
            "dataGraph": "scenario/data.ttl",
            "shapesGraph": "scenario/shapes.ttl",
            "candidates": "scenario/candidates.json",
        },
        "baseline": {
            "conforms": conforms,
            "fingerprint": graph_fingerprint(baseline),
            "violationCount": len(baseline),
            "violations": baseline,
        },
        "sharedDependencies": [
            {
                "path": path,
                "propertyShape": info["propertyShape"],
                "constraints": info["constraints"],
            }
            for path, info in sorted(constraints_by_path.items())
        ],
        "candidates": [
            candidate_results[c["id"]] for c in config["candidates"]
        ],
        "combinations": combinations,
    }


def canonical_json(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def render_text(report: dict[str, Any]) -> str:
    lines: list[str] = []
    base = report["baseline"]
    lines.append(f"基线验证 conforms={base['conforms']} 违规数={base['violationCount']} 指纹={base['fingerprint'][:12]}")
    for v in base["violations"]:
        lines.append(
            f"  - {v['focusNode']} {v['resultPath']} {v['sourceConstraintComponent'].split('#')[-1]}"
            f" value={v['value'] or '∅'} :: {v['resultMessage']}"
        )

    lines.append("\n共同依赖（同一路径上的完整相关约束）:")
    for dep in report["sharedDependencies"]:
        joined = ", ".join(f"{c.split('#')[-1]}({p})" for c, p in dep["constraints"].items())
        lines.append(f"  {dep['path']} [{dep['propertyShape']}]: {joined}")

    for cand in report["candidates"]:
        lines.append(f"\n候选 {cand['id']}: {cand['description']}")
        for triple in cand["deletedTriples"]:
            lines.append(f"  删除: {triple[0]} {triple[1]} {triple[2]}")
        lines.append(
            f"  重验证 conforms={cand['conforms']} 指纹={cand['fingerprint'][:12]} "
            f"消失={cand['counts']['removed']} 新增={cand['counts']['introduced']} "
            f"保留={cand['counts']['retained']} 净变化={cand['counts']['netViolationChange']}"
        )
        for v in cand["removedViolations"]:
            lines.append(f"  [消失] {label_violation(v)}")
        for v in cand["introducedViolations"]:
            lines.append(f"  [新增] {label_violation(v)}")
        for v in cand["retainedViolations"]:
            lines.append(f"  [保留] {label_violation(v)}")
        for _, impact in cand["constraintImpact"].items():
            lines.append(f"  焦点 {impact['focusNode']} 路径 {impact['path']} 的完整约束影响:")
            for component, state in impact["components"].items():
                lines.append(
                    f"    {component.split('#')[-1]}({state['parameter']}): "
                    f"{state['baseline']} -> {state['after']}"
                )

    for combo in report["combinations"]:
        lines.append(f"\n组合 {combo['id']}: {combo['description']}")
        lines.append(f"  整体重验证 conforms={combo['conforms']} 指纹={combo['fingerprint'][:12]} 可加={combo['additive']}")
        na = combo["nonAdditivity"]
        for key, items in na.items():
            for v in items:
                lines.append(f"  不可加证据[{key}]: {label_violation(v)}")
        if combo["additive"]:
            lines.append("  组合效应与朴素求和一致（本场景中该组合不应出现此结果）")
    return "\n".join(lines) + "\n"


def label_violation(v: dict[str, str]) -> str:
    return (
        f"{v['focusNode']} {v['resultPath']} {v['sourceConstraintComponent'].split('#')[-1]}"
        f" value={v['value'] or '∅'} :: {v['resultMessage']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="清洗候选的完整约束影响分析")
    parser.add_argument("--scenario", type=Path, default=SCENARIO_DIR)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "report.json")
    parser.add_argument(
        "--check",
        type=Path,
        default=None,
        help="对给定 expected_diff.json 做重建比对，一致则退出码 0",
    )
    args = parser.parse_args()

    report = build_report(args.scenario)
    if args.check:
        expected = json.loads(args.check.read_text(encoding="utf-8"))
        if canonical_json(expected) != canonical_json(report):
            raise SystemExit(
                "重建的验证差异与预期不一致：候选影响不可复现，禁止按当前建议执行清洗"
            )
        print(f"OK: 重建结果与 {args.check} 完全一致（确定性指纹比对通过）")
        return

    args.out.write_text(canonical_json(report), encoding="utf-8")
    print(render_text(report))
    print(f"结构化报告已写入 {args.out}")


if __name__ == "__main__":
    main()
