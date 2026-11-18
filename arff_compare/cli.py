"""Command line interface for stratified ARFF comparisons."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Sequence

from .core import (
    ClassMapping,
    ComparisonError,
    compare_files,
)


def _fmt(value: Optional[float], digits: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return ("%." + str(digits) + "f") % value
    return str(value)


def _table(rows: Sequence[Sequence[str]], header: Sequence[str]) -> List[str]:
    widths = [len(cell) for cell in header]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    lines = ["  ".join(cell.ljust(widths[index])
                       for index, cell in enumerate(header))]
    lines.append("  ".join("-" * width for width in widths))
    for row in rows:
        lines.append("  ".join(cell.ljust(widths[index])
                               for index, cell in enumerate(row)))
    return lines


def render_text(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    inputs = report["inputs"]
    lines.append("ARFF stratified class-share comparison")
    lines.append("baseline: %s  (%d rows, total weight %s)"
                 % (inputs["baseline"]["name"], inputs["baseline"]["rows"],
                    _fmt(inputs["baseline"]["total_weight"])))
    lines.append("comparison: %s  (%d rows, total weight %s)"
                 % (inputs["comparison"]["name"], inputs["comparison"]["rows"],
                    _fmt(inputs["comparison"]["total_weight"])))
    lines.append("strata: %s; class attribute: %s; weights: %s"
                 % (", ".join(inputs["stratum_attributes"]),
                    inputs["class_attribute"],
                    inputs["weight_attribute"] or "unweighted (1 per row)"))
    lines.append("")

    lines.append("Actual weighted shares (observed rows; [low, high] includes "
                 "unknown class mass)")
    header = ["period"] + [
        "%s point [low, high]" % group for group in report["classes"]]
    rows = []
    for period in ("baseline", "comparison"):
        summary = report["actual"][period]
        row = [period]
        for group in report["classes"]:
            point = summary["completed_shares"][group]
            low, high = summary["bounds"][group]
            row.append("%s [%s, %s]" % (_fmt(point), _fmt(low), _fmt(high)))
        rows.append(row)
    lines.extend(_table(rows, header))
    for period in ("baseline", "comparison"):
        residual = report["actual"][period]["residual_total"]
        if residual > 0:
            labels = report["actual"][period]["residual_shares"]
            lines.append("%s residual mass: %s %s"
                         % (period, _fmt(residual),
                            json.dumps(labels, ensure_ascii=False)))
    lines.append("")

    standardized = report["standardized"]
    lines.append("Standardized to baseline stratum weights "
                 "(primary: common strata only)")
    coverage = standardized["coverage"]
    lines.append("  coverage: %.2f%% of baseline weight, %.2f%% of comparison "
                 "weight in common strata"
                 % (100 * coverage["baseline_weight_share_primary"],
                    100 * coverage["comparison_weight_share_in_common"]))
    rows = []
    for group in report["classes"]:
        point = standardized["primary"]["point"][group]
        low, high = standardized["primary"]["bounds"][group]
        actual = report["actual"]["comparison"]["completed_shares"][group]
        rows.append([group, _fmt(actual), _fmt(point),
                     ("%s [%s, %s]" % (_fmt(point), _fmt(low), _fmt(high))
                      if point is not None else "not identified"),
                     _fmt(report["actual"]["baseline"]["completed_shares"][group])])
    lines.extend(_table(rows, ["class", "actual", "standardized", "bounds",
                               "baseline"]))
    lines.append("  %s" % standardized["primary"]["note"])
    if coverage["baseline_weight_share_baseline_only"] > 0:
        lines.append("  baseline-only weight %.2f%%: comparison rate unknown "
                     "(full range [0, 1])"
                     % (100 * coverage["baseline_weight_share_baseline_only"]))
    if coverage["comparison_weight_share_comparison_only"] > 0:
        lines.append("  comparison-only weight %.2f%%: no baseline weight; "
                     "reported separately, not zero-filled"
                     % (100 * coverage["comparison_weight_share_comparison_only"]))
    full = standardized["full_baseline"]
    lines.append("  full-baseline bounds (one-sided strata included):")
    for group in report["classes"]:
        low, high = full["bounds"][group]
        lines.append("    %s: [%s, %s]" % (group, _fmt(low), _fmt(high)))
    lines.append("")

    lines.append("Standardized per-stratum contributions")
    rows = []
    for row in standardized["strata"]:
        for group in report["classes"]:
            entry = row["classes"][group]
            low, high = entry["contribution_bounds"]
            rows.append([
                row["stratum"], group,
                _fmt(row["standardized_weight_share"]),
                _fmt(entry["comparison_share"]),
                "%s [%s, %s]" % (_fmt(entry["contribution"]),
                                 _fmt(low), _fmt(high)),
            ])
    if rows:
        lines.extend(_table(rows, ["stratum", "class", "baseline weight",
                                   "comparison rate", "contribution"]))
    else:
        lines.append("  (no stratum is present in both periods)")
    lines.append("")

    lines.append("Weights-only counterfactual "
                 "(comparison weights x baseline rates)")
    rows = []
    for group in report["classes"]:
        point = report["fixed_rates"]["point"][group]
        low, high = report["fixed_rates"]["bounds"][group]
        rows.append([group, _fmt(point),
                     "%s [%s, %s]" % (_fmt(point), _fmt(low), _fmt(high)),
                     _fmt(report["actual"]["comparison"]["completed_shares"][group])])
    lines.extend(_table(rows, ["class", "fixed rates", "bounds", "observed"]))
    coverage = report["fixed_rates"]["coverage"]
    if coverage["comparison_only_weight_share"] > 0:
        lines.append("  comparison-only weight %.2f%%: baseline rate unknown "
                     "(full range [0, 1])"
                     % (100 * coverage["comparison_only_weight_share"]))
    lines.append("")

    lines.append("Explanation")
    for group in report["classes"]:
        item = report["explanation"][group]
        standardized_change = item["standardized_change_point"]
        lines.append("  %s: actual change %s; standardized change %s; "
                     "rate component %s"
                     % (group, _fmt(item["actual_change_point"]),
                        _fmt(standardized_change),
                        _fmt(item["actual_comparison_point"]
                             - item["fixed_rate_point"])))
        lines.append("    verdict: %s" % item["verdict"])
        if item["strata_blocking_identity"]:
            gaps = item["stratum_rate_gaps"]
            detail = ", ".join("%s (%s)" % (name, _fmt(gaps.get(name)))
                               for name in item["strata_blocking_identity"])
            lines.append("    strata blocking identical rates: %s" % detail)
        if item["witness"]:
            witness = item["witness"]
            lines.append("    witness: target %s; comparison allocations %d; "
                         "baseline-rate allocations %d"
                         % (_fmt(witness["target_share"]),
                            len(witness["comparison_completion"]["allocations"]),
                            len(witness["baseline_rate_completion"]["allocations"])))
    lines.append("")

    lines.append("Per-stratum contributions (mix / rate / interaction)")
    rows = []
    for row in report["decomposition"]["strata"]:
        support = row["support"]
        for group in report["classes"]:
            entry = row["classes"][group]
            if support == "both":
                rows.append([
                    row["stratum"], group,
                    _fmt(row["baseline_weight_share"]),
                    _fmt(row["comparison_weight_share"]),
                    _fmt(entry["baseline_share"]),
                    _fmt(entry["comparison_share"]),
                    _fmt(entry["mix_effect"]),
                    _fmt(entry["rate_effect"]),
                    _fmt(entry["interaction"]),
                ])
            else:
                rows.append([
                    row["stratum"], group,
                    _fmt(row["baseline_weight_share"]),
                    _fmt(row["comparison_weight_share"]),
                    _fmt(entry["baseline_share"]),
                    _fmt(entry["comparison_share"]),
                    "-", "-", "-",
                ])
    lines.extend(_table(rows, ["stratum", "class", "wA", "wB", "pA", "pB",
                               "mix", "rate", "interaction"]))
    lines.append("  Only 'both' rows enter the mix/rate totals; one-sided rows "
                 "are listed but not assigned a contribution.")
    lines.append("")

    lines.append("Missingness audit")
    missing = report["missing"]
    lines.append("  baseline unassigned stratum weight: %s"
                 % _fmt(missing["baseline_unassigned_weight"]))
    lines.append("  comparison unassigned stratum weight: %s"
                 % _fmt(missing["comparison_unassigned_weight"]))
    lines.append("  baseline unknown labels: %s"
                 % json.dumps(missing["baseline_unknown_labels"],
                              ensure_ascii=False))
    lines.append("  comparison unknown labels: %s"
                 % json.dumps(missing["comparison_unknown_labels"],
                              ensure_ascii=False))
    lines.append("  baseline-only strata: %s"
                 % (", ".join(missing["baseline_only_strata"]) or "none"))
    lines.append("  comparison-only strata: %s"
                 % (", ".join(missing["comparison_only_strata"]) or "none"))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m arff_compare",
        description=(
            "Compare class shares in two weighted ARFF periods by stratum, "
            "with direct standardisation and identified bounds."))
    parser.add_argument("baseline", help="baseline-period ARFF file")
    parser.add_argument("comparison", help="comparison-period ARFF file")
    parser.add_argument("--stratum", action="append", default=[],
                        metavar="ATTRIBUTE",
                        help="stratum attribute; repeat for a composite stratum")
    parser.add_argument("--strata", metavar="A,B",
                        help="comma-separated stratum attributes")
    parser.add_argument("--class-attribute", required=True,
                        help="nominal class attribute")
    parser.add_argument("--mapping",
                        help="JSON class-equivalence mapping; default is the "
                             "identity on labels declared in both ARFF headers")
    parser.add_argument("--weight-attribute",
                        help="numeric observation weight; default 1 per row")
    parser.add_argument("--missing-weight", choices=("error", "one", "mean"),
                        default="error",
                        help="rule for a missing observation weight")
    parser.add_argument("--material-threshold", type=float, default=0.01,
                        help="threshold used only for verdict wording")
    parser.add_argument("--allow-string-class", action="store_true",
                        help="accept a STRING class attribute when --mapping "
                             "is supplied")
    parser.add_argument("--format", choices=("json", "text"), default="json")
    parser.add_argument("--output", "-o", help="write to this file")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    strata = list(args.stratum)
    if args.strata:
        strata.extend(part.strip() for part in args.strata.split(",")
                      if part.strip())
    if not strata:
        parser.error("at least one --stratum or --strata is required")
    try:
        mapping = ClassMapping.from_file(args.mapping) if args.mapping else None
        report = compare_files(
            args.baseline, args.comparison, strata, args.class_attribute,
            mapping=mapping, weight_attribute=args.weight_attribute,
            missing_weight=args.missing_weight,
            material_threshold=args.material_threshold,
            allow_string_class=args.allow_string_class)
    except ComparisonError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2

    output = (json.dumps(report, indent=2, ensure_ascii=False, sort_keys=False)
              if args.format == "json" else render_text(report))
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as handle:
                handle.write(output)
                if not output.endswith("\n"):
                    handle.write("\n")
        except OSError as exc:
            print("error: cannot write %s: %s" % (args.output, exc),
                  file=sys.stderr)
            return 2
    else:
        print(output)
    return 0
