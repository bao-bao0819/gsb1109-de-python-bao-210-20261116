"""Stratified class-share comparison for two weighted ARFF periods.

The report keeps these quantities separate:

``actual``
    Weighted class shares of the observed rows in each period.  A mapped
    class is reported as a known lower share plus an upper share that includes
    every unknown or unmapped class.  Bounds do not reallocate anything.

``standardized``
    The comparison period re-weighted to baseline-period stratum weights
    (direct standardisation).  The primary figure is conditional on strata
    observed in both periods; its coverage is reported next to it.  A
    secondary range includes baseline-only strata with an unknown comparison
    rate in ``[0, 1]``, so those strata are never treated as zero-rate.

``fixed_rates``
    The weights-only counterfactual: comparison-period weights applied to
    baseline-period class rates.  A comparison-only stratum keeps its observed
    comparison weight and receives an unknown baseline rate in ``[0, 1]``.
    The pooled interval is necessary but not sufficient: offsetting rate
    changes can cancel.  ``stratumwise_equality`` therefore checks the exact
    joint condition in every common stratum, and that is what the verdicts use.

``decomposition``
    Per-stratum weighted contributions and, for strata observed in both
    periods, Kitagawa mix / rate / interaction terms.  One-sided strata are
    reported as non-comparable mass, never as zero contributions.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import arff

UNASSIGNED_STRATUM = "__UNASSIGNED_STRATUM__"
MISSING_CLASS = "__MISSING_CLASS__"
MISSING_LABEL = "?"


class ComparisonError(ValueError):
    """Raised for invalid inputs or configurations."""


def _text(value: Any) -> str:
    if value is None:
        return MISSING_LABEL
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(path: str) -> Dict[str, Any]:
    """Load an ARFF file, keeping undeclared nominal labels as strings."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return arff.load(handle, lenient_nominal=True)
    except OSError as exc:
        raise ComparisonError("cannot read %s: %s" % (path, exc)) from exc
    except arff.ArffException as exc:
        raise ComparisonError("invalid ARFF %s: %s" % (path, exc)) from exc


def _attribute_position(dataset: Mapping[str, Any], name: str) -> int:
    names = [_text(attr[0]) for attr in dataset["attributes"]]
    try:
        return names.index(name)
    except ValueError:
        raise ComparisonError(
            "attribute %r is not present; available attributes: %s"
            % (name, ", ".join(names))
        )


def _nominal_values(dataset: Mapping[str, Any], name: str) -> Optional[List[str]]:
    position = _attribute_position(dataset, name)
    kind = dataset["attributes"][position][1]
    if isinstance(kind, (list, tuple)):
        return [_text(value) for value in kind]
    return None


class ClassMapping:
    """Raw-label to class-group equivalence mapping.

    Accepted JSON forms::

        {"positive": ["yes", "Y"], "negative": ["no", "N"]}
        {"classes": ["positive", "negative"],
         "map": {"yes": "positive", "no": "negative"}}
        [{"group": "positive", "values": ["yes", "Y"]}]

    A raw label may occur in at most one group.  Unmapped labels remain
    residual mass and are named in the report; they are never dropped or
    forced into the first group.
    """

    def __init__(self, groups: Sequence[str],
                 raw_to_group: Mapping[str, str]):
        ordered: List[str] = []
        for group in groups:
            group = _text(group)
            if group not in ordered:
                ordered.append(group)
        if not ordered:
            raise ComparisonError("the class mapping must contain at least one group")
        self.groups = ordered
        self.raw_to_group = {_text(raw): _text(group)
                             for raw, group in raw_to_group.items()}
        for raw, group in self.raw_to_group.items():
            if group not in self.groups:
                raise ComparisonError(
                    "raw label %r maps to unknown class group %r" % (raw, group)
                )

    @classmethod
    def identity(cls, labels: Iterable[str]) -> "ClassMapping":
        ordered: List[str] = []
        for label in labels:
            if label != MISSING_LABEL and label not in ordered:
                ordered.append(label)
        if not ordered:
            raise ComparisonError("no class labels found for an identity mapping")
        return cls(ordered, {label: label for label in ordered})

    @classmethod
    def from_obj(cls, obj: Any) -> "ClassMapping":
        if isinstance(obj, list):
            groups: List[str] = []
            raw_to_group: Dict[str, str] = {}
            for item in obj:
                if not isinstance(item, Mapping) or "group" not in item:
                    raise ComparisonError("mapping list entries need a 'group'")
                group = _text(item["group"])
                groups.append(group)
                for value in item.get("values", []):
                    raw = _text(value)
                    if raw in raw_to_group and raw_to_group[raw] != group:
                        raise ComparisonError(
                            "raw label %r is mapped to both %r and %r"
                            % (raw, raw_to_group[raw], group))
                    raw_to_group[raw] = group
            return cls(groups, raw_to_group)

        if not isinstance(obj, Mapping):
            raise ComparisonError("mapping must be a JSON object or list")
        if "map" in obj or "classes" in obj:
            groups = [_text(group) for group in obj.get("classes", [])]
            raw_to_group = {}
            for raw, group in obj.get("map", {}).items():
                raw_to_group[_text(raw)] = _text(group)
                if _text(group) not in groups:
                    groups.append(_text(group))
            return cls(groups, raw_to_group)

        groups = []
        raw_to_group = {}
        for group, values in obj.items():
            group = _text(group)
            groups.append(group)
            if isinstance(values, str):
                values = [values]
            for value in values:
                raw = _text(value)
                if raw in raw_to_group and raw_to_group[raw] != group:
                    raise ComparisonError(
                        "raw label %r is mapped to both %r and %r"
                        % (raw, raw_to_group[raw], group))
                raw_to_group[raw] = group
        return cls(groups, raw_to_group)

    @classmethod
    def from_file(cls, path: str) -> "ClassMapping":
        try:
            with open(path, "r", encoding="utf-8") as handle:
                obj = json.load(handle)
        except OSError as exc:
            raise ComparisonError("cannot read mapping %s: %s" % (path, exc)) from exc
        except ValueError as exc:
            raise ComparisonError("invalid mapping JSON %s: %s" % (path, exc)) from exc
        return cls.from_obj(obj)

    def group_of(self, raw: str) -> Optional[str]:
        return self.raw_to_group.get(raw)

    def to_obj(self) -> Dict[str, Any]:
        return {"classes": list(self.groups),
                "map": OrderedDict(sorted(self.raw_to_group.items()))}


@dataclass
class Cell:
    key: Any
    label: str
    weight: float
    mapped: Dict[str, float]
    residual: float

    @property
    def known_total(self) -> float:
        return math.fsum(self.mapped.values())

    def low(self, group: str) -> float:
        return self.mapped.get(group, 0.0) / self.weight if self.weight else 0.0

    def high(self, group: str) -> float:
        if not self.weight:
            return 0.0
        return min(1.0, (self.mapped.get(group, 0.0) + self.residual) / self.weight)

    def point(self, group: str) -> float:
        """Proportional completion of residual mass inside this stratum."""
        if self.weight <= 0:
            return 0.0
        extra = 0.0
        if self.residual > 0:
            if self.known_total > 0:
                extra = self.residual * self.mapped.get(group, 0.0) / self.known_total
            else:
                extra = self.residual / len(self.mapped) if self.mapped else 0.0
        return (self.mapped.get(group, 0.0) + extra) / self.weight


@dataclass
class Period:
    name: str
    path: Optional[str]
    relation: str
    rows: int
    total: float
    cells: "OrderedDict[Any, Cell]"
    residual_by_label: Dict[str, float]
    unknown_labels: Dict[str, float]
    missing_class_weight: float
    zero_weight_rows: int
    missing_weight: str
    declared_classes: List[str]

    def weight_share(self, key: Any) -> float:
        cell = self.cells.get(key)
        return cell.weight / self.total if cell else 0.0

    def point(self, group: str) -> float:
        return math.fsum(cell.weight * cell.point(group)
                         for cell in self.cells.values()) / self.total

    def bounds(self, group: str) -> Tuple[float, float]:
        known = math.fsum(cell.mapped.get(group, 0.0)
                          for cell in self.cells.values())
        residual = math.fsum(cell.residual for cell in self.cells.values())
        return known / self.total, (known + residual) / self.total


def _parse_weight(raw: Any, policy: str, mean: Optional[float]) -> float:
    if raw is None or raw == "":
        if policy == "error":
            raise ComparisonError(
                "missing observation weight; pass --missing-weight one or "
                "--missing-weight mean to state an explicit rule")
        if policy == "one":
            return 1.0
        if policy == "mean":
            if mean is None:
                raise ComparisonError(
                    "cannot use --missing-weight mean without an observed weight")
            return mean
        raise ComparisonError("unknown missing-weight policy %r" % policy)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ComparisonError("weight %r is not numeric" % (raw,))
    if not math.isfinite(value):
        raise ComparisonError("weight %r is not finite" % (raw,))
    if value < 0:
        raise ComparisonError("weight %r is negative" % (raw,))
    return value


def _stratum_key(parts: Sequence[Any], names: Sequence[str]) -> Any:
    if any(part is None for part in parts):
        return UNASSIGNED_STRATUM
    if len(names) == 1:
        return _text(parts[0])
    return tuple(_text(part) for part in parts)


def stratum_label(key: Any, names: Sequence[str]) -> str:
    if key == UNASSIGNED_STRATUM:
        return "(missing stratum)"
    if len(names) == 1:
        return str(key)
    return " | ".join("%s=%s" % (name, value)
                      for name, value in zip(names, key))


def build_period(
    dataset: Mapping[str, Any],
    name: str,
    stratum_attributes: Sequence[str],
    class_attribute: str,
    mapping: ClassMapping,
    weight_attribute: Optional[str],
    missing_weight: str,
    path: Optional[str],
    allow_string_class: bool = False,
) -> Period:
    """Aggregate one ARFF dataset into weighted stratum/class cells."""
    if not stratum_attributes:
        raise ComparisonError("at least one stratum attribute is required")
    stratum_positions = [_attribute_position(dataset, attr)
                         for attr in stratum_attributes]
    class_position = _attribute_position(dataset, class_attribute)
    weight_position = (_attribute_position(dataset, weight_attribute)
                       if weight_attribute else None)
    declared = _nominal_values(dataset, class_attribute)
    if declared is None and not allow_string_class:
        raise ComparisonError(
            "class attribute %r must be nominal; declare its labels in the "
            "header, or pass allow_string_class with an explicit mapping"
            % class_attribute)

    rows = list(dataset["data"])
    observed_weights = []
    if weight_position is not None:
        for row in rows:
            raw = row[weight_position]
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value >= 0:
                observed_weights.append(value)
    mean_weight = (math.fsum(observed_weights) / len(observed_weights)
                   if observed_weights else None)

    cells: "OrderedDict[Any, Cell]" = OrderedDict()
    residual_by_label: Dict[str, float] = {}
    unknown_labels: Dict[str, float] = {}
    missing_class_weight = 0.0
    total = 0.0
    zero_rows = 0

    for row in rows:
        weight = (1.0 if weight_position is None else
                  _parse_weight(row[weight_position], missing_weight, mean_weight))
        if weight == 0:
            zero_rows += 1
            continue
        key = _stratum_key([row[position] for position in stratum_positions],
                           stratum_attributes)
        cell = cells.get(key)
        if cell is None:
            cell = Cell(key=key, label=stratum_label(key, stratum_attributes),
                        weight=0.0,
                        mapped={group: 0.0 for group in mapping.groups},
                        residual=0.0)
            cells[key] = cell

        raw_class = row[class_position]
        group = None if raw_class is None else mapping.group_of(_text(raw_class))
        cell.weight += weight
        total += weight
        if group is not None:
            cell.mapped[group] += weight
        else:
            cell.residual += weight
            if raw_class is None:
                label_key = MISSING_CLASS
                missing_class_weight += weight
            else:
                label_key = _text(raw_class)
                unknown_labels[label_key] = (unknown_labels.get(label_key, 0.0)
                                             + weight)
            residual_by_label[label_key] = (residual_by_label.get(label_key, 0.0)
                                            + weight)
    if total <= 0:
        raise ComparisonError("period %r has no positive observation weight" % name)
    return Period(name=name, path=path,
                  relation=_text(dataset.get("relation", "")),
                  rows=len(rows), total=total, cells=cells,
                  residual_by_label=residual_by_label,
                  unknown_labels=unknown_labels,
                  missing_class_weight=missing_class_weight,
                  zero_weight_rows=zero_rows, missing_weight=missing_weight,
                  declared_classes=list(declared or []))


def _period_summary(period: Period, groups: Sequence[str]) -> Dict[str, Any]:
    known, upper, point = {}, {}, {}
    for group in groups:
        low, high = period.bounds(group)
        known[group] = low
        upper[group] = high
        point[group] = period.point(group)
    return {
        "rows": period.rows,
        "relation": period.relation,
        "total_weight": period.total,
        "strata": len(period.cells),
        "zero_weight_rows": period.zero_weight_rows,
        "known_mapped_shares": known,
        "bounds": {group: [known[group], upper[group]] for group in groups},
        "completed_shares": point,
        "known_mapped_total": math.fsum(known.values()),
        "residual_total": math.fsum(period.residual_by_label.values()) / period.total,
        "residual_shares": {label: weight / period.total
                            for label, weight in period.residual_by_label.items()},
        "unassigned_stratum_weight": (
            period.cells[UNASSIGNED_STRATUM].weight
            if UNASSIGNED_STRATUM in period.cells else 0.0),
    }


def _split(period_a: Period, period_b: Period) -> Tuple[List[Any], List[Any], List[Any]]:
    a_keys = [key for key in period_a.cells if key != UNASSIGNED_STRATUM]
    b_keys = [key for key in period_b.cells if key != UNASSIGNED_STRATUM]
    common = [key for key in a_keys if key in period_b.cells]
    return common, [key for key in a_keys if key not in period_b.cells], \
        [key for key in b_keys if key not in period_a.cells]


def _standardized(baseline: Period, comparison: Period,
                  groups: Sequence[str], names: Sequence[str]) -> Dict[str, Any]:
    common, baseline_only, comparison_only = _split(baseline, comparison)
    base_unassigned = UNASSIGNED_STRATUM in baseline.cells
    comp_unassigned = UNASSIGNED_STRATUM in comparison.cells
    W0 = baseline.total
    common_den = math.fsum(baseline.cells[key].weight for key in common)

    point, low, high = {}, {}, {}
    full_low, full_high, full_point = {}, {}, {}
    pooled_b = {group: comparison.point(group) for group in groups}
    for group in groups:
        num_p = num_l = num_h = 0.0
        for key in common:
            w = baseline.cells[key].weight
            cell = comparison.cells[key]
            num_p += w * cell.point(group)
            num_l += w * cell.low(group)
            num_h += w * cell.high(group)
        one_sided_weight = math.fsum(baseline.cells[key].weight
                                     for key in baseline_only)
        if base_unassigned:
            one_sided_weight += baseline.cells[UNASSIGNED_STRATUM].weight
        point[group] = num_p / common_den if common_den else None
        low[group] = num_l / common_den if common_den else None
        high[group] = num_h / common_den if common_den else None
        full_point[group] = (
            (num_p + one_sided_weight * pooled_b[group]) / W0)
        full_low[group] = num_l / W0
        full_high[group] = min(1.0, (num_h + one_sided_weight) / W0)

    return {
        "target": (
            "comparison-period class shares re-weighted with baseline total "
            "observation weight per stratum"),
        "primary": {
            "scope": "strata observed in both periods",
            "denominator_weight": common_den,
            "point": point,
            "bounds": {group: [low[group], high[group]] for group in groups},
            "identified": bool(common_den),
            "note": (
                "no stratum occurs in both periods; direct standardisation "
                "has no jointly observed rate and is not identified"
                if not common_den else
                "rates are identified from rows in the common strata only"),
            "baseline_comparator": {
                "point": ({group: _common_point(baseline, common, group)
                           for group in groups} if common_den else
                          {group: None for group in groups}),
                "bounds": ({group: list(_common_bounds(baseline, common, group))
                            for group in groups} if common_den else
                           {group: None for group in groups}),
            },
        },
        "strata": _standardized_strata(
            baseline, comparison, common, groups, common_den, names),
        "full_baseline": {
            "scope": (
                "all observed baseline weight; baseline-only strata receive "
                "an unknown comparison rate in [0, 1] rather than zero"),
            "denominator_weight": W0,
            "point": full_point,
            "bounds": {group: [full_low[group], full_high[group]]
                       for group in groups},
            "one_sided_weight": math.fsum(
                baseline.cells[key].weight for key in baseline_only)
            + (baseline.cells[UNASSIGNED_STRATUM].weight
               if base_unassigned else 0.0),
        },
        "coverage": {
            "baseline_weight_share_primary": common_den / W0,
            "baseline_weight_share_baseline_only": math.fsum(
                baseline.cells[key].weight for key in baseline_only) / W0,
            "baseline_unassigned_weight_share": (
                baseline.cells[UNASSIGNED_STRATUM].weight / W0
                if base_unassigned else 0.0),
            "comparison_weight_share_in_common": math.fsum(
                comparison.cells[key].weight for key in common) / comparison.total,
            "comparison_weight_share_comparison_only": math.fsum(
                comparison.cells[key].weight for key in comparison_only)
            / comparison.total,
            "comparison_unassigned_weight_share": (
                comparison.cells[UNASSIGNED_STRATUM].weight / comparison.total
                if comp_unassigned else 0.0),
        },
        "baseline_only_strata": [
            {"stratum": stratum_label(key, names),
             "baseline_weight_share": baseline.weight_share(key),
             "comparison_rate_bounds": [0.0, 1.0]}
            for key in baseline_only],
        "comparison_only_strata": [
            {"stratum": stratum_label(key, names),
             "comparison_weight_share": comparison.weight_share(key),
             "comparison_share": {
                 group: comparison.cells[key].point(group)
                 for group in groups},
             "note": ("no baseline rows; baseline weight is unidentified.  "
                      "The primary standard excludes this stratum; "
                      "comparison_only_weight_free gives the range obtained "
                      "if its baseline weight is allowed to be any "
                      "nonnegative value")}
            for key in comparison_only],
        "comparison_only_weight_free": _free_weight_range(
            comparison_only, groups, comparison, point),
        "sensitivity": {
            "note": (
                "the primary standard uses only strata present in both "
                "periods.  The full_baseline block adds baseline-only strata "
                "with a comparison rate in [0, 1].  A comparison-only stratum "
                "has no baseline weight at all: it is neither given zero nor "
                "silently dropped.  Its effect is shown in fixed_rates, where "
                "it keeps its observed comparison weight and its baseline "
                "rate ranges over [0, 1]."),
        },
    }


def _free_weight_range(comparison_only: Sequence[Any],
                       groups: Sequence[str],
                       comparison: Period,
                       common_point: Mapping[str, Optional[float]],
                       ) -> Dict[str, List[Optional[float]]]:
    """Range of standardized shares over any baseline weight for new strata.

    A comparison-only stratum has no observed baseline weight.  Allowing that
    weight to be any nonnegative value, and its class rate to lie in the
    identified interval, makes the standardized share range from the
    common-strata value to the extremes of those class rates.  This is a
    reachable range, not an imputation.
    """
    result: Dict[str, List[Optional[float]]] = {}
    for group in groups:
        candidates_low = []
        candidates_high = []
        anchor = common_point.get(group)
        if anchor is not None:
            candidates_low.append(anchor)
            candidates_high.append(anchor)
        for key in comparison_only:
            cell = comparison.cells[key]
            candidates_low.append(cell.low(group))
            candidates_high.append(cell.high(group))
        if not candidates_low:
            result[group] = [None, None]
        else:
            result[group] = [min(candidates_low), max(candidates_high)]
    return result


def _standardized_strata(baseline: Period, comparison: Period,
                         common: Sequence[Any], groups: Sequence[str],
                         denominator: float,
                         names: Sequence[str]) -> List[Dict[str, Any]]:
    """Per-stratum baseline-weight contributions to the standardized share."""
    rows = []
    for key in common:
        b0 = baseline.cells[key]
        b1 = comparison.cells[key]
        row: Dict[str, Any] = {
            "stratum": stratum_label(key, names),
            "baseline_weight": b0.weight,
            "comparison_weight": b1.weight,
            "standardized_weight_share": (
                b0.weight / denominator if denominator else None),
            "classes": {},
        }
        for group in groups:
            row["classes"][group] = {
                "comparison_share": b1.point(group),
                "comparison_share_bounds": [b1.low(group), b1.high(group)],
                "contribution": (
                    b0.weight * b1.point(group) / denominator
                    if denominator else None),
                "contribution_bounds": [
                    b0.weight * b1.low(group) / denominator,
                    b0.weight * b1.high(group) / denominator,
                ] if denominator else None,
            }
        rows.append(row)
    return rows


def _common_point(period: Period, keys: Sequence[Any], group: str) -> Optional[float]:
    den = math.fsum(period.cells[key].weight for key in keys)
    if not den:
        return None
    return math.fsum(period.cells[key].weight * period.cells[key].point(group)
                     for key in keys) / den


def _common_bounds(period: Period, keys: Sequence[Any],
                   group: str) -> Optional[Tuple[float, float]]:
    den = math.fsum(period.cells[key].weight for key in keys)
    if not den:
        return None
    low = math.fsum(period.cells[key].weight * period.cells[key].low(group)
                    for key in keys) / den
    high = math.fsum(period.cells[key].weight * period.cells[key].high(group)
                     for key in keys) / den
    return low, high


def _fixed_rates(baseline: Period, comparison: Period,
                 groups: Sequence[str], names: Sequence[str]) -> Dict[str, Any]:
    common, baseline_only, comparison_only = _split(baseline, comparison)
    comp_unassigned = UNASSIGNED_STRATUM in comparison.cells
    W1 = comparison.total
    pooled_a = {group: baseline.point(group) for group in groups}

    point, low, high = {}, {}, {}
    for group in groups:
        num_p = num_l = num_h = 0.0
        for key in common:
            w = comparison.cells[key].weight
            cell = baseline.cells[key]
            num_p += w * cell.point(group)
            num_l += w * cell.low(group)
            num_h += w * cell.high(group)
        one_sided = math.fsum(comparison.cells[key].weight
                              for key in comparison_only)
        if comp_unassigned:
            one_sided += comparison.cells[UNASSIGNED_STRATUM].weight
        point[group] = (num_p + one_sided * pooled_a[group]) / W1
        low[group] = num_l / W1
        high[group] = min(1.0, (num_h + one_sided) / W1)

    return {
        "scope": (
            "baseline-period class rates held fixed, re-weighted with the "
            "comparison period's observed stratum weights"),
        "denominator_weight": W1,
        "point": point,
        "bounds": {group: [low[group], high[group]] for group in groups},
        "coverage": {
            "comparison_weight_share_with_baseline_rate": math.fsum(
                comparison.cells[key].weight for key in common) / W1,
            "comparison_only_weight_share": math.fsum(
                comparison.cells[key].weight for key in comparison_only) / W1,
            "comparison_unassigned_weight_share": (
                comparison.cells[UNASSIGNED_STRATUM].weight / W1
                if comp_unassigned else 0.0),
            "baseline_weight_not_represented": math.fsum(
                baseline.cells[key].weight for key in baseline_only)
            / baseline.total,
        },
        "comparison_only_strata": [
            {"stratum": stratum_label(key, names),
             "comparison_weight_share": comparison.weight_share(key),
             "baseline_rate_bounds": [0.0, 1.0]}
            for key in comparison_only],
    }


def _decomposition(baseline: Period, comparison: Period,
                   groups: Sequence[str], names: Sequence[str]) -> Dict[str, Any]:
    W0, W1 = baseline.total, comparison.total
    common, baseline_only, comparison_only = _split(baseline, comparison)
    base_unassigned = UNASSIGNED_STRATUM in baseline.cells
    comp_unassigned = UNASSIGNED_STRATUM in comparison.cells
    common_den = math.fsum(baseline.cells[key].weight for key in common)

    rows = []
    keys = list(baseline.cells) + [key for key in comparison.cells
                                   if key not in baseline.cells]
    for key in keys:
        in_a, in_b = key in baseline.cells, key in comparison.cells
        if key == UNASSIGNED_STRATUM:
            support = "unassigned"
        elif in_a and in_b:
            support = "both"
        elif in_a:
            support = "baseline_only"
        else:
            support = "comparison_only"
        row: Dict[str, Any] = {
            "stratum": stratum_label(key, names),
            "support": support,
            "baseline_weight": baseline.cells[key].weight if in_a else None,
            "comparison_weight": comparison.cells[key].weight if in_b else None,
            "baseline_weight_share": (baseline.cells[key].weight / W0
                                      if in_a else None),
            "comparison_weight_share": (comparison.cells[key].weight / W1
                                        if in_b else None),
            "standardized_weight_share": (
                baseline.cells[key].weight / common_den
                if in_a and in_b and common_den else None),
            "classes": {},
        }
        for group in groups:
            entry: Dict[str, Any] = {
                "baseline_weight_share": row["baseline_weight_share"],
                "comparison_weight_share": row["comparison_weight_share"],
                "baseline_share": (baseline.cells[key].point(group)
                                   if in_a else None),
                "comparison_share": (comparison.cells[key].point(group)
                                     if in_b else None),
                "baseline_share_bounds": ([baseline.cells[key].low(group),
                                           baseline.cells[key].high(group)]
                                          if in_a else None),
                "comparison_share_bounds": ([comparison.cells[key].low(group),
                                             comparison.cells[key].high(group)]
                                            if in_b else None),
            }
            if in_a and in_b:
                p0, p1 = entry["baseline_share"], entry["comparison_share"]
                pi0 = entry["baseline_weight_share"]
                pi1 = entry["comparison_weight_share"]
                entry.update({
                    "contribution_baseline": pi0 * p0,
                    "contribution_comparison": pi1 * p1,
                    "contribution_change": pi1 * p1 - pi0 * p0,
                    "contribution_change_bounds": [
                        pi1 * entry["comparison_share_bounds"][0]
                        - pi0 * entry["baseline_share_bounds"][1],
                        pi1 * entry["comparison_share_bounds"][1]
                        - pi0 * entry["baseline_share_bounds"][0]],
                    "mix_effect": (pi1 - pi0) * p0,
                    "rate_effect": pi0 * (p1 - p0),
                    "interaction": (pi1 - pi0) * (p1 - p0),
                })
            elif in_a:
                pi0 = entry["baseline_weight_share"]
                entry.update({
                    "contribution_baseline": pi0 * entry["baseline_share"],
                    "contribution_comparison": None,
                    "contribution_change": None,
                    "contribution_change_bounds": None,
                    "note": (
                        "no comparison rows in this stratum: the comparison "
                        "weight and class rate are unidentified without a "
                        "stratum frame, so no change or range is asserted"),
                })
            else:
                pi1 = entry["comparison_weight_share"]
                entry.update({
                    "contribution_baseline": None,
                    "contribution_comparison": pi1 * entry["comparison_share"],
                    "contribution_change": None,
                    "contribution_change_bounds": None,
                    "note": (
                        "no baseline rows in this stratum: no baseline "
                        "contribution is asserted"),
                })
            row["classes"][group] = entry
        rows.append(row)

    totals = {}
    for group in groups:
        a = baseline.point(group)
        b = comparison.point(group)
        actual_change = b - a
        mix = rate = inter = comparable = 0.0
        for row in rows:
            if row["support"] != "both":
                continue
            entry = row["classes"][group]
            mix += entry["mix_effect"]
            rate += entry["rate_effect"]
            inter += entry["interaction"]
            comparable += entry["contribution_change"]
        totals[group] = {
            "actual_change_point": actual_change,
            "comparable_change_point": comparable,
            "mix_effect_point": mix,
            "rate_effect_point": rate,
            "interaction_point": inter,
            "noncomparable_change_point": actual_change - comparable,
            "identity_check": mix + rate + inter - comparable,
        }

    return {
        "definition": (
            "for strata present in both periods: mix=(pi_B-pi_A)*p_A, "
            "rate=pi_A*(p_B-p_A), interaction=(pi_B-pi_A)*(p_B-p_A), with "
            "pi the period weight share and p the completed class share; "
            "totals cover only those strata"),
        "strata": rows,
        "totals": totals,
        "coverage": {
            "baseline_weight_share_in_common": common_den / W0,
            "comparison_weight_share_in_common": math.fsum(
                comparison.cells[key].weight for key in common) / W1,
            "baseline_only_weight_share": math.fsum(
                baseline.cells[key].weight for key in baseline_only) / W0,
            "baseline_unassigned_weight_share": (
                baseline.cells[UNASSIGNED_STRATUM].weight / W0
                if base_unassigned else 0.0),
            "comparison_only_weight_share": math.fsum(
                comparison.cells[key].weight for key in comparison_only) / W1,
            "comparison_unassigned_weight_share": (
                comparison.cells[UNASSIGNED_STRATUM].weight / W1
                if comp_unassigned else 0.0),
        },
    }


def _target_choice(actual_low: float, actual_high: float,
                   fixed_low: float, fixed_high: float,
                   preferred: float) -> Tuple[bool, Optional[float], Optional[float]]:
    """Return overlap, target, and the gap if the intervals are disjoint."""
    low = max(actual_low, fixed_low)
    high = min(actual_high, fixed_high)
    if low <= high + 1e-12:
        target = min(max(preferred, low), high)
        return True, target, None
    if fixed_high < actual_low:
        gap = actual_low - fixed_high
    else:
        gap = fixed_low - actual_high
    return False, None, gap


def _witness_actual(comparison: Period, group: str, target: float) -> Dict[str, Any]:
    """Concretely allocate comparison residual mass to reproduce ``target``."""
    known = math.fsum(cell.mapped.get(group, 0.0)
                      for cell in comparison.cells.values())
    needed = target * comparison.total - known
    residual_cells = [cell for cell in comparison.cells.values()
                      if cell.residual > 0]
    allocations = []
    remaining = max(0.0, needed)
    for cell in residual_cells:
        if remaining <= 1e-15:
            break
        take = min(cell.residual, remaining)
        remaining -= take
        allocations.append({
            "stratum": cell.label,
            "action": "assign residual class mass to %r" % group,
            "residual_weight": take,
            "residual_share": take / cell.weight if cell.weight else None,
        })
    if remaining > 1e-9:
        raise ComparisonError("comparison residual mass is insufficient")
    return {"target_share": target, "known_mapped_weight": known,
            "allocated_residual_weight": needed, "allocations": allocations}


def _witness_baseline_rate(baseline: Period, comparison: Period,
                           group: str, target: float) -> Dict[str, Any]:
    """Concretely raise baseline rates so the fixed-rate share is ``target``."""
    W1 = comparison.total
    common, _, comparison_only = _split(baseline, comparison)
    comp_unassigned = UNASSIGNED_STRATUM in comparison.cells
    base = math.fsum(comparison.cells[key].weight
                     * baseline.cells[key].low(group) for key in common)
    sources = []  # (kind, key, capacity in weighted units)
    for key in common:
        cell = baseline.cells[key]
        if cell.residual > 0:
            sources.append(("baseline_residual", key,
                            comparison.cells[key].weight * cell.residual
                            / cell.weight))
    for key in comparison_only:
        sources.append(("unobserved_rate", key, comparison.cells[key].weight))
    if comp_unassigned:
        sources.append(("unobserved_rate", UNASSIGNED_STRATUM,
                        comparison.cells[UNASSIGNED_STRATUM].weight))
    needed = target * W1 - base
    allocations = []
    remaining = max(0.0, needed)
    for kind, key, capacity in sources:
        if remaining <= 1e-15:
            break
        take = min(capacity, remaining)
        remaining -= take
        if kind == "baseline_residual":
            cell = baseline.cells[key]
            weight = take * cell.weight / comparison.cells[key].weight
            allocations.append({
                "stratum": cell.label,
                "action": "assign baseline residual class mass to %r" % group,
                "residual_weight": weight,
                "residual_share_of_stratum": weight / cell.weight,
            })
        else:
            cell = (comparison.cells[key] if key != UNASSIGNED_STRATUM
                    else comparison.cells[UNASSIGNED_STRATUM])
            allocations.append({
                "stratum": cell.label,
                "action": "set the unobserved baseline class rate",
                "rate": take / cell.weight if cell.weight else None,
            })
    if remaining > 1e-9:
        raise ComparisonError("baseline residual mass is insufficient")
    return {"target_share": target, "known_rate_component": base / W1,
            "allocated_component": needed / W1, "allocations": allocations}


def _stratumwise_equality(baseline: Period, comparison: Period,
                          groups: Sequence[str],
                          names: Sequence[str]) -> Dict[str, Any]:
    """Check whether one identical share vector is feasible in every stratum.

    In a stratum, a period's feasible share vector is
    ``p_g >= mapped_g / weight`` and ``sum_g p_g <= 1`` (residual mass may stay
    outside the mapped classes).  Two periods can share a vector exactly when
    the sum of the larger lower bounds does not exceed one.  This joint test is
    stronger than checking each class interval separately, which can report a
    false positive for three or more classes.
    """
    common, baseline_only, comparison_only = _split(baseline, comparison)
    rows = []
    all_feasible = bool(common)
    by_class: Dict[str, Dict[str, Any]] = {
        group: {"marginal_feasible": bool(common), "marginal_blocking": [],
                "marginal_gaps": {}}
        for group in groups
    }
    for key in common:
        cell_a = baseline.cells[key]
        cell_b = comparison.cells[key]
        label = stratum_label(key, names)
        lows = {}
        entry_classes = {}
        for group in groups:
            low = max(cell_a.low(group), cell_b.low(group))
            lows[group] = low
            a_low, a_high = cell_a.low(group), cell_a.high(group)
            b_low, b_high = cell_b.low(group), cell_b.high(group)
            feasible = max(a_low, b_low) <= min(a_high, b_high) + 1e-12
            entry_classes[group] = {
                "baseline_bounds": [a_low, a_high],
                "comparison_bounds": [b_low, b_high],
                "common_rate_interval": (
                    [max(a_low, b_low), min(a_high, b_high)]
                    if feasible else None),
                "marginal_feasible": feasible,
            }
            if not feasible:
                by_class[group]["marginal_feasible"] = False
                by_class[group]["marginal_blocking"].append(label)
                by_class[group]["marginal_gaps"][label] = (
                    b_low - a_high if b_low > a_high else a_low - b_high)
        required = math.fsum(lows.values())
        feasible = required <= 1.0 + 1e-12
        if not feasible:
            all_feasible = False
        rows.append({
            "stratum": label,
            "classes": entry_classes,
            "required_minimum_total": required,
            "joint_excess": max(0.0, required - 1.0),
            "identical_rates_feasible": feasible,
        })
    # Exact per-class condition: hold this class share fixed and let the
    # remaining classes absorb the rest.  Other classes must still respect
    # their own lower bounds, which caps this class's feasible upper endpoint.
    by_class_exact: Dict[str, Dict[str, Any]] = {}
    for group in groups:
        exact_rows = []
        feasible_all = bool(common)
        for key in common:
            cell_a = baseline.cells[key]
            cell_b = comparison.cells[key]
            label = stratum_label(key, names)
            low = max(cell_a.low(group), cell_b.low(group))
            high = min(cell_a.high(group), cell_b.high(group),
                       1.0 - math.fsum(
                           max(cell_a.low(other), cell_b.low(other))
                           for other in groups if other != group
                           if cell_a.low(other) > 0 or cell_b.low(other) > 0))
            ok = low <= high + 1e-12
            feasible_all = feasible_all and ok
            exact_rows.append({
                "stratum": label,
                "common_rate_interval": [low, high] if ok else None,
                "feasible": ok,
            })
        by_class_exact[group] = {
            "all_common_strata_feasible": feasible_all,
            "strata": exact_rows,
        }

    result = {
        "strata": rows,
        "all_common_strata_feasible": all_feasible,
        "common_strata": len(common),
        "by_class": by_class_exact,
        "baseline_only_strata": [
            {"stratum": stratum_label(key, names),
             "comparison_rate_bounds": [0.0, 1.0]} for key in baseline_only],
        "comparison_only_strata": [
            {"stratum": stratum_label(key, names),
             "baseline_rate_bounds": [0.0, 1.0]} for key in comparison_only],
        "note": (
            "a stratum is consistent with unchanged within-stratum shares "
            "exactly when the sum over classes of max(baseline lower bound, "
            "comparison lower bound) is at most 1; one-sided strata cannot be "
            "checked and remain explicitly unidentified"),
    }
    return result


def _explain(baseline: Period, comparison: Period, groups: Sequence[str],
             standardized: Mapping[str, Any], fixed: Mapping[str, Any],
             material_threshold: float,
             stratumwise: Mapping[str, Any]) -> Dict[str, Any]:
    result = {}
    for group in groups:
        b_low, b_high = comparison.bounds(group)
        f_low, f_high = fixed["bounds"][group]
        b_point = comparison.point(group)
        reachable, target, gap = _target_choice(
            b_low, b_high, f_low, f_high, b_point)
        identified = (
            fixed["coverage"]["comparison_weight_share_with_baseline_rate"] > 1e-12
            and (f_high - f_low) < 1.0 - 1e-9
            and (b_high - b_low) < 1.0 - 1e-9)
        classwise = stratumwise["by_class"][group]
        feasible = classwise["all_common_strata_feasible"]
        if not identified:
            verdict = "not_identifiable"
        elif feasible:
            verdict = "weight_only_change_not_ruled_out"
        else:
            verdict = "weight_only_change_ruled_out"
        comparable = [row["stratum"] for row in classwise["strata"]
                      if row["feasible"]]
        blocking = [row["stratum"] for row in classwise["strata"]
                    if not row["feasible"]]
        stratum_gaps = {}
        for row in classwise["strata"]:
            if row["feasible"]:
                continue
            stratum_key = next(
                (item for item in stratumwise["strata"]
                 if item["stratum"] == row["stratum"]), None)
            if stratum_key is not None:
                stratum_gaps[row["stratum"]] = stratum_key["joint_excess"]
        actual_change = b_point - baseline.point(group)
        std_point = standardized["primary"]["point"][group]
        std_base = standardized["primary"]["baseline_comparator"]["point"][group]
        std_change = (std_point - std_base
                      if std_point is not None and std_base is not None else None)
        entry: Dict[str, Any] = {
            "actual_baseline_point": baseline.point(group),
            "actual_comparison_point": b_point,
            "actual_change_point": actual_change,
            "actual_change_bounds": [b_low - baseline.bounds(group)[1],
                                     b_high - baseline.bounds(group)[0]],
            "actual_baseline_bounds": list(baseline.bounds(group)),
            "actual_comparison_bounds": [b_low, b_high],
            "standardized_comparison_point": std_point,
            "standardized_baseline_point": std_base,
            "standardized_change_point": std_change,
            "standardized_comparison_bounds":
                standardized["primary"]["bounds"][group],
            "material": abs(actual_change) > material_threshold,
            "fixed_rate_point": fixed["point"][group],
            "fixed_rate_bounds": [f_low, f_high],
            "weight_only_interval": (
                [max(b_low, f_low), min(b_high, f_high)]
                if reachable and feasible else None),
            "weight_only_can_reproduce_observed": reachable and feasible,
            "identified": identified,
            "strata_with_identical_rates_feasible": comparable,
            "strata_blocking_identity": blocking,
            "stratum_rate_gaps": stratum_gaps,
            "minimum_rate_component_needed": gap,
            "verdict": verdict,
            "witness": None,
        }
        if identified and feasible and target is not None:
            try:
                entry["witness"] = {
                    "target_share": target,
                    "comparison_completion": _witness_actual(
                        comparison, group, target),
                    "baseline_rate_completion": _witness_baseline_rate(
                        baseline, comparison, group, target),
                    "statement": (
                        "with these residual allocations the comparison share "
                        "equals the fixed-rate share, so a completion exists "
                        "in which every within-stratum rate is unchanged; "
                        "this shows the weights-only explanation is possible, "
                        "not that it is the only one; the allocations are for "
                        "this class and need not be jointly compatible with "
                        "the witnesses reported for other classes"),
                }
            except ComparisonError as exc:
                entry["witness"] = {"error": str(exc)}
        result[group] = entry
    return result


def _missing(baseline: Period, comparison: Period, names: Sequence[str],
             actual_a: Mapping[str, Any], actual_b: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "baseline_unassigned_weight": actual_a["unassigned_stratum_weight"],
        "comparison_unassigned_weight": actual_b["unassigned_stratum_weight"],
        "baseline_missing_class_weight": baseline.missing_class_weight,
        "comparison_missing_class_weight": comparison.missing_class_weight,
        "baseline_unknown_labels": dict(baseline.unknown_labels),
        "comparison_unknown_labels": dict(comparison.unknown_labels),
        "baseline_declared_but_unobserved": [
            label for label in baseline.declared_classes
            if label not in baseline.unknown_labels
            and not any(label in cell.mapped for cell in baseline.cells.values())
            and label not in [key for key in baseline.residual_by_label]
        ],
        "comparison_declared_but_unobserved": [
            label for label in comparison.declared_classes
            if label not in comparison.unknown_labels
            and not any(label in cell.mapped for cell in comparison.cells.values())
            and label not in [key for key in comparison.residual_by_label]
        ],
        "baseline_only_strata": [
            stratum_label(key, names) for key in baseline.cells
            if key != UNASSIGNED_STRATUM and key not in comparison.cells],
        "comparison_only_strata": [
            stratum_label(key, names) for key in comparison.cells
            if key != UNASSIGNED_STRATUM and key not in baseline.cells],
        "policy": (
            "unknown labels and missing class values are residual mass per "
            "stratum and are bounded; a stratum absent from one period is "
            "reported and its rate or weight is ranged over, never set to zero"),
    }


def compare(
    baseline: Mapping[str, Any],
    comparison: Mapping[str, Any],
    stratum_attributes: Sequence[str],
    class_attribute: str,
    mapping: ClassMapping,
    weight_attribute: Optional[str] = None,
    missing_weight: str = "error",
    baseline_name: str = "baseline",
    comparison_name: str = "comparison",
    baseline_path: Optional[str] = None,
    comparison_path: Optional[str] = None,
    material_threshold: float = 0.01,
    allow_string_class: bool = False,
) -> Dict[str, Any]:
    """Compare two in-memory ARFF objects.

    ``material_threshold`` affects only verdict wording.  All reported values
    are exact for the stated completion or bounds, not thresholded.
    """
    if material_threshold < 0:
        raise ComparisonError("material_threshold must be non-negative")
    if missing_weight not in ("error", "one", "mean"):
        raise ComparisonError("missing_weight must be error, one or mean")

    period_a = build_period(baseline, baseline_name, stratum_attributes,
                            class_attribute, mapping, weight_attribute,
                            missing_weight, baseline_path, allow_string_class)
    period_b = build_period(comparison, comparison_name, stratum_attributes,
                            class_attribute, mapping, weight_attribute,
                            missing_weight, comparison_path, allow_string_class)
    groups = list(mapping.groups)
    names = list(stratum_attributes)
    actual_a = _period_summary(period_a, groups)
    actual_b = _period_summary(period_b, groups)
    standardized = _standardized(period_a, period_b, groups, names)
    fixed = _fixed_rates(period_a, period_b, groups, names)
    decomposition = _decomposition(period_a, period_b, groups, names)
    stratumwise = _stratumwise_equality(period_a, period_b, groups, names)
    explanation = _explain(period_a, period_b, groups, standardized, fixed,
                           material_threshold, stratumwise)

    return {
        "schema": "arff-stratified-comparison/1",
        "inputs": {
            "baseline": {
                "name": baseline_name, "path": baseline_path,
                "sha256": _sha256(baseline_path) if baseline_path else None,
                "relation": period_a.relation, "rows": period_a.rows,
                "total_weight": period_a.total, "strata": len(period_a.cells),
                "zero_weight_rows": period_a.zero_weight_rows,
                "missing_weight_policy": missing_weight,
            },
            "comparison": {
                "name": comparison_name, "path": comparison_path,
                "sha256": _sha256(comparison_path) if comparison_path else None,
                "relation": period_b.relation, "rows": period_b.rows,
                "total_weight": period_b.total, "strata": len(period_b.cells),
                "zero_weight_rows": period_b.zero_weight_rows,
                "missing_weight_policy": missing_weight,
            },
            "stratum_attributes": names,
            "class_attribute": class_attribute,
            "weight_attribute": weight_attribute,
            "mapping": mapping.to_obj(),
            "material_threshold": material_threshold,
        },
        "classes": groups,
        "actual": {"baseline": actual_a, "comparison": actual_b},
        "standardized": standardized,
        "fixed_rates": fixed,
        "decomposition": decomposition,
        "stratumwise_equality": stratumwise,
        "explanation": explanation,
        "missing": _missing(period_a, period_b, names, actual_a, actual_b),
        "completions": {
            "proportional": (
                "residual class mass in a stratum is split in proportion to "
                "its known mapped mass; if none is known it is split equally"),
            "actual_point": "the proportional completion of each period",
            "standardized_primary_point": (
                "common strata use the comparison proportional completion; "
                "one-sided strata are excluded and their coverage is reported"),
            "standardized_full_point": (
                "baseline-only strata receive the pooled comparison rate "
                "(a named completion)"),
            "fixed_rate_point": (
                "comparison-only strata receive the pooled baseline rate "
                "(a named completion)"),
            "bounds": (
                "per-class marginal ranges; unknown mass may move to at most "
                "one class at a time, so bounds for different classes cannot "
                "generally be attained simultaneously"),
            "witness": (
                "allocates only the residual mass needed to move the observed "
                "comparison share to the fixed-rate value; all other residual "
                "mass stays unassigned and is reported"),
        },
    }


def compare_files(
    baseline_path: str,
    comparison_path: str,
    stratum_attributes: Sequence[str],
    class_attribute: str,
    mapping: Optional[ClassMapping] = None,
    weight_attribute: Optional[str] = None,
    missing_weight: str = "error",
    baseline_name: Optional[str] = None,
    comparison_name: Optional[str] = None,
    material_threshold: float = 0.01,
    allow_string_class: bool = False,
) -> Dict[str, Any]:
    baseline = load_dataset(baseline_path)
    comparison = load_dataset(comparison_path)
    if mapping is None:
        labels: List[str] = []
        for dataset in (baseline, comparison):
            labels.extend(_nominal_values(dataset, class_attribute) or [])
        mapping = ClassMapping.identity(labels)
    return compare(
        baseline, comparison, stratum_attributes, class_attribute, mapping,
        weight_attribute=weight_attribute,
        missing_weight=missing_weight,
        baseline_name=baseline_name or os.path.basename(baseline_path),
        comparison_name=comparison_name or os.path.basename(comparison_path),
        baseline_path=baseline_path, comparison_path=comparison_path,
        material_threshold=material_threshold,
        allow_string_class=allow_string_class)
