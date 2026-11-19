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
#
# Installation-history auditing across two offline registry hives.
#
# The caller supplies a rule set describing a product: its product codes
# (GUIDs), uninstall keys, component keys and the support/exclusion
# relationships between versions.  This module never treats a stale path
# string as proof of installation: path-bearing values are weak evidence and
# are reported separately as inactive residue.  When the evidence supports
# two competing interpretations, the report names the exact artifact that
# would discriminate between them.

from __future__ import print_function
from __future__ import unicode_literals

import re

from Registry import Registry
from Registry.RegistryParse import UnknownTypeException


# Verdict constants.  The Chinese labels are only used by the text report.
PROVABLE = "provable"
CANDIDATE = "candidate"
CONFLICT = "conflict"
RESIDUE = "residue"
NONE = "none"

VERDICT_LABELS = {
    PROVABLE: "可证",      # 可证
    CANDIDATE: "候选",      # 候选
    CONFLICT: "冲突",       # 冲突
    RESIDUE: "空闲残留",  # 空闲残留
    NONE: "无证据",      # 无证据
}

GUID_RE = re.compile(r"^\{?[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                     r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\}?$")
VERSION_RE = re.compile(r"\d+(?:\.\d+)*")


def normalize_guid(guid):
    """Lower-case a GUID and ensure surrounding braces for comparison."""
    if guid is None:
        return None
    text = guid.strip().lower()
    if not text:
        return None
    if not text.startswith("{"):
        text = "{" + text
    if not text.endswith("}"):
        text = text + "}"
    return text


def is_guid(text):
    return bool(text) and bool(GUID_RE.match(text.strip()))


def parse_version(text):
    """Extract the leading dotted-numeric version as a tuple of ints."""
    if text is None:
        return None
    match = VERSION_RE.search(str(text))
    if not match:
        return None
    return tuple(int(part) for part in match.group(0).split("."))


def _single_constraint(observed_tuple, constraint):
    for operator in (">=", "<=", "!=", ">", "<", "="):
        if constraint.startswith(operator):
            wanted = parse_version(constraint[len(operator):])
            if wanted is None:
                return False
            if operator == "=":
                return observed_tuple == wanted
            if operator == "!=":
                return observed_tuple != wanted
            if operator == ">=":
                return observed_tuple >= wanted
            if operator == "<=":
                return observed_tuple <= wanted
            if operator == ">":
                return observed_tuple > wanted
            if operator == "<":
                return observed_tuple < wanted
    wanted = parse_version(constraint)
    return wanted is not None and observed_tuple == wanted


def version_satisfies(observed, constraint):
    """
    Compare an observed version string against a rule constraint.
    Constraints: '=1.0', '>=1.0', '<=2.5', '>1.0', '<2.0' or a bare
    version string (exact match).  Comma-separated constraints are ANDed,
    e.g. '>=1.0,<2.0'.
    """
    observed_tuple = parse_version(observed)
    if observed_tuple is None:
        return False
    return all(_single_constraint(observed_tuple, part.strip())
               for part in constraint.split(","))


class Evidence(object):
    """A single observation in one hive, tied to a concrete key/value."""

    def __init__(self, kind, hive_label, key_path, value_name=None,
                 data=None, key_timestamp=None, note=None):
        self.kind = kind              # uninstall_key|product_code|component_key|version|path
        self.hive_label = hive_label
        self.key_path = key_path
        self.value_name = value_name
        self.data = data
        self.key_timestamp = key_timestamp
        self.note = note

    def location(self):
        if self.value_name:
            return "%s!%s" % (self.key_path, self.value_name)
        return self.key_path

    def describe(self):
        base = "[%s] %s (%s)" % (self.hive_label, self.location(), self.kind)
        if self.data is not None:
            base += " = %r" % (self.data,)
        if self.note:
            base += " // %s" % self.note
        return base


class Conflict(object):
    def __init__(self, reason, evidence, rule_id=None):
        self.reason = reason
        self.evidence = evidence      # list[Evidence]
        self.rule_id = rule_id

    def describe(self):
        lines = [self.reason]
        for item in self.evidence:
            lines.append("    - " + item.describe())
        return "\n".join(lines)


class Verdict(object):
    """Per-product, per-hive classification with full provenance."""

    def __init__(self, product, hive_label, status):
        self.product = product
        self.hive_label = hive_label
        self.status = status
        self.versions = []            # provable/candidate version strings
        self.evidence = []            # Evidence used
        self.conflicts = []           # Conflict
        self.missing = []             # human-readable gaps
        self.discriminators = []      # artifacts that would resolve ambiguity

    def describe(self):
        return "%s/%s: %s %s" % (
            self.product, self.hive_label,
            VERDICT_LABELS[self.status], self.versions)


class HiveView(object):
    """Abstract read-only hive access used by the auditor."""

    def exists(self, path):
        raise NotImplementedError

    def key_timestamp(self, path):
        raise NotImplementedError

    def subkeys(self, path):
        raise NotImplementedError

    def value(self, path, name):
        """Return (type, data) or None. Unreadable data returns a marker."""
        raise NotImplementedError


class RegistryHiveView(HiveView):
    """HiveView backed by a python-registry Registry object."""

    UNREADABLE = "<unreadable value data>"

    def __init__(self, registry, label):
        self._registry = registry
        self.label = label

    def _open(self, path):
        try:
            return self._registry.open(path)
        except Registry.RegistryKeyNotFoundException:
            return None

    def exists(self, path):
        return self._open(path) is not None

    def key_timestamp(self, path):
        key = self._open(path)
        return str(key.timestamp()) if key is not None else None

    def subkeys(self, path):
        key = self._open(path)
        if key is None:
            return []
        return [sub.path() for sub in key.subkeys()]

    def value(self, path, name):
        key = self._open(path)
        if key is None:
            return None
        try:
            reg_value = key.value(name)
        except Registry.RegistryValueNotFoundException:
            return None
        try:
            data = reg_value.value()
        except UnknownTypeException:
            return reg_value.value_type(), self.UNREADABLE
        return reg_value.value_type(), data


class InMemoryHiveView(HiveView):
    """Dict-backed hive for unit tests.

    layout: {path: {"timestamp": str, "values": {name: (type, data)}}}
    """

    def __init__(self, layout, label):
        self.label = label
        self._layout = layout

    def exists(self, path):
        return path.lower() in self._layout

    def key_timestamp(self, path):
        entry = self._layout.get(path.lower())
        return entry.get("timestamp") if entry else None

    def subkeys(self, path):
        prefix = path.lower().rstrip("\\") + "\\"
        found = []
        for key_path in self._layout:
            if key_path.startswith(prefix) and "\\" not in key_path[len(prefix):]:
                found.append(key_path)
        return found

    def value(self, path, name):
        entry = self._layout.get(path.lower())
        if not entry:
            return None
        return entry.get("values", {}).get(name)


class ProductRule(object):
    def __init__(self, raw):
        self.name = raw["name"]
        self.product_codes = [normalize_guid(g) for g in raw.get("product_codes", [])]
        self.uninstall_keys = list(raw.get("uninstall_keys", []))
        self.uninstall_roots = list(raw.get("uninstall_roots", []))
        self.component_keys = [normalize_guid(g) if is_guid(g) else g
                               for g in raw.get("component_keys", [])]
        self.component_roots = list(raw.get("component_roots", []))
        self.version_values = list(raw.get("version_values", ["DisplayVersion"]))
        self.path_values = list(raw.get("path_values",
                                        ["InstallLocation", "DisplayIcon",
                                         "UninstallString"]))
        # Suspect locations that may only contain stale path strings. Values
        # found here are collected as weak path evidence exclusively.
        self.path_keys = list(raw.get("path_keys", []))
        # Explicit keys outside Uninstall that may carry a ProductCode value
        # (for example per-product registration keys).
        self.product_code_keys = list(raw.get("product_code_keys", []))
        self.requires = list(raw.get("requires", []))
        self.excludes = list(raw.get("excludes", []))
        self.versions = raw.get("versions", [])

    def owns_guid(self, guid):
        return normalize_guid(guid) in self.product_codes

    def owns_component(self, key_name):
        normalized = normalize_guid(key_name) if is_guid(key_name) else key_name.lower()
        return normalized in [c.lower() for c in self.component_keys]


class RuleSet(object):
    """Parsed caller rule set: products plus global exclusion rules."""

    def __init__(self, raw):
        self.products = [ProductRule(item) for item in raw.get("products", [])]
        self.global_exclusions = list(raw.get("exclusions", []))

    @classmethod
    def from_dict(cls, raw):
        return cls(raw)

    @classmethod
    def from_file(cls, path):
        import json
        with open(path, "rb") as handle:
            return cls(json.loads(handle.read().decode("utf-8")))


class InstallAuditor(object):
    """
    Correlate installation evidence for a rule set across two offline hives.
    """

    def __init__(self, rule_set, hive_a, hive_b):
        self.rules = rule_set
        self.hives = [hive_a, hive_b]
        self.hive_labels = [hive_a.label, hive_b.label]

    # ------------------------------------------------------------------ #
    # Evidence collection
    # ------------------------------------------------------------------ #
    def _collect(self, product, view):
        found = {"uninstall_key": [], "product_code": [], "component_key": [],
                 "version": [], "path": []}
        seen = set()

        def add(kind, key_path, value_name=None, data=None, note=None):
            dedupe_key = (kind, key_path.lower(),
                          value_name.lower() if value_name else None)
            if dedupe_key in seen:
                return
            seen.add(dedupe_key)
            evidence = Evidence(kind, view.label, key_path, value_name, data,
                                view.key_timestamp(key_path), note)
            found[kind].append(evidence)

        for key_path in product.uninstall_keys:
            if view.exists(key_path):
                add("uninstall_key", key_path)
                self._collect_key_values(product, view, key_path, add)

        for root in product.uninstall_roots:
            for child in view.subkeys(root):
                leaf = child.rstrip("\\").split("\\")[-1]
                if is_guid(leaf) and product.owns_guid(leaf):
                    add("uninstall_key", child,
                        note="GUID-named subkey discovered under %s" % root)
                    self._collect_key_values(product, view, child, add)

        for component in product.component_keys:
            for candidate in self._resolve_component(product, view, component):
                add("component_key", candidate)

        for root in product.component_roots:
            for child in view.subkeys(root):
                leaf = child.rstrip("\\").split("\\")[-1]
                if product.owns_component(leaf):
                    add("component_key", child,
                        note="component discovered under %s" % root)

        for key_path in product.path_keys:
            if view.exists(key_path):
                for value_name in product.path_values:
                    result = view.value(key_path, value_name)
                    if result is not None:
                        add("path", key_path, value_name, result[1],
                            note="path-only key: weak evidence, never an "
                                 "install proof")

        for key_path in product.product_code_keys:
            if view.exists(key_path):
                result = view.value(key_path, "ProductCode")
                if result is not None and product.owns_guid(str(result[1])):
                    add("product_code", key_path, "ProductCode", result[1],
                        note="product code outside an uninstall key; still "
                             "needs uninstall-key corroboration")
                for value_name in product.version_values:
                    result = view.value(key_path, value_name)
                    if result is not None:
                        add("version", key_path, value_name, result[1])

        return found

    def _resolve_component(self, product, view, component):
        # Explicit component entries may be full paths or bare GUIDs that a
        # component root would normally anchor.
        if view.exists(component):
            return [component]
        matches = []
        for root in product.component_roots:
            for child in view.subkeys(root):
                leaf = child.rstrip("\\").split("\\")[-1]
                if leaf.lower() == component.lower():
                    matches.append(child)
        return matches

    def _collect_key_values(self, product, view, key_path, add):
        for value_name in product.version_values:
            result = view.value(key_path, value_name)
            if result is not None:
                add("version", key_path, value_name, result[1])
        for value_name in product.path_values:
            result = view.value(key_path, value_name)
            if result is not None:
                add("path", key_path, value_name, result[1],
                    note="path-bearing value is weak evidence")
        result = view.value(key_path, "ProductCode")
        if result is not None and product.owns_guid(str(result[1])):
            add("product_code", key_path, "ProductCode", result[1])
        leaf = key_path.rstrip("\\").split("\\")[-1]
        if is_guid(leaf) and product.owns_guid(leaf):
            add("product_code", key_path, None, leaf,
                note="uninstall key name is a registered product code")

    # ------------------------------------------------------------------ #
    # Rule predicates
    # ------------------------------------------------------------------ #
    def _predicate_hit(self, predicate, view, evidence_bag):
        """
        Predicate forms:
          "key:SOFTWARE\\...", "value:path|name[=data]",
          "product:{GUID}", "component:{GUID-or-path}",
          or dicts with explicit type/path/name/data fields.
        """
        if isinstance(predicate, dict):
            kind = predicate.get("type", "key")
            target = predicate.get("path", "")
            name = predicate.get("name")
            expected = predicate.get("data")
        else:
            kind, _, target = predicate.partition(":")
            name = expected = None
            if kind == "value" and "|" in target:
                target, _, rest = target.partition("|")
                if "=" in rest:
                    name, _, expected = rest.partition("=")
                else:
                    name = rest
        if kind == "key":
            return view.exists(target), target
        if kind == "value":
            result = view.value(target, name) if name else None
            if result is None:
                return False, "%s!%s" % (target, name)
            if expected is not None and str(result[1]).lower() != expected.lower():
                return False, "%s!%s=%r (want %s)" % (target, name, result[1], expected)
            return True, "%s!%s" % (target, name)
        if kind == "product":
            hit = any(normalize_guid(target) == normalize_guid(e.data)
                      for e in evidence_bag["product_code"])
            return hit, "product code %s" % target
        if kind == "component":
            target_norm = normalize_guid(target) if is_guid(target) else target.lower()
            hit = any((normalize_guid(e.key_path.split("\\")[-1])
                       if is_guid(e.key_path.split("\\")[-1])
                       else e.key_path.lower()) == target_norm
                      for e in evidence_bag["component_key"])
            return hit, "component %s" % target
        return False, "unknown predicate %r" % predicate

    def _check_rules(self, clauses, view, evidence_bag):
        hits, failures = [], []
        for clause in clauses:
            ok, detail = self._predicate_hit(clause, view, evidence_bag)
            if ok:
                hits.append(detail)
            else:
                failures.append(detail)
        return hits, failures

    # ------------------------------------------------------------------ #
    # Classification
    # ------------------------------------------------------------------ #
    def audit_product(self, product, view):
        bag = self._collect(product, view)
        all_evidence = [item for group in bag.values() for item in group]
        verdict = Verdict(product.name, view.label, NONE)
        verdict.evidence = all_evidence

        strong_keys = bag["uninstall_key"]
        product_codes = bag["product_code"]
        components = bag["component_key"]
        versions = bag["version"]
        paths = bag["path"]

        _, require_fail = self._check_rules(product.requires, view, bag)
        exclude_hits, _ = self._check_rules(product.excludes, view, bag)
        if exclude_hits:
            verdict.conflicts.append(Conflict(
                "product-level exclusion triggered: %s" % "; ".join(exclude_hits),
                all_evidence, product.name))

        observed_versions = [str(item.data) for item in versions]

        if strong_keys or product_codes:
            provable, version_failures = self._evaluate_version_rules(
                product, view, bag, observed_versions, verdict)
            complete = bool(strong_keys) and bool(product_codes)
            version_ok = not product.versions or bool(provable)
            if (complete and version_ok and not require_fail
                    and not verdict.conflicts):
                verdict.status = PROVABLE
                verdict.versions = provable or observed_versions
                self._mutual_exclusion_conflicts(product, view, bag, verdict)
            else:
                verdict.status = CONFLICT if verdict.conflicts else CANDIDATE
                verdict.versions = observed_versions or [
                    v.get("constraint", v.get("version", "?"))
                    for v in product.versions]
                if not strong_keys:
                    verdict.missing.append(
                        "uninstall key absent; expected one of: %s"
                        % ", ".join(product.uninstall_keys
                                    or [r for r in product.uninstall_roots]))
                if not product_codes:
                    verdict.missing.append(
                        "no ProductCode value / GUID-named key for %s"
                        % ", ".join(product.product_codes))
                if product.versions and not provable:
                    verdict.missing.append(
                        "observed versions %s satisfy no supported version "
                        "rule" % observed_versions)
                verdict.missing.extend(version_failures)
                if not observed_versions:
                    verdict.missing.append(
                        "no version value (%s) on uninstall keys"
                        % ", ".join(product.version_values))
                verdict.missing.extend(
                    "unsupported requirement: %s" % item for item in require_fail)
                for conflict in verdict.conflicts:
                    verdict.missing.append("conflict: " + conflict.reason)
                self._discriminators(verdict, product, strong_keys,
                                     product_codes, observed_versions)
        elif components or paths:
            verdict.status = RESIDUE
            if components and not paths:
                verdict.note = "orphan component key without product code or uninstall key"
            self._discriminators(verdict, product, strong_keys,
                                 product_codes, observed_versions)
        else:
            verdict.discriminators.append(
                "no registry evidence; to test the install hypothesis, acquire "
                "%s or MSI transaction logs / AmCache" %
                ", ".join(product.uninstall_keys))
        return verdict

    def _evaluate_version_rules(self, product, view, bag, observed_versions,
                                verdict):
        """Return (provable versions, human-readable rule failures).

        Version-level exclusions are always evaluated and recorded as
        conflicts, so an unsupported combination is visible even when the
        overall verdict is only a candidate.
        """
        if not product.versions:
            return list(observed_versions), []
        provable, failures = [], []
        for version_rule in product.versions:
            constraint = version_rule.get("constraint", version_rule.get("version"))
            label = version_rule.get("version", constraint)
            matching = [v for v in observed_versions
                        if version_satisfies(v, constraint)]
            if not matching:
                continue
            _, require_fail = self._check_rules(
                version_rule.get("requires", []), view, bag)
            excl_hits, _ = self._check_rules(
                version_rule.get("excludes", []), view, bag)
            for hit in excl_hits:
                failures.append("version %s exclusion hit: %s" % (label, hit))
                verdict.conflicts.append(Conflict(
                    "version %s exclusion: %s" % (label, hit),
                    verdict.evidence, product.name))
            for failure in require_fail:
                failures.append("version %s unsupported requirement: %s"
                                % (label, failure))
            if not require_fail and not excl_hits:
                provable.extend(matching)
        return sorted(set(provable)), failures

    def _mutual_exclusion_conflicts(self, product, view, bag, verdict):
        # Version-level exclusions are evaluated in _evaluate_version_rules.
        # Here we only flag several coexisting versions when the rules never
        # explicitly allow that combination.
        if len(verdict.versions) > 1 and product.versions and not any(
                rule.get("allows_coexist") for rule in product.versions):
            verdict.conflicts.append(Conflict(
                "multiple versions provable without allows_coexist: %s"
                % verdict.versions, verdict.evidence, product.name))

    def _discriminators(self, verdict, product, strong_keys, product_codes,
                        observed_versions):
        if not strong_keys:
            verdict.discriminators.append(
                "acquire uninstall key %s -> its presence supports an actual "
                "install, its absence supports never-installed residue"
                % ", ".join(product.uninstall_keys))
        if not product_codes:
            verdict.discriminators.append(
                "acquire ProductCode value / GUID key for %s -> distinguishes "
                "real MSI registration from copied path strings"
                % ", ".join(product.product_codes))
        if not observed_versions:
            verdict.discriminators.append(
                "read %s on the uninstall key to pin the version instead of "
                "inferring it from a path" % ", ".join(product.version_values))
        verdict.discriminators.append(
            "outside the hive: MSI transaction logs, AmCache.hve and $MFT "
            "timestamps discriminate installed-then-removed from never-installed")

    # ------------------------------------------------------------------ #
    # Global exclusions and cross-hive correlation
    # ------------------------------------------------------------------ #
    def _global_conflicts(self, view):
        conflicts = []
        for rule in self.rules.global_exclusions:
            paths = rule.get("paths", [])
            reason = rule.get("reason", "global exclusion")
            existing = [path for path in paths if view.exists(path)]
            if len(existing) == len(paths) and paths:
                evidence = [Evidence("key", view.label, path)
                            for path in existing]
                conflicts.append(Conflict(reason, evidence))
        return conflicts

    def run(self):
        report = AuditReport(self.rules, self.hive_labels)
        for view in self.hives:
            for product in self.rules.products:
                report.add_verdict(self.audit_product(product, view))
            report.add_global_conflicts(view.label, self._global_conflicts(view))
        report.correlate()
        return report


class AuditReport(object):
    def __init__(self, rule_set, hive_labels):
        self.rules = rule_set
        self.hive_labels = hive_labels
        self.verdicts = []
        self.global_conflicts = {}

    def add_verdict(self, verdict):
        self.verdicts.append(verdict)

    def add_global_conflicts(self, hive_label, conflicts):
        self.global_conflicts.setdefault(hive_label, []).extend(conflicts)

    def correlate(self):
        """Cross-hive timeline notes; verdicts themselves stay per-hive."""
        self.timeline = []
        for product in self.rules.products:
            per_hive = {}
            for verdict in self.verdicts:
                if verdict.product == product.name:
                    per_hive.setdefault(verdict.hive_label, verdict)
            labels = list(per_hive)
            if len(labels) < 2:
                continue
            earlier = per_hive[labels[0]]
            later = per_hive[labels[1]]
            if earlier.status == PROVABLE and later.status == RESIDUE:
                self.timeline.append(
                    "%s: provable in %s, only inactive residue remains in %s "
                    "-> installed then removed (residue is not activity)"
                    % (product.name, labels[0], labels[1]))
            elif earlier.status == RESIDUE and later.status == PROVABLE:
                self.timeline.append(
                    "%s: residue in %s, provable in %s -> first hive post-dates "
                    "an uninstall, second hive shows a fresh install"
                    % (product.name, labels[0], labels[1]))
            elif earlier.status != later.status:
                self.timeline.append(
                    "%s: status differs across hives (%s=%s, %s=%s)"
                    % (product.name, labels[0], earlier.status,
                       labels[1], later.status))

    # ------------------------------------------------------------------ #
    # Text rendering
    # ------------------------------------------------------------------ #
    def to_dict(self):
        def evidence_dict(item):
            return {
                "kind": item.kind,
                "hive": item.hive_label,
                "key_path": item.key_path,
                "value_name": item.value_name,
                "data": item.data,
                "key_timestamp": item.key_timestamp,
                "note": item.note,
            }

        return {
            "hives": self.hive_labels,
            "verdicts": [
                {
                    "product": verdict.product,
                    "hive": verdict.hive_label,
                    "status": verdict.status,
                    "status_label": VERDICT_LABELS[verdict.status],
                    "versions": verdict.versions,
                    "note": getattr(verdict, "note", None),
                    "evidence": [evidence_dict(e) for e in verdict.evidence],
                    "conflicts": [
                        {"reason": c.reason, "rule_id": c.rule_id,
                         "evidence": [evidence_dict(e) for e in c.evidence]}
                        for c in verdict.conflicts],
                    "missing": verdict.missing,
                    "discriminators": verdict.discriminators,
                }
                for verdict in self.verdicts],
            "global_conflicts": {
                label: [{"reason": c.reason,
                         "evidence": [evidence_dict(e) for e in c.evidence]}
                        for c in conflicts]
                for label, conflicts in self.global_conflicts.items()},
            "timeline": self.timeline,
        }

    def to_text(self):
        lines = []
        lines.append("=" * 72)
        lines.append("Installation-history audit across hives: %s"
                     % " vs ".join(self.hive_labels))
        lines.append("=" * 72)

        sections = [
            (PROVABLE, "Provable versions / 可证版本"),
            (CANDIDATE, "Candidates / 候选"),
            (CONFLICT, "Conflicts / 冲突依据"),
            (RESIDUE, "Inactive residue (not activity) / 空闲残留（不当活动配置）"),
        ]
        for status, title in sections:
            lines.append("")
            lines.append("-- %s --" % title)
            scoped = [v for v in self.verdicts if v.status == status]
            if status == CONFLICT:
                scoped = [v for v in self.verdicts if v.conflicts]
            if not scoped and not (status == CONFLICT and self.global_conflicts):
                lines.append("  (none)")
                continue
            for verdict in scoped:
                lines.append("  * %s [%s] %s"
                             % (verdict.product, verdict.hive_label,
                                verdict.versions or ""))
                if getattr(verdict, "note", None):
                    lines.append("      note: " + verdict.note)
                for evidence in verdict.evidence:
                    lines.append("      evidence: " + evidence.describe())
                for conflict in verdict.conflicts:
                    for piece in conflict.describe().splitlines():
                        lines.append("      " + piece)
                for gap in verdict.missing:
                    lines.append("      gap: " + gap)
                for discriminator in verdict.discriminators:
                    lines.append("      discriminator: " + discriminator)
            if status == CONFLICT:
                for hive_label, conflicts in self.global_conflicts.items():
                    for conflict in conflicts:
                        lines.append("  * global [%s] %s"
                                     % (hive_label, conflict.reason))
                        for evidence in conflict.evidence:
                            lines.append("      evidence: " + evidence.describe())

        lines.append("")
        lines.append("-- Cross-hive correlation / 跨 hive 核对 --")
        if not self.timeline:
            lines.append("  (none)")
        for item in self.timeline:
            lines.append("  " + item)

        lines.append("")
        lines.append("-- Discrimination gaps / 缺证据时的补证项 --")
        any_gap = False
        for verdict in self.verdicts:
            if verdict.status in (CANDIDATE, CONFLICT, RESIDUE):
                any_gap = True
                lines.append("  * %s [%s] currently=%s"
                             % (verdict.product, verdict.hive_label,
                                VERDICT_LABELS[verdict.status]))
                for discriminator in verdict.discriminators:
                    lines.append("      - " + discriminator)
        if not any_gap:
            lines.append("  (none)")
        return "\n".join(lines)
