#!/usr/bin/python
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
"""
Logical diff of two Windows Registry hives.

Two versions of the same hive may be physically reorganized (cells
compacted or rewritten at new offsets), so cell addresses can never be
used to decide whether two records are "the same" configuration item.
This module instead matches keys and values by their logical, case
insensitive path and reports:

 - keys added / removed
 - values added / removed
 - value type changes
 - value data changes

Every reported item keeps the physical provenance (the NK/VK record
offset) of each hive version it came from.

Paths that are ambiguous -- duplicated names, or names that differ only
in case within the same parent -- are never paired arbitrarily.  They
are reported as conflicts and excluded from matching.
"""
from __future__ import print_function
from __future__ import unicode_literals


class ValueSnapshot(object):
    """
    A logical view of a single registry value, keeping the physical
    offset of the VK record it was read from.
    """
    def __init__(self, name, value_type, value_type_str, data, offset):
        self.name = name
        self.value_type = value_type
        self.value_type_str = value_type_str
        self.data = data
        self.offset = offset

    def display_name(self):
        return self.name if self.name != "" else "(default)"

    def __repr__(self):
        return 'ValueSnapshot(name="{0}", type="{1}", offset=0x{2:x})'.format(
            self.name, self.value_type_str, self.offset)


class KeySnapshot(object):
    """
    A logical view of a single registry key, keeping the physical
    offset of the NK record it was read from.
    """
    def __init__(self, path, name, offset, timestamp):
        self.path = path
        self.name = name
        self.offset = offset
        self.timestamp = timestamp
        # canonical (case insensitive) value name -> [ValueSnapshot, ...]
        self.values = {}

    def __repr__(self):
        return 'KeySnapshot(path="{0}", offset=0x{1:x})'.format(self.path, self.offset)


class HiveSnapshot(object):
    """
    A logical index of one hive version.

    `keys` maps a canonical (case insensitive, root-relative) key path
    to the list of keys found there.  A well formed hive has exactly
    one key per path; more than one means duplicate or case-conflicting
    names, which must not be paired arbitrarily during a diff.
    """
    def __init__(self, hive_name):
        self.hive_name = hive_name
        self.keys = {}

    def add_key(self, canonical_path, key):
        self.keys.setdefault(canonical_path, []).append(key)


def _canonical(name):
    return name.lower()


def snapshot_registry(registry):
    """
    Walk a Registry.Registry and build a HiveSnapshot.

    The canonical path is relative to the root key, so a difference in
    the root key (hive) name alone does not turn every key into a
    delete/recreate pair.
    """
    snapshot = HiveSnapshot(registry.hive_name())

    def walk(key, canonical_path, display_path):
        snap = KeySnapshot(display_path, key.name(),
                           key._nkrecord.offset(), key.timestamp())
        for value in key.values():
            try:
                data = value.raw_data()
            except Exception:
                data = None
            vsnap = ValueSnapshot(value.name(), value.value_type(),
                                  value.value_type_str(), data,
                                  value._vkrecord.offset())
            snap.values.setdefault(_canonical(vsnap.name), []).append(vsnap)
        snapshot.add_key(canonical_path, snap)
        for subkey in key.subkeys():
            sub_display = display_path + "\\" + subkey.name() if display_path else subkey.name()
            sub_canonical = canonical_path + "\\" + _canonical(subkey.name()) if canonical_path else _canonical(subkey.name())
            walk(subkey, sub_canonical, sub_display)

    walk(registry.root(), "", registry.root().name())
    return snapshot


class KeyConflict(object):
    """
    A canonical key path that could not be paired unambiguously.
    """
    def __init__(self, canonical_path, old_keys, new_keys):
        self.canonical_path = canonical_path
        self.old_keys = old_keys
        self.new_keys = new_keys


class ValueConflict(object):
    """
    A canonical value name, within one matched key, that could not be
    paired unambiguously.
    """
    def __init__(self, key_path, canonical_name, old_values, new_values):
        self.key_path = key_path
        self.canonical_name = canonical_name
        self.old_values = old_values
        self.new_values = new_values


class ValueChange(object):
    """
    A value that exists in both versions but whose type or data changed.
    """
    def __init__(self, key_path, old_value, new_value):
        self.key_path = key_path
        self.old_value = old_value
        self.new_value = new_value


class HiveDiff(object):
    def __init__(self, old_snapshot, new_snapshot):
        self.old_snapshot = old_snapshot
        self.new_snapshot = new_snapshot
        self.keys_removed = []       # [KeySnapshot] (from the old hive)
        self.keys_added = []         # [KeySnapshot] (from the new hive)
        self.keys_matched = []       # [(old KeySnapshot, new KeySnapshot)]
        self.key_conflicts = []      # [KeyConflict]
        self.values_removed = []     # [(key path, ValueSnapshot)]
        self.values_added = []       # [(key path, ValueSnapshot)]
        self.values_type_changed = []  # [ValueChange]
        self.values_data_changed = []  # [ValueChange]
        self.value_conflicts = []    # [ValueConflict]

    def has_changes(self):
        return any([
            self.keys_removed, self.keys_added, self.key_conflicts,
            self.values_removed, self.values_added,
            self.values_type_changed, self.values_data_changed,
            self.value_conflicts,
        ])


def _pair_up(old_map, new_map):
    """
    Pair entries of two canonical-name maps.

    Returns (matched, only_old, only_new, conflicts) where matched is a
    list of (old, new) tuples.  An entry is only paired when exactly one
    candidate exists on each side; anything else is a conflict and is
    never paired arbitrarily.
    """
    matched = []
    only_old = []
    only_new = []
    conflicts = []
    for canonical in sorted(set(old_map) | set(new_map)):
        olds = old_map.get(canonical, [])
        news = new_map.get(canonical, [])
        if len(olds) > 1 or len(news) > 1:
            conflicts.append((canonical, olds, news))
        elif len(olds) == 1 and len(news) == 1:
            matched.append((olds[0], news[0]))
        elif len(olds) == 1:
            only_old.append(olds[0])
        else:
            only_new.append(news[0])
    return matched, only_old, only_new, conflicts


def diff_snapshots(old_snapshot, new_snapshot):
    """
    Compute the logical diff between two HiveSnapshots.
    """
    diff = HiveDiff(old_snapshot, new_snapshot)

    matched, removed, added, conflicts = _pair_up(old_snapshot.keys,
                                                  new_snapshot.keys)
    diff.keys_matched = matched
    diff.keys_removed = removed
    diff.keys_added = added
    diff.key_conflicts = [KeyConflict(c, o, n) for (c, o, n) in conflicts]

    for old_key, new_key in matched:
        key_path = new_key.path
        v_matched, v_removed, v_added, v_conflicts = _pair_up(old_key.values,
                                                              new_key.values)
        diff.values_removed.extend((key_path, v) for v in v_removed)
        diff.values_added.extend((key_path, v) for v in v_added)
        diff.value_conflicts.extend(
            ValueConflict(key_path, c, o, n) for (c, o, n) in v_conflicts)
        for old_value, new_value in v_matched:
            if old_value.value_type != new_value.value_type:
                diff.values_type_changed.append(
                    ValueChange(key_path, old_value, new_value))
            elif old_value.data != new_value.data:
                diff.values_data_changed.append(
                    ValueChange(key_path, old_value, new_value))
    return diff


def diff_registries(old_registry, new_registry):
    """
    Compute the logical diff between two Registry.Registry instances.
    """
    return diff_snapshots(snapshot_registry(old_registry),
                          snapshot_registry(new_registry))


def _offset(offset):
    return "0x%x" % offset


def format_report(diff):
    """
    Render a HiveDiff as a human readable text report.
    """
    lines = []
    lines.append("Hive diff: {0} -> {1}".format(diff.old_snapshot.hive_name,
                                               diff.new_snapshot.hive_name))
    lines.append("")
    lines.append("== Summary ==")
    lines.append("keys:   {0} matched, {1} added, {2} removed, {3} conflicted".format(
        len(diff.keys_matched), len(diff.keys_added),
        len(diff.keys_removed), len(diff.key_conflicts)))
    lines.append("values: {0} added, {1} removed, {2} type changed, "
                 "{3} data changed, {4} conflicted".format(
                     len(diff.values_added), len(diff.values_removed),
                     len(diff.values_type_changed),
                     len(diff.values_data_changed),
                     len(diff.value_conflicts)))

    lines.append("")
    lines.append("== Keys removed ==")
    for key in diff.keys_removed:
        lines.append("[-] {0}  (old nk @ {1})".format(
            key.path or key.name, _offset(key.offset)))

    lines.append("")
    lines.append("== Keys added ==")
    for key in diff.keys_added:
        lines.append("[+] {0}  (new nk @ {1})".format(
            key.path or key.name, _offset(key.offset)))

    lines.append("")
    lines.append("== Key path conflicts (not paired) ==")
    for conflict in diff.key_conflicts:
        lines.append("[!] {0}".format(conflict.canonical_path))
        for key in conflict.old_keys:
            lines.append("      old: {0}  (nk @ {1})".format(
                key.path or key.name, _offset(key.offset)))
        for key in conflict.new_keys:
            lines.append("      new: {0}  (nk @ {1})".format(
                key.path or key.name, _offset(key.offset)))

    lines.append("")
    lines.append("== Values removed ==")
    for key_path, value in diff.values_removed:
        lines.append("[-] {0} :: {1} ({2})  (old vk @ {3})".format(
            key_path, value.display_name(), value.value_type_str,
            _offset(value.offset)))

    lines.append("")
    lines.append("== Values added ==")
    for key_path, value in diff.values_added:
        lines.append("[+] {0} :: {1} ({2})  (new vk @ {3})".format(
            key_path, value.display_name(), value.value_type_str,
            _offset(value.offset)))

    lines.append("")
    lines.append("== Value type changes ==")
    for change in diff.values_type_changed:
        lines.append("[~] {0} :: {1}  {2} -> {3}  (old vk @ {4}, new vk @ {5})".format(
            change.key_path, change.new_value.display_name(),
            change.old_value.value_type_str, change.new_value.value_type_str,
            _offset(change.old_value.offset), _offset(change.new_value.offset)))

    lines.append("")
    lines.append("== Value data changes ==")
    for change in diff.values_data_changed:
        lines.append("[~] {0} :: {1} ({2})  (old vk @ {3}, new vk @ {4})".format(
            change.key_path, change.new_value.display_name(),
            change.new_value.value_type_str,
            _offset(change.old_value.offset), _offset(change.new_value.offset)))

    lines.append("")
    lines.append("== Value name conflicts (not paired) ==")
    for conflict in diff.value_conflicts:
        lines.append("[!] {0} :: {1}".format(conflict.key_path,
                                             conflict.canonical_name))
        for value in conflict.old_values:
            lines.append("      old: {0} ({1})  (vk @ {2})".format(
                value.display_name(), value.value_type_str, _offset(value.offset)))
        for value in conflict.new_values:
            lines.append("      new: {0} ({1})  (vk @ {2})".format(
                value.display_name(), value.value_type_str, _offset(value.offset)))

    return "\n".join(lines)
