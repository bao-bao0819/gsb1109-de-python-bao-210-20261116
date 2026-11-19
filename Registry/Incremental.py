#!/usr/bin/python

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
Incremental configuration collection for Windows Registry hives.

The transaction log may recover a hive by writing a batch of dirty pages.
A collector that already has a snapshot of the hive does not need to scan
the complete hive again.  It only needs to know which value identities and
value contents are physically reachable from the dirty pages.

This module builds an index of the physical structures that make up the
logical registry configuration:

  * NK cells (keys), including parent/name/directory/value-list fields,
  * subkey lists (li/lf/lh and ri indirection),
  * value lists (pointers to VK cells),
  * VK cells (value identity and data location), and
  * every cell that holds value data, including each fragment of a large
    value stored through a DB/indirect block.

Given a set of dirty pages, the index produces a recollection plan.  The
plan is conservative about identity: a clean content page does not prove
that the logical path is still the same.  If a directory or value list
changes, the identities below it are recollected.  A pointer to a cell
that cannot be confirmed from the baseline index is reported as pending
rediscovery rather than silently guessed.

The replay API applies only the plan to a recovered hive.  Its exported
snapshot is equivalent to a full collection, and every recollected item
records the change trigger that caused it.
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import bisect
import copy
import datetime
import struct
from collections import OrderedDict, defaultdict

from . import Registry
from . import RegistryParse

PAGE_SIZE = 0x1000
FIRST_HBIN_OFFSET = 0x1000

STATUS_RECOLLECT = "recollect"
STATUS_PENDING = "pending_rediscovery"

KIND_KEY = "key"
KIND_SUBKEY_LIST = "subkey_list"
KIND_VALUE_LIST = "value_list"
KIND_VALUE = "value"
KIND_VALUE_DATA = "value_data"
KIND_CLASS = "class"

_LIST_IDS = (b"li", b"lf", b"lh", b"ri")
_NEW_STRUCTURE_IDS = (b"nk", b"vk", b"li", b"lf", b"lh", b"ri", b"db")


class RecollectError(RegistryParse.RegistryException):
    """
    Raised when a recollection plan cannot be built or replayed.
    """
    def __init__(self, value):
        super(RecollectError, self).__init__(value)

    def __str__(self):
        return "Recollect Error: %s" % (self._value)


def _aligned_pages(start, size):
    """Return the page offsets touched by [start, start + size)."""
    if size <= 0:
        return []
    first = start - (start % PAGE_SIZE)
    last = start + size - 1
    last = last - (last % PAGE_SIZE)
    return list(range(first, last + PAGE_SIZE, PAGE_SIZE))


def _intersects(start, size, intervals):
    end = start + size
    for interval_start, interval_end in intervals:
        if start < interval_end and end > interval_start:
            return True
    return False


def _changed_ranges(before, after, start, size):
    """
    Compare two byte strings over a structure extent.

    Returns a list of (absolute_start, absolute_end) ranges that differ.
    `before` and `after` may be None, in which case the complete extent is
    considered changed.
    """
    if size <= 0:
        return []
    if before is None or after is None:
        return [(start, start + size)]

    ranges = []
    range_start = None
    limit = min(size, len(before), len(after))
    for index in range(limit):
        if before[index] != after[index]:
            if range_start is None:
                range_start = index
        elif range_start is not None:
            ranges.append((start + range_start, start + index))
            range_start = None
    if range_start is not None:
        ranges.append((start + range_start, start + limit))
    if len(before) != len(after):
        tail = min(len(before), len(after))
        ranges.append((start + tail, start + max(len(before), len(after))))
    return ranges


def _field_changed(before, after, relative_offset, length):
    if before is None or after is None:
        return True
    return before[relative_offset:relative_offset + length] != \
        after[relative_offset:relative_offset + length]


def _decode_name(raw):
    if isinstance(raw, tuple):
        raw = raw[0] if raw else b""
    if raw is None:
        return None
    try:
        return raw.decode("windows-1252")
    except Exception:
        try:
            return raw.decode("utf-16le")
        except Exception:
            return None


def _json_safe(value):
    """Convert parsed registry values into deterministic JSON-safe data."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    if isinstance(value, bytearray):
        return {"__bytes__": bytes(value).hex()}
    if isinstance(value, datetime.datetime):
        return {"__datetime__": value.isoformat()}
    if isinstance(value, datetime.timedelta):
        return {"__timedelta__": value.total_seconds()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return OrderedDict((str(key), _json_safe(item))
                           for key, item in value.items())
    return repr(value)


class DirtyPageRegion(object):
    """
    One dirty page range supplied by the transaction log.

    `offset` is absolute in the hive by default.  Use
    `from_log_reference()` for offsets that are relative to the first HBIN,
    as stored in HvLE dirty page references.
    """
    def __init__(self, offset, size, data=None, sequence=None, source=None,
                 relative=False):
        self.offset = offset + FIRST_HBIN_OFFSET if relative else offset
        self.size = size
        self.data = data
        self.sequence = sequence
        self.source = source

    @classmethod
    def from_log_reference(cls, reference, data=None, sequence=None,
                           source=None):
        """
        Build a region from a RegistryParse.DirtyPageReference and its data.
        """
        return cls(reference.offset(), reference.size(), data=data,
                   sequence=sequence, source=source, relative=True)

    def pages(self):
        return _aligned_pages(self.offset, self.size)

    def to_dict(self):
        return OrderedDict([
            ("offset", self.offset),
            ("size", self.size),
            ("sequence", self.sequence),
            ("source", self.source),
            ("has_data", self.data is not None),
        ])

    def __repr__(self):
        return "DirtyPageRegion(offset=0x%x, size=0x%x, sequence=%r)" % \
            (self.offset, self.size, self.sequence)


def dirty_pages_from_log(log, source=None):
    """
    Yield DirtyPageRegion objects for every dirty page in a RegistryLog.

    The yielded regions are ordered exactly as the log entries are applied.
    """
    for entry in log._regf.log_entries():
        for reference, page in entry.dirty_pages_with_references():
            yield DirtyPageRegion.from_log_reference(
                reference, data=page.data(), sequence=entry.sequence(),
                source=source)


def dirty_pages_from_logs(logs):
    """
    Yield dirty page regions from an iterable of RegistryLog objects.
    """
    for index, log in enumerate(logs):
        for region in dirty_pages_from_log(log, source="log[%d]" % index):
            yield region


class Dependency(object):
    """
    A physical structure that participates in the logical configuration.
    """
    def __init__(self, kind, start, size, key_path=None, value_name=None,
                 record_offset=None, fields=None, fragments=None,
                 list_id=None, list_count=None, parent_path=None,
                 scope_path=None, value_type=None):
        self.kind = kind
        self.start = start
        self.size = max(size, 0)
        self.key_path = key_path
        self.value_name = value_name
        self.record_offset = record_offset
        self.fields = fields or OrderedDict()
        self.fragments = fragments or []
        self.list_id = list_id
        self.list_count = list_count
        self.parent_path = parent_path
        self.scope_path = scope_path
        self.value_type = value_type

    @property
    def end(self):
        return self.start + self.size

    @property
    def identity(self):
        if self.value_name is None:
            return (self.key_path,)
        return (self.key_path, self.value_name)

    def pages(self):
        return _aligned_pages(self.start, self.size)

    def field(self, name):
        return self.fields.get(name)

    def to_dict(self):
        return OrderedDict([
            ("kind", self.kind),
            ("start", self.start),
            ("size", self.size),
            ("key_path", self.key_path),
            ("value_name", self.value_name),
            ("list_id", self.list_id),
            ("list_count", self.list_count),
            ("fragments", [{"start": start, "size": size}
                           for start, size in self.fragments]),
        ])

    def __repr__(self):
        return "Dependency(kind=%s, start=0x%x, size=0x%x, key=%r, value=%r)" % \
            (self.kind, self.start, self.size, self.key_path, self.value_name)


class ChangeTrigger(object):
    """
    Describes the exact change that caused a recollect item.
    """
    def __init__(self, kind, page=None, structure_offset=None, fields=None,
                 pages=None, sequence=None, source=None, detail=None,
                 propagated_from=None):
        self.kind = kind
        self.page = page
        self.structure_offset = structure_offset
        self.fields = list(fields or [])
        self.pages = list(pages or ([] if page is None else [page]))
        self.sequence = sequence
        self.source = source
        self.detail = detail
        self.propagated_from = propagated_from

    def to_dict(self):
        return OrderedDict([
            ("kind", self.kind),
            ("page", self.page),
            ("pages", self.pages),
            ("structure_offset", self.structure_offset),
            ("fields", self.fields),
            ("sequence", self.sequence),
            ("source", self.source),
            ("detail", self.detail),
            ("propagated_from", self.propagated_from),
        ])

    def __str__(self):
        location = "page 0x%x" % self.page if self.page is not None else "unknown page"
        structure = " at 0x%x" % self.structure_offset \
            if self.structure_offset is not None else ""
        fields = ""
        if self.fields:
            fields = " [%s]" % ", ".join(self.fields)
        propagated = ""
        if self.propagated_from:
            propagated = " via %s" % self.propagated_from
        return "%s %s%s%s%s" % (self.kind, location, structure, fields, propagated)


class RecollectItem(object):
    """
    One identity or discovery scope that must be processed.
    """
    def __init__(self, kind, key_path=None, value_name=None, status=STATUS_RECOLLECT,
                 triggers=None, scope_path=None, parent_path=None, detail=None,
                 nk_offset=None, vk_offset=None, parent_nk_offset=None,
                 mode="value", owner_nk_offset=None):
        self.kind = kind
        self.key_path = key_path
        self.value_name = value_name
        self.status = status
        self.triggers = list(triggers or [])
        self.scope_path = scope_path
        self.parent_path = parent_path
        self.detail = detail
        self.nk_offset = nk_offset
        self.vk_offset = vk_offset
        self.parent_nk_offset = parent_nk_offset
        self.mode = mode
        self.owner_nk_offset = owner_nk_offset

    @property
    def identity(self):
        if self.value_name is None:
            return (self.key_path,)
        return (self.key_path, self.value_name)

    @property
    def pending(self):
        return self.status == STATUS_PENDING

    def add_trigger(self, trigger):
        for current in self.triggers:
            if current.to_dict() == trigger.to_dict():
                return
        self.triggers.append(trigger)

    def to_dict(self):
        return OrderedDict([
            ("kind", self.kind),
            ("key_path", self.key_path),
            ("value_name", self.value_name),
            ("status", self.status),
            ("scope_path", self.scope_path),
            ("parent_path", self.parent_path),
            ("nk_offset", self.nk_offset),
            ("vk_offset", self.vk_offset),
            ("mode", self.mode),
            ("detail", self.detail),
            ("triggers", [trigger.to_dict() for trigger in self.triggers]),
        ])

    def __str__(self):
        target = self.key_path or "(unknown key)"
        if self.value_name is not None:
            target = "%s : %s" % (target, self.value_name)
        triggers = "; ".join(str(trigger) for trigger in self.triggers)
        return "%s %s (%s)" % (self.status, target, triggers)

    def __repr__(self):
        return "RecollectItem(kind=%s, status=%s, key=%r, value=%r)" % \
            (self.kind, self.status, self.key_path, self.value_name)


class Removal(object):
    """
    A baseline key or value that is no longer present after replay.
    """
    def __init__(self, key_path, value_name=None, subtree=False, trigger=None):
        self.key_path = key_path
        self.value_name = value_name
        self.subtree = subtree
        self.trigger = trigger

    @property
    def identity(self):
        if self.value_name is None:
            return (self.key_path,)
        return (self.key_path, self.value_name)

    def to_dict(self):
        return OrderedDict([
            ("key_path", self.key_path),
            ("value_name", self.value_name),
            ("subtree", self.subtree),
            ("trigger", self.trigger.to_dict() if self.trigger else None),
        ])

    def __repr__(self):
        return "Removal(key=%r, value=%r, subtree=%r)" % \
            (self.key_path, self.value_name, self.subtree)


class RecollectPlan(object):
    """
    The result of applying a dirty page batch to a dependency index.
    """
    def __init__(self, dirty_pages=None, items=None, removals=None,
                 unknown_pages=None):
        self.dirty_pages = list(dirty_pages or [])
        self.items = list(items or [])
        self.removals = list(removals or [])
        self.unknown_pages = list(unknown_pages or [])
        self._item_index = {}
        self._removal_index = {}
        for item in self.items:
            self._item_index.setdefault(
                (item.identity, item.kind, item.status), []).append(item)
        for removal in self.removals:
            self._removal_index[removal.identity] = removal

    @property
    def pending(self):
        return [item for item in self.items if item.pending]

    @property
    def recollections(self):
        return [item for item in self.items if not item.pending]

    @property
    def is_empty(self):
        return not self.items and not self.removals

    def add_item(self, item):
        bucket = self._item_index.setdefault(
            (item.identity, item.kind, item.status), [])
        for current in bucket:
            if not item.pending:
                # Stable identities are merged by identity and kind.
                matches = True
            else:
                # A pending item is anchored by its physical location; two
                # new pointers under the same parent are distinct discoveries.
                matches = current.nk_offset == item.nk_offset and \
                    current.vk_offset == item.vk_offset and \
                    current.detail == item.detail
            if not matches:
                continue
            for trigger in item.triggers:
                current.add_trigger(trigger)
            for attribute in ("nk_offset", "vk_offset", "parent_nk_offset",
                              "owner_nk_offset", "parent_path"):
                if getattr(current, attribute, None) is None and \
                        getattr(item, attribute, None) is not None:
                    setattr(current, attribute, getattr(item, attribute))
            return current
        self.items.append(item)
        bucket.append(item)
        return item

    def find_item(self, key_path, value_name=None, kind=None):
        for item in self.items:
            if item.key_path == key_path and item.value_name == value_name and \
                    (kind is None or item.kind == kind):
                return item
        return None

    def add_removal(self, removal):
        current = self._removal_index.get(removal.identity)
        if current is not None:
            return current
        self.removals.append(removal)
        self._removal_index[removal.identity] = removal
        return removal

    def identities(self):
        return set(item.identity for item in self.items)

    def value_identities(self):
        return set(item.identity for item in self.items
                   if item.value_name is not None)

    def to_dict(self):
        return OrderedDict([
            ("dirty_pages", [page.to_dict() for page in self.dirty_pages]),
            ("items", [item.to_dict() for item in self.items]),
            ("removals", [removal.to_dict() for removal in self.removals]),
            ("unknown_pages", self.unknown_pages),
        ])

    def explain(self):
        """Return a deterministic list of human readable plan lines."""
        lines = []
        for item in self.items:
            target = item.key_path or "(unknown key)"
            if item.value_name is not None:
                target = "%s : %s" % (target, item.value_name)
            triggers = "; ".join(str(trigger) for trigger in item.triggers)
            lines.append("%s\t%s\t%s" % (item.status, target, triggers))
        for removal in self.removals:
            target = removal.key_path
            if removal.value_name is not None:
                target = "%s : %s" % (target, removal.value_name)
            lines.append("remove\t%s\t%s" % (target, removal.trigger or ""))
        return lines

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)


class ConfigValue(object):
    """
    A value in an exported configuration snapshot.
    """
    def __init__(self, key_path, name, value_type, raw_data, value=None,
                 error=None):
        self.key_path = key_path
        self.name = name
        self.value_type = value_type
        self.raw_data = raw_data
        self.value = value
        self.error = error

    @property
    def identity(self):
        return (self.key_path, self.name)

    def to_dict(self):
        return OrderedDict([
            ("path", self.key_path),
            ("name", self.name),
            ("type", self.value_type),
            ("raw_data", None if self.raw_data is None else self.raw_data.hex()),
            ("value", _json_safe(self.value)),
            ("error", self.error),
        ])

    def __eq__(self, other):
        return isinstance(other, ConfigValue) and \
            self.identity == other.identity and \
            self.value_type == other.value_type and \
            self.raw_data == other.raw_data

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return "ConfigValue(%r, %r, type=0x%x, len=%r)" % \
            (self.key_path, self.name, self.value_type,
             None if self.raw_data is None else len(self.raw_data))


class ConfigKey(object):
    """
    A key in an exported configuration snapshot.
    """
    def __init__(self, path, name, parent_path, subkey_paths, value_names,
                 timestamp=None):
        self.path = path
        self.name = name
        self.parent_path = parent_path
        self.subkey_paths = list(subkey_paths)
        self.value_names = list(value_names)
        self.timestamp = timestamp

    def to_dict(self):
        return OrderedDict([
            ("path", self.path),
            ("name", self.name),
            ("parent_path", self.parent_path),
            ("subkeys", list(self.subkey_paths)),
            ("values", list(self.value_names)),
            ("timestamp", _json_safe(self.timestamp)),
        ])

    def __eq__(self, other):
        return isinstance(other, ConfigKey) and \
            self.path == other.path and \
            self.subkey_paths == other.subkey_paths and \
            self.value_names == other.value_names and \
            self.timestamp == other.timestamp

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return "ConfigKey(path=%r)" % (self.path,)


def _make_config_value(registry_value):
    key_path = registry_value._vkrecord.path() if hasattr(
        registry_value._vkrecord, "path") else None
    # VKRecord does not know its parent key path; callers fill this in.
    name = registry_value.name()
    value_type = registry_value.value_type()
    raw_data = None
    value = None
    error = None
    try:
        raw_data = registry_value.raw_data()
    except Exception as exc:
        error = "%s: %s" % (type(exc).__name__, exc)
    try:
        value = registry_value.value()
    except Exception as exc:
        if error is None:
            error = "%s: %s" % (type(exc).__name__, exc)
    return ConfigValue(key_path, name, value_type, raw_data, value, error)


def _collect_value(key_path, registry_value):
    item = _make_config_value(registry_value)
    item.key_path = key_path
    item.name = registry_value.name()
    return item


def _safe_timestamp(key):
    try:
        return key.timestamp()
    except Exception:
        return None


def _collect_key_records(registry_key, keys, values):
    """
    Collect one key (not its descendants) into the snapshot maps.
    """
    path = registry_key.path()
    try:
        subkey_paths = [subkey.path() for subkey in registry_key.subkeys()]
    except Exception:
        subkey_paths = []
    try:
        registry_values = registry_key.values()
    except Exception:
        registry_values = []

    value_names = []
    for registry_value in registry_values:
        config_value = _collect_value(path, registry_value)
        values[config_value.identity] = config_value
        value_names.append(config_value.name)

    parent_path = None
    try:
        parent_path = registry_key.parent().path()
    except Exception:
        parent_path = None

    keys[path] = ConfigKey(path, registry_key.name(), parent_path,
                           subkey_paths, value_names, _safe_timestamp(registry_key))


def collect_config(registry):
    """
    Fully collect the configuration of a hive into a ConfigSnapshot.

    This is the baseline operation.  Incremental replays use it only for
    affected branches and are checked against this result.
    """
    keys = OrderedDict()
    values = OrderedDict()

    def walk(registry_key):
        _collect_key_records(registry_key, keys, values)
        for subkey in registry_key.subkeys():
            walk(subkey)

    walk(registry.root())
    return ConfigSnapshot(registry, keys, values)


class ConfigSnapshot(object):
    """
    A deterministic exported view of the registry configuration.
    """
    def __init__(self, registry=None, keys=None, values=None):
        self.registry = registry
        self.keys = OrderedDict(keys or {})
        self.values = OrderedDict(values or {})

    def copy(self):
        return ConfigSnapshot(
            self.registry,
            OrderedDict((path, copy.copy(key)) for path, key in self.keys.items()),
            OrderedDict((identity, copy.copy(value))
                        for identity, value in self.values.items()))

    def key_paths(self):
        return list(self.keys.keys())

    def value_identities(self):
        return list(self.values.keys())

    def export(self):
        """
        Export all keys and values in a stable, JSON-friendly structure.
        """
        return OrderedDict([
            ("keys", [self.keys[path].to_dict()
                      for path in sorted(self.keys.keys())]),
            ("values", [self.values[identity].to_dict()
                        for identity in sorted(self.values.keys())]),
        ])

    to_dict = export

    def __eq__(self, other):
        return isinstance(other, ConfigSnapshot) and \
            dict(self.keys) == dict(other.keys) and \
            dict(self.values) == dict(other.values)

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return "ConfigSnapshot(keys=%d, values=%d)" % \
            (len(self.keys), len(self.values))


class DependencyIndex(object):
    """
    A physical-to-logical index for one hive snapshot.

    The index is built once from a fully collected hive.  Planning then only
    inspects the structures that intersect the supplied dirty pages.
    """
    def __init__(self, registry):
        self.registry = registry
        self.buf = registry._buf
        self.dependencies = []
        self._by_page = defaultdict(list)
        self._by_start = {}
        self._keys = OrderedDict()
        self._values = OrderedDict()
        self._value_dependencies = OrderedDict()
        self._list_dependencies = OrderedDict()
        self._key_dependencies = OrderedDict()
        self._build()

    def _add(self, dependency):
        if dependency.size <= 0:
            return dependency
        existing = self._by_start.get((dependency.kind, dependency.start))
        if existing is not None:
            return existing
        self.dependencies.append(dependency)
        self._by_start[(dependency.kind, dependency.start)] = dependency
        for page in dependency.pages():
            self._by_page[page].append(dependency)
        return dependency

    def _read_cell_size(self, absolute_offset):
        try:
            size = struct.unpack_from("<i", self.buf, absolute_offset)[0]
            return abs(size)
        except Exception:
            return 0

    def _build(self):
        root = self.registry.root()
        self._walk_key(root, None)
        self._keys = self._key_dependencies
        self._sorted_key_paths = sorted(self._key_dependencies.keys())
        self._keys_by_parent = defaultdict(list)
        for path, dependency in self._key_dependencies.items():
            self._keys_by_parent[dependency.parent_path].append(path)
        self._values_by_key = defaultdict(list)
        for identity in self._value_dependencies:
            self._values_by_key[identity[0]].append(identity)
        self._key_offsets = {}
        for path, dependency in self._key_dependencies.items():
            self._key_offsets[dependency.record_offset] = path
            self._key_offsets[dependency.record_offset - 0x4] = path
        self._value_offsets = {}
        for identity, dependency in self._value_dependencies.items():
            self._value_offsets[dependency.record_offset] = identity
            self._value_offsets[dependency.record_offset - 0x4] = identity
        self._page_index_sets = {}

    def _walk_key(self, registry_key, parent_path):
        nk = registry_key._nkrecord
        record_offset = nk.offset()
        cell_offset = record_offset - 0x4
        cell_size = self._read_cell_size(cell_offset)

        path = registry_key.path()
        value_name_length = nk.unpack_word(0x48)
        flags = nk.unpack_word(0x2)
        key_fields = OrderedDict([
            ("timestamp", (0x8, 0x8)),
            ("access_bits", (0x10, 0x4)),
            ("parent", (0x14, 0x4)),
            ("subkey_count", (0x18, 0x4)),
            ("subkey_list_offset", (0x20, 0x4)),
            ("value_count", (0x28, 0x4)),
            ("value_list_offset", (0x2C, 0x4)),
            ("classname_offset", (0x34, 0x4)),
            ("name_length", (0x4C, 0x2)),
            ("classname_length", (0x4E, 0x2)),
        ])
        dependency = Dependency(
            KIND_KEY, cell_offset, cell_size, key_path=path,
            record_offset=record_offset, fields=key_fields,
            parent_path=parent_path, scope_path=path)
        dependency.ascii_name = bool(flags & 0x0020)
        dependency.name_length = value_name_length
        dependency.name_offset = record_offset + 0x4C
        self._add(dependency)
        self._key_dependencies[path] = dependency

        # Class name, if any.
        if nk.has_classname():
            try:
                class_offset = FIRST_HBIN_OFFSET + nk.unpack_dword(0x30)
                class_name_length = nk.unpack_word(0x4A)
                class_cell_size = self._read_cell_size(class_offset)
                class_dependency = Dependency(
                    KIND_CLASS, class_offset, class_cell_size, key_path=path,
                    parent_path=parent_path, scope_path=path,
                    fields=OrderedDict([("classname_length", (0x0, 0x0))]))
                class_dependency.name_length = class_name_length
                self._add(class_dependency)
            except Exception:
                pass

        # Directory (subkey list) and its indirection.
        if nk.subkey_number():
            try:
                list_offset = FIRST_HBIN_OFFSET + nk.unpack_dword(0x1C)
                self._add_subkey_list(list_offset, path, parent_path,
                                      nk.subkey_number(), dependency)
            except Exception:
                pass

        # Value list and the VK records that it locates.
        if nk.values_number():
            try:
                self._add_value_list(registry_key, dependency, path, parent_path)
            except Exception:
                pass

        try:
            subkeys = registry_key.subkeys()
        except Exception:
            subkeys = []
        for subkey in subkeys:
            try:
                self._walk_key(subkey, path)
            except Exception:
                continue

    def _add_value_list(self, registry_key, key_dependency, path, parent_path):
        nk = registry_key._nkrecord
        list_offset = FIRST_HBIN_OFFSET + nk.unpack_dword(0x28)
        list_size = self._read_cell_size(list_offset)
        count = nk.values_number()
        dependency = Dependency(
            KIND_VALUE_LIST, list_offset, list_size, key_path=path,
            parent_path=parent_path, scope_path=path, list_count=count,
            fields=OrderedDict([("list", (0x0, list_size))]))
        dependency.count = count
        dependency.value_offsets = []
        self._add(dependency)
        self._list_dependencies[(KIND_VALUE_LIST, list_offset)] = dependency

        try:
            value_list = nk.values_list()
        except Exception:
            return
        for offset in range(count):
            try:
                value_offset = FIRST_HBIN_OFFSET + value_list.unpack_dword(4 * offset)
                dependency.value_offsets.append(value_offset)
            except Exception:
                break

        try:
            registry_values = registry_key.values()
        except Exception:
            registry_values = []

        for registry_value in registry_values:
            vk = registry_value._vkrecord
            vk_record_offset = vk.offset()
            vk_cell_offset = vk_record_offset - 0x4
            vk_cell_size = self._read_cell_size(vk_cell_offset)
            value_fields = OrderedDict([
                ("name_length", (0x6, 0x2)),
                ("data_size", (0x8, 0x4)),
                ("data_offset", (0xC, 0x4)),
                ("data_type", (0x10, 0x4)),
                ("name_flags", (0x14, 0x2)),
                ("name", (0x18, 0)),
            ])
            value_dependency = Dependency(
                KIND_VALUE, vk_cell_offset, vk_cell_size, key_path=path,
                value_name=registry_value.name(), record_offset=vk_record_offset,
                fields=value_fields, parent_path=parent_path, scope_path=path,
                value_type=registry_value.value_type())
            value_dependency.name_length = vk.unpack_word(0x2)
            value_dependency.name_offset = vk_record_offset + 0x14
            value_dependency.ascii_name = bool(vk.unpack_word(0x10) & 1)
            self._add(value_dependency)
            self._value_dependencies[(path, registry_value.name())] = value_dependency

            try:
                fragments = vk.data_fragments()
            except Exception:
                fragments = []
            value_dependency.fragments = list(fragments)
            for fragment_index, (fragment_start, fragment_size) in enumerate(fragments):
                data_dependency = Dependency(
                    KIND_VALUE_DATA, fragment_start, fragment_size,
                    key_path=path, value_name=registry_value.name(),
                    parent_path=parent_path, scope_path=path,
                    value_type=registry_value.value_type())
                data_dependency.fragment_index = fragment_index
                data_dependency.owner = value_dependency
                self._add(data_dependency)

    def _add_subkey_list(self, list_offset, path, parent_path, count, owner,
                         seen=None):
        if seen is None:
            seen = set()
        if list_offset in seen:
            return
        seen.add(list_offset)

        list_size = self._read_cell_size(list_offset)
        list_id = self.buf[list_offset + 0x4:list_offset + 0x6]
        dependency = Dependency(
            KIND_SUBKEY_LIST, list_offset, list_size, key_path=path,
            parent_path=parent_path, scope_path=path, list_id=list_id,
            list_count=count,
            fields=OrderedDict([("list", (0x0, list_size))]))
        dependency.owner = owner
        dependency.child_offsets = []
        self._add(dependency)
        self._list_dependencies[(KIND_SUBKEY_LIST, list_offset)] = dependency

        if list_id == b"ri":
            child_count = struct.unpack_from("<H", self.buf, list_offset + 0x6)[0]
            for index in range(child_count):
                try:
                    child_offset = FIRST_HBIN_OFFSET + \
                        struct.unpack_from("<I", self.buf,
                                           list_offset + 0x8 + 4 * index)[0]
                except Exception:
                    break
                dependency.child_offsets.append(child_offset)
                self._add_subkey_list(child_offset, path, parent_path, 0,
                                      owner, seen)
            return

        if list_id not in _LIST_IDS:
            return

        entry_size = 4 if list_id == b"li" else 8
        entry_count = 0
        try:
            entry_count = struct.unpack_from("<H", self.buf, list_offset + 0x6)[0]
        except Exception:
            entry_count = count
        for index in range(entry_count):
            try:
                child_offset = FIRST_HBIN_OFFSET + struct.unpack_from(
                    "<I", self.buf, list_offset + 0x8 + entry_size * index)[0]
            except Exception:
                break
            dependency.child_offsets.append(child_offset)

    def structures_on_page(self, page):
        """Return dependencies that intersect a page (absolute offset)."""
        return list(self._by_page.get(page, []))

    def dependency_for(self, kind, start):
        return self._by_start.get((kind, start))

    def value_dependency(self, key_path, value_name):
        return self._value_dependencies.get((key_path, value_name))

    def key_dependency(self, key_path):
        return self._key_dependencies.get(key_path)

    def list_dependency(self, kind, start):
        return self._list_dependencies.get((kind, start))

    def key_paths(self):
        return list(self._key_dependencies.keys())

    def value_identities(self):
        return list(self._value_dependencies.keys())


def build_dependency_index(registry):
    """
    Build a DependencyIndex from a Registry instance (or a file-like/buffer).
    """
    if not isinstance(registry, Registry.Registry):
        registry = Registry.Registry(registry)
    return DependencyIndex(registry)


def _normalize_pages(dirty_pages):
    normalized = []
    for page in dirty_pages:
        if isinstance(page, DirtyPageRegion):
            normalized.append(page)
        elif isinstance(page, dict):
            normalized.append(DirtyPageRegion(
                page.get("offset", 0), page.get("size", 0),
                data=page.get("data"), sequence=page.get("sequence"),
                source=page.get("source"),
                relative=page.get("relative", False)))
        elif isinstance(page, (tuple, list)):
            offset, size = page[0], page[1]
            data = page[2] if len(page) > 2 else None
            sequence = page[3] if len(page) > 3 else None
            normalized.append(DirtyPageRegion(offset, size, data=data,
                                              sequence=sequence))
        else:
            # A RegistryParse.DirtyPageReference-like object.
            normalized.append(DirtyPageRegion.from_log_reference(page))
    return normalized


class _Overlay(object):
    """
    Final bytes for dirty regions, falling back to the baseline hive.
    """
    def __init__(self, baseline, dirty_pages):
        self.baseline = baseline
        # Preserve input order: later log entries must win over earlier
        # entries that touch the same bytes.
        self.regions = []
        for page in dirty_pages:
            if page.data is not None:
                self.regions.append((page.offset, page.offset + page.size,
                                     page.data))

    def has_data(self, start, size):
        end = start + size
        for region_start, region_end, data in self.regions:
            if region_start < end and region_end > start:
                return True
        return False

    def get(self, start, size):
        """
        Return the final bytes for [start, start + size), or None when a
        changed region has no supplied data.
        """
        end = start + size
        covered = False
        for region_start, region_end, data in self.regions:
            if region_start < end and region_end > start:
                covered = True
                break
        if not covered:
            return None
        result = bytearray(self.baseline[start:end])
        for region_start, region_end, data in self.regions:
            overlap_start = max(start, region_start)
            overlap_end = min(end, region_end)
            if overlap_start >= overlap_end:
                continue
            source_start = overlap_start - region_start
            target_start = overlap_start - start
            chunk = data[source_start:source_start + (overlap_end - overlap_start)]
            result[target_start:target_start + len(chunk)] = chunk
        return bytes(result)

    def page_bytes(self, page):
        return self.get(page, PAGE_SIZE)


_FIELD_IDENTITY = {
    ("key", "parent"),
    ("key", "name_length"),
    ("key", "name"),
    ("value", "name_length"),
    ("value", "name"),
    ("value", "name_flags"),
}
_FIELD_DIRECTORY = {
    ("key", "subkey_count"),
    ("key", "subkey_list_offset"),
}
_FIELD_VALUES = {
    ("key", "value_count"),
    ("key", "value_list_offset"),
}
_FIELD_VALUE_CONTENT = {
    ("value", "data_size"),
    ("value", "data_offset"),
    ("value", "data_type"),
}
_FIELD_VALUE_LAYOUT = {
    ("value", "name_length"),
    ("value", "name"),
    ("value", "name_flags"),
}


def _field_names_for(dependency, before, after):
    names = []
    if before is None or after is None:
        return ["cell"]
    for name, (offset, length) in dependency.fields.items():
        if name == "name":
            if dependency.name_length is not None:
                length = dependency.name_length
            else:
                length = 0
        if length <= 0 and name != "list":
            continue
        if _field_changed(before, after, offset, length):
            names.append(name)
    return names


def _trigger_for(dependency, ranges, fields, page, sequence, source,
                 detail=None, propagated_from=None):
    return ChangeTrigger(
        dependency.kind, page=page, structure_offset=dependency.record_offset
        if dependency.record_offset is not None else dependency.start,
        fields=fields, pages=[item[0] for item in ranges] or [page],
        sequence=sequence, source=source, detail=detail,
        propagated_from=propagated_from)


def _structure_bytes(overlay, dependency):
    return overlay.get(dependency.start, dependency.size)


def _baseline_bytes(index, dependency):
    return index.buf[dependency.start:dependency.start + dependency.size]


def _changed_structure(index, overlay, dependency, dirty_intervals):
    if not _intersects(dependency.start, dependency.size, dirty_intervals):
        return None
    if not overlay.has_data(dependency.start, dependency.size):
        # Dirty page bytes were not supplied.  Be conservative and treat
        # the structure as changed.
        return _changed_ranges(None, None, dependency.start, dependency.size)
    after = _structure_bytes(overlay, dependency)
    before = _baseline_bytes(index, dependency)
    if after is None:
        return _changed_ranges(None, None, dependency.start, dependency.size)
    if before == after:
        return []
    return _changed_ranges(before, after, dependency.start, dependency.size)


def _children_from_overlay(overlay, offset):
    """
    Return child NK offsets for a subkey list, reading final cell bytes from
    the overlay and following RI indirection.
    """
    header = overlay.get(offset + 0x4, 0x4)
    if header is None or len(header) < 0x4:
        return []
    list_id = header[0x0:0x2]
    try:
        count = struct.unpack_from("<H", header, 0x2)[0]
    except Exception:
        return []
    if list_id == b"ri":
        children = []
        pointers = overlay.get(offset + 0x8, 4 * count)
        if pointers is None:
            return []
        for index_ in range(count):
            try:
                child_list = FIRST_HBIN_OFFSET + struct.unpack_from(
                    "<I", pointers, 4 * index_)[0]
            except Exception:
                break
            children.extend(_children_from_overlay(overlay, child_list))
        return children
    if list_id not in _LIST_IDS:
        return []
    entry_size = 4 if list_id == b"li" else 8
    pointers = overlay.get(offset + 0x8, entry_size * count)
    if pointers is None:
        return []
    children = []
    for index_ in range(count):
        try:
            children.append(FIRST_HBIN_OFFSET + struct.unpack_from(
                "<I", pointers, entry_size * index_)[0])
        except Exception:
            break
    return children


def _value_offsets_from_overlay(overlay, offset, count):
    pointers = overlay.get(offset + 0x4, 4 * count)
    if pointers is None:
        return []
    offsets = []
    for index_ in range(count):
        try:
            offsets.append(FIRST_HBIN_OFFSET + struct.unpack_from(
                "<I", pointers, 4 * index_)[0])
        except Exception:
            break
    return offsets


def _baseline_value_for_vk(index, vk_offset):
    return index._value_offsets.get(vk_offset)


def _subtree_paths(index, key_path):
    """Return key_path and all descendant key paths using a sorted prefix
    range, not a scan of every key in the hive."""
    paths = index._sorted_key_paths
    prefix = key_path + "\\"
    start = bisect.bisect_left(paths, key_path)
    end = bisect.bisect_left(paths, prefix + "\U0010FFFF")
    result = []
    for candidate in paths[start:end]:
        if candidate == key_path or candidate.startswith(prefix):
            result.append(candidate)
        elif candidate > prefix:
            break
    return result


def _subtree_values(index, key_path):
    """Return the identities of values owned by a key or its descendants."""
    values = []
    for path in _subtree_paths(index, key_path):
        values.extend(index._values_by_key.get(path, []))
    return values


def _new_structure_dependencies(index, overlay, dirty_intervals):
    """
    Find structures on dirty pages that are not part of the baseline index.

    Only pages that actually contain dirty bytes are inspected.  This is a
    bounded scan of the affected allocation blocks, not a scan of the hive.
    """
    candidates = []
    seen_cells = set()
    dirty_pages = set()
    for start, end in dirty_intervals:
        for page in _aligned_pages(start, end - start):
            dirty_pages.add(page)

    # Walk only HBIN blocks that overlap a dirty page.  The HBIN headers are
    # at fixed offsets, so this does not require a full hive traversal.
    hbin_candidates = set()
    for page in dirty_pages:
        hbin_candidates.add(0x1000 + ((page - 0x1000) // 0x1000) * 0x1000)

    # HBIN structures are 0x1000 aligned but may be larger than one block.
    # Locate the containing HBIN by following the linked list from the first
    # HBIN; only a handful of headers are visited.
    hbin_offsets = set()
    try:
        hbin = RegistryParse.HBINBlock(index.buf, FIRST_HBIN_OFFSET, None)
        while True:
            hbin_start = hbin.offset()
            hbin_end = hbin._offset_next_hbin
            if any(hbin_start < page + PAGE_SIZE and hbin_end > page
                   for page in dirty_pages):
                hbin_offsets.add(hbin_start)
            if not hbin.has_next():
                break
            hbin = hbin.next()
    except Exception:
        pass

    for hbin_start in sorted(hbin_offsets):
        try:
            hbin = RegistryParse.HBINBlock(index.buf, hbin_start, None)
            cells = hbin.cells()
            for cell in cells:
                offset = cell.offset()
                size = cell.size()
                if size <= 0:
                    break
                if not _intersects(offset, size, dirty_intervals):
                    if offset >= max(end for _, end in dirty_intervals):
                        break
                    continue
                if offset in seen_cells:
                    continue
                seen_cells.add(offset)
                if (KIND_KEY, offset) in index._by_start or \
                        (KIND_VALUE, offset) in index._by_start or \
                        (KIND_VALUE_LIST, offset) in index._by_start or \
                        (KIND_SUBKEY_LIST, offset) in index._by_start:
                    continue
                try:
                    data_id = index.buf[offset + 0x4:offset + 0x6]
                except Exception:
                    continue
                if data_id in _NEW_STRUCTURE_IDS:
                    candidates.append((offset, size, data_id))
        except Exception:
            continue
    return candidates


def _value_name_from_after(index, overlay, dependency):
    """
    Read the value name from the final version of a VK record.

    `raw` is cell-relative, so the VK fields are shifted by the 4-byte
    HBIN cell length: name length at 0x6, flags at 0x14, name at 0x18.
    """
    raw = overlay.get(dependency.start, dependency.size)
    if raw is None:
        return None, None, None
    if raw[0x4:0x6] != b"vk":
        return None, None, None
    try:
        name_length = struct.unpack_from("<H", raw, 0x6)[0]
        flags = struct.unpack_from("<H", raw, 0x14)[0]
        name_raw = raw[0x18:0x18 + name_length]
        if flags & 1:
            name = name_raw.decode("windows-1252")
        else:
            name = name_raw.decode("utf-16le")
        if name_length == 0:
            name = "(default)"
        return name, name_length, flags
    except Exception:
        return None, None, None


def plan_recollection(index, dirty_pages, after_pages=None):
    """
    Build a RecollectPlan from dirty page regions.

    Arguments:
      - `index`: a DependencyIndex built from the baseline hive.
      - `dirty_pages`: DirtyPageRegion objects, (offset, size[, data]) tuples,
            or RegistryParse.DirtyPageReference objects.  Offsets are absolute
            unless `relative=True` is specified with DirtyPageRegion.
      - `after_pages`: optional map of page offset -> final page bytes.  If
            omitted, the dirty page data from the regions is used.

    Returns a RecollectPlan.  The plan never scans the whole hive; it inspects
    the structures that intersect the supplied dirty pages and follows new
    pointers only far enough to mark them for rediscovery.
    """
    regions = _normalize_pages(dirty_pages)
    if after_pages:
        extra = []
        for page, data in after_pages.items():
            extra.append(DirtyPageRegion(page, len(data), data=data))
        regions = regions + extra

    if not regions:
        return RecollectPlan()

    dirty_intervals = [(region.offset, region.offset + region.size)
                       for region in regions if region.size > 0]
    if not dirty_intervals:
        return RecollectPlan(regions)

    overlay = _Overlay(index.buf, regions)
    sequence_for_page = {}
    source_for_page = {}
    for region in regions:
        for page in region.pages():
            if region.data is not None:
                sequence_for_page[page] = region.sequence
                source_for_page[page] = region.source

    plan = RecollectPlan(regions)
    changed_structures = []
    directory_triggers = []
    value_list_triggers = []
    value_identity_triggers = []
    value_content_triggers = []

    # Inspect only structures that intersect dirty pages.
    dependencies = []
    seen_dependencies = set()
    for region in regions:
        for page in region.pages():
            for dependency in index.structures_on_page(page):
                key = (dependency.kind, dependency.start, dependency.value_name)
                if key in seen_dependencies:
                    continue
                seen_dependencies.add(key)
                dependencies.append(dependency)

    for dependency in dependencies:
        if dependency.kind not in (KIND_KEY, KIND_VALUE, KIND_VALUE_LIST,
                                   KIND_SUBKEY_LIST, KIND_VALUE_DATA):
            continue
        changes = _changed_structure(index, overlay, dependency, dirty_intervals)
        if changes is None or len(changes) == 0:
            continue
        page = changes[0][0] - (changes[0][0] % PAGE_SIZE)
        sequence = sequence_for_page.get(page)
        source = source_for_page.get(page)
        has_payload = overlay.has_data(dependency.start, dependency.size)

        if dependency.kind == KIND_KEY:
            before = _baseline_bytes(index, dependency)
            after = _structure_bytes(overlay, dependency)
            fields = _field_names_for(dependency, before, after)
            identity_fields = [field for field in fields
                               if (KIND_KEY, field) in _FIELD_IDENTITY]
            directory_fields = [field for field in fields
                                if (KIND_KEY, field) in _FIELD_DIRECTORY]
            value_fields = [field for field in fields
                            if (KIND_KEY, field) in _FIELD_VALUES]
            metadata_fields = [field for field in fields
                               if field in ("timestamp", "access_bits",
                                            "classname_offset",
                                            "classname_length")]
            if not (identity_fields or directory_fields or value_fields or
                    metadata_fields):
                continue
            changed_structures.append((dependency, changes, fields))
            trigger = _trigger_for(dependency, changes, fields, page, sequence,
                                   source)
            if not has_payload:
                # Without the final bytes the fields cannot be classified;
                # treat the key and its identity scope as unconfirmed.
                directory_triggers.append((dependency, trigger,
                                           fields or ["cell"], []))
            else:
                if identity_fields or directory_fields:
                    directory_triggers.append((dependency, trigger,
                                               identity_fields,
                                               directory_fields))
                if value_fields:
                    value_list_triggers.append((dependency, trigger,
                                                value_fields))
                if metadata_fields and not identity_fields and \
                        not directory_fields and not value_fields:
                    plan.add_item(RecollectItem(
                        KIND_KEY, key_path=dependency.key_path,
                        status=STATUS_RECOLLECT, triggers=[trigger],
                        scope_path=dependency.key_path,
                        detail="key metadata changed: %s" %
                               ", ".join(metadata_fields),
                        nk_offset=dependency.record_offset, mode="metadata"))

        elif dependency.kind == KIND_SUBKEY_LIST:
            changed_structures.append((dependency, changes, ["list"]))
            trigger = _trigger_for(dependency, changes, ["list"], page,
                                   sequence, source)
            directory_triggers.append((dependency, trigger, ["list"], []))

        elif dependency.kind == KIND_VALUE_LIST:
            changed_structures.append((dependency, changes, ["list"]))
            trigger = _trigger_for(dependency, changes, ["list"], page,
                                   sequence, source)
            value_list_triggers.append((dependency, trigger, ["list"]))

        elif dependency.kind == KIND_VALUE:
            before = _baseline_bytes(index, dependency)
            after = _structure_bytes(overlay, dependency)
            fields = _field_names_for(dependency, before, after)
            identity_fields = [field for field in fields
                               if (KIND_VALUE, field) in _FIELD_IDENTITY or
                               (KIND_VALUE, field) in _FIELD_VALUE_LAYOUT]
            content_fields = [field for field in fields
                              if (KIND_VALUE, field) in _FIELD_VALUE_CONTENT]
            if not fields:
                continue
            changed_structures.append((dependency, changes, fields))
            trigger = _trigger_for(dependency, changes, fields, page,
                                   sequence, source)
            if identity_fields:
                value_identity_triggers.append((dependency, trigger,
                                                identity_fields))
            if content_fields or not identity_fields:
                value_content_triggers.append((dependency, trigger,
                                               content_fields or fields))

        elif dependency.kind == KIND_VALUE_DATA:
            owner = getattr(dependency, "owner", None)
            if owner is None:
                continue
            changed_structures.append((dependency, changes,
                                       ["fragment"]))
            trigger = _trigger_for(dependency, changes,
                                   ["data_fragment_%d" %
                                    getattr(dependency, "fragment_index", 0)],
                                   page, sequence, source)
            value_content_triggers.append((owner, trigger, ["data"]))

    # Directory changes invalidate the identity of everything below the key.
    # Every affected key and value is listed explicitly with its physical
    # record offset so replay does not need to scan the hive.
    for dependency, trigger, identity_fields, directory_fields in directory_triggers:
        path = dependency.key_path
        plan.add_item(RecollectItem(
            KIND_KEY, key_path=path, status=STATUS_RECOLLECT,
            triggers=[trigger], scope_path=path,
            detail="directory changed: %s" % (", ".join(
                identity_fields + directory_fields) or "subkey list"),
            nk_offset=dependency.record_offset, mode="key"))
        for value_identity in _subtree_values(index, path):
            value_dependency = index.value_dependency(*value_identity)
            plan.add_item(RecollectItem(
                KIND_VALUE, key_path=value_identity[0],
                value_name=value_identity[1], status=STATUS_RECOLLECT,
                triggers=[ChangeTrigger(
                    "directory", page=trigger.page,
                    structure_offset=trigger.structure_offset,
                    fields=identity_fields + directory_fields,
                    pages=trigger.pages, sequence=trigger.sequence,
                    source=trigger.source,
                    detail="identity invalidated by directory change",
                    propagated_from=path)],
                scope_path=path,
                vk_offset=(value_dependency.record_offset
                           if value_dependency else None),
                owner_nk_offset=(index.key_dependency(value_identity[0]).record_offset
                                 if index.key_dependency(value_identity[0]) else None),
                mode="vk" if value_dependency else "value"))
        for child_path in _subtree_paths(index, path):
            if child_path == path:
                continue
            child_dependency = index.key_dependency(child_path)
            plan.add_item(RecollectItem(
                KIND_KEY, key_path=child_path, status=STATUS_RECOLLECT,
                triggers=[ChangeTrigger(
                    "directory", page=trigger.page,
                    structure_offset=trigger.structure_offset,
                    fields=identity_fields + directory_fields,
                    pages=trigger.pages, sequence=trigger.sequence,
                    source=trigger.source,
                    detail="identity invalidated by ancestor directory",
                    propagated_from=path)], scope_path=path,
                nk_offset=(child_dependency.record_offset
                           if child_dependency else None),
                mode="key"))

    # A changed value list invalidates the identity and order of the values
    # directly owned by the key.
    for dependency, trigger, fields in value_list_triggers:
        path = dependency.key_path
        key_dependency = index.key_dependency(path)
        plan.add_item(RecollectItem(
            KIND_KEY, key_path=path, status=STATUS_RECOLLECT,
            triggers=[trigger], scope_path=path,
            detail="value list changed: %s" % ", ".join(fields),
            nk_offset=(key_dependency.record_offset
                       if key_dependency else dependency.record_offset),
            mode="key"))
        for value_identity in _subtree_values(index, path):
            if value_identity[0] != path:
                continue
            value_dependency = index.value_dependency(*value_identity)
            plan.add_item(RecollectItem(
                KIND_VALUE, key_path=path, value_name=value_identity[1],
                status=STATUS_RECOLLECT,
                triggers=[ChangeTrigger(
                    "value_list", page=trigger.page,
                    structure_offset=trigger.structure_offset,
                    fields=fields, pages=trigger.pages,
                    sequence=trigger.sequence, source=trigger.source,
                    detail="value identity may have changed",
                    propagated_from=path)],
                scope_path=path,
                vk_offset=(value_dependency.record_offset
                           if value_dependency else None),
                owner_nk_offset=(index.key_dependency(value_identity[0]).record_offset
                                 if index.key_dependency(value_identity[0]) else None),
                mode="vk" if value_dependency else "value"))

    # Changed VK identity fields: the old identity must be removed and the
    # new location rediscovered from the changed VK cell.
    for dependency, trigger, fields in value_identity_triggers:
        path = dependency.key_path
        old_name = dependency.value_name
        new_name, _new_length, _new_flags = _value_name_from_after(
            index, overlay, dependency)
        if new_name is None:
            plan.add_item(RecollectItem(
                KIND_VALUE, key_path=path, value_name=old_name,
                status=STATUS_PENDING, triggers=[trigger], scope_path=path,
                detail="new value identity cannot be confirmed",
                vk_offset=dependency.record_offset,
                mode="rediscover_value"))
        elif new_name != old_name:
            plan.add_item(RecollectItem(
                KIND_VALUE, key_path=path, value_name=old_name,
                status=STATUS_PENDING, triggers=[ChangeTrigger(
                    "value_identity", page=trigger.page,
                    structure_offset=trigger.structure_offset,
                    fields=trigger.fields, pages=trigger.pages,
                    sequence=trigger.sequence, source=trigger.source,
                    detail="old identity removed by rename",
                    propagated_from=trigger.detail)], scope_path=path,
                detail="value renamed; old identity removed",
                vk_offset=dependency.record_offset,
                mode="rediscover_value"))
            plan.add_removal(Removal(path, old_name, trigger=trigger))
            plan.add_item(RecollectItem(
                KIND_VALUE, key_path=path, value_name=new_name,
                status=STATUS_PENDING, triggers=[trigger], scope_path=path,
                detail="value renamed; new identity requires rediscovery",
                vk_offset=dependency.record_offset,
                mode="rediscover_value"))
        else:
            plan.add_item(RecollectItem(
                KIND_VALUE, key_path=path, value_name=old_name,
                status=STATUS_RECOLLECT, triggers=[trigger], scope_path=path,
                detail="value name metadata changed",
                vk_offset=dependency.record_offset, mode="vk",
                owner_nk_offset=(index.key_dependency(path).record_offset
                                 if index.key_dependency(path) else None)))

    # Changed value content: stable identity, recollect that value only.
    for dependency, trigger, fields in value_content_triggers:
        path = dependency.key_path
        name = dependency.value_name
        plan.add_item(RecollectItem(
            KIND_VALUE, key_path=path, value_name=name,
            status=STATUS_RECOLLECT, triggers=[trigger], scope_path=path,
            detail="value content changed: %s" % (", ".join(fields) or "data"),
            vk_offset=dependency.record_offset, mode="vk",
            owner_nk_offset=(index.key_dependency(path).record_offset
                             if index.key_dependency(path) else None)))

    # Check pointers from changed value lists and subkey lists.  A new
    # pointer that cannot be resolved from the baseline index is never
    # guessed; it becomes a pending rediscovery item.
    for dependency, trigger, fields in value_list_triggers:
        path = dependency.key_path
        before = _baseline_bytes(index, dependency)
        after = _structure_bytes(overlay, dependency)
        old_offsets = list(getattr(dependency, "value_offsets", []))
        count = getattr(dependency, "count", dependency.list_count or 0)
        if after is None:
            # The final value list is unavailable.  Do not interpret the
            # missing bytes as an empty list (which would fabricate
            # removals); require rediscovery of the list instead.
            plan.add_item(RecollectItem(
                KIND_VALUE, key_path=path, value_name=None,
                status=STATUS_PENDING, triggers=[trigger], scope_path=path,
                parent_path=path,
                detail="value list contents cannot be confirmed",
                mode="rediscover_values"))
            continue
        new_offsets = _value_offsets_from_overlay(
            overlay, dependency.start, count)

        known_vks = set(index.value_dependency(*identity).record_offset - 0x4
                        for identity in index._values_by_key.get(path, []))
        for offset in new_offsets:
            if offset not in known_vks and \
                    index.dependency_for(KIND_VALUE, offset) is None:
                plan.add_item(RecollectItem(
                    KIND_VALUE, key_path=path, value_name=None,
                    status=STATUS_PENDING, triggers=[trigger], scope_path=path,
                    parent_path=path,
                    detail="new value pointer at 0x%x requires rediscovery" %
                           offset, mode="rediscover_values"))
        for offset in old_offsets:
            if offset not in new_offsets:
                identity = _baseline_value_for_vk(index, offset + 0x4)
                if identity is not None:
                    plan.add_removal(Removal(identity[0], identity[1],
                                             trigger=trigger))

    for dependency, trigger, identity_fields, directory_fields in directory_triggers:
        path = dependency.key_path
        before = _baseline_bytes(index, dependency)
        after = _structure_bytes(overlay, dependency)
        list_offset = None
        if dependency.kind == KIND_SUBKEY_LIST:
            list_offset = dependency.start
        elif dependency.kind == KIND_KEY and dependency.record_offset is not None:
            pointer = overlay.get(dependency.record_offset + 0x1C, 0x4)
            if pointer is not None and len(pointer) == 0x4:
                list_offset = FIRST_HBIN_OFFSET + struct.unpack_from(
                    "<I", pointer, 0)[0]
        if list_offset is None:
            plan.add_item(RecollectItem(
                KIND_KEY, key_path=path, status=STATUS_PENDING,
                triggers=[trigger], scope_path=path, parent_path=path,
                detail="subkey list contents cannot be confirmed",
                mode="rediscover_key"))
            continue
        new_children = _children_from_overlay(overlay, list_offset)
        if not overlay.has_data(list_offset + 0x4, 0x4):
            plan.add_item(RecollectItem(
                KIND_KEY, key_path=path, status=STATUS_PENDING,
                triggers=[trigger], scope_path=path, parent_path=path,
                detail="subkey list contents cannot be confirmed",
                mode="rediscover_key"))
            continue
        known_nks = index._key_offsets
        for child in new_children:
            if child not in known_nks and \
                    index.dependency_for(KIND_KEY, child) is None:
                plan.add_item(RecollectItem(
                    KIND_KEY, key_path=path, status=STATUS_PENDING,
                    triggers=[trigger], scope_path=path, parent_path=path,
                    detail="new subkey pointer at 0x%x requires rediscovery" %
                           child, mode="rediscover_key",
                    nk_offset=child + 0x4))
        old_paths = list(index._keys_by_parent.get(path, []))
        new_offsets = set(new_children)
        for old_path in old_paths:
            old_dep = index._keys[old_path]
            if old_dep.record_offset - 0x4 not in new_offsets:
                plan.add_removal(Removal(old_path, subtree=True,
                                         trigger=trigger))

    # A structural cell on a dirty page whose payload is available but that
    # is not reachable from any indexed pointer is an unconfirmed location.
    # When no payload is supplied the dirty pages are already covered by the
    # conservative pending items above, so nothing is inferred here.
    if any(region.data is not None for region in regions):
        for offset, _size, data_id in _new_structure_dependencies(
                index, overlay, dirty_intervals):
            page = offset - (offset % PAGE_SIZE)
            sequence = sequence_for_page.get(page)
            source = source_for_page.get(page)
            if data_id == b"nk":
                kind = KIND_KEY
                detail = "new key structure at 0x%x" % offset
            elif data_id == b"vk":
                kind = KIND_VALUE
                detail = "new value structure at 0x%x" % offset
            elif data_id in (b"li", b"lf", b"lh", b"ri"):
                kind = KIND_SUBKEY_LIST
                detail = "new directory structure at 0x%x" % offset
            else:
                kind = KIND_VALUE_DATA
                detail = "new data structure at 0x%x" % offset
            trigger = ChangeTrigger(
                "new_structure", page=page, structure_offset=offset,
                fields=["cell"], pages=[page], sequence=sequence,
                source=source, detail=detail)
            plan.add_item(RecollectItem(
                kind, key_path=None, status=STATUS_PENDING,
                triggers=[trigger], detail=detail))

    # Pages that changed but contain no indexed structure.  They are only
    # recorded; they do not invalidate a logical identity unless a changed
    # pointer points into them (handled above).
    indexed_pages = set()
    for dependency in index.dependencies:
        indexed_pages.update(dependency.pages())
    for page in sorted(set(page for region in regions for page in region.pages())):
        if page not in indexed_pages:
            plan.unknown_pages.append(page)

    return plan


def _open_key(registry, path):
    """
    Open a key by its full path, including the root key name.

    Replay attaches a small cache to the Registry instance so each planned
    path is resolved only once.
    """
    cache = getattr(registry, "_incremental_key_cache", None)
    if cache is not None and path in cache:
        return cache[path]
    if not path:
        key = registry.root()
    else:
        key = None
        try:
            key = registry.open(path)
        except Exception:
            key = None
        if key is None:
            root_name = registry.root().name()
            if path == root_name:
                key = registry.root()
            else:
                relative = path[len(root_name) + 1:] \
                    if path.startswith(root_name + "\\") else path
                key = registry.open(relative)
    if cache is not None:
        cache[path] = key
    return key


def _hbin_parent(registry):
    """
    Return an HBINBlock usable as the parent of detached NK/VK records.

    The parser only needs this parent to convert HBIN-relative offsets to
    absolute offsets, which always start at the first HBIN.
    """
    return RegistryParse.HBINBlock(registry._buf, FIRST_HBIN_OFFSET, None)


def _key_record_at(registry, record_offset):
    """Build an NKRecord directly from a physical record offset."""
    return RegistryParse.NKRecord(registry._buf, record_offset,
                                  _hbin_parent(registry))


def _value_record_at(registry, record_offset):
    """Build a VKRecord directly from a physical record offset."""
    return RegistryParse.VKRecord(registry._buf, record_offset,
                                  _hbin_parent(registry))


def _snapshot_key_from_record(registry, nk, fallback_path=None):
    """
    Build a ConfigKey (and its direct ConfigValues) from an NKRecord.
    """
    path = None
    try:
        path = nk.path()
    except Exception:
        path = fallback_path
    if path is None:
        return None, []
    try:
        name = nk.name()
    except Exception:
        name = path.rsplit("\\", 1)[-1]

    subkey_paths = []
    try:
        if nk.subkey_number():
            for child in nk.subkey_list().keys():
                try:
                    subkey_paths.append(child.path())
                except Exception:
                    continue
    except Exception:
        pass

    values = []
    value_names = []
    try:
        if nk.values_number():
            from . import Registry as RegistryModule
            for record in nk.values_list().values():
                try:
                    value = RegistryModule.RegistryValue(record)
                    config_value = _collect_value(path, value)
                    values.append(config_value)
                    value_names.append(config_value.name)
                except Exception:
                    continue
    except Exception:
        pass

    parent_path = None
    try:
        parent_path = nk.parent_key().path()
    except Exception:
        parent_path = None

    from . import RegistryParse as _Parse
    timestamp = None
    try:
        timestamp = _Parse.parse_windows_timestamp(nk.unpack_qword(0x4))
    except Exception:
        timestamp = None

    return ConfigKey(path, name, parent_path, subkey_paths, value_names,
                     timestamp), values


def _snapshot_value_from_record(registry, vk, key_path):
    """Build a ConfigValue from a VKRecord and its owning key path."""
    from . import Registry as RegistryModule
    value = RegistryModule.RegistryValue(vk)
    config_value = _collect_value(key_path, value)
    return config_value


def _all_values_for_key(registry, key_path):
    """
    Read the values of one key only.  Used when the value list changed and
    every pointer in that list must be rediscovered.
    """
    key = _open_key(registry, key_path)
    values = []
    for registry_value in key.values():
        values.append(_collect_value(key_path, registry_value))
    return values


def _collect_branch(registry, path, snapshot):
    """
    Recollect a key and all descendants into `snapshot`.
    """
    try:
        key = _open_key(registry, path)
    except Exception:
        return False

    prefix = path + "\\"
    for existing in list(snapshot.keys.keys()):
        if existing == path or existing.startswith(prefix):
            del snapshot.keys[existing]
    for identity in list(snapshot.values.keys()):
        if identity[0] == path or identity[0].startswith(prefix):
            del snapshot.values[identity]

    def walk(registry_key):
        _collect_key_records(registry_key, snapshot.keys, snapshot.values)
        try:
            for subkey in registry_key.subkeys():
                walk(subkey)
        except Exception:
            return

    walk(key)
    return True


def _recollect_single_value(registry, key_path, value_name, snapshot):
    try:
        key = _open_key(registry, key_path)
    except Exception:
        return False
    try:
        registry_value = key.value(value_name)
    except Exception:
        for identity in list(snapshot.values.keys()):
            if identity[0] == key_path and identity[1] == value_name:
                del snapshot.values[identity]
        return True
    config_value = _collect_value(key_path, registry_value)
    snapshot.values[config_value.identity] = config_value
    return True


def _apply_removals(result, removals):
    for removal in removals:
        if removal.subtree:
            prefix = removal.key_path + "\\"
            for path in list(result.keys.keys()):
                if path == removal.key_path or path.startswith(prefix):
                    del result.keys[path]
            for identity in list(result.values.keys()):
                if identity[0] == removal.key_path or \
                        identity[0].startswith(prefix):
                    del result.values[identity]
        elif removal.value_name is not None:
            result.values.pop((removal.key_path, removal.value_name), None)
        else:
            result.keys.pop(removal.key_path, None)
            for identity in list(result.values.keys()):
                if identity[0] == removal.key_path:
                    del result.values[identity]


def _replay_key_item(registry, item, result):
    """Refresh one key node without traversing its descendants."""
    if item.nk_offset is None:
        key = _open_key(registry, item.key_path)
        config_key, values = _snapshot_key_from_record(
            registry, key._nkrecord, item.key_path)
    else:
        nk = _key_record_at(registry, item.nk_offset)
        config_key, values = _snapshot_key_from_record(
            registry, nk, item.key_path)
    if config_key is None:
        return
    # The key's own value list is authoritative for the identities it owns.
    # Drop only the previously known values of this key, then insert the
    # freshly read values.  This stays bounded by the key's value count.
    previous = result.keys.get(config_key.path)
    if previous is not None:
        for name in previous.value_names:
            result.values.pop((config_key.path, name), None)
    result.keys[config_key.path] = config_key
    for config_value in values:
        result.values[config_value.identity] = config_value


def _value_offsets_for_nk(registry, nk_offset):
    """
    Return the set of value-record cell offsets referenced by an NK record,
    reading the record and its value list directly.
    """
    cache = getattr(registry, "_incremental_nk_value_cache", None)
    if cache is not None and nk_offset in cache:
        return cache[nk_offset]
    offsets = set()
    try:
        nk = _key_record_at(registry, nk_offset)
        if nk.values_number():
            values_list = nk.values_list()
            for item in range(nk.values_number()):
                try:
                    offsets.add(FIRST_HBIN_OFFSET +
                                values_list.unpack_dword(4 * item))
                except Exception:
                    break
    except Exception:
        offsets = set()
    if cache is not None:
        cache[nk_offset] = offsets
    return offsets


def _value_belongs_to_key(registry, key_path, vk_record_offset,
                          nk_offset=None):
    """
    Check that a VK record is currently referenced by the value list of the
    owning key.  The value-list offsets are cached per key for the replay.
    """
    if nk_offset is not None:
        return (vk_record_offset - 0x4) in _value_offsets_for_nk(
            registry, nk_offset)

    cache = getattr(registry, "_incremental_key_value_cache", None)
    if cache is not None and key_path in cache:
        return (vk_record_offset - 0x4) in cache[key_path]
    try:
        nk = _open_key(registry, key_path)._nkrecord
        offsets = _value_offsets_for_nk(registry, nk.offset())
    except Exception:
        offsets = set()
    if cache is not None:
        cache[key_path] = offsets
    return (vk_record_offset - 0x4) in offsets


def _replay_value_item(registry, item, result):
    """Refresh one VK record, discovering its identity from the record."""
    key_path = item.key_path
    vk = None
    if item.vk_offset is not None:
        try:
            vk = _value_record_at(registry, item.vk_offset)
        except Exception:
            vk = None
        if vk is not None and not _value_belongs_to_key(
                registry, key_path, item.vk_offset,
                nk_offset=item.owner_nk_offset):
            # The cell was reused by another key.  Its owner's own checklist
            # entry will collect it; do not attach it to the old identity.
            vk = None
    if vk is None:
        # The physical VK cell was freed or reused; the baseline identity no
        # longer exists.  Do not guess a replacement.
        result.values.pop((key_path, item.value_name), None)
        config_key = result.keys.get(key_path)
        if config_key is not None and item.value_name in config_key.value_names:
            config_key.value_names = [name for name in config_key.value_names
                                      if name != item.value_name]
        return

    try:
        config_value = _snapshot_value_from_record(registry, vk, key_path)
    except Exception:
        return

    if item.value_name is not None and config_value.name != item.value_name:
        # The physical record now names a different identity.  Remove the
        # baseline identity and add the discovered one.
        result.values.pop((key_path, item.value_name), None)
        config_key = result.keys.get(key_path)
        if config_key is not None:
            config_key.value_names = [name for name in config_key.value_names
                                      if name != item.value_name]
    result.values[config_value.identity] = config_value


def _replay_value_list(registry, item, result):
    """Rediscover all values owned by the changed value list."""
    prefix_identity = item.key_path
    for identity in list(result.values.keys()):
        if identity[0] == prefix_identity:
            del result.values[identity]
    value_names = []
    for config_value in _all_values_for_key(registry, item.key_path):
        result.values[config_value.identity] = config_value
        value_names.append(config_value.name)
    config_key = result.keys.get(item.key_path)
    if config_key is not None:
        config_key.value_names = value_names


def _replay_rediscover_key(registry, item, result):
    """
    Rediscover a key from a new physical NK pointer.

    A newly pointed-to key may itself have descendants, so its subtree is
    collected.  This is still bounded by the new subtree rather than a
    scan of the hive.
    """
    if item.nk_offset is None:
        _replay_key_item(registry, item, result)
        return
    try:
        nk = _key_record_at(registry, item.nk_offset)
    except Exception:
        # The pointer does not resolve to an NK record in the recovered
        # hive; it was a transient or stale pointer.  Leave the baseline
        # identity untouched rather than guessing.
        return
    try:
        path = nk.path()
    except Exception:
        path = item.key_path
    if path is None:
        return
    _collect_branch(registry, path, result)


def replay_recollection(baseline_snapshot, plan, recovered_registry):
    """
    Apply a RecollectPlan to the baseline snapshot using a recovered hive.

    Only the planned identities are read; no full-hive traversal is used.
    The result is a ConfigSnapshot.  Compare it to
    `collect_config(recovered_registry)` to verify equivalence.
    """
    if not isinstance(recovered_registry, Registry.Registry):
        recovered_registry = Registry.Registry(recovered_registry)

    result = baseline_snapshot.copy()
    result.registry = recovered_registry
    recovered_registry._incremental_key_cache = {}
    recovered_registry._incremental_key_value_cache = {}
    recovered_registry._incremental_nk_value_cache = {}

    try:
        return _replay_plan_items(baseline_snapshot, plan, recovered_registry,
                                  result)
    finally:
        for attribute in ("_incremental_key_cache",
                          "_incremental_key_value_cache",
                          "_incremental_nk_value_cache"):
            try:
                delattr(recovered_registry, attribute)
            except AttributeError:
                pass


def _replay_plan_items(baseline_snapshot, plan, recovered_registry, result):

    # Apply explicit removals first so stale identities do not survive.
    _apply_removals(result, plan.removals)

    # Deduplicate by physical work item.  The plan lists each identity
    # explicitly, so replay only visits those exact records.
    handled_keys = set()
    handled_values = set()
    handled_lists = set()

    ordered = sorted(plan.items, key=lambda entry: (
        entry.mode not in ("rediscover_key", "key", "metadata"),
        (entry.key_path or "").count("\\"),
        entry.key_path or "",
        entry.value_name or ""))
    for item in ordered:
        if item.status == STATUS_PENDING and item.mode not in (
                "rediscover_value", "rediscover_values", "rediscover_key"):
            continue
        if item.mode == "metadata":
            handled_keys.add(item.key_path)
            _replay_key_item(recovered_registry, item, result)
        elif item.mode == "key":
            if item.key_path in handled_keys:
                continue
            handled_keys.add(item.key_path)
            _replay_key_item(recovered_registry, item, result)
        elif item.mode in ("vk", "rediscover_value"):
            token = (item.key_path, item.value_name, item.vk_offset)
            if token in handled_values:
                continue
            handled_values.add(token)
            _replay_value_item(recovered_registry, item, result)
        elif item.mode == "rediscover_values":
            if item.key_path in handled_lists:
                continue
            handled_lists.add(item.key_path)
            _replay_value_list(recovered_registry, item, result)
        elif item.mode == "rediscover_key":
            token = (item.nk_offset, item.key_path)
            if token in handled_keys or item.nk_offset is None:
                continue
            handled_keys.add(token)
            _replay_rediscover_key(recovered_registry, item, result)
        elif item.mode == "value":
            token = (item.key_path, item.value_name)
            if token in handled_values:
                continue
            handled_values.add(token)
            _recollect_single_value(recovered_registry, item.key_path,
                                    item.value_name, result)

    return result


class IncrementalCollector(object):
    """
    A cached collector that can plan and replay incremental updates.

    Example:
        collector = IncrementalCollector(Registry.Registry(baseline))
        plan = collector.plan(dirty_pages)
        snapshot = collector.replay(plan, Registry.Registry(recovered))
        assert snapshot == collect_config(Registry.Registry(recovered))
    """
    def __init__(self, registry):
        if not isinstance(registry, Registry.Registry):
            registry = Registry.Registry(registry)
        self.registry = registry
        self.index = DependencyIndex(registry)
        self.baseline = collect_config(registry)
        self.current = self.baseline.copy()

    def plan(self, dirty_pages, after_pages=None):
        return plan_recollection(self.index, dirty_pages,
                                 after_pages=after_pages)

    def replay(self, plan, recovered_registry):
        self.current = replay_recollection(self.baseline, plan,
                                           recovered_registry)
        return self.current

    def export(self):
        return self.current.export()

    def verify(self, recovered_registry):
        """
        Return True when the replayed current snapshot equals a full
        collection of the recovered hive.
        """
        if not isinstance(recovered_registry, Registry.Registry):
            recovered_registry = Registry.Registry(recovered_registry)
        return self.current == collect_config(recovered_registry)


# Names that make the intended workflow discoverable from the package.
build_index = build_dependency_index
plan = plan_recollection
replay = replay_recollection

__all__ = [
    "DirtyPageRegion",
    "dirty_pages_from_log",
    "dirty_pages_from_logs",
    "Dependency",
    "DependencyIndex",
    "build_dependency_index",
    "build_index",
    "ChangeTrigger",
    "RecollectItem",
    "Removal",
    "RecollectPlan",
    "plan_recollection",
    "plan",
    "ConfigValue",
    "ConfigKey",
    "ConfigSnapshot",
    "collect_config",
    "replay_recollection",
    "replay",
    "IncrementalCollector",
    "RecollectError",
    "STATUS_RECOLLECT",
    "STATUS_PENDING",
]
