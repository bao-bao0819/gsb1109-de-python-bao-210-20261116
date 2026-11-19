#!/bin/python

#    This file is part of python-registry.
#
#   Copyright 2011 Will Ballenthin <william.ballenthin@mandiant.com>
#                    while at Mandiant <http://www.mandiant.com>
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
Offset cache support for incremental collection of registry hives.

A hive reorganization (compaction) moves NK/VK cells around inside the
HBINs.  An offset cached from a previous snapshot may therefore still point
at a perfectly readable cell whose contents now belong to a different key
or value.  The file name and file size cannot detect this situation, so the
cache is bound to:

  1. the source version markers stored inside the REGF header (hive name,
     reorganized timestamp, modification timestamp, sequence numbers and
     HBIN size), and
  2. the logical object addressed by an entry, i.e. the full key path plus
     the value name.

Cached offsets are never trusted on their own.  Every resolution walks the
live hive from the root key to the logical object, verifies the cell found
at the cached offset, and relocates the entry when the hive was reorganized
or the cell moved.  A content fingerprint then decides whether the value has
to be booked by the incremental collector: values whose content is unchanged
are reported as UNCHANGED and must not enter the business ledger twice.
"""
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import struct
import json
import hashlib
import datetime
from collections import namedtuple
from enum import Enum

from .Registry import RegistryKeyNotFoundException, RegistryValueNotFoundException
from . import RegistryParse


CACHE_FORMAT_VERSION = 1

# Marker written into a cache entry when no live offset is known.
UNKNOWN_OFFSET = 0xFFFFFFFF


class StaleOffsetCacheException(RegistryParse.RegistryException):
    """
    Raised when a cache was produced by a different source hive.
    """
    def __str__(self):
        return "Stale Offset Cache Exception: %s" % (self._value)


class ChangeType(Enum):
    """
    The content state of a cached value during an incremental run.
    Only NEW and CHANGED entries are meant to be booked.
    """
    NEW = "new"
    CHANGED = "changed"
    UNCHANGED = "unchanged"
    MISSING = "missing"


# Markers describing the exact source version a cache applies to.
SourceState = namedtuple("SourceState", [
    "hive_name",
    "reorganized_timestamp",
    "modification_timestamp",
    "hive_sequence1",
    "hive_sequence2",
    "hbins_size",
])

# A single cached offset.
CacheEntry = namedtuple("CacheEntry", [
    "key_path",
    "value_name",
    "offset",
    "fingerprint",
    "data_type",
    "data_length",
])

# Result of resolving one logical object against the live hive.
Resolution = namedtuple("Resolution", [
    "key_path",
    "value_name",
    "change",
    "reorganized",
    "offset_changed",
    "cached_offset",
    "current_offset",
    "cached_fingerprint",
    "current_fingerprint",
    "value",
])


def value_fingerprint(value):
    """
    Compute a stable content fingerprint for a RegistryValue.

    The fingerprint covers the data type and the raw value bytes, so it
    survives cell relocation and hive reorganization as long as the logical
    value's content is untouched.
    """
    raw = value.raw_data()
    digest = hashlib.sha256()
    digest.update(struct.pack(str("<I"), value.value_type()))
    digest.update(struct.pack(str("<Q"), len(raw)))
    digest.update(raw)
    return digest.hexdigest()


def _walk_values(key):
    """
    Yield (RegistryKey, RegistryValue) pairs for every value below `key`.
    """
    for value in key.values():
        yield (key, value)
    for subkey in key.subkeys():
        for pair in _walk_values(subkey):
            yield pair


def _relative_key_path(registry, key):
    """
    The logical key path relative to the hive root, as accepted by
    Registry.open().  The root key itself is the empty string.
    """
    root_path = registry.root().path()
    path = key.path()
    if path == root_path:
        return ""
    if path.startswith(root_path + "\\"):
        return path[len(root_path) + 1:]
    return path


class OffsetCache(object):
    """
    A persistable offset cache bound to a specific hive source version.

    Entries are keyed by the logical object (key path, value name).  The
    stored offset is a convenience for unchanged hives only; the logical
    identity and the content fingerprint are always authoritative.
    """
    def __init__(self, source=None, entries=None):
        self._source = source
        self._entries = entries if entries is not None else {}

    #
    # construction
    #
    @classmethod
    def build(cls, registry):
        """
        Build a fresh cache from the current state of a Registry.
        """
        cache = cls(source=cls._source_state(registry))
        root = registry.root()
        for key, value in _walk_values(root):
            key_path = _relative_key_path(registry, key)
            cache._entries[cls._entry_key(key_path, value.name())] = CacheEntry(
                key_path=key_path,
                value_name=value.name(),
                offset=value._vkrecord.offset(),
                fingerprint=value_fingerprint(value),
                data_type=value.value_type(),
                data_length=len(value.raw_data()),
            )
        return cache

    @staticmethod
    def _entry_key(key_path, value_name):
        return (key_path, value_name)

    @staticmethod
    def _source_state(registry):
        regf = registry._regf
        return SourceState(
            hive_name=regf.hive_name(),
            reorganized_timestamp=regf.reorganized_timestamp(),
            modification_timestamp=regf.modification_timestamp(),
            hive_sequence1=regf.hive_sequence1(),
            hive_sequence2=regf.hive_sequence2(),
            hbins_size=regf.hbins_size(),
        )

    #
    # source version validation
    #
    def source_state(self):
        return self._source

    def source_matches(self, registry):
        """
        Does this cache apply to the given hive?
        Only internal REGF markers are considered; the external file name
        and file size are intentionally ignored.
        """
        if self._source is None:
            return False
        return self._source == self._source_state(registry)

    def reorganization_detected(self, registry):
        """
        True when the live hive carries a different reorganized timestamp,
        i.e. its cells were rearranged after the cache was built.
        """
        if self._source is None:
            return False
        return self._source.reorganized_timestamp != \
            registry._regf.reorganized_timestamp()

    def validate_source(self, registry):
        """
        Ensure the cache applies to this exact hive source version.
        Raises StaleOffsetCacheException otherwise and returns the live
        SourceState on success.
        """
        current = self._source_state(registry)
        if self._source is None:
            raise StaleOffsetCacheException("cache has no source version markers")
        if self._source.hive_name != current.hive_name:
            raise StaleOffsetCacheException(
                "hive name mismatch: cache is for %r, got %r" %
                (self._source.hive_name, current.hive_name))
        if self._source.reorganized_timestamp != current.reorganized_timestamp:
            raise StaleOffsetCacheException(
                "hive reorganization detected: cached reorganized timestamp %s, current %s" %
                (self._source.reorganized_timestamp, current.reorganized_timestamp))
        if self._source != current:
            raise StaleOffsetCacheException(
                "source version mismatch: cache %r, current %r" %
                (self._source, current))
        return current

    #
    # offset verification
    #
    @staticmethod
    def _cell_at_offset(registry, offset):
        """
        Read the HBINCell whose data block starts at `offset`.
        Returns None when the offset does not point at an allocated cell.
        """
        if offset is None or offset == UNKNOWN_OFFSET or offset < 4:
            return None
        cell_offset = offset - 4
        try:
            cell = RegistryParse.HBINCell(registry._buf, cell_offset, registry._regf)
        except (RegistryParse.ParseException, struct.error, IndexError):
            return None
        if cell.is_free():
            return None
        return cell

    def cached_offset_is_valid(self, registry, key_path, value_name):
        """
        Verify that the cached offset still points at an allocated VK cell
        carrying the expected value name.  A readable cell is not enough by
        itself: callers must still confirm the logical owner through the
        key path before trusting any data.
        """
        entry = self._entries.get(self._entry_key(key_path, value_name))
        if entry is None:
            return False
        cell = self._cell_at_offset(registry, entry.offset)
        if cell is None or cell.data_id() != b"vk":
            return False
        try:
            vk = RegistryParse.VKRecord(registry._buf, cell.data_offset(), cell)
        except (RegistryParse.ParseException, struct.error):
            return False
        return vk.name() == self._public_to_internal_name(value_name)

    @staticmethod
    def _public_to_internal_name(value_name):
        return "" if value_name == "(default)" else value_name

    #
    # logical resolution and incremental accounting
    #
    def _resolve_value(self, registry, key_path, value_name):
        """
        Locate a value through its logical object.  This walk is the sole
        source of truth for the returned data; cached offsets never feed
        data back to the caller directly.
        """
        try:
            key = registry.open(key_path)
            return key.value(self._public_to_internal_name(value_name))
        except (RegistryKeyNotFoundException, RegistryValueNotFoundException):
            return None

    def resolve(self, registry, key_path, value_name):
        """
        Resolve one logical value against the live hive and classify it.

        The cached offset is validated against the live cell, but the value
        always comes from a root-to-object traversal, so a stale offset that
        lands on a foreign cell is detected and relocated instead of being
        read as the requested value.
        """
        reorganized = self.reorganization_detected(registry)
        entry = self._entries.get(self._entry_key(key_path, value_name))
        cached_offset = entry.offset if entry is not None else UNKNOWN_OFFSET
        cached_fingerprint = entry.fingerprint if entry is not None else None

        value = self._resolve_value(registry, key_path, value_name)
        if value is None:
            return Resolution(
                key_path=key_path,
                value_name=value_name,
                change=ChangeType.MISSING,
                reorganized=reorganized,
                offset_changed=entry is not None,
                cached_offset=cached_offset,
                current_offset=UNKNOWN_OFFSET,
                cached_fingerprint=cached_fingerprint,
                current_fingerprint=None,
                value=None,
            )

        current_offset = value._vkrecord.offset()
        current_fingerprint = value_fingerprint(value)
        offset_changed = entry is None or entry.offset != current_offset

        if entry is None:
            change = ChangeType.NEW
        elif entry.fingerprint != current_fingerprint:
            change = ChangeType.CHANGED
        else:
            change = ChangeType.UNCHANGED

        return Resolution(
            key_path=key_path,
            value_name=value_name,
            change=change,
            reorganized=reorganized,
            offset_changed=offset_changed,
            cached_offset=cached_offset,
            current_offset=current_offset,
            cached_fingerprint=cached_fingerprint,
            current_fingerprint=current_fingerprint,
            value=value,
        )

    def changes(self, registry):
        """
        Run an incremental collection pass.

        Yields one Resolution per logical value in the live hive, plus a
        MISSING resolution for cached objects that no longer exist.  Values
        with unchanged content are reported as UNCHANGED even when the hive
        was reorganized and every offset had to be relocated.
        """
        seen = set()
        for key, value in _walk_values(registry.root()):
            key_path = _relative_key_path(registry, key)
            logical = self._entry_key(key_path, value.name())
            seen.add(logical)
            yield self.resolve(registry, key_path, value.name())

        for logical, entry in self._entries.items():
            if logical not in seen:
                yield Resolution(
                    key_path=entry.key_path,
                    value_name=entry.value_name,
                    change=ChangeType.MISSING,
                    reorganized=self.reorganization_detected(registry),
                    offset_changed=True,
                    cached_offset=entry.offset,
                    current_offset=UNKNOWN_OFFSET,
                    cached_fingerprint=entry.fingerprint,
                    current_fingerprint=None,
                    value=None,
                )

    def bookable(self, registry):
        """
        Return the resolutions that must enter the business ledger:
        new values and values whose content changed.  Relocated but
        content-identical values are excluded so they are not booked twice.
        """
        return [r for r in self.changes(registry)
                if r.change in (ChangeType.NEW, ChangeType.CHANGED)]

    def relocate(self, registry):
        """
        Re-bind every cache entry to the live hive after a reorganization.

        Missing objects are dropped.  The source markers and offsets are
        refreshed, while fingerprints are taken from the live content so the
        next incremental run compares against the new baseline.
        """
        relocated = OffsetCache(source=self._source_state(registry))
        for key, value in _walk_values(registry.root()):
            key_path = _relative_key_path(registry, key)
            relocated._entries[self._entry_key(key_path, value.name())] = CacheEntry(
                key_path=key_path,
                value_name=value.name(),
                offset=value._vkrecord.offset(),
                fingerprint=value_fingerprint(value),
                data_type=value.value_type(),
                data_length=len(value.raw_data()),
            )
        self._source = relocated._source
        self._entries = relocated._entries
        return self

    #
    # persistence
    #
    def to_dict(self):
        return {
            "format_version": CACHE_FORMAT_VERSION,
            "source": self._source_to_dict(self._source),
            "entries": [
                {
                    "key_path": entry.key_path,
                    "value_name": entry.value_name,
                    "offset": entry.offset,
                    "fingerprint": entry.fingerprint,
                    "data_type": entry.data_type,
                    "data_length": entry.data_length,
                }
                for entry in self._entries.values()
            ],
        }

    @classmethod
    def from_dict(cls, data):
        if data.get("format_version") != CACHE_FORMAT_VERSION:
            raise StaleOffsetCacheException(
                "unsupported cache format version: %r" %
                (data.get("format_version"),))
        source = cls._source_from_dict(data.get("source"))
        entries = {}
        for raw in data.get("entries", []):
            entry = CacheEntry(
                key_path=raw["key_path"],
                value_name=raw["value_name"],
                offset=raw["offset"],
                fingerprint=raw["fingerprint"],
                data_type=raw["data_type"],
                data_length=raw["data_length"],
            )
            entries[cls._entry_key(entry.key_path, entry.value_name)] = entry
        return cls(source=source, entries=entries)

    def save(self, filelikeobject):
        """
        Serialize the cache to a binary file-like object or a filename.
        """
        payload = _json_dumps(self.to_dict()).encode("utf-8")
        try:
            filelikeobject.write(payload)
        except AttributeError:
            with open(filelikeobject, "wb") as f:
                f.write(payload)

    @classmethod
    def load(cls, filelikeobject):
        """
        Load a cache from a binary file-like object or a filename.
        Use validate_source() against the live hive before relying on it.
        """
        try:
            payload = filelikeobject.read()
        except AttributeError:
            with open(filelikeobject, "rb") as f:
                payload = f.read()
        return cls.from_dict(_json_loads(payload.decode("utf-8")))

    #
    # source (de)serialization helpers
    #
    @staticmethod
    def _source_to_dict(source):
        if source is None:
            return None
        return {
            "hive_name": source.hive_name,
            "reorganized_timestamp": source.reorganized_timestamp.isoformat(),
            "modification_timestamp": source.modification_timestamp.isoformat(),
            "hive_sequence1": source.hive_sequence1,
            "hive_sequence2": source.hive_sequence2,
            "hbins_size": source.hbins_size,
        }

    @staticmethod
    def _source_from_dict(data):
        if data is None:
            return None
        return SourceState(
            hive_name=data["hive_name"],
            reorganized_timestamp=_parse_timestamp(data["reorganized_timestamp"]),
            modification_timestamp=_parse_timestamp(data["modification_timestamp"]),
            hive_sequence1=data["hive_sequence1"],
            hive_sequence2=data["hive_sequence2"],
            hbins_size=data["hbins_size"],
        )


def _parse_timestamp(text):
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise StaleOffsetCacheException("invalid timestamp in cache: %r" % (text,))


def _json_dumps(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _json_loads(text):
    return json.loads(text)
