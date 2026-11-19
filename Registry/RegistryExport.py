#!/bin/python

#    This file is part of python-registry.
#
#   Copyright 2026
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
Export effective registry configuration separately from forensic residue.

``Registry.Registry`` follows only the live key tree.  That is the right source
for "currently effective configuration", but records can remain readable in
unallocated (free) cells long after they were deleted.  This module builds a
report with three disjoint evidence classes:

``active``
    Values referenced by the value list of a key reachable from the hive root,
    whose value record and value data are both in allocated cells.  Only these
    rows describe current configuration.

``residual``
    Value records physically located in free cells.  They are observations of
    old data and are never added to ``active``.

``uncertain``
    Rows whose reachability cannot be established: a free record whose owner
    key has a broken parent chain, an allocated record that no live key
    references, or a value referenced by a live key whose cell/data is not
    allocated.  Such rows are not labelled as deleted.

Textual paths are *not* identities.  A residual record whose reconstructed
path and value name equal an active record is kept as a separate observation
and receives a ``collision`` marker.  Consumers must group by physical
identity, not by path alone.

Every observation carries a typed ``evidence`` list and a ``location`` mapping
with absolute file offsets, HBIN offsets and cell offsets for the record and
for its value data.
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import argparse
import binascii
import bisect
import datetime
import json
import struct
import sys

from . import RegistryParse

ACTIVE = "active"
RESIDUAL = "residual"
UNCERTAIN = "uncertain"

SCHEMA = "python-registry/config-export/1"

_DEVPROP_MASK_TYPE = 0x00000FFF
_MAX_NAME = 0x4000
_MAX_VALUE_DATA = 0x7FFFFFFF
_MAX_EXPORTED_DATA = 0x00100000
_MAX_ENTRIES = 0x00100000
_NK_FLAG_MASK = 0x007F

try:
    _TEXT_TYPES = (str, unicode)
except NameError:  # Python 3
    _TEXT_TYPES = (str,)

if str is bytes:  # Python 2: str is a byte string, but still a valid path.
    _BINARY_SOURCE_TYPES = (bytearray,)
else:
    _BINARY_SOURCE_TYPES = (bytes, bytearray)


class ConfigExportReport(object):
    """
    Result of :func:`classify`.

    ``active`` is the only list that describes currently effective
    configuration.  ``residual`` and ``uncertain`` are forensic observations.
    """

    def __init__(self, hive, active, residual, uncertain, collisions,
                 structural_issues, query=None):
        self.hive = hive
        self.active = active
        self.residual = residual
        self.uncertain = uncertain
        self.collisions = collisions
        self.structural_issues = structural_issues
        self.query = query or {}

    def to_dict(self):
        return {
            "schema": SCHEMA,
            "hive": self.hive,
            "query": self.query,
            "summary": {
                "active": len(self.active),
                "residual": len(self.residual),
                "uncertain": len(self.uncertain),
                "collisions": len(self.collisions),
                "structural_issues": len(self.structural_issues),
                "unique_active_paths": len(set(
                    item.get("path") for item in self.active
                    if item.get("path") is not None)),
                "residual_cell_states": _cell_state_counts(self.residual),
                "uncertain_cell_states": _cell_state_counts(self.uncertain),
            },
            "active": self.active,
            "residual": self.residual,
            "uncertain": self.uncertain,
            "collisions": self.collisions,
            "structural_issues": self.structural_issues,
        }

    def to_json(self, indent=2, **kwargs):
        return json.dumps(self.to_dict(), indent=indent,
                          ensure_ascii=False, sort_keys=False, **kwargs)


def _cell_state_counts(items):
    counts = {}
    for item in items:
        state = item.get("location", {}).get("cell_state") or "unknown"
        counts[state] = counts.get(state, 0) + 1
    return counts


def _hex(value):
    if value is None:
        return None
    return "0x%x" % value


class _CellInfo(object):
    __slots__ = ("offset", "size", "is_free", "end", "hbin_offset",
                 "hbin_end")

    def __init__(self, offset, raw_size, hbin_offset, hbin_end):
        self.offset = offset
        self.size = abs(raw_size)
        self.is_free = raw_size > 0
        self.end = offset + self.size
        self.hbin_offset = hbin_offset
        self.hbin_end = hbin_end

    @property
    def data_offset(self):
        return self.offset + 4

    @property
    def state(self):
        return "free" if self.is_free else "allocated"


class _KeyInfo(object):
    __slots__ = ("record_offset", "cell_offset", "name", "flags",
                 "parent_rel", "subkey_number", "values_number",
                 "values_list_rel", "cell_state", "record_size")

    def __init__(self, header, cell):
        self.record_offset = header["record_offset"]
        self.cell_offset = cell.offset if cell is not None else None
        self.name = header["name"]
        self.flags = header["flags"]
        self.parent_rel = header["parent_rel"]
        self.subkey_number = header["subkey_number"]
        self.values_number = header["values_number"]
        self.values_list_rel = header["values_list_rel"]
        self.cell_state = cell.state if cell is not None else "missing"
        self.record_size = header["record_size"]

    @property
    def is_root(self):
        return bool(self.flags & 0x0004)


class _CellIndex(object):
    def __init__(self, cells):
        self._cells = sorted(cells, key=lambda cell: cell.offset)
        self._starts = [cell.offset for cell in self._cells]
        self._by_offset = dict((cell.offset, cell) for cell in self._cells)

    def at(self, offset):
        return self._by_offset.get(offset)

    def containing(self, offset):
        index = bisect.bisect_right(self._starts, offset) - 1
        if index < 0:
            return None
        cell = self._cells[index]
        if cell.offset <= offset < cell.end:
            return cell
        return None

    def __iter__(self):
        return iter(self._cells)


def _hbin_ranges(buf, regf, issues):
    """
    Yield (hbin_offset, hbin_end) walking HBIN headers directly.

    This is intentionally independent of ``REGFBlock.hbins()`` so a single
    corrupt link does not hide every later cell from the export.
    """
    first = regf.first_hbin_offset()
    hbins_size = regf.hbins_size()
    file_end = len(buf)
    if hbins_size <= 0 or first + hbins_size > file_end:
        hbins_size = max(0, file_end - first)

    offset = first
    limit = first + hbins_size
    seen = set()
    while offset + 0x20 <= limit:
        if offset in seen:
            issues.append({
                "reason": "HBIN cycle",
                "location": {"file_offset": offset},
            })
            break
        seen.add(offset)
        if buf[offset:offset + 4] != b"hbin":
            issues.append({
                "reason": "invalid HBIN signature",
                "location": {"file_offset": offset},
            })
            break
        try:
            size = struct.unpack_from("<I", buf, offset + 0x8)[0]
        except struct.error:
            break
        if size < 0x1000 or offset + size > file_end:
            issues.append({
                "reason": "HBIN size is invalid",
                "location": {"file_offset": offset},
                "evidence": {"hbin_size": size},
            })
            break
        yield offset, offset + size
        offset += size


def _read_cells(buf, regf, issues):
    cells = []
    for hbin_offset, hbin_end in _hbin_ranges(buf, regf, issues):
        offset = hbin_offset + 0x20
        while offset + 4 <= hbin_end:
            try:
                raw_size = struct.unpack_from("<i", buf, offset)[0]
            except struct.error:
                break
            if raw_size == 0:
                issues.append({
                    "reason": "zero-sized cell stops HBIN traversal",
                    "location": {"file_offset": offset,
                                 "hbin_offset": hbin_offset},
                })
                break
            size = abs(raw_size)
            if size < 8 or offset + size > hbin_end:
                issues.append({
                    "reason": "cell extends outside its HBIN",
                    "location": {
                        "file_offset": offset,
                        "cell_offset": offset,
                        "cell_size": size,
                        "cell_state": "free" if raw_size > 0 else "allocated",
                        "hbin_offset": hbin_offset,
                        "hbin_end": hbin_end,
                    },
                })
                break
            cells.append(_CellInfo(offset, raw_size, hbin_offset, hbin_end))
            if offset + size == hbin_end:
                break
            offset += size
    return cells


def _decode_name(raw, is_ascii, name_length):
    if name_length == 0:
        return "", True
    if not is_ascii and name_length % 2 != 0:
        return None, False
    try:
        name = raw.decode("windows-1252" if is_ascii else "utf-16le")
    except (UnicodeDecodeError, LookupError):
        return None, False
    if "\x00" in name:
        return None, False
    for char in name:
        if ord(char) < 0x20 and char != "\t":
            return None, False
    return name, True


def _parse_vk_header(buf, record_offset, cell_end):
    """
    Validate the fixed part and name of a VK record.

    Random occurrences of ``vk`` inside free cells are rejected unless the
    rest of the header is structurally plausible.
    """
    limit = min(cell_end, len(buf))
    if record_offset < 0 or record_offset + 0x14 > limit:
        return None
    if buf[record_offset:record_offset + 2] != b"vk":
        return None

    name_length = struct.unpack_from("<H", buf, record_offset + 2)[0]
    flags = struct.unpack_from("<H", buf, record_offset + 0x10)[0]
    data_length_raw = struct.unpack_from("<I", buf, record_offset + 4)[0]
    data_offset_rel = struct.unpack_from("<I", buf, record_offset + 8)[0]
    data_type = struct.unpack_from("<I", buf, record_offset + 0xC)[0]
    data_type &= _DEVPROP_MASK_TYPE

    if name_length > _MAX_NAME or flags & ~0x0001:
        return None
    if data_type > 0x11F:
        return None
    record_end = record_offset + 0x14 + name_length
    if record_end > limit:
        return None

    if data_length_raw >= 0x80000000:
        data_length = data_length_raw - 0x80000000
        resident = True
    elif data_length_raw < 5:
        data_length = data_length_raw
        resident = True
    else:
        data_length = data_length_raw
        resident = False
    if data_length > _MAX_VALUE_DATA:
        return None

    name, name_valid = _decode_name(
        buf[record_offset + 0x14:record_end], flags & 0x0001, name_length)
    if name_length and not name_valid:
        return None

    return {
        "record_offset": record_offset,
        "record_end": record_end,
        "record_size": record_end - record_offset,
        "name": name,
        "name_length": name_length,
        "name_valid": name_valid,
        "data_length_raw": data_length_raw,
        "data_length": data_length,
        "data_offset_rel": data_offset_rel,
        "data_type": data_type,
        "flags": flags,
        "resident": resident,
    }


def _parse_nk_header(buf, record_offset, cell_end):
    limit = min(cell_end, len(buf))
    if record_offset < 0 or record_offset + 0x4C > limit:
        return None
    if buf[record_offset:record_offset + 2] != b"nk":
        return None

    flags = struct.unpack_from("<H", buf, record_offset + 2)[0]
    if flags & ~_NK_FLAG_MASK:
        return None
    parent_rel = struct.unpack_from("<I", buf, record_offset + 0x10)[0]
    subkey_number = struct.unpack_from("<I", buf, record_offset + 0x14)[0]
    subkey_list_rel = struct.unpack_from("<I", buf, record_offset + 0x1C)[0]
    values_number = struct.unpack_from("<I", buf, record_offset + 0x24)[0]
    values_list_rel = struct.unpack_from("<I", buf, record_offset + 0x28)[0]
    name_length = struct.unpack_from("<H", buf, record_offset + 0x48)[0]

    if name_length > _MAX_NAME or subkey_number > _MAX_ENTRIES or \
            values_number > _MAX_ENTRIES:
        return None

    # The library treats the all-ones sentinel as "none".
    if subkey_number == 0xFFFFFFFF:
        subkey_number = 0
    if values_number == 0xFFFFFFFF:
        values_number = 0
    record_end = record_offset + 0x4C + name_length
    if record_end > limit:
        return None

    name, name_valid = _decode_name(
        buf[record_offset + 0x4C:record_end], flags & 0x0020, name_length)
    if name_length and not name_valid:
        return None

    # A plausible record must not claim children or values without a list
    # pointer.  This is what keeps random "nk" byte pairs inside free cells
    # or value data from becoming key candidates.
    if values_number not in (0, 0xFFFFFFFF) and values_list_rel == 0xFFFFFFFF:
        return None
    if subkey_number not in (0, 0xFFFFFFFF) and subkey_list_rel == 0xFFFFFFFF:
        return None

    return {
        "record_offset": record_offset,
        "record_end": record_end,
        "record_size": record_end - record_offset,
        "name": name,
        "name_length": name_length,
        "name_valid": name_valid,
        "flags": flags,
        "parent_rel": parent_rel,
        "subkey_number": subkey_number,
        "subkey_list_rel": subkey_list_rel,
        "values_number": values_number,
        "values_list_rel": values_list_rel,
    }


def _candidate_offsets(buf, cell):
    """
    Possible NK/VK record offsets inside a cell.

    An allocated cell has exactly one record at its data offset.  A free cell
    may have been coalesced from several old cells, so aligned signatures are
    searched across the whole payload.
    """
    if not cell.is_free:
        return [cell.data_offset]

    offsets = set()
    for signature in (b"nk", b"vk"):
        index = buf.find(signature, cell.data_offset, cell.end)
        while index != -1:
            if (index - cell.data_offset) % 4 == 0:
                offsets.add(index)
            index = buf.find(signature, index + 1, cell.end)
    return sorted(offsets)


def _structural_cell(index, pointer):
    """
    Resolve a pointer that must name a cell start.

    Live structural pointers (value lists, subkey lists, RI entries) address a
    cell header.  A carved record inside a coalesced free cell may still point
    at its old header, which is embedded in a larger free cell.  Pointers into
    allocated payload bytes are rejected rather than guessed at.
    """
    cell = index.at(pointer)
    if cell is not None:
        return cell
    container = index.containing(pointer)
    if container is not None and container.is_free:
        return container
    return None


def _list_payload_offset(buf, list_cell, pointer):
    """
    Resolve the byte offset of a value list.

    A live list pointer names a cell, so its entries start after the 4-byte
    cell header.  A pointer into a coalesced free cell names an old cell that
    no longer exists as its own allocation; its header still precedes the
    entries.  Treat both the same way.
    """
    if pointer == list_cell.offset:
        return list_cell.data_offset
    if pointer < list_cell.data_offset:
        return list_cell.data_offset
    return pointer + 4


def _read_value_list(buf, list_cell, count, first_hbin, index, pointer=None):
    """Return [(value_cell_offset, list_cell_offset, entry_index, pointer)]."""
    entries = []
    if list_cell is None:
        return entries
    if pointer is None:
        pointer = list_cell.offset
    start = _list_payload_offset(buf, list_cell, pointer)
    available = max(0, (list_cell.end - start) // 4)
    for entry_index in range(min(count, available)):
        try:
            rel = struct.unpack_from("<I", buf, start + entry_index * 4)[0]
        except struct.error:
            break
        if rel == 0xFFFFFFFF:
            continue
        value_pointer = first_hbin + rel
        cell = index.at(value_pointer) or index.containing(value_pointer)
        if cell is not None:
            entries.append((cell.offset, list_cell.offset, entry_index,
                            value_pointer))
    return entries


def _read_subkey_list(buf, list_cell, first_hbin, index, issues, depth=0):
    """
    Return (key_cell_offsets, list_cell_offsets) following RI indirection.
    """
    if list_cell is None or depth > 8:
        return [], []
    start = list_cell.data_offset
    signature = buf[start:start + 2]
    list_cells = [list_cell.offset]

    if signature not in (b"lf", b"lh", b"li", b"ri"):
        issues.append({
            "reason": "unsupported or corrupt subkey list",
            "location": _cell_location(list_cell),
            "evidence": {"type": "subkey_list",
                         "signature": signature.decode("latin-1")},
        })
        return [], list_cells
    try:
        count = struct.unpack_from("<H", buf, start + 2)[0]
    except struct.error:
        return [], list_cells

    if signature == b"ri":
        keys = []
        available = max(0, (list_cell.end - start - 4) // 4)
        for entry_index in range(min(count, available)):
            rel = struct.unpack_from("<I", buf, start + 4 + entry_index * 4)[0]
            sublist = _structural_cell(index, first_hbin + rel)
            if sublist is None:
                issues.append({
                    "reason": "bad RI subkey-list reference",
                    "location": dict(_cell_location(list_cell),
                                     referenced_offset=first_hbin + rel),
                    "evidence": {"type": "subkey_list",
                                 "signature": "ri",
                                 "entry_index": entry_index},
                })
                continue
            subkeys, sublists = _read_subkey_list(
                buf, sublist, first_hbin, index, issues, depth + 1)
            keys.extend(subkeys)
            list_cells.extend(sublists)
        return keys, list_cells

    entry_size = 8 if signature in (b"lf", b"lh") else 4
    available = max(0, (list_cell.end - start - 4) // entry_size)
    keys = []
    for entry_index in range(min(count, available)):
        position = start + 4 + entry_index * entry_size
        rel = struct.unpack_from("<I", buf, position)[0]
        keys.append(first_hbin + rel)
    return keys, list_cells


def _cell_location(cell, record_offset=None):
    location = {
        "file_offset": record_offset,
        "file_offset_hex": _hex(record_offset),
        "cell_offset": cell.offset if cell is not None else None,
        "cell_offset_hex": _hex(cell.offset) if cell is not None else None,
        "cell_size": cell.size if cell is not None else None,
        "cell_state": cell.state if cell is not None else "missing",
        "hbin_offset": cell.hbin_offset if cell is not None else None,
        "hbin_offset_hex": _hex(cell.hbin_offset) if cell is not None else None,
    }
    if cell is not None and record_offset is not None:
        location["record_offset_in_cell"] = record_offset - cell.data_offset
        location["embedded_in_free_cell"] = bool(
            cell.is_free and record_offset != cell.data_offset)
    return location


def _payload_span(buf, index, pointer, cell):
    """
    Resolve a data pointer to (payload_offset, payload_end, owner_cell).

    ``pointer`` may address a live cell or an old cell header that survives
    inside a coalesced free cell.  In the latter case the old cell size still
    bounds the data.
    """
    live = index.at(pointer)
    if live is not None:
        return live.data_offset, live.end, live

    container = index.containing(pointer)
    if container is None:
        return pointer + 4, pointer + 4, None

    end = container.end
    if pointer + 4 <= container.end:
        try:
            old_size = struct.unpack_from("<i", buf, pointer)[0]
        except struct.error:
            old_size = 0
        if old_size != 0:
            old_end = pointer + abs(old_size)
            if pointer < old_end <= container.end:
                end = old_end
    return pointer + 4, end, container


def _data_description(buf, header, first_hbin, index, cell):
    """
    Return (raw_bytes, description).  ``raw_bytes`` may be None if the data
    cell is missing.  The description always records the pointer and whether
    the destination is allocated.
    """
    data_length = header["data_length"]

    if header["resident"] and header["data_type"] == RegistryParse.RegMultiSZ \
            and header["data_length_raw"] >= 0x80000000:
        # The library treats a high-bit multi-string length as empty.
        return b"", {
            "resident": True,
            "pointer": None,
            "pointer_hex": None,
            "file_offset": header["record_offset"] + 8,
            "file_offset_hex": _hex(header["record_offset"] + 8),
            "length": 0,
            "available": 0,
            "cell_offset": cell.offset if cell is not None else None,
            "cell_offset_hex": _hex(cell.offset) if cell is not None else None,
            "cell_state": cell.state if cell is not None else None,
            "allocated": True,
            "chunks": [],
        }

    if header["resident"]:
        start = header["record_offset"] + 8
        available = max(0, min(data_length, 4, len(buf) - start))
        raw = buf[start:start + available]
        return raw, {
            "resident": True,
            "pointer": None,
            "pointer_hex": None,
            "file_offset": start,
            "file_offset_hex": _hex(start),
            "length": data_length,
            "available": available,
            "cell_offset": cell.offset if cell is not None else None,
            "cell_offset_hex": _hex(cell.offset) if cell is not None else None,
            "cell_state": cell.state if cell is not None else None,
            "allocated": True,
            "chunks": [],
        }

    pointer = first_hbin + header["data_offset_rel"]
    payload_offset, payload_end, data_cell = _payload_span(
        buf, index, pointer, cell)

    description = {
        "resident": False,
        "pointer": pointer,
        "pointer_hex": _hex(pointer),
        "file_offset": payload_offset,
        "file_offset_hex": _hex(payload_offset),
        "length": data_length,
        "available": 0,
        "cell_offset": data_cell.offset if data_cell is not None else None,
        "cell_offset_hex": _hex(data_cell.offset) if data_cell is not None else None,
        "cell_state": data_cell.state if data_cell is not None else "missing",
        "allocated": bool(data_cell is not None and not data_cell.is_free),
        "chunks": [],
    }
    if data_cell is None:
        return None, description

    # Large values use a db record: [db][pointer to indirect cell], and the
    # indirect cell holds pointers to the actual data cells.
    if data_length > 0x3FD8 and buf[payload_offset:payload_offset + 2] == b"db":
        try:
            indirect_rel = struct.unpack_from("<I", buf, payload_offset + 4)[0]
        except struct.error:
            indirect_rel = 0
        indirect_pointer = first_hbin + indirect_rel
        indirect_payload, indirect_end, indirect_cell = _payload_span(
            buf, index, indirect_pointer, cell)
        if indirect_cell is None:
            description["chunks"] = []
            return None, description

        chunks = []
        parts = []
        remaining = data_length
        count = (data_length + 0x3FD7) // 0x3FD8
        for chunk_index in range(min(count, 0x10000)):
            position = indirect_payload + chunk_index * 4
            if position + 4 > indirect_end:
                break
            rel = struct.unpack_from("<I", buf, position)[0]
            chunk_pointer = first_hbin + rel
            chunk_payload, chunk_end, chunk_cell = _payload_span(
                buf, index, chunk_pointer, cell)
            if chunk_cell is None:
                chunks.append({
                    "pointer": chunk_pointer,
                    "pointer_hex": _hex(chunk_pointer),
                    "cell_offset": None,
                    "cell_offset_hex": None,
                    "cell_state": "missing",
                    "file_offset": chunk_payload,
                    "file_offset_hex": _hex(chunk_payload),
                    "length": 0,
                })
                break
            take = min(0x3FD8, remaining, max(0, chunk_end - chunk_payload))
            raw = buf[chunk_payload:chunk_payload + take]
            chunks.append({
                "pointer": chunk_pointer,
                "pointer_hex": _hex(chunk_pointer),
                "cell_offset": chunk_cell.offset,
                "cell_offset_hex": _hex(chunk_cell.offset),
                "cell_state": chunk_cell.state,
                "file_offset": chunk_payload,
                "file_offset_hex": _hex(chunk_payload),
                "length": len(raw),
            })
            parts.append(raw)
            remaining -= len(raw)
            if remaining <= 0 or not raw:
                break
        description["chunks"] = chunks
        raw = b"".join(parts)
        description["available"] = len(raw)
        return raw, description

    available = max(0, min(data_length, _MAX_EXPORTED_DATA,
                           payload_end - payload_offset))
    raw = buf[payload_offset:payload_offset + available]
    description["available"] = available
    return raw, description


def _decode_value(raw, data_type):
    """
    Return (value, error).  ``raw`` may be None when the data cell is gone.
    """
    if raw is None:
        return None, "value data cell is missing"
    try:
        if data_type in (RegistryParse.RegSZ, RegistryParse.RegExpandSZ):
            if len(raw) % 2:
                raw = raw + b"\x00"
            try:
                text = raw.decode("utf-16le")
            except UnicodeDecodeError as exc:
                # Keep a readable best-effort value; the exact bytes remain in
                # value_hex and the decoding problem is reported.
                text = raw.decode("utf-16le", "replace")
                return text.split("\x00")[0], "invalid UTF-16LE: %s" % exc
            return text.split("\x00")[0], None
        if data_type == RegistryParse.RegMultiSZ:
            if len(raw) % 2:
                raw = raw + b"\x00"
            # Match Registry.RegistryValue.value(): the terminator remains as
            # trailing empty strings, which are part of the on-disk encoding.
            return raw.decode("utf-16le").split("\x00"), None
        if data_type == RegistryParse.RegDWord:
            return struct.unpack_from("<I", raw, 0)[0], None
        if data_type == RegistryParse.RegQWord:
            return struct.unpack_from("<Q", raw, 0)[0], None
        if data_type == RegistryParse.RegBigEndian:
            return struct.unpack_from(">I", raw, 0)[0], None
        if data_type == RegistryParse.RegFileTime:
            ticks = struct.unpack_from("<Q", raw, 0)[0]
            epoch = datetime.datetime(1601, 1, 1)
            return (epoch + datetime.timedelta(microseconds=ticks // 10)).isoformat("T") + "Z", None
        # RegBin, RegNone, RegLink and the remaining DEVPROP types are kept
        # as hex; exposing a fabricated Python value would be misleading.
        if data_type in (RegistryParse.RegBin, RegistryParse.RegNone,
                         RegistryParse.RegLink):
            return None, None
        return None, "unsupported value type 0x%x; raw bytes exported as hex" % data_type
    except (UnicodeDecodeError, struct.error, ValueError) as exc:
        return None, str(exc)


def _json_safe(value):
    if isinstance(value, (bytes, bytearray)):
        return None
    if isinstance(value, datetime.datetime):
        return value.isoformat("T") + "Z"
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, _TEXT_TYPES + (int, float, bool)) or value is None:
        return value
    return str(value)


def _type_name(data_type):
    names = {
        RegistryParse.RegNone: "RegNone",
        RegistryParse.RegSZ: "RegSZ",
        RegistryParse.RegExpandSZ: "RegExpandSZ",
        RegistryParse.RegBin: "RegBin",
        RegistryParse.RegDWord: "RegDWord",
        RegistryParse.RegBigEndian: "RegBigEndian",
        RegistryParse.RegLink: "RegLink",
        RegistryParse.RegMultiSZ: "RegMultiSZ",
        RegistryParse.RegResourceList: "RegResourceList",
        RegistryParse.RegFullResourceDescriptor: "RegFullResourceDescriptor",
        RegistryParse.RegResourceRequirementsList: "RegResourceRequirementsList",
        RegistryParse.RegQWord: "RegQWord",
        RegistryParse.RegFileTime: "RegFileTime",
    }
    return names.get(data_type, "0x%x" % data_type)


def _make_observation(header, cell, raw, data_description,
                      state, path, path_status, owner, reason,
                      root_reachable, value_list=None, parent_links=None,
                      parent_broken=None):
    value, value_error = _decode_value(raw, header["data_type"])
    if raw is not None and len(raw) > _MAX_EXPORTED_DATA:
        raw_hex = binascii.hexlify(raw[:_MAX_EXPORTED_DATA]).decode("ascii")
        truncated = True
    elif raw is not None:
        raw_hex = binascii.hexlify(raw).decode("ascii")
        truncated = False
    else:
        raw_hex = None
        truncated = False

    location = _cell_location(cell, header["record_offset"])
    location["data"] = data_description

    evidence = [{
        "type": "record",
        "signature": "vk",
        "record_offset": header["record_offset"],
        "record_offset_hex": _hex(header["record_offset"]),
        "record_size": header["record_size"],
        "name_length": header["name_length"],
        "data_type": header["data_type"],
        "data_length": header["data_length"],
        "resident": header["resident"],
    }, {
        "type": "allocation",
        "cell_offset": cell.offset,
        "cell_offset_hex": _hex(cell.offset),
        "cell_size": cell.size,
        "cell_state": cell.state,
        "embedded_in_free_cell": bool(
            cell.is_free and header["record_offset"] != cell.data_offset),
    }]

    if value_list is not None:
        evidence.append({
            "type": "value_list_reference",
            "status": ("referenced" if root_reachable else
                       "stale_owner_reference"),
            "list_cell_offset": value_list["list_cell_offset"],
            "list_cell_offset_hex": _hex(value_list["list_cell_offset"]),
            "entry_index": value_list["entry_index"],
            "value_cell_pointer": value_list["pointer"],
            "value_cell_pointer_hex": _hex(value_list["pointer"]),
            "key_offset": value_list.get("key_offset"),
            "key_offset_hex": _hex(value_list.get("key_offset")),
        })

    evidence.append({
        "type": "root_reachability",
        "status": ("root_reachable" if state == ACTIVE else
                   "indeterminate" if state == UNCERTAIN else
                   "not_root_reachable"),
        "reason": reason,
        "root_reachable": root_reachable,
    })

    evidence.append({
        "type": "value_data",
        "status": data_description.get("cell_state", "missing"),
        "allocated": data_description.get("allocated"),
        "pointer": data_description.get("pointer"),
        "pointer_hex": data_description.get("pointer_hex"),
        "file_offset": data_description.get("file_offset"),
        "file_offset_hex": data_description.get("file_offset_hex"),
        "length": data_description.get("length"),
        "available": data_description.get("available"),
    })

    if owner is not None:
        evidence.append({
            "type": "owner_key",
            "status": "resolved",
            "resolution": ("live_value_list" if root_reachable else
                           "carved_cell_reference"),
            "key_offset": owner.cell_offset,
            "key_offset_hex": _hex(owner.cell_offset),
            "key_record_offset": owner.record_offset,
            "key_record_offset_hex": _hex(owner.record_offset),
            "key_name": owner.name,
            "key_cell_state": owner.cell_state,
            "path_status": path_status,
        })
        evidence.append({
            "type": "parent_chain",
            "status": path_status,
            "link_kind": ("live_traversal" if root_reachable else
                          "reconstructed_from_parent_fields"),
            "links": parent_links or [],
        })
        if parent_broken is not None:
            evidence.append({
                "type": "parent_chain_broken",
                "status": "broken",
                "parent_cell_offset": parent_broken.get("parent_cell_offset"),
                "parent_cell_offset_hex": _hex(parent_broken.get("parent_cell_offset")),
                "reason": parent_broken.get("reason"),
            })
    else:
        evidence.append({
            "type": "owner_key",
            "status": "unknown",
            "reason": "no key candidate references this value cell",
        })
        evidence.append({
            "type": "parent_chain",
            "status": "unknown",
            "links": [],
        })

    return {
        "state": state,
        "identity": {
            "record_offset": header["record_offset"],
            "record_offset_hex": _hex(header["record_offset"]),
            "cell_offset": cell.offset,
            "cell_offset_hex": _hex(cell.offset),
            "owner_key_offset": owner.cell_offset if owner is not None else None,
        },
        "path": path,
        "key_name": owner.name if owner is not None else None,
        "value_name": header["name"],
        "value_type": header["data_type"],
        "value_type_name": _type_name(header["data_type"]),
        "value": _json_safe(value),
        "value_hex": raw_hex,
        "value_truncated": truncated,
        "value_error": value_error,
        "root_reachable": root_reachable,
        "path_status": path_status,
        "classification_reason": reason,
        "location": location,
        "evidence": evidence,
        "collision": None,
    }


def _walk_root(buf, regf, index, issues):
    """
    Walk only references originating at the root key.

    Returns (active, uncertain, referenced_cells, handled_record_offsets).
    """
    first_hbin = regf.first_hbin_offset()
    active = []
    uncertain = []
    referenced = set()
    handled = set()

    try:
        root_cell_offset = first_hbin + regf.unpack_dword(0x24)
    except struct.error as exc:
        issues.append({
            "reason": "root key pointer cannot be read",
            "location": {"file_offset": 0x24},
            "evidence": {"type": "root", "error": str(exc)},
        })
        return active, uncertain, referenced, handled, set()

    root_cell = index.at(root_cell_offset) or index.containing(root_cell_offset)
    if root_cell is None:
        issues.append({
            "reason": "root key cell is missing",
            "location": {"file_offset": root_cell_offset},
            "evidence": {"type": "root"},
        })
        return active, uncertain, referenced, handled, set()

    root_header = _parse_nk_header(buf, root_cell.data_offset, root_cell.end)
    if root_header is None:
        issues.append({
            "reason": "root key does not contain a valid NK record",
            "location": _cell_location(root_cell, root_cell.data_offset),
            "evidence": {"type": "root"},
        })
        return active, uncertain, referenced, handled, set()

    root_record = root_header["record_offset"]
    stack = [(root_header, root_cell, [], root_cell_offset)]
    visited_keys = set()
    visited_key_cells = set()

    while stack:
        nk, key_cell, parent_parts, parent_cell_offset = stack.pop()
        key_record = nk["record_offset"]
        referenced.add(key_cell.offset)
        visited_key_cells.add(key_cell.offset)

        if key_record in visited_keys:
            issues.append({
                "reason": "path cycle encountered during root traversal",
                "location": _cell_location(key_cell, key_record),
                "evidence": {"type": "path_cycle",
                             "key_offset": key_cell.offset},
            })
            continue
        visited_keys.add(key_record)

        path_parts = parent_parts + [nk["name"]]
        path = "\\".join(path_parts)

        if key_record != root_record:
            expected_parent_rel = parent_cell_offset - first_hbin
            if nk["parent_rel"] != expected_parent_rel:
                issues.append({
                    "reason": "parent reference does not match traversal parent",
                    "location": _cell_location(key_cell, key_record),
                    "evidence": {
                        "type": "parent_reference",
                        "recorded_parent_rel": nk["parent_rel"],
                        "recorded_parent_rel_hex": _hex(nk["parent_rel"]),
                        "expected_parent_rel": expected_parent_rel,
                        "expected_parent_rel_hex": _hex(expected_parent_rel),
                    },
                })

        # Mark auxiliary cells referenced by this key.
        for pointer_field in (0x2C, 0x30):
            try:
                rel = struct.unpack_from("<I", buf, key_record + pointer_field)[0]
            except struct.error:
                continue
            if rel != 0xFFFFFFFF:
                auxiliary = index.at(first_hbin + rel) or index.containing(first_hbin + rel)
                if auxiliary is not None:
                    referenced.add(auxiliary.offset)

        value_list_cell = None
        if nk["values_number"] and nk["values_list_rel"] != 0xFFFFFFFF:
            value_list_cell = _structural_cell(
                index, first_hbin + nk["values_list_rel"])
            if value_list_cell is not None:
                referenced.add(value_list_cell.offset)
            entries = _read_value_list(buf, value_list_cell,
                                       nk["values_number"], first_hbin, index,
                                       pointer=first_hbin + nk["values_list_rel"])
            if len(entries) < nk["values_number"]:
                issues.append({
                    "reason": "value list is shorter than the NK value count",
                    "location": _cell_location(key_cell, key_record),
                    "evidence": {
                        "type": "value_list",
                        "declared": nk["values_number"],
                        "read": len(entries),
                    },
                })
            for value_cell_offset, list_cell_offset, entry_index, pointer in entries:
                value_cell = index.at(value_cell_offset) or index.containing(value_cell_offset)
                if value_cell is None:
                    issues.append({
                        "reason": "value-list entry points outside every cell",
                        "location": {
                            "file_offset": value_cell_offset,
                            "cell_offset": value_cell_offset,
                            "pointer_hex": _hex(pointer),
                        },
                        "evidence": {
                            "type": "value_list_reference",
                            "key_offset": key_cell.offset,
                            "entry_index": entry_index,
                        },
                    })
                    continue
                if not value_cell.is_free:
                    referenced.add(value_cell.offset)

                if pointer == value_cell.offset:
                    record_offset = value_cell.data_offset
                else:
                    record_offset = pointer + 4
                header = _parse_vk_header(buf, record_offset, value_cell.end)
                if header is None:
                    issues.append({
                        "reason": "value-list entry does not point to a valid VK record",
                        "location": _cell_location(value_cell, record_offset),
                        "evidence": {
                            "type": "value_list_reference",
                            "key_offset": key_cell.offset,
                            "entry_index": entry_index,
                            "pointer_hex": _hex(pointer),
                        },
                    })
                    continue

                handled.add(header["record_offset"])
                raw, data_description = _data_description(
                    buf, header, first_hbin, index, value_cell)
                if data_description.get("cell_offset") is not None:
                    referenced.add(data_description["cell_offset"])
                for chunk in data_description.get("chunks", []):
                    if chunk.get("cell_offset") is not None:
                        referenced.add(chunk["cell_offset"])
                if value_cell.is_free or not data_description.get("allocated"):
                    reason = ("value record is referenced by a root-reachable key, "
                              "but its cell or value data is not allocated")
                    observation = _make_observation(
                        header, value_cell, raw, data_description,
                        UNCERTAIN, path, "root", _KeyInfo(nk, key_cell),
                        reason, True,
                        value_list={
                            "list_cell_offset": list_cell_offset,
                            "entry_index": entry_index,
                            "pointer": pointer,
                            "key_offset": key_cell.offset,
                        })
                    observation["key_name"] = nk["name"]
                    uncertain.append(observation)
                    issues.append({
                        "reason": reason,
                        "location": observation["location"],
                        "evidence": {
                            "type": "value_data",
                            "key_offset": key_cell.offset,
                            "value_offset": header["record_offset"],
                        },
                    })
                else:
                    reason = ("value record is referenced by the value list of a key "
                              "reachable from the hive root")
                    observation = _make_observation(
                        header, value_cell, raw, data_description,
                        ACTIVE, path, "root", _KeyInfo(nk, key_cell),
                        reason, True,
                        value_list={
                            "list_cell_offset": list_cell_offset,
                            "entry_index": entry_index,
                            "pointer": pointer,
                            "key_offset": key_cell.offset,
                        })
                    observation["key_name"] = nk["name"]
                    active.append(observation)
        elif nk["values_number"]:
            issues.append({
                "reason": "NK declares values but has no value-list pointer",
                "location": _cell_location(key_cell, key_record),
                "evidence": {"type": "value_list",
                             "declared": nk["values_number"]},
            })

        if nk["subkey_number"]:
            if nk["subkey_number"] == 0xFFFFFFFF:
                continue
            subkey_list_cell = _structural_cell(
                index,
                first_hbin + struct.unpack_from("<I", buf, key_record + 0x1C)[0])
            key_offsets, list_offsets = _read_subkey_list(
                buf, subkey_list_cell, first_hbin, index, issues)
            referenced.update(list_offsets)
            children = []
            for child_cell_offset in key_offsets:
                child_cell = index.at(child_cell_offset) or index.containing(child_cell_offset)
                if child_cell is None or child_cell.is_free:
                    issues.append({
                        "reason": "subkey reference points to a missing or free cell",
                        "location": {
                            "file_offset": child_cell_offset,
                            "cell_offset": child_cell_offset,
                            "cell_state": child_cell.state if child_cell else "missing",
                        },
                        "evidence": {"type": "subkey_reference",
                                     "parent_key_offset": key_cell.offset},
                    })
                    continue
                child_header = _parse_nk_header(buf, child_cell.data_offset,
                                                child_cell.end)
                if child_header is None:
                    issues.append({
                        "reason": "referenced subkey does not contain a valid NK record",
                        "location": _cell_location(child_cell, child_cell.data_offset),
                        "evidence": {"type": "subkey_reference",
                                     "parent_key_offset": key_cell.offset},
                    })
                    continue
                children.append((child_header, child_cell, path_parts,
                                 key_cell.offset))
            for child in reversed(children):
                stack.append(child)

    return active, uncertain, referenced, handled, visited_key_cells


def _scan_keys(buf, index, first_hbin, visited_key_cells, issues):
    """
    Build _KeyInfo candidates from allocated and free cells.

    Parent links only ever address a cell header.  A live key record sits at
    its cell's data offset; a record carved out of a coalesced free cell was
    itself a cell, so its old header starts four bytes before the record.
    Records elsewhere in a cell cannot be named by a parent link and are not
    used to resolve paths.

    Returns (candidates, by_cell_offset, by_record_offset).
    """
    candidates = []
    by_cell = {}
    by_record = {}
    seen = set()
    for cell in index:
        if not cell.is_free and cell.offset not in visited_key_cells:
            # Allocated cells outside the live tree may hold values or data,
            # but a key there is unreferenced and therefore cannot help
            # resolve a path through parent links.
            continue
        for record_offset in _candidate_offsets(buf, cell):
            if record_offset in seen:
                continue
            header = _parse_nk_header(buf, record_offset, cell.end)
            if header is None:
                continue
            seen.add(record_offset)
            info = _KeyInfo(header, cell)
            candidates.append(info)
            by_record[record_offset] = info
            # Parent pointers address a cell start.  A record at a cell's data
            # offset is addressed by that cell.  A record carved from a
            # coalesced free cell was itself once a cell, so its old header
            # begins four bytes before the record.
            if record_offset == cell.data_offset:
                by_cell[cell.offset] = info
            elif cell.is_free and record_offset - 4 >= cell.data_offset:
                by_cell[record_offset - 4] = info
    return candidates, by_cell, by_record


def _value_owners(buf, key_candidates, first_hbin, index):
    """
    Return (owner_by_pointer, list_cell_by_key_pointer) where pointers are
    absolute value-cell offsets named by each key's value list.
    """
    owners = {}
    owner_entries = {}
    lists = {}
    for key in key_candidates:
        if not key.values_number or key.values_list_rel == 0xFFFFFFFF:
            continue
        list_cell = _structural_cell(index, first_hbin + key.values_list_rel)
        if list_cell is None:
            continue
        lists[key.cell_offset] = list_cell.offset
        for value_cell_offset, list_cell_offset, entry_index, pointer in \
                _read_value_list(buf, list_cell, key.values_number,
                                 first_hbin, index,
                                 pointer=first_hbin + key.values_list_rel):
            if pointer not in owners:
                owners[pointer] = key
                owner_entries[pointer] = entry_index
    return owners, owner_entries, lists


def _resolve_path(key, by_cell, first_hbin, root_record, index):
    """
    Follow parent references, the only path evidence for unreferenced keys.

    Returns (path, status, links, broken).
    """
    if key is None:
        return None, "unknown", [], None

    names = []
    links = []
    seen = set()
    current = key

    while True:
        if current.cell_offset in seen:
            names.append("[path cycle]")
            return ("\\".join(reversed(names)), "broken", links, {
                "parent_cell_offset": current.cell_offset,
                "reason": "path cycle",
            })
        seen.add(current.cell_offset)
        names.append(current.name)
        links.append({
            "key_offset": current.cell_offset,
            "key_offset_hex": _hex(current.cell_offset),
            "key_record_offset": current.record_offset,
            "key_record_offset_hex": _hex(current.record_offset),
            "key_name": current.name,
            "cell_state": current.cell_state,
            "parent_rel": current.parent_rel,
            "parent_rel_hex": _hex(current.parent_rel),
        })

        if current.record_offset == root_record:
            return "\\".join(reversed(names)), "root", links, None

        if current.is_root:
            # A root flag on a record other than the real root is a stale or
            # forged copy; its position in the tree is indeterminate.
            return ("\\".join(reversed(names)), "broken", links, {
                "parent_cell_offset": current.cell_offset,
                "reason": "root flag set on a non-root record",
            })

        if current.parent_rel == 0xFFFFFFFF:
            return ("\\".join(reversed(names)), "broken", links, {
                "parent_cell_offset": current.cell_offset,
                "reason": "non-root key has no parent reference",
            })

        parent_offset = first_hbin + current.parent_rel
        parent = by_cell.get(parent_offset)
        if parent is None:
            # Distinguish "the cell exists but holds no NK" from "missing".
            parent_cell = index.at(parent_offset) or index.containing(parent_offset)
            if parent_cell is None:
                reason = "parent reference points to a missing cell"
            else:
                reason = "parent cell does not contain a valid NK record at its data offset"
            return ("\\".join(reversed(names)), "broken", links, {
                "parent_cell_offset": parent_offset,
                "reason": reason,
            })
        current = parent


def _scan_residual(buf, index, first_hbin, key_candidates, by_cell,
                   root_record, referenced, handled):
    """
    Return (residual, uncertain) observations for records not reachable from
    the root.  Physically-free records are residual unless their parent chain
    is broken, in which case reachability is indeterminate.
    """
    residual = []
    uncertain = []
    owners, owner_entries, owner_lists = _value_owners(
        buf, key_candidates, first_hbin, index)

    # First pass: collect structurally plausible VK records outside the live
    # tree.  Data cells are identified before any record is classified so
    # pass two does not depend on physical cell order.
    candidates = []
    seen_records = set()
    for cell in index:
        # Cells already accounted for by the live tree are key/value/list/data
        # structures.  Scanning them again would turn data bytes into VK hits.
        if cell.offset in referenced:
            continue
        for record_offset in _candidate_offsets(buf, cell):
            if record_offset in handled or record_offset in seen_records:
                continue
            header = _parse_vk_header(buf, record_offset, cell.end)
            if header is None:
                continue
            if record_offset + header["record_size"] > cell.end:
                continue
            seen_records.add(record_offset)
            raw, data_description = _data_description(
                buf, header, first_hbin, index, cell)
            candidates.append((cell, header, raw, data_description))

    data_cells = set()
    for cell, header, raw, data_description in candidates:
        data_cell_offset = data_description.get("cell_offset")
        if data_cell_offset is not None and data_cell_offset != cell.offset:
            data_cells.add(data_cell_offset)
        for chunk in data_description.get("chunks", []):
            if chunk.get("cell_offset") is not None:
                data_cells.add(chunk["cell_offset"])

    for cell, header, raw, data_description in candidates:
        if cell.offset in data_cells:
            # Payload of another record, not a value record of its own.
            continue

        record_offset = header["record_offset"]
        pointer_candidates = [record_offset - 4]
        if record_offset == cell.data_offset:
            pointer_candidates.append(cell.offset)
        owner = None
        owner_value_pointer = record_offset - 4
        owner_entry_index = None
        for pointer in pointer_candidates:
            if pointer in owners:
                owner = owners[pointer]
                owner_entry_index = owner_entries.get(pointer)
                owner_value_pointer = pointer
                break

        path, path_status, links, broken = _resolve_path(
            owner, by_cell, first_hbin, root_record, index)

        if cell.is_free:
            if path_status == "broken":
                state = UNCERTAIN
                reason = ("value record is in a free cell, but its owner's parent "
                          "chain is broken, so reachability is indeterminate")
            else:
                state = RESIDUAL
                reason = "value record is physically located in a free cell"
            root_reachable = False
        else:
            state = UNCERTAIN
            reason = ("value record is in an allocated cell but is not referenced "
                      "by any root-reachable value list")
            root_reachable = False

        observation = _make_observation(
            header, cell, raw, data_description, state, path,
            path_status, owner, reason, root_reachable,
            value_list=({
                "list_cell_offset": owner_lists.get(owner.cell_offset),
                "entry_index": owner_entry_index,
                # This pointer comes from the carved owner's old value
                # list; it is not a live root-reachable reference.
                "pointer": owner_value_pointer,
                "key_offset": owner.cell_offset if owner is not None else None,
            } if owner is not None else None),
            parent_links=links, parent_broken=broken)
        if owner is not None:
            observation["identity"]["owner_key_offset"] = owner.cell_offset

        if state == RESIDUAL:
            residual.append(observation)
        else:
            uncertain.append(observation)

    return residual, uncertain


def _collision_key(observation):
    path = observation.get("path")
    if path is None:
        return None
    return (path.lower(), (observation.get("value_name") or "").lower())


def _annotate_collisions(active, residual, uncertain):
    """
    Link equal textual path/value names across evidence classes without
    merging them.  Identity remains physical.
    """
    groups = {}
    for state, items in ((ACTIVE, active), (RESIDUAL, residual),
                         (UNCERTAIN, uncertain)):
        for observation in items:
            key = _collision_key(observation)
            if key is None:
                continue
            groups.setdefault(key, []).append((state, observation))

    collisions = []
    for key, members in sorted(groups.items(), key=lambda pair: pair[0]):
        states = set(state for state, _ in members)
        if ACTIVE not in states or len(states) == 1:
            continue
        group_id = "collision-%d" % (len(collisions) + 1)
        by_state = {}
        for state, observation in members:
            by_state.setdefault(state, []).append(observation["identity"])
        collision = {
            "group": group_id,
            "path": members[0][1].get("path"),
            "value_name": members[0][1].get("value_name"),
            "states": sorted(states),
            "active_identities": by_state.get(ACTIVE, []),
            "residual_identities": by_state.get(RESIDUAL, []),
            "uncertain_identities": by_state.get(UNCERTAIN, []),
            "note": ("textual path is not identity; residual and uncertain rows "
                     "are observations, not active configuration"),
        }
        collisions.append(collision)
        for state, observation in members:
            others = []
            for other_state, other in members:
                if other is observation:
                    continue
                others.append({
                    "state": other_state,
                    "record_offset": other["identity"]["record_offset"],
                    "cell_offset": other["identity"]["cell_offset"],
                })
            observation["collision"] = {
                "group": group_id,
                "states": sorted(states),
                "other_identities": others,
                "note": collision["note"],
            }
    return collisions


def _match(needle, observation, case_sensitive):
    haystack = [
        observation.get("path"),
        observation.get("key_name"),
        observation.get("value_name"),
        observation.get("value_type_name"),
    ]
    value = observation.get("value")
    if isinstance(value, list):
        haystack.extend(value)
    elif value is not None:
        haystack.append(str(value))
    value_hex = observation.get("value_hex")
    if value_hex:
        haystack.append(value_hex)

    if case_sensitive:
        return any(needle in item for item in haystack
                   if isinstance(item, _TEXT_TYPES))
    lowered = needle.lower()
    return any(lowered in item.lower() for item in haystack
               if isinstance(item, _TEXT_TYPES))


def classify(source, needle=None, case_sensitive=False):
    """
    Analyze a hive and return a :class:`ConfigExportReport`.

    ``source`` may be a filename, a file-like object, or a
    :class:`Registry.Registry` instance.
    """
    if hasattr(source, "_buf") and hasattr(source, "_regf"):
        registry = source
    else:
        from .Registry import Registry
        if isinstance(source, _BINARY_SOURCE_TYPES):
            import io
            registry = Registry(io.BytesIO(bytes(source)))
        else:
            registry = Registry(source)
    buf = registry._buf
    regf = registry._regf

    issues = []
    cells = _read_cells(buf, regf, issues)
    index = _CellIndex(cells)
    first_hbin = regf.first_hbin_offset()

    active, uncertain, referenced, handled, visited_key_cells = _walk_root(
        buf, regf, index, issues)

    try:
        root_cell = index.at(first_hbin + regf.unpack_dword(0x24))
        root_record = root_cell.data_offset if root_cell is not None else None
    except struct.error:
        root_record = None

    key_candidates, by_cell, _ = _scan_keys(
        buf, index, first_hbin, visited_key_cells, issues)
    residual, scan_uncertain = _scan_residual(
        buf, index, first_hbin, key_candidates, by_cell, root_record,
        referenced, handled)
    uncertain.extend(scan_uncertain)

    active.sort(key=lambda item: (item["identity"]["record_offset"],
                                  item.get("path") or ""))
    residual.sort(key=lambda item: (item["identity"]["record_offset"],
                                    item.get("path") or ""))
    uncertain.sort(key=lambda item: (item["identity"]["record_offset"],
                                     item.get("path") or ""))

    collisions = _annotate_collisions(active, residual, uncertain)

    query = {"needle": needle, "case_sensitive": case_sensitive}
    if needle:
        active = [item for item in active
                  if _match(needle, item, case_sensitive)]
        residual = [item for item in residual
                    if _match(needle, item, case_sensitive)]
        uncertain = [item for item in uncertain
                     if _match(needle, item, case_sensitive)]
        exported = set()
        for item in active + residual + uncertain:
            marker = item.get("collision")
            if marker:
                exported.add(marker["group"])
        collisions = [item for item in collisions
                      if item["group"] in exported]

    recovery = regf.recovery_required()
    hive = {
        "name": _safe_call(registry.hive_name),
        "type": _safe_call(lambda: getattr(registry.hive_type(), "value",
                                           str(registry.hive_type()))),
        "root_key_name": _safe_root_name(buf, index, regf, first_hbin),
        "root_key_offset": root_record,
        "root_key_offset_hex": _hex(root_record),
        "first_hbin_offset": first_hbin,
        "first_hbin_offset_hex": _hex(first_hbin),
        "hbins_size": regf.hbins_size(),
        "file_size": len(buf),
        "sequence1": regf.hive_sequence1(),
        "sequence2": regf.hive_sequence2(),
        "recovery_required": {
            "header": recovery.recover_header,
            "data": recovery.recover_data,
        },
    }

    return ConfigExportReport(hive, active, residual, uncertain, collisions,
                              issues, query)


def _safe_call(callable_):
    try:
        return callable_()
    except Exception:
        return None


def _safe_root_name(buf, index, regf, first_hbin):
    try:
        cell = index.at(first_hbin + regf.unpack_dword(0x24))
        if cell is None:
            return None
        header = _parse_nk_header(buf, cell.data_offset, cell.end)
        return header["name"] if header is not None else None
    except Exception:
        return None


def export_config(source, needle=None, case_sensitive=False):
    """Return the report as a JSON-compatible dictionary."""
    return classify(source, needle=needle,
                    case_sensitive=case_sensitive).to_dict()


def export_json(source, needle=None, case_sensitive=False, indent=2):
    """Return the report as a JSON string."""
    return classify(source, needle=needle,
                    case_sensitive=case_sensitive).to_json(indent=indent)


# Descriptive aliases: the report is an analysis, and exporting it yields the
# same dictionary either way.
analyze = classify
export = export_config


def _format_value(observation):
    if observation.get("value") is not None:
        value = observation["value"]
        if isinstance(value, list):
            return "[" + ", ".join(repr(item) for item in value) + "]"
        return repr(value)
    if observation.get("value_hex"):
        return "hex:" + observation["value_hex"]
    return "(unavailable)"


def format_text(report):
    """Render a human-readable report from a ConfigExportReport."""
    lines = []
    hive = report.hive
    lines.append("Hive: %s (%s)" % (hive.get("name"), hive.get("type")))
    lines.append("Root: %s at %s" % (hive.get("root_key_name"),
                                     hive.get("root_key_offset_hex")))
    if report.query.get("needle"):
        lines.append("Query: %r (case_sensitive=%s)" % (
            report.query["needle"], report.query["case_sensitive"]))
    lines.append("")

    def section(title, items, note):
        lines.append("%s (%d) - %s" % (title, len(items), note))
        for item in items:
            location = item["location"]
            lines.append("  [%s] %s\\%s = %s" % (
                item["state"], item.get("path") or "(path unknown)",
                item.get("value_name") or "(default)", _format_value(item)))
            lines.append("       record=%s cell=%s(%s) data=%s(%s)" % (
                location.get("file_offset_hex"),
                location.get("cell_offset_hex"),
                location.get("cell_state"),
                location["data"].get("file_offset_hex"),
                location["data"].get("cell_state")))
            if item.get("collision"):
                lines.append("       collision=%s" % item["collision"]["group"])
            if item.get("path_status") not in (None, "root"):
                lines.append("       path_status=%s" % item["path_status"])
            for evidence in item["evidence"]:
                if evidence["type"] == "parent_chain_broken":
                    lines.append("       parent_broken_at=%s (%s)" % (
                        evidence.get("parent_cell_offset_hex"),
                        evidence.get("reason")))
                    break
        lines.append("")

    section("ACTIVE", report.active, "root-reachable, allocated")
    section("RESIDUAL", report.residual, "records in free cells")
    section("UNCERTAIN", report.uncertain,
            "reachability indeterminate; not classified as deleted")

    lines.append("COLLISIONS (%d) - path equality is not identity" %
                 len(report.collisions))
    for collision in report.collisions:
        lines.append("  %s %s\\%s states=%s" % (
            collision["group"], collision["path"], collision["value_name"],
            ",".join(collision["states"])))
    if not report.collisions:
        lines.append("  (none)")
    lines.append("")

    lines.append("STRUCTURAL ISSUES (%d)" % len(report.structural_issues))
    for issue in report.structural_issues:
        location = issue.get("location", {})
        lines.append("  %s @ %s" % (issue.get("reason"),
                                    location.get("file_offset_hex")))
    if not report.structural_issues:
        lines.append("  (none)")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=("Export a registry hive with root-reachable active values "
                     "kept separate from free-cell residue and uncertain rows."))
    parser.add_argument("hive", help="Windows Registry hive to analyze")
    parser.add_argument("--search", dest="needle", default=None,
                        help="only export rows containing this text")
    parser.add_argument("--case-sensitive", action="store_true",
                        help="make --search case sensitive")
    parser.add_argument("--format", choices=("json", "text"), default="json",
                        help="output format (default: json)")
    parser.add_argument("--output", default=None,
                        help="write to this file instead of stdout")
    parser.add_argument("--indent", type=int, default=2,
                        help="JSON indentation (default: 2)")
    args = parser.parse_args(argv)

    report = classify(args.hive, needle=args.needle,
                      case_sensitive=args.case_sensitive)
    if args.format == "json":
        output = report.to_json(indent=args.indent)
    else:
        output = format_text(report)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
            if not output.endswith("\n"):
                f.write("\n")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
