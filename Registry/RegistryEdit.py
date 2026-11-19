#!/bin/python

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
Snapshot based, rehearsable edits for Windows Registry keys and values.

This module adds a planning layer on top of the read-only parser in
:mod:`Registry.Registry`.  A :class:`HiveSnapshot` is an immutable logical
copy of a hive.  :class:`EditPlan` accepts :class:`EditGroup` instances and
first *rehearses* them against that snapshot.  Only groups that are valid and
mutually independent are then applied, producing a new snapshot and a
verification report.

The logical identity of a value is ``(key path, stored value name)`` with
case-insensitive comparison.  The unnamed default value has the empty stored
name.  A literal value named ``(Default)`` is a normal named value and has a
different logical identity, even though both may be displayed the same way by
some tools.
"""
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import collections
import struct
from enum import Enum

from . import Registry
from .Registry import DEFAULT_VALUE
from .RegistryParse import decode_utf16le

__all__ = [
    "DEFAULT_VALUE",
    "HiveSnapshot",
    "EditPlan",
    "EditGroup",
    "RenameValue",
    "ConvertDefaultValue",
    "DeleteValue",
    "ChangeValueType",
    "ClearValue",
    "ValueIdentity",
    "ValuePresence",
    "ValueState",
    "ValueSummary",
    "ValueOrigin",
    "KeyState",
    "EditError",
    "KeyNotFoundException",
    "MissingSourceValueError",
    "ValueOccupiedError",
    "UnknownValueTypeError",
    "RehearsalVerificationError",
    "Rehearsal",
    "GroupResult",
    "GroupStatus",
    "GroupDependency",
    "OperationResult",
    "OperationStatus",
    "ApplyResult",
    "VerificationResult",
    "VerificationMismatch",
    "snapshot_from_registry",
    "build_plan",
    "rehearse",
    "apply_edits",
    "normalize_key_path",
    "parse_value_type",
    "value_type_name",
]


RegSZ = Registry.RegSZ
RegExpandSZ = Registry.RegExpandSZ
RegBin = Registry.RegBin
RegDWord = Registry.RegDWord
RegMultiSZ = Registry.RegMultiSZ
RegQWord = Registry.RegQWord
RegNone = Registry.RegNone
RegBigEndian = Registry.RegBigEndian
RegLink = Registry.RegLink
RegResourceList = Registry.RegResourceList
RegFullResourceDescriptor = Registry.RegFullResourceDescriptor
RegResourceRequirementsList = Registry.RegResourceRequirementsList
RegFileTime = Registry.RegFileTime
RegUint8 = 0x101
RegInt16 = 0x102
RegUint16 = 0x103
RegInt32 = 0x104
RegUint32 = 0x105
RegInt64 = 0x106
RegUint64 = 0x107
RegFloat = 0x108
RegDouble = 0x109
RegUnicodeChar = 0x10A
RegBoolean = 0x10B
RegUnicodeString = 0x10C
RegCompositeValue = 0x10D
RegDateTimeOffset = 0x10E
RegTimeSpan = 0x10F
RegGUID = 0x110
RegUnk111 = 0x111
RegUnk112 = 0x112
RegUnk113 = 0x113
RegBytesArray = 0x114
RegInt16Array = 0x115
RegUint16Array = 0x116
RegInt32Array = 0x117
RegUInt32Array = 0x118
RegInt64Array = 0x119
RegUInt64Array = 0x11A
RegFloatArray = 0x11B
RegDoubleArray = 0x11C
RegUnicodeCharArray = 0x11D
RegBooleanArray = 0x11E
RegUnicodeStringArray = 0x11F


VALUE_TYPE_NAMES = {
    RegSZ: "RegSZ",
    RegExpandSZ: "RegExpandSZ",
    RegBin: "RegBin",
    RegDWord: "RegDWord",
    RegMultiSZ: "RegMultiSZ",
    RegQWord: "RegQWord",
    RegNone: "RegNone",
    RegBigEndian: "RegBigEndian",
    RegLink: "RegLink",
    RegResourceList: "RegResourceList",
    RegFullResourceDescriptor: "RegFullResourceDescriptor",
    RegResourceRequirementsList: "RegResourceRequirementsList",
    RegFileTime: "RegFileTime",
    RegUint8: "RegUint8",
    RegInt16: "RegInt16",
    RegUint16: "RegUint16",
    RegInt32: "RegInt32",
    RegUint32: "RegUint32",
    RegInt64: "RegInt64",
    RegUint64: "RegUint64",
    RegFloat: "RegFloat",
    RegDouble: "RegDouble",
    RegUnicodeChar: "RegUnicodeChar",
    RegBoolean: "RegBoolean",
    RegUnicodeString: "RegUnicodeString",
    RegCompositeValue: "RegCompositeValue",
    RegDateTimeOffset: "RegDateTimeOffset",
    RegTimeSpan: "RegTimeSpan",
    RegGUID: "RegGUID",
    RegUnk111: "RegUnk111",
    RegUnk112: "RegUnk112",
    RegUnk113: "RegUnk113",
    RegBytesArray: "RegBytesArray",
    RegInt16Array: "RegInt16Array",
    RegUint16Array: "RegUint16Array",
    RegInt32Array: "RegInt32Array",
    RegUInt32Array: "RegUInt32Array",
    RegInt64Array: "RegInt64Array",
    RegUInt64Array: "RegUInt64Array",
    RegFloatArray: "RegFloatArray",
    RegDoubleArray: "RegDoubleArray",
    RegUnicodeCharArray: "RegUnicodeCharArray",
    RegBooleanArray: "RegBooleanArray",
    RegUnicodeStringArray: "RegUnicodeStringArray",
}

VALUE_TYPE_BY_NAME = {}
for _type_value, _type_name in VALUE_TYPE_NAMES.items():
    VALUE_TYPE_BY_NAME[_type_name.casefold()] = _type_value
    VALUE_TYPE_BY_NAME[_type_name.lower()] = _type_value
VALUE_TYPE_BY_NAME["regsz"] = RegSZ


_UNSET = object()


class EditError(Exception):
    """Base class for edit planning and application errors."""


class KeyNotFoundException(EditError):
    """A key referenced by an operation does not exist in the snapshot."""


class MissingSourceValueError(EditError):
    """An operation references a value that is absent from the snapshot."""


class ValueOccupiedError(EditError):
    """A rename target is already occupied by a different logical value."""


class UnknownValueTypeError(EditError):
    """A value type is neither a known constant nor a known Reg* name."""


class RehearsalVerificationError(EditError):
    """The state collected after applying a plan does not match the rehearsal."""


def _casefold(value):
    if hasattr(value, "casefold"):
        return value.casefold()
    return value.lower()


def normalize_key_path(path):
    """Normalize a key path for display and lookup.

    Leading and trailing separators are removed.  A leading ``/`` is accepted
    as a convenience, but ``\\`` remains the canonical separator because value
    names may legally contain ``/``.
    """
    if path is None:
        return ""
    path = path.strip()
    while path.startswith("\\") or path.startswith("/"):
        path = path[1:]
    while path.endswith("\\") or path.endswith("/"):
        path = path[:-1]
    return path


def parse_value_type(value_type):
    """Return the integer constant for a Reg* name or integer value."""
    if isinstance(value_type, int):
        if value_type in VALUE_TYPE_NAMES:
            return value_type
        raise UnknownValueTypeError("Unknown registry value type: %s" % (value_type,))

    if isinstance(value_type, str):
        key = value_type.strip().casefold()
        key = key.replace("-", "").replace("_", "")
        aliases = {
            "dword": RegDWord,
            "qword": RegQWord,
            "sz": RegSZ,
            "string": RegSZ,
            "expandez": RegExpandSZ,
            "expandsz": RegExpandSZ,
            "binary": RegBin,
            "bin": RegBin,
            "multisz": RegMultiSZ,
            "none": RegNone,
        }
        if key in aliases:
            return aliases[key]
        if key in VALUE_TYPE_BY_NAME:
            return VALUE_TYPE_BY_NAME[key]
        normalized = "reg" + key[3:] if key.startswith("reg") else "reg" + key
        if normalized in VALUE_TYPE_BY_NAME:
            return VALUE_TYPE_BY_NAME[normalized]

    raise UnknownValueTypeError("Unknown registry value type: %r" % (value_type,))


def value_type_name(value_type):
    return VALUE_TYPE_NAMES.get(value_type, "Unknown(0x%x)" % value_type)


def _decode_value(value_type, raw_data):
    """Best effort decode of raw value bytes into a Python object."""
    raw_data = bytes(raw_data or b"")
    try:
        if value_type in (RegSZ, RegExpandSZ):
            return decode_utf16le(raw_data) if raw_data else ""
        if value_type == RegMultiSZ:
            if not raw_data:
                return []
            text = raw_data.decode("utf-16")
            return [part for part in text.split("\x00") if part != ""]
        if value_type == RegDWord and len(raw_data) >= 4:
            return struct.unpack_from("<I", raw_data, 0)[0]
        if value_type == RegQWord and len(raw_data) >= 8:
            return struct.unpack_from("<Q", raw_data, 0)[0]
        if value_type == RegBigEndian and len(raw_data) >= 4:
            return struct.unpack_from(">I", raw_data, 0)[0]
    except (UnicodeDecodeError, struct.error, ValueError):
        return raw_data
    return raw_data


def _encode_value(value_type, value):
    """Encode a Python object as raw registry value bytes."""
    if value is None:
        return b""
    if isinstance(value, bytearray):
        value = bytes(value)
    if isinstance(value, bytes):
        return value

    if value_type in (RegSZ, RegExpandSZ, RegUnicodeString):
        if isinstance(value, (int, float, bool)):
            value = str(value)
        if not isinstance(value, str):
            raise ValueError("A registry string value must be text")
        text = value
        if not text.endswith("\x00"):
            text += "\x00"
        return text.encode("utf-16le")
    if value_type == RegMultiSZ:
        items = list(value)
        text = "\x00".join(items)
        if not text.endswith("\x00\x00"):
            text += "\x00\x00"
        return text.encode("utf-16le")
    if value_type == RegDWord:
        return struct.pack("<I", int(value) & 0xFFFFFFFF)
    if value_type == RegQWord:
        return struct.pack("<Q", int(value) & 0xFFFFFFFFFFFFFFFF)
    if value_type == RegBigEndian:
        return struct.pack(">I", int(value) & 0xFFFFFFFF)
    if value_type == RegUint8:
        return struct.pack("<B", int(value) & 0xFF)
    if value_type == RegInt16:
        return struct.pack("<h", int(value))
    if value_type == RegUint16:
        return struct.pack("<H", int(value) & 0xFFFF)
    if value_type == RegInt32:
        return struct.pack("<i", int(value))
    if value_type == RegUint32:
        return struct.pack("<I", int(value) & 0xFFFFFFFF)
    if value_type == RegInt64:
        return struct.pack("<q", int(value))
    if value_type == RegUint64:
        return struct.pack("<Q", int(value) & 0xFFFFFFFFFFFFFFFF)
    if value_type == RegFloat:
        return struct.pack("<f", float(value))
    if value_type == RegDouble:
        return struct.pack("<d", float(value))
    if value_type == RegBoolean:
        return b"\x01" if value else b"\x00"
    raise UnknownValueTypeError(
        "Cannot encode %r as registry value type %s" % (value, value_type_name(value_type)))


def _is_empty_value(value):
    if value is None:
        return True
    if isinstance(value, (bytes, bytearray)):
        return len(value) == 0
    if isinstance(value, str):
        return value == ""
    if isinstance(value, (list, tuple, dict)):
        return len(value) == 0
    return False


class ValuePresence(Enum):
    """Presence of a logical value within a key."""
    MISSING = "missing"
    DEFAULT = "default"
    NAMED = "named"


class ValueIdentity(object):
    """
    The logical identity of a registry value.

    ``name`` is the *stored* name.  The empty string denotes the unnamed
    default value.  A literal value named ``(Default)`` is stored as
    ``"(Default)"`` and therefore remains distinct.
    """
    __slots__ = ("key_path", "name")

    def __init__(self, key_path, name=DEFAULT_VALUE):
        self.key_path = normalize_key_path(key_path)
        if name is DEFAULT_VALUE or name is None:
            name = ""
        self.name = name

    @property
    def is_default(self):
        return self.name == ""

    @property
    def display_name(self):
        return "(default)" if self.is_default else self.name

    @property
    def logical(self):
        if self.is_default:
            return (_casefold(self.key_path), _DefaultNameMarker)
        return (_casefold(self.key_path), _casefold(self.name))

    def with_key_path(self, key_path):
        return ValueIdentity(key_path, self.name)

    def with_name(self, name):
        return ValueIdentity(self.key_path, name)

    def __eq__(self, other):
        return isinstance(other, ValueIdentity) and self.logical == other.logical

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.logical)

    def __repr__(self):
        return "ValueIdentity(key_path=%r, name=%r)" % (self.key_path, self.name)

    def __str__(self):
        if self.is_default:
            return "%s\\<default>" % (self.key_path,)
        return "%s\\%s" % (self.key_path, self.name)


_DefaultNameMarker = object()


class ValueOrigin(object):
    """The original identity and data from which a value was captured."""
    __slots__ = ("revision", "identity", "value_type", "raw_data")

    def __init__(self, revision, identity, value_type, raw_data):
        self.revision = revision
        self.identity = identity
        self.value_type = value_type
        self.raw_data = bytes(raw_data or b"")

    def __repr__(self):
        return "ValueOrigin(revision=%r, identity=%r, value_type=%r)" % (
            self.revision, self.identity, value_type_name(self.value_type))



class ValueSummary(object):
    """A small, history-free description of a value state."""
    __slots__ = ("identity", "value_type", "raw_data", "value", "is_empty",
                 "exists", "origin")

    def __init__(self, identity, value_type, raw_data, value, exists=True,
                 origin=None):
        self.identity = identity
        self.value_type = value_type
        self.raw_data = bytes(raw_data or b"")
        self.value = value
        self.exists = exists
        self.is_empty = exists and _is_empty_value(value)
        self.origin = origin

    @property
    def value_type_name(self):
        return value_type_name(self.value_type)

    @property
    def is_default(self):
        return self.identity.is_default

    @property
    def original_identity(self):
        return self.origin.identity if self.origin is not None else self.identity

    def to_dict(self):
        return {
            "key_path": self.identity.key_path,
            "name": self.identity.name,
            "display_name": self.identity.display_name,
            "is_default": self.is_default,
            "exists": self.exists,
            "value_type": self.value_type_name,
            "is_empty": self.is_empty,
            "data_length": len(self.raw_data),
            "original_key_path": self.original_identity.key_path,
            "original_name": self.original_identity.name,
        }

    def __repr__(self):
        return "ValueSummary(identity=%r, type=%s, empty=%r, exists=%r)" % (
            self.identity, self.value_type_name, self.is_empty, self.exists)


class ChangeRecord(object):
    """One value transition, retained for provenance."""
    __slots__ = ("revision", "group", "operation", "before", "after")

    def __init__(self, revision, group, operation, before, after):
        self.revision = revision
        self.group = group
        self.operation = operation
        self.before = before
        self.after = after

    def __repr__(self):
        return "ChangeRecord(group=%r, operation=%r, before=%r, after=%r)" % (
            self.group, self.operation, self.before, self.after)


class ValueState(object):
    """
    The state of one existing registry value.

    ``origin`` always points at the value captured from the original snapshot,
    even after several edits.  ``history`` contains each transition that led
    to this state.
    """
    __slots__ = (
        "identity", "value_type", "raw_data", "value", "origin", "history",
    )

    def __init__(self, identity, value_type, raw_data=b"", value=_UNSET,
                 origin=None, history=()):
        self.identity = identity
        self.value_type = value_type
        self.raw_data = bytes(raw_data or b"")
        if value is _UNSET:
            value = _decode_value(value_type, self.raw_data)
        self.value = value
        if origin is None:
            origin = ValueOrigin(0, identity, value_type, self.raw_data)
        self.origin = origin
        self.history = tuple(history)

    @classmethod
    def from_registry_value(cls, key_path, registry_value, revision=0):
        identity = ValueIdentity(key_path, registry_value.stored_name())
        try:
            parsed = registry_value.value()
        except Exception:
            parsed = registry_value.raw_data()
        raw_data = registry_value.raw_data()
        origin = ValueOrigin(revision, identity, registry_value.value_type(), raw_data)
        return cls(identity, registry_value.value_type(), raw_data, parsed,
                   origin=origin)

    @property
    def name(self):
        return self.identity.name

    @property
    def display_name(self):
        return self.identity.display_name

    @property
    def is_default(self):
        return self.identity.is_default

    @property
    def value_type_name(self):
        return value_type_name(self.value_type)

    @property
    def is_empty(self):
        return _is_empty_value(self.value)

    @property
    def is_empty_data(self):
        return len(self.raw_data) == 0

    def summary(self, exists=True):
        return ValueSummary(self.identity, self.value_type, self.raw_data,
                            self.value, exists=exists, origin=self.origin)

    def copy_with(self, identity=None, value_type=None, raw_data=_UNSET,
                  value=_UNSET, history=None):
        if identity is None:
            identity = self.identity
        if value_type is None:
            value_type = self.value_type
        if raw_data is _UNSET:
            raw_data = self.raw_data
        if value is _UNSET:
            value = self.value
        return ValueState(identity, value_type, raw_data, value,
                          origin=self.origin,
                          history=self.history if history is None else history)

    def with_history(self, history):
        return ValueState(self.identity, self.value_type, self.raw_data,
                          self.value, origin=self.origin, history=history)

    def __repr__(self):
        return "ValueState(identity=%r, type=%s, empty=%r)" % (
            self.identity, self.value_type_name, self.is_empty)


class KeyState(object):
    """The immutable state of one key and its values."""
    __slots__ = ("path", "values")

    def __init__(self, path, values=()):
        self.path = normalize_key_path(path)
        self.values = tuple(values)

    @property
    def name(self):
        return self.path.rsplit("\\", 1)[-1]

    def value(self, name=DEFAULT_VALUE):
        if name is DEFAULT_VALUE or name is None:
            name = ""
        lower = _casefold(name)
        for value in self.values:
            if _casefold(value.name) == lower:
                return value
        return None

    def __repr__(self):
        return "KeyState(path=%r, values=%d)" % (self.path, len(self.values))


class HiveSnapshot(object):
    """
    An immutable logical snapshot of a registry hive.

    Use :meth:`from_registry` to capture a parsed
    :class:`Registry.Registry.Registry` instance.  A snapshot never changes in
    place; applying an :class:`EditPlan` returns a new snapshot linked to its
    parent, which keeps the original values traceable.
    """
    def __init__(self, root_path, keys=(), revision=1, parent=None, changes=()):
        self.root_path = normalize_key_path(root_path)
        self.revision = revision
        self.parent = parent
        self.changes = tuple(changes)
        self._keys = collections.OrderedDict()
        self._index = {}

        for key in keys:
            self._add_key(key)

    @classmethod
    def from_registry(cls, registry, revision=1):
        """Capture an immutable snapshot from a parsed Registry instance."""
        root = registry.root()
        keys = []

        def walk(key):
            values = []
            for registry_value in key.values():
                values.append(ValueState.from_registry_value(
                    key.path(), registry_value, revision=revision))
            keys.append(KeyState(key.path(), values))
            for subkey in key.subkeys():
                walk(subkey)

        walk(root)
        return cls(root.path(), keys, revision=revision)

    # Compatibility aliases with intuitive names.
    capture = from_registry

    @classmethod
    def from_mapping(cls, root_path, mapping, revision=1):
        """
        Build a snapshot from a mapping of ``key path -> value descriptions``.

        Each value description may contain ``name`` (empty or missing for the
        unnamed default), ``type`` (a constant or Reg* name), ``value``, and
        optionally ``raw_data``.  This is primarily useful for tests and for
        planning against a small synthetic configuration.
        """
        keys = []
        root_path = normalize_key_path(root_path)
        for path, descriptions in mapping.items():
            values = []
            for description in descriptions:
                name = description.get("name", "")
                value_type = parse_value_type(description.get("type", RegSZ))
                if "raw_data" in description:
                    raw_data = bytes(description["raw_data"])
                    value = description.get("value", _UNSET)
                    if value is _UNSET:
                        value = _decode_value(value_type, raw_data)
                else:
                    value = description.get("value")
                    raw_data = _encode_value(value_type, value)
                identity = ValueIdentity(path, name)
                origin = ValueOrigin(revision, identity, value_type, raw_data)
                values.append(ValueState(identity, value_type, raw_data, value,
                                         origin=origin))
            keys.append(KeyState(path, values))
        return cls(root_path, keys, revision=revision)

    def _add_key(self, key):
        normalized = self._canonical_path(key.path)
        values = []
        for value in key.values:
            identity = ValueIdentity(normalized, value.name)
            if value.identity != identity:
                value = value.copy_with(identity=identity)
            if value.origin.revision == self.revision and \
                    value.origin.identity != identity:
                value = ValueState(
                    value.identity, value.value_type, value.raw_data,
                    value.value,
                    origin=ValueOrigin(value.origin.revision, identity,
                                       value.origin.value_type,
                                       value.origin.raw_data),
                    history=value.history)
            values.append(value)
        key = KeyState(normalized, values)
        self._keys[_casefold(normalized)] = key
        for value in key.values:
            self._index[value.identity.logical] = value

    def _canonical_path(self, path):
        path = normalize_key_path(path)
        if not path:
            return self.root_path
        if "/" in path and "\\" not in path:
            path = normalize_key_path(path.replace("/", "\\"))
        if not self.root_path:
            return path
        root_lower = _casefold(self.root_path)
        path_lower = _casefold(path)
        if path_lower == root_lower or path_lower.startswith(root_lower + "\\"):
            return path
        return self.root_path + "\\" + path

    def canonical_path(self, path):
        """Resolve a possibly root-relative path to a full key path."""
        path = normalize_key_path(path)
        candidates = [path]
        if "/" in path:
            candidates.append(normalize_key_path(path.replace("/", "\\")))

        for candidate in candidates:
            canonical = self._canonical_path(candidate)
            key = self._keys.get(_casefold(canonical))
            if key is not None:
                return key.path

        raise KeyNotFoundException("Registry key not found in snapshot: %s" % (path,))

    def canonical_identity(self, key_path, name=DEFAULT_VALUE):
        """Resolve an identity against the snapshot's actual key paths."""
        path = self.canonical_path(key_path)
        if name is DEFAULT_VALUE or name is None:
            name = ""
        return ValueIdentity(path, name)

    def keys(self):
        return list(self._keys.values())

    def key(self, path):
        try:
            canonical = self.canonical_path(path)
        except KeyNotFoundException:
            raise
        return self._keys[_casefold(canonical)]

    def values_at(self, path):
        return list(self.key(path).values)

    def find_value(self, key_path, name=DEFAULT_VALUE):
        """
        Return the exact logical value, or None.

        The empty string and :data:`Registry.DEFAULT_VALUE` address the unnamed
        default.  The literal string ``(Default)`` addresses a *named* value
        with that literal name.
        """
        try:
            identity = self.canonical_identity(key_path, name)
        except KeyNotFoundException:
            return None
        return self._index.get(identity.logical)

    def value(self, key_path, name=DEFAULT_VALUE):
        """Alias for :meth:`find_value`."""
        return self.find_value(key_path, name)

    def resolve(self, key_path, name=DEFAULT_VALUE):
        """
        Resolve a value using the friendly ``(default)`` spelling.

        The empty string, :data:`DEFAULT_VALUE`, and ``(default)`` (any case)
        address the unnamed default value, consistent with
        :meth:`Registry.Registry.RegistryKey.value`.  A literal value named
        ``(Default)`` stays distinct and is reached with :meth:`find_value`
        or :meth:`Registry.Registry.RegistryKey.value_exact`.
        """
        if name is DEFAULT_VALUE or name is None or name == "" or \
                _casefold(name) == "(default)":
            return self.find_value(key_path, "")
        return self.find_value(key_path, name)

    def presence(self, key_path, name=DEFAULT_VALUE):
        """
        Classify the exact logical value as MISSING, DEFAULT, or NAMED.

        ``DEFAULT_VALUE`` and the empty string address the unnamed default.
        The literal name ``(Default)`` is an ordinary named value, so it
        reports NAMED rather than DEFAULT.  Use :meth:`resolve` for the
        friendly ``(default)`` spelling.
        """
        if name is DEFAULT_VALUE or name is None or name == "":
            value = self.find_value(key_path, "")
        else:
            value = self.find_value(key_path, name)
        if value is None:
            return ValuePresence.MISSING
        return ValuePresence.DEFAULT if value.is_default else ValuePresence.NAMED

    def presence_friendly(self, key_path, name=DEFAULT_VALUE):
        """Classify a lookup using the friendly ``(default)`` spelling."""
        value = self.resolve(key_path, name)
        if value is None:
            return ValuePresence.MISSING
        return ValuePresence.DEFAULT if value.is_default else ValuePresence.NAMED

    def history_for(self, identity):
        """
        Return the chronological change history of a logical value.

        Renames are followed through the captured origin, so the history of a
        renamed value also includes transitions recorded under its earlier
        names.  Deleted values remain queryable by their last identity.
        """
        if not isinstance(identity, ValueIdentity):
            raise TypeError("identity must be a ValueIdentity")

        chain = []
        snapshot = self
        while snapshot is not None:
            chain.append(snapshot)
            snapshot = snapshot.parent

        all_changes = []
        for snapshot in reversed(chain):
            all_changes.extend(snapshot.changes)

        value = self._index.get(identity.logical)
        origins = set()
        if value is not None:
            origins.add(value.origin.identity.logical)

        for change in all_changes:
            before = change.before
            after = change.after
            touches = (before is not None and before.identity == identity) or \
                      (after is not None and after.identity == identity)
            if not touches:
                continue
            if before is not None:
                origins.add(before.original_identity.logical)
            if after is not None:
                origins.add(after.original_identity.logical)

        if not origins:
            return []

        history = []
        for change in all_changes:
            before = change.before
            after = change.after
            if before is not None and before.original_identity.logical in origins:
                history.append(change)
            elif after is not None and after.original_identity.logical in origins:
                history.append(change)
        return history

    def recollect(self):
        """
        Rebuild a snapshot from this snapshot's current states.

        This simulates reading the hive again after changes were committed.
        The rebuilt snapshot has no parent and no pending changes, but every
        value retains its captured origin and history.
        """
        keys = []
        for key in self._keys.values():
            keys.append(KeyState(key.path, key.values))
        return HiveSnapshot(self.root_path, keys, revision=self.revision,
                            parent=None, changes=())

    def rebuild(self, revision=None, parent=None):
        """Rebuild this snapshot's spaces, optionally re-parenting it."""
        keys = [KeyState(key.path, key.values) for key in self._keys.values()]
        return HiveSnapshot(self.root_path, keys,
                            revision=self.revision if revision is None else revision,
                            parent=parent, changes=self.changes)

    def to_dict(self):
        return {
            "root_path": self.root_path,
            "revision": self.revision,
            "keys": [
                {
                    "path": key.path,
                    "values": [value.summary().to_dict() for value in key.values],
                }
                for key in self._keys.values()
            ],
        }

    def __len__(self):
        return len(self._keys)

    def __repr__(self):
        return "HiveSnapshot(root=%r, keys=%d, revision=%r)" % (
            self.root_path, len(self._keys), self.revision)


class _WorkingHive(object):
    """Mutable scratch copy used while validating and applying groups."""
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.keys = collections.OrderedDict()
        for path, key in snapshot._keys.items():
            self.keys[path] = [key.path, list(key.values)]

    def key(self, path):
        canonical = self.snapshot.canonical_path(path)
        return self.keys[_casefold(canonical)]

    def find(self, identity):
        canonical = self.snapshot.canonical_identity(identity.key_path, identity.name)
        entry = self.keys.get(_casefold(canonical.key_path))
        if entry is None:
            return None, -1
        for index, value in enumerate(entry[1]):
            if value.identity == canonical:
                return value, index
        return None, -1

    def put(self, value, position=None):
        entry = self.key(value.identity.key_path)
        existing, index = self.find(value.identity)
        if existing is not None:
            entry[1][index] = value
            return
        if position is None or position > len(entry[1]):
            entry[1].append(value)
        else:
            entry[1].insert(position, value)

    def remove(self, identity):
        entry = self.key(identity.key_path)
        existing, index = self.find(identity)
        if existing is not None:
            del entry[1][index]
        return existing

    def build_snapshot(self, revision, parent, changes):
        keys = [KeyState(path, values) for path, values in self.keys.values()]
        return HiveSnapshot(self.snapshot.root_path, keys, revision=revision,
                            parent=parent, changes=changes)


class EditOperation(object):
    """Base class for one value edit."""
    def __init__(self, key_path):
        self.key_path = key_path

    def apply(self, working, group_name, revision):
        raise NotImplementedError()

    def describe(self):
        return self.__class__.__name__

    def _identity(self, working, name):
        return working.snapshot.canonical_identity(self.key_path, name)

    def _require(self, working, name):
        identity = self._identity(working, name)
        value, _ = working.find(identity)
        if value is None:
            raise MissingSourceValueError(
                "Source value does not exist in the original snapshot: %s" % (identity,))
        return value

    def _commit(self, working, source, changed, group_name, revision):
        """Record one transition and return the operation result."""
        before = source.summary()
        after = changed.summary()
        change = ChangeRecord(revision, group_name, self.describe(), before, after)
        changed = changed.with_history(source.history + (change,))
        working.put(changed)
        after = changed.summary()
        return OperationResult(
            self, source_identity=source.identity, target_identity=changed.identity,
            before=before, after=after, read={source.identity},
            write={source.identity, changed.identity}, change=change)


class RenameValue(EditOperation):
    """
    Rename one value, preserving its type and data.

    A default value is renamed by passing ``DEFAULT_VALUE`` or ``""`` as the
    source and a non-empty literal name as the target.  The distinct literal
    value ``(Default)`` can be produced by using that exact target string.
    """
    def __init__(self, key_path, source, target):
        super(RenameValue, self).__init__(key_path)
        self.source = source
        self.target = target

    def _target_name(self):
        if self.target is DEFAULT_VALUE or self.target is None:
            return ""
        return self.target

    def apply(self, working, group_name, revision):
        source = self._require(working, self.source)
        target_identity = self._identity(working, self._target_name())
        before = source.summary()

        if target_identity != source.identity:
            occupied, _ = working.find(target_identity)
            if occupied is not None:
                raise ValueOccupiedError(
                    "Target identity is occupied: %s" % (target_identity,))

        _, position = working.find(source.identity)
        new_identity = ValueIdentity(target_identity.key_path, target_identity.name)
        moved = source.copy_with(identity=new_identity)
        change = ChangeRecord(revision, group_name, self.describe(),
                              before, moved.summary())
        moved = moved.with_history(source.history + (change,))
        if target_identity != source.identity:
            working.remove(source.identity)
        working.put(moved, position)
        return OperationResult(
            self, source_identity=source.identity, target_identity=moved.identity,
            before=before, after=moved.summary(), read={source.identity},
            write={source.identity, moved.identity}, change=change)


class ConvertDefaultValue(RenameValue):
    """
    Convert the unnamed default value into a named value.

    This is an explicit form of rename so that callers cannot accidentally
    address a literal value named ``(Default)`` while intending the unnamed
    default.
    """
    def __init__(self, key_path, target):
        super(ConvertDefaultValue, self).__init__(key_path, DEFAULT_VALUE, target)
        if target is DEFAULT_VALUE or target is None or target == "":
            raise ValueError("A converted default value needs a non-empty target name")

    def _require(self, working, name):
        identity = working.snapshot.canonical_identity(self.key_path, DEFAULT_VALUE)
        value, _ = working.find(identity)
        if value is None:
            raise MissingSourceValueError(
                "The unnamed default value does not exist in the original snapshot: %s"
                % (identity,))
        return value


class DeleteValue(EditOperation):
    """Delete one value from its key."""
    def __init__(self, key_path, name=DEFAULT_VALUE):
        super(DeleteValue, self).__init__(key_path)
        self.name = name

    def apply(self, working, group_name, revision):
        source = self._require(working, self.name)
        before = source.summary()
        working.remove(source.identity)
        after = ValueSummary(source.identity, source.value_type, source.raw_data,
                             source.value, exists=False, origin=source.origin)
        change = ChangeRecord(revision, group_name, self.describe(), before, after)
        return OperationResult(
            self, source_identity=source.identity, target_identity=source.identity,
            before=before, after=after, read={source.identity},
            write={source.identity}, change=change)


class ChangeValueType(EditOperation):
    """
    Change the type of an existing value.

    When ``value`` is omitted the existing data is retained: if the bytes are
    already a valid representation for the new type the raw data is kept,
    otherwise the parsed value is re-encoded.  When ``value`` is provided it
    is encoded with the new type.  An explicit ``value=None`` clears the data.
    """
    def __init__(self, key_path, name, value_type, value=_UNSET):
        super(ChangeValueType, self).__init__(key_path)
        self.name = name
        self.value_type = parse_value_type(value_type)
        self.value = value

    def apply(self, working, group_name, revision):
        source = self._require(working, self.name)
        if self.value is _UNSET:
            value = source.value
            if source.value_type == self.value_type:
                raw_data = source.raw_data
            else:
                raw_data = None
                if isinstance(value, (bytes, bytearray)):
                    if self.value_type in (RegBin, RegNone, RegLink,
                                           RegResourceList,
                                           RegFullResourceDescriptor,
                                           RegResourceRequirementsList):
                        raw_data = bytes(value)
                if raw_data is None:
                    try:
                        raw_data = _encode_value(self.value_type, value)
                    except (UnknownValueTypeError, TypeError, ValueError,
                            struct.error):
                        raw_data = source.raw_data
                        value = _decode_value(self.value_type, raw_data)
                value = _decode_value(self.value_type, raw_data)
        else:
            value = self.value
            raw_data = _encode_value(self.value_type, value)
        changed = source.copy_with(value_type=self.value_type,
                                   raw_data=raw_data, value=value)
        return self._commit(working, source, changed, group_name, revision)


class ClearValue(EditOperation):
    """
    Clear a value's data while keeping the value present.

    The type is preserved, so the result is an existing value with empty data,
    which is distinct from a deleted value and from a value that never existed.
    """
    def __init__(self, key_path, name=DEFAULT_VALUE):
        super(ClearValue, self).__init__(key_path)
        self.name = name

    def apply(self, working, group_name, revision):
        source = self._require(working, self.name)
        if source.is_empty and len(source.raw_data) == 0:
            changed = source
        else:
            changed = source.copy_with(raw_data=b"", value=None)
        return self._commit(working, source, changed, group_name, revision)


class OperationStatus(Enum):
    OK = "ok"
    REJECTED = "rejected"
    SKIPPED = "skipped"


class OperationResult(object):
    """The rehearsal or application outcome of one operation."""
    __slots__ = (
        "operation", "source_identity", "target_identity", "before", "after",
        "read", "write", "status", "reason", "change",
    )

    def __init__(self, operation, source_identity=None, target_identity=None,
                 before=None, after=None, read=None, write=None,
                 status=OperationStatus.OK, reason=None, change=None):
        self.operation = operation
        self.source_identity = source_identity
        self.target_identity = target_identity
        self.before = before
        self.after = after
        self.read = set(read or ())
        self.write = set(write or ())
        self.status = status
        self.reason = reason
        self.change = change

    def describe(self):
        return self.operation.describe()

    def to_dict(self):
        return {
            "operation": self.describe(),
            "status": self.status.value,
            "reason": self.reason,
            "source": str(self.source_identity) if self.source_identity else None,
            "target": str(self.target_identity) if self.target_identity else None,
            "before": self.before.to_dict() if self.before else None,
            "after": self.after.to_dict() if self.after else None,
        }

    def __repr__(self):
        return "OperationResult(%s, status=%s)" % (self.describe(), self.status.value)


class EditGroup(object):
    """
    One atomic group of operations.

    Operations in a group may depend on each other, but the group is either
    applied completely or rejected completely.  Groups are the unit of
    independence checking between configuration changes.
    """
    def __init__(self, name, operations=None, description=None):
        self.name = name
        self.description = description
        self.operations = list(operations or ())

    def add(self, operation):
        self.operations.append(operation)
        return self

    def rename(self, key_path, source, target):
        return self.add(RenameValue(key_path, source, target))

    rename_value = rename

    def convert_default(self, key_path, target):
        return self.add(ConvertDefaultValue(key_path, target))

    convert_default_value = convert_default
    promote_default = convert_default

    def delete(self, key_path, name=DEFAULT_VALUE):
        return self.add(DeleteValue(key_path, name))

    delete_value = delete

    def change_type(self, key_path, name, value_type, value=_UNSET):
        return self.add(ChangeValueType(key_path, name, value_type, value))

    set_type = change_type
    retype = change_type

    def clear(self, key_path, name=DEFAULT_VALUE):
        return self.add(ClearValue(key_path, name))

    clear_value = clear

    def __len__(self):
        return len(self.operations)

    def __repr__(self):
        return "EditGroup(name=%r, operations=%d)" % (self.name, len(self.operations))


class GroupStatus(Enum):
    APPLIED = "applied"
    SKIPPED = "skipped"
    REJECTED = "rejected"


class GroupDependency(object):
    """A cross-group dependency discovered by logical identity."""
    __slots__ = ("left", "right", "identities", "kinds")

    def __init__(self, left, right, identities, kinds):
        self.left = left
        self.right = right
        self.identities = set(identities)
        self.kinds = tuple(sorted(set(kinds)))

    def to_dict(self):
        return {
            "left": self.left,
            "right": self.right,
            "identities": [str(identity) for identity in sorted(self.identities, key=str)],
            "kinds": list(self.kinds),
        }

    def __repr__(self):
        return "GroupDependency(%r, %r, %r)" % (
            self.left, self.right, self.kinds)


class GroupResult(object):
    """Rehearsal outcome for one group."""
    __slots__ = (
        "group", "name", "status", "reason", "operations", "read", "write",
        "dependencies",
    )

    def __init__(self, group, status, reason=None, operations=None,
                 read=None, write=None, dependencies=None):
        self.group = group
        self.name = group.name
        self.status = status
        self.reason = reason
        self.operations = list(operations or ())
        self.read = set(read or ())
        self.write = set(write or ())
        self.dependencies = list(dependencies or ())

    @property
    def is_applied(self):
        return self.status == GroupStatus.APPLIED

    @property
    def is_skipped(self):
        return self.status == GroupStatus.SKIPPED

    @property
    def is_rejected(self):
        return self.status == GroupStatus.REJECTED

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.group.description,
            "status": self.status.value,
            "reason": self.reason,
            "operations": [operation.to_dict() for operation in self.operations],
            "dependencies": [dependency.to_dict() for dependency in self.dependencies],
        }

    def __repr__(self):
        return "GroupResult(name=%r, status=%s, reason=%r)" % (
            self.name, self.status.value, self.reason)


class Rehearsal(object):
    """
    A complete dry run.

    ``expected_snapshot`` is the projected snapshot produced by applying only
    the groups that are both legal and mutually independent.  No input is
    modified.
    """
    def __init__(self, snapshot, groups, dependencies, expected_snapshot):
        self.snapshot = snapshot
        self.groups = list(groups)
        self.dependencies = list(dependencies)
        self.expected_snapshot = expected_snapshot

    @property
    def applied_groups(self):
        return [group for group in self.groups if group.is_applied]

    @property
    def skipped_groups(self):
        return [group for group in self.groups if group.is_skipped]

    @property
    def rejected_groups(self):
        return [group for group in self.groups if group.is_rejected]

    @property
    def is_clean(self):
        return not self.skipped_groups and not self.rejected_groups

    def group(self, name):
        for group in self.groups:
            if group.name == name:
                return group
        raise KeyError(name)

    def expected_state(self, key_path, name=DEFAULT_VALUE):
        if isinstance(key_path, ValueIdentity):
            return self.expected_snapshot.find_value(key_path.key_path, key_path.name)
        return self.expected_snapshot.find_value(key_path, name)

    def expected_summary(self, key_path, name=DEFAULT_VALUE):
        state = self.expected_state(key_path, name)
        if state is None:
            if isinstance(key_path, ValueIdentity):
                identity = key_path
            else:
                identity = ValueIdentity(key_path, name)
            return ValueSummary(identity, RegNone, b"", None, exists=False)
        return state.summary()

    def to_dict(self):
        return {
            "snapshot_revision": self.snapshot.revision,
            "expected_revision": self.expected_snapshot.revision,
            "groups": [group.to_dict() for group in self.groups],
            "dependencies": [dependency.to_dict() for dependency in self.dependencies],
            "applied": [group.name for group in self.applied_groups],
            "skipped": [group.name for group in self.skipped_groups],
            "rejected": [group.name for group in self.rejected_groups],
        }

    def __repr__(self):
        return "Rehearsal(applied=%r, skipped=%r, rejected=%r)" % (
            [group.name for group in self.applied_groups],
            [group.name for group in self.skipped_groups],
            [group.name for group in self.rejected_groups])


class VerificationMismatch(object):
    """One difference between rehearsal expectations and collected state."""
    __slots__ = ("group", "identity", "expected", "actual", "reason")

    def __init__(self, group, identity, expected, actual, reason):
        self.group = group
        self.identity = identity
        self.expected = expected
        self.actual = actual
        self.reason = reason

    def to_dict(self):
        return {
            "group": self.group,
            "identity": str(self.identity) if self.identity else None,
            "reason": self.reason,
            "expected": self.expected.to_dict() if self.expected else None,
            "actual": self.actual.to_dict() if self.actual else None,
        }

    def __repr__(self):
        return "VerificationMismatch(%r, %r, %s)" % (
            self.group, self.identity, self.reason)


class VerificationResult(object):
    """Result of re-collecting state and comparing it with the rehearsal."""
    __slots__ = ("passed", "checked", "mismatches")

    def __init__(self, passed, checked, mismatches):
        self.passed = passed
        self.checked = checked
        self.mismatches = list(mismatches)

    def raise_if_failed(self):
        if not self.passed:
            raise RehearsalVerificationError(
                "Post-apply state differs from rehearsal: %r" % (self.mismatches,))

    def to_dict(self):
        return {
            "passed": self.passed,
            "checked": self.checked,
            "mismatches": [mismatch.to_dict() for mismatch in self.mismatches],
        }

    def __repr__(self):
        return "VerificationResult(passed=%r, checked=%d, mismatches=%d)" % (
            self.passed, self.checked, len(self.mismatches))


class ApplyResult(object):
    """The outcome of applying a plan, including provenance."""
    def __init__(self, rehearsal, post_snapshot, verification):
        self.rehearsal = rehearsal
        self.snapshot = post_snapshot
        self.verification = verification

    @property
    def original_snapshot(self):
        return self.rehearsal.snapshot

    @property
    def applied_groups(self):
        return self.rehearsal.applied_groups

    @property
    def skipped_groups(self):
        return self.rehearsal.skipped_groups

    @property
    def rejected_groups(self):
        return self.rehearsal.rejected_groups

    @property
    def changes(self):
        return list(self.rehearsal.expected_snapshot.changes)

    @property
    def deleted_identities(self):
        return [change.before.identity for change in self.changes
                if change.after is not None and not change.after.exists]

    @property
    def deleted_values(self):
        """Deletion change records, including both captured before and after states."""
        return [
            change for change in self.changes
            if change.after is not None and not change.after.exists
        ]

    @property
    def is_success(self):
        return self.verification.passed

    def history_for(self, identity):
        return self.snapshot.history_for(identity)

    def origin(self, key_path, name=DEFAULT_VALUE):
        """
        Return the original captured state for a logical value, or None.

        The lookup follows renames: asking for a new name returns the origin
        captured under the value's original identity.  Deleted values remain
        queryable by the identity they had at deletion time.
        """
        snapshots = [self.snapshot, self.rehearsal.expected_snapshot,
                     self.rehearsal.snapshot]
        for snapshot in snapshots:
            try:
                identity = snapshot.canonical_identity(key_path, name)
            except KeyNotFoundException:
                continue
            value = snapshot.find_value(identity.key_path, identity.name)
            if value is not None:
                return value.origin

        try:
            identity = self.rehearsal.snapshot.canonical_identity(key_path, name)
        except KeyNotFoundException:
            return None
        for change in self.changes:
            if change.before is not None and change.before.identity == identity:
                return change.before.origin
        return None

    def to_dict(self):
        return {
            "rehearsal": self.rehearsal.to_dict(),
            "verification": self.verification.to_dict(),
            "changes": [
                {
                    "group": change.group,
                    "operation": change.operation,
                    "before": change.before.to_dict(),
                    "after": change.after.to_dict(),
                }
                for change in self.changes
            ],
            "deleted": [str(identity) for identity in self.deleted_identities],
        }

    def __repr__(self):
        return "ApplyResult(applied=%r, verification=%r)" % (
            [group.name for group in self.applied_groups], self.verification)


class EditPlan(object):
    """
    A rehearsal/apply plan over one immutable snapshot.

    Example::

        snapshot = HiveSnapshot.from_registry(reg)
        plan = EditPlan(snapshot)
        plan.add_group(
            EditGroup("timezone")
                .convert_default("TimeZoneInformation", "TimeZoneKeyName")
                .change_type("TimeZoneInformation", "TimeZoneKeyName", RegSZ)
        )
        result = plan.apply()
        result.verification.raise_if_failed()
    """
    def __init__(self, snapshot, groups=None):
        self.snapshot = snapshot
        self.groups = list(groups or ())

    def add_group(self, group, operations=None):
        if isinstance(group, EditGroup):
            if operations:
                for operation in operations:
                    group.add(operation)
            self.groups.append(group)
        else:
            self.groups.append(EditGroup(group, operations or ()))
        return self

    def _execute(self, groups, base_snapshot=None):
        if base_snapshot is None:
            base_snapshot = self.snapshot
        group_results = []
        valid = collections.OrderedDict()
        dependencies = []
        revision = base_snapshot.revision + 1
        name_counts = collections.Counter(group.name for group in groups)

        for group in groups:
            if name_counts[group.name] > 1:
                group_results.append(GroupResult(
                    group, GroupStatus.REJECTED,
                    reason="Duplicate group name: %s" % (group.name,),
                    operations=[
                        OperationResult(operation, status=OperationStatus.REJECTED,
                                        reason="Duplicate group name")
                        for operation in group.operations
                    ]))
                continue

            working = _WorkingHive(base_snapshot)
            operation_results = []
            read = set()
            write = set()
            rejected = False
            rejection_reason = None

            for index, operation in enumerate(group.operations):
                try:
                    result = operation.apply(working, group.name, revision)
                except (EditError, ValueError, KeyError) as error:
                    rejected = True
                    rejection_reason = str(error)
                    operation_results.append(OperationResult(
                        operation, status=OperationStatus.REJECTED,
                        reason=str(error)))
                    for pending in group.operations[index + 1:]:
                        operation_results.append(OperationResult(
                            pending, status=OperationStatus.SKIPPED,
                            reason="Group rejected before this operation"))
                    break
                operation_results.append(result)
                read.update(result.read)
                write.update(result.write)

            if not group.operations:
                group_results.append(GroupResult(
                    group, GroupStatus.REJECTED,
                    reason="Group has no operations",
                    operations=operation_results))
                continue

            if rejected:
                group_results.append(GroupResult(
                    group, GroupStatus.REJECTED,
                    reason=rejection_reason or
                    "One or more operations are not valid against the snapshot",
                    operations=operation_results, read=read, write=write))
                continue

            valid[group.name] = (group, working, operation_results, read, write)
            group_results.append(GroupResult(
                group, GroupStatus.APPLIED, operations=operation_results,
                read=read, write=write))

        # Discover dependencies between otherwise-valid groups.  A group is
        # only applied when it is legal and has no dependency on another legal
        # group; this is the conservative interpretation of "mutually
        # independent legal groups".
        valid_names = list(valid.keys())
        dependency_by_name = collections.defaultdict(list)
        for left_index, left_name in enumerate(valid_names):
            left = valid[left_name]
            for right_name in valid_names[left_index + 1:]:
                right = valid[right_name]
                identities = set()
                kinds = set()
                if left[4] & right[4]:
                    identities.update(left[4] & right[4])
                    kinds.add("write-write")
                if left[4] & right[3]:
                    identities.update(left[4] & right[3])
                    kinds.add("write-read")
                if left[3] & right[4]:
                    identities.update(left[3] & right[4])
                    kinds.add("read-write")
                if identities:
                    dependency = GroupDependency(left_name, right_name,
                                                 identities, kinds)
                    dependencies.append(dependency)
                    dependency_by_name[left_name].append(dependency)
                    dependency_by_name[right_name].append(dependency)

        # A group is applied only when it is legal and is independent of every
        # other legal group.  If two legal groups touch the same logical value
        # they are dependent, so neither is applied: this is the conservative
        # reading of "only apply mutually independent legal groups".
        for result in group_results:
            if result.status != GroupStatus.APPLIED:
                continue
            dependencies = dependency_by_name.get(result.name, ())
            if dependencies:
                result.status = GroupStatus.SKIPPED
                result.dependencies = list(dependencies)
                others = []
                for dependency in dependencies:
                    other = dependency.right if dependency.left == result.name \
                        else dependency.left
                    others.append(other)
                result.reason = "Group is not independent of %s" % (
                    ", ".join(sorted(set(others))))

        working = _WorkingHive(base_snapshot)
        changes = []
        for result in group_results:
            if result.status != GroupStatus.APPLIED:
                continue
            for operation in result.group.operations:
                outcome = operation.apply(working, result.name, revision)
                if outcome.change is not None:
                    changes.append(outcome.change)

        expected = working.build_snapshot(revision, base_snapshot, changes)
        return Rehearsal(base_snapshot, group_results, dependencies, expected)

    def rehearse(self):
        """
        Perform the dry run.

        Returns a :class:`Rehearsal` whose ``expected_snapshot`` is the state
        that a compliant apply operation must reproduce.
        """
        return self._execute(self.groups)

    def apply(self, collector=None):
        """
        Apply the legal, independent groups and verify the collected state.

        The original snapshot is never modified.  The returned
        :class:`ApplyResult` contains the new snapshot, the verification
        report, and every before/after transition.

        By default the state is re-collected from the projected snapshot, so
        the verification compares the rehearsal with what a fresh read of the
        applied configuration would yield.  Pass ``collector`` to supply an
        external store; it receives the projected snapshot and must return a
        snapshot representing the re-collected state.
        """
        rehearsal = self._execute(self.groups)
        post = rehearsal.expected_snapshot.rebuild(parent=rehearsal.snapshot)
        if collector is not None:
            post = collector(post)

        verification = self._verify(rehearsal, post)
        return ApplyResult(rehearsal, post, verification)

    def _verify(self, rehearsal, post):
        mismatches = []
        checked = 0
        expected_snapshot = rehearsal.expected_snapshot
        touched = {}

        for result in rehearsal.applied_groups:
            for outcome in result.operations:
                if outcome.status != OperationStatus.OK:
                    mismatches.append(VerificationMismatch(
                        result.name, outcome.target_identity, outcome.after, None,
                        "An applied group contains a non-OK operation"))
                    continue
                touched[outcome.target_identity.logical] = result.name

        for expected_key in expected_snapshot.keys():
            actual_key = None
            try:
                actual_key = post.key(expected_key.path)
            except KeyNotFoundException:
                actual_key = None

            for expected in expected_key.values:
                group = touched.get(expected.identity.logical, "unchanged")
                actual_state = None
                if actual_key is not None:
                    for candidate in actual_key.values:
                        if candidate.identity == expected.identity:
                            actual_state = candidate
                            break
                actual = actual_state.summary() if actual_state is not None else \
                    ValueSummary(expected.identity, expected.value_type,
                                 b"", None, exists=False,
                                 origin=expected.origin)
                checked += 1

                if expected.value_type != actual.value_type:
                    mismatches.append(VerificationMismatch(
                        group, expected.identity, expected.summary(), actual,
                        "Value type differs from rehearsal"))
                    continue
                if expected.is_empty != actual.is_empty:
                    mismatches.append(VerificationMismatch(
                        group, expected.identity, expected.summary(), actual,
                        "Empty state differs from rehearsal"))
                    continue
                if expected.raw_data != actual.raw_data:
                    mismatches.append(VerificationMismatch(
                        group, expected.identity, expected.summary(), actual,
                        "Value data differs from rehearsal"))

        expected_logical = set(
            value.identity.logical
            for key in expected_snapshot.keys() for value in key.values)
        for actual_key in post.keys():
            for actual in actual_key.values:
                if actual.identity.logical not in expected_logical:
                    checked += 1
                    mismatches.append(VerificationMismatch(
                        "unexpected", actual.identity, None, actual.summary(),
                        "Unexpected value exists after apply but not in rehearsal"))

        for change in expected_snapshot.changes:
            if change.after is not None and not change.after.exists:
                actual_state = post.find_value(change.after.identity.key_path,
                                               change.after.identity.name)
                checked += 1
                if actual_state is not None:
                    mismatches.append(VerificationMismatch(
                        change.group, change.after.identity, change.after,
                        actual_state.summary(),
                        "Deleted value is still present after apply"))

        return VerificationResult(not mismatches, checked, mismatches)


def snapshot_from_registry(registry, revision=1):
    """Capture an immutable snapshot from a parsed Registry instance."""
    return HiveSnapshot.from_registry(registry, revision=revision)


def build_plan(snapshot, groups=()):
    """Create an :class:`EditPlan` for `snapshot` and optional groups."""
    return EditPlan(snapshot, groups)


def rehearse(snapshot, groups=()):
    """Dry-run `groups` against `snapshot` without modifying anything."""
    return EditPlan(snapshot, groups).rehearse()


def apply_edits(snapshot, groups=(), collector=None):
    """Apply the legal, independent groups against `snapshot` and verify."""
    return EditPlan(snapshot, groups).apply(collector=collector)
