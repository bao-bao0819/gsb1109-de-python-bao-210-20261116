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
Resolve an effective configuration from layered sources.

A configuration may be described at three levels:

  - ``system``: machine-wide defaults,
  - ``user``: per-user preferences that override the system defaults,
  - ``policy``: administrator policy that overrides both.

Values are either explicit values or deletion markers.  A deletion marker
does not itself become a value: it hides every value below it and leaves a
``DELETED`` entry in the effective configuration.  This makes it possible to
tell three different situations apart:

  - ``ABSENT``:  the name was never mentioned by any layer,
  - ``EMPTY``:   a layer explicitly set an empty value (``""``, ``b""``, ``[]``,
                 ``{}`` or ``None``); lower layers are *not* consulted,
  - ``DELETED``: a deletion marker hides the lower layers.

Every effective item carries a source chain.  The chain is ordered from the
highest-priority source to the lowest-priority source and records which entry
won and which entries were masked.

Effective configurations are computed from the raw layer entries on every
call.  Withdrawing (or replacing) the policy layer therefore re-reveals the
correct lower-layer values; a previously computed overlay is never reused.

Example::

    from Registry.LayeredConfig import DELETE, LayeredConfig, State

    config = LayeredConfig()
    config.set("system", "browser.homepage", "https://intranet/", "REG_SZ")
    config.set("user", "browser.homepage", "https://example.com/", "REG_SZ")
    config.set("policy", "browser.homepage", DELETE)

    effective = config.effective()
    effective["browser.homepage"].state   # State.DELETED
    effective["browser.homepage"].value   # None

    config.withdraw("policy")
    config.effective()["browser.homepage"].value  # "https://example.com/"

Names are case-insensitive by default, matching registry semantics.  Pass
``case_sensitive=True`` to keep distinct names distinct.
"""
from __future__ import print_function
from __future__ import unicode_literals

from collections import OrderedDict
from enum import Enum
import binascii

try:
    string_types = (basestring,)  # noqa: F821 - Python 2
    text_type = unicode  # noqa: F821 - Python 2
    binary_type = str
except NameError:
    string_types = (str,)
    text_type = str
    binary_type = bytes

__all__ = [
    "Layer",
    "State",
    "EntryKind",
    "SYSTEM",
    "USER",
    "POLICY",
    "DELETE",
    "EMPTY",
    "DeleteTree",
    "ConfigEntry",
    "Source",
    "EffectiveValue",
    "Conflict",
    "EffectiveConfig",
    "LayeredConfig",
    "compute_effective_config",
    "resolve_layers",
    "resolve_config",
]


class Layer(str, Enum):
    """Convenience constants for the three standard configuration layers."""

    SYSTEM = "system"
    USER = "user"
    POLICY = "policy"

    def __str__(self):
        return self.value


SYSTEM = Layer.SYSTEM
USER = Layer.USER
POLICY = Layer.POLICY

STATE_ABSENT = "absent"
STATE_PRESENT = "present"
STATE_EMPTY = "empty"
STATE_DELETED = "deleted"


class State(str, Enum):
    """The state of an item in an effective configuration."""

    ABSENT = STATE_ABSENT
    PRESENT = STATE_PRESENT
    EMPTY = STATE_EMPTY
    DELETED = STATE_DELETED

    MISSING = STATE_ABSENT
    EXPLICIT_EMPTY = STATE_EMPTY

    def __str__(self):
        return self.value


class EntryKind(str, Enum):
    """The kind of an entry supplied by a layer."""

    SET = "set"
    DELETE = "delete"
    DELETE_TREE = "delete_tree"

    def __str__(self):
        return self.value


class DeleteValue(object):
    """Singleton marker that deletes one name from lower layers."""

    __slots__ = ()

    def __repr__(self):
        return "DELETE"

    def __reduce__(self):
        return (_get_delete_value, ())


class DeleteTree(object):
    """Marker that deletes a name and every name below it."""

    __slots__ = ("prefix",)

    def __init__(self, prefix):
        if not isinstance(prefix, string_types) or not prefix:
            raise ValueError("delete tree prefix must be a non-empty string")
        self.prefix = prefix

    def __repr__(self):
        return "DeleteTree({0!r})".format(self.prefix)


def _get_delete_value():
    return DELETE


DELETE = DeleteValue()


class _EmptyValue(object):
    """Explicitly empty value whose type is unknown."""

    __slots__ = ()

    def __repr__(self):
        return "EMPTY"

    def __reduce__(self):
        return (_get_empty_value, ())


def _get_empty_value():
    return EMPTY


EMPTY = _EmptyValue()


def _coerce_layer(layer):
    if isinstance(layer, Layer):
        return layer.value
    if isinstance(layer, string_types):
        return layer.lower()
    return str(layer).lower()


def _split_value_type(value):
    """
    Split the ``(value, value_type)`` shorthand used by ``set_many``.

    A two-tuple whose second item names a known value type is interpreted as
    a value plus its type.  This keeps ordinary two-item sequences (for
    example REG_MULTI_SZ data) usable as plain values.
    """
    if isinstance(value, tuple) and len(value) == 2:
        candidate = value[1]
        if isinstance(candidate, string_types):
            if _canonical_type(candidate) in _KNOWN_TYPES:
                return value[0], candidate
        elif isinstance(candidate, int) and not isinstance(candidate, bool):
            if candidate in _REGISTRY_TYPE_NUMBERS:
                return value[0], candidate
    return value, None


def _infer_type(value):
    """Return a human readable type for a Python value."""
    if value is EMPTY or value is None:
        return "none"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    if isinstance(value, string_types):
        return "string"
    if isinstance(value, (bytes, bytearray)):
        return "binary"
    if isinstance(value, (list, tuple)):
        return "list"
    if isinstance(value, dict):
        return "dict"
    if isinstance(value, set):
        return "set"
    return type(value).__name__


_TYPE_ALIASES = {
    "str": "reg_sz",
    "string": "reg_sz",
    "sz": "reg_sz",
    "unicode": "reg_sz",
    "regsz": "reg_sz",
    "reg_sz": "reg_sz",
    "expand_sz": "reg_expand_sz",
    "regexpandsz": "reg_expand_sz",
    "reg_expand_sz": "reg_expand_sz",
    "int": "reg_dword",
    "integer": "reg_dword",
    "dword": "reg_dword",
    "regdword": "reg_dword",
    "reg_dword": "reg_dword",
    "int32": "reg_dword",
    "uint32": "reg_dword",
    "long": "reg_dword",
    "qword": "reg_qword",
    "regqword": "reg_qword",
    "reg_qword": "reg_qword",
    "int64": "reg_qword",
    "uint64": "reg_qword",
    "binary": "reg_binary",
    "bytes": "reg_binary",
    "regbin": "reg_binary",
    "reg_binary": "reg_binary",
    "multi_sz": "reg_multi_sz",
    "multisz": "reg_multi_sz",
    "regmulti_sz": "reg_multi_sz",
    "reg_multi_sz": "reg_multi_sz",
    "bool": "bool",
    "boolean": "bool",
    "float": "float",
    "double": "reg_double",
    "regdouble": "reg_double",
    "reg_double": "reg_double",
    "none": "none",
    "null": "none",
    "regnone": "none",
}


_REGISTRY_TYPE_NUMBERS = {
    0x0000: "none",
    0x0001: "reg_sz",
    0x0002: "reg_expand_sz",
    0x0003: "reg_binary",
    0x0004: "reg_dword",
    0x0005: "reg_big_endian",
    0x0006: "reg_link",
    0x0007: "reg_multi_sz",
    0x0008: "reg_resource_list",
    0x0009: "reg_full_resource_descriptor",
    0x000A: "reg_resource_requirements_list",
    0x000B: "reg_qword",
    0x0010: "reg_filetime",
}

_KNOWN_TYPES = frozenset(
    set(_TYPE_ALIASES) | set(_REGISTRY_TYPE_NUMBERS.values()) | {
        "regsz",
        "regexpandsz",
        "regbin",
        "regdword",
        "regqword",
        "regmulti_sz",
        "regnone",
        "regdouble",
        "regfiletime",
        "regbig_endian",
        "reglink",
        "reg_resource_list",
        "reg_full_resource_descriptor",
        "reg_resource_requirements_list",
    }
)


def _canonical_type(value_type):
    if isinstance(value_type, bool):
        raise ValueError("value_type must not be a bool")
    if isinstance(value_type, int):
        try:
            return _REGISTRY_TYPE_NUMBERS[value_type]
        except KeyError:
            return "0x{0:x}".format(value_type)
    value_type = value_type.strip().lower().replace("-", "_")
    return _TYPE_ALIASES.get(value_type, value_type)


def _type_token(label):
    """Return a canonical type for conflict checks, or ``None`` when untyped."""
    if not label:
        return None
    canonical = _canonical_type(label)
    if canonical == "none":
        return None
    return canonical


def _is_empty(value):
    if value is EMPTY or value is None:
        return True
    if isinstance(value, (string_types, bytes, bytearray, list, tuple, dict, set)):
        return len(value) == 0
    return False


def _jsonable(value):
    if isinstance(value, binary_type):
        return binascii.hexlify(value).decode("ascii")
    if isinstance(value, bytearray):
        return binascii.hexlify(value).decode("ascii")
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (text_type, bool, int, float)) or value is None:
        return value
    return repr(value)


class ConfigEntry(object):
    """One entry contributed by one layer."""

    __slots__ = ("layer", "name", "kind", "value", "value_type", "origin", "index")

    def __init__(self, layer, name, kind, value=None, value_type=None, origin=None, index=0):
        self.layer = layer
        self.name = name
        self.kind = kind
        self.value = value
        self.value_type = value_type
        self.origin = origin
        self.index = index

    @property
    def is_deletion(self):
        return self.kind in (EntryKind.DELETE, EntryKind.DELETE_TREE)

    @property
    def inferred_type(self):
        if self.kind is not EntryKind.SET:
            return None
        return _infer_type(self.value)

    @property
    def type_label(self):
        """The declared type, or the inferred type when none was declared."""
        if self.kind is not EntryKind.SET:
            return None
        if self.value_type is None:
            return self.inferred_type
        if isinstance(self.value_type, int):
            return _canonical_type(self.value_type)
        return self.value_type

    def __repr__(self):
        return "ConfigEntry(layer={0!r}, name={1!r}, kind={2!r}, value={3!r}, type={4!r})".format(
            self.layer, self.name, self.kind.value, self.value, self.type_label
        )

    def to_dict(self):
        return {
            "layer": self.layer,
            "name": self.name,
            "kind": self.kind.value,
            "value": _jsonable(self.value) if self.kind is EntryKind.SET else None,
            "value_type": self.type_label,
            "origin": self.origin,
            "index": self.index,
        }


class Source(object):
    """A source-chain link: one layer entry relevant to an effective item."""

    __slots__ = ("entry", "outcome", "scope", "matched_name")

    def __init__(self, entry, outcome, scope="name", matched_name=None):
        self.entry = entry
        self.outcome = outcome
        self.scope = scope
        self.matched_name = matched_name

    @property
    def layer(self):
        return self.entry.layer

    @property
    def name(self):
        return self.entry.name

    @property
    def origin(self):
        return self.entry.origin

    @property
    def kind(self):
        return self.entry.kind

    @property
    def value(self):
        return self.entry.value

    @property
    def value_type(self):
        return self.entry.type_label

    @property
    def is_winner(self):
        return self.outcome in ("winner", "tombstone")

    @property
    def is_masked(self):
        return self.outcome == "masked"

    def __repr__(self):
        return "Source(layer={0!r}, name={1!r}, kind={2!r}, outcome={3!r})".format(
            self.layer, self.name, self.kind.value, self.outcome
        )

    def to_dict(self):
        return {
            "layer": self.layer,
            "name": self.name,
            "kind": self.kind.value,
            "value": _jsonable(self.value) if self.kind is EntryKind.SET else None,
            "value_type": self.value_type,
            "origin": self.origin,
            "index": self.entry.index,
            "outcome": self.outcome,
            "scope": self.scope,
            "matched_name": self.matched_name,
        }


class EffectiveValue(object):
    """The effective state, value and source chain for one name."""

    __slots__ = ("name", "state", "value", "value_type", "source_chain")

    def __init__(self, name, state, value=None, value_type=None, source_chain=()):
        self.name = name
        self.state = state
        self.value = value
        self.value_type = value_type
        self.source_chain = tuple(source_chain)

    @property
    def is_absent(self):
        return self.state is State.ABSENT

    @property
    def is_empty(self):
        return self.state is State.EMPTY

    @property
    def is_deleted(self):
        return self.state is State.DELETED

    @property
    def is_present(self):
        return self.state in (State.PRESENT, State.EMPTY)

    @property
    def winner(self):
        for source in self.source_chain:
            if source.is_winner:
                return source
        return None

    @property
    def layers(self):
        return tuple(source.layer for source in self.source_chain)

    @property
    def sources(self):
        """Alias for :attr:`source_chain`."""
        return self.source_chain

    @property
    def status(self):
        """Alias for :attr:`state`."""
        return self.state

    @property
    def origin(self):
        """The origin of the winning source, if any."""
        source = self.winner
        return source.origin if source is not None else None

    def __repr__(self):
        return "EffectiveValue(name={0!r}, state={1!r}, value={2!r}, type={3!r})".format(
            self.name, self.state.value, self.value, self.value_type
        )

    def to_dict(self):
        return {
            "name": self.name,
            "state": self.state.value,
            "value": _jsonable(self.value) if self.is_present else None,
            "value_type": self.value_type,
            "is_empty": self.is_empty,
            "sources": [source.to_dict() for source in self.source_chain],
        }


class Conflict(object):
    """A name whose sources declare different types."""

    __slots__ = ("name", "entries", "types")

    def __init__(self, name, entries, types):
        self.name = name
        self.entries = tuple(entries)
        self.types = tuple(types)

    @property
    def layers(self):
        return tuple(entry.layer for entry in self.entries)

    def __repr__(self):
        return "Conflict(name={0!r}, types={1!r}, layers={2!r})".format(
            self.name, self.types, self.layers
        )

    def to_dict(self):
        return {
            "name": self.name,
            "types": list(self.types),
            "layers": list(self.layers),
            "entries": [entry.to_dict() for entry in self.entries],
        }


class EffectiveConfig(object):
    """
    A snapshot of an effective configuration.

    The object behaves like a read-only mapping of names to
    :class:`EffectiveValue`.  Unknown names are not present in the mapping;
    use :meth:`resolve` or :meth:`state` to get an explicit ``ABSENT`` result.
    """

    def __init__(self, items, conflicts, order, revision=0, case_sensitive=False):
        self._items = OrderedDict(sorted(items.items(), key=lambda pair: pair[0]))
        self._conflicts = tuple(conflicts)
        self.order = tuple(order)
        self.revision = revision
        self.case_sensitive = bool(case_sensitive)

    def __getitem__(self, name):
        return self._items[self._key(name)]

    def __contains__(self, name):
        return self._key(name) in self._items

    def __iter__(self):
        for item in self._items.values():
            yield item.name

    def __len__(self):
        return len(self._items)

    def __repr__(self):
        return "EffectiveConfig(items={0}, conflicts={1}, revision={2})".format(
            len(self._items), len(self._conflicts), self.revision
        )

    @property
    def conflicts(self):
        return self._conflicts

    def _key(self, name):
        return name if self.case_sensitive else name.lower()

    def keys(self):
        return [item.name for item in self._items.values()]

    def values(self):
        return list(self._items.values())

    def items(self):
        return [(item.name, item) for item in self._items.values()]

    def get(self, name, default=None):
        try:
            return self[name]
        except KeyError:
            return default

    def resolve(self, name):
        """
        Return the effective value for a name, or an explicit ``ABSENT``
        value when no layer mentioned it.
        """
        try:
            return self[name]
        except KeyError:
            return EffectiveValue(name, State.ABSENT)

    def state(self, name):
        return self.resolve(name).state

    def source_chain(self, name):
        return self.resolve(name).source_chain

    def to_dict(self):
        return {
            "order": list(self.order),
            "revision": self.revision,
            "effective": OrderedDict(
                (item.name, item.to_dict()) for item in self._items.values()
            ),
            "conflicts": [conflict.to_dict() for conflict in self._conflicts],
        }

    def conflict_names(self):
        return [conflict.name for conflict in self._conflicts]

    def is_conflicted(self, name):
        key = self._key(name)
        for conflict in self._conflicts:
            if self._key(conflict.name) == key:
                return True
        return False

    def absent(self):
        """Return the names of every candidate that is not present."""
        absent = []
        for item in self._items.values():
            if item.state is State.ABSENT:
                absent.append(item.name)
        return absent

    def plain(self):
        """
        Return a plain ``{name: value}`` mapping.

        ``DELETED`` and ``ABSENT`` items map to ``None`` and ``EMPTY`` items
        keep their empty value.  Use :meth:`resolve` when the state matters.
        """
        return OrderedDict(
            (item.name, item.value if item.is_present else None)
            for item in self._items.values()
        )

    as_dict = plain

    def states(self):
        return OrderedDict((item.name, item.state) for item in self._items.values())

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


class LayeredConfig(object):
    """
    A set of configuration layers that can be resolved to an effective view.

    The default priority, from lowest to highest, is ``system``, ``user``,
    ``policy``.  Entries within one layer are applied in insertion order, so
    the last write wins.  Deletion markers work exactly like values for the
    purpose of ordering: a marker masks everything below it, and a higher
    layer or a later entry can override the marker again.

    ``effective()`` always rebuilds the result from the raw entries.  It does
    not mutate lower layers and does not cache an overlay, so ``withdraw()``
    (or ``replace_layer()``) correctly re-reveals lower values.
    """

    DEFAULT_ORDER = ("system", "user", "policy")

    def __init__(self, order=None, case_sensitive=False, system=None,
                 user=None, policy=None):
        if order is None:
            order = self.DEFAULT_ORDER
        layers = []
        for layer in order:
            name = _coerce_layer(layer)
            if name in layers:
                raise ValueError("duplicate layer: {0}".format(name))
            layers.append(name)
        if not layers:
            raise ValueError("at least one layer is required")

        self._order = tuple(layers)
        self._ranks = {name: rank for rank, name in enumerate(layers)}
        self._layers = OrderedDict((name, []) for name in layers)
        self._case_sensitive = bool(case_sensitive)
        self._revision = 0

        for layer_name, values in (
            ("system", system),
            ("user", user),
            ("policy", policy),
        ):
            if values:
                if layer_name not in self._ranks:
                    continue
                self.set_many(layer_name, values)

    def __repr__(self):
        counts = ", ".join(
            "{0}={1}".format(name, len(self._layers[name])) for name in self._order
        )
        return "LayeredConfig({0})".format(counts)

    @property
    def order(self):
        return self._order

    @property
    def revision(self):
        return self._revision

    def _normalize_layer(self, layer):
        name = _coerce_layer(layer)
        if name not in self._ranks:
            raise ValueError(
                "unknown layer: {0!r} (known layers: {1})".format(
                    layer, ", ".join(self._order)
                )
            )
        return name

    def _name_key(self, name):
        if not isinstance(name, string_types) or not name:
            raise ValueError("name must be a non-empty string")
        return name if self._case_sensitive else name.lower()

    def _append(self, entry):
        entries = self._layers[entry.layer]
        entry.index = len(entries)
        entries.append(entry)
        self._revision += 1
        return entry

    def set(self, layer, name, value, value_type=None, origin=None):
        """Set a value in a layer."""
        layer = self._normalize_layer(layer)
        if isinstance(value, DeleteTree):
            return self.delete_tree(layer, value.prefix, origin=origin)
        if value is DELETE:
            return self.delete(layer, name, origin=origin)
        if value is EMPTY:
            value = None
            if value_type is None:
                value_type = "none"
        return self._append(
            ConfigEntry(
                layer=layer,
                name=name,
                kind=EntryKind.SET,
                value=value,
                value_type=value_type,
                origin=origin,
            )
        )

    set_value = set

    def set_many(self, layer, values):
        """
        Set multiple values in a layer.

        ``values`` may be a mapping or an iterable of ``(name, value)`` pairs.
        A value of ``DELETE`` becomes a deletion marker.  A value of
        ``DeleteTree(prefix)`` becomes a subtree deletion marker.  A pair of
        ``(value, value_type)`` is accepted when the second item names a known
        type, for example ``{"app.theme": ("dark", "REG_SZ")}``.
        """
        if hasattr(values, "items"):
            values = values.items()
        entries = []
        for name, value in values:
            value, value_type = _split_value_type(value)
            entries.append(self.set(layer, name, value, value_type))
        return entries

    apply = set_many

    def delete(self, layer, name, origin=None):
        """Add a marker that deletes one name from lower layers."""
        layer = self._normalize_layer(layer)
        self._name_key(name)
        return self._append(
            ConfigEntry(
                layer=layer,
                name=name,
                kind=EntryKind.DELETE,
                origin=origin,
            )
        )

    def delete_tree(self, layer, prefix, origin=None):
        """Add a marker that deletes a name and every name below it."""
        layer = self._normalize_layer(layer)
        self._name_key(prefix)
        return self._append(
            ConfigEntry(
                layer=layer,
                name=prefix,
                kind=EntryKind.DELETE_TREE,
                origin=origin,
            )
        )

    def withdraw(self, layer):
        """
        Remove every entry contributed by a layer.

        The next call to :meth:`effective` rebuilds the result from the
        remaining raw entries, so lower-layer values become visible again.
        """
        layer = self._normalize_layer(layer)
        self._layers[layer] = []
        self._revision += 1

    clear_layer = withdraw

    def revoke(self, layer):
        """Alias for :meth:`withdraw`."""
        return self.withdraw(layer)

    def remove(self, layer, name=None):
        """
        Remove raw entries contributed by a layer.

        With ``name`` omitted this behaves like :meth:`withdraw`.  With a
        ``name`` only the entries for that name are dropped, which is how a
        single withdrawn policy item is expressed.  This is different from
        :meth:`delete`, which adds a marker that hides lower layers.
        """
        layer = self._normalize_layer(layer)
        if name is None:
            return self.withdraw(layer)
        key = self._name_key(name)
        entries = self._layers[layer]
        kept = [entry for entry in entries if self._name_key(entry.name) != key]
        removed = len(entries) - len(kept)
        if removed:
            self._layers[layer] = kept
            for index, entry in enumerate(kept):
                entry.index = index
            self._revision += 1
        return removed

    def has_entries(self, layer):
        return bool(self._layers[self._normalize_layer(layer)])

    def layers(self):
        """Return the layer names in ascending priority order."""
        return self._order

    def raw(self):
        """Return the raw layer entries as ``{layer: [ConfigEntry, ...]}``."""
        return OrderedDict(
            (name, list(self._layers[name])) for name in self._order
        )

    def set_layer(self, layer, values):
        """Withdraw a layer and populate it from scratch."""
        layer = self._normalize_layer(layer)
        self.withdraw(layer)
        self.set_many(layer, values)

    replace_layer = set_layer

    def entries(self, layer=None):
        """Return the raw entries, optionally limited to one layer."""
        if layer is None:
            result = []
            for name in self._order:
                result.extend(self._layers[name])
            return tuple(result)
        return tuple(self._layers[self._normalize_layer(layer)])

    def _prefix_matches(self, prefix, name_key):
        prefix_key = self._name_key(prefix).rstrip("\\/.")
        if not prefix_key:
            return False
        if name_key == prefix_key:
            return True
        for separator in ("\\", "/", "."):
            if name_key.startswith(prefix_key + separator):
                return True
        return False

    @staticmethod
    def _sort_key(entry, ranks):
        return (ranks[entry.layer], entry.index)

    def _type_conflicts(self, set_entries):
        tokens = []
        for entry in set_entries:
            token = _type_token(entry.type_label)
            if token is not None and token not in tokens:
                tokens.append(token)
        return len(tokens) > 1

    def _compute(self):
        set_entries = OrderedDict()
        exact_deletions = OrderedDict()
        tree_deletions = []

        for layer in self._order:
            for entry in self._layers[layer]:
                key = self._name_key(entry.name)
                if entry.kind is EntryKind.SET:
                    set_entries.setdefault(key, []).append(entry)
                elif entry.kind is EntryKind.DELETE:
                    exact_deletions.setdefault(key, []).append(entry)
                else:
                    tree_deletions.append(entry)

        candidate_keys = list(set_entries.keys())
        seen_keys = set(candidate_keys)
        for key in exact_deletions:
            if key not in seen_keys:
                candidate_keys.append(key)
                seen_keys.add(key)
        for tree in tree_deletions:
            matched = False
            for key in set_entries:
                if key not in seen_keys and self._prefix_matches(tree.name, key):
                    candidate_keys.append(key)
                    seen_keys.add(key)
                    matched = True
            if not matched:
                prefix_key = self._name_key(tree.name)
                if prefix_key not in seen_keys:
                    candidate_keys.append(prefix_key)
                    seen_keys.add(prefix_key)

        items = {}
        conflicts = []

        for key in candidate_keys:
            relevant = list(set_entries.get(key, ()))
            relevant.extend(exact_deletions.get(key, ()))
            for entry in tree_deletions:
                if self._prefix_matches(entry.name, key):
                    relevant.append(entry)
            relevant.sort(key=lambda entry: self._sort_key(entry, self._ranks))
            winner = relevant[-1]

            if winner.kind is EntryKind.DELETE:
                display_name = winner.name
            elif winner.kind is EntryKind.DELETE_TREE:
                lower_sets = [entry for entry in relevant if entry.kind is EntryKind.SET]
                if lower_sets:
                    display_name = lower_sets[-1].name
                else:
                    exact = [entry for entry in relevant if entry.kind is EntryKind.DELETE]
                    display_name = exact[-1].name if exact else winner.name
            else:
                display_name = winner.name

            if winner.kind in (EntryKind.DELETE, EntryKind.DELETE_TREE):
                state = State.DELETED
                value = None
                value_type = None
            else:
                state = State.EMPTY if _is_empty(winner.value) else State.PRESENT
                value = None if winner.value is EMPTY else winner.value
                value_type = winner.type_label

            chain = []
            for chain_index, entry in enumerate(reversed(relevant)):
                if chain_index == 0:
                    outcome = "tombstone" if entry.is_deletion else "winner"
                else:
                    outcome = "masked"
                scope = "subtree" if entry.kind is EntryKind.DELETE_TREE else "name"
                chain.append(
                    Source(
                        entry=entry,
                        outcome=outcome,
                        scope=scope,
                        matched_name=display_name,
                    )
                )

            items[key] = EffectiveValue(
                name=display_name,
                state=state,
                value=value,
                value_type=value_type,
                source_chain=chain,
            )

            candidates = set_entries.get(key, ())
            if len(candidates) > 1 and self._type_conflicts(candidates):
                ordered = sorted(
                    candidates,
                    key=lambda entry: self._sort_key(entry, self._ranks),
                    reverse=True,
                )
                types = []
                for entry in ordered:
                    label = entry.type_label
                    if label not in types:
                        types.append(label)
                conflicts.append(Conflict(display_name, ordered, types))

        conflicts.sort(key=lambda conflict: self._name_key(conflict.name))
        return EffectiveConfig(
            items,
            conflicts,
            order=self._order,
            revision=self._revision,
            case_sensitive=self._case_sensitive,
        )

    def effective(self):
        """Compute and return a fresh :class:`EffectiveConfig`."""
        return self._compute()

    def get(self, name):
        """Return the effective value for a name (freshly computed)."""
        return self._compute().resolve(name)

    def state(self, name):
        return self._compute().state(name)

    def resolve(self, name):
        return self._compute().resolve(name)

    def source_chain(self, name):
        return self._compute().source_chain(name)

    def sources(self, name):
        """Alias for :meth:`source_chain`."""
        return self.source_chain(name)

    def conflicts(self):
        return self._compute().conflicts

    def conflict_names(self):
        return self._compute().conflict_names()

    def states(self):
        return self._compute().states()

    def resolve_all(self, names):
        """Resolve several names at once, returning a list of values."""
        effective = self._compute()
        return [effective.resolve(name) for name in names]

    def to_dict(self):
        return self._compute().to_dict()

    @classmethod
    def from_mappings(cls, system=None, user=None, policy=None, order=None,
                      case_sensitive=False):
        """Build a configuration from three optional mappings."""
        return cls(
            order=order,
            case_sensitive=case_sensitive,
            system=system,
            user=user,
            policy=policy,
        )


def compute_effective_config(system=None, user=None, policy=None, order=None,
                             case_sensitive=False):
    """Convenience wrapper returning a fresh :class:`EffectiveConfig`."""
    return LayeredConfig.from_mappings(
        system=system,
        user=user,
        policy=policy,
        order=order,
        case_sensitive=case_sensitive,
    ).effective()


resolve_layers = compute_effective_config
resolve_config = compute_effective_config
