"""
Target-environment expansion for REG_EXPAND_SZ strings.

The acquisition host is not necessarily the system that produced the hive, so
this module never falls back to ``os.environ``.  Callers provide the target
environment explicitly and receive both the untouched source string and an
effective view, together with per-occurrence provenance.

A variable may refer to another variable.  Cycles and depth limits stop
recursion and leave the offending placeholder in place so that the result is
always finite.  Every textual occurrence of a variable is recorded separately:
two variables that resolve to the same text remain distinguishable in the
provenance records.

Variables absent from the target environment, or present with a value of
``None``, are left as placeholders and reported as unresolved/pending.  This is
deliberate: the decision may come from another hive, another control set, or an
external source that is not available yet.

The expansion result is a ``str`` subclass whose string value is the effective
view, so existing string-based code can use it directly.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from collections import OrderedDict

try:
    from collections.abc import Mapping
except ImportError:  # Python 2
    from collections import Mapping

try:
    string_types = (basestring,)
    text_type = unicode
except NameError:  # Python 3
    string_types = (str,)
    text_type = str


REG_EXPAND_SZ = 0x0002

STATUS_LITERAL = "literal"
STATUS_NOT_EXPANDABLE = "not_expandable"
STATUS_RESOLVED = "resolved"
STATUS_PARTIAL = "partial"
STATUS_UNRESOLVED = "unresolved"
STATUS_CYCLE = "cycle"
STATUS_DEPTH_LIMIT = "depth_limit"

_STATUS_PRECEDENCE = {
    STATUS_LITERAL: -1,
    STATUS_RESOLVED: 0,
    STATUS_PARTIAL: 1,
    STATUS_DEPTH_LIMIT: 2,
    STATUS_UNRESOLVED: 3,
    STATUS_CYCLE: 4,
}


def _to_text(value):
    if value is None:
        raise ValueError("value must not be None")
    if isinstance(value, text_type):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.decode("windows-1252")
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return text_type(value)
    return text_type(value)


def _merge_status(current, new):
    if _STATUS_PRECEDENCE.get(new, 0) > _STATUS_PRECEDENCE.get(current, 0):
        return new
    return current


class ExpansionPart(object):
    """
    One part of an expanded string.

    Literal text and variable replacements are both represented.  Variable
    parts carry the source placeholder, the resolved variable name, the chain
    of variables that was followed, and a status.  ``start`` and ``end`` are
    offsets into the effective string, so parts explain exactly where each
    piece of text came from.

    Attributes are also readable through ``item[...]`` and ``item.get(...)``
    with the names returned by ``as_dict()``.
    """

    _FIELDS = ("kind", "source", "name", "chain", "status", "text",
               "start", "end", "depth")

    def __init__(self, kind, text, start, end, source=None, name=None,
                 chain=(), status=STATUS_LITERAL, depth=0):
        self.kind = kind
        self._text = text
        self.start = start
        self.end = end
        self.source = source
        self.name = name
        self.chain = tuple(chain)
        self.status = status
        self.depth = depth

    @property
    def text(self):
        return self._text

    @property
    def value(self):
        return self._text

    @property
    def replacement(self):
        return self._text

    @property
    def resolved_value(self):
        if self.kind != "variable" or self.status not in (
                STATUS_RESOLVED, STATUS_PARTIAL):
            return None
        return self._text

    @property
    def variable(self):
        return self.name

    @property
    def variable_name(self):
        return self.name

    @property
    def placeholder(self):
        return self.source if self.kind == "variable" else None

    @property
    def is_variable(self):
        return self.kind == "variable"

    @property
    def is_literal(self):
        return self.kind == "literal"

    @property
    def is_resolved(self):
        return self.kind == "literal" or self.status in (
            STATUS_RESOLVED, STATUS_PARTIAL)

    @property
    def span(self):
        return (self.start, self.end)

    def shift(self, offset):
        self.start += offset
        self.end += offset

    def __getitem__(self, key):
        if isinstance(key, string_types):
            if key in self._FIELDS:
                return getattr(self, key)
            raise KeyError(key)
        return self._text[key]

    def get(self, key, default=None):
        if key in self._FIELDS:
            return getattr(self, key)
        return default

    def keys(self):
        return list(self._FIELDS)

    def items(self):
        return [(key, getattr(self, key)) for key in self._FIELDS]

    def __iter__(self):
        return iter(self._FIELDS)

    def __len__(self):
        return len(self._FIELDS)

    def __contains__(self, key):
        return key in self._FIELDS

    def explain(self):
        return self.as_dict()

    def as_dict(self):
        return {
            "kind": self.kind,
            "source": self.source,
            "placeholder": self.placeholder,
            "name": self.name,
            "variable": self.name,
            "chain": list(self.chain),
            "status": self.status,
            "text": self._text,
            "replacement": self._text,
            "start": self.start,
            "end": self.end,
            "depth": self.depth,
        }

    def __str__(self):
        return self._text

    def __repr__(self):
        return (
            'ExpansionPart(kind="{0}", source="{1}", text="{2}", '
            'status="{3}", chain={4})'
        ).format(self.kind, self.source, self._text, self.status,
                 list(self.chain))

    def __eq__(self, other):
        if isinstance(other, ExpansionPart):
            return self.as_dict() == other.as_dict()
        if isinstance(other, Mapping):
            return self.as_dict() == dict(other)
        if isinstance(other, string_types):
            return self._text == other
        return NotImplemented

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    __hash__ = None


class EnvironmentExpansionResult(text_type):
    """
    The effective view of a REG_EXPAND_SZ string.

    The object itself is a string containing the effective text.  ``raw`` is
    exactly the source string, while ``provenance`` contains one record per
    variable occurrence in the source string. ``trace`` additionally contains
    records for references nested inside variable values. Records are never
    merged by resolved text or variable name.
    """

    def __new__(cls, raw, effective, parts, environment, max_depth, status):
        self = text_type.__new__(cls, effective)
        self._raw = _to_text(raw)
        self._trace = tuple(sorted(
            parts,
            key=lambda part: (
                part.start,
                part.depth,
                part.end,
            )))
        self._parts = tuple(part for part in self._trace
                            if part.depth == 0)
        self._environment = environment
        self.max_depth = max_depth
        self._status = status
        return self


    @property
    def raw(self):
        return self._raw

    @property
    def original(self):
        return self._raw

    @property
    def original_value(self):
        return self._raw

    @property
    def stored(self):
        return self._raw

    @property
    def raw_value(self):
        return self._raw

    @property
    def raw_string(self):
        return self._raw

    @property
    def source(self):
        return self._raw

    @property
    def effective(self):
        return text_type(self)

    @property
    def expanded(self):
        return text_type(self)

    @property
    def expanded_string(self):
        return text_type(self)

    @property
    def effective_string(self):
        return text_type(self)

    @property
    def expanded_value(self):
        return text_type(self)

    @property
    def effective_value(self):
        return text_type(self)

    @property
    def effective_view(self):
        return text_type(self)

    @property
    def value(self):
        return text_type(self)

    @property
    def text(self):
        return text_type(self)

    @property
    def status(self):
        return self._status

    @property
    def is_complete(self):
        return self._status in (STATUS_RESOLVED, STATUS_NOT_EXPANDABLE)

    @property
    def is_resolved(self):
        return self.is_complete

    @property
    def is_partial(self):
        return self._status == STATUS_PARTIAL

    @property
    def has_unresolved(self):
        return any(part.status == STATUS_UNRESOLVED
                   for part in self._trace)

    @property
    def has_cycles(self):
        return any(part.status == STATUS_CYCLE
                   for part in self._trace)

    @property
    def has_depth_limited(self):
        return any(part.status == STATUS_DEPTH_LIMIT
                   for part in self._trace)

    @property
    def parts(self):
        return list(self._parts)

    @property
    def segments(self):
        return list(self._parts)

    @property
    def provenance(self):
        """
        One record per variable occurrence in the source string. Nested
        references are described by each record's ``chain`` and exposed
        separately through ``trace``.
        """
        return [part for part in self._parts
                if part.kind == "variable"]

    @property
    def trace(self):
        """
        Every variable occurrence, including those nested inside the values
        of other variables.
        """
        return [part for part in self._trace
                if part.kind == "variable"]

    @property
    def substitutions(self):
        return self.provenance

    @property
    def records(self):
        return self.provenance

    @property
    def occurrences(self):
        return self.provenance

    @property
    def origins(self):
        return self.provenance

    @property
    def sources(self):
        return self.provenance

    def _names_with_status(self, status):
        names = []
        seen = set()
        for part in self._trace:
            if part.kind != "variable" or part.status != status:
                continue
            name = part.name
            key = name.upper()
            if key in seen:
                continue
            seen.add(key)
            names.append(name)
        return names

    @property
    def unresolved(self):
        return self._names_with_status(STATUS_UNRESOLVED)

    @property
    def unresolved_variables(self):
        return self.unresolved

    @property
    def unresolved_names(self):
        return self.unresolved

    @property
    def unresolved_records(self):
        return [part for part in self.trace
                if part.status == STATUS_UNRESOLVED]

    @property
    def cycles(self):
        """
        Names that participate in a reference cycle.

        A cycle record is the point at which an already-active variable is
        encountered again. Its chain holds the variables along the loop, so
        every participant is reported, not just the repeated one.
        """
        names = []
        seen = set()
        for part in self.trace:
            if part.status != STATUS_CYCLE:
                continue
            chain = part.chain
            participants = list(chain)
            if chain:
                repeated = chain[-1].upper()
                start = None
                for index, name in enumerate(chain[:-1]):
                    if name.upper() == repeated:
                        start = index
                        break
                if start is not None:
                    participants = chain[start:]
            for name in participants:
                if name.upper() in seen:
                    continue
                seen.add(name.upper())
                names.append(name)
        return names

    @property
    def cycle_variables(self):
        return self.cycles

    @property
    def cycle_records(self):
        return [part for part in self.trace
                if part.status == STATUS_CYCLE]

    @property
    def depth_limited(self):
        return self._names_with_status(STATUS_DEPTH_LIMIT)

    @property
    def depth_limited_variables(self):
        return self.depth_limited

    @property
    def pending(self):
        """
        Names still requiring a decision, in first-seen order.

        This includes unbound variables, cycle participants and variables
        stopped by the depth limit.
        """
        pending = []
        seen = set()
        cycle_names = self.cycles
        for part in self._trace:
            if part.kind != "variable" or part.status not in (
                    STATUS_UNRESOLVED, STATUS_CYCLE, STATUS_DEPTH_LIMIT):
                continue
            if part.status == STATUS_CYCLE:
                names = cycle_names
            else:
                names = (part.name,)
            for name in names:
                if name.upper() in seen:
                    continue
                seen.add(name.upper())
                pending.append(name)
        return pending

    @property
    def pending_variables(self):
        return self.pending

    @property
    def variable_names(self):
        names = []
        seen = set()
        for part in self.provenance:
            if part.kind != "variable":
                continue
            if part.name.upper() in seen:
                continue
            seen.add(part.name.upper())
            names.append(part.name)
        return names

    @property
    def environment(self):
        return self._environment.as_dict()

    @property
    def variables(self):
        """Variable names referenced by the source string, in order."""
        return self.variable_names

    @property
    def target_environment(self):
        return self._environment

    @property
    def env(self):
        return self._environment.as_dict()

    def with_environment(self, environment, max_depth=None):
        """
        Re-expand the original string with this environment extended by
        ``environment``. Useful to resolve variables that were left pending
        until more target facts are known.
        """
        if max_depth is None:
            max_depth = self.max_depth
        merged = self._environment.merged(environment)
        return expand_environment_strings(self._raw, merged, max_depth)

    def resolve(self, environment, max_depth=None):
        return self.with_environment(environment, max_depth)

    def complete(self, environment, max_depth=None):
        return self.with_environment(environment, max_depth)

    def source_map(self):
        """
        Group provenance records by their effective replacement text.

        Values in the returned lists are records, not merged strings, so equal
        replacements from different variables remain separate entries.
        """
        mapping = OrderedDict()
        for part in self.provenance:
            mapping.setdefault(part.text, []).append(part)
        return mapping

    def sources_for(self, text):
        """
        Return every provenance record whose replacement text equals ``text``.
        """
        return [part for part in self.provenance if part.text == text]

    def trace_map(self):
        """
        Like ``source_map`` but includes nested resolution records.
        """
        mapping = OrderedDict()
        for part in self.trace:
            mapping.setdefault(part.text, []).append(part)
        return mapping

    def explain(self):
        """
        Return a serializable explanation of the raw and effective views.
        """
        return self.as_dict()

    def as_dict(self):
        provenance = [part.as_dict() for part in self.provenance]
        return {
            "raw": self.raw,
            "raw_value": self.raw,
            "original": self.raw,
            "effective": self.effective,
            "expanded": self.effective,
            "effective_view": self.effective,
            "value": self.effective,
            "status": self.status,
            "is_complete": self.is_complete,
            "unresolved": list(self.unresolved),
            "unresolved_variables": list(self.unresolved),
            "cycles": list(self.cycles),
            "depth_limited": list(self.depth_limited),
            "pending": list(self.pending),
            "environment": self.environment,
            "provenance": provenance,
            "origins": provenance,
            "sources": provenance,
            "trace": [part.as_dict() for part in self.trace],
            "segments": [part.as_dict() for part in self.parts],
        }

    def to_dict(self):
        return self.as_dict()

    def get(self, key, default=None):
        return self.as_dict().get(key, default)

    def keys(self):
        return list(self.as_dict().keys())

    def items(self):
        return list(self.as_dict().items())

    def to_mapping(self):
        return self.as_dict()

    def __getitem__(self, key):
        if isinstance(key, string_types):
            values = self.as_dict()
            if key in values:
                return values[key]
        return text_type(self)[key]

    def __repr__(self):
        return (
            'EnvironmentExpansionResult(raw="{0}", effective="{1}", '
            'status="{2}", unresolved={3}, cycles={4})'
        ).format(self._raw, text_type(self), self._status,
                 self.unresolved, self.cycles)


class TargetEnvironment(object):
    """
    A case-insensitive environment supplied by the caller.

    Variable names preserve the spelling used in the mapping while lookup is
    case-insensitive, matching Windows behavior. Values are kept raw; nested
    references are resolved by the expansion engine.
    """

    def __init__(self, values=None):
        self._entries = OrderedDict()
        if values is not None:
            self._load(values)

    def _load(self, values):
        if isinstance(values, TargetEnvironment):
            for name, value in values.items():
                self.add(name, value)
            return

        if isinstance(values, string_types):
            name, separator, value = values.partition("=")
            if not separator:
                raise ValueError(
                    "environment entry must contain '=': %s" % values)
            self.add(name, value)
            return

        if isinstance(values, Mapping):
            items = values.items()
        elif hasattr(values, "values") and hasattr(values, "path"):
            for name, value in environment_from_key(values).items():
                self.add(name, value)
            return
        else:
            items = values

        for item in items:
            if isinstance(item, string_types):
                name, separator, value = item.partition("=")
                if not separator:
                    raise ValueError(
                        "environment entry must contain '=': %s" % item)
            else:
                try:
                    name, value = item
                except (TypeError, ValueError):
                    raise ValueError(
                        "environment entry must be a pair or NAME=value")
            self.add(name, value)

    def add(self, name, value):
        name = _to_text(name)
        if name is None or name == "":
            raise ValueError("environment variable name must not be empty")
        if value is not None:
            value = _to_text(value)
        self._entries[name.upper()] = (name, value)

    def get(self, name, default=None):
        entry = self._entries.get(_to_text(name).upper())
        if entry is None:
            return default
        return entry[1]

    def get_entry(self, name):
        return self._entries.get(_to_text(name).upper())

    def is_known(self, name):
        """
        True if the name is present with a decided (non-None) value.
        """
        entry = self.get_entry(name)
        return entry is not None and entry[1] is not None

    def is_pending(self, name):
        entry = self.get_entry(name)
        return entry is None or entry[1] is None

    def __getitem__(self, name):
        entry = self.get_entry(name)
        if entry is None:
            raise KeyError(name)
        return entry[1]

    def __contains__(self, name):
        return _to_text(name).upper() in self._entries

    def __len__(self):
        return len(self._entries)

    def __iter__(self):
        for original_name, value in self._entries.values():
            yield original_name

    def keys(self):
        return [original_name for original_name, value
                in self._entries.values()]

    def values(self):
        return [value for original_name, value in self._entries.values()]

    def items(self):
        return [(original_name, value) for original_name, value
                in self._entries.values()]

    def as_dict(self):
        return OrderedDict(self.items())

    def update(self, values):
        other = normalize_environment(values)
        for name, value in other.items():
            self.add(name, value)
        return self

    def merged(self, values):
        result = TargetEnvironment(self)
        result.update(values)
        return result

    def copy(self):
        return TargetEnvironment(self)

    def __repr__(self):
        return "TargetEnvironment({0})".format(dict(self.as_dict()))


def normalize_environment(environment):
    if isinstance(environment, TargetEnvironment):
        return environment
    if environment is None:
        return TargetEnvironment()
    return TargetEnvironment(environment)


def environment_from_key(key):
    """
    Build a target environment from a RegistryKey such as
    ``...\\Session Manager\\Environment``.

    Values are kept raw: nested ``%VAR%`` references are resolved by the
    expansion engine, with the same cycle and provenance handling as any other
    target environment.
    """
    values = OrderedDict()
    for value in key.values():
        name = value.name()
        if name == "(default)":
            continue
        data = value.value()
        if not isinstance(data, string_types):
            continue
        values[name] = data
    return values


def _iter_tokens(text):
    index = 0
    length = len(text)
    while index < length:
        start = text.find("%", index)
        if start < 0:
            yield "literal", text[index:], index
            break
        if start > index:
            yield "literal", text[index:start], index
        end = text.find("%", start + 1)
        if end < 0:
            yield "literal", text[start:], start
            break
        name = text[start + 1:end]
        if name == "":
            yield "literal", text[start:end + 1], start
        else:
            yield "variable", (name, text[start:end + 1]), start
        index = end + 1


def _expand_text(text, environment, active, chain, depth, max_depth):
    output = []
    parts = []
    position = 0
    status = STATUS_RESOLVED

    for kind, payload, _source_start in _iter_tokens(text):
        if kind == "literal":
            literal = payload
            parts.append(ExpansionPart(
                "literal", literal, position, position + len(literal),
                source=literal, depth=depth))
            output.append(literal)
            position += len(literal)
            continue

        name, placeholder = payload
        key = name.upper()
        part_status = STATUS_RESOLVED

        if key in active:
            replacement = placeholder
            parts.append(ExpansionPart(
                "variable", replacement, position, position + len(replacement),
                source=placeholder, name=name, chain=chain + (name,),
                status=STATUS_CYCLE, depth=depth))
            part_status = STATUS_CYCLE
        elif not environment.is_known(name):
            replacement = placeholder
            parts.append(ExpansionPart(
                "variable", replacement, position, position + len(replacement),
                source=placeholder, name=name, chain=chain + (name,),
                status=STATUS_UNRESOLVED, depth=depth))
            part_status = STATUS_UNRESOLVED
        elif depth > max_depth:
            replacement = placeholder
            parts.append(ExpansionPart(
                "variable", replacement, position, position + len(replacement),
                source=placeholder, name=name, chain=chain + (name,),
                status=STATUS_DEPTH_LIMIT, depth=depth))
            part_status = STATUS_DEPTH_LIMIT
        else:
            variable_value = environment.get(name)
            child_text, child_parts, child_status = _expand_text(
                variable_value, environment, active | frozenset([key]),
                chain + (name,), depth + 1, max_depth)
            replacement = child_text
            for child_part in child_parts:
                child_part.shift(position)
            parts.extend(child_parts)
            if child_status == STATUS_RESOLVED:
                part_status = STATUS_RESOLVED
            else:
                part_status = STATUS_PARTIAL
            parts.append(ExpansionPart(
                "variable", replacement, position, position + len(replacement),
                source=placeholder, name=name, chain=chain + (name,),
                status=part_status, depth=depth))
            status = _merge_status(status, child_status)

        output.append(replacement)
        position += len(replacement)
        status = _merge_status(status, part_status)

    return "".join(output), parts, status


def expand_environment_strings(text, environment=None, max_depth=32):
    """
    Expand ``%NAME%`` references using only the supplied target environment.

    The return value is a string containing the effective text; it also
    exposes ``raw``, ``unresolved``, ``cycles``, ``provenance`` and
    ``segments``.
    """
    if max_depth is None:
        max_depth = 32
    if max_depth < 0:
        raise ValueError("max_depth must not be negative")

    target_environment = normalize_environment(environment)
    raw = _to_text(text)
    effective, parts, status = _expand_text(
        raw, target_environment, frozenset(), (), 0, max_depth)
    return EnvironmentExpansionResult(
        raw, effective, parts, target_environment, max_depth, status)


def expand_environment_string(text, environment=None, max_depth=32):
    return expand_environment_strings(text, environment, max_depth)


def expand_environment(text, environment=None, max_depth=32):
    return expand_environment_strings(text, environment, max_depth)


def expand(value, environment=None, max_depth=32):
    """
    Expand text or a RegistryValue-compatible object.
    """
    if isinstance(value, string_types):
        return expand_environment_strings(value, environment, max_depth)
    return expand_value(value, environment, max_depth)


def expand_string(text, environment=None, max_depth=32):
    return expand_environment_strings(text, environment, max_depth)


def expand_value(value, environment=None, max_depth=32):
    """
    Expand a RegistryValue-compatible object without changing its raw data.

    Only REG_EXPAND_SZ values are expanded. Other value types keep their
    stored text verbatim, matching Windows behavior.
    """
    if isinstance(value, string_types):
        raw = value
    elif hasattr(value, "raw_value") and callable(value.raw_value):
        raw = value.raw_value()
    elif hasattr(value, "value") and callable(value.value):
        raw = value.value()
    else:
        raw = _to_text(value)

    target_environment = normalize_environment(environment)
    if max_depth is None:
        max_depth = 32
    if max_depth < 0:
        raise ValueError("max_depth must not be negative")

    if hasattr(value, "value_type") and callable(value.value_type):
        if value.value_type() != REG_EXPAND_SZ:
            if not isinstance(raw, string_types):
                return raw
            raw = _to_text(raw)
            return EnvironmentExpansionResult(
                raw, raw, (), target_environment, max_depth,
                STATUS_NOT_EXPANDABLE)

    return expand_environment_strings(raw, target_environment, max_depth)


Environment = TargetEnvironment
Expansion = EnvironmentExpansionResult
ExpansionResult = EnvironmentExpansionResult


# Common spellings for callers that mirror Windows terminology.

def expand_environment_variables(text, environment=None, max_depth=32):
    return expand_environment_strings(text, environment, max_depth)


def expand_env(text, environment=None, max_depth=32):
    return expand_environment_strings(text, environment, max_depth)


def expand_registry_value(value, environment=None, max_depth=32):
    return expand_value(value, environment, max_depth)


def expand_reg_expand_sz(text, environment=None, max_depth=32):
    return expand_environment_strings(text, environment, max_depth)
