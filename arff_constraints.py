# -*- coding: utf-8 -*-
"""Cross-attribute constraint checking for ARFF observations.

ARFF treats every value on its own: a nominal value is valid if it is
declared, a real value is valid if it parses.  Device configurations usually
have additional rules that span attributes.  This module checks those rules
and, for each observation, decides whether the missing fields can be filled
in without violating them.

Rules
-----
``rules`` is a mapping with any of ``allowed``, ``implies`` and ``excludes``
(aliases are listed below).  It can also be a list of rule mappings with an
explicit ``type``.

``allowed``
    A disjunction of permitted combinations.  At least one must match every
    observation.  Accepted forms::

        {"allowed": [
            {"power": "on", "mode": "fast"},
            {"power": "off", "mode": "idle"},
        ]}

        {"allowed": {
            "attributes": ["power", "mode"],
            "combinations": [
                ["on", "fast"],
                ["off", "idle"],
            ],
        }}

    With the explicit ``attributes`` form, every combination gives all
    listed attributes.  With a list, each item is one alternative; a single
    alternative may list several attributes as a mapping, as a flat
    ``(name, value, name, value)`` tuple, or as a
    ``name=value and name=value`` string.  To keep one combination together,
    put it in one tuple or mapping, or use the group form.

``implies``
    ``{"if": CONDITION, "then": CONDITION}``, or a pair of conditions, or a
    string such as ``"power=on -> cooling=active"``.  The consequent is
    required only when the whole antecedent holds.

``excludes``
    A list of forbidden combinations.  Each item is one combination (all of
    its conditions may not hold together), in one of these forms::

        {"power": "on", "bypass": "enabled"}       # one combination
        ("power", "on", "bypass", "enabled")        # same, flat pairs
        [("power", "on"), ("bypass", "enabled")]    # same, pairs
        "power=on and bypass=enabled"               # same, string
        {"not": [{"power": "on"}, {"bypass": "enabled"}]}
        {"if": {"power": "on"}, "excludes": {"bypass": "enabled"}}

    Separate list items are separate exclusions.

A condition is a mapping of attribute names to values, a flat
``(name, value, name, value, ...)`` sequence, or a string using
``name=value`` joined by ``and``.  All rule values must be members of the
declared domain.

Missing values
--------------
``None``, ``''`` and NaN are missing.  Missing fields are variables: they may
take any value in their declared domain.  They are never automatically
reported as errors.  If several completions exist, the result keeps the
whole range of possible values for every originally missing field instead of
choosing one.  Known values are never replaced in the result.

Conflicts
---------
When no completion exists, the result contains an inclusion-minimal set of
observed field/value pairs that cannot hold together.  Removing any one of
those fields admits a completion.  If the rules themselves are inconsistent,
the conflict has an empty ``attributes`` list and ``rule_conflict`` is true.

For numeric and string domains the set of possible values can be infinite.
``possible_values`` lists the concrete values that are possible and
``any_value`` is true when other values are possible as well.  Example
completions (``include_completions=True``) use ``ANY_VALUE`` (``'*'``) to
stand for one value outside the finite list.

Missing values in sparse ARFF
-----------------------------
Sparse observations may omit attributes.  Omitted fields are treated as
missing, just like ``?``, because a sparse line does not assert the nominal
first value or zero that a dense decoder would infer.
"""

import argparse
import itertools
import json
import math
import sys
from collections.abc import Mapping

import arff


__all__ = [
    'ANY_VALUE',
    'ArffConstraintError',
    'BadObservation',
    'BadRule',
    'ConstraintChecker',
    'ConstraintSet',
    'check_constraints',
    'main',
]


class ArffConstraintError(ValueError):
    """Base class for invalid constraint input."""


class BadRule(ArffConstraintError):
    """A constraint rule is malformed or uses an undeclared value."""


class BadObservation(ArffConstraintError):
    """The ARFF object or one of its observations is malformed."""


class _AnyValue(object):
    """Representative for any value not named by a rule constant."""

    def __init__(self, attribute):
        self.attribute = attribute

    def __repr__(self):
        return '<any %s>' % self.attribute


ANY_VALUE = '*'


# Section names ---------------------------------------------------------------
_ALLOWED_KEYS = frozenset([
    'allowed', 'allowed_combinations', 'allowed_combination', 'combinations',
])
_IMPLIES_KEYS = frozenset([
    'implies', 'implications', 'implication', 'requires',
])
_EXCLUDES_KEYS = frozenset([
    'excludes', 'exclusions', 'exclusion', 'mutex', 'mutual_exclusions',
    'mutually_exclusive', 'forbidden_combinations',
])
_SECTION_KEYS = _ALLOWED_KEYS | _IMPLIES_KEYS | _EXCLUDES_KEYS

_RULE_KEYS = frozenset([
    'type', 'rule', 'if', 'when', 'antecedent', 'condition', 'left',
    'then', 'require', 'required', 'consequent', 'right', 'excludes',
    'exclude', 'not_with', 'conditions', 'not', 'forbidden',
])

# Descriptive keys that never affect the logic are ignored in rule mappings.
_METADATA_KEYS = frozenset(['id', 'name', 'description', 'comment', 'why'])

_ANTECEDENT_KEYS = ('if', 'when', 'antecedent', 'condition', 'left')
_CONSEQUENT_KEYS = ('then', 'require', 'required', 'consequent', 'right')


def _is_missing(value):
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if value == '':
        return True
    return False


def _sort_key(value):
    if isinstance(value, _AnyValue):
        return (1, 0, '')
    return (0, type(value).__name__, value)


def _public(value):
    return ANY_VALUE if isinstance(value, _AnyValue) else value


class _Domain(object):
    """The declared values of one attribute."""

    def __init__(self, name, declaration):
        self.name = name
        self.constants = set()
        self.mentioned = False
        if isinstance(declaration, (list, tuple)):
            self.kind = 'NOMINAL'
            self.values = []
            seen = set()
            for value in declaration:
                if _is_missing(value):
                    raise BadObservation(
                        'Nominal attribute %r has a missing value in its '
                        'declaration' % name)
                atom = str(value)
                if atom not in seen:
                    seen.add(atom)
                    self.values.append(atom)
            if not self.values:
                raise BadObservation(
                    'Nominal attribute %r has no declared values' % name)
            self.value_set = frozenset(self.values)
        elif isinstance(declaration, str):
            self.kind = declaration.upper()
            if self.kind not in ('NUMERIC', 'REAL', 'INTEGER', 'STRING'):
                raise BadObservation(
                    'Unsupported attribute type %r for %r'
                    % (declaration, name))
            self.values = []
            self.value_set = frozenset()
        else:
            raise BadObservation(
                'Attribute %r has an unsupported declaration %r'
                % (name, declaration))

    @property
    def nominal(self):
        return self.kind == 'NOMINAL'

    def parse(self, value, what, strict=True):
        """Convert a rule value (strict) or an ARFF value to an atom."""
        if _is_missing(value):
            raise BadRule('%s gives a missing value for attribute %r'
                          % (what, self.name))
        atom = self._convert(value, what)
        if strict and self.nominal and atom not in self.value_set:
            raise BadRule(
                '%s uses value %r, which is not declared for nominal '
                'attribute %r' % (what, atom, self.name))
        return atom

    def _convert(self, value, what):
        try:
            if self.nominal:
                return str(value)
            if self.kind == 'INTEGER':
                return int(float(value))
            if self.kind in ('NUMERIC', 'REAL'):
                return float(value)
            return str(value)
        except (TypeError, ValueError, OverflowError):
            raise BadRule('%s gives invalid value %r for %s attribute %r'
                          % (what, value, self.kind, self.name))

    def parse_input(self, value, what):
        """Return the atom for an observed value, or ``INVALID_ATOM``.

        An observed value outside the declared domain is an individual data
        error, not a rule authoring error.  It is reported as a conflict for
        that observation.
        """
        if _is_missing(value):
            return None
        try:
            atom = self._convert(value, what)
        except BadRule:
            return INVALID_ATOM
        if self.nominal and atom not in self.value_set:
            return INVALID_ATOM
        return atom

    def all_values(self):
        """All values that need to be considered for satisfiability."""
        if self.nominal:
            return list(self.values)
        return sorted(self.constants, key=_sort_key) + [_AnyValue(self.name)]

    def finite_values(self, include_other=True):
        """Concrete possible values, plus an any-value marker if needed."""
        if self.nominal:
            return list(self.values), False
        return sorted(self.constants, key=_sort_key), include_other

    @staticmethod
    def is_other(value):
        return isinstance(value, _AnyValue)


class _RuleSet(object):
    def __init__(self, domains):
        self.domains = domains
        self.by_name = {domain.name: index
                        for index, domain in enumerate(domains)}
        self.allowed = []       # list of disjunctions of {index: atom}
        self.implications = []  # list of ({index: atom}, {index: atom})
        self.exclusions = []    # list of lists of {index: atom}

    def lookup(self, name, what):
        if not isinstance(name, str):
            raise BadRule('%s uses non-string attribute name %r'
                          % (what, name))
        try:
            return self.by_name[name]
        except KeyError:
            raise BadRule('%s refers to unknown attribute %r' % (what, name))

    def condition(self, value, what):
        pairs = _condition_items(value, what)
        result = {}
        for name, value in pairs:
            index = self.lookup(name, what)
            if index in result:
                raise BadRule('%s sets attribute %r twice'
                              % (what, name))
            domain = self.domains[index]
            atom = domain.parse(value, what)
            result[index] = atom
            domain.constants.add(atom)
            domain.mentioned = True
        return result


def _condition_items(value, what):
    """Normalise one condition to a list of (attribute, value) pairs."""
    if isinstance(value, str):
        result = []
        for part in value.split(' and '):
            if '=' not in part:
                raise BadRule('%s condition %r must use name=value'
                              % (what, part.strip()))
            name, atom = part.split('=', 1)
            name = name.strip()
            atom = atom.strip()
            if len(atom) >= 2 and atom[0] == atom[-1] and atom[0] in '\'"':
                atom = atom[1:-1]
            if not name:
                raise BadRule('%s contains an empty attribute name' % what)
            result.append((name, atom))
        return result
    if isinstance(value, Mapping):
        return list(value.items())
    if isinstance(value, (list, tuple)):
        items = list(value)
        if not items:
            return []
        if all(isinstance(item, (list, tuple)) and len(item) == 2
               for item in items):
            return [(item[0], item[1]) for item in items]
        if all(isinstance(item, str) and '=' in item for item in items):
            result = []
            for item in items:
                result.extend(_condition_items(item, what))
            return result
        if len(items) % 2:
            raise BadRule('%s has an odd number of entries' % what)
        return [(items[i], items[i + 1]) for i in range(0, len(items), 2)]
    raise BadRule('%s must be a mapping, list of pairs or string' % what)


def _condition_like(value):
    if isinstance(value, Mapping):
        return True
    if isinstance(value, str):
        return '=' in value or ' and ' in value
    if isinstance(value, (list, tuple)):
        if not value:
            return False
        if all(isinstance(item, (list, tuple)) and len(item) == 2
               for item in value):
            return True
        if len(value) == 2 and not any(isinstance(item, (list, tuple))
                                       for item in value):
            return True
    return False


def _rule_list(value, what):
    if value is None:
        return []
    if isinstance(value, Mapping) or isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    raise BadRule('%s must be a mapping, string or list' % what)


def _even_pairs(value):
    return isinstance(value, (list, tuple)) and len(value) >= 2 and \
        len(value) % 2 == 0 and \
        not any(isinstance(item, (list, tuple, Mapping)) for item in value)


def _group_spec(value):
    return isinstance(value, Mapping) and any(
        key in value for key in ('attributes', 'fields', 'columns'))


class _RuleParser(object):
    def __init__(self, ruleset):
        self.rules = ruleset

    def parse(self, value):
        if value is None:
            return
        if isinstance(value, ConstraintSet):
            value = value.as_mapping()
        if isinstance(value, Mapping):
            if self._single_rule(value):
                self.add_rule(value)
            else:
                self.add_sections(value)
            return
        if isinstance(value, (list, tuple)):
            if not value:
                return
            if all(isinstance(item, Mapping) and self._single_rule(item)
                   for item in value):
                # A list of typed rules is one constraint set.  Plain
                # allowed entries form one disjunction; entries that name
                # their own attribute group are independent constraints.
                allowed = []
                for item in value:
                    kind = _kind(item.get('type', item.get('rule')))
                    if kind != 'allowed':
                        self.add_rule(item)
                        continue
                    body = {key: entry for key, entry in item.items()
                            if _norm(key) not in _METADATA_KEYS and
                            _norm(key) not in ('type', 'rule')}
                    if _group_spec(body):
                        self.add_allowed(body)
                        continue
                    for key in list(body):
                        if _norm(key) in _ALLOWED_KEYS:
                            body = body[key]
                            break
                    allowed = self._collect_allowed(allowed, body)
                if allowed:
                    self.add_allowed(allowed)
                return
            raise BadRule('A rule list must contain mappings with an explicit '
                          'rule type')
        raise BadRule('Rules must be a mapping or a list of mappings')

    def _collect_allowed(self, allowed, body):
        if isinstance(body, (list, tuple)) and body and \
                all(isinstance(entry, Mapping) and _group_spec(entry)
                    for entry in body):
            self.add_allowed(body)
        elif isinstance(body, str):
            allowed.append(body)
        elif isinstance(body, (list, tuple)):
            allowed.extend(body)
        elif isinstance(body, Mapping):
            allowed.append(body)
        else:
            raise BadRule('Allowed rule body has an unsupported type')
        return allowed

    @staticmethod
    def _single_rule(value):
        kind = value.get('type', value.get('rule'))
        if isinstance(kind, str) and _kind(kind) is not None:
            return True
        keys = set(value)
        if keys & _SECTION_KEYS:
            return False
        return bool(keys & _RULE_KEYS)

    def add_sections(self, value):
        for key, section in value.items():
            norm = _norm(key)
            if norm in _METADATA_KEYS:
                continue
            if norm in _ALLOWED_KEYS:
                self.add_allowed(section)
            elif norm in _IMPLIES_KEYS:
                self.add_implies(section)
            elif norm in _EXCLUDES_KEYS:
                self.add_excludes(section)
            else:
                raise BadRule('Unknown constraint section %r' % key)
        # An empty mapping means "no constraints", which is useful when the
        # same code path handles optional rule files.

    def add_rule(self, value):
        kind = value.get('type', value.get('rule'))
        norm = _kind(kind)
        if norm is None and kind is None:
            keys = set(value)
            if keys & set(_ANTECEDENT_KEYS) and keys & set(_CONSEQUENT_KEYS):
                norm = 'implies'
            elif keys & {'not', 'conditions', 'forbidden'}:
                norm = 'excludes'
            elif _group_spec(value) or keys & set(_ALLOWED_KEYS):
                norm = 'allowed'
        if norm is None:
            raise BadRule('Unknown rule type %r' % kind)
        body = {key: item for key, item in value.items()
                if _norm(key) not in _METADATA_KEYS and
                _norm(key) not in ('type', 'rule')}
        section_keys = {'allowed': _ALLOWED_KEYS, 'implies': _IMPLIES_KEYS,
                        'excludes': _EXCLUDES_KEYS}[norm]
        for key in list(body):
            if _norm(key) in section_keys:
                body = body[key]
                break
        if norm == 'allowed':
            self.add_allowed(body)
        elif norm == 'implies':
            self.add_implies(body)
        else:
            self.add_excludes(body)

    # allowed ---------------------------------------------------------------
    def add_allowed(self, value):
        if value is None:
            return
        if isinstance(value, str):
            self.rules.allowed.append([self._alternative(value, None, value)])
            return
        if isinstance(value, Mapping):
            attributes, rows = _group(value)
            self._allowed_group(attributes, rows, value)
            return
        if isinstance(value, (list, tuple)):
            items = list(value)
            if not items:
                self.rules.allowed.append([])
                return
            if all(isinstance(item, Mapping) and _group_spec(item)
                   for item in items):
                for item in items:
                    attributes, rows = _group(item)
                    self._allowed_group(attributes, rows, item)
                return
            # A plain list is a list of alternatives, never one combination.
            # To express one combination, use a mapping, a group with
            # "combinations", or a list of conditions.
            self._allowed_group(None, items, value)
            return
        raise BadRule('allowed must be a mapping or a list')

    def _allowed_group(self, attributes, rows, original):
        if isinstance(rows, Mapping):
            rows = [rows]
        if not isinstance(rows, (list, tuple)):
            raise BadRule('Allowed combinations must be a list')
        alternatives = []
        for number, row in enumerate(rows):
            what = 'allowed combination %d' % number
            alternatives.append(self._alternative(row, attributes, what))
        self.rules.allowed.append(alternatives)

    def _alternative(self, row, attributes, what):
        if attributes is None:
            pairs = _condition_items(row, what)
            result = {}
            for name, value in pairs:
                index = self.rules.lookup(name, what)
                if index in result:
                    raise BadRule('%s sets attribute %r twice'
                                  % (what, name))
                domain = self.rules.domains[index]
                atom = domain.parse(value, what)
                result[index] = atom
                domain.constants.add(atom)
                domain.mentioned = True
            # An empty alternative constrains nothing, so it matches every
            # observation.  This is a legitimate catch-all.
            return result

        if not isinstance(attributes, (list, tuple)):
            raise BadRule('%s has an invalid attribute list' % what)
        attributes = list(attributes)
        if isinstance(row, Mapping):
            extra = set(row) - set(attributes)
            if extra:
                raise BadRule(
                    '%s contains attributes outside the group: %s'
                    % (what, ', '.join(sorted(map(str, extra)))))
            pairs = [(name, row[name]) for name in attributes
                     if name in row]
            missing = [name for name in attributes if name not in row]
            if missing:
                raise BadRule('%s is missing group attributes: %s'
                              % (what, ', '.join(map(str, missing))))
        elif isinstance(row, (list, tuple)):
            if len(row) != len(attributes):
                raise BadRule('%s has %d values, expected %d'
                              % (what, len(row), len(attributes)))
            pairs = list(zip(attributes, row))
        else:
            raise BadRule('%s must be a list or mapping' % what)
        result = {}
        for name, value in pairs:
            index = self.rules.lookup(name, what)
            domain = self.rules.domains[index]
            atom = domain.parse(value, what)
            result[index] = atom
            domain.constants.add(atom)
            domain.mentioned = True
        return result

    # implies ---------------------------------------------------------------
    def add_implies(self, value):
        for number, item in enumerate(_rule_list(value, 'implies')):
            what = 'implication %d' % number
            antecedent, consequent = self._implication(item, what)
            self.rules.implications.append((
                self.rules.condition(antecedent, what),
                self.rules.condition(consequent, what),
            ))

    def _implication(self, item, what):
        if isinstance(item, str):
            for token in ('->', '=>', ' implies '):
                if token in item:
                    left, right = item.split(token, 1)
                    return left.strip(), right.strip()
            raise BadRule('%s must contain ->, => or implies' % what)
        if isinstance(item, Mapping):
            left = next((key for key in _ANTECEDENT_KEYS if key in item),
                        None)
            right = next((key for key in _CONSEQUENT_KEYS if key in item),
                         None)
            if left is None or right is None:
                raise BadRule('%s needs both a condition and a consequence'
                              % what)
            allowed = set(_ANTECEDENT_KEYS + _CONSEQUENT_KEYS)
            extra = set(item) - allowed
            if extra:
                raise BadRule('%s has unknown keys: %s'
                              % (what, ', '.join(sorted(map(str, extra)))))
            return item[left], item[right]
        if isinstance(item, (list, tuple)):
            items = list(item)
            if len(items) == 2 and all(_condition_like(part)
                                       for part in items):
                return items[0], items[1]
            if len(items) == 4:
                return items[:2], items[2:]
        raise BadRule('%s must be a mapping, string or two conditions'
                      % what)

    # excludes --------------------------------------------------------------
    def add_excludes(self, value):
        for number, item in enumerate(_exclusion_rules(value)):
            what = 'exclusion %d' % number
            conditions = self._exclusion(item, what)
            if not conditions:
                raise BadRule('%s has no conditions' % what)
            self.rules.exclusions.append([
                self.rules.condition(condition, what)
                for condition in conditions
            ])

    def _exclusion(self, item, what):
        if isinstance(item, str):
            if ' and ' in item or ' & ' in item:
                token = ' and ' if ' and ' in item else ' & '
                return item.split(token)
            return [item]
        if isinstance(item, Mapping):
            if 'not' in item or 'forbidden' in item:
                key = 'not' if 'not' in item else 'forbidden'
                conditions = item[key]
                if isinstance(conditions, Mapping):
                    return [conditions]
                if isinstance(conditions, (list, tuple)):
                    return list(conditions)
                raise BadRule('%s not must be a condition or list' % what)
            if 'conditions' in item:
                conditions = item['conditions']
                if not isinstance(conditions, (list, tuple)):
                    raise BadRule('%s conditions must be a list' % what)
                return list(conditions)
            left = next((key for key in _ANTECEDENT_KEYS if key in item),
                        None)
            right = next((key for key in ('excludes', 'exclude', 'not_with',
                                          'right') if key in item), None)
            if left is not None and right is not None:
                return [item[left], item[right]]
            if left is not None or right is not None:
                raise BadRule('%s needs both sides of the exclusion' % what)
            return [item]
        if isinstance(item, (list, tuple)):
            items = list(item)
            if not items:
                raise BadRule('%s is empty' % what)
            if all(isinstance(part, Mapping) for part in items):
                return items
            if all(isinstance(part, str) and '=' in part for part in items):
                return items
            if all(isinstance(part, (list, tuple)) and len(part) == 2
                   and not isinstance(part[0], (list, tuple, Mapping))
                   for part in items):
                return items
            if len(items) % 2 == 0 and \
                    not any(isinstance(part, (list, tuple, Mapping))
                            for part in items):
                return [items]
            if len(items) == 2 and all(_condition_like(part)
                                       for part in items):
                return items
            raise BadRule('%s has an unsupported exclusion shape' % what)
        raise BadRule('%s must be a mapping, string or list' % what)


def _exclusion_rules(value):
    """Split the excludes section into individual forbidden combinations."""
    if value is None:
        return []
    if isinstance(value, Mapping) or isinstance(value, str):
        return [value]
    if not isinstance(value, (list, tuple)):
        raise BadRule('excludes must be a mapping, string or list')
    items = list(value)
    if not items:
        return []
    if all(isinstance(item, Mapping) for item in items):
        return items
    if _even_pairs(value):
        # ("power", "on", "bypass", "enabled") or
        # ["a=1", "b=2"] is one combination.
        return [value]
    if all(isinstance(item, str) and '=' in item for item in items):
        return [value]
    if all(isinstance(item, (list, tuple)) and len(item) == 2
           and not isinstance(item[0], (list, tuple, Mapping))
           for item in items):
        # [("power", "on"), ("bypass", "enabled")] is one combination.
        return [items]
    if all(isinstance(item, (list, tuple)) for item in items):
        return items
    raise BadRule('excludes has an unsupported list shape')


def _group(value):
    attributes = None
    for key in ('attributes', 'fields', 'columns'):
        if key in value:
            attributes = value[key]
            break
    rows = None
    for key in ('combinations', 'rows', 'tuples', 'values', 'allowed'):
        if key in value:
            rows = value[key]
            break
    if rows is None:
        if attributes is not None:
            raise BadRule('An allowed group with attributes needs '
                          'combinations')
        # A plain attribute/value mapping is one alternative.
        return None, [{key: item for key, item in value.items()
                       if _norm(key) not in _METADATA_KEYS}]
    return attributes, rows


def _norm(value):
    if not isinstance(value, str):
        return ''
    return value.lower().replace('-', '_').replace(' ', '_')


def _kind(value):
    if not isinstance(value, str):
        return None
    value = _norm(value)
    if value in ('allowed', 'allowed_combination', 'allowed_combinations',
                 'combination', 'combinations'):
        return 'allowed'
    if value in ('implies', 'implication', 'implications'):
        return 'implies'
    if value in ('excludes', 'exclusion', 'exclusions', 'mutex',
                 'mutual_exclusion', 'mutual_exclusions', 'not_both'):
        return 'excludes'
    return None


class ConstraintSet(object):
    """Programmatic rule container accepted anywhere ``rules`` is accepted."""

    def __init__(self, allowed=None, implies=None, excludes=None):
        self.allowed = allowed
        self.implies = implies
        self.excludes = excludes

    def as_mapping(self):
        result = {}
        if self.allowed is not None:
            result['allowed'] = self.allowed
        if self.implies is not None:
            result['implies'] = self.implies
        if self.excludes is not None:
            result['excludes'] = self.excludes
        return result


_INVALID = object()
INVALID_ATOM = _INVALID


class _Solver(object):
    """Depth-first search over the values possible for each field."""

    def __init__(self, domains, ruleset):
        self.domains = domains
        self.rules = ruleset

    def domains_for(self, fixed):
        result = []
        for index, domain in enumerate(self.domains):
            if index not in fixed:
                result.append(tuple(domain.all_values()))
            elif fixed[index] is INVALID_ATOM:
                result.append(())
            else:
                result.append((fixed[index],))
        return result

    def solve(self, fixed, extra=None, limit=None, collect=None):
        """Return True if a completion exists.

        ``fixed`` maps field indices to observed atoms.  ``extra`` pins one
        additional field, used when enumerating possible values.  If
        ``collect`` is a list, up to ``limit`` completions are appended.
        """
        domains = self.domains_for(fixed)
        assignment = [None] * len(domains)
        for index, value in fixed.items():
            if value is INVALID_ATOM:
                return False
            assignment[index] = value
        if extra:
            for index, value in extra.items():
                assignment[index] = value
                domains[index] = (value,)
        for index, value in enumerate(assignment):
            if value is None and not domains[index]:
                return False

        pending = [index for index, value in enumerate(assignment)
                   if value is None]
        found = [False]

        def search(remaining):
            if collect is not None and limit is not None and \
                    len(collect) >= limit:
                return True
            if found[0] and collect is None:
                return True
            if not remaining:
                if not self._consistent(assignment, domains):
                    return False
                found[0] = True
                if collect is not None:
                    collect.append({
                        self.domains[index].name: _public(assignment[index])
                        for index in range(len(assignment))
                        if index not in fixed
                    })
                return collect is None
            index = min(remaining, key=lambda i: (
                len(domains[i]), i))
            remaining = [i for i in remaining if i != index]
            for value in domains[index]:
                assignment[index] = value
                if self._consistent(assignment, domains) and search(remaining):
                    if collect is None:
                        assignment[index] = None
                        return True
            assignment[index] = None
            return False

        search(pending)
        return found[0]

    def _consistent(self, assignment, domains):
        for alternatives in self.rules.allowed:
            if not any(self._compatible(alternative, assignment, domains)
                       for alternative in alternatives):
                return False
        for antecedent, consequent in self.rules.implications:
            if self._holds(antecedent, assignment) and \
                    not self._compatible(consequent, assignment, domains):
                return False
        for conditions in self.rules.exclusions:
            if all(self._holds(condition, assignment)
                   for condition in conditions):
                return False
        return True

    @staticmethod
    def _holds(condition, assignment):
        for index, atom in condition.items():
            if assignment[index] is None or assignment[index] != atom:
                return False
        return True

    @staticmethod
    def _compatible(condition, assignment, domains):
        for index, atom in condition.items():
            value = assignment[index]
            if value is not None:
                if value != atom:
                    return False
            elif atom not in domains[index]:
                return False
        return True


def _looks_sparse(text):
    """True when the data section of ARFF text uses sparse rows."""
    in_data = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('%'):
            continue
        upper = stripped.upper()
        if not in_data:
            if upper.startswith('@DATA'):
                in_data = True
            continue
        if stripped.startswith('{'):
            return True
    return False


def _decode(source):
    if isinstance(source, Mapping):
        return source
    if hasattr(source, 'read'):
        text = source.read()
        if isinstance(text, bytes):
            text = text.decode('utf-8')
    elif isinstance(source, str):
        text = source
    else:
        raise BadObservation('ARFF input must be a mapping, text or a '
                             'file-like object')
    try:
        if _looks_sparse(text):
            # Decode sparse rows without inventing the implicit nominal
            # first value or numeric zero.  Omitted fields stay absent and
            # are later treated as missing.
            return arff.loads(text, return_type=arff.LOD)
        return arff.loads(text)
    except arff.ArffException as exc:
        raise BadObservation('Invalid ARFF input: %s' % exc)


def _prepare(source):
    obj = _decode(source)
    attributes = obj.get('attributes')
    if not isinstance(attributes, (list, tuple)) or not attributes:
        raise BadObservation('The ARFF object has no attributes')
    domains = []
    names = {}
    for declared in attributes:
        if not isinstance(declared, (list, tuple)) or len(declared) != 2:
            raise BadObservation('Invalid attribute declaration %r'
                                 % (declared,))
        name, declaration = declared
        if not isinstance(name, str):
            raise BadObservation('Attribute name %r is not a string'
                                 % (name,))
        if name in names:
            raise BadObservation('Duplicate attribute name %r' % name)
        names[name] = len(domains)
        domains.append(_Domain(name, declaration))
    data = obj.get('data', [])
    if not isinstance(data, (list, tuple)):
        if isinstance(data, Mapping) or isinstance(data, (str, bytes)):
            raise BadObservation('ARFF data must be a list of observations')
        try:
            data = list(data)
        except TypeError:
            raise BadObservation('ARFF data must be a list of observations')
    rows = []
    for number, row in enumerate(data):
        if isinstance(row, Mapping):
            values = [None] * len(domains)
            extra = []
            for key, value in row.items():
                if isinstance(key, int) and not isinstance(key, bool) and \
                        0 <= key < len(domains):
                    values[key] = value
                elif isinstance(key, str) and key in names:
                    values[names[key]] = value
                else:
                    extra.append(key)
            if extra:
                raise BadObservation(
                    'Observation %d has unknown attributes: %s'
                    % (number, ', '.join(sorted(map(str, extra)))))
            rows.append(values)
        elif isinstance(row, (list, tuple)):
            if len(row) != len(domains):
                raise BadObservation(
                    'Observation %d has %d values, expected %d'
                    % (number, len(row), len(domains)))
            rows.append(list(row))
        else:
            raise BadObservation('Observation %d is not a list or mapping'
                                 % number)
    return obj, domains, rows


def _fixed_values(values, domains, number):
    fixed = {}
    missing = []
    invalid = []
    for index, (domain, value) in enumerate(zip(domains, values)):
        atom = domain.parse_input(value, 'observation %d' % number)
        if atom is None:
            missing.append(index)
        elif atom is INVALID_ATOM:
            fixed[index] = INVALID_ATOM
            invalid.append(index)
        else:
            fixed[index] = atom
    return fixed, missing, invalid


def _possible_values(solver, fixed, missing, domains, ruleset):
    possible = {}
    any_value = {}
    for index in missing:
        domain = domains[index]
        if not domain.mentioned:
            values, other = domain.finite_values(include_other=not
                                                 domain.nominal)
            possible[domain.name] = values
            any_value[domain.name] = other
            continue
        values = []
        other = False
        for candidate in domain.all_values():
            if solver.solve(fixed, extra={index: candidate}):
                if domain.is_other(candidate):
                    other = True
                else:
                    values.append(_public(candidate))
        possible[domain.name] = values
        any_value[domain.name] = other
    return possible, any_value


def _minimal_conflict(solver, fixed, domains, raw):
    """Return an inclusion-minimal set of observed fields with no model.

    The check is made on the candidate set alone: fields outside the
    candidate set are free (equivalently, treated as missing).  That is what
    makes the result reviewable -- the reported fields by themselves have no
    legal completion, and removing any one of them restores one.

    A bounded search also looks for a smaller-cardinality core.  When it
    succeeds, ``minimality`` is ``'minimum'``; otherwise the greedy core is
    still inclusion-minimal and ``minimality`` is ``'inclusion'``.
    """
    core = dict(fixed)
    changed = True
    while changed:
        changed = False
        for index in sorted(core, key=lambda i: domains[i].name):
            trial = dict(core)
            del trial[index]
            if not solver.solve(trial):
                core = trial
                changed = True
    if not core:
        # The rules reject every assignment even with no observed fields.
        return {
            'attributes': [],
            'values': {},
            'rule_conflict': True,
            'minimality': 'minimum',
            'reason': ('the constraint rules have no satisfying assignment '
                       'even when every field is treated as missing'),
        }

    smaller = _smaller_core(core, fixed, solver, domains)
    if smaller[0] == 'found':
        core = {index: fixed[index] for index in smaller[1]}
        minimality = 'minimum'
    elif smaller[0] == 'none':
        minimality = 'minimum'
    elif len(core) == 1:
        minimality = 'minimum'
    else:
        minimality = 'inclusion'

    order = sorted(core, key=lambda i: i)
    names = [domains[index].name for index in order]
    values = {}
    for index in order:
        if not _is_missing(raw[index]):
            values[domains[index].name] = raw[index]
    return {
        'attributes': names,
        'values': values,
        'rule_conflict': False,
        'minimality': minimality,
        'violated': _violated_rules(core, domains, solver.rules),
        'reason': ('these observed fields have no legal completion on their '
                   'own; removing any one of them admits one'),
    }


_CORE_SEARCH_BUDGET = 20000


def _smaller_core(core, fixed, solver, domains):
    """Search for a core with fewer fields than ``core``.

    Returns ``('found', fields)``, ``('none', None)`` when no smaller core
    exists, or ``('exhausted', None)`` when the search budget ran out.
    """
    indices = sorted(fixed, key=lambda i: domains[i].name)
    if len(core) <= 1:
        return ('none', None)
    budget = [_CORE_SEARCH_BUDGET]

    def limited_solve(subset):
        if budget[0] <= 0:
            raise _BudgetExhausted
        budget[0] -= 1
        return solver.solve(subset)

    try:
        for size in range(1, len(core)):
            for combination in itertools.combinations(indices, size):
                trial = {index: fixed[index] for index in combination}
                if not limited_solve(trial):
                    return ('found', set(combination))
    except _BudgetExhausted:
        return ('exhausted', None)
    return ('none', None)


class _BudgetExhausted(Exception):
    """Internal: the bounded core search ran out of solver calls."""


def _condition_names(condition, domains):
    return {domains[index].name: atom for index, atom in condition.items()}


def _violated_rules(core, domains, ruleset):
    """Describe rules directly contradicted by the core fields.

    This is an explanatory aid, not the definition of unsatisfiability.
    Rules whose violation depends on other fields completed in a particular
    way may not appear here; the authoritative answer is that the core has
    no completion.
    """
    violated = []
    for alternatives in ruleset.allowed:
        if all(_condition_conflicts(alternative, core)
               for alternative in alternatives):
            names = set()
            for alternative in alternatives:
                names.update(domains[index].name for index in alternative)
            violated.append({
                'kind': 'allowed',
                'attributes': sorted(names),
                'allowed': [
                    _condition_names(alternative, domains)
                    for alternative in alternatives
                ],
            })
    for antecedent, consequent in ruleset.implications:
        if _condition_matches(antecedent, core) and \
                _condition_conflicts(consequent, core):
            violated.append({
                'kind': 'implies',
                'attributes': sorted(
                    {domains[index].name for index in antecedent} |
                    {domains[index].name for index in consequent}),
                'if': _condition_names(antecedent, domains),
                'then': _condition_names(consequent, domains),
            })
    for conditions in ruleset.exclusions:
        if all(_condition_matches(condition, core)
               for condition in conditions):
            names = set()
            for condition in conditions:
                names.update(domains[index].name for index in condition)
            violated.append({
                'kind': 'excludes',
                'attributes': sorted(names),
                'combination': [
                    _condition_names(condition, domains)
                    for condition in conditions
                ],
            })
    return violated


def _condition_matches(condition, fixed):
    return all(index in fixed and fixed[index] == atom
               for index, atom in condition.items())


def _condition_conflicts(condition, fixed):
    return any(index in fixed and fixed[index] != atom
               for index, atom in condition.items())


def _check_row(number, values, domains, ruleset):
    fixed, missing, invalid = _fixed_values(values, domains, number)
    solver = _Solver(domains, ruleset)
    result = {
        'index': number,
        'values': list(values),
        'missing': [domains[index].name for index in missing],
        'complete': not missing,
    }
    if not solver.solve(fixed):
        result['has_completion'] = False
        result['valid'] = False
        result['status'] = 'conflict'
        conflict = _minimal_conflict(solver, fixed, domains, values)
        conflict['invalid'] = [domains[index].name for index in invalid]
        if invalid:
            conflict['reason'] = (
                'these fields contain a value outside the declared domain; '
                'removing the offending field admits a legal completion')
        result['conflict'] = conflict
        result['conflict_fields'] = conflict['attributes']
        result['possible_values'] = {}
        result['any_value'] = {}
        return result

    result['has_completion'] = True
    result['valid'] = True
    result['status'] = 'complete' if not missing else 'completable'
    possible, any_value = _possible_values(
        solver, fixed, missing, domains, ruleset)
    result['possible_values'] = possible
    result['any_value'] = any_value
    return result


def check_constraints(source, rules, include_completions=False,
                      max_completions=100):
    """Check all observations of an ARFF source against constraints.

    :param source: a decoded ARFF mapping, ARFF text or file-like object.
    :param rules: allowed combinations, implications and exclusions.  See the
        module documentation for the accepted forms.
    :param include_completions: if true, include up to ``max_completions``
        example completions for each satisfiable observation that has missing
        fields.
    :param max_completions: limit for the example completion lists.
    :return: a mapping with ``rows`` (one result per observation),
        ``satisfiable`` and ``unsatisfiable_rows``.
    """
    obj, domains, rows = _prepare(source)
    ruleset = _RuleSet(domains)
    _RuleParser(ruleset).parse(rules)
    results = []
    for number, values in enumerate(rows):
        result = _check_row(number, values, domains, ruleset)
        if include_completions:
            if result['has_completion'] and result['missing']:
                fixed, _, _ = _fixed_values(values, domains, number)
                completions = []
                # Ask for one extra completion so the truncated flag is
                # exact rather than guessed from the list length.
                _Solver(domains, ruleset).solve(
                    fixed, limit=max_completions + 1, collect=completions)
                truncated = len(completions) > max_completions
                result['completions'] = completions[:max_completions]
                result['completions_truncated'] = truncated
            else:
                result['completions'] = []
                result['completions_truncated'] = False
        results.append(result)
    satisfiable = all(row['has_completion'] for row in results)
    return {
        'relation': obj.get('relation', ''),
        'attributes': [
            (domain.name, obj['attributes'][index][1])
            for index, domain in enumerate(domains)
        ],
        'rows': results,
        'satisfiable': satisfiable,
        'unsatisfiable_rows': [row['index'] for row in results
                               if not row['has_completion']],
    }


class ConstraintChecker(object):
    """Reusable checker for one rule set.

    >>> checker = ConstraintChecker(rules)
    >>> report = checker.check('devices.arff')
    """

    def __init__(self, rules):
        self.rules = rules

    def check(self, source, include_completions=False, max_completions=100):
        return check_constraints(
            source, self.rules, include_completions=include_completions,
            max_completions=max_completions)


def _load_json(path):
    with open(path, 'r') as handle:
        return json.load(handle)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='python -m arff_constraints',
        description='Check ARFF observations against cross-attribute rules.')
    parser.add_argument('arff_file', help='ARFF file to check')
    parser.add_argument('rules_file', help='JSON file with constraint rules')
    parser.add_argument('--completions', action='store_true',
                        help='include example completions')
    parser.add_argument('--max-completions', type=int, default=100,
                        help='limit the example completions per observation')
    args = parser.parse_args(argv)
    rules = _load_json(args.rules_file)
    with open(args.arff_file, 'rb') as handle:
        report = check_constraints(
            handle, rules, include_completions=args.completions,
            max_completions=args.max_completions)
    json.dump(report, sys.stdout, indent=2, sort_keys=True,
              default=str)
    sys.stdout.write('\n')
    return 0 if report['satisfiable'] else 1


if __name__ == '__main__':
    sys.exit(main())
