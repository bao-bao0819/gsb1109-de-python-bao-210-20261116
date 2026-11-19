#!/bin/python

#    This file is part of python-registry.
#
#   Copyright 2011, 2012 Willi Ballenthin <william.ballenthin@mandiant.com>
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
Cross-snapshot registry rename mapping.

The planner works on one registry level (the sibling subkeys of a key, or the
values of one key).  It compares names the way a lookup does: the stored
format of each name decides how it is upcased, and a query with no stored
format is considered in both formats.  Ordinary ``str.lower`` is never used,
because it is not the inverse of the upcase Windows applies and it silently
merges names that a query can still tell apart.

A plan reports four things:

* the accepted renames, including their original bytes and formats;
* the conflict groups that require manual unbinding, with the minimum number
  of renames to drop;
* the independent migration components, each of which can be applied alone;
* deterministic rename steps that break cycles through temporary names.

Nothing here writes to a hive.  Names and bytes are copied, never rewritten in
the source snapshot.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

from . import RegistryName


NAME_KEY = RegistryName.NAME_KEY
NAME_VALUE = RegistryName.NAME_VALUE


class RegistryDiffException(Exception):
    pass


class DuplicateIdentityError(RegistryDiffException):
    pass


class UnknownIdentityError(RegistryDiffException):
    pass


class AmbiguousQueryError(RegistryDiffException):
    def __init__(self, query, candidates, kind):
        super(AmbiguousQueryError, self).__init__(query)
        self.query = query
        self.candidates = list(candidates)
        self.kind = kind

    def __str__(self):
        return "query %r is ambiguous among %s" % (
            self.query, ", ".join(repr(c) for c in self.candidates))


def _identity_sort_key(identity):
    if isinstance(identity, tuple):
        return (0, repr(tuple(_identity_sort_key(part) for part in identity)))
    return (1, "%s:%r" % (type(identity).__name__, identity))


class Entry(object):
    """
    One named entry at a single registry level.

    `identity` names the entity across snapshots; renames are recognized by
    matching identities, never by names.  `payload` is optional evidence used
    only by callers that want to derive an identity from content.
    """

    __slots__ = ("identity", "name", "payload", "metadata")

    def __init__(self, identity, name, payload=None, metadata=None):
        if isinstance(name, RegistryName.Name):
            self.name = name
        else:
            self.name = RegistryName.Name(text=name)
        self.identity = identity
        self.payload = payload
        self.metadata = metadata

    def kind(self):
        return self.name.kind

    def sort_key(self):
        return (_identity_sort_key(self.identity), self.name.sort_key())

    def __repr__(self):
        return "Entry(identity=%r, name=%r)" % (self.identity, self.name)


class Snapshot(object):
    """
    An immutable-by-convention level of registry names.

    Entries are stored in the order given, but every derived result is sorted
    by a canonical key so that caller iteration order never changes a plan.
    """

    def __init__(self, kind, entries=None, label=None):
        self.kind = RegistryName.normalize_kind(kind)
        self.label = label
        self._entries = []
        self._by_identity = {}
        if entries:
            for entry in entries:
                self.add(entry)

    @classmethod
    def from_key(cls, key, kind=NAME_KEY, identity=None, label=None):
        """
        Capture one level of a parsed RegistryKey.

        With ``kind="key"`` the subkeys are captured, with ``kind="value"``
        the values are captured.  Identity defaults to a content fingerprint
        that excludes names, so a rename does not change it.
        """
        kind = RegistryName.normalize_kind(kind)
        entries = []
        if kind == NAME_KEY:
            for index, child in enumerate(key.subkeys()):
                entries.append(Entry(
                    identity or _key_content_fingerprint(child, index=index),
                    child.registry_name(), payload=child.path()))
        else:
            for index, value in enumerate(key.values()):
                entries.append(Entry(
                    identity or _value_fingerprint(value, index=index),
                    value.registry_name(), payload=value.value_type()))
        return cls(kind, entries, label=label)

    @classmethod
    def from_names(cls, kind, names, identities=None, label=None):
        """
        Build a snapshot from strings or RegistryName.Name objects.

        `identities` may be a parallel sequence; when omitted, each entry is
        identified by its position.  Positions are stable only for the same
        caller, so pass real identities when matching two snapshots.
        """
        kind = RegistryName.normalize_kind(kind)
        names = list(names)
        if identities is None:
            identities = list(range(len(names)))
        else:
            identities = list(identities)
        if len(names) != len(identities):
            raise ValueError("names and identities must have the same length")
        entries = []
        for identity, name in zip(identities, names):
            if isinstance(name, RegistryName.Name):
                name = name.with_text(name.text, is_ascii=name.is_ascii,
                                      kind=kind,
                                      format_explicit=name.format_explicit)
            else:
                name = RegistryName.Name(text=name, kind=kind)
            entries.append(Entry(identity, name))
        return cls(kind, entries, label=label)

    def add(self, entry):
        if not isinstance(entry, Entry):
            raise TypeError("Snapshot entries must be Entry instances")
        if entry.kind() != self.kind:
            raise ValueError("entry kind %r does not match snapshot kind %r" % (
                entry.kind(), self.kind))
        if entry.identity in self._by_identity:
            raise DuplicateIdentityError(
                "duplicate identity %r in snapshot" % (entry.identity,))
        self._by_identity[entry.identity] = entry
        self._entries.append(entry)

    def entries(self):
        return list(self._entries)

    def by_identity(self):
        return dict(self._by_identity)

    def get(self, identity):
        try:
            return self._by_identity[identity]
        except KeyError:
            raise UnknownIdentityError("unknown identity %r" % (identity,))

    def __len__(self):
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def sorted_entries(self):
        return sorted(self._entries, key=lambda entry: entry.sort_key())

    def fold_groups(self):
        """
        Return the sets of sibling names that a single query can collapse.

        A query is considered in both storage formats, because the query text
        itself has no storage flag.  For each stored text, the names it would
        match are collected; duplicate sets and sets contained in a larger set
        are dropped.  The result therefore does not depend on entry order and
        does not overstate folding by chaining unrelated queries together.
        """
        entries = self.sorted_entries()
        candidate_texts = set()
        for entry in entries:
            if entry.name.present:
                candidate_texts.add(entry.name.text)
                candidate_texts.add(entry.name.folded())

        found = set()
        for text in candidate_texts:
            allowed = set(RegistryName.candidate_lookup_keys(text, self.kind))
            members = tuple(entry.identity for entry in entries
                            if entry.name.lookup_key() in allowed)
            if len(members) > 1:
                found.add(frozenset(members))

        maximal = []
        for group in found:
            if any(group < other for other in found):
                continue
            maximal.append(group)

        by_id = self.by_identity()
        result = []
        for group in maximal:
            members = [by_id[identity] for identity in group]
            result.append(FoldGroup(self.kind, members))
        result.sort(key=lambda group: group.sort_key())
        return result

    def resolve(self, query):
        """
        Resolve a query string against this level.

        Returns the single matching entry, or raises UnknownIdentityError when
        there is no match and AmbiguousQueryError when more than one stored
        name matches.
        """
        matches = [entry for entry in self.sorted_entries()
                   if RegistryName.query_matches(
                       entry.name.text, entry.name.is_ascii, query, self.kind,
                       present=entry.name.present)]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise UnknownIdentityError("no entry matches %r" % (query,))
        raise AmbiguousQueryError(query, [entry.name.display() for entry in matches],
                                  self.kind)


class FoldGroup(object):
    """
    A set of sibling names that one query can collapse together.
    """

    def __init__(self, kind, entries):
        self.kind = kind
        self.entries = list(entries)

    identities = property(lambda self: [entry.identity for entry in self.entries])
    names = property(lambda self: [entry.name.display() for entry in self.entries])

    def sort_key(self):
        return tuple(entry.sort_key() for entry in self.entries)

    def to_dict(self):
        return {
            "kind": self.kind,
            "identities": self.identities,
            "names": self.names,
            "raw": [entry.name.raw.hex() for entry in self.entries],
            "formats": [RegistryName.FORMAT_ASCII if entry.name.is_ascii
                        else RegistryName.FORMAT_UTF16 for entry in self.entries],
        }

    def __repr__(self):
        return "FoldGroup(kind=%r, names=%r)" % (self.kind, self.names)


class RenameIntent(object):
    """
    A request to give one source entry a new name.

    The old Name keeps its exact bytes.  The new Name also keeps the bytes the
    caller supplied; when the caller supplies only text, the source format is
    used if the text fits, otherwise the name is encoded as UTF-16LE.
    """

    __slots__ = ("source", "old_name", "new_name")

    def __init__(self, source, new_name):
        if not isinstance(source, Entry):
            raise TypeError("rename source must be an Entry")
        self.source = source
        self.old_name = source.name
        if isinstance(new_name, RegistryName.Name):
            self.new_name = new_name
        else:
            self.new_name = _make_target_name(source.name, new_name)

    def source_identity(self):
        return self.source.identity

    def target_key(self):
        return self.new_name.lookup_key()

    def sort_key(self):
        return (self.source.sort_key(), self.new_name.sort_key())

    def to_dict(self, accepted=None):
        result = {
            "identity": self.source.identity,
            "old": self.source.name.to_dict(),
            "new": self.new_name.to_dict(),
        }
        if accepted is not None:
            result["accepted"] = bool(accepted)
        return result

    def __repr__(self):
        return "RenameIntent(%r -> %r)" % (
            self.source.name.display(), self.new_name.display())


def _make_target_name(old_name, text):
    if old_name.kind == NAME_VALUE and text in ("", RegistryName.DEFAULT_DISPLAY_NAME):
        return RegistryName.Name.default_value()
    if isinstance(text, RegistryName.Name):
        return text.with_text(text.text, kind=old_name.kind)
    if old_name.is_ascii:
        try:
            return RegistryName.Name(text=text, is_ascii=True,
                                     kind=old_name.kind, format_explicit=False)
        except UnicodeEncodeError:
            pass
    return RegistryName.Name(text=text, is_ascii=False, kind=old_name.kind,
                             format_explicit=False)


def _value_fingerprint(value, index=0):
    raw = value.raw_data()
    return ("value", index, value.value_type(), len(raw), raw)


def _key_content_fingerprint(key, depth=2, index=0):
    """
    A name-independent identity for a key.

    This is what `Snapshot.from_key` uses by default, so renaming a key does
    not change the identity used to match it across snapshots.  `index` keeps
    identical siblings distinct without consulting their names.
    """
    value_types = tuple(value.value_type() for value in key.values())
    value_sizes = tuple(len(value.raw_data()) for value in key.values())
    child_count = key.subkeys_number()
    if depth > 1:
        children = tuple(_key_content_fingerprint(subkey, depth - 1,
                                                  index=child_index)
                         for child_index, subkey in enumerate(key.subkeys()))
    else:
        children = ()
    return ("key", index, value_types, value_sizes, child_count, children)


def _key_fingerprint(key, depth=2, index=0):
    """
    A name-sensitive identity for a key.

    Values and a bounded number of descendant names are included.  The bound
    keeps capture cheap on large hives; pass explicit identities when a
    caller has stronger evidence.
    """
    values = tuple((value.name_bytes(), value.value_type(),
                    len(value.raw_data())) for value in key.values())
    subkeys = tuple(("key", subkey.name_bytes())
                    for subkey in key.subkeys())
    if depth > 1:
        children = tuple(_key_fingerprint(subkey, depth - 1,
                                          index=child_index)
                         for child_index, subkey in enumerate(key.subkeys()))
    else:
        children = ()
    return ("key", index, values, subkeys, children)


class RenameMapping(object):
    """
    The correspondence between two snapshots, matched by identity.
    """

    def __init__(self, source, destination, intents, unchanged, removed, added):
        self.source = source
        self.destination = destination
        self.intents = list(intents)
        self.unchanged = list(unchanged)
        self.removed = list(removed)
        self.added = list(added)
        self._forward = dict((intent.source_identity(), intent)
                             for intent in intents)
        self._unchanged = dict((entry.identity, entry) for entry in unchanged)

    def forward(self, identity):
        if identity in self._forward:
            return self._forward[identity]
        if identity in self._unchanged:
            return self._unchanged[identity]
        raise UnknownIdentityError("identity %r is not in the mapping" % (identity,))

    def reverse(self, target):
        key = target.lookup_key() if isinstance(target, RegistryName.Name) \
            else RegistryName.lookup_key(target, self.source.kind, False)
        matches = [intent for intent in self.intents
                   if intent.target_key() == key]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise UnknownIdentityError("no rename targets %r" % (target,))
        raise AmbiguousQueryError(getattr(target, "text", target),
                                  [m.new_name.display() for m in matches],
                                  self.source.kind)

    def to_dict(self):
        return {
            "kind": self.source.kind,
            "intents": [intent.to_dict() for intent in self.intents],
            "unchanged": [entry.identity for entry in self.unchanged],
            "removed": [entry.identity for entry in self.removed],
            "added": [entry.identity for entry in self.added],
        }


def build_rename_map(source, destination, name_targets=None):
    """
    Match two snapshots and produce rename intents.

    Matching is by identity.  A pair is a rename when the stored name differs
    in text, bytes, or storage format.  Destination-only entries are reported
    as additions; they still occupy target names when a plan is evaluated.
    """
    if source.kind != destination.kind:
        raise ValueError("source and destination snapshots must have the same kind")

    source_by_id = source.by_identity()
    destination_by_id = destination.by_identity()

    intents = []
    unchanged = []
    removed = []

    for identity in sorted(source_by_id, key=_identity_sort_key):
        entry = source_by_id[identity]
        if identity not in destination_by_id:
            removed.append(entry)
            continue
        target = destination_by_id[identity]
        if name_targets is not None and identity in name_targets:
            new_name = name_targets[identity]
            if not isinstance(new_name, RegistryName.Name):
                new_name = _make_target_name(entry.name, new_name)
        else:
            new_name = target.name
        if new_name != entry.name:
            intents.append(RenameIntent(entry, new_name))
        else:
            unchanged.append(entry)

    added = [destination_by_id[identity]
             for identity in sorted(destination_by_id, key=_identity_sort_key)
             if identity not in source_by_id]
    return RenameMapping(source, destination, intents, unchanged, removed, added)


class RenameStep(object):
    """
    One deterministic step of a rename plan.
    """

    def __init__(self, entry_id, old_name, new_name, temporary=False):
        self.entry_id = entry_id
        self.old_name = old_name
        self.new_name = new_name
        self.temporary = temporary

    def to_dict(self):
        return {
            "identity": self.entry_id,
            "old": self.old_name.to_dict(),
            "new": self.new_name.to_dict(),
            "temporary": self.temporary,
        }

    def __repr__(self):
        return "RenameStep(%r, %r -> %r%s)" % (
            self.entry_id, self.old_name.display(), self.new_name.display(),
            ", temporary" if self.temporary else "")


class ConflictGroup(object):
    """
    A set of rename intents that cannot all be applied.

    `unbound` is a minimum set to hand to a human; the remaining `accepted`
    intents are safe.  `reasons` names the target keys that collide.
    """

    def __init__(self, component_id, intents, accepted, unbound, reasons):
        self.component_id = component_id
        self.intents = list(intents)
        self.accepted = list(accepted)
        self.unbound = list(unbound)
        self.reasons = list(reasons)

    def to_dict(self):
        return {
            "component": self.component_id,
            "intents": [intent.to_dict() for intent in self.intents],
            "accepted": [intent.source_identity() for intent in self.accepted],
            "unbound": [intent.to_dict() for intent in self.unbound],
            "reasons": self.reasons,
        }

    def __repr__(self):
        return "ConflictGroup(accepted=%d, unbound=%d)" % (
            len(self.accepted), len(self.unbound))


class MigrationComponent(object):
    """
    An interaction component whose accepted renames can be applied alone.
    """

    def __init__(self, component_id, intents, accepted, unbound, steps):
        self.component_id = component_id
        self.intents = list(intents)
        self.accepted = list(accepted)
        self.unbound = list(unbound)
        self.steps = list(steps)

    def is_complete(self):
        return not self.unbound

    def to_dict(self):
        return {
            "component": self.component_id,
            "intents": [intent.to_dict() for intent in self.intents],
            "accepted": [intent.to_dict(accepted=True) for intent in self.accepted],
            "unbound": [intent.to_dict(accepted=False) for intent in self.unbound],
            "steps": [step.to_dict() for step in self.steps],
            "complete": self.is_complete(),
        }

    def __repr__(self):
        return "MigrationComponent(%r, accepted=%d, unbound=%d)" % (
            self.component_id, len(self.accepted), len(self.unbound))


class _Feasibility(object):
    """
    Decide whether a set of rename intents can be applied at one level.

    For every entry, compute the lookup key it would occupy afterwards: the
    target key when its intent is accepted, its original key otherwise.  A
    selection is feasible only when no final key has more than one occupant.
    This covers duplicate targets, overwriting a stationary entry, and a
    target that only becomes free when its current holder also moves.
    """

    def __init__(self, intents, universe):
        self.intents = list(intents)
        self.universe = list(universe)
        self.by_identity = dict((intent.source_identity(), intent)
                                for intent in self.intents)
        self.by_old_key = {}
        for intent in self.intents:
            self.by_old_key.setdefault(
                intent.old_name.lookup_key(), []).append(intent)

        intent_ids = set(self.by_identity)
        self.persistent_keys = set(
            entry.name.lookup_key() for entry in self.universe
            if entry.identity not in intent_ids)
        self.static_key_holders = set(
            entry.identity for entry in self.universe
            if entry.identity not in intent_ids)

    def forced(self, identity):
        """
        Intents that must also be accepted when `identity` is accepted.

        If an accepted rename wants the name another intent-bearing entry
        currently holds, that entry must move too; otherwise its old name
        would collide with the new one.
        """
        intent = self.by_identity.get(identity)
        if intent is None:
            return []
        target = intent.target_key()
        result = []
        for dependent in self.by_old_key.get(target, ()):
            dependent_id = dependent.source_identity()
            if dependent_id == identity:
                continue
            if dependent.target_key() == target:
                # The dependent stays on the wanted key, so it can never free
                # it.  The caller treats this as a hard conflict.
                result.append((dependent_id, True))
            else:
                result.append((dependent_id, False))
        return result

    def closure(self, selected_ids):
        chosen = set(selected_ids)
        pending = list(chosen)
        blocked = False
        while pending:
            identity = pending.pop()
            for dependent_id, stays in self.forced(identity):
                if stays:
                    blocked = True
                if dependent_id not in chosen:
                    chosen.add(dependent_id)
                    pending.append(dependent_id)
        return frozenset(chosen), blocked

    def forbidden_ids(self):
        """
        Intents that can never be accepted, with implications propagated.

        An intent is forbidden when its target is permanently occupied by a
        non-participant, or when the entry holding that target would stay on
        it even after moving.  If accepting A forces B and B is forbidden,
        then A is forbidden too, so whole chains collapse in one pass.
        """
        forbidden = set()
        changed = True
        while changed:
            changed = False
            for intent in self.intents:
                identity = intent.source_identity()
                if identity in forbidden:
                    continue
                target = intent.target_key()
                if target in self.persistent_keys:
                    forbidden.add(identity)
                    changed = True
                    continue
                for dependent_id, stays in self.forced(identity):
                    if stays or dependent_id in forbidden:
                        forbidden.add(identity)
                        changed = True
                        break
        # Propagate backwards: any intent that forces a forbidden one is also
        # forbidden.  Repeat until stable.
        changed = True
        while changed:
            changed = False
            for intent in self.intents:
                identity = intent.source_identity()
                if identity in forbidden:
                    continue
                for dependent_id, stays in self.forced(identity):
                    if dependent_id in forbidden:
                        forbidden.add(identity)
                        changed = True
                        break
        return forbidden

    def has_implications(self):
        for intent in self.intents:
            for _ in self.forced(intent.source_identity()):
                return True
        return False

    def _final_keys(self, selected_ids):
        chosen, blocked = self.closure(selected_ids)
        problems = []
        if blocked:
            problems.append(("blocked_target", tuple(sorted(chosen, key=repr))))
        moved = {}
        for identity in chosen:
            moved[identity] = self.by_identity[identity].target_key()

        counts = {}
        owners = {}
        for entry in self.universe:
            identity = entry.identity
            key = moved.get(identity, entry.name.lookup_key())
            counts[key] = counts.get(key, 0) + 1
            owners.setdefault(key, []).append(identity)

        for key, count in counts.items():
            if count > 1:
                problems.append(("collision", key, tuple(owners[key])))
        return problems

    def problems(self, selected):
        return self._final_keys(set(intent.source_identity()
                                    for intent in selected))

    def feasible(self, selected):
        return not self.problems(selected)


def _minimum_removal(component, universe):
    """
    Return (accepted, unbound) maximizing the accepted set.

    Only safe reductions are applied before the exact search:

    * intents whose target is permanently held by a non-participant are
      rejected, and rejection propagates backwards through forced moves;
    * a rename whose forced closure is feasible and whose targets no other
      rename wants is safe to accept;
    * when no rename forces another, targets are independent and one
      canonically-chosen intent per target key is optimal.

    The remaining contested core is searched exactly.  Components are the
    connected pieces of one level, so this is bounded by the number of renames
    that actually interact.  Ties break on canonical intent order, making the
    result independent of caller traversal order.
    """
    ordered = sorted(component, key=lambda intent: intent.sort_key())
    checker = _Feasibility(ordered, universe)
    forbidden = checker.forbidden_ids()

    allowed = [intent for intent in ordered
               if intent.source_identity() not in forbidden]
    blocked = [intent for intent in ordered
               if intent.source_identity() in forbidden]

    if not allowed:
        return [], ordered

    peeled, contested = _peel_free(allowed, checker)
    if not contested:
        accepted = sorted(peeled, key=lambda intent: intent.sort_key())
        unbound = sorted(blocked, key=lambda intent: intent.sort_key())
        return accepted, unbound

    if not checker.has_implications():
        extra_accepted, extra_unbound = _select_one_per_target(contested)
        accepted = sorted(peeled + extra_accepted,
                          key=lambda intent: intent.sort_key())
        unbound = sorted(blocked + extra_unbound,
                         key=lambda intent: intent.sort_key())
        return accepted, unbound

    by_identity = dict((intent.source_identity(), intent)
                       for intent in contested)
    identities = [intent.source_identity() for intent in contested]
    count = len(identities)
    peeled_ids = frozenset(intent.source_identity() for intent in peeled)
    target_keys = dict((identity, by_identity[identity].target_key())
                       for identity in identities)

    best = {"accepted": frozenset(), "key": None}

    def key_for(chosen):
        return tuple(sorted(by_identity[i].sort_key() for i in chosen))

    def better(chosen, candidate_key):
        current = best["accepted"]
        if len(chosen) > len(current):
            return True
        if len(chosen) == len(current):
            return best["key"] is None or candidate_key < best["key"]
        return False

    def visit(index, chosen):
        if len(chosen) + (count - index) < len(best["accepted"]):
            return
        if index == count:
            members = [by_identity[i] for i in chosen]
            if not checker.feasible(members):
                return
            candidate_key = key_for(chosen)
            if better(chosen, candidate_key):
                best["accepted"] = frozenset(chosen)
                best["key"] = candidate_key
            return

        identity = identities[index]
        expanded, blocked_now = checker.closure(chosen | set([identity]))
        expanded = frozenset(i for i in expanded if i in by_identity)
        if not blocked_now and checker.feasible(
                [by_identity[i] for i in expanded]):
            visit(index + 1, expanded)
        visit(index + 1, chosen)

    visit(0, frozenset())
    accepted = peeled + [by_identity[identity] for identity in identities
                         if identity in best["accepted"]]
    unbound = [by_identity[identity] for identity in identities
               if identity not in best["accepted"]] + blocked
    accepted.sort(key=lambda intent: intent.sort_key())
    unbound.sort(key=lambda intent: intent.sort_key())
    return accepted, unbound


def _peel_free(component, checker):
    """
    Accept renames that cannot make any other rename infeasible.

    A candidate is safe when its forced closure is feasible and every target
    it uses is either claimed only by that closure or is its own target.
    Accepting it frees its old key and adds no competition for a key another
    rename still needs.
    """
    remaining = list(component)
    accepted = []
    accepted_ids = set()
    changed = True
    while changed and remaining:
        changed = False
        target_use = {}
        for intent in remaining:
            target_use.setdefault(intent.target_key(), set()).add(
                intent.source_identity())
        for intent in sorted(remaining, key=lambda item: item.sort_key()):
            identity = intent.source_identity()
            closure, blocked_now = checker.closure([identity])
            closure = frozenset(i for i in closure if i in checker.by_identity
                                and i not in accepted_ids)
            if blocked_now:
                continue
            members = [checker.by_identity[i] for i in closure]
            exclusive = True
            for member in members:
                claimants = target_use.get(member.target_key(), set())
                if claimants - closure:
                    exclusive = False
                    break
                if member.target_key() in checker.persistent_keys:
                    exclusive = False
                    break
            if not exclusive:
                continue
            if not checker.feasible(members):
                continue
            accepted.extend(members)
            accepted_ids.update(closure)
            remaining = [candidate for candidate in remaining
                         if candidate.source_identity() not in closure]
            changed = True
            break
    return accepted, remaining


def _select_one_per_target(allowed):
    """
    Maximize an implication-free component.

    Without forced moves the only conflicts are shared target keys.  At most
    one intent per target can be accepted, so the canonically first intent of
    each target group is optimal.
    """
    accepted = []
    unbound = []
    used_targets = set()
    for intent in sorted(allowed, key=lambda item: item.sort_key()):
        target = intent.target_key()
        if target in used_targets:
            unbound.append(intent)
            continue
        used_targets.add(target)
        accepted.append(intent)
    return accepted, unbound


class RenamePlan(object):
    """
    The result of planning renames for one registry level.
    """

    def __init__(self, source, intents, accepted, unbound, components,
                 universe):
        self.source = source
        self.universe = list(universe)
        self.intents = list(intents)
        self.accepted = list(accepted)
        self.unbound = list(unbound)
        self._components = list(components)

    def accepted_intents(self):
        return list(self.accepted)

    def unbound_intents(self):
        return list(self.unbound)

    def components(self):
        return list(self._components)

    def conflict_groups(self):
        return [ConflictGroup(component.component_id, component.intents,
                              component.accepted, component.unbound,
                              _component_reasons(component, self.universe))
                for component in self._components if component.unbound]

    def minimum_unbound_count(self):
        return len(self.unbound)

    def steps(self):
        steps = []
        for component in self._components:
            steps.extend(component.steps)
        return steps

    def final_names(self):
        """
        Return identity -> Name after applying the accepted renames.

        Entries whose intent was unbound keep their original Name object,
        including its original bytes.
        """
        result = {}
        moved = dict((intent.source_identity(), intent)
                     for intent in self.accepted)
        for entry in self.universe:
            if entry.identity in moved:
                result[entry.identity] = moved[entry.identity].new_name
            else:
                result[entry.identity] = entry.name
        return result

    def to_dict(self):
        return {
            "kind": self.source.kind,
            "accepted": [intent.to_dict(accepted=True) for intent in self.accepted],
            "unbound": [intent.to_dict(accepted=False) for intent in self.unbound],
            "minimum_unbound": len(self.unbound),
            "conflict_groups": [group.to_dict()
                                for group in self.conflict_groups()],
            "components": [component.to_dict() for component in self._components],
            "steps": [step.to_dict() for step in self.steps()],
        }

    def __repr__(self):
        return "RenamePlan(accepted=%d, unbound=%d, components=%d)" % (
            len(self.accepted), len(self.unbound), len(self._components))


def _component_reasons(component, universe):
    intents = list(component.intents)
    intent_ids = set(intent.source_identity() for intent in intents)
    persistent_keys = set(entry.name.lookup_key() for entry in universe
                          if entry.identity not in intent_ids)
    by_target = {}
    for intent in intents:
        by_target.setdefault(intent.target_key(), []).append(intent)
    reasons = {"duplicate_target": [], "overwrites_persistent": [],
               "duplicate_source": []}

    for target, group in sorted(by_target.items()):
        if len(group) > 1:
            reasons["duplicate_target"].append(
                (target, [intent.source_identity() for intent in group]))
    for intent in intents:
        if intent.target_key() in persistent_keys:
            reasons["overwrites_persistent"].append(
                (intent.target_key(), intent.source_identity()))
    source_counts = {}
    for intent in intents:
        source_counts.setdefault(intent.source_identity(), []).append(intent)
    for identity, group in sorted(source_counts.items(),
                                  key=lambda item: repr(item[0])):
        if len(group) > 1:
            reasons["duplicate_source"].append(identity)
    return dict((key, value) for key, value in reasons.items() if value)


def _interaction_components(intents):
    """
    Group intents that can affect each other.

    Two intents interact when they share a source identity, share a target
    key, or one targets the key the other is vacating.  Components do not
    interact, so each can be migrated on its own.
    """
    intents = list(intents)
    parent = list(range(len(intents)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    source_index = {}
    for index, intent in enumerate(intents):
        source_index.setdefault(intent.source_identity(), []).append(index)

    for indices in source_index.values():
        first = indices[0]
        for other in indices[1:]:
            union(first, other)

    target_index = {}
    for index, intent in enumerate(intents):
        target_index.setdefault(intent.target_key(), []).append(index)
    for indices in target_index.values():
        first = indices[0]
        for other in indices[1:]:
            union(first, other)

    source_key_index = {}
    for index, intent in enumerate(intents):
        source_key_index.setdefault(intent.old_name.lookup_key(), []).append(index)

    for index, intent in enumerate(intents):
        if intent.target_key() in source_key_index:
            for other in source_key_index[intent.target_key()]:
                union(index, other)

    groups = {}
    for index in range(len(intents)):
        groups.setdefault(find(index), []).append(index)

    components = []
    for indices in groups.values():
        members = [intents[index] for index in indices]
        members.sort(key=lambda intent: intent.sort_key())
        components.append(members)
    components.sort(key=lambda members: tuple(i.sort_key() for i in members))
    return components


def _temporary_name(entry, existing_keys, serial):
    index = serial
    while True:
        text = "\x00rename-temp:%s:%d" % (entry.identity, index)
        candidate = RegistryName.Name(text=text, is_ascii=False,
                                      kind=entry.name.kind,
                                      format_explicit=False)
        if candidate.lookup_key() not in existing_keys:
            return candidate
        index += 1


def _execution_steps(accepted, universe):
    """
    Produce deterministic steps for one component.

    A cycle is broken by moving one entry to a temporary name that no other
    name at this level can occupy.  The first rename in a cycle is chosen by
    canonical sort, so traversal order cannot pick a different winner.
    """
    ordered = sorted(accepted, key=lambda intent: intent.sort_key())
    pending = list(ordered)
    current = {}
    steps = []

    intent_ids = set(intent.source_identity() for intent in ordered)
    occupied = set(entry.name.lookup_key() for entry in universe
                   if entry.identity not in intent_ids)
    for intent in ordered:
        current[intent.source_identity()] = intent.old_name
        occupied.add(intent.old_name.lookup_key())

    while pending:
        chosen = None
        for intent in pending:
            identity = intent.source_identity()
            target = intent.target_key()
            if target == current[identity].lookup_key():
                chosen = intent
                break
            if target not in occupied:
                chosen = intent
                break

        if chosen is not None:
            identity = chosen.source_identity()
            old_name = current[identity]
            occupied.discard(old_name.lookup_key())
            steps.append(RenameStep(identity, old_name, chosen.new_name))
            current[identity] = chosen.new_name
            occupied.add(chosen.new_name.lookup_key())
            pending.remove(chosen)
            continue

        chosen = pending[0]
        identity = chosen.source_identity()
        temp = _temporary_name(chosen.source, occupied, len(steps))
        old_name = current[identity]
        occupied.discard(old_name.lookup_key())
        steps.append(RenameStep(identity, old_name, temp, temporary=True))
        current[identity] = temp
        occupied.add(temp.lookup_key())

    return steps


def plan_renames(source, intents=None, destination=None, blockers=None):
    """
    Plan renames for one level and return a RenamePlan.

    `source` is the level being renamed.  `intents` are RenameIntent objects;
    when omitted they are inferred from `destination` by identity.  `blockers`
    are destination-only Entry objects that must keep their names and must
    never be overwritten.  When `destination` is given, its entries that do
    not exist in the source are added as blockers automatically.

    The result is independent of the order in which intents or blockers are
    supplied.  Every decision is made after a canonical sort.
    """
    if not isinstance(source, Snapshot):
        raise TypeError("source must be a Snapshot")

    if intents is None:
        if destination is None:
            raise ValueError("either intents or destination is required")
        intents = build_rename_map(source, destination).intents
    else:
        intents = list(intents)
        for intent in intents:
            if not isinstance(intent, RenameIntent):
                raise TypeError("intents must be RenameIntent objects")

    blockers = list(blockers or [])
    if destination is not None:
        source_ids = set(source.by_identity())
        blockers.extend(entry for entry in destination.sorted_entries()
                        if entry.identity not in source_ids)

    universe = list(source.sorted_entries())
    sources = set(source.by_identity())
    seen_blockers = set()
    for entry in blockers:
        if entry.kind() != source.kind:
            raise ValueError("blocker kind does not match source kind")
        if entry.identity in sources:
            raise ValueError("blocker identity %r is also a source entry"
                             % (entry.identity,))
        if entry.identity in seen_blockers:
            continue
        seen_blockers.add(entry.identity)
        universe.append(entry)

    for intent in intents:
        if intent.source_identity() not in sources:
            raise UnknownIdentityError(
                "intent source %r is not in the source snapshot"
                % (intent.source_identity(),))

    seen = {}
    for intent in sorted(intents, key=lambda item: item.sort_key()):
        identity = intent.source_identity()
        if identity in seen:
            raise DuplicateIdentityError(
                "more than one rename targets identity %r" % (identity,))
        seen[identity] = intent

    intents.sort(key=lambda intent: intent.sort_key())

    components = _interaction_components(intents)
    accepted = []
    unbound = []
    migration_components = []

    for component_id, component in enumerate(components):
        kept, removed = _minimum_removal(component, universe)
        accepted.extend(kept)
        unbound.extend(removed)
        steps = _execution_steps(kept, universe)
        migration_components.append(
            MigrationComponent(component_id, component, kept, removed, steps))

    accepted.sort(key=lambda intent: intent.sort_key())
    unbound.sort(key=lambda intent: intent.sort_key())
    for component in migration_components:
        component.accepted.sort(key=lambda intent: intent.sort_key())
        component.unbound.sort(key=lambda intent: intent.sort_key())

    return RenamePlan(source, intents, accepted, unbound, migration_components,
                      universe)


def rename_map(source, destination, name_targets=None):
    """
    Convenience wrapper returning the identity-matched mapping.
    """
    return build_rename_map(source, destination, name_targets=name_targets)


def plan_from_mapping(mapping, blockers=None):
    """
    Plan the renames inferred by a RenameMapping.
    """
    return plan_renames(mapping.source, mapping.intents, blockers=blockers)
