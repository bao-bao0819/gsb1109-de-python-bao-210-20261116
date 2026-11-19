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
Control set resolution for offline SYSTEM hives.

An offline SYSTEM hive usually contains several physical control sets
(ControlSet001, ControlSet002, ...), while collection rules are written
against the virtual path "CurrentControlSet" that only exists on a live
system.  This module builds the mapping from such virtual query paths to
the physical keys of one snapshot of the hive.

The roles Current, Default, Failed and LastKnownGood are interpreted
from the Select values of the *same* snapshot.  Roles are distinct
entities: two roles pointing at the same physical key are still two
separate resolutions.

A role is *undefined* -- and never falls back to the lowest numbered
control set -- when:

 - the Select key or the role's Select value is missing,
 - the Select value has an unexpected type (source format mismatch), or
 - the Select value points to a control set that does not exist in this
   snapshot.

Snapshots are immutable: after the current control set switches, an old
snapshot still resolves to its old physical keys.  ControlSetHistory
ingests successive snapshots, exports each round's effective
configuration together with the rationale for every role change, and
stores each physical control set only once even when several roles (or
rounds) reference it.
"""
from __future__ import print_function

import re
from enum import Enum

from . import RegistryParse
from .Registry import RegDWord
from .Registry import RegistryKeyNotFoundException
from .Registry import RegistryValueNotFoundException

VIRTUAL_CURRENT_CONTROL_SET = "CurrentControlSet"

_CONTROL_SET_NAME_RE = re.compile(r"^ControlSet\d{3}$", re.IGNORECASE)


class ControlSetRole(Enum):
    """
    The four control set roles described by the Select key.
    The enum value is the name of the corresponding Select value.
    """
    CURRENT = "Current"
    DEFAULT = "Default"
    FAILED = "Failed"
    LAST_KNOWN_GOOD = "LastKnownGood"


class UndefinedReason(Enum):
    """
    Why a control set role could not be resolved.
    """
    SELECT_KEY_MISSING = "the Select key does not exist in this snapshot"
    SELECT_VALUE_MISSING = "the Select value for this role does not exist"
    SELECT_VALUE_TYPE_MISMATCH = "the Select value is not a REG_DWORD (source format mismatch)"
    CONTROL_SET_NOT_FOUND = "the Select value points to a control set that does not exist in this snapshot"


class ControlSetUndefinedException(RegistryParse.RegistryException):
    """
    Raised when a virtual path cannot be mapped to a physical key
    because the underlying role is undefined in this snapshot.
    """
    def __init__(self, role, reason):
        super(ControlSetUndefinedException, self).__init__(
            "control set role {0} is undefined: {1}".format(role.value, reason.value))
        self.role = role
        self.reason = reason


class ControlSetResolution(object):
    """
    The resolution of one control set role within one snapshot.

    A resolution is either defined (a physical control set name is
    available) or undefined (a reason is available).  It is never
    silently mapped to a fallback control set.
    """
    def __init__(self, role, select_value=None, control_set_name=None, reason=None):
        self._role = role
        self._select_value = select_value
        self._control_set_name = control_set_name
        self._reason = reason

    def __repr__(self):
        if self.is_defined():
            return 'ControlSetResolution(role="{0}", control_set="{1}")'.format(
                self._role.value, self._control_set_name)
        return 'ControlSetResolution(role="{0}", undefined="{1}")'.format(
            self._role.value, self._reason.name)

    def role(self):
        return self._role

    def is_defined(self):
        return self._reason is None

    def select_value(self):
        """
        The raw integer read from the Select key, or None if it could
        not be read.
        """
        return self._select_value

    def control_set_name(self):
        """
        The physical control set key name (for example "ControlSet002"),
        or None if the role is undefined.
        """
        return self._control_set_name

    def reason(self):
        """
        The UndefinedReason if the role is undefined, else None.
        """
        return self._reason

    def rationale(self):
        """
        A human readable explanation of how this resolution was reached.
        """
        if self.is_defined():
            return "Select\\{0} is {1}, mapped to {2}".format(
                self._role.value, self._select_value, self._control_set_name)
        if self._select_value is not None:
            return "Select\\{0} is {1}: {2}".format(
                self._role.value, self._select_value, self._reason.value)
        return "Select\\{0}: {1}".format(self._role.value, self._reason.value)

    def export(self):
        """
        A JSON-serializable description of this resolution.
        """
        record = {
            "role": self._role.value,
            "status": "resolved" if self.is_defined() else "undefined",
            "select_value": self._select_value,
            "control_set": self._control_set_name,
            "rationale": self.rationale(),
        }
        if not self.is_defined():
            record["reason"] = self._reason.name
        return record


def _control_set_name_for_number(number):
    return "ControlSet%03d" % (number)


class ControlSetSnapshot(object):
    """
    An immutable view of the control set layout of one SYSTEM hive.

    The Select values are read once at construction; every role is
    interpreted against that single read, so all resolutions are
    consistent with each other.  The snapshot keeps its own reference to
    the registry, therefore it keeps resolving to the same physical keys
    even after the current control set switches in a later snapshot.
    """
    def __init__(self, registry):
        """
        Arguments:
        - `registry`: A Registry.Registry opened on a SYSTEM hive.
        """
        self._registry = registry
        self._control_sets = self._enumerate_control_sets(registry)
        self._resolutions = self._resolve_all(registry, self._control_sets)

    @staticmethod
    def _enumerate_control_sets(registry):
        control_sets = {}
        for subkey in registry.root().subkeys():
            name = subkey.name()
            if _CONTROL_SET_NAME_RE.match(name):
                control_sets[name.lower()] = name
        return control_sets

    @staticmethod
    def _resolve_all(registry, control_sets):
        try:
            select_key = registry.open("Select")
        except RegistryKeyNotFoundException:
            return dict((role, ControlSetResolution(role, reason=UndefinedReason.SELECT_KEY_MISSING))
                        for role in ControlSetRole)

        resolutions = {}
        for role in ControlSetRole:
            resolutions[role] = ControlSetSnapshot._resolve_role(select_key, control_sets, role)
        return resolutions

    @staticmethod
    def _resolve_role(select_key, control_sets, role):
        try:
            select_value = select_key.value(role.value)
        except RegistryValueNotFoundException:
            return ControlSetResolution(role, reason=UndefinedReason.SELECT_VALUE_MISSING)

        if select_value.value_type() != RegDWord:
            return ControlSetResolution(role, reason=UndefinedReason.SELECT_VALUE_TYPE_MISMATCH)

        number = select_value.value()
        name = _control_set_name_for_number(number)
        physical_name = control_sets.get(name.lower())
        if physical_name is None:
            return ControlSetResolution(role, select_value=number,
                                        reason=UndefinedReason.CONTROL_SET_NOT_FOUND)
        return ControlSetResolution(role, select_value=number, control_set_name=physical_name)

    def resolution(self, role):
        """
        Return the ControlSetResolution for the given ControlSetRole.
        """
        return self._resolutions[role]

    def is_defined(self, role):
        return self._resolutions[role].is_defined()

    def control_set_name(self, role):
        """
        Return the physical control set key name for the role, or None
        if the role is undefined in this snapshot.
        """
        return self._resolutions[role].control_set_name()

    def physical_control_sets(self):
        """
        Return the physical control set key names referenced by the
        defined roles, each listed exactly once no matter how many
        roles point at it, sorted by name.
        """
        names = set()
        for resolution in self._resolutions.values():
            if resolution.is_defined():
                names.add(resolution.control_set_name())
        return sorted(names)

    def translate_path(self, path, role=ControlSetRole.CURRENT):
        """
        Map a virtual query path to the physical key path of this
        snapshot.  A path rooted at "CurrentControlSet" (matched case
        insensitively) is rewritten to the physical control set of the
        given role; any other path is returned unchanged.

        Raises ControlSetUndefinedException if the role is undefined.
        """
        (immediate, sep, future) = path.partition("\\")
        if immediate.lower() != VIRTUAL_CURRENT_CONTROL_SET.lower():
            return path
        resolution = self._resolutions[role]
        if not resolution.is_defined():
            raise ControlSetUndefinedException(role, resolution.reason())
        physical = resolution.control_set_name()
        if sep:
            return physical + sep + future
        return physical

    def open(self, path, role=ControlSetRole.CURRENT):
        """
        Open a RegistryKey by virtual or physical path within this
        snapshot.  Virtual paths are translated via translate_path().

        Raises ControlSetUndefinedException if the role is undefined.
        """
        return self._registry.open(self.translate_path(path, role=role))

    def effective_config(self):
        """
        Export this round's effective configuration: every role's
        resolution (roles stay distinct entities even when they share a
        physical key) plus the deduplicated list of physical control
        sets in use.
        """
        roles = {}
        for role in ControlSetRole:
            roles[role.value] = self._resolutions[role].export()
        return {
            "roles": roles,
            "physical_control_sets": self.physical_control_sets(),
        }


class RoleChange(object):
    """
    How one role's resolution changed between two rounds (or how it was
    first observed, when there is no previous round).
    """
    def __init__(self, role, previous, current):
        self._role = role
        self._previous = previous
        self._current = current

    def __repr__(self):
        return 'RoleChange(role="{0}", rationale="{1}")'.format(self._role.value, self.rationale())

    def role(self):
        return self._role

    def previous(self):
        """
        The previous ControlSetResolution, or None on the first round.
        """
        return self._previous

    def current(self):
        return self._current

    def changed(self):
        if self._previous is None:
            return True
        return (self._previous.is_defined() != self._current.is_defined() or
                self._previous.control_set_name() != self._current.control_set_name() or
                self._previous.reason() != self._current.reason())

    def rationale(self):
        """
        A human readable justification for the role's current state.
        """
        current = self._current
        if self._previous is None:
            prefix = "role '{0}' initially ".format(self._role.value)
        else:
            prefix = "role '{0}' ".format(self._role.value)
        if current.is_defined():
            description = "resolved to {0} ({1})".format(
                current.control_set_name(), current.rationale())
        else:
            description = "undefined ({0})".format(current.rationale())
        if self._previous is None:
            return prefix + description
        previous = self._previous
        if previous.is_defined():
            was = "was {0}".format(previous.control_set_name())
        else:
            was = "was undefined ({0})".format(previous.reason().value)
        return "{0}{1}, {2}".format(prefix, description, was)

    def export(self):
        return {
            "role": self._role.value,
            "previous": self._previous.export() if self._previous is not None else None,
            "current": self._current.export(),
            "rationale": self.rationale(),
        }


class ControlSetRoundReport(object):
    """
    The export of one ingested round: the effective configuration of
    that round's snapshot, the rationale for every role change relative
    to the previous round, and the physical control sets that were newly
    stored (each physical key enters the store only once).
    """
    def __init__(self, round_number, snapshot, role_changes, new_physical_control_sets):
        self._round_number = round_number
        self._snapshot = snapshot
        self._role_changes = role_changes
        self._new_physical_control_sets = new_physical_control_sets

    def round_number(self):
        return self._round_number

    def snapshot(self):
        return self._snapshot

    def effective_config(self):
        return self._snapshot.effective_config()

    def role_changes(self):
        return list(self._role_changes)

    def new_physical_control_sets(self):
        return list(self._new_physical_control_sets)

    def export(self):
        return {
            "round": self._round_number,
            "effective_config": self.effective_config(),
            "role_changes": [change.export() for change in self._role_changes],
            "new_physical_control_sets": self.new_physical_control_sets(),
        }


class ControlSetHistory(object):
    """
    An ordered store of control set snapshots.

    Ingesting a snapshot produces a ControlSetRoundReport describing
    that round.  Physical control set keys are deduplicated across
    rounds and across roles: a key that has been stored once is never
    stored again.  Previously ingested snapshots are not modified, so
    they still resolve to their original physical keys.
    """
    def __init__(self):
        self._reports = []
        self._physical_store = {}

    def ingest(self, snapshot):
        """
        Ingest a ControlSetSnapshot and return its ControlSetRoundReport.
        """
        previous = self._reports[-1].snapshot() if self._reports else None
        role_changes = []
        for role in ControlSetRole:
            current = snapshot.resolution(role)
            old = previous.resolution(role) if previous is not None else None
            change = RoleChange(role, old, current)
            if change.changed():
                role_changes.append(change)

        new_physical = []
        for name in snapshot.physical_control_sets():
            if name not in self._physical_store:
                self._physical_store[name] = len(self._reports)
                new_physical.append(name)

        report = ControlSetRoundReport(len(self._reports), snapshot,
                                       role_changes, new_physical)
        self._reports.append(report)
        return report

    def reports(self):
        return list(self._reports)

    def physical_control_sets(self):
        """
        All physical control set keys stored so far, sorted by name.
        Each key appears exactly once.
        """
        return sorted(self._physical_store)
