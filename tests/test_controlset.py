import os.path

import pytest

from Registry import Registry
from Registry.ControlSet import (
    ControlSetHistory,
    ControlSetRole,
    ControlSetSnapshot,
    ControlSetUndefinedException,
    UndefinedReason,
)

SYSTEM_HIVE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "testing", "reg_samples", "new_log_1", "SYSTEM")


class FakeValue(object):
    def __init__(self, name, value_type, value):
        self._name = name
        self._value_type = value_type
        self._value = value

    def name(self):
        return self._name

    def value_type(self):
        return self._value_type

    def value(self):
        return self._value


class FakeKey(object):
    def __init__(self, name):
        self._name = name
        self._values = {}
        self._subkeys = {}

    def name(self):
        return self._name

    def add_value(self, name, value_type, value):
        self._values[name.lower()] = FakeValue(name, value_type, value)
        return self

    def add_subkey(self, key):
        self._subkeys[key.name().lower()] = key
        return self

    def subkeys(self):
        return list(self._subkeys.values())

    def value(self, name):
        try:
            return self._values[name.lower()]
        except KeyError:
            raise Registry.RegistryValueNotFoundException(name)

    def find_key(self, path):
        key = self
        for component in [c for c in path.split("\\") if c]:
            try:
                key = key._subkeys[component.lower()]
            except KeyError:
                raise Registry.RegistryKeyNotFoundException(path)
        return key


class FakeRegistry(object):
    def __init__(self, root):
        self._root = root

    def root(self):
        return self._root

    def open(self, path):
        return self._root.find_key(path)


def make_registry(select_values, control_sets):
    root = FakeKey("ROOT")
    if select_values is not None:
        select = FakeKey("Select")
        for name, (value_type, value) in select_values.items():
            select.add_value(name, value_type, value)
        root.add_subkey(select)
    for name in control_sets:
        root.add_subkey(FakeKey(name))
    return FakeRegistry(root)


def dword(value):
    return (Registry.RegDWord, value)


@pytest.fixture
def system_snapshot():
    return ControlSetSnapshot(Registry.Registry(SYSTEM_HIVE))


def test_real_hive_roles_resolve_against_same_select(system_snapshot):
    assert system_snapshot.control_set_name(ControlSetRole.CURRENT) == "ControlSet001"
    assert system_snapshot.control_set_name(ControlSetRole.DEFAULT) == "ControlSet001"
    assert system_snapshot.control_set_name(ControlSetRole.LAST_KNOWN_GOOD) == "ControlSet001"


def test_real_hive_failed_points_to_missing_control_set(system_snapshot):
    resolution = system_snapshot.resolution(ControlSetRole.FAILED)
    assert not resolution.is_defined()
    assert resolution.select_value() == 0
    assert resolution.reason() == UndefinedReason.CONTROL_SET_NOT_FOUND


def test_roles_are_distinct_entities(system_snapshot):
    current = system_snapshot.resolution(ControlSetRole.CURRENT)
    default = system_snapshot.resolution(ControlSetRole.DEFAULT)
    assert current is not default
    config = system_snapshot.effective_config()
    assert set(config["roles"]) == {"Current", "Default", "Failed", "LastKnownGood"}
    assert config["roles"]["Current"] is not config["roles"]["Default"]


def test_shared_physical_key_stored_once(system_snapshot):
    assert system_snapshot.physical_control_sets() == ["ControlSet001"]
    config = system_snapshot.effective_config()
    assert config["physical_control_sets"] == ["ControlSet001"]


def test_virtual_path_translation(system_snapshot):
    assert system_snapshot.translate_path(r"CurrentControlSet\Services\Tcpip") == \
        r"ControlSet001\Services\Tcpip"
    assert system_snapshot.translate_path("currentcontrolset") == "ControlSet001"
    assert system_snapshot.translate_path(r"ControlSet002\Services") == \
        r"ControlSet002\Services"


def test_open_via_virtual_path(system_snapshot):
    virtual = system_snapshot.open(r"CurrentControlSet\Control\ProductOptions")
    physical = system_snapshot.open(r"ControlSet001\Control\ProductOptions")
    assert virtual.path() == physical.path()


def test_missing_select_value_is_undefined_without_fallback():
    registry = make_registry({"Default": dword(1)}, ["ControlSet001"])
    snapshot = ControlSetSnapshot(registry)
    resolution = snapshot.resolution(ControlSetRole.CURRENT)
    assert not resolution.is_defined()
    assert resolution.reason() == UndefinedReason.SELECT_VALUE_MISSING
    with pytest.raises(ControlSetUndefinedException):
        snapshot.translate_path(r"CurrentControlSet\Services")


def test_missing_select_key_makes_all_roles_undefined():
    snapshot = ControlSetSnapshot(make_registry(None, ["ControlSet001"]))
    for role in ControlSetRole:
        resolution = snapshot.resolution(role)
        assert not resolution.is_defined()
        assert resolution.reason() == UndefinedReason.SELECT_KEY_MISSING


def test_non_dword_select_value_is_source_mismatch():
    registry = make_registry({"Current": (Registry.RegSZ, "1")}, ["ControlSet001"])
    snapshot = ControlSetSnapshot(registry)
    resolution = snapshot.resolution(ControlSetRole.CURRENT)
    assert not resolution.is_defined()
    assert resolution.reason() == UndefinedReason.SELECT_VALUE_TYPE_MISMATCH
    with pytest.raises(ControlSetUndefinedException):
        snapshot.open("CurrentControlSet")


def test_select_value_pointing_to_missing_control_set():
    registry = make_registry({"Current": dword(7)}, ["ControlSet001", "ControlSet002"])
    snapshot = ControlSetSnapshot(registry)
    resolution = snapshot.resolution(ControlSetRole.CURRENT)
    assert not resolution.is_defined()
    assert resolution.select_value() == 7
    assert resolution.reason() == UndefinedReason.CONTROL_SET_NOT_FOUND
    assert snapshot.physical_control_sets() == []


def test_switching_current_control_set_exports_round_and_rationale():
    before = ControlSetSnapshot(make_registry(
        {"Current": dword(1), "Default": dword(1)},
        ["ControlSet001", "ControlSet002"]))
    after = ControlSetSnapshot(make_registry(
        {"Current": dword(2), "Default": dword(1)},
        ["ControlSet001", "ControlSet002"]))

    history = ControlSetHistory()
    first = history.ingest(before)
    assert first.round_number() == 0
    assert first.effective_config()["roles"]["Current"]["control_set"] == "ControlSet001"
    assert {change.role() for change in first.role_changes()} == set(ControlSetRole)
    assert first.new_physical_control_sets() == ["ControlSet001"]

    second = history.ingest(after)
    assert second.round_number() == 1
    changes = {change.role(): change for change in second.role_changes()}
    assert set(changes) == {ControlSetRole.CURRENT}
    rationale = changes[ControlSetRole.CURRENT].rationale()
    assert "ControlSet001" in rationale and "ControlSet002" in rationale
    export = second.export()
    assert export["effective_config"]["roles"]["Current"]["control_set"] == "ControlSet002"
    assert export["role_changes"][0]["rationale"] == rationale
    assert second.new_physical_control_sets() == ["ControlSet002"]


def test_old_snapshot_still_points_to_old_physical_keys():
    before = ControlSetSnapshot(make_registry({"Current": dword(1)},
                                              ["ControlSet001", "ControlSet002"]))
    after = ControlSetSnapshot(make_registry({"Current": dword(2)},
                                             ["ControlSet001", "ControlSet002"]))
    history = ControlSetHistory()
    history.ingest(before)
    history.ingest(after)

    assert before.translate_path("CurrentControlSet") == "ControlSet001"
    assert after.translate_path("CurrentControlSet") == "ControlSet002"


def test_physical_keys_not_stored_twice_across_roles_and_rounds():
    shared = ControlSetSnapshot(make_registry(
        {"Current": dword(1), "Default": dword(1), "LastKnownGood": dword(1)},
        ["ControlSet001", "ControlSet002"]))
    switched = ControlSetSnapshot(make_registry(
        {"Current": dword(2), "Default": dword(1)},
        ["ControlSet001", "ControlSet002"]))
    back = ControlSetSnapshot(make_registry(
        {"Current": dword(1), "Default": dword(1)},
        ["ControlSet001", "ControlSet002"]))

    history = ControlSetHistory()
    history.ingest(shared)
    history.ingest(switched)
    third = history.ingest(back)

    assert history.physical_control_sets() == ["ControlSet001", "ControlSet002"]
    assert third.new_physical_control_sets() == []
