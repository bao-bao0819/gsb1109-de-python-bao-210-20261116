import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "samples"))

from Registry import Registry
import installed_software as sw


def make_record(view, name, version, key_name="key", uninstall="uninstall.exe"):
    return {
        "view": view,
        "key_name": key_name,
        "key_path": "ROOT\\%s\\%s" % (key_name, key_name),
        "DisplayName": name,
        "DisplayVersion": version,
        "Publisher": "pub",
        "InstallDate": None,
        "InstallLocation": None,
        "UninstallString": uninstall,
        "QuietUninstallString": None,
    }


class FakeValue(object):
    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value


class FakeKey(object):
    def __init__(self, name, values=None, subkeys=None, path=None):
        self._name = name
        self._values = values or {}
        self._subkeys = subkeys or []
        self._path = path or name

    def name(self):
        return self._name

    def path(self):
        return self._path

    def subkeys(self):
        return self._subkeys

    def value(self, name):
        if name not in self._values:
            raise Registry.RegistryValueNotFoundException(name)
        return FakeValue(self._values[name])


class FakeRegistry(object):
    def __init__(self, views):
        self._views = views

    def open(self, path):
        if path not in self._views:
            raise Registry.RegistryKeyNotFoundException(path)
        return self._views[path]


def test_normalize_name_strips_arch_annotations():
    assert sw.normalize_name("Widget (64-bit)") == sw.normalize_name("Widget (x86)")
    assert sw.normalize_name("Widget") == sw.normalize_name("  WIDGET   ")


def test_correlate_keeps_both_views_as_separate_installs():
    records = [
        make_record("64-bit", "Widget", "2.0", key_name="widget64",
                    uninstall="uninstall64.exe"),
        make_record("32-bit", "Widget (x86)", "1.5", key_name="widget32",
                    uninstall="uninstall32.exe"),
    ]
    software = sw.correlate(records)
    assert len(software) == 1
    installs = list(software.values())[0]["installs"]
    assert len(installs) == 2
    versions = dict((r["view"], r["DisplayVersion"]) for r in installs)
    assert versions == {"64-bit": "2.0", "32-bit": "1.5"}
    uninstalls = dict((r["view"], r["UninstallString"]) for r in installs)
    assert uninstalls == {"64-bit": "uninstall64.exe", "32-bit": "uninstall32.exe"}


def test_correlate_does_not_collapse_same_view_duplicates():
    records = [
        make_record("64-bit", "Widget", "1.0", key_name="widget_a"),
        make_record("64-bit", "Widget", "2.0", key_name="widget_b"),
    ]
    software = sw.correlate(records)
    installs = list(software.values())[0]["installs"]
    assert len(installs) == 2
    assert sorted(r["DisplayVersion"] for r in installs) == ["1.0", "2.0"]


def test_missing_view_is_absent_not_merged():
    records = [make_record("64-bit", "Widget", "2.0")]
    software = sw.correlate(records)
    installs = list(software.values())[0]["installs"]
    assert len(installs) == 1
    assert installs[0]["view"] == "64-bit"


def test_collect_installs_tags_view_and_skips_missing_view():
    entry64 = FakeKey("widget", {"DisplayName": "Widget", "DisplayVersion": "2.0"},
                      path="ROOT\\Uninstall\\widget")
    uninstall64 = FakeKey("Uninstall", subkeys=[entry64])
    registry = FakeRegistry({sw.UNINSTALL_VIEWS["64-bit"]: uninstall64})

    records = sw.collect_installs(registry)
    assert len(records) == 1
    assert records[0]["view"] == "64-bit"
    assert records[0]["DisplayVersion"] == "2.0"
    assert records[0]["InstallDate"] is None
    assert records[0]["key_path"] == "ROOT\\Uninstall\\widget"
