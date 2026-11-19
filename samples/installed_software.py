#!/usr/bin/python

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
#
#   Enumerates installed software from the Uninstall keys of a SOFTWARE
#   hive.  The same product may appear in both the 64-bit (native) view
#   and the 32-bit (Wow6432Node) view; the two registry paths differ only
#   by the Wow6432Node component.  Every collected record keeps its view
#   identity, and software-level correlation is applied afterwards by an
#   explicit rule (normalized DisplayName).  Records are never merged:
#   two views may legitimately carry two different version numbers and
#   two different uninstall strings at the same time, and a missing view
#   only means "not present in that view", never "uninstalled".
from __future__ import print_function
from __future__ import unicode_literals

import re
import sys
from collections import OrderedDict

from Registry import Registry

# View identity is part of the collected data.  The paths are identical
# except for the Wow6432Node component, but they are enumerated and
# reported as two distinct views.
UNINSTALL_VIEWS = OrderedDict([
    ("64-bit", "Microsoft\\Windows\\CurrentVersion\\Uninstall"),
    ("32-bit", "Wow6432Node\\Microsoft\\Windows\\CurrentVersion\\Uninstall"),
])

VALUE_NAMES = [
    "DisplayName",
    "DisplayVersion",
    "Publisher",
    "InstallDate",
    "InstallLocation",
    "UninstallString",
    "QuietUninstallString",
]

# Explicit correlation rules, applied in order to a DisplayName to
# produce the software-level correlation key.  Architecture annotations
# are removed because the same product may label itself differently per
# view; versions are deliberately kept so distinct releases do not
# collapse into one another.
_ARCH_ANNOTATION = re.compile(
    r"\s*\((?:64-bit|32-bit|x64|x86|win64|win32)\)\s*", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


def normalize_name(display_name):
    """
    Explicit correlation rule: case-fold, collapse whitespace, and strip
    architecture annotations such as "(64-bit)" or "(x86)".
    """
    name = _ARCH_ANNOTATION.sub(" ", display_name)
    name = _WHITESPACE.sub(" ", name)
    return name.strip().lower()


def _read_values(uninstall_entry_key):
    values = {}
    for name in VALUE_NAMES:
        try:
            values[name] = uninstall_entry_key.value(name).value()
        except Registry.RegistryValueNotFoundException:
            values[name] = None
    return values


def collect_installs(registry):
    """
    Collect one record per Uninstall subkey, per view.  Each record keeps
    its view identity and full key path.  A view whose Uninstall key is
    absent from the hive is simply skipped.
    """
    records = []
    for view, path in UNINSTALL_VIEWS.items():
        try:
            uninstall_key = registry.open(path)
        except Registry.RegistryKeyNotFoundException:
            continue
        for entry_key in uninstall_key.subkeys():
            record = _read_values(entry_key)
            record["view"] = view
            record["key_name"] = entry_key.name()
            record["key_path"] = entry_key.path()
            records.append(record)
    return records


def correlate(records):
    """
    Group records into software-level entries by the explicit
    correlation rule (normalize_name on DisplayName).  Records are
    appended, never merged: every view keeps its own version number and
    uninstall strings, and a software entry may hold several installs
    from the same view as well as from both views.
    """
    software = OrderedDict()
    for record in records:
        display_name = record.get("DisplayName") or record["key_name"]
        key = normalize_name(display_name)
        entry = software.setdefault(key, {"name": display_name, "installs": []})
        entry["installs"].append(record)
    return software


def _or_dash(value):
    return value if value else "-"


def print_report(software):
    for entry in software.values():
        print(entry["name"])
        for record in entry["installs"]:
            print("  [%s] %s" % (record["view"], record["key_path"]))
            print("    version:    %s" % _or_dash(record["DisplayVersion"]))
            print("    publisher:  %s" % _or_dash(record["Publisher"]))
            print("    uninstall:  %s" % _or_dash(record["UninstallString"]))
        views = set(r["view"] for r in entry["installs"])
        missing = [v for v in UNINSTALL_VIEWS if v not in views]
        if missing:
            print("  note: no %s view entry in this hive (not evidence of uninstall)"
                  % "/".join(missing))
        print("")


def usage():
    return "  USAGE:\n\t%s <SOFTWARE hive>" % sys.argv[0]


def main():
    if len(sys.argv) != 2:
        print(usage())
        sys.exit(-1)

    registry = Registry.Registry(sys.argv[1])
    records = collect_installs(registry)
    print_report(correlate(records))


if __name__ == '__main__':
    main()
