#!/usr/bin/python
#
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
# Logical diff of two versions of a registry hive.
#
# Keys and values are matched by their logical, case insensitive path,
# never by cell offset, so a hive that was merely reorganized (cells
# rewritten at new addresses) produces an empty diff instead of a
# delete/recreate of every item.  Duplicate or case-conflicting paths
# are reported as conflicts and are never paired arbitrarily.  Every
# reported change keeps the physical NK/VK record offset from each
# hive version for provenance.
from __future__ import print_function

import sys

from Registry import Registry
from Registry import RegistryDiff


def usage():
    return "  USAGE:\n\t%s <old Windows Registry file> <new Windows Registry file>" % sys.argv[0]


def main():
    if len(sys.argv) != 3:
        print(usage())
        sys.exit(2)

    old_registry = Registry.Registry(sys.argv[1])
    new_registry = Registry.Registry(sys.argv[2])

    diff = RegistryDiff.diff_registries(old_registry, new_registry)
    print("old hive: {0}\nnew hive: {1}".format(sys.argv[1], sys.argv[2]))
    print(RegistryDiff.format_report(diff))

    sys.exit(1 if diff.has_changes() else 0)


if __name__ == "__main__":
    main()
