#!/usr/bin/env python
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
# Cross-hive installation-history auditor.
#
# Usage:
#   python samples/install_audit.py rules.json HIVE_A HIVE_B \
#       [--label-a NAME] [--label-b NAME] [--json]
#
# rules.json encodes the caller's product codes, uninstall keys, component
# keys and the support/exclusion rules between versions.  See
# samples/install_audit_rules.example.json for the schema.

from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Registry import Registry
from Registry.install_audit import (InstallAuditor, RegistryHiveView,
                                    RuleSet)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Audit software installation history across two offline "
                    "registry hives using caller-supplied support/exclusion "
                    "rules.")
    parser.add_argument("rules", help="JSON rule set")
    parser.add_argument("hive_a", help="first offline registry hive")
    parser.add_argument("hive_b", help="second offline registry hive")
    parser.add_argument("--label-a", default=None,
                        help="display name for the first hive")
    parser.add_argument("--label-b", default=None,
                        help="display name for the second hive")
    parser.add_argument("--json", action="store_true",
                        help="emit a machine-readable JSON report")
    args = parser.parse_args(argv)

    rule_set = RuleSet.from_file(args.rules)
    label_a = args.label_a or os.path.basename(os.path.dirname(args.hive_a)) \
        or "hive_a"
    label_b = args.label_b or os.path.basename(os.path.dirname(args.hive_b)) \
        or "hive_b"

    view_a = RegistryHiveView(Registry.Registry(args.hive_a), label_a)
    view_b = RegistryHiveView(Registry.Registry(args.hive_b), label_b)

    report = InstallAuditor(rule_set, view_a, view_b).run()
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False,
                         sort_keys=True))
    else:
        print(report.to_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
