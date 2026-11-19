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
"""
Export effective configuration values together with free-cell residue.

    python -m Registry.RegistryExport <hive> [--search TEXT] [--format text]

The JSON report contains three evidence classes:

``active``
    Values referenced from the value list of a key that is reachable from the
    hive root, with both the value record and its data in allocated cells.

``residual``
    Value records that survive in unallocated (free) cells.  These are
    observations of old data, never current configuration.

``uncertain``
    Rows whose reachability is indeterminate: a free record whose owner's
    parent chain is broken, an allocated record no live key references, or a
    value referenced by a live key whose cell/data is not allocated.  These
    are not reported as deleted.

Textual paths are not identities.  When a residual row has the same key path
and value name as an active row, both rows are kept and linked through a
``collision`` record; only the ``active`` identity is effective configuration.

Run the CLI with ``--format text`` for a human-readable view.
"""
from __future__ import print_function
from __future__ import absolute_import

import sys

from Registry import RegistryExport


def main(argv=None):
    return RegistryExport.main(argv)


if __name__ == "__main__":
    sys.exit(main())
