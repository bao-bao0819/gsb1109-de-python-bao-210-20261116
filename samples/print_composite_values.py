#!/usr/bin/python
# -*- coding: utf-8 -*-
# Print settings.dat composite values, including fields whose types this
# library does not know yet.  Unknown fields are shown with their type code and
# exact byte range so the enclosing stream and its item count stay verifiable.

from __future__ import print_function
from __future__ import unicode_literals

import sys

from Registry import Registry, SettingsParse


def print_value(value, indent=0):
    pad = " " * indent
    if isinstance(value, SettingsParse.CompositeValue):
        print("%s{%d items, %d bytes%s}" % (
            pad, value.item_count, value.size,
            "" if value.verified else ", INCONSISTENT"))
        for name, item in value.items():
            print("%s  %r:" % (pad, name))
            print_value(item, indent + 4)
        if value.trailing is not None:
            print("%s  <trailing>:" % pad)
            print_value(value.trailing, indent + 4)
    elif isinstance(value, SettingsParse.UnknownCompositeValue):
        print("%sunknown type 0x%X, range=[0x%X,0x%X), %d bytes%s" % (
            pad, value.item_type, value.offset, value.end, value.size,
            ", truncated" if value.truncated else ""))
    else:
        print("%s%r" % (pad, value))


def walk(key):
    for value in key.values():
        parsed = value.value()
        if isinstance(parsed, SettingsParse.CompositeValue):
            print("%s : %s : %s" % (key.path(), value.name(),
                                    value.value_type_str()))
            print_value(parsed, 2)
    for subkey in key.subkeys():
        walk(subkey)


def usage():
    return "  USAGE:\n\t%s <settings.dat>" % sys.argv[0]


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(usage())
        sys.exit(-1)
    walk(Registry.Registry(sys.argv[1]).root())
