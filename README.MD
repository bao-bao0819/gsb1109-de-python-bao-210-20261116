
python-registry
===============

Introduction
------------
python-registry is a pure Python library that provides read-only
access to Windows NT Registry files.
These include NTUSER.DAT, userdiff, and SAM. The interface is two-fold:
a high-level interface suitable for most tasks, and a low level
set of parsing objects and methods which may be used for advanced
study of the Windows NT Registry. This library is compatible with
Windows,Linux, and MacOS, making it accessible and functional
across all major operating systems.

Usage
-----

Most users will find the `Registry.Registry` module most appropriate.
The module exposes three classes: the `Registry`, the `RegistryKey`,
and the `RegistryValue`. The `Registry` organizes parsing and access
to the Windows Registry file. The `RegistryKey` is a convenient
interface into the tree-like structure of the Windows NT Registry.
A `RegistryKey` may have children `RegistryKeys`, and may also have
values associated with it. A `RegistryValue` can be thought of as
the tuple (name, datatype, value) associated with a `RegistryKey`.
python-registry supports all major datatypes, such as `RegSZ`,
`RegDWord`, and `RegBin`.

To open a Windows Registry file, its this easy:


    import sys
    from Registry import Registry
    
    reg = Registry.Registry(sys.argv[1])


Print all keys in a Registry


    def rec(key, depth=0):
        print "\t" * depth + key.path()
    
        for subkey in key.subkeys():
            rec(subkey, depth + 1)
    
    rec(reg.root())


Find a key and print all string values


    try:
        key = reg.open("SOFTWARE\\Microsoft\\Windows\\Current Version\\Run")
    except Registry.RegistryKeyNotFoundException:
        print "Couldn't find Run key. Exiting..."
        sys.exit(-1)
    
    for value in [v for v in key.values() \
                       if v.value_type() == Registry.RegSZ or \
                          v.value_type() == Registry.RegExpandSZ]:
        print "%s: %s" % (value.name(), value.value())


Advanced users who wish to study the structure of the Windows
Registry may find the `Registry.RegistryParse` module useful.
This module implements all known structures of the Windows Registry.

Wanted
------
  - Bug reports.
  - Feedback.

Comparing and migrating names across snapshots
-----------------------------------------------
`Registry.RegistryName` keeps each stored name together with its raw bytes
and storage format (compressed Windows-1252 or UTF-16LE).  Comparison uses
that format and an upcase mapping, not `str.lower`: a name stored compressed
folds only its ASCII letters, while an uncompressed name folds by code point.
A query string has no storage format, so it is considered in both forms.

`Registry.RegistryDiff` builds a rename mapping for one level, either the
sibling subkeys of a key or the values of one key.  Entries are matched by
identity, so a rename does not change which entities correspond.
Destination-only entries act as blockers and are never overwritten.

    import Registry
    from Registry import RegistryDiff

    old = Registry.Registry("old.hive").open("Software\\Vendor")
    new = Registry.Registry("new.hive").open("Software\\Vendor")

    source = RegistryDiff.Snapshot.from_key(old, kind="key")
    destination = RegistryDiff.Snapshot.from_key(new, kind="key")
    plan = RegistryDiff.plan_renames(source, destination=destination)

    for intent in plan.accepted_intents():
        print(intent.old_name.text, "->", intent.new_name.text)
    for group in plan.conflict_groups():
        print("needs manual unbinding:", group.names)
    for component in plan.components():
        print("independent component:", component.is_complete())

The plan reports the minimum number of renames that must be unbound by hand,
groups the rest into components that can be migrated independently, and emits
ordered steps.  A rename cycle is broken with a temporary name rather than
overwriting an entry, and unbound or untouched entries keep their original
Name objects and bytes.  All decisions are made after a canonical sort, so the
order in which intents are supplied does not choose the winner.

License
-------
As of version 0.2.0, python-registry is released under the Apache 2.0 license.
Before that, python-registry was released under the GPLv3.


Sources
-------
Nearly all structure definitions used in python-registry
came from one of two sources:
1) WinReg.txt, by B.H., which may be accessed at:
   http://pogostick.net/~pnh/ntpasswd/WinReg.txt
2) The Windows NT Registry File Format version 0.4, by
   Timothy D. Morgan, which may be accessed at:
   https://docs.google.com/viewer?url=http%3A%2F%2Fsentinelchicken.com%2Fdata%2FTheWindowsNTRegistryFileFormat.pdf
Copies of these resources are included in the
`documentation/` directory of the python-registry source.


The source directory for python-registry contains a `sample/`
subdirectory that contains small programs that use python-registry.
For example, `regview.py` is a read-only clone of Microsoft Window's
Regedit, implemented in a few hundred lines.
