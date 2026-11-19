
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

Rehearsed edits
---------------

`Registry.RegistryEdit` adds a planning layer over an immutable snapshot.
It never rewrites the hive file.  Instead, it validates groups of value
edits against the snapshot, applies only legal and mutually independent
groups, and reports the projected state before anything is committed.

    from Registry import Registry, RegistryEdit

    reg = Registry.Registry("NTUSER.DAT")
    snapshot = RegistryEdit.HiveSnapshot.from_registry(reg)

    plan = RegistryEdit.EditPlan(snapshot)
    plan.add_group(
        RegistryEdit.EditGroup("timezone")
            .rename("TimeZoneInformation", "Bias", "BiasRenamed")
            .change_type("TimeZoneInformation", "BiasRenamed", "RegQWord")
    )

    rehearsal = plan.rehearse()          # dry run only
    result = plan.apply()                # validates and verifies again
    result.verification.raise_if_failed()

The unnamed default value is addressed with
`Registry.Registry.DEFAULT_VALUE` (or the empty stored name).  It is not the
same as a missing value, a value with empty data, or a normal value whose
literal name is `(Default)`:

    key.value("(default)")               # historical friendly spelling
    key.value(Registry.DEFAULT_VALUE)    # explicitly the unnamed value
    key.value_exact("(Default)")         # a literal named value
    value.is_default()                   # True only for the unnamed value

In a snapshot, use `find_value(...)` for exact logical identity and
`resolve(...)` when the friendly `(default)` spelling should address the
unnamed value:

Wanted
------
  - Bug reports.
  - Feedback.

python-registry was originally developed to scratch one of
the author's itches.  Now he hopes it can be of use to
someone outside of his lonely NYC apartment.


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
