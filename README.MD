
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

Exporting effective configuration vs. residual data
---------------------------------------------------

Keys and values that survive in unallocated (free) cells are not part of the
live configuration. The `Registry.RegistryExport` module builds a report that
keeps the two apart:

    from Registry import RegistryExport

    report = RegistryExport.classify("SYSTEM")
    print(report.to_json())

    for item in report.active:
        print(item["path"], item["value_name"], item["value"])

The same split is available directly on a loaded hive:

    reg = Registry.Registry("SYSTEM")
    active = reg.active_values()
    residual = reg.residual_values()
    uncertain = reg.uncertain_values()

The report has three disjoint lists:

- `active`: values referenced from the value list of a key reachable from the
  hive root, with both the value record and its data in allocated cells. Only
  these describe current configuration.
- `residual`: value records physically located in free cells.
- `uncertain`: rows whose reachability cannot be established, such as a free
  record whose owner key has a broken parent chain or an allocated record that
  no live key references. These are not reported as deleted.

Each row carries an `evidence` list and a `location` mapping with absolute
file, HBIN and cell offsets for the record and its data. A residual row that
shares a textual path and value name with an active row is kept separate and
linked through `collisions`; path equality is never treated as identity.

The same output is available on the command line:

    python -m Registry.RegistryExport SYSTEM --format text
    python -m Registry.RegistryExport SYSTEM --search "Program Files"


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
