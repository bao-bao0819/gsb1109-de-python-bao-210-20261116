
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


Expand REG_EXPAND_SZ against a target machine
----------------------------------------------
`value()` always returns the stored string, so `%USERPROFILE%` is preserved
even when the acquisition host has a value for that variable. To build an
effective view, pass the target machine's environment explicitly; the
acquisition host's environment is never consulted.

    value = reg.open("SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Explorer") \
               .value("User Shell Folders")

    result = value.expand({
        "USERPROFILE": "C:\\Users\\target",
        "SystemRoot": "C:\\Windows",
    })

    print result.raw          # exactly as stored
    print result.effective    # target-machine view
    print result.unresolved   # names left as placeholders
    print result.provenance   # one record per variable occurrence

Variables may refer to other variables. Cycles and excessive nesting stop at
the placeholder instead of expanding forever, and are reported through
`result.cycles` and `result.depth_limited`. Provenance records are never merged
by displayed text: if two different variables resolve to the same string, both
occurrences remain separate records with their own source placeholder and
chain.

Environment values can also be read from a hive key:

    env = reg.open("ControlSet001\\Control\\Session Manager\\Environment") \
             .environment()
    result = value.expand(env)

The standalone helpers live in `Registry.Expand`:

    from Registry import Expand

    Expand.expand_environment_strings("%USERPROFILE%\\Desktop", env)

Only `RegExpandSZ` values are expanded. Other value types are returned
verbatim with status `not_expandable`, even if their text contains `%`. If
more target facts arrive later, a pending result can be re-evaluated without
losing the original:

    result = value.expand({"USERPROFILE": "C:\\Users\\target"})
    result = result.with_environment({"SystemRoot": "C:\\Windows"})

`result.pending` lists the names still awaiting a decision. A variable may also
be marked as known-but-undecided by mapping it to `None`; it stays a
placeholder and is reported as pending rather than silently bound.


Advanced users who wish to study the structure of the Windows
Registry may find the `Registry.RegistryParse` module useful.
This module implements all known structures of the Windows Registry.

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
