
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


Layered configuration
---------------------
`Registry.LayeredConfig` resolves an effective configuration from
three layers.  The default priority, from lowest to highest, is
`system`, then `user`, then `policy`.


    from Registry.LayeredConfig import DELETE, LayeredConfig

    config = LayeredConfig()
    config.set("system", "browser.homepage", "https://intranet/", "REG_SZ")
    config.set("user", "browser.homepage", "https://example.com/", "REG_SZ")
    config.set("policy", "browser.homepage", DELETE)

    item = config.effective()["browser.homepage"]
    item.state          # State.DELETED: the policy marker hides both lower values
    item.source_chain   # policy tombstone, then the masked user and system values

    config.withdraw("policy")
    config.effective()["browser.homepage"].value   # "https://example.com/"

Every effective item reports one of four states:

- `ABSENT`: no layer mentioned the name.
- `PRESENT`: the winning layer set a non-empty value.
- `EMPTY`: the winning layer set an explicit empty value (`""`, `b""`,
  `[]`, `{}` or `None`); lower layers are not consulted.
- `DELETED`: a deletion marker hides the lower values.

A deletion marker is data, not a mutation: `effective()` always rebuilds
from the raw entries, so withdrawing or replacing the policy layer
re-reveals the correct lower values instead of reusing a stale overlay.
Names whose sources declare different value types are listed separately
in `effective().conflicts` (each conflict is also returned by
`config.conflicts()`); the winning value is still reported normally.

`DELETE` removes one name.  `DeleteTree("app")` removes a name and every
name below it.  `config.remove(layer, name)` withdraws a single raw entry
without adding a marker, and `config.revoke(layer)` is an alias for
`withdraw(layer)`.  `to_dict()` renders the whole result, including source
chains and conflicts, as JSON-friendly data.


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
