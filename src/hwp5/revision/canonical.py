# -*- coding: utf-8 -*-
#
#   pyhwp : hwp file format parser in python
#   Copyright (C) 2010-2023 mete0r <https://github.com/mete0r>
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Affero General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Affero General Public License for more details.
#
#   You should have received a copy of the GNU Affero General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
'''Canonical, comparable representations of parsed HWP binary models.

The comparison code never compares ``Flags``/``Enum`` instances directly.
Those are integer subclasses whose identity depends on the parser run.  A
canonical value is a plain JSON-compatible value, so the same model always
produces the same fingerprint, even after the DocInfo resource tables have
been reordered.
'''
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from binascii import hexlify
from hashlib import sha1
import json


PY3 = __import__('sys').version_info.major == 3
if PY3:
    integer_types = (int,)
    string_types = (str,)
    binary_type = bytes
else:  # pragma: no cover - preserved for the py2 branch of pyhwp
    integer_types = (int, long)  # noqa: F821
    string_types = (basestring,)  # noqa: F821
    binary_type = str


def _key_sort_key(item):
    return item[0]


def canonical(value):
    '''Return a plain deterministic representation of ``value``.'''
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, string_types):
        return value
    if isinstance(value, binary_type):
        return {'__bytes__': hexlify(value).decode('ascii')}

    # dataio.Flags is an int subclass; keep its named bit fields, otherwise
    # ``head_shape``/``storage`` and friends would collapse into one number.
    dictvalue = getattr(value, 'dictvalue', None)
    if dictvalue is not None:
        try:
            return canonical(dictvalue())
        except TypeError:
            pass

    if isinstance(value, integer_types):
        return int(value)
    if isinstance(value, float):
        return value
    if isinstance(value, dict):
        return dict((str(k), canonical(value[k]))
                    for k in sorted(value, key=str))
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((canonical(item) for item in value),
                      key=lambda item: repr(item))
    try:
        return int(value)
    except (TypeError, ValueError):
        return repr(value)


def canonical_json(value):
    return json.dumps(canonical(value), sort_keys=True,
                      ensure_ascii=False, separators=(',', ':'))


def fingerprint(value):
    data = canonical_json(value).encode('utf-8')
    return sha1(data).hexdigest()


def fingerprint_of(kind, value):
    return fingerprint({'kind': kind, 'value': canonical(value)})


def canonical_diff(old, new, path=''):
    '''Return a list of changed leaf fields between two canonical values.'''
    changes = []
    if type(old) is not type(new):
        # int/bool and dict/list distinctions are meaningful for a report.
        changes.append(dict(path=path or '.', old=old, new=new))
        return changes
    if isinstance(old, dict):
        keys = sorted(set(old) | set(new), key=str)
        for key in keys:
            child = key if not path else path + '.' + key
            if key not in old:
                changes.append(dict(path=child, old=None, new=new[key]))
            elif key not in new:
                changes.append(dict(path=child, old=old[key], new=None))
            else:
                changes.extend(canonical_diff(old[key], new[key], child))
    elif isinstance(old, list):
        if len(old) != len(new):
            changes.append(dict(path=path or '.', old=old, new=new))
        else:
            for index, (a, b) in enumerate(zip(old, new)):
                child = '%s[%d]' % (path or '.', index)
                changes.extend(canonical_diff(a, b, child))
    elif old != new:
        changes.append(dict(path=path or '.', old=old, new=new))
    return changes
