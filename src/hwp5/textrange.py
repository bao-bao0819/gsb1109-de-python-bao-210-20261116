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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


def utf16_len(text):
    ''' number of UTF-16 code units of a unicode string

    HWP paragraph positions -- ParaCharShape and ParaLineSeg boundaries as
    well as the ranges of ParaText chunks -- are counted in UTF-16 code
    units. Python strings instead are sequences of unicode code points:
    characters outside the Basic Multilingual Plane take two code units
    (a high/low surrogate pair) but only one code point.
    '''
    return len(text.encode('utf-16-le')) // 2


def split_utf16(text, units):
    ''' split ``text`` after ``units`` UTF-16 code units

    returns a ``(left, right)`` tuple whose concatenation is always the
    original string. Unlike slicing a python string by code-unit offset,
    a boundary landing in the middle of a surrogate pair is moved forward,
    so the whole supplementary character stays in the left piece. The
    number of code units actually placed in the left piece can be derived
    with :func:`utf16_len`.
    '''
    if units <= 0:
        return '', text
    total = utf16_len(text)
    if units >= total:
        return text, ''

    index = 0
    seen = 0
    while seen < units:
        if ord(text[index]) > 0xFFFF:
            # a supplementary code point is encoded as a high/low
            # surrogate pair, i.e. two UTF-16 code units
            seen += 2
        else:
            seen += 1
        index += 1
    return text[:index], text[index:]
