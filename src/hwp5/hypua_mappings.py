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
''' Bundled sample Hanyang-PUA mapping tables.

The full Hanyang PUA -> Unicode mapping is a large, separately maintained
data set.  The tables here are small, explicit samples that exercise every
provenance case (one-to-many normalization, unmapped codepoints and
version-to-version changes); real deployments load their own tables via
:func:`hwp5.pua.load_mapping_table`.
'''
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from .pua import MappingEntry
from .pua import PuaMappingTable


#: First sample table version.
SAMPLE_TABLE_V1 = 'sample-hypua/1'

#: Revised sample table version; some mappings are corrected on purpose.
SAMPLE_TABLE_V2 = 'sample-hypua/2'


_ENTRIES_V1 = [
    # U+EBD4 is a Hanyang-PUA Hangul syllable glyph found in HWP text.
    MappingEntry(0xEBD4, u'나', note='hanyang hangul syllable'),
    # A distinct source glyph normalized onto the same business character:
    # this is the one-to-many case (two PUA codepoints -> one syllable).
    MappingEntry(0xEBD7, u'나',
                 note='variant glyph normalized to U+B098'),
    # Symbol glyphs, mirroring hwp5.xmlformat.PUA_SYMBOLS.
    MappingEntry(0xF06C, u'●', note='black circle'),
    MappingEntry(0xF09F, u'•', note='bullet'),
    # U+F100 is intentionally absent: an unmapped codepoint.
]


_ENTRIES_V2 = [
    MappingEntry(0xEBD4, u'나', note='hanyang hangul syllable'),
    # Corrected mapping: the variant glyph is a different syllable.
    MappingEntry(0xEBD7, u'낙',
                 note='corrected: variant glyph is U+B099'),
    MappingEntry(0xF06C, u'●', note='black circle'),
    # Corrected symbol normalization.
    MappingEntry(0xF09F, u'·', note='corrected: middle dot'),
    # Previously unknown, now mapped.
    MappingEntry(0xF100, u'가', note='newly mapped in v2'),
]


def sample_table_v1():
    return PuaMappingTable(_ENTRIES_V1, version=SAMPLE_TABLE_V1)


def sample_table_v2():
    return PuaMappingTable(_ENTRIES_V2, version=SAMPLE_TABLE_V2)


#: Default table used when a caller does not supply one.
def default_table():
    return sample_table_v1()
