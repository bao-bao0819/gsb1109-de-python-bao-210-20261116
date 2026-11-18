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
from __future__ import print_function
from __future__ import unicode_literals
from unittest import TestCase

from hwp5.textrange import split_utf16
from hwp5.textrange import utf16_len


class TestUtf16Len(TestCase):

    def test_bmp_characters_count_one_unit(self):
        self.assertEqual(0, utf16_len(''))
        self.assertEqual(3, utf16_len('abc'))
        self.assertEqual(2, utf16_len('한글'))

    def test_supplementary_characters_count_two_units(self):
        self.assertEqual(2, utf16_len('\U0001F4A1'))
        self.assertEqual(4, utf16_len('a\U0001F4A1b'))


class TestSplitUtf16(TestCase):

    def setUp(self):
        # unit layout: a(0) high(1) low(2) b(3)
        self.text = 'a\U0001F4A1b'

    def test_split_at_ends(self):
        self.assertEqual(('', self.text), split_utf16(self.text, 0))
        self.assertEqual((self.text, ''), split_utf16(self.text, 4))
        self.assertEqual((self.text, ''), split_utf16(self.text, 99))

    def test_split_on_code_point_boundary(self):
        self.assertEqual(('a', '\U0001F4A1b'), split_utf16(self.text, 1))
        self.assertEqual(('a\U0001F4A1', 'b'), split_utf16(self.text, 3))

    def test_split_inside_surrogate_pair_keeps_pair_in_left(self):
        head, tail = split_utf16(self.text, 2)
        self.assertEqual(('a\U0001F4A1', 'b'), (head, tail))
        self.assertEqual(3, utf16_len(head))
        self.assertEqual(self.text, head + tail)

    def test_concatenation_is_lossless(self):
        for units in range(5):
            head, tail = split_utf16(self.text, units)
            self.assertEqual(self.text, head + tail)
