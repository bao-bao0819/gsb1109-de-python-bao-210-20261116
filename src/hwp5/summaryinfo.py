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
from uuid import UUID

from .msoleprops import PropertyIdentifier
from .msoleprops import RESERVED_PROPERTIES
from .msoleprops import SUMMARY_INFORMATION_PROPERTIES
from .msoleprops import FILETIME
from .msoleprops import RawTextValue
from .msoleprops import UnreadableValue
from .msoleprops import render_property_value


CLSID_HWP_SUMMARY_INFORMATION = UUID(
    '9fa2b660-1061-11d4-b4c6-006097c09d8c'
)

FMTID_HWP_SUMMARY_INFORMATION = CLSID_HWP_SUMMARY_INFORMATION

HWPPIDSI_DATE_STR = PropertyIdentifier(
    id=0x00000014,
    label='HWPPIDSI_DATE_STR',
)

HWPPIDSI_PARACOUNT = PropertyIdentifier(
    id=0x00000015,
    label='HWPPIDSI_PARACOUNT',
)

HWP_PROPERTIES = RESERVED_PROPERTIES + SUMMARY_INFORMATION_PROPERTIES + (
    HWPPIDSI_DATE_STR,
    HWPPIDSI_PARACOUNT,
)


class HwpSummaryInfoTextFormatter(object):

    def formatTextLines(self, hwpsummaryinfo):
        yield 'Title: {}'.format(
            self._text(self._get(hwpsummaryinfo, 'title'))
        )
        yield 'Subject: {}'.format(
            self._text(self._get(hwpsummaryinfo, 'subject'))
        )
        yield 'Author: {}'.format(
            self._text(self._get(hwpsummaryinfo, 'author'))
        )
        yield 'Keywords: {}'.format(
            self._text(self._get(hwpsummaryinfo, 'keywords'))
        )
        yield 'Comments: {}'.format(
            self._text(self._get(hwpsummaryinfo, 'comments'))
        )
        yield 'Last saved by: {}'.format(
            self._text(self._get(hwpsummaryinfo, 'lastSavedBy'))
        )
        yield 'Revision Number: {}'.format(
            self._text(self._get(hwpsummaryinfo, 'revisionNumber'))
        )
        for line in self._value_lines(
            'Last Printed at', hwpsummaryinfo.lastPrintedTimes()
        ):
            yield line
        for line in self._value_lines(
            'Created at', hwpsummaryinfo.createdTimes()
        ):
            yield line
        for line in self._value_lines(
            'Last saved at', hwpsummaryinfo.lastSavedTimes()
        ):
            yield line
        yield 'Number of pages: {}'.format(
            self._text(self._get(hwpsummaryinfo, 'numberOfPages'))
        )
        for line in self._value_lines(
            'Date', hwpsummaryinfo.dateStrings()
        ):
            yield line
        yield 'Number of paragraphs: {}'.format(
            self._text(self._get(hwpsummaryinfo, 'numberOfParagraphs'))
        )

        yield ''
        yield 'Time sources (kept separate; none overrides another):'
        for record in hwpsummaryinfo.time_sources():
            value = record.get('datetime')
            if value is None:
                value = record.get('value')
            if value is None and 'raw_text' in record:
                value = 'raw-bytes({})=0x{}'.format(
                    record['raw_text']['reason'],
                    record['raw_text']['raw_hex'],
                )
            if value is None and 'unreadable' in record:
                value = 'unreadable({})'.format(
                    record['unreadable']['error'],
                )
            yield '- {} ({}): {} [bytes {}-{}]'.format(
                record['source'],
                record['type'],
                value,
                record['byte_range']['start'],
                record['byte_range']['end'],
            )

    def _value_lines(self, label, properties):
        multiple = len(properties) > 1
        for index, property in enumerate(properties):
            suffix = ' [{}]'.format(index) if multiple else ''
            yield '{}: {}'.format(
                label + suffix, self._text(property.value),
            )

    def _get(self, hwpsummaryinfo, attr):
        try:
            return getattr(hwpsummaryinfo, attr)
        except KeyError:
            return None

    def _text(self, value):
        if isinstance(value, (RawTextValue, UnreadableValue)):
            return render_property_value(value)
        if isinstance(value, FILETIME):
            return str(value.datetime)
        if value is None:
            return ''
        return render_property_value(value)
