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
'''Revision comparison for HWPv5 documents.

Typical use::

    from hwp5.revision import compare_hwp5
    comparison = compare_hwp5('old.hwp', 'new.hwp')
    report = comparison.to_dict()

The report separates ``resource_reorder``, ``resource_content_change``, and
``body_change``.  See :mod:`hwp5.revision.compare` for the full model.
'''
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from .compare import AFFECTED_STATUSES
from .compare import BODY_STATUSES
from .compare import STATUS_ADDED
from .compare import STATUS_REFERENCE_CHANGE
from .compare import STATUS_REMAPPED
from .compare import STATUS_REORDER_AFFECTED
from .compare import STATUS_REMOVED
from .compare import STATUS_RESOURCE_CHANGE
from .compare import STATUS_TEXT_CHANGE
from .compare import STATUS_UNCHANGED
from .compare import ParagraphResult
from .compare import RevisionComparison
from .compare import RevisionDocument
from .compare import compare_hwp5
from .compare import compare_snapshots
from .paragraphs import ParagraphSnapshot
from .paragraphs import SectionParagraphs
from .paragraphs import read_sections
from .resources import FONT
from .resources import LANGUAGES
from .resources import Resource
from .resources import ResourceMatch
from .resources import ResourcePair
from .resources import ResourceTable


__all__ = [
    'AFFECTED_STATUSES',
    'BODY_STATUSES',
    'STATUS_ADDED',
    'STATUS_REFERENCE_CHANGE',
    'STATUS_REMAPPED',
    'STATUS_REORDER_AFFECTED',
    'STATUS_REMOVED',
    'STATUS_RESOURCE_CHANGE',
    'STATUS_TEXT_CHANGE',
    'STATUS_UNCHANGED',
    'FONT',
    'LANGUAGES',
    'ParagraphResult',
    'ParagraphSnapshot',
    'Resource',
    'ResourceMatch',
    'ResourcePair',
    'ResourceTable',
    'RevisionComparison',
    'RevisionDocument',
    'SectionParagraphs',
    'compare_hwp5',
    'compare_snapshots',
    'read_sections',
]
