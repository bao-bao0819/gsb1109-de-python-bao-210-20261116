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
''' Region-aware text occurrence collection.

A HWP document stores page headers/footers as control subtrees
(``Header``/``Footer``) that the layout engine repeats on every applicable
page, while the ordinary body text lives in top-level paragraphs. The two
regions may contain identical text with identical character shapes, so text
content alone cannot tell whether an occurrence is a repeated page
decoration or genuine body text.

This module walks the model event stream of an :class:`Hwp5File` and keeps
the structural region evidence (body/header/footer) alongside every text
occurrence, so identical strings are never deduplicated by content.
'''
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import json
import logging

from .binmodel import Footer
from .binmodel import Header
from .binmodel import Paragraph
from .binmodel import Text
from .binmodel.controls import SectionDef
from .treeop import STARTEVENT
from .treeop import ENDEVENT


logger = logging.getLogger(__name__)


REGION_BODY = 'body'
REGION_HEADER = 'header'
REGION_FOOTER = 'footer'

REGION_PAGE_DECORATION = (REGION_HEADER, REGION_FOOTER)

IDENTITY_BODY_TEXT = 'body_text'
IDENTITY_PAGE_DECORATION = 'page_decoration'
IDENTITY_UNKNOWN = 'unknown'

PLACES_NAMES = {
    0: 'both_pages',
    1: 'even_page',
    2: 'odd_page',
}


def _region_for_control(model):
    if model is Header:
        return REGION_HEADER
    if model is Footer:
        return REGION_FOOTER
    return None


def _places_name(flags):
    if flags is None:
        return 'unknown'
    places = getattr(flags, 'places', None)
    name = getattr(places, 'name', None)
    if name is not None:
        return name.lower()
    value = int(places) if places is not None else int(flags)
    return PLACES_NAMES.get(value, 'unknown')


def collect_text_occurrences(events):
    ''' Collect every :class:`Text` occurrence with region evidence.

    :param events: model event stream, e.g. ``Hwp5File.events()``
    :returns: list of occurrence dicts in document order. Each occurrence
              records the text fragment, its structural region
              (``body``/``header``/``footer``), its section index and the
              globally unique ``paragraph_id``.
    '''
    occurrences = []
    region_stack = [REGION_BODY]
    section_index = None
    paragraph_id = None

    for event, item in events:
        model, attrs, context = item
        if event == STARTEVENT:
            region = _region_for_control(model)
            if region is not None:
                region_stack.append(region)
            elif model is SectionDef:
                section_index = attrs.get('section_id')
            elif model is Paragraph:
                paragraph_id = attrs.get('paragraph_id')
            elif model is Text:
                text = attrs.get('text') or ''
                if text:
                    occurrences.append({
                        'text': text,
                        'region': region_stack[-1],
                        'section_index': section_index,
                        'paragraph_id': paragraph_id,
                    })
        elif event == ENDEVENT:
            region = _region_for_control(model)
            if region is not None:
                region_stack.pop()

    return occurrences


def group_occurrences_by_paragraph(occurrences):
    ''' Group text fragment occurrences by their paragraph.

    Language tokenization may split a single visual string into several
    :class:`Text` fragments, so matching is performed on the joined
    paragraph text while each fragment keeps its own identity.
    '''
    paragraphs = []
    index_by_paragraph = {}
    for occurrence in occurrences:
        key = (occurrence['section_index'],
               occurrence['region'],
               occurrence['paragraph_id'])
        if key not in index_by_paragraph:
            index_by_paragraph[key] = len(paragraphs)
            paragraphs.append({
                'region': occurrence['region'],
                'section_index': occurrence['section_index'],
                'paragraph_id': occurrence['paragraph_id'],
                'fragments': [],
            })
        paragraphs[index_by_paragraph[key]]['fragments'].append(occurrence)

    for paragraph in paragraphs:
        paragraph['text'] = ''.join(fragment['text']
                                    for fragment in paragraph['fragments'])
    return paragraphs


def find_in_paragraphs(paragraphs, needle):
    ''' Find paragraphs containing ``needle``; never dedupe by content. '''
    matches = []
    for paragraph in paragraphs:
        text = paragraph['text']
        positions = []
        start = 0
        while True:
            index = text.find(needle, start)
            if index < 0:
                break
            positions.append(index)
            start = index + len(needle)
        if positions:
            matches.append({
                'needle': needle,
                'text': text,
                'region': paragraph['region'],
                'section_index': paragraph['section_index'],
                'paragraph_id': paragraph['paragraph_id'],
                'positions': positions,
                'fragment_count': len(paragraph['fragments']),
            })
    return matches


def collect_header_footer_decorations(events):
    ''' Describe each header/footer control exactly as stored in the model.

    A header is stored once per section but rendered on every page where
    the section's page settings apply; the repetition count is a layout
    concern unavailable at this model level, hence reported separately from
    the single model occurrence.
    '''
    decorations = []
    section_index = None
    current = None

    for event, item in events:
        model, attrs, context = item
        if event == STARTEVENT:
            region = _region_for_control(model)
            if region is not None:
                current = {
                    'control': model.__name__,
                    'region': region,
                    'section_index': section_index,
                    'places': _places_name(attrs.get('flags')),
                    'model_occurrences': 1,
                    'paragraph_ids': [],
                    'texts': [],
                }
                decorations.append(current)
            elif model is SectionDef:
                section_index = attrs.get('section_id')
            elif current is not None and model is Paragraph:
                current['paragraph_ids'].append(attrs.get('paragraph_id'))
            elif current is not None and model is Text:
                text = attrs.get('text') or ''
                if text:
                    current['texts'].append(text)
        elif event == ENDEVENT:
            region = _region_for_control(model)
            if region is not None and current is not None \
                    and current['region'] == region:
                current['text'] = ''.join(current['texts'])
                del current['texts']
                current = None

    for decoration in decorations:
        decoration['rendering_note'] = (
            'stored once in the model; the layout engine repeats it on '
            'every applicable page, so rendered page positions are not '
            'enumerable at this model level'
        )
    return decorations


def classify_matches(matches):
    ''' Assign an identity to each match with region evidence.

    Matches with a known region get a definite identity. Matches whose
    region evidence is missing or unknown stay candidates, and the totals
    for each identity are therefore reported as min/max ranges.
    '''
    classified = []
    counts = {
        IDENTITY_BODY_TEXT: {'definite': 0, 'possible': 0},
        IDENTITY_PAGE_DECORATION: {'definite': 0, 'possible': 0},
    }

    for match in matches:
        region = match['region']
        if region in REGION_PAGE_DECORATION:
            identity = IDENTITY_PAGE_DECORATION
            candidates = [IDENTITY_PAGE_DECORATION]
            counts[identity]['definite'] += 1
        elif region == REGION_BODY:
            identity = IDENTITY_BODY_TEXT
            candidates = [IDENTITY_BODY_TEXT]
            counts[identity]['definite'] += 1
        else:
            identity = IDENTITY_UNKNOWN
            candidates = [IDENTITY_BODY_TEXT, IDENTITY_PAGE_DECORATION]
            for candidate in candidates:
                counts[candidate]['possible'] += 1

        classified.append(dict(match, identity=identity,
                               candidate_identities=candidates))

    summary = {}
    for identity, count in counts.items():
        definite = count['definite']
        possible = count['possible']
        summary[identity] = {
            'min_count': definite,
            'max_count': definite + possible,
            'definite_count': definite,
            'uncertain_count': possible,
        }
    return classified, summary


def build_report(hwp5file, needle):
    ''' Build the full region-aware report for ``needle``. '''
    events = list(hwp5file.events())

    occurrences = collect_text_occurrences(iter(events))
    paragraphs = group_occurrences_by_paragraph(occurrences)
    matches = find_in_paragraphs(paragraphs, needle)
    classified, identity_summary = classify_matches(matches)
    decorations = collect_header_footer_decorations(iter(events))

    decoration_hits = [
        {
            'control': decoration['control'],
            'region': decoration['region'],
            'section_index': decoration['section_index'],
            'paragraph_ids': decoration['paragraph_ids'],
            'places': decoration['places'],
            'model_occurrences': decoration['model_occurrences'],
        }
        for decoration in decorations
        if needle in decoration.get('text', '')
    ]

    return {
        'needle': needle,
        'total_paragraph_matches': len(classified),
        'identity_summary': identity_summary,
        'matches': classified,
        'page_decorations_containing_needle': decoration_hits,
        'header_footer_decorations': decorations,
        'note': (
            'identical text is kept per occurrence; header/footer text is '
            'stored once per section and repeated by the layout engine on '
            'each applicable page'
        ),
    }


def report_to_json(report):
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
