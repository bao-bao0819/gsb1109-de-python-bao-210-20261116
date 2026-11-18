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
'''Paragraph snapshots with references resolved to resource contents.

A paragraph stores only numeric ids.  Two revisions that merely reorder the
DocInfo tables may therefore keep every paragraph byte-for-byte identical
while the meaning of those ids changes completely.  Conversely, a reorder
can be accompanied by rewritten ids that still point to the same content.

``ParagraphSnapshot`` captures the resolved meaning of a paragraph, not its
ids.  It is the unit the comparison uses to decide whether a paragraph was
really affected, and only then to rebuild its language/style index.
'''
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from bisect import bisect_right

from ..charsets import tokenize_unicode_by_lang
from .canonical import canonical
from .canonical import fingerprint_of
from .resources import LANGUAGES
from .resources import Unresolved


def _chunk_marker(chunk):
    if isinstance(chunk, dict):
        if chunk.get('chid'):
            return u'￼{%s}' % chunk['chid']
        code = chunk.get('code')
        return u'￼{%02x}' % code if code is not None else u'￼'
    return chunk


def _text_chunks(paratext):
    if paratext is None:
        return [((0, 0), u'')]
    return paratext['content']['chunks']


def _charshapes(paracharshape):
    if paracharshape is None:
        return [(0, 0)]
    shapes = paracharshape['content'].get('charshapes') or []
    if not shapes:
        return [(0, 0)]
    return [(int(pos), int(charshape_id))
            for pos, charshape_id in shapes]


class TextSpan(object):
    '''A run of text sharing one CharShape reference.'''

    def __init__(self, start, end, text, charshape_id):
        self.start = start
        self.end = end
        self.text = text
        self.charshape_id = charshape_id

    def languages(self):
        result = []
        for lang, text in tokenize_unicode_by_lang(self.text):
            result.append((lang, text))
        return result

    def __repr__(self):
        return '<TextSpan %d:%d %r cs=%r>' % (self.start, self.end,
                                              self.text, self.charshape_id)


class ParagraphSnapshot(object):
    '''One paragraph with all resource references resolved.'''

    def __init__(self, section, ordinal, model, paratext, paracharshape,
                 table):
        self.section = section
        self.ordinal = ordinal
        self.model = model
        self.table = table
        header = model['content']
        self.seqno = model['seqno']
        self.level = model['level']
        self.parashape_id = header.get('parashape_id')
        self.style_id = header.get('style_id')
        self.controlmask = canonical(header.get('controlmask'))
        self.spans = _make_spans(paratext, paracharshape)
        self.text = u''.join(span.text for span in self.spans)
        self.semantic = self._semantic()
        self.fingerprint = fingerprint_of('paragraph', self.semantic)
        self.language_index = self._language_index()

    # -- resolution ---------------------------------------------------------

    def _resolve(self, kind, index):
        resource = self.table.get(kind, index)
        if resource is None:
            return Unresolved(kind, index)
        return self.table.semantic(resource)

    def _span_semantic(self, span):
        charshape = self.table.get('charshape', span.charshape_id)
        if charshape is None:
            charshape_value = Unresolved('charshape', span.charshape_id)
        else:
            charshape_value = self.table.semantic(charshape)

        fonts = {}
        for lang, text in span.languages():
            if charshape is None:
                fonts[lang] = None
                continue
            font = self.table.charshape_font(charshape, lang)
            if font is None:
                fonts[lang] = None
            else:
                fonts[lang] = {
                    'name': font.content.get('name'),
                    'fingerprint': self.table.semantic_fingerprint(font),
                }
        return {
            'range': [span.start, span.end],
            'text': span.text,
            'languages': dict((lang, text) for lang, text
                              in span.languages()),
            'fonts': fonts,
            'charshape': charshape_value,
        }

    def _semantic(self):
        return {
            'text': self.text,
            'parashape': self._resolve('parashape', self.parashape_id),
            'style': self._resolve('style', self.style_id),
            'spans': [self._span_semantic(span) for span in self.spans],
        }

    def _language_index(self):
        '''Per-language view: language -> list of resolved font runs.'''
        index = dict((lang, []) for lang in LANGUAGES)
        for span in self.spans:
            charshape = self.table.get('charshape', span.charshape_id)
            for lang, text in span.languages():
                font = None
                if charshape is not None:
                    font = self.table.charshape_font(charshape, lang)
                index.setdefault(lang, []).append({
                    'text': text,
                    'charshape_id': span.charshape_id,
                    'charshape': self.table.semantic_fingerprint(charshape)
                    if charshape is not None else None,
                    'font': font.content.get('name') if font else None,
                    'font_fingerprint': self.table.semantic_fingerprint(font)
                    if font is not None else None,
                })
        return index

    def style_index(self):
        '''Resolved paragraph style: style, parashape, and span charshapes.'''
        style = self.table.get('style', self.style_id)
        parashape = self.table.get('parashape', self.parashape_id)
        return {
            'style_id': self.style_id,
            'style': self.table.label(style),
            'style_fingerprint': self.table.semantic_fingerprint(style)
            if style is not None else None,
            'parashape_id': self.parashape_id,
            'parashape_fingerprint': self.table.semantic_fingerprint(
                parashape) if parashape is not None else None,
            'spans': [{
                'charshape_id': span.charshape_id,
                'charshape_fingerprint': (
                    self.table.semantic_fingerprint(
                        self.table.get('charshape', span.charshape_id))
                    if self.table.get('charshape',
                                      span.charshape_id) is not None
                    else None),
            } for span in self.spans],
        }

    def numeric_references(self):
        return {
            'parashape_id': self.parashape_id,
            'style_id': self.style_id,
            'charshape_ids': [span.charshape_id for span in self.spans],
        }

    def __repr__(self):
        return '<Paragraph %s#%d %r>' % (self.section, self.ordinal,
                                         self.text[:20])


def _charshape_at(shapes, positions, position):
    index = bisect_right(positions, position) - 1
    if index < 0:
        return shapes[0][1]
    return shapes[index][1]


def _make_spans(paratext, paracharshape):
    shapes = _charshapes(paracharshape)
    positions = [pos for pos, _ in shapes]

    spans = []

    def append(start, end, text, charshape_id):
        if spans and spans[-1].charshape_id == charshape_id and \
                spans[-1].end == start:
            spans[-1].end = end
            spans[-1].text += text
        else:
            spans.append(TextSpan(start, end, text, charshape_id))

    for (start, end), chunk in _text_chunks(paratext):
        if isinstance(chunk, dict):
            # A control marker occupies its own range and never carries text.
            append(start, end, _chunk_marker(chunk),
                   _charshape_at(shapes, positions, start))
            continue

        # A text chunk may span several char shape ranges; split it so that
        # each slice belongs to exactly one CharShape reference.
        boundaries = [start]
        boundaries.extend(pos for pos in positions if start < pos < end)
        boundaries.append(end)
        for slice_start, slice_end in zip(boundaries, boundaries[1:]):
            text = chunk[slice_start - start:slice_end - start]
            append(slice_start, slice_end, text,
                   _charshape_at(shapes, positions, slice_start))

    if not spans:
        charshape_id = shapes[0][1]
        spans.append(TextSpan(0, 0, u'', charshape_id))
    return spans


class SectionParagraphs(object):
    '''All paragraphs of a section, in document order.'''

    def __init__(self, section_index, paragraphs):
        self.section_index = section_index
        self.paragraphs = paragraphs

    @classmethod
    def from_models(cls, section_index, models, table):
        paragraphs = []
        pending = {}
        ordinal = 0
        for model in models:
            name = model['type'].__name__
            if name == 'Paragraph':
                snap = ParagraphSnapshot(section_index, ordinal, model,
                                         pending.get('ParaText'),
                                         pending.get('ParaCharShape'),
                                         table)
                paragraphs.append(snap)
                pending = {}
                ordinal += 1
            elif name in ('ParaText', 'ParaCharShape'):
                pending[name] = model
        return cls(section_index, paragraphs)

    @classmethod
    def from_section(cls, section_index, section, table):
        return cls.from_models(section_index, section.models(), table)

    def __iter__(self):
        return iter(self.paragraphs)

    def __len__(self):
        return len(self.paragraphs)


def read_sections(hwp5file, table):
    sections = []
    for index in hwp5file.bodytext.section_indexes():
        section = hwp5file.bodytext.section(index)
        sections.append(SectionParagraphs.from_section(index, section, table))
    return sections
