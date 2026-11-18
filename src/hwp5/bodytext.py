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
''' Merge paragraph body texts while keeping hyperlink/field range markers.

A hyperlink (or any other range tag, e.g. a field) may start at the beginning
of one paragraph and end in the next paragraph, with other markers nested in
between. When the paragraph texts are merged into one contiguous string, the
information about which text belongs to which link or field must survive:

* every start/end marker is kept, including markers that span paragraph
  boundaries and markers that never get a matching end;
* an inner marker ending must never tear down an enclosing outer marker;
* editing a single paragraph only shifts the markers that are actually
  affected by the real length delta, instead of translating every later node
  by a stale offset.

The module works on plain data so it is independent from the XML event
pipeline; :func:`paragraphs_from_modelevents` adapts the binmodel event stream
to the input shape used here.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import sys

from .binmodel import Paragraph
from .binmodel import ParaText
from .binmodel import ParaRangeTag
from .treeop import STARTEVENT
from .treeop import ENDEVENT


if sys.version_info.major == 3:
    unichr = chr


logger = logging.getLogger(__name__)


def _tag_kind_data(range_tag):
    if 'kind' in range_tag:
        return range_tag['kind'], range_tag.get('data')
    tag = int(range_tag['tag'])
    return tag >> 24, tag & 0x00ffffff


class BodyTextMarker(object):
    ''' A hyperlink/field range over the merged body text.

    ``start`` and ``end`` are character offsets of the merged text.
    ``closed`` is False when the marker starts but never meets its end; such
    unpaired markers are kept instead of being dropped.
    '''

    def __init__(self, kind, key, start, end, tag=None, closed=True):
        self.kind = kind
        self.key = key
        self.start = start
        self.end = end
        self.tag = tag
        self.closed = closed

    def __repr__(self):
        return 'BodyTextMarker(%r, %r, %r, %r, closed=%r)' % (
            self.kind, self.key, self.start, self.end, self.closed)

    def __eq__(self, other):
        if not isinstance(other, BodyTextMarker):
            return NotImplemented
        return (self.kind == other.kind and self.key == other.key
                and self.start == other.start and self.end == other.end
                and self.closed == other.closed)

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __hash__(self):
        return hash((self.kind, self.key, self.start, self.end))

    def copy(self):
        return BodyTextMarker(self.kind, self.key, self.start, self.end,
                              tag=self.tag, closed=self.closed)


class BodyText(object):
    ''' Merged body text plus the markers that annotate it. '''

    def __init__(self, text, markers, paragraph_offsets):
        self.text = text
        self.markers = markers
        self.paragraph_offsets = list(paragraph_offsets)

    @property
    def paragraph_count(self):
        return len(self.paragraph_offsets) - 1

    def paragraph_range(self, paragraph_index):
        ''' Return the (start, end) offsets of one paragraph. '''
        start = self.paragraph_offsets[paragraph_index]
        end = self.paragraph_offsets[paragraph_index + 1]
        return start, end

    def markers_at(self, pos):
        ''' Markers covering ``pos``, outermost first. '''
        return [marker for marker in self.markers
                if marker.start <= pos < marker.end]

    def segments(self):
        ''' Yield ((start, end), text, active markers) pieces. '''
        return split_by_markers(self)


def merge_bodytext(paragraphs):
    ''' Merge paragraph texts and their range tags into a BodyText.

    :param paragraphs: iterable of ``(text, range_tags)`` pairs; each
        ``range_tags`` item is a mapping with ``start``/``end`` paragraph-local
        character offsets and either a ``tag`` value (int or Flags) or
        explicit ``kind``/``data`` entries.
    '''
    merged_texts = []
    markers = []
    # active: end beyond the paragraph length, a continuation is required
    active = {}
    # at_boundary: end equals the paragraph length; the marker may be
    # continued by a tag starting at offset 0 of the next paragraph
    at_boundary = {}
    offsets = [0]
    offset = 0

    for paragraph_index, (text, range_tags) in enumerate(paragraphs):
        if paragraph_index > 0:
            # markers that ended at the previous paragraph boundary without a
            # continuation are closed there
            for marker in at_boundary.values():
                marker.closed = True
            at_boundary = {}

        paragraph_end = offset + len(text)
        for range_tag in range_tags:
            kind, data = _tag_kind_data(range_tag)
            key = (kind, data)
            local_start = int(range_tag['start'])
            local_end = int(range_tag['end'])
            start = offset + local_start
            end = offset + local_end

            if local_start == 0 and key in active:
                marker = active.pop(key)
                marker.end = end
            elif local_start == 0 and key in at_boundary:
                marker = at_boundary.pop(key)
                marker.end = end
            else:
                marker = BodyTextMarker(kind, key, start, end,
                                        tag=range_tag.get('tag'))
                markers.append(marker)

            marker.closed = True
            if end > paragraph_end:
                # the marker runs past this paragraph: without a continuation
                # it stays open (an unpaired marker is never dropped)
                marker.closed = False
                active[key] = marker
            elif end == paragraph_end:
                at_boundary[key] = marker

        merged_texts.append(text)
        offset = paragraph_end
        offsets.append(offset)

    total_length = offset

    # markers ending at the final boundary are closed there; markers still
    # active beyond it have no matching end and remain unpaired
    for marker in at_boundary.values():
        marker.closed = True
    for marker in active.values():
        marker.closed = False

    for marker in markers:
        marker.start = max(0, min(marker.start, total_length))
        marker.end = max(marker.start, min(marker.end, total_length))

    markers.sort(key=lambda marker: (marker.start, -marker.end))
    return BodyText(''.join(merged_texts), markers, offsets)


def split_by_markers(bodytext):
    ''' Split the merged text at every marker boundary.

    Each yielded piece reports the full stack of markers that cover it, so the
    ownership of every character (link, field or none) is known even when
    markers nest or cross paragraph boundaries.
    '''
    text = bodytext.text
    bounds = set([0, len(text)])
    for marker in bodytext.markers:
        bounds.add(marker.start)
        bounds.add(marker.end)
    boundaries = sorted(bounds)

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if start == end:
            continue
        active_markers = [marker for marker in bodytext.markers
                          if marker.start <= start and end <= marker.end]
        yield (start, end), text[start:end], active_markers


def edit_paragraph(bodytext, paragraph_index, start, end, replacement):
    ''' Replace ``text[start:end]`` of one paragraph and fix marker offsets.

    Only nodes after the actual edit point are moved, by the real length
    delta; markers enclosing the edit keep their start (an inner end never
    clears the outer relationship) and merely extend/shorten their end.
    '''
    paragraph_start, paragraph_end = bodytext.paragraph_range(paragraph_index)
    paragraph_length = paragraph_end - paragraph_start
    start = max(0, min(start, paragraph_length))
    end = max(start, min(end, paragraph_length))

    abs_start = paragraph_start + start
    abs_end = paragraph_start + end
    replacement_end = abs_start + len(replacement)
    delta = len(replacement) - (end - start)

    new_text = (bodytext.text[:abs_start] + replacement
                + bodytext.text[abs_end:])

    new_markers = []
    for old_marker in bodytext.markers:
        marker = old_marker.copy()

        # map one endpoint across the edit: points inside the replaced span
        # collapse to the replacement span; later points move by the delta
        if marker.start < abs_start:
            new_start = marker.start
        elif marker.start < abs_end:
            new_start = replacement_end
        else:
            new_start = marker.start + delta

        if marker.end < abs_start:
            new_end = marker.end
        elif marker.end < abs_end:
            new_end = replacement_end
        else:
            new_end = marker.end + delta

        # a pure insertion at a marker start keeps the start boundary:
        # the new characters belong to the marker rather than pushing it
        if (old_marker.start == abs_start
                and old_marker.start < old_marker.end):
            new_start = abs_start

        if new_end <= new_start:
            # nothing of the marker remains after the edit
            logger.info('marker removed by edit: %r', marker)
            continue

        marker.start = new_start
        marker.end = new_end
        new_markers.append(marker)

    new_markers.sort(key=lambda marker: (marker.start, -marker.end))

    new_offsets = list(bodytext.paragraph_offsets)
    for idx in range(paragraph_index + 1, len(new_offsets)):
        new_offsets[idx] += delta

    return BodyText(new_text, new_markers, new_offsets)


def paragraphs_from_modelevents(event_prefixed_mac):
    ''' Collect (text, range_tags) pairs from a binmodel event stream.

    Consumes the same event shape produced by
    :func:`hwp5.xmlmodel.prefix_binmodels_with_event`: Paragraph start/end
    events delimit paragraphs, ParaText supplies the text and ParaRangeTag
    supplies the range markers. Control characters keep one character slot
    (their code point) so that range tag offsets stay aligned.
    '''
    paragraphs = []
    stack = []

    for event, item in event_prefixed_mac:
        model, attributes, context = item

        if model is Paragraph:
            if event == STARTEVENT:
                stack.append(([], []))
            elif event == ENDEVENT and stack:
                text_parts, range_tags = stack.pop()
                paragraphs.append((''.join(text_parts), range_tags))
        elif not stack:
            continue
        elif model is ParaText and event == STARTEVENT:
            text_parts, range_tags = stack[-1]
            for chunk_range, chunk in attributes['chunks']:
                if isinstance(chunk, dict):
                    text_parts.append(unichr(chunk['code']))
                else:
                    text_parts.append(chunk)
        elif model is ParaRangeTag and event == STARTEVENT:
            text_parts, range_tags = stack[-1]
            range_tags.extend(attributes['range_tags'])

    return paragraphs


def merge_bodytext_from_modelevents(event_prefixed_mac):
    return merge_bodytext(paragraphs_from_modelevents(event_prefixed_mac))
