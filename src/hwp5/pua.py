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
''' Versioned, provenance-preserving mapping of Hanyang private-use-area
(PUA) codepoints for Hangul search views.

Historical HWP documents store Hangul in the Unicode Private Use Area
(U+E000..U+F8FF).  Several distinct source glyphs may be normalized onto the
same business character, and a glyph without a known mapping must never be
silently collapsed into a shared placeholder.

This module therefore keeps, for every normalized character:

* the original codepoint(s) it came from,
* the mapping table version that produced the normalization,
* the one-to-many range relation between source codepoints and normalized
  characters, and
* an explicit ``unknown`` marker instead of a shared fallback character.

A :class:`SearchView` built this way can answer a search and trace every hit
back to concrete source glyphs; :func:`remap` rebuilds the view with another
mapping table and :func:`diff_views` reports which hits changed.
'''
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import io
import json
import logging
import sys

PY3 = sys.version_info.major == 3
if PY3:
    unichr = chr
    text_type = str
else:
    text_type = unicode  # noqa: F821

logger = logging.getLogger(__name__)


# Unicode Private Use Area: U+E000..U+F8FF
HANGUL_PUA_START = 0xE000
HANGUL_PUA_END = 0xF8FF

#: Bumped whenever the bundled mapping data changes.
MAPPING_VERSION = 'pyhwp-pua/2026.11-draft1'

#: Sentinel stored in normalized text for unmapped PUA codepoints.
#: It is outside the PUA so it cannot collide with a source glyph, and every
#: unknown codepoint keeps its own distinct replacement (see
#: :func:`unknown_char`); unknown glyphs are never unified with each other.
UNKNOWN_SENTINEL_BASE = 0xE0000


def is_pua_codepoint(codepoint):
    return HANGUL_PUA_START <= codepoint <= HANGUL_PUA_END


def unknown_char(codepoint):
    ''' Distinct replacement character for an *unmapped* PUA codepoint.

    Each unknown codepoint gets a unique private-use character on the
    supplementary PUA plane, so two unknown glyphs never become the same
    business character while the original codepoint stays recoverable.
    '''
    if not is_pua_codepoint(codepoint):
        raise ValueError('not a PUA codepoint: %r' % codepoint)
    return unichr(UNKNOWN_SENTINEL_BASE + (codepoint - HANGUL_PUA_START))


def codepoint_of_unknown(char):
    ''' Recover the original PUA codepoint from an unknown sentinel char. '''
    codepoint = ord(char) - UNKNOWN_SENTINEL_BASE + HANGUL_PUA_START
    if not is_pua_codepoint(codepoint):
        raise ValueError('not an unknown-mapping sentinel: %r' % char)
    return codepoint


def codepoint_text(codepoint):
    return 'U+%04X' % codepoint


def parse_codepoint(text):
    text = text.strip()
    if text.upper().startswith('U+'):
        text = text[2:]
    return int(text, 16)


def coalesce_ranges(codepoints):
    ''' Coalesce sorted codepoints into inclusive ``(start, end)`` ranges. '''
    ranges = []
    for codepoint in sorted(codepoints):
        if ranges and codepoint <= ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], codepoint)
        else:
            ranges.append((codepoint, codepoint))
    return ranges


def format_ranges(ranges):
    return ['U+%04X..U+%04X' % (start, end) for start, end in ranges]


class MappingEntry(object):
    ''' One row of a mapping table: one source codepoint -> one target.

    A normalized target may be fed by several entries (one-to-many); the
    reverse relation is indexed by :class:`PuaMappingTable`.
    '''

    __slots__ = ('source', 'target', 'note')

    def __init__(self, source, target, note=None):
        if not is_pua_codepoint(source):
            raise ValueError('mapping source outside PUA: %r' % source)
        if not isinstance(target, text_type) or len(target) != 1:
            raise ValueError('mapping target must be one character: %r'
                             % target)
        self.source = source
        self.target = target
        self.note = note

    def as_dict(self):
        data = dict(source=codepoint_text(self.source), target=self.target)
        if self.note is not None:
            data['note'] = self.note
        return data


class PuaMappingTable(object):
    ''' An immutable, versioned PUA -> normalized-character mapping table. '''

    def __init__(self, entries, version=MAPPING_VERSION):
        self.version = version
        self._by_source = {}
        reverse = {}
        for entry in entries:
            if entry.source in self._by_source:
                raise ValueError('duplicate mapping source: %s'
                                 % codepoint_text(entry.source))
            self._by_source[entry.source] = entry
            reverse.setdefault(entry.target, set()).add(entry.source)
        self._reverse = dict(
            (target, frozenset(sources))
            for target, sources in reverse.items()
        )

    @property
    def entries(self):
        return [self._by_source[source]
                for source in sorted(self._by_source)]

    def get(self, codepoint):
        return self._by_source.get(codepoint)

    def is_known(self, codepoint):
        return codepoint in self._by_source

    def sources_for_target(self, target):
        ''' All source codepoints normalized onto *target* (one-to-many). '''
        return sorted(self._reverse.get(target, ()))

    def ranges_for_target(self, target):
        ''' Inclusive source-codepoint ranges normalized onto *target*. '''
        return coalesce_ranges(self.sources_for_target(target))

    @property
    def mapped_ranges(self):
        return coalesce_ranges(self._by_source.keys())

    def as_dict(self):
        return dict(
            version=self.version,
            pua_range='U+%04X..U+%04X' % (HANGUL_PUA_START, HANGUL_PUA_END),
            mappings=dict(
                (codepoint_text(entry.source),
                 dict(target=entry.target,
                      **(dict(note=entry.note)
                         if entry.note is not None else dict())))
                for entry in self.entries
            ),
        )

    def dump(self, stream):
        json.dump(self.as_dict(), stream, ensure_ascii=False,
                  indent=2, sort_keys=True)

    def dumps(self):
        return json.dumps(self.as_dict(), ensure_ascii=False,
                          indent=2, sort_keys=True)


def load_mapping_table(stream_or_path, version=None):
    ''' Load a mapping table from the JSON format of :meth:`dump`. '''
    if hasattr(stream_or_path, 'read'):
        data = json.load(stream_or_path)
    else:
        with io.open(stream_or_path, encoding='utf-8') as f:
            data = json.load(f)
    table_version = version or data.get('version') or MAPPING_VERSION
    mappings = data.get('mappings', {})
    entries = []
    for source_text, spec in mappings.items():
        source = parse_codepoint(source_text)
        if isinstance(spec, dict):
            target = spec['target']
            note = spec.get('note')
        else:
            target = spec
            note = None
        entries.append(MappingEntry(source, target, note))
    return PuaMappingTable(entries, version=table_version)


def iter_pua_chars(text):
    ''' Yield ``(index, char, codepoint)`` for PUA characters in *text*. '''
    for index, char in enumerate(text):
        codepoint = ord(char)
        if is_pua_codepoint(codepoint):
            yield index, char, codepoint


def split_pua_ranges(text):
    ''' Split *text* into runs, tagging PUA codepoint spans.

    Returns a list of ``(kind, value)`` where *kind* is ``'text'`` (a plain
    unicode run) or ``'pua'`` (a tuple of PUA codepoints).
    '''
    runs = []
    buf = []

    def flush_text():
        if buf:
            runs.append(('text', ''.join(buf)))
            del buf[:]

    for char in text:
        if is_pua_codepoint(ord(char)):
            flush_text()
            runs.append(('pua', (ord(char),)))
        else:
            buf.append(char)
    flush_text()
    return runs


class MappedChar(object):
    ''' Provenance record for one character of a normalized text. '''

    __slots__ = ('index', 'normalized', 'source_codepoints', 'mapping_version',
                 'unknown', 'note')

    def __init__(self, index, normalized, source_codepoints,
                 mapping_version, unknown=False, note=None):
        self.index = index
        self.normalized = normalized
        self.source_codepoints = tuple(source_codepoints)
        self.mapping_version = mapping_version
        self.unknown = unknown
        self.note = note

    def as_dict(self):
        data = dict(
            index=self.index,
            normalized=self.normalized,
            source_codepoints=[codepoint_text(c)
                               for c in self.source_codepoints],
            mapping_version=self.mapping_version,
            unknown=self.unknown,
        )
        if self.note is not None:
            data['note'] = self.note
        return data


def apply_mapping(text, table):
    ''' Normalize *text* under *table* without losing provenance.

    Returns ``(normalized_text, records)``: the normalized string (in which
    unmapped PUA codepoints become distinct sentinels, never a shared
    placeholder) and a per-character :class:`MappedChar` list aligned by
    index.
    '''
    normalized = []
    records = []
    for index, char in enumerate(text):
        codepoint = ord(char)
        if not is_pua_codepoint(codepoint):
            normalized.append(char)
            records.append(MappedChar(index, char, (), table.version))
            continue
        entry = table.get(codepoint)
        if entry is None:
            replacement = unknown_char(codepoint)
            normalized.append(replacement)
            logger.warning(
                'no mapping for PUA codepoint %s; keeping distinct unknown '
                'sentinel %s', codepoint_text(codepoint),
                codepoint_text(ord(replacement)))
            records.append(MappedChar(
                index, replacement, (codepoint,), table.version,
                unknown=True))
        else:
            normalized.append(entry.target)
            records.append(MappedChar(
                index, entry.target, (codepoint,), table.version,
                note=entry.note))
    return ''.join(normalized), records


class SearchHit(object):
    ''' A search match with a trace back to the source glyphs. '''

    __slots__ = ('start', 'end', 'text', 'records')

    def __init__(self, start, end, text, records):
        self.start = start
        self.end = end
        self.text = text
        self.records = records

    @property
    def source_codepoints(self):
        codepoints = []
        for record in self.records:
            codepoints.extend(record.source_codepoints)
        return tuple(codepoints)

    @property
    def pua_ranges(self):
        return coalesce_ranges(self.source_codepoints)

    def as_dict(self):
        return dict(
            start=self.start,
            end=self.end,
            text=self.text,
            source_codepoints=[codepoint_text(c)
                               for c in self.source_codepoints],
            pua_ranges=format_ranges(self.pua_ranges),
            characters=[record.as_dict() for record in self.records],
        )


class SearchView(object):
    ''' A normalized text with everything needed to trace and re-explain it.

    :ivar source_text: original text with raw PUA codepoints.
    :ivar normalized_text: text normalized under ``mapping_version``.
    :ivar mapping_version: version of the table used to build the view.
    :ivar records: per-character :class:`MappedChar` provenance records.
    :ivar location: optional ``(stream, seqno)`` pointer back into the HWP
        document.
    '''

    def __init__(self, source_text, normalized_text, records,
                 mapping_version, location=None):
        self.source_text = source_text
        self.normalized_text = normalized_text
        self.records = records
        self.mapping_version = mapping_version
        self.location = location

    @property
    def pua_ranges(self):
        ''' One-to-many relation: PUA spans present in this view. '''
        codepoints = [codepoint
                      for record in self.records
                      for codepoint in record.source_codepoints]
        return coalesce_ranges(codepoints)

    @property
    def unknown_codepoints(self):
        return sorted(record.source_codepoints[0]
                      for record in self.records
                      if record.unknown and record.source_codepoints)

    def search(self, query):
        ''' Find *query* in the normalized text; hits keep source glyphs. '''
        if not query:
            return []
        hits = []
        view = self.normalized_text
        start = 0
        while True:
            found = view.find(query, start)
            if found < 0:
                break
            end = found + len(query)
            hits.append(SearchHit(
                found, end, view[found:end],
                self.records[found:end]))
            start = end
        return hits

    def explain_hit(self, hit):
        ''' Human-readable explanation tracing *hit* to source glyphs. '''
        lines = ['%r at %d..%d (mapping %s)'
                 % (hit.text, hit.start, hit.end, self.mapping_version)]
        if self.location is not None:
            lines.append('location: %s seqno=%s' % self.location)
        for record in hit.records:
            if record.source_codepoints:
                sources = ', '.join(codepoint_text(c)
                                    for c in record.source_codepoints)
                kind = 'unknown' if record.unknown else 'mapped'
                lines.append('  %r <- %s [%s] %s'
                             % (record.normalized, sources, kind,
                                record.note or ''))
        if hit.pua_ranges:
            lines.append('source PUA ranges: '
                         + ', '.join(format_ranges(hit.pua_ranges)))
        return '\n'.join(lines)

    def as_dict(self):
        data = dict(
            source_text=self.source_text,
            normalized_text=self.normalized_text,
            mapping_version=self.mapping_version,
            pua_ranges=format_ranges(self.pua_ranges),
            unknown_codepoints=[codepoint_text(c)
                                for c in self.unknown_codepoints],
            characters=[record.as_dict() for record in self.records],
        )
        if self.location is not None:
            data['location'] = dict(stream=self.location[0],
                                    seqno=self.location[1])
        return data

    def dumps(self):
        return json.dumps(self.as_dict(), ensure_ascii=False, indent=2,
                          sort_keys=True)


def build_search_view(text, table, location=None):
    ''' Build a :class:`SearchView` from raw *text* and a mapping *table*. '''
    normalized, records = apply_mapping(text, table)
    return SearchView(text, normalized, records, table.version, location)


def remap(view, table):
    ''' Rebuild *view* under a different mapping *table*.

    The original text is preserved untouched, so rebuilding only changes the
    normalization layer; the result carries the new mapping version.
    '''
    return build_search_view(view.source_text, table, view.location)


def diff_views(old_view, new_view):
    ''' Explain what changes between two views of the same source text.

    Returns a dict with the mapping versions, per-index changes and the
    affected source PUA codepoint ranges.  Used to answer "which hits change
    after swapping the mapping table?".
    '''
    if old_view.source_text != new_view.source_text:
        raise ValueError('views do not describe the same source text')
    changes = []
    affected = set()
    for old, new in zip(old_view.records, new_view.records):
        if old.normalized == new.normalized \
                and old.unknown == new.unknown:
            continue
        affected.update(old.source_codepoints)
        affected.update(new.source_codepoints)
        changes.append(dict(
            index=old.index,
            source_codepoints=[codepoint_text(c)
                               for c in (old.source_codepoints
                                         or new.source_codepoints)],
            old=dict(normalized=old.normalized,
                     mapping_version=old.mapping_version,
                     unknown=old.unknown),
            new=dict(normalized=new.normalized,
                     mapping_version=new.mapping_version,
                     unknown=new.unknown),
        ))
    return dict(
        source_text=old_view.source_text,
        old_mapping_version=old_view.mapping_version,
        new_mapping_version=new_view.mapping_version,
        changed_characters=changes,
        affected_pua_ranges=format_ranges(coalesce_ranges(affected)),
    )


def search_hit_changes(old_view, new_view, query):
    ''' Compare search hits for *query* under two mapping versions.

    Returns ``(old_hits, new_hits, changed)`` where *changed* lists the
    source-codepoint ranges whose normalization moved between versions.
    '''
    old_hits = old_view.search(query)
    new_hits = new_view.search(query)
    diff = diff_views(old_view, new_view)
    changed = diff['affected_pua_ranges']
    return old_hits, new_hits, changed


def iter_paratext_views(hwp5file, table):
    ''' Yield a :class:`SearchView` for every ``ParaText`` in a HWP file.

    Each view records its ``(stream, seqno)`` location so a hit can be traced
    back to the concrete paragraph and source glyph in the document.
    '''
    from .binmodel import ParaText

    for section in hwp5file.bodytext:
        stream = 'BodyText/' + section
        for model in hwp5file.bodytext[section].models():
            if model['type'] is not ParaText:
                continue
            text = ''.join(
                chunk for range_, chunk in model['content']['chunks']
                if isinstance(chunk, text_type)
            )
            if not text:
                continue
            yield build_search_view(
                text, table,
                location=(stream, model.get('seqno')))
