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
'''Inspect and remove memo annotations from HWP5 fragment archives.

A fragment archive is a directory containing the decompressed internal
streams of a HWP5 document (``FileHeader``, ``DocInfo``,
``BodyText/SectionN`` and optional ``BinData/*`` files) together with a
``manifest.json`` describing the selected paragraphs and resources.

Memo annotations are represented in a paragraph text record by a pair of
inline control characters:

* ``FIELD_START`` (0x03) with a memo CHID (``%%me``)
* ``FIELD_END`` (0x04) whose CHID is the first start CHID character
  replaced by NUL (``\\x00%me``)

The actual note text is stored in a ``HWPTAG_MEMO_LIST`` record and is
referenced by the field ``id`` (and, in some producers, by the control
character parameter).  The text between the two inline markers is ordinary
document body text and is deliberately left untouched.

The module is conservative.  It refuses to remove an annotation when its
markers cannot be paired unambiguously, when its range crosses another
annotation range, or when the note body is still referenced by another
annotation.  Independent annotations can still be removed.
'''
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
from io import BytesIO
import json
import os
import struct

from .binmodel.controlchar import CHID
from .binmodel.tagid51_para_text import ParaTextChunks
from .dataio import BSTR
from .dataio import UINT32
from .recordstream import encode_record_header
from .tagids import HWPTAG_BIN_DATA
from .tagids import HWPTAG_CTRL_HEADER
from .tagids import HWPTAG_MEMO_LIST
from .tagids import HWPTAG_PARA_CHAR_SHAPE
from .tagids import HWPTAG_PARA_HEADER
from .tagids import HWPTAG_PARA_LINE_SEG
from .tagids import HWPTAG_PARA_RANGE_TAG
from .tagids import HWPTAG_PARA_TEXT
from .tagids import HWPTAG_SHAPE_COMPONENT_OLE
from .tagids import HWPTAG_SHAPE_COMPONENT_PICTURE


FRAGMENT_FORMAT = 'hwp5-fragment/1'
ANNOTATION_FORMAT = 'hwp5-annotation-inventory/1'

MEMO_CHID = '%%me'
MEMO_END_CHID = '\x00%me'

MESSAGE_NEWLINE = '\n'


FIELD_START = 0x03
FIELD_END = 0x04


class FragError(Exception):
    '''Base class for fragment/annotation errors.'''


class FragmentFormatError(FragError):
    '''The supplied archive is not a readable HWP5 fragment.'''


class SelectionError(FragError):
    '''The requested annotation selection is not valid.'''


def _unichr(value):
    try:
        return unichr(value)
    except NameError:
        return chr(value)


def _read_uint32(payload, offset=0):
    if offset + 4 > len(payload):
        return None
    return struct.unpack_from('<I', payload, offset)[0]


def _read_uint16(payload, offset=0):
    if offset + 2 > len(payload):
        return None
    return struct.unpack_from('<H', payload, offset)[0]


def chid_from_payload(payload):
    '''Return the CHID stored at the beginning of a control payload.'''
    if payload is None or len(payload) < 4:
        return None
    return CHID.decode(payload[:4])


def _end_chid_for(start_chid):
    return '\x00' + start_chid[1:]


# ---------------------------------------------------------------------------
# fragment archive access
# ---------------------------------------------------------------------------

class FragmentArchive(object):
    '''A directory-based HWP5 fragment archive.

    ``streams`` maps a POSIX-style relative path (``BodyText/Section0``) to
    the raw bytes of that stream.  ``manifest`` keeps the original manifest
    so unknown producer metadata survives a round trip.
    '''

    def __init__(self, streams, manifest=None, source=None):
        self.streams = streams
        self.manifest = manifest or {}
        self.source = source

    def section_paths(self):
        names = [name for name in self.streams
                 if name.startswith('BodyText/') or
                 name.startswith('ViewText/')]
        names.sort()
        return names

    def section_index(self, path):
        name = path.rsplit('/', 1)[-1]
        if name.startswith('Section'):
            try:
                return int(name[len('Section'):])
            except ValueError:
                pass
        return 0

    def copy(self):
        clone = dict((k, bytes(v)) for k, v in self.streams.items())
        return FragmentArchive(clone, json.loads(json.dumps(self.manifest)),
                               self.source)

    def save(self, destination):
        if not os.path.isdir(destination):
            os.makedirs(destination)
        # A reused destination must not retain streams that are absent from
        # this archive, otherwise stale body/memo records would reappear.
        for dirpath, dirnames, filenames in os.walk(destination):
            dirnames[:] = [d for d in dirnames
                           if not d.startswith('.') and d != '__pycache__']
            for filename in filenames:
                if filename == 'manifest.json':
                    continue
                full = os.path.join(dirpath, filename)
                rel = os.path.relpath(full, destination).replace(os.sep, '/')
                if rel not in self.streams:
                    os.remove(full)
        for name, data in self.streams.items():
            path = os.path.join(destination, name.replace('/', os.sep))
            parent = os.path.dirname(path)
            if parent and not os.path.isdir(parent):
                os.makedirs(parent)
            with open(path, 'wb') as f:
                f.write(data)
        manifest_path = os.path.join(destination, 'manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2, sort_keys=True)
            f.write('\n')


def _is_fragment_directory(path):
    return (os.path.isdir(path) and
            os.path.exists(os.path.join(path, 'FileHeader')) and
            os.path.isdir(os.path.join(path, 'BodyText')))


def _read_directory_streams(root):
    streams = {}
    for dirpath, dirnames, filenames in os.walk(root):
        # Do not recurse into unrelated hidden or output directories.
        dirnames[:] = [d for d in dirnames
                       if not d.startswith('.') and d != '__pycache__']
        for filename in filenames:
            if filename == 'manifest.json':
                continue
            full = os.path.join(dirpath, filename)
            rel = os.path.relpath(full, root).replace(os.sep, '/')
            with open(full, 'rb') as f:
                streams[rel] = f.read()
    return streams


def _read_ole_streams(source):
    from .filestructure import Hwp5File
    from .storage import iter_storage_leafs

    hwp5file = Hwp5File(source)
    streams = {}
    for path in iter_storage_leafs(hwp5file):
        item = hwp5file
        for name in path.split('/'):
            item = item[name]
        f = item.open()
        try:
            streams[path] = f.read()
        finally:
            f.close()
    return streams


def open_fragment(source):
    '''Open a fragment directory or a HWP5 OLE document.

    OLE input is read through its decompressed stream layer and returned as
    an in-memory fragment.  A fragment directory is the canonical archive
    copy used for edits.
    '''
    if _is_fragment_directory(source):
        streams = _read_directory_streams(source)
        manifest = {}
        manifest_path = os.path.join(source, 'manifest.json')
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        return FragmentArchive(streams, manifest, source=source)

    if os.path.isfile(source):
        return FragmentArchive(_read_ole_streams(source), {}, source=source)

    raise FragmentFormatError('not a fragment directory or HWP5 file: %s'
                              % (source,))


# ---------------------------------------------------------------------------
# record stream primitives
# ---------------------------------------------------------------------------

class Record(object):
    __slots__ = ('tagid', 'level', 'size', 'payload', 'header_start',
                 'payload_start', 'end', 'seqno')

    def __init__(self, **kwargs):
        for name in self.__slots__:
            setattr(self, name, kwargs.get(name))

    def raw(self, data):
        return data[self.header_start:self.end]

    def encode(self):
        rec = dict(tagid=self.tagid, level=self.level, payload=self.payload)
        return encode_record_header(rec) + self.payload


def iter_records(data):
    '''Yield :class:`Record` objects with their original byte offsets.'''
    pos = 0
    seqno = 0
    length = len(data)
    while pos + 4 <= length:
        header_start = pos
        header = struct.unpack_from('<I', data, pos)[0]
        pos += 4
        tagid = header & 0x3ff
        level = (header >> 10) & 0x3ff
        size = (header >> 20) & 0xfff
        if size == 0xfff:
            if pos + 4 > length:
                break
            size = struct.unpack_from('<I', data, pos)[0]
            pos += 4
        payload_start = pos
        end = pos + size
        if end > length:
            raise FragmentFormatError(
                'truncated record %d (needs %d bytes, has %d)'
                % (seqno, size, length - pos))
        yield Record(tagid=tagid, level=level, size=size,
                     payload=data[payload_start:end],
                     header_start=header_start, payload_start=payload_start,
                     end=end, seqno=seqno)
        pos = end
        seqno += 1


def dump_records(records):
    return b''.join(record.encode() for record in records)


# ---------------------------------------------------------------------------
# paragraph / marker analysis
# ---------------------------------------------------------------------------

class ParagraphGroup(object):
    def __init__(self, index, records):
        self.index = index
        self.records = records
        self.header = records[0]
        self.paratext = None
        self.controls = []
        self.by_tag = defaultdict(list)
        for record in records:
            self.by_tag[record.tagid].append(record)
            if record.tagid == HWPTAG_PARA_TEXT and self.paratext is None:
                self.paratext = record
            elif record.tagid == HWPTAG_CTRL_HEADER:
                self.controls.append(record)

    @property
    def starts_paragraph(self):
        return self.header.tagid == HWPTAG_PARA_HEADER


class Marker(object):
    def __init__(self, paragraph, record, chunk_start, chunk_end, payload,
                 chunk):
        self.paragraph = paragraph
        self.record = record
        self.chunk_start = chunk_start
        self.chunk_end = chunk_end
        self.chunk = chunk
        self.chid = chunk.get('chid')
        self.code = chunk.get('code')
        self.param = chunk.get('param', b'')
        self.control = None
        self.field = None
        self.field_id = None
        self.body_candidates = []

    @property
    def key(self):
        return (self.paragraph.index, self.chunk_start)

    def byte_range(self):
        return (self.record.payload_start + self.chunk_start * 2,
                self.record.payload_start + self.chunk_end * 2)

    def to_dict(self, section=None):
        return {
            'section': section,
            'paragraph': self.paragraph.index,
            'paragraph_seqno': self.paragraph.header.seqno,
            'record_seqno': self.record.seqno,
            'code': self.code,
            'chid': self.chid,
            'code_units': [self.chunk_start, self.chunk_end],
            'byte_range': list(self.byte_range()),
            'field_id': self.field_id,
            'body_candidates': list(self.body_candidates),
            'control_seqno': (self.control.seqno
                              if self.control is not None else None),
        }


def split_paragraphs(records):
    '''Split a record stream into level-0 groups.

    HWP requires paragraph header records to be level 0.  Records between
    two paragraph headers belong to the first paragraph even when they are
    controls or nested list records.
    '''
    groups = []
    current = []
    for record in records:
        if record.level == 0 and current:
            groups.append(ParagraphGroup(len(groups), current))
            current = []
        current.append(record)
    if current:
        groups.append(ParagraphGroup(len(groups), current))
    return groups


def _parse_field_control(payload):
    if payload is None or len(payload) < 4:
        return None
    result = {'chid': chid_from_payload(payload)}
    stream = BytesIO(payload[4:])
    try:
        result['flags'] = UINT32.read(stream)
        result['extra_attr'] = struct.unpack('<B', stream.read(1))[0]
        result['command'] = BSTR.read(stream)
        result['id'] = UINT32.read(stream)
    except Exception:
        return result
    return result


def _control_chid(record):
    return chid_from_payload(record.payload)


def _match_controls(paragraph):
    '''Associate inline extended control chunks with their control records.'''
    controls = list(paragraph.controls)
    used = set()
    matches = {}

    if paragraph.paratext is None:
        return matches

    chunks = list(ParaTextChunks.parse_chunks(paragraph.paratext.payload))
    for index, (bounds, chunk) in enumerate(chunks):
        if not isinstance(chunk, dict) or 'chid' not in chunk:
            continue
        for control in controls:
            if id(control) in used:
                continue
            if _control_chid(control) == chunk.get('chid'):
                matches[index] = control
                used.add(id(control))
                break
    return matches


def _decode_chunk_text(chunk):
    if isinstance(chunk, str):
        return chunk
    if not isinstance(chunk, dict):
        return ''
    code = chunk.get('code')
    if code == 0x09:
        return '\t'
    if code in (0x0a, 0x0d):
        return '\n'
    if code in (0x1e, 0x1f):
        return ' '
    return ''


def paragraph_text_chunks(paragraph):
    if paragraph.paratext is None:
        return []
    return list(ParaTextChunks.parse_chunks(paragraph.paratext.payload))


def paragraph_text(paragraph):
    pieces = []
    for bounds, chunk in paragraph_text_chunks(paragraph):
        pieces.append(_decode_chunk_text(chunk))
    return ''.join(pieces)


def _extract_range(paragraph, start, end):
    '''Extract text from a paragraph between two code-unit positions.'''
    if end <= start:
        return ''
    pieces = []
    for (chunk_start, chunk_end), chunk in paragraph_text_chunks(paragraph):
        if chunk_end <= start:
            continue
        if chunk_start >= end:
            break
        left = max(start, chunk_start)
        right = min(end, chunk_end)
        if left >= right:
            continue
        if isinstance(chunk, str):
            pieces.append(chunk[left - chunk_start:right - chunk_start])
        else:
            text = _decode_chunk_text(chunk)
            if text:
                pieces.append(text * max(0, right - left))
    return ''.join(pieces)


def _collect_markers(archive):
    sections = []
    for path in archive.section_paths():
        data = archive.streams[path]
        records = list(iter_records(data))
        paragraphs = split_paragraphs(records)
        section_index = archive.section_index(path)
        section = {
            'path': path,
            'index': section_index,
            'data': data,
            'records': records,
            'paragraphs': paragraphs,
            'markers': [],
        }
        for paragraph in paragraphs:
            if not paragraph.starts_paragraph:
                continue
            control_matches = _match_controls(paragraph)
            chunks = paragraph_text_chunks(paragraph)
            for chunk_index, (bounds, chunk) in enumerate(chunks):
                if not isinstance(chunk, dict):
                    continue
                if chunk.get('code') not in (FIELD_START, FIELD_END):
                    continue
                if chunk.get('code') == FIELD_END:
                    if chunk.get('chid') != MEMO_END_CHID:
                        # A non-memo field end is irrelevant, but keep it in
                        # the stream untouched.
                        continue
                if chunk.get('code') == FIELD_START:
                    if chunk.get('chid') != MEMO_CHID:
                        continue
                marker = Marker(paragraph, paragraph.paratext,
                                bounds[0], bounds[1],
                                paragraph.paratext.payload, chunk)
                if chunk.get('code') == FIELD_START:
                    control = control_matches.get(chunk_index)
                    marker.control = control
                    if control is not None:
                        marker.field = _parse_field_control(control.payload)
                        if marker.field:
                            marker.field_id = marker.field.get('id')
                    if marker.param:
                        for offset in (0, 4):
                            candidate = _read_uint32(marker.param, offset)
                            if candidate:
                                marker.body_candidates.append(candidate)
                    if marker.field_id:
                        marker.body_candidates.insert(0, marker.field_id)
                section['markers'].append(marker)
        sections.append(section)
    return sections


def _marker_ids(marker):
    '''Return non-zero 32-bit ids carried in a marker parameter block.'''
    values = []
    param = marker.param or b''
    for offset in (0, 4):
        value = _read_uint32(param, offset)
        if value and value not in values:
            values.append(value)
    if marker.field_id and marker.field_id not in values:
        values.append(marker.field_id)
    return values


def _pair_markers(section):
    '''Pair FIELD_START/FIELD_END markers.

    HWP writes memo markers with the same four-character CHID.  When the
    marker parameter (or the field control) carries an id, that id is used
    first, which makes nested annotations unambiguous.  Without ids the
    standard LIFO field semantics are used; crossing ranges are refused
    later by range comparison.
    '''
    pairs = []
    unmatched_ends = []
    unmatched_starts = []
    ambiguous = []

    stack = []
    for marker in section['markers']:
        if marker.code == FIELD_START:
            stack.append(marker)
            continue

        if not stack:
            unmatched_ends.append(marker)
            continue

        end_ids = _marker_ids(marker)
        candidates = []
        if end_ids:
            candidates = [
                candidate for candidate in stack
                if set(_marker_ids(candidate)) & set(end_ids)
            ]

        if len(candidates) == 1:
            start = candidates[0]
            stack.remove(start)
            pairs.append((start, marker))
            continue
        if len(candidates) > 1:
            ambiguous.append({
                'kind': 'multiple_field_candidates',
                'markers': [candidate.to_dict(section['index'])
                            for candidate in candidates] +
                           [marker.to_dict(section['index'])],
                'reasons': [
                    'field end matches more than one open field',
                ],
            })
            for start in candidates:
                if start in stack:
                    stack.remove(start)
            continue

        if stack[-1].chid == marker.chid:
            start = stack.pop()
            pairs.append((start, marker))
            continue

        # The top of the stack does not match.  Finding a unique lower match
        # means the intervening starts were never closed, which has two
        # readings; refuse all involved ranges.
        candidates = [candidate for candidate in stack
                      if marker.chid == _end_chid_for(candidate.chid)]
        if len(candidates) == 1:
            start = candidates[0]
            index = stack.index(start)
            involved = stack[index:]
            del stack[index:]
            pairs.append((start, marker))
            if len(involved) > 1:
                ambiguous.append({
                    'kind': 'unclosed_nested_field',
                    'markers': [item.to_dict(section['index'])
                                for item in involved] +
                               [start.to_dict(section['index']),
                                marker.to_dict(section['index'])],
                    'reasons': [
                        'field end skips a nested field start',
                        'enclosing and nested ranges have two readings',
                    ],
                })
        elif len(candidates) > 1:
            ambiguous.append({
                'kind': 'multiple_field_candidates',
                'markers': [candidate.to_dict(section['index'])
                            for candidate in candidates] +
                           [marker.to_dict(section['index'])],
                'reasons': [
                    'field end matches more than one open field',
                ],
            })
            for start in candidates:
                if start in stack:
                    stack.remove(start)
        else:
            unmatched_ends.append(marker)

    unmatched_starts.extend(stack)
    return pairs, unmatched_starts, unmatched_ends, ambiguous


def _manifest_storage_id(value):
    '''Normalize a manifest storage id.

    HWP5 fragment manifests may store either a plain integer or a remapping
    object ``{"fragment": N, "original": M}`` after a document is folded
    into a fragment.  The record stream always uses the *fragment* id.
    '''
    if isinstance(value, dict):
        return value.get('fragment')
    return value


def _storage_id_aliases(value):
    '''Return every numeric id that denotes the same BinData resource.'''
    if isinstance(value, dict):
        ids = [value.get('fragment'), value.get('original')]
    else:
        ids = [value]
    return [item for item in ids if item is not None]


def _docinfo_bindata_map(archive):
    '''Map storage IDs from embedded BinData records to stream names.'''
    result = {}
    docinfo = archive.streams.get('DocInfo')
    if not docinfo:
        return result
    for record in iter_records(docinfo):
        if record.tagid != HWPTAG_BIN_DATA:
            continue
        payload = record.payload
        if len(payload) < 2:
            continue
        flags = _read_uint16(payload, 0) or 0
        storage_type = flags & 0x0f
        if storage_type != 1:  # embedding
            continue
        storage_id = _read_uint16(payload, 2)
        ext = ''
        if len(payload) >= 4:
            try:
                stream = BytesIO(payload[4:])
                ext = BSTR.read(stream)
            except Exception:
                ext = ''
        if storage_id is None:
            continue
        result[storage_id] = 'BIN%04X.%s' % (storage_id, ext)
    return result


def _record_bindata_ids(record, version=None):
    ids = []
    if record.tagid == HWPTAG_SHAPE_COMPONENT_PICTURE:
        if len(record.payload) >= 73:
            ids.append(_read_uint16(record.payload, 71))
        else:
            try:
                from .binmodel import ShapePicture
                from .bintype import read_type
                content = read_type(ShapePicture, {'version': version},
                                    BytesIO(record.payload))
                picture = content.get('picture') or {}
                if picture.get('bindata_id'):
                    ids.append(picture['bindata_id'])
            except Exception:
                pass
    elif record.tagid == HWPTAG_SHAPE_COMPONENT_OLE:
        if len(record.payload) >= 14:
            ids.append(_read_uint16(record.payload, 12))
    return [value for value in ids if value]


def _resource_refs(records, version=None):
    refs = []
    seen = set()
    for record in records:
        for storage_id in _record_bindata_ids(record, version):
            if storage_id in seen:
                continue
            seen.add(storage_id)
            refs.append({'kind': 'bindata', 'storage_id': storage_id})
    return refs


def _body_record_spans(records):
    '''Return memo-list record spans.

    A memo list record owns every following record with a deeper level.
    A list record may be followed by sibling list records, in which case the
    span stops at the first record with the same or a lower level.
    '''
    spans = []
    for index, record in enumerate(records):
        if record.tagid != HWPTAG_MEMO_LIST:
            continue
        end = index + 1
        while end < len(records):
            if records[end].level <= record.level:
                break
            end += 1
        spans.append((index, end, record))
    return spans


def _body_id(record, span_records):
    '''Derive the annotation id that a memo-list record belongs to.

    The exact MEMO_LIST payload is not documented in the public HWP 5.0
    specification.  Producers in the wild use the first 32-bit word as the
    memo id, and the field control uses the same value in its ``id`` member.
    When the record is short, fall back to the record sequence number so an
    annotation still has a stable identity for reporting.
    '''
    candidates = []
    if len(record.payload) >= 4:
        value = _read_uint32(record.payload, 0)
        if value:
            candidates.append(value)
    if not candidates:
        candidates.append(record.seqno)
    return candidates[0]


def _match_bodies_to_annotations(bodies, annotations):
    '''Fallback association for memo bodies without an explicit ID.

    Some producers store the field control before the memo list and omit a
    usable id in the list payload.  In that case pair unmatched bodies and
    annotations in document order, but only when the counts agree.  Any
    surplus leaves both sides unassociated rather than guessing.
    '''
    unmatched_annotations = [a for a in annotations
                             if a.body is None and a.body_candidates]
    unmatched_bodies = [b for b in bodies if not b.annotation_refs]
    if len(unmatched_annotations) != 1 or len(unmatched_bodies) != 1:
        return

    annotation = unmatched_annotations[0]
    body = unmatched_bodies[0]
    annotation.body = body
    annotation.body_id = body.body_id
    annotation.reasons.append(
        'memo body was associated by document order')
    body.annotation_refs.append(annotation)


def _body_text(span_records):
    pieces = []
    for record in span_records:
        if record.tagid != HWPTAG_PARA_TEXT:
            continue
        text = []
        for bounds, chunk in ParaTextChunks.parse_chunks(record.payload):
            text.append(_decode_chunk_text(chunk))
        piece = ''.join(text).rstrip('\n')
        if piece:
            pieces.append(piece)
    return MESSAGE_NEWLINE.join(pieces)


class Body(object):
    def __init__(self, path, section_index, span, record, records):
        self.path = path
        self.section_index = section_index
        self.span = span
        self.record = record
        self.records = records
        self.body_id = _body_id(record, records)
        self.text = _body_text(records)
        self.resources = _resource_refs(records)
        self.external_refs = []
        self.annotation_refs = []

    def to_dict(self):
        return {
            'id': self.body_id,
            'record_seqno': self.record.seqno,
            'stream': self.path,
            'record_span': list(self.span),
            'text': self.text,
            'resources': list(self.resources),
            'shared': len(self.annotation_refs) > 1,
            'reference_count': len(self.annotation_refs),
            'external_references': len(self.external_refs),
        }


def _collect_bodies(archive, version=None):
    bodies = []
    for path in archive.section_paths():
        records = list(iter_records(archive.streams[path]))
        section_index = archive.section_index(path)
        for start, end, record in _body_record_spans(records):
            body = Body(path, section_index, (start, end), record,
                        records[start:end])
            for item in body.records:
                for storage_id in _record_bindata_ids(item, version):
                    ref = {'kind': 'bindata', 'storage_id': storage_id}
                    if ref not in body.resources:
                        body.resources.append(ref)
            bodies.append(body)

    docinfo = archive.streams.get('DocInfo')
    if docinfo:
        records = list(iter_records(docinfo))
        for start, end, record in _body_record_spans(records):
            body = Body('DocInfo', -1, (start, end), record,
                        records[start:end])
            for item in body.records:
                for storage_id in _record_bindata_ids(item, version):
                    ref = {'kind': 'bindata', 'storage_id': storage_id}
                    if ref not in body.resources:
                        body.resources.append(ref)
            bodies.append(body)
    return bodies


def _find_external_body_refs(records, body, span=None):
    '''Find ID references outside the body's own record span.

    Memo field controls are tracked through annotations; only records of a
    different kind count as external protection here.
    '''
    needle = struct.pack('<I', body.body_id)
    span = span or body.span
    refs = []
    for index, record in enumerate(records):
        if span[0] <= index < span[1]:
            continue
        if record.tagid != HWPTAG_CTRL_HEADER:
            continue
        if chid_from_payload(record.payload) == MEMO_CHID:
            continue
        if needle in record.payload[4:]:
            refs.append(record.seqno)
    return refs


# ---------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------

class Annotation(object):
    def __init__(self, index, section, start, end, inventory=None):
        self.index = index
        self.section = section
        self.start = start
        self.end = end
        self.inventory = inventory
        self.annotation_id = 'ann-%04d' % (index + 1)
        self.text = ''
        self.body = None
        self.body_id = None
        self.body_candidates = []
        self.status = 'paired'
        self.reasons = []
        self.related = []
        self.resources = []
        self.markers = [start, end]

    @property
    def start_key(self):
        return (self.section['index'], self.start.paragraph.index,
                self.start.chunk_start)

    @property
    def end_key(self):
        return (self.section['index'], self.end.paragraph.index,
                self.end.chunk_start)

    @property
    def sort_key(self):
        return (self.start_key, self.end_key)

    @property
    def deletable(self):
        return self.status == 'paired'

    def to_dict(self):
        return {
            'id': self.annotation_id,
            'index': self.index,
            'status': self.status,
            'deletable': self.deletable,
            'reasons': list(self.reasons),
            'text': self.text,
            'field_id': self.start.field_id,
            'body_id': self.body_id,
            'body': self.body.to_dict() if self.body is not None else None,
            'markers': {
                'start': self.start.to_dict(self.section['index']),
                'end': self.end.to_dict(self.section['index']),
            },
            'range': {
                'section': self.section['index'],
                'start': [self.start.paragraph.index, self.start.chunk_end],
                'end': [self.end.paragraph.index, self.end.chunk_start],
            },
            'resources': list(self.resources),
            'related': list(self.related),
            'impact': {
                'markers': [
                    self.start.to_dict(self.section['index']),
                    self.end.to_dict(self.section['index']),
                ],
                'body': (self.body.to_dict()
                         if self.body is not None else None),
                'resources': list(self.resources),
            },
        }


class Inventory(object):
    def __init__(self, archive):
        self.archive = archive
        self.annotations = []
        self.refusals = []
        self.warnings = []
        self.sections = []
        self.bodies = []
        self.resources = {}
        self.body_refs = defaultdict(list)
        self.external_resource_refs = set()
        self._build()

    def _build(self):
        version = None
        manifest = self.archive.manifest or {}
        source = manifest.get('source') or {}
        if source.get('version'):
            version = tuple(source['version'])

        self.sections = _collect_markers(self.archive)
        self.bodies = _collect_bodies(self.archive, version)
        bodies_by_id = {}
        for body in self.bodies:
            bodies_by_id.setdefault(body.body_id, []).append(body)

        annotation_index = 0
        for section in self.sections:
            pairs, unmatched_starts, unmatched_ends, ambiguous = \
                _pair_markers(section)

            for marker in unmatched_starts:
                self.refusals.append({
                    'kind': 'unmatched_start',
                    'marker': marker.to_dict(section['index']),
                    'reasons': ['field start has no matching field end'],
                })
            for marker in unmatched_ends:
                self.refusals.append({
                    'kind': 'unmatched_end',
                    'marker': marker.to_dict(section['index']),
                    'reasons': ['field end has no matching field start'],
                })
            for item in ambiguous:
                self.refusals.append(dict(item, section=section['index']))

            for start, end in pairs:
                annotation = Annotation(annotation_index, section,
                                        start, end, inventory=self)
                annotation_index += 1
                annotation.text = self._annotation_text(annotation)
                self._resolve_body(annotation, bodies_by_id)
                self._resolve_resources(annotation)
                self.annotations.append(annotation)

        _match_bodies_to_annotations(self.bodies, self.annotations)
        for annotation in self.annotations:
            if annotation.body is not None:
                self._resolve_resources(annotation)

        # Record every anchor (paired or not) that references a body so that
        # an unpaired start still protects a shared body from deletion.
        for annotation in self.annotations:
            if annotation.body_id is not None:
                if annotation not in self.body_refs[annotation.body_id]:
                    self.body_refs[annotation.body_id].append(annotation)
                if annotation.body is not None and \
                        annotation not in annotation.body.annotation_refs:
                    annotation.body.annotation_refs.append(annotation)
        for refusal in self.refusals:
            if refusal.get('kind') != 'unmatched_start':
                continue
            marker = refusal['marker']
            for candidate in marker.get('body_candidates', []):
                if candidate in bodies_by_id:
                    self.body_refs[candidate].append(refusal)
                    break

        self._mark_ambiguous_pairs()
        self._mark_unpaired_overlaps()
        self._mark_body_ambiguities(bodies_by_id)
        self._recheck_shared_bodies()
        self._collect_external_resource_refs(version)
        self._collect_resources()

    def _annotation_text(self, annotation):
        start = annotation.start
        end = annotation.end
        paragraphs = annotation.section['paragraphs']
        start_index = start.paragraph.index
        end_index = end.paragraph.index
        if start_index == end_index:
            return _extract_range(start.paragraph, start.chunk_end,
                                  end.chunk_start)
        pieces = []
        pieces.append(_extract_range(start.paragraph, start.chunk_end,
                                     len(start.record.payload) // 2)
                      .rstrip('\n'))
        for index in range(start_index + 1, end_index):
            paragraph = paragraphs[index]
            if paragraph.starts_paragraph:
                pieces.append(paragraph_text(paragraph).rstrip('\n'))
        pieces.append(_extract_range(end.paragraph, 0, end.chunk_start))
        return MESSAGE_NEWLINE.join(pieces)

    def _resolve_body(self, annotation, bodies_by_id):
        candidates = []
        for value in annotation.start.body_candidates:
            if value and value not in candidates:
                candidates.append(value)
        matches = []
        for value in candidates:
            if value in bodies_by_id:
                for body in bodies_by_id[value]:
                    if body not in matches:
                        matches.append(body)
        annotation.body_candidates = list(candidates)
        if len(matches) > 1:
            annotation.status = 'ambiguous_body'
            annotation.reasons.append(
                'memo field references more than one memo body')
            annotation.related = [body.to_dict() for body in matches]
            return
        if not matches:
            if candidates:
                self.warnings.append({
                    'kind': 'missing_body',
                    'annotation': annotation.annotation_id,
                    'body_candidates': candidates,
                    'reason': 'memo body record was not found',
                })
            return
        body = matches[0]
        annotation.body = body
        annotation.body_id = body.body_id

    def _resolve_resources(self, annotation):
        refs = []
        if annotation.body is not None:
            refs.extend(annotation.body.resources)
        # A memo marker may carry a direct resource reference in its
        # parameter block; only keep IDs that the archive actually contains.
        refs = [ref for ref in refs
                if self._resource_exists(ref.get('storage_id'))]
        seen = set()
        unique = []
        for ref in refs:
            key = (ref.get('kind'), ref.get('storage_id'))
            if key in seen:
                continue
            seen.add(key)
            unique.append(ref)
        annotation.resources = unique

    def _resource_exists(self, storage_id):
        if storage_id is None:
            return False
        for alias in _storage_id_aliases(storage_id):
            manifest = self.archive.manifest or {}
            for entry in (manifest.get('resources', {})
                          .get('bindata') or []):
                if alias in _storage_id_aliases(entry.get('storage_id')):
                    return True
            prefix = 'BIN%04X.' % alias
            for name in self.archive.streams:
                if name.startswith('BinData/' + prefix):
                    return True
        return False

    def _mark_ambiguous_pairs(self):
        '''Refuse crossing or shared-boundary annotations as one component.'''
        parent = {}

        def find(item):
            while parent[item] != item:
                parent[item] = parent[parent[item]]
                item = parent[item]
            return item

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for annotation in self.annotations:
            parent[annotation.annotation_id] = annotation.annotation_id

        annotations = self.annotations
        for i, left in enumerate(annotations):
            for right in annotations[i + 1:]:
                if left.start_key == right.start_key or \
                        left.end_key == right.end_key:
                    union(left.annotation_id, right.annotation_id)
                    continue
                if left.section is not right.section:
                    continue
                a_start, a_end = left.sort_key
                b_start, b_end = right.sort_key
                # Proper nesting is unambiguous.  Crossing ranges (one
                # starts first and ends last) have two possible pairings.
                if (a_start < b_start and b_end < a_end) or \
                        (b_start < a_start and a_end < b_end):
                    continue
                if a_start < b_start < a_end or \
                        b_start < a_start < b_end:
                    union(left.annotation_id, right.annotation_id)

        by_root = defaultdict(list)
        for annotation in annotations:
            by_root[find(annotation.annotation_id)].append(annotation)
        for group in by_root.values():
            if len(group) < 2:
                continue
            for annotation in group:
                if annotation.status == 'paired':
                    annotation.status = 'ambiguous'
                annotation.reasons.append(
                    'annotation range overlaps another annotation with '
                    'more than one interpretation')
                annotation.related = [other.annotation_id
                                      for other in group
                                      if other is not annotation]

        for refusal in self.refusals:
            if refusal.get('kind') not in ('unclosed_nested_field',
                                           'multiple_field_candidates'):
                continue
            markers = refusal.get('markers')
            if markers:
                keys = set((item.get('section'), item.get('paragraph'),
                            item.get('code_units', [None, None])[0])
                           for item in markers)
            else:
                marker = refusal.get('marker', {})
                keys = set([(marker.get('section'), marker.get('paragraph'),
                             marker.get('code_units', [None, None])[0])])
            for annotation in annotations:
                if annotation.start.key in keys or \
                        annotation.end.key in keys:
                    if annotation.status == 'paired':
                        annotation.status = 'ambiguous'
                    reason = ('annotation marker participates in an '
                              'ambiguous field pairing')
                    if reason not in annotation.reasons:
                        annotation.reasons.append(reason)

    def _mark_unpaired_overlaps(self):
        '''Refuse a paired annotation containing an unpaired memo marker.

        An unpaired start/end inside another annotation's range means the
        pairing of that range has an alternative reading, so the containing
        annotation cannot be deleted without guessing.
        '''
        unpaired = []
        for refusal in self.refusals:
            if refusal.get('kind') not in ('unmatched_start',
                                           'unmatched_end'):
                continue
            marker = refusal.get('marker') or {}
            section = marker.get('section')
            paragraph = marker.get('paragraph')
            units = marker.get('code_units') or [None, None]
            if section is None or paragraph is None or units[0] is None:
                continue
            unpaired.append((section, paragraph, units[0]))

        for annotation in self.annotations:
            section_index = annotation.section['index']
            start_key = annotation.start_key
            end_key = annotation.end_key
            for key in unpaired:
                if key[0] != section_index:
                    continue
                if start_key < key < end_key:
                    if annotation.status == 'paired':
                        annotation.status = 'ambiguous'
                    reason = ('unpaired memo marker falls inside this '
                              'annotation range')
                    if reason not in annotation.reasons:
                        annotation.reasons.append(reason)

    def _mark_body_ambiguities(self, bodies_by_id):
        for body in self.bodies:
            records = list(iter_records(self.archive.streams[body.path])) \
                if body.path in self.archive.streams else []
            body.external_refs = _find_external_body_refs(records, body,
                                                          body.span)
        for annotation in self.annotations:
            if annotation.body_id is None:
                continue
            refs = self.body_refs.get(annotation.body_id, [])
            if len(refs) > 1:
                annotation.reasons.append(
                    'memo body is shared by other annotations')
            if annotation.body is not None and \
                    annotation.body.external_refs:
                annotation.reasons.append(
                    'memo body is referenced outside the annotation')

    def _recheck_shared_bodies(self):
        '''Recompute reasons after all anchors have been associated.

        Missing-body warnings are advisory and do not make an otherwise
        independent annotation unsafe.
        '''
        for annotation in self.annotations:
            if annotation.body_id is None:
                continue
            refs = [ref for ref in self.body_refs.get(annotation.body_id, [])
                    if isinstance(ref, Annotation)]
            if len(refs) > 1:
                reason = 'memo body is shared by other annotations'
                if reason not in annotation.reasons:
                    annotation.reasons.append(reason)
            if annotation.body is not None and \
                    annotation.body.external_refs:
                reason = 'memo body is referenced outside the annotation'
                if reason not in annotation.reasons:
                    annotation.reasons.append(reason)

    def _collect_external_resource_refs(self, version=None):
        '''Collect resources used by ordinary (non-memo) document content.'''
        body_spans = defaultdict(list)
        for body in self.bodies:
            body_spans[body.path].append(body.span)
        for path in self.archive.section_paths():
            data = self.archive.streams[path]
            records = list(iter_records(data))
            spans = body_spans.get(path, [])
            for index, record in enumerate(records):
                if any(start <= index < end for start, end in spans):
                    continue
                for storage_id in _record_bindata_ids(record, version):
                    self.external_resource_refs.add(storage_id)

    def _collect_resources(self):
        manifest = self.archive.manifest or {}
        for entry in (manifest.get('resources', {}).get('bindata') or []):
            storage_id = _manifest_storage_id(entry.get('storage_id'))
            if storage_id is None:
                continue
            self.resources[storage_id] = {
                'storage_id': storage_id,
                'manifest_storage_id': entry.get('storage_id'),
                'stream': entry.get('stream'),
                'references': [],
            }
        for annotation in self.annotations:
            for ref in annotation.resources:
                storage_id = _manifest_storage_id(ref.get('storage_id'))
                if storage_id not in self.resources:
                    self.resources[storage_id] = {
                        'storage_id': storage_id,
                        'stream': None,
                        'references': [],
                    }
                self.resources[storage_id]['references'].append(
                    annotation.annotation_id)
        for body in self.bodies:
            for ref in body.resources:
                storage_id = _manifest_storage_id(ref.get('storage_id'))
                if storage_id not in self.resources:
                    self.resources[storage_id] = {
                        'storage_id': storage_id,
                        'stream': None,
                        'references': [],
                    }
                self.resources[storage_id]['references'].append(
                    'body:%s' % body.body_id)

    def by_id(self, annotation_id):
        for annotation in self.annotations:
            if annotation.annotation_id == annotation_id:
                return annotation
        return None

    def select(self, values=None, select_all=False, texts=None,
               field_ids=None, indexes=None):
        '''Resolve a user selection to annotation objects.'''
        values = list(values or [])
        texts = list(texts or [])
        field_ids = list(field_ids or [])
        indexes = list(indexes or [])
        selected = []

        if select_all:
            selected = list(self.annotations)
        else:
            for value in values:
                for part in str(value).split(','):
                    part = part.strip()
                    if not part:
                        continue
                    selected.extend(self._select_one(part))
            for index in indexes:
                try:
                    selected.append(self.annotations[int(index)])
                except (ValueError, IndexError):
                    raise SelectionError('no annotation at index %r' % index)
            for text in texts:
                matches = [a for a in self.annotations if text in a.text]
                if not matches:
                    raise SelectionError('no annotation text contains %r'
                                         % text)
                selected.extend(matches)
            for field_id in field_ids:
                try:
                    field_id = int(field_id)
                except ValueError:
                    raise SelectionError('invalid field id %r' % field_id)
                matches = [a for a in self.annotations
                           if a.start.field_id == field_id]
                if not matches:
                    raise SelectionError('no annotation has field id %r'
                                         % field_id)
                selected.extend(matches)

        unique = []
        seen = set()
        for annotation in selected:
            if annotation.annotation_id in seen:
                continue
            seen.add(annotation.annotation_id)
            unique.append(annotation)
        return unique

    def _select_one(self, value):
        if value.startswith('id:'):
            value = value[3:]
        if value.startswith('index:'):
            try:
                return [self.annotations[int(value[6:])]]
            except (ValueError, IndexError):
                raise SelectionError('no annotation at %s' % value)
        if value.startswith('text:'):
            text = value[5:]
            matches = [a for a in self.annotations if text in a.text]
            if not matches:
                raise SelectionError('no annotation text contains %r' % text)
            return matches
        if value.startswith('field:'):
            try:
                field_id = int(value[6:])
            except ValueError:
                raise SelectionError('invalid field id %r' % value)
            matches = [a for a in self.annotations
                       if a.start.field_id == field_id]
            if not matches:
                raise SelectionError('no annotation has field id %r'
                                     % field_id)
            return matches
        exact = self.by_id(value)
        if exact is not None:
            return [exact]
        matches = [a for a in self.annotations
                   if a.text and (value in a.text or a.text in value)]
        if matches:
            return matches
        raise SelectionError('unknown annotation selection: %r' % value)

    def to_dict(self, include_archive=True):
        data = {
            'format': ANNOTATION_FORMAT,
            'source': self.archive.source,
            'annotations': [a.to_dict() for a in self.annotations],
            'refusals': list(self.refusals),
            'warnings': list(self.warnings),
            'bodies': [b.to_dict() for b in self.bodies],
            'resources': list(self.resources.values()),
            'counts': {
                'annotations': len(self.annotations),
                'deletable': len([a for a in self.annotations
                                  if a.deletable]),
                'ambiguous': len([a for a in self.annotations
                                  if a.status != 'paired']),
                'bodies': len(self.bodies),
            },
        }
        if include_archive:
            data['sections'] = [
                {'path': s['path'], 'index': s['index']}
                for s in self.sections
            ]
        return data

    def to_report(self):
        '''Return the explicit "list everything first" report.

        For every annotation this includes its text, both paired control
        markers with byte ranges, the resolved body, and the actual impact
        on resource references.
        '''
        return {
            'format': ANNOTATION_FORMAT,
            'source': self.archive.source,
            'annotations': [a.to_dict() for a in self.annotations],
            'refusals': list(self.refusals),
            'warnings': list(self.warnings),
            'bodies': [b.to_dict() for b in self.bodies],
            'resources': list(self.resources.values()),
            'counts': self.to_dict(include_archive=False)['counts'],
            'shared_bodies': [
                {
                    'id': body_id,
                    'annotations': [
                        ref.annotation_id if isinstance(ref, Annotation)
                        else {'kind': ref.get('kind'),
                              'marker': ref.get('marker')}
                        for ref in refs
                    ],
                }
                for body_id, refs in self.body_refs.items()
                if len(refs) > 1
            ],
        }


# ---------------------------------------------------------------------------
# deletion planning and application
# ---------------------------------------------------------------------------

class DeletionPlan(object):
    def __init__(self, inventory, selected):
        self.inventory = inventory
        self.selected = selected
        self.deletions = []
        self.refusals = []
        self.kept_bodies = []
        self.removed_bodies = []
        self.kept_resources = []
        self.removed_resources = []
        self._build()

    def _build(self):
        for annotation in self.selected:
            if not annotation.deletable:
                self.refusals.append({
                    'id': annotation.annotation_id,
                    'text': annotation.text,
                    'status': annotation.status,
                    'reasons': list(annotation.reasons) or
                               ['annotation is not safely deletable'],
                })
                continue
            self.deletions.append(annotation)

        selected_ids = set(a.annotation_id for a in self.deletions)
        selected_bodies = defaultdict(list)
        for annotation in self.deletions:
            if annotation.body_id is not None:
                selected_bodies[annotation.body_id].append(annotation)

        all_refs = self.inventory.body_refs
        for body_id, annotations in selected_bodies.items():
            refs = all_refs.get(body_id, [])
            external = []
            for ref in refs:
                if isinstance(ref, Annotation) and \
                        ref.annotation_id in selected_ids:
                    continue
                external.append(ref)
            body = annotations[0].body
            if refs and len(external) > 0:
                self.kept_bodies.append({
                    'id': body_id,
                    'reason': 'memo body is still referenced elsewhere',
                    'references': len(external),
                    'body': body.to_dict() if body is not None else None,
                })
            elif body is not None and body.external_refs:
                self.kept_bodies.append({
                    'id': body_id,
                    'reason': 'memo body is referenced outside annotations',
                    'references': len(body.external_refs),
                    'body': body.to_dict(),
                })
            elif body is not None:
                self.removed_bodies.append(body)

        removed_body_ids = set(body.body_id for body in self.removed_bodies)
        # Resources on removed bodies may still be used by ordinary document
        # content or by annotations that were not selected.
        before = self._resource_refcounts()
        after = self._resource_refcounts(removed_body_ids=removed_body_ids,
                                         removed_annotations=selected_ids)
        for storage_id, info in self.inventory.resources.items():
            if before.get(storage_id, 0) <= 0:
                continue
            if after.get(storage_id, 0) <= 0:
                self.removed_resources.append(storage_id)
            else:
                self.kept_resources.append({
                    'storage_id': storage_id,
                    'manifest_storage_id': info.get('manifest_storage_id'),
                    'references': after.get(storage_id, 0),
                })

    def _resource_refcounts(self, removed_body_ids=None,
                            removed_annotations=None):
        '''Count references to resources from surviving archive content.

        Ordinary body records count too: a resource referenced by a picture
        outside any memo body must never be deleted just because the memo
        that mentioned it goes away.
        '''
        removed_body_ids = removed_body_ids or set()
        removed_annotations = removed_annotations or set()
        counts = defaultdict(int)
        for body in self.inventory.bodies:
            if body.body_id in removed_body_ids:
                continue
            for ref in body.resources:
                counts[ref.get('storage_id')] += 1
        for annotation in self.inventory.annotations:
            if annotation.annotation_id in removed_annotations:
                continue
            for ref in annotation.resources:
                counts[ref.get('storage_id')] += 1
        for storage_id in self.inventory.external_resource_refs:
            counts[storage_id] += 1
        return counts

    def to_dict(self):
        return {
            'selected': [a.annotation_id for a in self.selected],
            'deletions': [
                {
                    'id': a.annotation_id,
                    'text': a.text,
                    'markers': {
                        'start': a.start.to_dict(a.section['index']),
                        'end': a.end.to_dict(a.section['index']),
                    },
                    'body_id': a.body_id,
                }
                for a in self.deletions
            ],
            'refusals': list(self.refusals),
            'bodies': {
                'removed': [body.to_dict() for body in self.removed_bodies],
                'kept': list(self.kept_bodies),
            },
            'resources': {
                'removed': list(self.removed_resources),
                'kept': list(self.kept_resources),
            },
        }


def plan_annotation_deletion(inventory, selection=None, select_all=False,
                             texts=None, field_ids=None, indexes=None):
    '''Return a :class:`DeletionPlan` without modifying any archive.'''
    selected = inventory.select(selection, select_all=select_all,
                                texts=texts, field_ids=field_ids,
                                indexes=indexes)
    return DeletionPlan(inventory, selected)


plan_deletions = plan_annotation_deletion


def _map_position(position, removals):
    '''Map an old code-unit position after some ranges are removed.

    ``removals`` is a list of ``(start, end)`` ranges in ascending order.
    Positions inside a removed range collapse to the range start.
    '''
    removed = 0
    for start, end in removals:
        if position >= end:
            removed += end - start
            continue
        if position > start:
            return start - removed
        break
    return position - removed


def _update_paragraph_header(record, delta, charshapes=None,
                             rangetags=None, linesegs=None):
    payload = bytearray(record.payload)
    if len(payload) < 4:
        return bytes(payload)
    value = struct.unpack_from('<I', payload, 0)[0]
    chars = value & 0x7fffffff
    chars = max(0, chars - delta)
    value = (value & 0x80000000) | (chars & 0x7fffffff)
    struct.pack_into('<I', payload, 0, value)
    if len(payload) >= 18:
        if charshapes is not None:
            struct.pack_into('<H', payload, 12, charshapes)
        if rangetags is not None:
            struct.pack_into('<H', payload, 14, rangetags)
        if linesegs is not None:
            struct.pack_into('<H', payload, 16, linesegs)
    return bytes(payload)


def _update_char_shapes(record, removals):
    payload = record.payload
    unitsize = 8
    result = []
    for offset in range(0, len(payload) - unitsize + 1, unitsize):
        position, shape = struct.unpack_from('<II', payload, offset)
        inside = False
        for start, end in removals:
            if start < position < end:
                inside = True
                break
        if inside:
            continue
        mapped = _map_position(position, removals)
        if result and result[-1][0] == mapped:
            # The last boundary at a collapsed position governs the text
            # that follows the removed marker.
            result[-1] = (mapped, shape)
        else:
            result.append((mapped, shape))
    if not result:
        return b''
    return b''.join(struct.pack('<II', position, shape)
                    for position, shape in result)


def _update_line_segs(record, removals):
    payload = record.payload
    unitsize = struct.calcsize('<iiiiiiiiHH')
    result = []
    previous = None
    for offset in range(0, len(payload) - unitsize + 1, unitsize):
        values = list(struct.unpack_from('<iiiiiiiiHH', payload, offset))
        position = values[0]
        mapped = _map_position(position, removals)
        if previous is not None and mapped < previous:
            mapped = previous
        values[0] = mapped
        result.append(struct.pack('<iiiiiiiiHH', *values))
        previous = mapped
    return b''.join(result)


def _update_range_tags(record, removals):
    payload = record.payload
    unitsize = 12
    result = []
    for offset in range(0, len(payload) - unitsize + 1, unitsize):
        start, end, tag = struct.unpack_from('<III', payload, offset)
        start = _map_position(start, removals)
        end = _map_position(end, removals)
        if end < start:
            end = start
        result.append(struct.pack('<III', start, end, tag))
    return b''.join(result)


def _remove_byte_ranges(payload, ranges):
    '''Remove byte ranges from a payload, rightmost first.'''
    result = bytearray(payload)
    for start, end in sorted(ranges, reverse=True):
        del result[start:end]
    return bytes(result)


def _all_annotations_for(deletions):
    '''Return annotations visible from a deletion set, including peers.

    ``Annotation.section`` holds every marker in the section, not only the
    annotations.  Sharing information must therefore come from the
    inventory snapshot; annotations carry a back-reference to it.
    '''
    inventory = None
    for annotation in deletions:
        inventory = annotation.inventory
        if inventory is not None:
            break
    if inventory is None:
        return list(deletions)
    return inventory.annotations


def _edit_section_stream(data, deletions):
    '''Rewrite one section stream for the given annotations.

    Only annotation markers and their field-control records are removed.
    Text, character shapes (except shifted positions), line segments and
    range tags are preserved.
    '''
    records = list(iter_records(data))
    paragraphs = split_paragraphs(records)

    by_paragraph = defaultdict(list)
    requested_controls = set()
    for annotation in deletions:
        by_paragraph[annotation.start.paragraph.index].append(
            (annotation.start, annotation.end))
        by_paragraph[annotation.end.paragraph.index].append(
            (annotation.start, annotation.end))
        if annotation.start.control is not None:
            requested_controls.add(annotation.start.control.seqno)
        if annotation.end.control is not None:
            requested_controls.add(annotation.end.control.seqno)

    # A control record may be shared by more than one anchor.  Only remove
    # it when every anchor that uses it is part of this deletion set.
    selected_ids = set(annotation.annotation_id
                       for annotation in deletions)
    control_users = defaultdict(set)
    for annotation in _all_annotations_for(deletions):
        for marker in (annotation.start, annotation.end):
            if marker.control is not None:
                control_users[marker.control.seqno].add(
                    annotation.annotation_id)
    control_seqnos = set()
    for seqno in requested_controls:
        users = control_users.get(seqno, set())
        if users and users <= selected_ids:
            control_seqnos.add(seqno)

    payloads = {}
    removed_seqnos = set(control_seqnos)

    for paragraph in paragraphs:
        if not paragraph.starts_paragraph:
            continue
        pairs = by_paragraph.get(paragraph.index)
        if not pairs:
            continue
        paratext = paragraph.paratext
        if paratext is None:
            continue

        ranges = []
        for start, end in pairs:
            if start.paragraph.index == paragraph.index:
                ranges.append((start.chunk_start, start.chunk_end))
            if end.paragraph.index == paragraph.index:
                ranges.append((end.chunk_start, end.chunk_end))
        ranges.sort()
        merged = []
        for start, end in ranges:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        removed_units = sum(end - start for start, end in merged)
        byte_ranges = [(start * 2, end * 2) for start, end in merged]
        new_payload = _remove_byte_ranges(paratext.payload, byte_ranges)
        payloads[paratext.seqno] = new_payload

        header = paragraph.header
        charshape_records = paragraph.by_tag.get(HWPTAG_PARA_CHAR_SHAPE, [])
        lineseg_records = paragraph.by_tag.get(HWPTAG_PARA_LINE_SEG, [])
        rangetag_records = paragraph.by_tag.get(HWPTAG_PARA_RANGE_TAG, [])

        new_charshapes = None
        if charshape_records:
            new_charshapes_payload = _update_char_shapes(
                charshape_records[0], merged)
            payloads[charshape_records[0].seqno] = new_charshapes_payload
            new_charshapes = len(new_charshapes_payload) // 8
            for record in charshape_records[1:]:
                payloads[record.seqno] = record.payload

        new_linesegs = None
        if lineseg_records:
            new_linesegs_payload = _update_line_segs(
                lineseg_records[0], merged)
            payloads[lineseg_records[0].seqno] = new_linesegs_payload
            new_linesegs = (len(new_linesegs_payload) //
                            struct.calcsize('<iiiiiiiiHH'))
            for record in lineseg_records[1:]:
                payloads[record.seqno] = record.payload

        new_rangetags = None
        if rangetag_records:
            new_rangetags_payload = _update_range_tags(
                rangetag_records[0], merged)
            payloads[rangetag_records[0].seqno] = new_rangetags_payload
            new_rangetags = len(new_rangetags_payload) // 12
            for record in rangetag_records[1:]:
                payloads[record.seqno] = record.payload

        payloads[header.seqno] = _update_paragraph_header(
            header, removed_units, charshapes=new_charshapes,
            rangetags=new_rangetags, linesegs=new_linesegs)

    chunks = []
    for record in records:
        if record.seqno in removed_seqnos:
            continue
        if record.seqno in payloads:
            record.payload = payloads[record.seqno]
            chunks.append(record.encode())
        else:
            chunks.append(record.raw(data))
    return b''.join(chunks)


def _remove_body_records(data, spans):
    '''Remove record spans (list indexes are relative to this stream).'''
    if not spans:
        return data
    records = list(iter_records(data))
    removed = set()
    for start, end in spans:
        for index in range(start, end):
            removed.add(records[index].seqno)
    chunks = []
    for record in records:
        if record.seqno in removed:
            continue
        chunks.append(record.raw(data))
    return b''.join(chunks)


def _body_spans_in_stream(data, body_ids):
    records = list(iter_records(data))
    spans = []
    for start, end, record in _body_record_spans(records):
        body_id = _body_id(record, records[start:end])
        if body_id in body_ids:
            spans.append((start, end))
    return spans


def _update_manifest(archive, inventory, plan, destination,
                     resources_deleted=False):
    manifest = json.loads(json.dumps(archive.manifest or {}))
    manifest.setdefault('format', FRAGMENT_FORMAT)
    manifest.setdefault('complete', True)
    manifest.setdefault('incomplete_reasons', [])
    manifest.setdefault('warnings', [])

    removed_streams = set()
    if resources_deleted:
        for storage_id in plan.removed_resources:
            prefix = 'BIN%04X.' % storage_id
            for name in list(archive.streams):
                if name.startswith('BinData/' + prefix):
                    removed_streams.add(name)

        resources = manifest.setdefault('resources', {})
        bindata = resources.get('bindata')
        if isinstance(bindata, list):
            resources['bindata'] = [
                entry for entry in bindata
                if entry.get('stream') not in removed_streams and
                not any(alias in plan.removed_resources for alias in
                        _storage_id_aliases(entry.get('storage_id')))
            ]

    manifest['annotations'] = [a.to_dict() for a in inventory.annotations]
    manifest['deletion'] = plan.to_dict()
    manifest['deleted_annotations'] = [
        a.annotation_id for a in plan.deletions]
    manifest['source'] = manifest.get('source') or {
        'file': archive.source,
    }
    return manifest


def delete_annotations(source, destination, selection=None, select_all=False,
                       texts=None, field_ids=None, indexes=None,
                       dry_run=False, in_place=False,
                       delete_unreferenced_resources=False):
    '''Remove selected annotations from *source* into *destination*.

    The returned dictionary is a JSON-serialisable report.  Body text is not
    removed; only memo anchors, their field control records, and unreferenced
    memo bodies are removed.

    Resource streams are reported in the plan but are kept by default.  A
    BinData stream is referenced from DocInfo/IdMappings, and removing the
    stream alone would leave those fields inconsistent with the archive.
    Deleting only the notes must not rewrite unrelated fields; callers that
    also rewrite the document's resource tables can opt in with
    ``delete_unreferenced_resources``.
    '''
    archive = open_fragment(source)
    inventory = Inventory(archive)
    plan = plan_annotation_deletion(
        inventory, selection=selection, select_all=select_all,
        texts=texts, field_ids=field_ids, indexes=indexes)

    report = {
        'format': 'hwp5-annotation-deletion/1',
        'source': source,
        'destination': destination,
        'dry_run': dry_run,
        'inventory': inventory.to_report(),
        'plan': plan.to_dict(),
        'deleted': [a.annotation_id for a in plan.deletions],
        'refused': plan.to_dict()['refusals'],
    }

    if dry_run:
        return report

    if in_place:
        if destination is None:
            destination = source
        elif os.path.abspath(destination) != os.path.abspath(source):
            raise FragError('in_place requires destination to be source')
    if destination is None:
        raise FragError('destination is required unless dry_run is used')
    if not in_place and os.path.abspath(destination) == \
            os.path.abspath(source):
        raise FragError('source and destination must differ; '
                        'use in_place=True to edit in place')

    output = archive.copy()

    for section in inventory.sections:
        selected = [a for a in plan.deletions
                    if a.section is section]
        if selected:
            output.streams[section['path']] = _edit_section_stream(
                output.streams[section['path']], selected)

    removed_body_ids = set(body.body_id for body in plan.removed_bodies)
    if removed_body_ids:
        for path in list(output.streams):
            if not (path.startswith('BodyText/') or
                    path.startswith('ViewText/') or path == 'DocInfo'):
                continue
            spans = _body_spans_in_stream(output.streams[path],
                                          removed_body_ids)
            if spans:
                output.streams[path] = _remove_body_records(
                    output.streams[path], spans)

    for stream in list(output.streams):
        if not delete_unreferenced_resources:
            break
        for storage_id in plan.removed_resources:
            candidate = 'BIN%04X.' % storage_id
            if stream.startswith('BinData/' + candidate):
                del output.streams[stream]
                break

    output.manifest = _update_manifest(
        archive, inventory, plan, destination,
        resources_deleted=delete_unreferenced_resources)
    output.manifest['deletion']['resources_deleted'] = bool(
        delete_unreferenced_resources)
    output.manifest['deletion']['resources_reported_only'] = not bool(
        delete_unreferenced_resources)
    output.manifest['deletion']['streams_removed'] = sorted(
        set(archive.streams) - set(output.streams))
    output.save(destination)
    report['manifest'] = os.path.join(destination, 'manifest.json')
    return report


remove_annotations = delete_annotations
prune_annotations = delete_annotations


def inspect_annotations(source):
    return Inventory(open_fragment(source))


list_annotations = inspect_annotations
inventory = inspect_annotations


def format_inventory_text(inv):
    lines = []
    lines.append('annotations: %d' % len(inv.annotations))
    for annotation in inv.annotations:
        lines.append('%s [%s] %s' % (
            annotation.annotation_id,
            annotation.status,
            annotation.text.replace('\n', '\\n')))
        start = annotation.start.to_dict(annotation.section['index'])
        end = annotation.end.to_dict(annotation.section['index'])
        lines.append('  start: para=%s pos=%s bytes=%s chid=%s' % (
            start['paragraph'], start['code_units'][0],
            start['byte_range'], start['chid']))
        lines.append('  end:   para=%s pos=%s bytes=%s chid=%s' % (
            end['paragraph'], end['code_units'][0],
            end['byte_range'], end['chid']))
        if annotation.body_id is not None:
            lines.append('  body: %s' % annotation.body_id)
        if annotation.resources:
            lines.append('  resources: %s' % ', '.join(
                str(r.get('storage_id')) for r in annotation.resources))
        if annotation.reasons:
            lines.append('  reasons: %s' % '; '.join(annotation.reasons))
    for refusal in inv.refusals:
        lines.append('refused: %s %s' % (
            refusal.get('kind'), refusal.get('reasons')))
    return '\n'.join(lines)
