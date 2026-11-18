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
''' Stable element identity for XML exports.

The XML export used to number elements (paragraphs, tables, shapes) by
their document order, so inserting a single paragraph at the beginning
renumbered every following element and made the whole document look
changed to downstream collectors.

This module derives element identity from evidence instead:

- original records: a baseline index of previously collected elements
  can be supplied so that matched elements keep their original ids;
- structural position: the ancestor path of each element is recorded
  as evidence in the collected index;
- textual evidence: the text contained in an element (and the ids of
  nested tracked elements) determines its content fingerprint.

Identical (duplicate) elements are disambiguated by an occurrence
ordinal.  When duplicates are split or merged between revisions,
:func:`reconcile` records the remaining candidates together with the
contributing character ranges, and fresh ids are allocated without
reusing ids retired from the baseline index.
'''
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from collections import defaultdict
from hashlib import sha1
import logging
import sys

from .binmodel import ControlChar
from .binmodel import GShapeObjectControl
from .binmodel import Paragraph
from .binmodel import ShapeComponent
from .binmodel import TableControl
from .binmodel import Text
from .treeop import STARTEVENT, ENDEVENT
from .treeop import build_subtree
from .treeop import iter_subevents
from .treeop import tree_events_multi
from .xmlformat import xmlattrval


PY3 = sys.version_info.major == 3
if PY3:
    basestring = str


logger = logging.getLogger(__name__)


INDEX_FORMAT = 'pyhwp-stable-id-index/1'
CORRESPONDENCE_FORMAT = 'pyhwp-revision-correspondence/1'

# model -> (xml attribute name, kind name, id prefix)
TRACKED_MODELS = {
    Paragraph: ('paragraph_id', 'paragraph', 'p'),
    TableControl: ('table_id', 'table', 't'),
    GShapeObjectControl: ('gshape_id', 'gshape', 'g'),
    ShapeComponent: ('shape_id', 'shape', 's'),
}


def own_attributes_evidence(attributes):
    ''' Render scalar attributes of an element as stable evidence. '''
    parts = []
    for name in sorted(attributes):
        value = attributes[name]
        if isinstance(value, (dict, list, tuple)):
            continue
        try:
            rendered = xmlattrval(value)
        except Exception:
            continue
        if not isinstance(rendered, basestring):
            rendered = str(rendered)
        parts.append('%s=%s' % (name, rendered))
    return ';'.join(parts)


def collect_text_evidence(events):
    ''' Collect textual evidence from a tracked element subtree.

    Nested tracked elements contribute their already-assigned stable id
    instead of their textual content, so the parent fingerprint depends
    on the identity of its children.
    '''
    parts = []
    skip_depth = 0
    for event, item in events:
        model, attributes = item[0], item[1]
        if event is STARTEVENT:
            if skip_depth > 0:
                skip_depth += 1
            elif model in TRACKED_MODELS:
                attribute_name = TRACKED_MODELS[model][0]
                parts.append('\x01%s\x01' % attributes.get(attribute_name, ''))
                skip_depth = 1
            elif model is Text:
                parts.append(attributes.get('text', ''))
            elif model is ControlChar:
                parts.append('\x00%s\x00' % attributes.get('name', ''))
        elif event is ENDEVENT:
            if skip_depth > 0:
                skip_depth -= 1
    return ''.join(parts)


def compute_fingerprint(kind, attributes_evidence, text_evidence):
    sha = sha1()
    sha.update(kind.encode('utf-8'))
    sha.update(b'\x00')
    sha.update(attributes_evidence.encode('utf-8'))
    sha.update(b'\x00')
    sha.update(text_evidence.encode('utf-8'))
    return sha.hexdigest()


def text_excerpt(evidence, limit=80):
    excerpt = evidence.replace('\x00', ' ').replace('\x01', ' ')
    excerpt = ' '.join(excerpt.split())
    return excerpt[:limit]


class StableIdAssigner(object):
    ''' Assign stable ids to tracked elements of one export.

    When baseline records (an index collected from a previous revision)
    are given, elements matched by (kind, fingerprint, occurrence) keep
    their original ids.  Fresh ids are allocated for new elements and
    are never allowed to overwrite an id present in the baseline index.
    '''

    def __init__(self, baseline_records=()):
        self.records = []
        self.occurrences = defaultdict(int)
        self.baseline_ids = set()
        self.baseline_by_key = dict()
        for record in baseline_records or ():
            key = (record['kind'],
                   record['fingerprint'],
                   record.get('occurrence', 1))
            self.baseline_by_key[key] = record['id']
            self.baseline_ids.add(record['id'])

    def assign(self, model, attributes, position, text_evidence):
        attribute_name, kind, prefix = TRACKED_MODELS[model]
        attributes_evidence = own_attributes_evidence(attributes)
        fingerprint = compute_fingerprint(kind,
                                          attributes_evidence,
                                          text_evidence)
        key = (kind, fingerprint)
        self.occurrences[key] += 1
        occurrence = self.occurrences[key]

        stable_id = self.baseline_by_key.get((kind, fingerprint,
                                              occurrence))
        if stable_id is None:
            stable_id = prefix + fingerprint[:12]
            if occurrence > 1:
                stable_id = '%s#%d' % (stable_id, occurrence)
            candidate = stable_id
            salt = 0
            while candidate in self.baseline_ids:
                salt += 1
                candidate = '%s.%d' % (stable_id, salt)
            stable_id = candidate

        record = dict(
            id=stable_id,
            kind=kind,
            fingerprint=fingerprint,
            occurrence=occurrence,
            position=[list(step) for step in position],
            evidence=text_evidence,
        )
        self.records.append(record)
        return stable_id


def give_elements_stable_id(event_prefixed_mac, assigner=None, _position=()):
    ''' Replace order-based ids with evidence-based stable ids. '''
    if assigner is None:
        assigner = StableIdAssigner()
    counters = defaultdict(int)
    event_prefixed_mac = iter(event_prefixed_mac)
    for event, item in event_prefixed_mac:
        model = item[0]
        if event is STARTEVENT:
            counters[model.__name__] += 1
            position = _position + ((model.__name__,
                                     counters[model.__name__]),)
            if model in TRACKED_MODELS:
                root, children = build_subtree(event_prefixed_mac)
                child_events = tree_events_multi(children)
                child_events = give_elements_stable_id(child_events,
                                                       assigner,
                                                       position)
                child_events = list(child_events)
                text_evidence = collect_text_evidence(child_events)
                rmodel, rattributes, rcontext = root
                stable_id = assigner.assign(rmodel, rattributes,
                                            position, text_evidence)
                attribute_name = TRACKED_MODELS[rmodel][0]
                rattributes[attribute_name] = stable_id
                yield STARTEVENT, root
                for child_event in child_events:
                    yield child_event
                yield ENDEVENT, root
            else:
                subevents = iter_subevents(event_prefixed_mac)
                subevents = give_elements_stable_id(subevents,
                                                    assigner,
                                                    position)
                yield event, item
                for subevent in subevents:
                    yield subevent
        else:
            yield event, item


def _group_by_fingerprint(records):
    groups = defaultdict(list)
    for record in records:
        key = (record['kind'], record['fingerprint'])
        groups[key].append(record)
    for group in groups.values():
        group.sort(key=lambda record: record.get('occurrence', 1))
    return groups


def _contribution(source, target):
    ''' Describe how source evidence contributes to target evidence. '''
    source_text = source['evidence']
    target_text = target['evidence']
    if not source_text or not target_text:
        return None
    if source_text == target_text:
        return dict(source_id=source['id'],
                    target_id=target['id'],
                    relation='duplicate',
                    source_range=[0, len(source_text)],
                    target_range=[0, len(target_text)])
    offset = source_text.find(target_text)
    if offset >= 0:
        return dict(source_id=source['id'],
                    target_id=target['id'],
                    relation='split',
                    source_range=[offset, offset + len(target_text)],
                    target_range=[0, len(target_text)])
    offset = target_text.find(source_text)
    if offset >= 0:
        return dict(source_id=source['id'],
                    target_id=target['id'],
                    relation='merge',
                    source_range=[0, len(source_text)],
                    target_range=[offset, offset + len(source_text)])
    return None


def reconcile(baseline_records, current_records):
    ''' Build a revision correspondence between two collected indexes.

    Elements are matched by original records (baseline ids), structural
    position and textual evidence.  Unmatched elements keep a list of
    candidates with contributing character ranges so that splits and
    merges of duplicate paragraphs remain traceable.
    '''
    baseline_groups = _group_by_fingerprint(baseline_records)
    current_groups = _group_by_fingerprint(current_records)

    matched = []
    new = []
    retired = []
    keys = set(baseline_groups) | set(current_groups)
    for key in keys:
        baseline_group = baseline_groups.get(key, [])
        current_group = current_groups.get(key, [])
        common = min(len(baseline_group), len(current_group))
        for idx in range(common):
            baseline_record = baseline_group[idx]
            current_record = current_group[idx]
            matched.append(dict(
                id=baseline_record['id'],
                kind=current_record['kind'],
                fingerprint=current_record['fingerprint'],
                occurrence=current_record.get('occurrence', 1),
                baseline_position=baseline_record['position'],
                current_position=current_record['position'],
                text_excerpt=text_excerpt(current_record['evidence']),
            ))
        retired.extend(baseline_group[common:])
        new.extend(current_group[common:])

    # splits/merges of (duplicate) paragraphs: leave candidates with
    # contributing ranges instead of silently dropping the relation
    new_entries = []
    for new_record in new:
        candidates = []
        for retired_record in retired:
            if retired_record['kind'] != new_record['kind']:
                continue
            contribution = _contribution(retired_record, new_record)
            if contribution is not None:
                candidates.append(contribution)
        new_entries.append(dict(
            id=new_record['id'],
            kind=new_record['kind'],
            fingerprint=new_record['fingerprint'],
            occurrence=new_record.get('occurrence', 1),
            position=new_record['position'],
            text_excerpt=text_excerpt(new_record['evidence']),
            candidates=candidates,
        ))

    retired_entries = []
    for retired_record in retired:
        candidates = []
        for new_record in new:
            if new_record['kind'] != retired_record['kind']:
                continue
            contribution = _contribution(retired_record, new_record)
            if contribution is not None:
                candidates.append(contribution)
        retired_entries.append(dict(
            id=retired_record['id'],
            kind=retired_record['kind'],
            fingerprint=retired_record['fingerprint'],
            occurrence=retired_record.get('occurrence', 1),
            position=retired_record['position'],
            text_excerpt=text_excerpt(retired_record['evidence']),
            candidates=candidates,
        ))

    matched.sort(key=lambda entry: entry['current_position'])
    new_entries.sort(key=lambda entry: entry['position'])
    retired_entries.sort(key=lambda entry: entry['position'])

    return dict(
        format=CORRESPONDENCE_FORMAT,
        matched=matched,
        new=new_entries,
        retired=retired_entries,
    )


def index_to_jsonable(records):
    return dict(format=INDEX_FORMAT, records=list(records))


def index_from_jsonable(document):
    if document.get('format') != INDEX_FORMAT:
        raise ValueError('not a %s document: %r' %
                         (INDEX_FORMAT, document.get('format')))
    return document['records']
