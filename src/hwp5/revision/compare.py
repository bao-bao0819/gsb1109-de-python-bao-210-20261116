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

The comparison is built around three questions that the HWP format keeps
separate:

* ``resource_reorder`` - DocInfo resources changed their table positions
  while their meaning stayed the same.  Paragraphs may carry rewritten ids,
  but their resolved language/font/style meaning is unchanged.
* ``resource_content_change`` - a resource definition changed (a font entry
  was edited, a char shape gained a colour, ...).  Paragraphs that use the
  changed resource are affected.
* ``body_change`` - the paragraph itself changed (text or its own direct
  references), independent of DocInfo definitions.

Duplicate resources are never paired one-to-one by position.  Only a
paragraph whose resolved meaning is proven different gets its language and
style index rebuilt; all other paragraphs keep the old (equivalent) index.
'''
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from difflib import SequenceMatcher

from .canonical import canonical_diff
from .paragraphs import read_sections
from .resources import FONT
from .resources import ResourceMatch
from .resources import ResourceTable


STATUS_UNCHANGED = 'unchanged'
STATUS_REMAPPED = 'remapped'
STATUS_RESOURCE_CHANGE = 'resource-change'
STATUS_REORDER_AFFECTED = 'reorder-affected'
STATUS_REFERENCE_CHANGE = 'reference-change'
STATUS_TEXT_CHANGE = 'text-change'
STATUS_ADDED = 'added'
STATUS_REMOVED = 'removed'

BODY_STATUSES = frozenset([
    STATUS_TEXT_CHANGE,
    STATUS_REFERENCE_CHANGE,
    STATUS_ADDED,
    STATUS_REMOVED,
])

AFFECTED_STATUSES = frozenset([
    STATUS_RESOURCE_CHANGE,
    STATUS_REORDER_AFFECTED,
    STATUS_REFERENCE_CHANGE,
    STATUS_TEXT_CHANGE,
    STATUS_ADDED,
])


class RevisionDocument(object):
    '''A document view with a resource table and paragraph snapshots.'''

    def __init__(self, table, sections):
        self.table = table
        self.sections = sections

    @classmethod
    def from_hwp5file(cls, hwp5file):
        table = ResourceTable.from_docinfo(hwp5file.docinfo)
        sections = read_sections(hwp5file, table)
        return cls(table, sections)

    @classmethod
    def open(cls, source):
        if isinstance(source, RevisionDocument):
            return source
        if hasattr(source, 'docinfo') and hasattr(source, 'bodytext'):
            return cls.from_hwp5file(source)
        from ..binmodel import Hwp5File
        return cls.from_hwp5file(Hwp5File(source))

    def section(self, index):
        if index < 0 or index >= len(self.sections):
            return None
        return self.sections[index]


class ParagraphResult(object):

    def __init__(self, section, old, new, status, reasons=None):
        self.section = section
        self.old = old
        self.new = new
        self.status = status
        self.reasons = reasons or []

    @property
    def affected(self):
        return self.status in AFFECTED_STATUSES

    @property
    def rebuilt(self):
        '''Only a proven-affected paragraph rebuilds its semantic indices.'''
        return self.affected

    def to_dict(self, include_indices=True):
        old, new = self.old, self.new
        data = {
            'section': self.section,
            'status': self.status,
            'affected': self.affected,
            'index_rebuilt': self.rebuilt,
            'old_ordinal': old.ordinal if old else None,
            'new_ordinal': new.ordinal if new else None,
            'old_seqno': old.seqno if old else None,
            'new_seqno': new.seqno if new else None,
            'old_text': old.text if old else None,
            'new_text': new.text if new else None,
            'reasons': self.reasons,
        }
        if old and new:
            data['numeric_ids_changed'] = \
                old.numeric_references() != new.numeric_references()
            data['old_ids'] = old.numeric_references()
            data['new_ids'] = new.numeric_references()
            if old.text == new.text and old.semantic != new.semantic:
                data['changes'] = canonical_diff(old.semantic, new.semantic)
        if include_indices:
            # Unchanged and remapped paragraphs keep the old semantic index.
            data['language_index_source'] = 'old' if not self.affected \
                else ('new' if new is not None else None)
            data['style_index_source'] = 'old' if not self.affected \
                else ('new' if new is not None else None)
            if not self.affected and old is not None:
                data['language_index'] = old.language_index
                data['style_index'] = old.style_index()
            elif new is not None:
                data['language_index'] = new.language_index
                data['style_index'] = new.style_index()
            else:
                data['language_index'] = old.language_index if old else None
                data['style_index'] = old.style_index() if old else None
        return data


class RevisionComparison(object):

    def __init__(self, old, new):
        self.old = old
        self.new = new
        self.match = None
        self.paragraphs = []
        self.sections = []

    # -- alignment ----------------------------------------------------------

    @staticmethod
    def _text_key(paragraph):
        return paragraph.text

    @staticmethod
    def _similarity(a, b):
        if a.text == b.text:
            return 1.0
        if not a.text or not b.text:
            return 0.0
        return SequenceMatcher(None, a.text, b.text).ratio()

    def _align_section(self, old_section, new_section):
        old_items = list(old_section or [])
        new_items = list(new_section or [])
        old_keys = [self._text_key(item) for item in old_items]
        new_keys = [self._text_key(item) for item in new_items]
        matcher = SequenceMatcher(None, old_keys, new_keys, autojunk=False)

        results = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                for k in range(i2 - i1):
                    results.append(ParagraphResult(
                        old_section.section_index if old_section else None,
                        old_items[i1 + k], new_items[j1 + k],
                        STATUS_UNCHANGED))
            elif tag == 'replace':
                results.extend(self._align_replaced(
                    old_section, old_items[i1:i2], new_items[j1:j2]))
            elif tag == 'delete':
                for item in old_items[i1:i2]:
                    results.append(ParagraphResult(
                        old_section.section_index if old_section else None,
                        item, None, STATUS_REMOVED))
            elif tag == 'insert':
                for item in new_items[j1:j2]:
                    results.append(ParagraphResult(
                        old_section.section_index if old_section else None,
                        None, item, STATUS_ADDED))
        return results

    def _align_replaced(self, old_section, old_items, new_items):
        section = old_section.section_index if old_section else None
        results = []
        used_old = set()
        used_new = set()

        # Greedy best-similarity pairing within the replaced block, so an
        # edited paragraph is reported as an edit rather than a delete plus
        # an insert even when other paragraphs were inserted around it.
        candidates = []
        for i, old_item in enumerate(old_items):
            for j, new_item in enumerate(new_items):
                similarity = self._similarity(old_item, new_item)
                if similarity >= 0.5:
                    candidates.append((similarity, i, j))
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

        for similarity, i, j in candidates:
            if i in used_old or j in used_new:
                continue
            used_old.add(i)
            used_new.add(j)
            results.append((i, j, ParagraphResult(
                section, old_items[i], new_items[j], STATUS_TEXT_CHANGE)))

        aligned = []
        paired = dict((i, (j, result)) for i, j, result in results)
        used_j = set(j for _, j, _ in results)

        for i, item in enumerate(old_items):
            if i in paired:
                aligned.append(paired[i][1])
            else:
                aligned.append(ParagraphResult(section, item, None,
                                               STATUS_REMOVED))
            # New-only paragraphs that sit before the next paired paragraph
            # keep document order.
        inserted = [ParagraphResult(section, None, new_items[j], STATUS_ADDED)
                    for j in range(len(new_items)) if j not in used_j]
        if inserted:
            # Preserve new document order: merge by new ordinal.
            merged = list(aligned)
            for result in inserted:
                position = len(merged)
                for index, existing in enumerate(merged):
                    new_item = existing.new
                    if new_item is not None and \
                            new_item.ordinal > result.new.ordinal:
                        position = index
                        break
                merged.insert(position, result)
            aligned = merged
        return aligned

    # -- usage evidence -----------------------------------------------------

    def _observe_usage(self):
        '''Feed aligned, text-identical paragraphs as resource usage evidence.

        A paragraph whose text is unchanged is strong evidence that the old
        resource id and the new resource id denote the same resource, even if
        the resource content changed.  Paragraphs whose text changed provide
        no such evidence and are never used to pair duplicates.
        '''
        match = self.match
        for result in self.paragraphs:
            old, new = result.old, result.new
            if old is None or new is None:
                continue
            if old.text != new.text:
                continue
            old_parashape = self.old.table.get('parashape',
                                               old.parashape_id)
            new_parashape = self.new.table.get('parashape',
                                               new.parashape_id)
            match.observe_usage(old_parashape, new_parashape)
            old_style = self.old.table.get('style', old.style_id)
            new_style = self.new.table.get('style', new.style_id)
            match.observe_usage(old_style, new_style)

            old_runs = _charshape_runs(old)
            new_runs = _charshape_runs(new)
            if len(old_runs) == len(new_runs):
                for (_, _, old_id), (_, _, new_id) in zip(old_runs, new_runs):
                    match.observe_usage(
                        self.old.table.get('charshape', old_id),
                        self.new.table.get('charshape', new_id))

    # -- paragraph classification ------------------------------------------

    def _classify_paragraph(self, result):
        old, new = result.old, result.new
        if old is None:
            result.status = STATUS_ADDED
            return
        if new is None:
            result.status = STATUS_REMOVED
            return

        if old.text != new.text:
            result.status = STATUS_TEXT_CHANGE
            result.reasons.append('text')
            return

        same_semantics = old.semantic == new.semantic
        same_ids = old.numeric_references() == new.numeric_references()

        if same_semantics:
            if same_ids:
                result.status = STATUS_UNCHANGED
            else:
                result.status = STATUS_REMAPPED
                result.reasons.append('resource-reorder')
            return

        attribution = self._reference_attribution(old, new)
        if attribution == 'resource-content':
            result.status = STATUS_RESOURCE_CHANGE
            result.reasons.append('resource-content')
        elif attribution == 'reorder':
            result.status = STATUS_REORDER_AFFECTED
            result.reasons.append('resource-reorder')
        else:
            result.status = STATUS_REFERENCE_CHANGE
            result.reasons.append('reference')

    def _reference_attribution(self, old, new):
        '''Explain why two text-identical paragraphs resolve differently.

        * ``resource-content``: a direct reference pairs with a changed
          resource (directly or through a dependency).
        * ``reorder``: every direct resource is unchanged, but its table
          position moved and the paragraph kept the old numeric id, so the
          id now names a different resource.
        * ``reference``: anything else, i.e. the paragraph itself points
          somewhere the resource history cannot justify.
        '''
        pairs = []
        same_ids = old.parashape_id == new.parashape_id and \
            old.style_id == new.style_id

        old_parashape = self.old.table.get('parashape', old.parashape_id)
        new_parashape = self.new.table.get('parashape', new.parashape_id)
        old_style = self.old.table.get('style', old.style_id)
        new_style = self.new.table.get('style', new.style_id)
        pairs.extend([(old_parashape, new_parashape),
                      (old_style, new_style)])

        old_runs = _charshape_runs(old)
        new_runs = _charshape_runs(new)
        if len(old_runs) == len(new_runs):
            for (old_start, old_end, old_id), (new_start, new_end, new_id) \
                    in zip(old_runs, new_runs):
                if (old_start, old_end) != (new_start, new_end):
                    return 'reference'
                pairs.append((
                    self.old.table.get('charshape', old_id),
                    self.new.table.get('charshape', new_id),
                ))
        else:
            return 'reference'

        any_content = False
        any_reordered = False
        for old_resource, new_resource in pairs:
            if old_resource is None and new_resource is None:
                continue
            if old_resource is None or new_resource is None:
                return 'reference'
            pair = self.match.pair_of_old(old_resource)
            if pair is None:
                return 'reference'
            if pair.status != 'unchanged':
                cause = self.match.change_cause(pair)
                if cause == 'content':
                    any_content = True
                else:
                    any_reordered = True
            elif pair.new.key != new_resource.key:
                # The resource itself is unchanged but the paragraph now
                # names another resource; only a reorder can explain that.
                any_reordered = True

        if any_content:
            return 'resource-content'
        if any_reordered and same_ids:
            return 'reorder'
        return 'reference'

    # -- resource usage summary --------------------------------------------

    def _changed_resource_users(self):
        users = {}
        for pair in self.match.changed_pairs():
            users[pair.old.key] = []
        for result in self.paragraphs:
            if not result.affected or result.old is None or \
                    result.new is None:
                continue
            for old_resource in _referenced_resources(self.old.table,
                                                      result.old):
                pair = self.match.pair_of_old(old_resource)
                if pair is None or pair.status == 'unchanged':
                    continue
                users.setdefault(pair.old.key, []).append({
                    'section': result.section,
                    'old_ordinal': result.old.ordinal,
                    'new_ordinal': result.new.ordinal,
                    'status': result.status,
                })
        return users

    # -- driver -------------------------------------------------------------

    def compare(self):
        section_count = max(len(self.old.sections), len(self.new.sections))
        for index in range(section_count):
            old_section = self.old.section(index)
            new_section = self.new.section(index)
            results = self._align_section(old_section, new_section)
            self.sections.append(results)
            self.paragraphs.extend(results)

        self.match = ResourceMatch(self.old.table, self.new.table)
        self._observe_usage()
        self.match.match()

        for result in self.paragraphs:
            self._classify_paragraph(result)
        return self

    def summary(self):
        counts = {}
        for result in self.paragraphs:
            counts[result.status] = counts.get(result.status, 0) + 1

        self_changed = [p for p in self.match.pairs if p.self_changed]
        dependency_changed = [p for p in self.match.pairs
                              if not p.self_changed and
                              p.cause == 'content']
        reorder_caused = [p for p in self.match.pairs
                          if not p.self_changed and
                          p.cause == 'reorder' and
                          p.effective_changed]
        resource_counts = {
            'matched': len(self.match.pairs),
            'unchanged': len([p for p in self.match.pairs
                              if not p.effective_changed]),
            'reordered': len(self.match.reordered_pairs()),
            'self_changed': len(self_changed),
            'dependency_changed': len(dependency_changed),
            'reorder_caused': len(reorder_caused),
            'content_changed': len(self.match.content_changed_pairs()),
            'effective_changed': len(self.match.changed_pairs()),
            'added': len(self.match.unmatched_new()),
            'removed': len(self.match.unmatched_old()),
            'duplicate_classes': len(self.match.duplicate_classes),
        }
        return {
            'documents': {
                'old': {
                    'sections': len(self.old.sections),
                    'paragraphs': sum(len(s) for s in self.old.sections),
                    'resources': _resource_counts(self.old.table),
                },
                'new': {
                    'sections': len(self.new.sections),
                    'paragraphs': sum(len(s) for s in self.new.sections),
                    'resources': _resource_counts(self.new.table),
                },
            },
            'resources': resource_counts,
            'paragraphs': counts,
            'categories': {
                'resource_reorder': len(self.match.reordered_pairs()),
                'resource_reorder_caused': len(reorder_caused),
                'resource_reorder_affected': len([
                    result for result in self.paragraphs
                    if result.status == STATUS_REORDER_AFFECTED]),
                'resource_content_change': len(self_changed),
                'resource_dependency_change': len(dependency_changed),
                'body_change': len([result for result in self.paragraphs
                                    if result.status in BODY_STATUSES]),
                'affected_paragraphs': len([result for result in
                                            self.paragraphs
                                            if result.affected]),
            },
        }

    def to_dict(self, include_indices=False):
        users = self._changed_resource_users()
        resource_report = self.match.to_dict()

        # annotate changed resources with the paragraphs that prove usage
        for item in resource_report['changed']:
            key = (FONT, item['old_lang'], item['old_index']) \
                if item['kind'] == FONT \
                else (item['kind'], item['old_index'])
            item['used_by'] = users.get(key, [])

        paragraph_items = [result.to_dict(include_indices=include_indices)
                           for result in self.paragraphs]

        changed = resource_report['changed']
        return {
            'summary': self.summary(),
            'resource_reorder': resource_report['reordered'],
            'resource_content_change': [item for item in changed
                                        if item['self_changed']],
            'resource_dependency_change': [item for item in changed
                                           if not item['self_changed']],
            'body_change': [item for item in paragraph_items
                            if item['status'] in BODY_STATUSES],
            'reorder_affected': [item for item in paragraph_items
                                 if item['status'] ==
                                 STATUS_REORDER_AFFECTED],
            'paragraphs': paragraph_items,
            'resources': {
                'added': resource_report['added'],
                'removed': resource_report['removed'],
                'duplicates': resource_report['duplicate_classes'],
                'ambiguous_usage': resource_report['ambiguous_usage'],
                'dangling_references': {
                    'old': self.old.table.validate(),
                    'new': self.new.table.validate(),
                },
            },
        }


def _charshape_runs(paragraph):
    return [(span.start, span.end, span.charshape_id)
            for span in paragraph.spans]


def _referenced_resources(table, paragraph):
    '''Resources a paragraph directly uses: parashape, style, charshapes.'''
    resources = []
    for kind, index in (('parashape', paragraph.parashape_id),
                        ('style', paragraph.style_id)):
        resource = table.get(kind, index)
        if resource is not None:
            resources.append(resource)
    for span in paragraph.spans:
        resource = table.get('charshape', span.charshape_id)
        if resource is not None and resource not in resources:
            resources.append(resource)
    return resources


def _resource_counts(table):
    return dict((kind, len(items))
                for kind, items in table.resources.items() if items)


def compare_hwp5(old, new):
    '''Compare two HWPv5 documents given filenames, storages, or Hwp5File
    instances.'''
    old_doc = RevisionDocument.open(old)
    new_doc = RevisionDocument.open(new)
    return RevisionComparison(old_doc, new_doc).compare()


def compare_snapshots(old_table, old_sections, new_table, new_sections):
    '''Compare two in-memory document snapshots (used by tests and tools).'''
    return RevisionComparison(
        RevisionDocument(old_table, old_sections),
        RevisionDocument(new_table, new_sections),
    ).compare()
