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
'''DocInfo resources: identity, references, and cross-revision matching.

A ``FaceName`` record has no id in the record stream; its id is its position
inside the per-language font table declared by ``IdMappings``.  Every other
DocInfo resource (``CharShape``, ``ParaShape``, ``Style``, ``TabDef``,
``Numbering``, ``Bullet``, ``BorderFill``) has a zero-based id inside its own
table, except ``BorderFill``/``Numbering``/``Bullet`` usages which reserve 0
for "none".

Paragraphs only store those numbers.  This module resolves a number to a
resource *content* inside one document, and separately matches resources
across two documents.  Matching is never done by numeric id.

Duplicate resources (same kind and same content) are kept as an equivalence
class.  They are only paired when identity or usage evidence is unique; a
duplicate group is never paired one-to-one merely by position.

Three notions of change are kept apart:

``self_changed``
    the resource definition itself changed, ignoring the resources it refers
    to (e.g. a font name was edited, a char shape gained a colour).
``effective_changed``
    the resource's resolved appearance changed, either directly or because a
    resource it refers to changed (e.g. the same char shape now points at an
    edited font).
``reordered``
    an unchanged resource moved to another table position.
'''
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict

from .canonical import canonical
from .canonical import canonical_diff
from .canonical import fingerprint_of


LANGUAGES = ('ko', 'en', 'cn', 'jp', 'other', 'symbol', 'user')

FONT = 'font'

TABLE_KINDS = (
    ('borderfill', 'borderfills', 'BorderFill'),
    ('charshape', 'charshapes', 'CharShape'),
    ('tabdef', 'tabdefs', 'TabDef'),
    ('numbering', 'numberings', 'Numbering'),
    ('bullet', 'bullets', 'Bullet'),
    ('parashape', 'parashapes', 'ParaShape'),
    ('style', 'styles', 'Style'),
    ('bindata', 'bindata', 'BinData'),
    ('memoshape', 'memoshapes', 'MemoShape'),
)

MATCHED_KINDS = tuple(kind for kind, _, _ in TABLE_KINDS) + (FONT,)

# Resource ids where 0 means "none".
ZERO_IS_NONE = frozenset(['borderfill', 'numbering', 'bullet'])

# Fields that are references to other resources, per kind.
REFERENCE_FIELDS = {
    'charshape': ('font_face',),
    'parashape': ('tabdef_id', 'numbering_bullet_id', 'borderfill_id'),
    'style': ('next_style_id', 'parashape_id', 'charshape_id'),
    'numbering': ('levels',),
    'bullet': ('charshape_id',),
}


class Unresolved(object):
    '''A reference that cannot be resolved or paired with certainty.'''

    def __init__(self, kind, ref):
        self.kind = kind
        self.ref = ref

    def __repr__(self):
        return 'Unresolved(%r)' % (self.ref,)

    def __eq__(self, other):
        return (isinstance(other, Unresolved) and self.kind == other.kind and
                self.ref == other.ref)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.kind, self.ref))


class Resource(object):
    '''One entry in a DocInfo resource table.'''

    def __init__(self, kind, index, content, model=None, lang=None):
        self.kind = kind
        self.index = index
        self.content = content
        self.model = model
        self.lang = lang
        self._semantic = None
        self._semantic_fingerprint = None

    @classmethod
    def from_model(cls, kind, index, model, lang=None):
        return cls(kind, index, model['content'], model=model, lang=lang)

    @property
    def seqno(self):
        return self.model['seqno'] if self.model is not None else None

    @property
    def key(self):
        if self.kind == FONT:
            return (FONT, self.lang, self.index)
        return (self.kind, self.index)

    @property
    def table_key(self):
        if self.kind == FONT:
            return (FONT, self.lang)
        return (self.kind,)

    def __repr__(self):
        return '<Resource %r>' % (self.key,)


def _models_of_type(models, class_name):
    return [model for model in models if model['type'].__name__ == class_name]


class ResourceTable(object):
    '''Resources of one document, addressable by the ids used by paragraphs.'''

    def __init__(self, resources=None, idmappings=None):
        self.resources = dict((kind, []) for kind in MATCHED_KINDS)
        if resources:
            for kind, items in resources.items():
                self.resources[kind] = list(items)
        self.idmappings = idmappings or {}
        self._semantic_memo = {}
        self._semantic_stack = []

    @classmethod
    def from_docinfo(cls, docinfo):
        models = list(docinfo.models())

        idmappings = None
        for model in models:
            if model['type'].__name__ == 'IdMappings':
                idmappings = model['content']
                break

        resources = {}
        for kind, count_name, class_name in TABLE_KINDS:
            items = _models_of_type(models, class_name)
            limit = None
            if idmappings is not None and count_name in idmappings:
                limit = idmappings[count_name]
            if limit is not None:
                items = items[:limit]
            resources[kind] = [Resource.from_model(kind, index, model)
                               for index, model in enumerate(items)]

        fonts = []
        facenames = _models_of_type(models, 'FaceName')
        offset = 0
        for lang in LANGUAGES:
            count = None
            if idmappings is not None:
                count = idmappings.get(lang + '_fonts')
            if count is None:
                count = len(facenames) - offset
            for index, model in enumerate(facenames[offset:offset + count]):
                fonts.append(Resource.from_model(FONT, index, model,
                                                 lang=lang))
            offset += count
        resources[FONT] = fonts
        return cls(resources, idmappings=idmappings)

    @classmethod
    def from_contents(cls, contents, font_contents=None, idmappings=None):
        '''Build a table from plain content dictionaries.

        ``contents`` maps a table kind to a list of content dicts.
        ``font_contents`` maps a language to a list of font content dicts.
        '''
        resources = {}
        for kind in MATCHED_KINDS:
            items = contents.get(kind, [])
            resources[kind] = [Resource(kind, index, content)
                               for index, content in enumerate(items)]
        fonts = []
        for lang in LANGUAGES:
            for index, content in enumerate((font_contents or {}).get(lang,
                                                                      [])):
                fonts.append(Resource(FONT, index, content, lang=lang))
        resources[FONT] = fonts
        return cls(resources, idmappings=idmappings)

    # -- address resolution -------------------------------------------------

    def get(self, kind, index):
        if index is None:
            return None
        try:
            index = int(index)
        except (TypeError, ValueError):
            return None
        if index < 0:
            return None
        items = self.resources.get(kind, [])
        if index >= len(items):
            return None
        return items[index]

    def get_used(self, kind, index):
        if index is None:
            return None
        try:
            index = int(index)
        except (TypeError, ValueError):
            return None
        if kind in ZERO_IS_NONE:
            if index == 0:
                return None
            index -= 1
        return self.get(kind, index)

    def font(self, lang, index):
        if index is None:
            return None
        try:
            index = int(index)
        except (TypeError, ValueError):
            return None
        if index < 0:
            return None
        items = [r for r in self.resources.get(FONT, []) if r.lang == lang]
        if index >= len(items):
            return None
        return items[index]

    def charshape_font(self, charshape, lang):
        if charshape is None or charshape.content is None:
            return None
        font_face = charshape.content.get('font_face') or {}
        return self.font(lang, font_face.get(lang))

    # -- labels -------------------------------------------------------------

    def label(self, resource):
        if resource is None:
            return None
        content = resource.content or {}
        if resource.kind == FONT:
            return content.get('name')
        if resource.kind == 'style':
            return content.get('local_name') or content.get('name')
        if resource.kind == 'numbering':
            levels = content.get('levels') or []
            if levels:
                return levels[0].get('numbering_format')
            return 'numbering'
        if resource.kind == 'bullet':
            return 'bullet:%r' % content.get('char')
        if resource.kind == 'bindata':
            inner = content.get('bindata') or {}
            return (inner.get('storage_id'), inner.get('ext'))
        return resource.kind

    def identity_label(self, resource):
        '''A label HWP itself treats as user-visible identity.

        Only fonts (name inside their language table), styles (local name),
        and binary data (storage id) get one.  Content-defined resources have
        no identity label, so they can only be paired by equality or usage.
        '''
        if resource is None:
            return None
        if resource.kind == FONT:
            name = (resource.content or {}).get('name')
            if name is None:
                return None
            return (FONT, resource.lang, name)
        if resource.kind == 'style':
            content = resource.content or {}
            name = content.get('local_name') or content.get('name')
            if name is None:
                return None
            return ('style', name)
        if resource.kind == 'bindata':
            inner = (resource.content or {}).get('bindata') or {}
            storage_id = inner.get('storage_id')
            if storage_id is None:
                return None
            return ('bindata', storage_id, inner.get('ext'))
        return None

    # -- semantic content ---------------------------------------------------

    def semantic(self, resource):
        '''Content of ``resource`` with all numeric references expanded.'''
        if not isinstance(resource, Resource):
            return resource
        key = resource.key
        if key in self._semantic_memo:
            return self._semantic_memo[key]
        if key in self._semantic_stack:
            return {'ref': resource.kind, 'label': self.label(resource),
                    'cycle': True}
        self._semantic_stack.append(key)
        try:
            value = self._semantic(resource)
        finally:
            self._semantic_stack.pop()
        self._semantic_memo[key] = value
        return value

    def _ref(self, kind, resource):
        if resource is None:
            return None
        return {'ref': kind,
                'label': self.label(resource),
                'fingerprint': self.semantic_fingerprint(resource)[:16]}

    def _semantic(self, resource):
        kind = resource.kind
        content = canonical(resource.content or {})
        if kind == FONT:
            return content

        if kind == 'charshape':
            fonts = content.get('font_face') or {}
            for lang in LANGUAGES:
                if lang in fonts:
                    fonts[lang] = self._ref(
                        FONT, self.font(lang, fonts[lang]))
            content['font_face'] = fonts
            return content

        if kind == 'parashape':
            content['tabdef_id'] = self._ref(
                'tabdef', self.get('tabdef', content.get('tabdef_id')))
            head_shape = _head_shape(content)
            number_id = content.get('numbering_bullet_id', 0)
            if head_shape == 3:
                content['numbering_bullet_id'] = self._ref(
                    'bullet', self.get_used('bullet', number_id))
            elif head_shape == 2:
                content['numbering_bullet_id'] = self._ref(
                    'numbering', self.get_used('numbering', number_id))
            else:
                content['numbering_bullet_id'] = None
            content['borderfill_id'] = self._ref(
                'borderfill', self.get_used('borderfill',
                                            content.get('borderfill_id')))
            return content

        if kind == 'style':
            content['next_style_id'] = self._ref(
                'style', self.get('style', content.get('next_style_id')))
            content['parashape_id'] = self._ref(
                'parashape', self.get('parashape',
                                      content.get('parashape_id')))
            content['charshape_id'] = self._ref(
                'charshape', self.get('charshape',
                                      content.get('charshape_id')))
            return content

        if kind == 'numbering':
            levels = []
            for level in content.get('levels') or []:
                level = dict(level)
                level['charshape_id'] = self._ref(
                    'charshape',
                    self.get_used('charshape', level.get('charshape_id')))
                levels.append(level)
            content['levels'] = levels
            return content

        if kind == 'bullet':
            content['charshape_id'] = self._ref(
                'charshape',
                self.get_used('charshape', content.get('charshape_id')))
            return content

        return content

    def own_content(self, resource):
        '''Content with reference fields removed.

        Numeric ids are deliberately dropped: ids may be rewritten during a
        reorder, and their meaning is carried by the paired resources.
        '''
        content = canonical(resource.content or {})
        kind = resource.kind
        for field in REFERENCE_FIELDS.get(kind, ()):
            content.pop(field, None)
        return content

    def own_fingerprint(self, resource):
        return fingerprint_of(resource.kind, self.own_content(resource))

    def semantic_fingerprint(self, resource):
        if not isinstance(resource, Resource):
            return fingerprint_of('unresolved', repr(resource))
        if resource._semantic_fingerprint is None:
            resource._semantic_fingerprint = fingerprint_of(
                resource.kind, self.semantic(resource))
        return resource._semantic_fingerprint

    def validate(self):
        '''Return a list of dangling references in this table.

        A dangling reference means a paragraph or resource points at an id
        that does not exist.  Comparison reports them instead of silently
        treating the reference as ``None``.
        '''
        problems = []
        for kind in MATCHED_KINDS:
            for resource in self.resources.get(kind, []):
                for target_kind, ref in self._raw_references(resource):
                    if ref is None:
                        continue
                    if self._resolve_raw(target_kind, ref) is None:
                        problems.append(dict(
                            source_kind=kind, source_index=resource.index,
                            target_kind=target_kind, ref=ref))
        return problems

    def _raw_references(self, resource):
        content = resource.content or {}
        kind = resource.kind
        if kind == 'charshape':
            for lang, index in (content.get('font_face') or {}).items():
                yield (FONT, (lang, index))
        elif kind == 'parashape':
            yield ('tabdef', content.get('tabdef_id'))
            head = _head_shape(content)
            if head == 2:
                yield ('numbering', _minus_one(
                    content.get('numbering_bullet_id')))
            elif head == 3:
                yield ('bullet', _minus_one(
                    content.get('numbering_bullet_id')))
            yield ('borderfill', _minus_one(content.get('borderfill_id')))
        elif kind == 'style':
            yield ('parashape', content.get('parashape_id'))
            yield ('charshape', content.get('charshape_id'))
            yield ('style', content.get('next_style_id'))
        elif kind == 'bullet':
            yield ('charshape', _non_negative(content.get('charshape_id')))
        elif kind == 'numbering':
            for level in content.get('levels') or []:
                yield ('charshape',
                       _non_negative(level.get('charshape_id')))

    def _resolve_raw(self, kind, ref):
        # ``_raw_references`` already removed the "0 means none" offset.
        if kind == FONT and isinstance(ref, tuple):
            lang, index = ref
            return self.font(lang, index)
        return self.get(kind, ref)


def _head_shape(content):
    flags = content.get('parashapeflags') or {}
    if isinstance(flags, dict):
        return flags.get('head_shape', 0)
    return 0


class ResourcePair(object):

    def __init__(self, old, new, basis):
        self.old = old
        self.new = new
        self.basis = basis        # content | identity | usage | dependency
        self.self_changed = False
        self.effective_changed = False
        self.cause = None
        self.changed_dependencies = []

    @property
    def kind(self):
        return self.old.kind

    @property
    def status(self):
        if self.self_changed:
            return 'changed'
        if self.effective_changed:
            return 'dependency-changed'
        return 'unchanged'

    @property
    def position_changed(self):
        return _position(self.old) != _position(self.new)

    @property
    def reordered(self):
        '''True if position moved without any content edit.

        A resource can stay effectively unchanged after a reorder (its ids
        were rewritten consistently), or it can effectively change only
        because an id now points at another position.  Both are reorders; a
        real definition edit is not.
        '''
        if not self.position_changed:
            return False
        if self.status == 'unchanged':
            return True
        return self.cause == 'reorder'

    def evaluate(self, old_table, new_table):
        self.self_changed = (old_table.own_fingerprint(self.old) !=
                             new_table.own_fingerprint(self.new))
        self.effective_changed = (
            old_table.semantic_fingerprint(self.old) !=
            new_table.semantic_fingerprint(self.new))
        return self

    def to_dict(self, old_table, new_table):
        data = {
            'kind': self.old.kind,
            'label': old_table.label(self.old),
            'old_index': self.old.index,
            'new_index': self.new.index,
            'old_lang': self.old.lang,
            'new_lang': self.new.lang,
            'status': self.status,
            'basis': self.basis,
            'cause': self.cause,
            'self_changed': self.self_changed,
            'effective_changed': self.effective_changed,
            'reordered': self.reordered,
        }
        if self.self_changed:
            data['own_changes'] = canonical_diff(
                old_table.own_content(self.old),
                new_table.own_content(self.new))
        if self.effective_changed:
            data['effective_changes'] = canonical_diff(
                old_table.semantic(self.old),
                new_table.semantic(self.new))
        if self.changed_dependencies:
            data['changed_dependencies'] = list(self.changed_dependencies)
        return data


def _position(resource):
    if resource.kind == FONT:
        return (resource.lang, resource.index)
    return resource.index


class ResourceMatch(object):
    '''Correspondence between the resources of two revisions.'''

    def __init__(self, old_table, new_table):
        self.old_table = old_table
        self.new_table = new_table
        self.pairs = []
        self._old_to_pair = {}
        self._new_to_pair = {}
        self._observations = defaultdict(lambda: defaultdict(int))
        self._dependency_observations = defaultdict(lambda: defaultdict(int))
        self.duplicate_classes = []
        self.ambiguous_usage = []

    # -- lookups ------------------------------------------------------------

    def pair_of_old(self, resource):
        if resource is None:
            return None
        return self._old_to_pair.get(resource.key)

    def pair_of_new(self, resource):
        if resource is None:
            return None
        return self._new_to_pair.get(resource.key)

    def new_of_old(self, resource):
        pair = self.pair_of_old(resource)
        return pair.new if pair else None

    def old_of_new(self, resource):
        pair = self.pair_of_new(resource)
        return pair.old if pair else None

    def _add(self, old, new, basis):
        if old.key in self._old_to_pair or new.key in self._new_to_pair:
            return None
        pair = ResourcePair(old, new, basis).evaluate(self.old_table,
                                                      self.new_table)
        self.pairs.append(pair)
        self._old_to_pair[old.key] = pair
        self._new_to_pair[new.key] = pair
        return pair

    def _all(self, table):
        for kind in MATCHED_KINDS:
            for resource in table.resources.get(kind, []):
                yield resource

    def _unmatched(self, side):
        table = self.old_table if side == 'old' else self.new_table
        pairs = self._old_to_pair if side == 'old' else self._new_to_pair
        equivalent = self._equivalent_keys()
        return [resource for resource in self._all(table)
                if resource.key not in pairs and
                resource.key not in equivalent]

    def _equivalent_keys(self):
        '''Keys of duplicate resources that have a counterpart group.

        Their content is interchangeable, so they are neither added nor
        removed; the duplicate class itself records the indirection.  If a
        group exists on only one side, its resources remain unmatched.
        '''
        keys = set()
        for duplicate in self.duplicate_classes:
            if duplicate['old'] and duplicate['new']:
                keys.update(resource.key for resource in duplicate['old'])
                keys.update(resource.key for resource in duplicate['new'])
        return keys

    def unmatched_old(self):
        return self._unmatched('old')

    def unmatched_new(self):
        return self._unmatched('new')

    def _add_duplicate_class(self, kind, lang, label, old_items, new_items,
                             fingerprint=None):
        '''Record one equivalence class, merging duplicate reports.

        The same members can be reported by the content pass and again by the
        identity-label pass, with different labels; callers should see one
        class per member group.  Merging is by membership, not by label.
        '''
        old_keys = set(resource.key for resource in old_items)
        new_keys = set(resource.key for resource in new_items)
        for existing in self.duplicate_classes:
            existing_old = set(r.key for r in existing['old'])
            existing_new = set(r.key for r in existing['new'])
            if ((old_keys & existing_old) and (new_keys & existing_new)) or \
                    (not old_keys and not existing_old and
                     new_keys & existing_new):
                if fingerprint is not None:
                    existing['fingerprint'] = fingerprint
                existing['old'] = list(dict.fromkeys(
                    existing['old'] + old_items))
                existing['new'] = list(dict.fromkeys(
                    existing['new'] + new_items))
                existing['old_count'] = len(existing['old'])
                existing['new_count'] = len(existing['new'])
                if label is not None and existing.get('label') is None:
                    existing['label'] = label
                return existing
        entry = dict(kind=kind, lang=lang, label=label,
                     fingerprint=fingerprint,
                     old=list(old_items), new=list(new_items),
                     old_count=len(old_items), new_count=len(new_items))
        self.duplicate_classes.append(entry)
        return entry

    # -- phase 1: equality --------------------------------------------------

    def match_identical(self):
        groups = defaultdict(lambda: {'old': [], 'new': []})
        for resource in self._all(self.old_table):
            key = (resource.table_key,
                   self.old_table.semantic_fingerprint(resource))
            groups[key]['old'].append(resource)
        for resource in self._all(self.new_table):
            key = (resource.table_key,
                   self.new_table.semantic_fingerprint(resource))
            groups[key]['new'].append(resource)

        for key in sorted(groups, key=repr):
            group = groups[key]
            old_items, new_items = group['old'], group['new']
            if not old_items or not new_items:
                continue
            if len(old_items) == 1 and len(new_items) == 1:
                self._add(old_items[0], new_items[0], 'content')
            else:
                table_key, fingerprint = key
                self._add_duplicate_class(
                    table_key[0],
                    table_key[1] if len(table_key) > 1 else None,
                    None, old_items, new_items, fingerprint=fingerprint)

    def match_own_identical(self):
        '''Pair resources whose own definition is equal and unique.

        A char shape whose font pointer moved keeps an identical own
        definition (font pointers are excluded); pairing it here lets the
        font change surface as a dependency change instead of an unrelated
        removal/addition.  Ambiguous groups are left unpaired.
        '''
        matched = False
        for kind in ('charshape', 'parashape', 'style', 'numbering', 'bullet',
                     'tabdef', 'borderfill'):
            groups = defaultdict(lambda: {'old': [], 'new': []})
            for resource in self.old_table.resources.get(kind, []):
                if resource.key in self._old_to_pair:
                    continue
                key = self.old_table.own_fingerprint(resource)
                groups[key]['old'].append(resource)
            for resource in self.new_table.resources.get(kind, []):
                if resource.key in self._new_to_pair:
                    continue
                key = self.new_table.own_fingerprint(resource)
                groups[key]['new'].append(resource)

            for key in sorted(groups, key=repr):
                group = groups[key]
                old_items, new_items = group['old'], group['new']
                if not old_items or not new_items:
                    continue
                if len(old_items) == 1 and len(new_items) == 1:
                    if self._add(old_items[0], new_items[0],
                                 'definition') is not None:
                        matched = True
        return matched

    # -- phase 2: identity labels ------------------------------------------

    def match_identity(self):
        matched = self._match_style_identity()
        matched = self._match_font_identity() or matched
        matched = self._match_bindata_identity() or matched
        return matched

    def _match_style_identity(self):
        return self._match_identity_kind('style')

    def _match_bindata_identity(self):
        return self._match_identity_kind('bindata')

    def _match_identity_kind(self, kind):
        old_by_label = defaultdict(list)
        for resource in self.old_table.resources.get(kind, []):
            if resource.key in self._old_to_pair:
                continue
            label = self.old_table.identity_label(resource)
            if label is not None:
                old_by_label[label].append(resource)
        new_by_label = defaultdict(list)
        for resource in self.new_table.resources.get(kind, []):
            if resource.key in self._new_to_pair:
                continue
            label = self.new_table.identity_label(resource)
            if label is not None:
                new_by_label[label].append(resource)

        matched = False
        for label in sorted(set(old_by_label) & set(new_by_label), key=repr):
            old_group = old_by_label[label]
            new_group = new_by_label[label]
            if len(old_group) != 1 or len(new_group) != 1:
                self._add_duplicate_class(kind, None, label, old_group,
                                          new_group)
                continue
            if self._add(old_group[0], new_group[0], 'identity') is not None:
                matched = True
        return matched

    def _match_font_identity(self):
        matched = False
        for lang in LANGUAGES:
            old_by_label = defaultdict(list)
            for resource in self.old_table.resources.get(FONT, []):
                if resource.lang != lang or resource.key in self._old_to_pair:
                    continue
                old_by_label[self.old_table.label(resource)].append(resource)
            new_by_label = defaultdict(list)
            for resource in self.new_table.resources.get(FONT, []):
                if resource.lang != lang or resource.key in self._new_to_pair:
                    continue
                new_by_label[self.new_table.label(resource)].append(resource)

            for label in sorted(set(old_by_label) & set(new_by_label),
                                key=repr):
                old_group = old_by_label[label]
                new_group = new_by_label[label]
                if len(old_group) != 1 or len(new_group) != 1:
                    self._add_duplicate_class(FONT, lang, label, old_group,
                                              new_group)
                    continue
                if self._add(old_group[0], new_group[0],
                             'identity') is not None:
                    matched = True
        return matched

    # -- phase 3: usage evidence -------------------------------------------

    def observe_usage(self, old, new):
        '''Record that one aligned paragraph used ``old`` in the old file and
        ``new`` in the new one.'''
        if old is None or new is None:
            return
        self._observations[old.key][new.key] += 1

    def match_usage(self, observations=None):
        if observations is None:
            observations = self._observations
        matched = False
        progress = True
        while progress:
            progress = False
            for old_key in sorted(observations, key=repr):
                old = self._resource_by_key(self.old_table, old_key)
                if old is None or old_key in self._old_to_pair:
                    continue
                counts = observations[old_key]
                candidates = []
                for new_key in counts:
                    new = self._resource_by_key(self.new_table, new_key)
                    if new is None or new_key in self._new_to_pair:
                        continue
                    candidates.append((new_key, new))

                if len(candidates) != 1:
                    if len(candidates) > 1:
                        self._record_ambiguity(old, [
                            key for key, _ in candidates], 'forward')
                    continue

                new_key, new = candidates[0]
                reverse = [key for key, observed in observations.items()
                           if new_key in observed and
                           key not in self._old_to_pair and
                           self._resource_by_key(self.old_table,
                                                 key) is not None]
                if len(reverse) != 1:
                    self._record_ambiguity(old, [new_key], 'reverse',
                                           reverse)
                    continue
                if self._add(old, new, 'usage') is not None:
                    matched = True
                    progress = True
        return matched

    def _record_ambiguity(self, old, new_keys, direction, old_keys=None):
        entry = dict(kind=old.kind, old=old.key, new=sorted(new_keys,
                                                            key=repr),
                     direction=direction)
        if old_keys is not None:
            entry['reverse_old'] = sorted(old_keys, key=repr)
        if entry not in self.ambiguous_usage:
            self.ambiguous_usage.append(entry)

    def _resource_by_key(self, table, key):
        kind = key[0]
        if kind == FONT:
            for resource in table.resources.get(FONT, []):
                if resource.key == key:
                    return resource
            return None
        return table.get(kind, key[1])

    # -- phase 4: dependency evidence --------------------------------------

    def match_dependencies(self):
        matched = False
        for pair in list(self.pairs):
            for kind, old_ref, new_ref in self._dependencies(pair.old,
                                                             pair.new):
                if old_ref is None or new_ref is None:
                    continue
                old = self._dependency_resource(self.old_table, kind, old_ref)
                new = self._dependency_resource(self.new_table, kind, new_ref)
                if old is None or new is None:
                    continue
                if old.key in self._old_to_pair or \
                        new.key in self._new_to_pair:
                    continue
                # Dependency evidence is an observation, not a pairing.  It
                # is only promoted when it is unique in both directions, so
                # duplicated resources are never paired by position.  The
                # referenced contents need not be equal: a unique reference
                # is exactly how an edited font is attributed to its users.
                self._dependency_observations[old.key][new.key] += 1
                matched = True
        return matched

    def _dependency_resource(self, table, kind, ref):
        # ``_dependencies`` normalizes every reference to a zero-based table
        # index (or a (lang, index) pair for fonts), so no second "0 means
        # none" conversion happens here.
        if isinstance(ref, tuple) and kind == FONT:
            lang, index = ref
            return table.font(lang, index)
        return table.get(kind, ref)

    def _dependencies(self, old, new):
        kind = old.kind
        old_content = old.content or {}
        new_content = new.content or {}

        if kind == 'charshape':
            old_fonts = old_content.get('font_face') or {}
            new_fonts = new_content.get('font_face') or {}
            for lang in LANGUAGES:
                if lang in old_fonts and lang in new_fonts:
                    yield (FONT, (lang, old_fonts[lang]),
                           (lang, new_fonts[lang]))
            return

        if kind == 'parashape':
            yield ('tabdef', old_content.get('tabdef_id'),
                   new_content.get('tabdef_id'))
            if _head_shape(old_content) == _head_shape(new_content):
                if _head_shape(old_content) == 2:
                    yield ('numbering',
                           _minus_one(old_content.get('numbering_bullet_id')),
                           _minus_one(new_content.get('numbering_bullet_id')))
                elif _head_shape(old_content) == 3:
                    yield ('bullet',
                           _minus_one(old_content.get('numbering_bullet_id')),
                           _minus_one(new_content.get('numbering_bullet_id')))
            yield ('borderfill',
                   _minus_one(old_content.get('borderfill_id')),
                   _minus_one(new_content.get('borderfill_id')))
            return

        if kind == 'style':
            yield ('parashape', old_content.get('parashape_id'),
                   new_content.get('parashape_id'))
            yield ('charshape', old_content.get('charshape_id'),
                   new_content.get('charshape_id'))
            yield ('style', old_content.get('next_style_id'),
                   new_content.get('next_style_id'))
            return

        if kind == 'bullet':
            yield ('charshape', _non_negative(old_content.get('charshape_id')),
                   _non_negative(new_content.get('charshape_id')))
            return

        if kind == 'numbering':
            old_levels = old_content.get('levels') or []
            new_levels = new_content.get('levels') or []
            for old_level, new_level in zip(old_levels, new_levels):
                yield ('charshape',
                       _non_negative(old_level.get('charshape_id')),
                       _non_negative(new_level.get('charshape_id')))

    # -- phase 5: exclusive leftovers --------------------------------------

    def match_exclusive_leftovers(self):
        '''Pair a lone unmatched resource on each side of one table.

        If exactly one resource of a kind is unmatched on both sides, there
        is no ambiguity: it is the same slot that was removed and re-added
        under a different definition (e.g. a renamed font).  With two or more
        leftovers the evidence is insufficient, so they stay unmatched.
        '''
        matched = False
        groups = []
        for kind in MATCHED_KINDS:
            if kind == FONT:
                for lang in LANGUAGES:
                    groups.append((FONT, lang))
            else:
                groups.append((kind, None))

        for kind, lang in groups:
            old_items = self._unmatched_in(kind, lang, 'old')
            new_items = self._unmatched_in(kind, lang, 'new')
            if len(old_items) != 1 or len(new_items) != 1:
                continue
            if self._add(old_items[0], new_items[0], 'exclusive') is not None:
                matched = True
        return matched

    def _unmatched_in(self, kind, lang, side):
        table = self.old_table if side == 'old' else self.new_table
        pairs = self._old_to_pair if side == 'old' else self._new_to_pair
        equivalent = self._equivalent_keys()
        items = []
        for resource in table.resources.get(kind, []):
            if resource.key in pairs or resource.key in equivalent:
                continue
            if kind == FONT and resource.lang != lang:
                continue
            items.append(resource)
        return items

    # -- dependency annotation ---------------------------------------------

    def inherit_effective_changes(self):
        '''Propagate effective changes from referenced resources to users.

        A char shape that points at an edited font did not change its own
        definition, but its effective appearance did.  Same for styles that
        point at changed paragraph/char shapes.  This is iterated to a fixed
        point so nested references are handled without recursion order
        assumptions.
        '''
        progress = True
        while progress:
            progress = False
            for pair in self.pairs:
                if pair.effective_changed:
                    continue
                for kind, old_ref, new_ref in self._dependencies(pair.old,
                                                                 pair.new):
                    if old_ref is None or new_ref is None:
                        continue
                    old = self._dependency_resource(self.old_table, kind,
                                                    old_ref)
                    new = self._dependency_resource(self.new_table, kind,
                                                    new_ref)
                    if old is None or new is None:
                        continue
                    child = self.pair_of_old(old)
                    if child is None or not child.effective_changed:
                        continue
                    pair.effective_changed = True
                    progress = True
                    break

    def change_cause(self, pair, _seen=None):
        '''Root cause of a pair's effective change.

        Returns ``None`` if the pair is unchanged, ``'content'`` if any
        resource in its dependency closure changed its own definition, and
        ``'reorder'`` if the difference comes purely from ids moving to
        different table positions.  Recursive reference cycles (e.g. a
        style's ``next_style_id`` chain) are cut by ``_seen``.
        '''
        if pair is None or not pair.effective_changed:
            return None
        if pair.self_changed:
            return 'content'
        if _seen is None:
            _seen = set()
        if pair.old.key in _seen:
            return None
        _seen.add(pair.old.key)

        for kind, old_ref, new_ref in self._dependencies(pair.old, pair.new):
            if old_ref is None or new_ref is None:
                continue
            old = self._dependency_resource(self.old_table, kind, old_ref)
            new = self._dependency_resource(self.new_table, kind, new_ref)
            if old is None or new is None:
                continue
            if self.old_table.semantic_fingerprint(old) == \
                    self.new_table.semantic_fingerprint(new):
                continue
            child = self.pair_of_old(old)
            if child is None:
                # No correspondence exists, so the reference itself moved.
                return 'reorder'
            cause = self.change_cause(child, _seen)
            if cause == 'content':
                return 'content'
            if cause is None and child.position_changed:
                return 'reorder'
        return 'reorder'

    def annotate_causes(self):
        for pair in self.pairs:
            if not pair.effective_changed:
                pair.cause = None
            elif pair.self_changed:
                pair.cause = 'content'
            else:
                pair.cause = self.change_cause(pair)

    def annotate_dependency_changes(self):
        for pair in self.pairs:
            if not pair.effective_changed or pair.self_changed:
                continue
            for kind, old_ref, new_ref in self._dependencies(pair.old,
                                                             pair.new):
                old = self._dependency_resource(self.old_table, kind, old_ref)
                new = self._dependency_resource(self.new_table, kind, new_ref)
                if old is None or new is None:
                    continue
                if self.old_table.semantic_fingerprint(old) == \
                        self.new_table.semantic_fingerprint(new):
                    continue
                child = self.pair_of_old(old)
                pair.changed_dependencies.append({
                    'kind': kind,
                    'old_index': old.index,
                    'new_index': new.index,
                    'paired': child is not None,
                })

    # -- driver -------------------------------------------------------------

    def match(self):
        self.match_identical()
        self.match_own_identical()
        self.match_identity()
        while True:
            changed = self.match_usage()
            changed = self.match_dependencies() or changed
            changed = self.match_usage(self._dependency_observations) or \
                changed
            if not changed:
                break
        self.match_exclusive_leftovers()
        self.inherit_effective_changes()
        self.annotate_causes()
        self.annotate_dependency_changes()
        return self

    # -- reporting ----------------------------------------------------------

    def changed_pairs(self):
        '''Pairs whose effective appearance changed, whatever the cause.'''
        return [pair for pair in self.pairs if pair.status != 'unchanged']

    def content_changed_pairs(self):
        return [pair for pair in self.pairs if pair.cause == 'content']

    def reordered_pairs(self):
        return [pair for pair in self.pairs if pair.reordered]

    def to_dict(self):
        old, new = self.old_table, self.new_table

        def items(resources, table):
            return [dict(kind=r.kind,
                         index=r.index,
                         lang=r.lang,
                         label=table.label(r),
                         fingerprint=table.semantic_fingerprint(r))
                    for r in resources]

        pairs = [pair.to_dict(old, new) for pair in self.pairs]
        return {
            'matched': pairs,
            'reordered': [p for p in pairs if p['reordered']],
            'changed': [p for p in pairs
                        if p['status'] in ('changed',
                                           'dependency-changed') and
                        p['cause'] == 'content'],
            'dependent': [p for p in pairs
                          if p['status'] == 'dependency-changed' and
                          p['cause'] == 'reorder'],
            'added': items(self.unmatched_new(), new),
            'removed': items(self.unmatched_old(), old),
            'duplicate_classes': [
                dict(kind=c['kind'], lang=c.get('lang'),
                     fingerprint=c.get('fingerprint'),
                     label=c.get('label'),
                     old=items(c['old'], old), new=items(c['new'], new),
                     old_count=len(c['old']), new_count=len(c['new']))
                for c in self.duplicate_classes
            ],
            'ambiguous_usage': self.ambiguous_usage,
        }


def _minus_one(value):
    value = _non_negative(value)
    if value is None or value == 0:
        return None
    return value - 1




def _non_negative(value):
    if value is None:
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None
