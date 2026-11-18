# -*- coding: utf-8 -*-
'''Extract selected paragraphs of an HWPv5 document into a self-contained
fragment package.

A fragment package is a directory which can be interpreted on its own,
without the original document::

    <outdir>/
        FileHeader          flags patched: 'compressed' bit cleared
        DocInfo             only the resources the selected paragraphs
                            actually depend on, renumbered densely
        BodyText/Section0   only the selected paragraph trees, with every
                            resource reference remapped to the new DocInfo
        BinData/BIN*        embedded objects referenced by the selected
                            paragraphs only, renamed after the new storage
                            ids (shared objects are collected only once)
        manifest.json       correspondence between fragment resource ids
                            and the original ones, plus completeness status

Unknown controls or records which may reference uncollected resources mark
the package as incomplete in the manifest; such a package is never claimed
to be self-sufficient.
'''
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from collections import namedtuple
import json
import logging
import os
import struct
import sys

from . import tagids
from .binmodel import Hwp5File
from .binmodel import UnknownTagModel
from .binmodel import parse_models
from .binmodel.controls._shared import control_models
from .binmodel.tagid56_list_header import TableCell
from .binmodel.tagid71_ctrl_data import ControlData
from .dataio import ParseError
from .recordstream import dump_record
from .storage import is_stream
from .storage.fs import FileSystemStorage
from .treeop import STARTEVENT, ENDEVENT


logger = logging.getLogger(__name__)


PY3 = sys.version_info.major == 3
if PY3:
    basestring = str


MANIFEST_NAME = 'manifest.json'
FRAGMENT_FORMAT = 'hwp5-fragment/1'

LANGS = ('ko', 'en', 'cn', 'jp', 'other', 'symbol', 'user')
FACENAME_KINDS = tuple('facename:' + lang for lang in LANGS)
DEP_KINDS = (FACENAME_KINDS +
             ('borderfill', 'charshape', 'tabdef', 'numbering', 'bullet',
              'parashape', 'style', 'bindata', 'binstorage'))

# DocInfo resource record tag -> resource kind
RESOURCE_TAG_KINDS = (
    (tagids.HWPTAG_BIN_DATA, 'bindata'),
    (tagids.HWPTAG_FACE_NAME, 'facename'),
    (tagids.HWPTAG_BORDER_FILL, 'borderfill'),
    (tagids.HWPTAG_CHAR_SHAPE, 'charshape'),
    (tagids.HWPTAG_TAB_DEF, 'tabdef'),
    (tagids.HWPTAG_NUMBERING, 'numbering'),
    (tagids.HWPTAG_BULLET, 'bullet'),
    (tagids.HWPTAG_PARA_SHAPE, 'parashape'),
    (tagids.HWPTAG_STYLE, 'style'),
)
TAG_TO_KIND = dict(RESOURCE_TAG_KINDS)

# DocInfo records which are known not to reference renumbered resources
SAFE_DOCINFO_TAGS = frozenset([
    tagids.HWPTAG_DOCUMENT_PROPERTIES,
    tagids.HWPTAG_DOC_DATA,
    tagids.HWPTAG_DISTRIBUTE_DOC_DATA,
    tagids.HWPTAG_COMPATIBLE_DOCUMENT,
    tagids.HWPTAG_LAYOUT_COMPATIBILITY,
    tagids.HWPTAG_FORBIDDEN_CHAR,
    tagids.HWPTAG_MEMO_SHAPE,
])

# section records which are known not to reference DocInfo resources
SAFE_BODY_TAGS = frozenset([
    tagids.HWPTAG_PARA_TEXT,
    tagids.HWPTAG_PARA_LINE_SEG,
    tagids.HWPTAG_PARA_RANGE_TAG,
    tagids.HWPTAG_PAGE_DEF,
    tagids.HWPTAG_FOOTNOTE_SHAPE,
    tagids.HWPTAG_SHAPE_COMPONENT,
    tagids.HWPTAG_SHAPE_COMPONENT_LINE,
    tagids.HWPTAG_SHAPE_COMPONENT_RECTANGLE,
    tagids.HWPTAG_SHAPE_COMPONENT_ELLIPSE,
    tagids.HWPTAG_SHAPE_COMPONENT_ARC,
    tagids.HWPTAG_SHAPE_COMPONENT_POLYGON,
    tagids.HWPTAG_SHAPE_COMPONENT_CURVE,
    tagids.HWPTAG_SHAPE_COMPONENT_CONTAINER,
    tagids.HWPTAG_SHAPE_COMPONENT_TEXTART,
    tagids.HWPTAG_CTRL_EQEDIT,
])

# section records carrying references handled by iter_model_references()
HANDLED_BODY_TAGS = frozenset([
    tagids.HWPTAG_PARA_HEADER,
    tagids.HWPTAG_PARA_CHAR_SHAPE,
    tagids.HWPTAG_CTRL_HEADER,
    tagids.HWPTAG_LIST_HEADER,
    tagids.HWPTAG_TABLE,
    tagids.HWPTAG_PAGE_BORDER_FILL,
    tagids.HWPTAG_SHAPE_COMPONENT_PICTURE,
    tagids.HWPTAG_SHAPE_COMPONENT_OLE,
    tagids.HWPTAG_CTRL_DATA,
])

IDMAPPING_COUNT_FIELDS = (
    ('bindata', 'bindata'),
    ('ko_fonts', 'facename:ko'),
    ('en_fonts', 'facename:en'),
    ('cn_fonts', 'facename:cn'),
    ('jp_fonts', 'facename:jp'),
    ('other_fonts', 'facename:other'),
    ('symbol_fonts', 'facename:symbol'),
    ('user_fonts', 'facename:user'),
    ('borderfills', 'borderfill'),
    ('charshapes', 'charshape'),
    ('tabdefs', 'tabdef'),
    ('numberings', 'numbering'),
    ('bullets', 'bullet'),
    ('parashapes', 'parashape'),
    ('styles', 'style'),
)


class FragmentError(Exception):
    pass


Reference = namedtuple('Reference', 'kind dep_id offset binfmt bias')


def leaf_binfmt(item):
    item_type = item.get('bin_type')
    if item_type is None:
        item_type = item['type']
    return getattr(item_type, 'binfmt', None)


def iter_leaves(binevents):
    '''Flatten model binevents into leaf fields with payload offsets.'''
    stack = []
    for event, item in binevents:
        if event is STARTEVENT:
            stack.append([item.get('name'), 0])
        elif event is ENDEVENT:
            if stack:
                stack.pop()
        elif event is None:
            names = [entry[0] for entry in stack if entry[0] is not None]
            if stack:
                index = stack[-1][1]
                stack[-1][1] += 1
            else:
                index = 0
            name = item.get('name')
            path = tuple(names + [name])
            yield dict(path=path, name=name, index=index,
                       offset=item['bin_offset'], binfmt=leaf_binfmt(item),
                       value=item.get('value'))


def ref(kind, dep_id, leaf, bias=0):
    return Reference(kind, dep_id, leaf['offset'], leaf['binfmt'], bias)


def iter_model_references(model):
    '''Yield References to DocInfo resources held by a parsed record model.

    Each Reference carries the payload offset of the field so that it can
    be rewritten later.  ``dep_id`` is the 0-based original resource id,
    except for the ``binstorage`` kind where it is a BinData storage id.
    ``bias`` is the 1-based-ness of the raw field: the raw value is
    ``dep_id + bias``; raw 0 means "none" for 1-based fields.
    '''
    tagid = model['tagid']
    leaves = list(iter_leaves(model['binevents']))
    by_path = {}
    for leaf in leaves:
        by_path.setdefault(leaf['path'], []).append(leaf)

    def simple(path, kind, bias=0, skip_negative=False):
        for leaf in by_path.get(path, ()):
            value = leaf['value']
            if bias and value == 0:
                # 1-based field: 0 means "none"
                continue
            if skip_negative and value < 0:
                continue
            yield ref(kind, value - bias, leaf, bias)

    if tagid == tagids.HWPTAG_PARA_HEADER:
        for x in simple(('parashape_id',), 'parashape'):
            yield x
        for x in simple(('style_id',), 'style'):
            yield x
    elif tagid == tagids.HWPTAG_PARA_CHAR_SHAPE:
        for leaf in by_path.get(('charshapes', None), ()):
            if leaf['index'] % 2 == 1:
                yield ref('charshape', leaf['value'], leaf)
    elif tagid == tagids.HWPTAG_CTRL_HEADER:
        chid = model['content'].get('chid')
        if chid == 'secd':
            # numbering_shape_id is 1-based; 0 means "none"
            for x in simple(('numbering_shape_id',), 'numbering', bias=1):
                yield x
    elif tagid == tagids.HWPTAG_LIST_HEADER:
        if model['type'] is TableCell:
            # borderfill_id is 1-based; 0 means "none"
            for x in simple(('borderfill_id',), 'borderfill', bias=1):
                yield x
    elif tagid == tagids.HWPTAG_TABLE:
        # borderfill_id is 1-based; 0 means "none"
        for leaf in leaves:
            if leaf['name'] == 'borderfill_id':
                if leaf['value'] == 0:
                    continue
                yield ref('borderfill', leaf['value'] - 1, leaf, 1)
    elif tagid == tagids.HWPTAG_PAGE_BORDER_FILL:
        # borderfill_id is 1-based; 0 means "none"
        for x in simple(('borderfill_id',), 'borderfill', bias=1):
            yield x
    elif tagid == tagids.HWPTAG_SHAPE_COMPONENT_PICTURE:
        # bindata_id is a 1-based index into the BinData record list
        for x in simple(('picture', 'bindata_id'), 'bindata', bias=1):
            yield x
    elif tagid == tagids.HWPTAG_SHAPE_COMPONENT_OLE:
        for x in simple(('storage_id',), 'binstorage'):
            yield x
    elif tagid == tagids.HWPTAG_BIN_DATA:
        for x in simple(('bindata', 'storage_id'), 'binstorage'):
            yield x
    elif tagid == tagids.HWPTAG_CHAR_SHAPE:
        for lang in LANGS:
            for x in simple(('font_face', lang), 'facename:' + lang):
                yield x
    elif tagid == tagids.HWPTAG_NUMBERING:
        for x in simple(('levels', 'charshape_id'), 'charshape',
                        skip_negative=True):
            yield x
    elif tagid == tagids.HWPTAG_BULLET:
        for x in simple(('charshape_id',), 'charshape', skip_negative=True):
            yield x
    elif tagid == tagids.HWPTAG_PARA_SHAPE:
        for x in simple(('tabdef_id',), 'tabdef'):
            yield x
        # borderfill_id is 1-based; 0 means "none"
        for x in simple(('borderfill_id',), 'borderfill', bias=1):
            yield x
        # numbering_bullet_id is 1-based; 0 means "none"
        head_shape = model['content']['parashapeflags'].head_shape
        head_shape = int(head_shape)
        if head_shape in (1, 2):        # OUTLINE, NUMBER
            for x in simple(('numbering_bullet_id',), 'numbering', bias=1):
                yield x
        elif head_shape == 3:           # BULLET
            for x in simple(('numbering_bullet_id',), 'bullet', bias=1):
                yield x
    elif tagid == tagids.HWPTAG_STYLE:
        for x in simple(('parashape_id',), 'parashape'):
            yield x
        for x in simple(('charshape_id',), 'charshape'):
            yield x
        for x in simple(('next_style_id',), 'style'):
            yield x


def parse_record_safely(context, record):
    '''Parse a single record; returns (model, error).'''
    try:
        models = list(parse_models(context, [record]))
    except ParseError as e:
        return None, e
    return models[0], None


def parse_tree_safely(context, records):
    '''Parse a record tree; returns (models, error).'''
    try:
        models = list(parse_models(context, list(records)))
    except ParseError as e:
        return None, e
    return models, None


class FragmentExtractor(object):
    '''Extract selected paragraphs into a fragment package.'''

    def __init__(self, hwpfile):
        self.source_name = None
        if isinstance(hwpfile, basestring):
            self.source_name = hwpfile
            if os.path.isdir(hwpfile):
                hwpfile = Hwp5File(FileSystemStorage(hwpfile))
            else:
                hwpfile = Hwp5File(hwpfile)
        self.hwpfile = hwpfile
        self.version = hwpfile.header.version
        self.context = dict(version=self.version)
        self.incomplete_reasons = []
        self.warnings = []

    def incomplete(self, fmt, *args):
        reason = fmt % args
        logger.warning('incomplete: %s', reason)
        self.incomplete_reasons.append(reason)

    def warn(self, fmt, *args):
        warning = fmt % args
        logger.warning('%s', warning)
        self.warnings.append(warning)

    # -- source analysis ------------------------------------------------

    def load_docinfo(self):
        records = list(self.hwpfile.docinfo.records())
        entries = []
        for seqno, record in enumerate(records):
            model, error = parse_record_safely(self.context, record)
            entries.append(dict(record=record, model=model, error=error,
                                seqno=seqno))
        self.docinfo_entries = entries

        idmappings = None
        for entry in entries:
            if entry['record']['tagid'] == tagids.HWPTAG_ID_MAPPINGS:
                idmappings = entry
                break
        if idmappings is None:
            raise FragmentError('DocInfo has no HWPTAG_ID_MAPPINGS record')
        if idmappings['model'] is None:
            raise FragmentError('cannot parse HWPTAG_ID_MAPPINGS: %s'
                                % idmappings['error'])
        self.idmappings_entry = idmappings
        self.idmappings = idmappings['model']['content']

        # original resource counts
        counts = dict((kind, 0) for kind in DEP_KINDS)
        for entry in entries:
            kind = TAG_TO_KIND.get(entry['record']['tagid'])
            if kind is None:
                continue
            if kind == 'facename':
                continue            # counted per language below
            counts[kind] += 1
        for field, kind in IDMAPPING_COUNT_FIELDS:
            if kind.startswith('facename:'):
                counts[kind] = self.idmappings.get(field, 0)
        # BinData storage ids are not indexes; they are resolved to BinData
        # records during the dependency closure
        counts['binstorage'] = None
        self.counts = counts

        # resource records indexed by (kind, old id)
        resource_kinds = ('facename', 'borderfill', 'charshape', 'tabdef',
                          'numbering', 'bullet', 'parashape', 'style',
                          'bindata')
        resources = dict((kind, []) for kind in resource_kinds)
        facename_lang = self.facename_languages()
        facename_lang_counters = dict((lang, 0) for lang in LANGS)
        for entry in entries:
            kind = TAG_TO_KIND.get(entry['record']['tagid'])
            if kind is None:
                continue
            old_id = len(resources[kind])
            entry['resource_kind'] = kind
            entry['resource_id'] = old_id
            if kind == 'facename':
                lang = facename_lang[old_id]
                entry['facename_lang'] = lang
                # font ids are indexes into the per-language font list
                entry['facename_lang_id'] = facename_lang_counters[lang]
                facename_lang_counters[lang] += 1
            resources[kind].append(entry)
        self.resources = resources

        # original storage ids of BinData records
        self.bindata_storage_ids = []
        for entry in resources['bindata']:
            model = entry['model']
            storage_id = None
            if model is not None:
                bindata = model['content'].get('bindata')
                if bindata is not None:
                    storage_id = bindata.get('storage_id')
            self.bindata_storage_ids.append(storage_id)

    def facename_languages(self):
        '''Assign a language to each FaceName record, in record order.'''
        langs = []
        for field, kind in IDMAPPING_COUNT_FIELDS:
            if not kind.startswith('facename:'):
                continue
            lang = kind.split(':', 1)[1]
            langs.extend([lang] * self.idmappings.get(field, 0))
        n_records = sum(1 for entry in self.docinfo_entries
                        if entry['record']['tagid'] == tagids.HWPTAG_FACE_NAME)
        if len(langs) != n_records:
            self.incomplete('DocInfo declares %d fonts but has %d '
                            'HWPTAG_FACE_NAME records; font language '
                            'assignment may be wrong',
                            len(langs), n_records)
            while len(langs) < n_records:
                langs.append(langs[-1] if langs else 'ko')
            langs = langs[:n_records]
        return langs

    def load_section(self, section_idx):
        sections = self.hwpfile.bodytext
        try:
            section = sections.section(section_idx)
        except KeyError:
            raise FragmentError('no such section: %d' % section_idx)
        groups = section.records_treegrouped()
        return [list(group) for group in groups]

    # -- dependency scanning --------------------------------------------

    def check_body_model(self, model, where):
        '''Flag records which may reference uncollected resources.'''
        tagid = model['tagid']
        tagname = model['tagname']
        if model.get('unparsed'):
            self.warn('%s: %s (%s) has %d unparsed trailing bytes',
                      where, tagname, model['type'].__name__,
                      len(model['unparsed']))
        if model['type'] is UnknownTagModel:
            self.incomplete('%s: unknown record %s (tagid %d) may reference '
                            'uncollected resources',
                            where, tagname, tagid)
            return
        if tagid in SAFE_BODY_TAGS:
            return
        if tagid in HANDLED_BODY_TAGS:
            if tagid == tagids.HWPTAG_CTRL_HEADER:
                chid = model['content'].get('chid')
                if chid not in control_models:
                    self.incomplete('%s: unknown control %r may reference '
                                    'uncollected resources', where, chid)
            elif tagid == tagids.HWPTAG_CTRL_DATA:
                if model['type'] is ControlData:
                    self.incomplete('%s: control data of unknown control '
                                    'may reference uncollected resources',
                                    where)
            return
        self.incomplete('%s: record %s (tagid %d) is not known to be free '
                        'of resource references', where, tagname, tagid)

    def scan_body_tree(self, tree_records, para_idx):
        '''Scan one selected paragraph tree.

        Returns (references, models_or_None).  Unknown/suspicious records
        are recorded as incomplete reasons.
        '''
        where = 'paragraph %d' % para_idx
        models, error = parse_tree_safely(self.context, tree_records)
        if models is None:
            self.incomplete('%s: cannot parse records (%s); resource '
                            'references inside are unknown', where, error)
            return [], None
        references = []
        for model in models:
            self.check_body_model(model, where)
            for reference in iter_model_references(model):
                references.append(reference)
        return references, models

    def add_dep(self, deps, reference, where):
        kind = reference.kind
        dep_id = reference.dep_id
        if dep_id < 0:
            return
        count = self.counts.get(kind)
        if count is not None and dep_id >= count:
            self.incomplete('%s: dangling %s reference to id %d '
                            '(only %d defined)', where, kind, dep_id, count)
            return
        deps[kind].add(dep_id)

    def close_deps(self, deps):
        '''Compute the dependency closure over DocInfo resources.'''
        resource_refs = {}
        for kind, entries in self.resources.items():
            if kind == 'facename':
                continue
            for old_id, entry in enumerate(entries):
                model = entry['model']
                if model is None:
                    continue
                for reference in iter_model_references(model):
                    resource_refs.setdefault((kind, old_id), []) \
                                 .append(reference)

        changed = True
        while changed:
            changed = False

            # resolve BinData storage ids to BinData records
            for storage_id in list(deps['binstorage']):
                if storage_id in self.bindata_storage_ids:
                    idx = self.bindata_storage_ids.index(storage_id)
                    if idx not in deps['bindata']:
                        deps['bindata'].add(idx)
                        changed = True
                else:
                    self.incomplete('no BinData record with storage id '
                                    '%d referenced by an OLE object',
                                    storage_id)
                    deps['binstorage'].discard(storage_id)

            for kind in DEP_KINDS:
                if kind == 'binstorage':
                    continue
                for old_id in list(deps[kind]):
                    for reference in resource_refs.get((kind, old_id), ()):
                        if reference.dep_id not in deps[reference.kind]:
                            deps[reference.kind].add(reference.dep_id)
                            changed = True

    # -- id remapping ----------------------------------------------------

    def build_maps(self, deps):
        '''Assign dense new ids, preserving the original relative order.'''
        maps = dict((kind, {}) for kind in DEP_KINDS)
        for lang in LANGS:
            kind = 'facename:' + lang
            new_id = 0
            for entry in self.resources['facename']:
                if entry['facename_lang'] != lang:
                    continue
                lang_old_id = entry['facename_lang_id']
                if lang_old_id in deps[kind]:
                    maps[kind][lang_old_id] = new_id
                    new_id += 1
        for kind in ('borderfill', 'charshape', 'tabdef', 'numbering',
                     'bullet', 'parashape', 'style', 'bindata'):
            new_id = 0
            for old_id, entry in enumerate(self.resources[kind]):
                if old_id in deps[kind]:
                    maps[kind][old_id] = new_id
                    new_id += 1
        # new BinData storage ids, assigned in kept-record order
        new_storage_id = 1
        for old_id, entry in enumerate(self.resources['bindata']):
            if old_id in deps['bindata']:
                old_storage_id = self.bindata_storage_ids[old_id]
                if old_storage_id is not None:
                    maps['binstorage'][old_storage_id] = new_storage_id
                    new_storage_id += 1
        return maps

    def reference_patches(self, references, maps, where):
        '''Turn references into (offset, packed bytes) payload patches.'''
        patches = []
        for reference in references:
            kind_map = maps[reference.kind]
            if reference.dep_id not in kind_map:
                self.incomplete('%s: %s id %d is referenced but not '
                                'collected', where, reference.kind,
                                reference.dep_id)
                continue
            new_value = kind_map[reference.dep_id] + reference.bias
            packed = struct.pack(reference.binfmt, new_value)
            patches.append((reference.offset, packed))
        return patches

    def patch_payload(self, payload, patches):
        if not patches:
            return payload
        data = bytearray(payload)
        for offset, packed in patches:
            data[offset:offset + len(packed)] = packed
        return bytes(data)

    def idmappings_patches(self):
        model = self.idmappings_entry['model']
        new_counts = dict((kind, len(self.maps[kind]))
                          for kind in DEP_KINDS)
        patches = []
        for leaf in iter_leaves(model['binevents']):
            for field, kind in IDMAPPING_COUNT_FIELDS:
                if leaf['path'] == (field,):
                    packed = struct.pack(leaf['binfmt'], new_counts[kind])
                    patches.append((leaf['offset'], packed))
        return patches

    # -- package writing --------------------------------------------------

    def extract(self, outdir, paragraph_indexes, section_idx=0):
        self.load_docinfo()
        trees = self.load_section(section_idx)
        n_paragraphs = len(trees)
        for para_idx in paragraph_indexes:
            if not 0 <= para_idx < n_paragraphs:
                raise FragmentError('paragraph index %d out of range '
                                    '(section %d has %d paragraphs)'
                                    % (para_idx, section_idx, n_paragraphs))
        for para_idx in paragraph_indexes:
            if trees[para_idx][0]['tagid'] != tagids.HWPTAG_PARA_HEADER:
                raise FragmentError('paragraph %d does not start with '
                                    'HWPTAG_PARA_HEADER' % para_idx)

        # 1. scan selected paragraph trees
        deps = dict((kind, set()) for kind in DEP_KINDS)
        scanned = []
        for para_idx in paragraph_indexes:
            references, models = self.scan_body_tree(trees[para_idx],
                                                     para_idx)
            scanned.append((para_idx, references, models))
            for reference in references:
                self.add_dep(deps, reference, 'paragraph %d' % para_idx)

        # 2. dependency closure over DocInfo resources
        self.close_deps(deps)

        # 3. dense id maps, shared resources collected only once
        self.maps = maps = self.build_maps(deps)

        # 4. rebuild DocInfo
        docinfo_records = []
        for entry in self.docinfo_entries:
            record = entry['record']
            tagid = record['tagid']
            model = entry['model']
            if tagid == tagids.HWPTAG_ID_MAPPINGS:
                patches = self.idmappings_patches()
                payload = self.patch_payload(record['payload'], patches)
                docinfo_records.append(self.replace_payload(record, payload))
                continue
            kind = TAG_TO_KIND.get(tagid)
            if kind is not None:
                if kind == 'facename':
                    old_id = entry['facename_lang_id']
                    dep_kind = 'facename:' + entry['facename_lang']
                else:
                    old_id = entry['resource_id']
                    dep_kind = kind
                if old_id not in deps[dep_kind]:
                    continue            # not referenced: drop
                if model is None:
                    self.incomplete('cannot parse kept DocInfo record %s '
                                    '(%s); copied unpatched',
                                    record['tagname'], entry['error'])
                    docinfo_records.append(record)
                    continue
                references = list(iter_model_references(model))
                patches = self.reference_patches(references, maps,
                                                 'DocInfo %s'
                                                 % record['tagname'])
                payload = self.patch_payload(record['payload'], patches)
                docinfo_records.append(self.replace_payload(record, payload))
                continue
            if tagid in SAFE_DOCINFO_TAGS:
                docinfo_records.append(record)
                continue
            if model is not None and model['type'] is UnknownTagModel:
                self.incomplete('DocInfo: unknown record %s (tagid %d) '
                                'may reference uncollected resources',
                                record['tagname'], tagid)
            elif model is None:
                self.incomplete('DocInfo: unparsable record %s kept '
                                'as-is (%s)', record['tagname'],
                                entry['error'])
            docinfo_records.append(record)

        # 5. rebuild the section stream
        section_records = []
        for para_idx, references, models in scanned:
            if models is None:
                section_records.extend(trees[para_idx])
                continue
            for record, model in zip(trees[para_idx], models):
                record_refs = list(iter_model_references(model))
                patches = self.reference_patches(
                    record_refs, maps, 'paragraph %d' % para_idx)
                payload = self.patch_payload(record['payload'], patches)
                section_records.append(
                    self.replace_payload(record, payload))

        # 6. write the package
        self.write_package(outdir, docinfo_records, section_records,
                           paragraph_indexes, section_idx)
        return self.manifest

    def replace_payload(self, record, payload):
        record = dict(record)
        record['payload'] = payload
        record['size'] = len(payload)
        return record

    def write_package(self, outdir, docinfo_records, section_records,
                      paragraph_indexes, section_idx):
        if os.path.exists(outdir):
            raise FragmentError('output directory already exists: %s'
                                % outdir)
        os.makedirs(outdir)

        # FileHeader: copy, but clear the 'compressed' flag bit since all
        # streams of the package are stored uncompressed
        header = self.hwpfile.fileheader.open().read()
        header = bytearray(header)
        flags = struct.unpack('<I', bytes(header[36:40]))[0]
        flags &= ~1
        header[36:40] = struct.pack('<I', flags)
        self.write_stream(outdir, 'FileHeader', bytes(header))

        self.write_records(outdir, 'DocInfo', docinfo_records)
        bodytext_dir = os.path.join(outdir, 'BodyText')
        os.mkdir(bodytext_dir)
        self.write_records(bodytext_dir, 'Section%d' % section_idx,
                           section_records)

        # embedded objects actually referenced, collected only once
        bindata_manifest = []
        copied = 0
        for old_id, entry in enumerate(self.resources['bindata']):
            if old_id not in self.maps['bindata']:
                continue
            new_id = self.maps['bindata'][old_id]
            old_storage_id = self.bindata_storage_ids[old_id]
            item_manifest = {'fragment': new_id + 1, 'original': old_id + 1}
            if old_storage_id is None:
                item_manifest['type'] = 'link'
                item_manifest['note'] = ('linked external data; no stream '
                                         'in the original file')
            else:
                new_storage_id = self.maps['binstorage'][old_storage_id]
                item_manifest['storage_id'] = {
                    'fragment': new_storage_id,
                    'original': old_storage_id,
                }
                model = entry['model']
                flags = model['content']['flags']
                storage_type = flags.storage
                if int(storage_type) == 1:      # EMBEDDING
                    ext = model['content']['bindata']['ext']
                    item_manifest['type'] = 'embedding'
                    src_name = 'BIN%04X.%s' % (old_storage_id, ext)
                    dst_name = 'BIN%04X.%s' % (new_storage_id, ext)
                else:                           # STORAGE
                    item_manifest['type'] = 'storage'
                    src_name = 'BIN%04X' % old_storage_id
                    dst_name = 'BIN%04X' % new_storage_id
                item_manifest['stream'] = 'BinData/' + dst_name
                item_manifest['original_stream'] = 'BinData/' + src_name
                src_item = self.bindata_item(src_name)
                if src_item is None:
                    self.incomplete('BinData stream %s is missing in the '
                                    'source file', src_name)
                else:
                    dst_dir = os.path.join(outdir, 'BinData')
                    if not os.path.exists(dst_dir):
                        os.mkdir(dst_dir)
                    self.copy_storage_item(src_item,
                                           os.path.join(dst_dir, dst_name))
                    copied += 1
            bindata_manifest.append(item_manifest)

        self.manifest = manifest = self.build_manifest(
            paragraph_indexes, section_idx, bindata_manifest)
        manifest_path = os.path.join(outdir, MANIFEST_NAME)
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2, sort_keys=True,
                      ensure_ascii=False)
        logger.info('fragment package written to %s '
                    '(%d paragraphs, %d BinData streams, complete=%s)',
                    outdir, len(paragraph_indexes), copied,
                    manifest['complete'])

    def bindata_item(self, name):
        try:
            bindata = self.hwpfile['BinData']
        except KeyError:
            return None
        try:
            return bindata[name]
        except KeyError:
            return None

    def copy_storage_item(self, item, dst_path):
        if is_stream(item):
            data = item.open().read()
            with open(dst_path, 'wb') as f:
                f.write(data)
        else:
            os.mkdir(dst_path)
            for name in item:
                self.copy_storage_item(item[name],
                                       os.path.join(dst_path, name))

    def write_stream(self, outdir, name, data):
        with open(os.path.join(outdir, name), 'wb') as f:
            f.write(data)

    def write_records(self, outdir, name, records):
        path = os.path.join(outdir, name)
        with open(path, 'wb') as f:
            for record in records:
                dump_record(f, record)

    def build_manifest(self, paragraph_indexes, section_idx,
                       bindata_manifest):
        resources = {}
        for lang in LANGS:
            kind = 'facename:' + lang
            entries = [{'fragment': new, 'original': old}
                       for old, new in sorted(self.maps[kind].items(),
                                              key=lambda kv: kv[1])]
            if entries:
                resources[kind] = entries
        for kind in ('borderfill', 'charshape', 'tabdef', 'numbering',
                     'bullet', 'parashape', 'style'):
            entries = [{'fragment': new, 'original': old}
                       for old, new in sorted(self.maps[kind].items(),
                                              key=lambda kv: kv[1])]
            if entries:
                resources[kind] = entries
        if bindata_manifest:
            resources['bindata'] = bindata_manifest
        source_name = self.source_name
        return {
            'format': FRAGMENT_FORMAT,
            'source': {
                'file': source_name,
                'version': list(self.version),
            },
            'section': section_idx,
            'paragraphs': list(paragraph_indexes),
            'complete': not self.incomplete_reasons,
            'incomplete_reasons': list(self.incomplete_reasons),
            'warnings': list(self.warnings),
            'resources': resources,
        }


def extract_fragment(source, outdir, paragraph_indexes, section_idx=0):
    '''Extract the given paragraphs of ``source`` into ``outdir``.

    ``source`` may be a filename or an open Hwp5File.  Returns the
    manifest dict of the created fragment package.
    '''
    extractor = FragmentExtractor(source)
    return extractor.extract(outdir, paragraph_indexes, section_idx)


def validate_fragment(outdir):
    '''Independently interpret a fragment package and check its integrity.

    Returns a list of problems; an empty list means the package parses
    cleanly and every resource reference resolves within the package.
    '''
    problems = []
    stg = FileSystemStorage(outdir)
    try:
        hwpfile = Hwp5File(stg)
    except Exception as e:
        return ['not a valid HWPv5 storage: %s' % e]

    version = hwpfile.header.version
    context = dict(version=version)

    # DocInfo must parse
    docinfo_models = []
    for record in hwpfile.docinfo.records():
        model, error = parse_record_safely(context, record)
        if model is None:
            problems.append('DocInfo: cannot parse %s: %s'
                            % (record['tagname'], error))
            continue
        docinfo_models.append(model)

    idmappings = None
    for model in docinfo_models:
        if model['tagid'] == tagids.HWPTAG_ID_MAPPINGS:
            idmappings = model['content']
    if idmappings is None:
        problems.append('DocInfo: HWPTAG_ID_MAPPINGS is missing')
        return problems

    counts = {}
    for field, kind in IDMAPPING_COUNT_FIELDS:
        counts[kind] = idmappings.get(field, 0)
    counts['binstorage'] = None

    storage_ids = set()
    for model in docinfo_models:
        if model['tagid'] != tagids.HWPTAG_BIN_DATA:
            continue
        bindata = model['content'].get('bindata')
        if bindata is None:
            continue
        storage_id = bindata.get('storage_id')
        if storage_id is None:
            continue
        storage_ids.add(storage_id)
        flags = model['content']['flags']
        if int(flags.storage) == 1:
            name = 'BIN%04X.%s' % (storage_id, bindata['ext'])
        else:
            name = 'BIN%04X' % storage_id
        path = os.path.join(outdir, 'BinData', name)
        if not os.path.exists(path):
            problems.append('BinData: stream %s is missing' % name)

    def check_references(models, where):
        for model in models:
            for reference in iter_model_references(model):
                if reference.kind == 'binstorage':
                    if reference.dep_id not in storage_ids:
                        problems.append('%s: unknown BinData storage id %d'
                                        % (where, reference.dep_id))
                    continue
                count = counts.get(reference.kind)
                if count is None:
                    continue
                if not 0 <= reference.dep_id < count:
                    problems.append('%s: %s id %d out of range (count %d)'
                                    % (where, reference.kind,
                                       reference.dep_id, count))

    check_references(docinfo_models, 'DocInfo')

    # every section stream must parse
    bodytext = hwpfile.bodytext
    for section_idx in bodytext.section_indexes():
        section = bodytext.section(section_idx)
        for group_idx, records in enumerate(section.records_treegrouped()):
            models, error = parse_tree_safely(context, records)
            if models is None:
                problems.append('section %d paragraph %d: cannot parse: %s'
                                % (section_idx, group_idx, error))
                continue
            check_references(models, 'section %d paragraph %d'
                             % (section_idx, group_idx))
    return problems
