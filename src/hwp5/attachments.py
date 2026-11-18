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
''' Inventory of binary attachments declared by a HWPv5 document.

The inventory is built from ``HWPTAG_BIN_DATA`` records, not from the
``BinData`` storage alone.  A record is the document's declaration of one
attachment: it carries the storage type, the storage identity used by the
body, and -- for external links -- the absolute and relative paths declared
by the document.

Consequences which are deliberately preserved by this module:

* Two records remain two inventory entries even when they use the same
  storage identity or contain identical bytes.
* Every body reference is kept as a separate occurrence.  References are
  never collapsed into a set.
* A relative link is resolved against the directory of the document (or an
  explicit base directory).  It is not resolved by searching ``BinData`` for
  a similarly named stream.
* A link whose declared file was not supplied is reported as ``pending``.
  It is never satisfied from another attachment that happens to share its
  base name.
'''
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from collections import OrderedDict
import hashlib
import io
import os
import os.path
import re

from .binmodel import BinData


SCHEMA_VERSION = 'hwp5-attachments/1'

STORAGE_TYPE_NAMES = {
    BinData.StorageType.LINK: 'link',
    BinData.StorageType.EMBEDDING: 'embedding',
    BinData.StorageType.STORAGE: 'storage',
}


def _text(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode('utf-8', 'replace')
    return value


def _storage_type_name(storage_type):
    name = STORAGE_TYPE_NAMES.get(storage_type)
    if name is not None:
        return name
    return getattr(storage_type, 'name', str(storage_type)).lower()


def _storage_id_text(storage_id):
    if storage_id is None:
        return None
    return 'BIN%04X' % int(storage_id)


def _open_item(item):
    if hasattr(item, 'open'):
        return item.open()
    raise TypeError('not an openable stream')


def _hash_fileobj(f):
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = f.read(65536)
        if not chunk:
            break
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _hash_stream(item):
    f = _open_item(item)
    try:
        return _hash_fileobj(f)
    finally:
        if hasattr(f, 'close'):
            f.close()


def _hash_path(path):
    f = io.open(path, 'rb')
    try:
        return _hash_fileobj(f)
    finally:
        f.close()


def _is_storage(item):
    return hasattr(item, '__iter__') and hasattr(item, '__getitem__')


def _storage_names(storage):
    if storage is None:
        return []
    try:
        return list(storage)
    except TypeError:
        return []


def _storage_name(storage, name):
    if storage is None or name is None:
        return None
    names = _storage_names(storage)
    if name in names:
        return name
    matches = [candidate for candidate in names
               if candidate.lower() == name.lower()]
    if len(matches) == 1:
        return matches[0]
    return None


def _storage_name_for_entry(entry, storage):
    ''' Find the stream that the declared storage identity denotes.

    ``embedding`` records carry both the identity and the extension, so the
    expected stream name is exact.  ``storage`` records carry only the
    identity; the extension is part of the stored stream name.  In that case
    a unique stream whose stem is the identity is accepted.  Ambiguity is
    reported as missing instead of guessing.
    '''
    storage_id = entry.get('storage_id')
    name = entry.get('name')
    if entry.get('storage_type') == 'embedding':
        return _storage_name(storage, name)

    if storage is None or storage_id is None:
        return None
    names = _storage_names(storage)
    if storage_id in names:
        return storage_id
    matches = [candidate for candidate in names
               if candidate.split('.', 1)[0].lower() == storage_id.lower()]
    if len(matches) == 1:
        return matches[0]
    return None


def _is_absolute_declared_path(path):
    if path is None:
        return False
    normalized = path.replace('\\', '/')
    if normalized.startswith('/'):
        return True
    # Windows drive-qualified path.  os.path.isabs() does not recognize it
    # on POSIX, but the declaration still is absolute.
    return re.match(r'^[A-Za-z]:/', normalized) is not None


def _normalize_declared_path(path):
    if path is None:
        return None
    return path.strip().replace('\\', '/')


def _resolve_declared_path(relpath, abspath, base_dir):
    ''' Resolve the path exactly as declared by the document.

    ``relpath`` wins over ``abspath`` when it is present.  There is no
    fallback to a basename search, and no substitution of another file when
    the declared one is absent.
    '''
    relpath = _normalize_declared_path(relpath)
    abspath = _normalize_declared_path(abspath)

    if relpath and not _is_absolute_declared_path(relpath):
        if base_dir:
            return os.path.normpath(os.path.join(base_dir, relpath)), \
                'document-relative'
        return relpath, 'unresolved-relative'

    if relpath:
        return relpath, 'declared-absolute'

    if abspath and _is_absolute_declared_path(abspath):
        return abspath, 'declared-absolute'

    if abspath:
        if base_dir:
            return os.path.normpath(os.path.join(base_dir, abspath)), \
                'document-relative'
        return abspath, 'unresolved-relative'

    return None, 'no-path-declared'


def _within_directory(path, directory):
    if not path or not directory:
        return None
    path = os.path.abspath(path)
    directory = os.path.abspath(directory)
    if path == directory:
        return True
    return path.startswith(directory + os.sep)


class _ModelReferenceScanner(object):
    ''' Walk a record-model stream and yield the direct binary references.

    Record models are emitted in pre-order with a ``level`` field.  Keeping
    a level-indexed stack lets us report a readable path for each occurrence
    without changing the model stream itself.
    '''

    def __init__(self, models):
        self.models = models

    def _path(self, stack, level, name):
        names = [stack[key][0] for key in sorted(stack) if key < level]
        names.append(name)
        return '/'.join(names)

    def _nearest_paragraph(self, stack, level):
        for key in sorted(stack, reverse=True):
            if key < level and stack[key][1] is not None:
                return stack[key][1]
        return None

    def references(self):
        stack = {}
        paragraph_counter = 0

        for model in self.models:
            level = model.get('level', 0)
            for key in list(stack):
                if key >= level:
                    del stack[key]

            model_type = model.get('type')
            name = getattr(model_type, '__name__', str(model_type))
            path = self._path(stack, level, name)
            paragraph = self._nearest_paragraph(stack, level)

            if name == 'Paragraph':
                paragraph = paragraph_counter
                paragraph_counter += 1

            content = model.get('content') or {}
            for bindata_id, field in _walk_ids(content, 'bindata_id'):
                yield {
                    'reference_type': 'bindata',
                    'model': name,
                    'record_seqno': model.get('seqno'),
                    'record_level': level,
                    'record_path': path,
                    'paragraph_index': paragraph,
                    'bindata_id': bindata_id,
                    'field': field,
                }

            for borderfill_id, field in _walk_ids(content, 'borderfill_id'):
                if borderfill_id:
                    yield {
                        'reference_type': 'borderfill',
                        'model': name,
                        'record_seqno': model.get('seqno'),
                        'record_level': level,
                        'record_path': path,
                        'paragraph_index': paragraph,
                        'borderfill_id': borderfill_id,
                        'field': field,
                    }

            for parashape_id, field in _walk_ids(content, 'parashape_id'):
                if parashape_id is not None:
                    yield {
                        'reference_type': 'parashape',
                        'model': name,
                        'record_seqno': model.get('seqno'),
                        'record_level': level,
                        'record_path': path,
                        'paragraph_index': paragraph,
                        'parashape_id': parashape_id,
                        'field': field,
                    }

            if name == 'ShapeOLE':
                storage_id = content.get('storage_id')
                if storage_id is not None:
                    yield {
                        'reference_type': 'storage',
                        'model': name,
                        'record_seqno': model.get('seqno'),
                        'record_level': level,
                        'record_path': path,
                        'paragraph_index': paragraph,
                        'storage_id': _storage_id_text(storage_id),
                        'field': 'storage_id',
                    }

            stack[level] = (name, paragraph)


def _walk_ids(value, name, prefix=''):
    if isinstance(value, dict):
        for key in value:
            child = value[key]
            field = prefix + key
            if key == name and isinstance(child, int):
                yield child, field
            else:
                for found in _walk_ids(child, name, field + '.'):
                    yield found
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            field = '%s%d' % (prefix, index)
            for found in _walk_ids(child, name, field + '.'):
                yield found


def _section_models(sections):
    ''' Normalize sections into ``(part, index, models)`` triples. '''
    for section in sections or ():
        if isinstance(section, dict):
            part = section.get('part', 'BodyText')
            index = section.get('index')
            models = section.get('models') or ()
            stream = section.get('stream')
        else:
            index, models = section
            part = 'BodyText'
            stream = None
        if stream is None and index is not None:
            stream = '%s/Section%d' % (part, index)
        yield part, index, stream, list(models)


def _attachment_from_model(model, bindata_id):
    content = model.get('content') or {}
    flags = BinData.Flags(content.get('flags', 0))
    storage_type = flags.storage
    storage_type_name = _storage_type_name(storage_type)
    bindata = content.get('bindata') or {}

    entry = OrderedDict()
    entry['bindata_id'] = bindata_id
    entry['record_seqno'] = model.get('seqno')
    entry['storage_type'] = storage_type_name
    entry['storage_id'] = None
    entry['storage_identity'] = None
    entry['identity'] = None
    entry['name'] = None
    entry['stream'] = None
    entry['status'] = 'unknown'
    entry['content'] = None
    entry['external'] = None
    entry['references'] = []
    entry['definition_references'] = []
    entry['shared'] = OrderedDict()
    entry['name_collision'] = False

    if storage_type == BinData.StorageType.LINK:
        abspath = _text(bindata.get('abspath'))
        relpath = _text(bindata.get('relpath'))
        entry['name'] = _basename(relpath) or _basename(abspath)
        entry['identity'] = 'link:' + (
            _normalize_declared_path(relpath or abspath) or ''
        )
        entry['storage_identity'] = OrderedDict([
            ('kind', 'external-file'),
            ('abspath', abspath),
            ('relpath', relpath),
        ])
        entry['external'] = {
            'abspath': abspath,
            'relpath': relpath,
        }
        return entry

    storage_id = bindata.get('storage_id')
    entry['storage_id'] = _storage_id_text(storage_id)

    if storage_type == BinData.StorageType.EMBEDDING:
        ext = _text(bindata.get('ext'))
        name = entry['storage_id']
        if ext:
            name += '.' + ext
        entry['name'] = name
        entry['storage_identity'] = OrderedDict([
            ('kind', 'bin-data-stream'),
            ('storage_id', entry['storage_id']),
            ('ext', ext),
        ])
    else:
        entry['name'] = entry['storage_id']
        entry['storage_identity'] = OrderedDict([
            ('kind', 'bin-data-storage'),
            ('storage_id', entry['storage_id']),
        ])

    entry['identity'] = 'BinData/' + entry['name']
    return entry


def _basename(path):
    if not path:
        return None
    path = _normalize_declared_path(path)
    if not path:
        return None
    return path.rstrip('/').split('/')[-1] or None


def _finish_embedded_entry(entry, storage, hash_content):
    name = entry.get('name')
    if storage is None:
        entry['status'] = 'missing'
        entry['reason'] = 'document has no BinData storage'
        return

    actual_name = _storage_name_for_entry(entry, storage)
    if actual_name is None:
        entry['status'] = 'missing'
        entry['reason'] = 'BinData stream is not present: %s' % name
        return

    entry['stream'] = 'BinData/' + actual_name
    item = storage[actual_name]
    if _is_storage(item):
        entry['status'] = 'available'
        entry['storage_kind'] = 'storage'
        return

    entry['storage_kind'] = 'stream'
    if not hash_content:
        entry['status'] = 'available'
        return

    try:
        sha256, size = _hash_stream(item)
    except (IOError, OSError) as e:
        entry['status'] = 'unreadable'
        entry['reason'] = str(e)
    else:
        entry['status'] = 'available'
        entry['content'] = OrderedDict([
            ('sha256', sha256),
            ('size', size),
        ])


def _finish_link_entry(entry, base_dir, hash_content):
    external = entry['external']
    relpath = external.get('relpath')
    abspath = external.get('abspath')
    resolved, resolution = _resolve_declared_path(relpath, abspath, base_dir)

    external['resolved_path'] = resolved
    external['resolution'] = resolution
    external['within_document_directory'] = _within_directory(
        resolved, base_dir
    )
    if resolution == 'unresolved-relative':
        external['exists'] = False
        external['delivered'] = False
        external['retrieval_required'] = True
        entry['status'] = 'pending'
        entry['reason'] = 'relative link cannot be resolved without a base'
        return

    external['exists'] = bool(resolved and os.path.exists(resolved))
    external['delivered'] = bool(external['exists'])
    external['retrieval_required'] = not external['exists']

    if not resolved:
        entry['status'] = 'pending'
        entry['reason'] = 'external link has no declared path'
        return

    if not external['exists']:
        entry['status'] = 'pending'
        entry['reason'] = 'external link was not supplied'
        return

    if hash_content and os.path.isfile(resolved):
        try:
            sha256, size = _hash_path(resolved)
        except (IOError, OSError) as e:
            entry['status'] = 'unreadable'
            entry['reason'] = str(e)
        else:
            entry['status'] = 'available'
            entry['content'] = OrderedDict([
                ('sha256', sha256),
                ('size', size),
            ])
    else:
        entry['status'] = 'available'


def _reference_with_context(reference, part, section, stream):
    result = OrderedDict()
    result['kind'] = reference.get('reference_type', 'bindata')
    result['part'] = part
    result['section'] = section
    result['stream'] = stream
    for key in ('model', 'record_seqno', 'record_level', 'record_path',
                'paragraph_index', 'bindata_id', 'borderfill_id',
                'parashape_id', 'storage_id', 'field'):
        if key in reference:
            result[key] = reference[key]
    return result


def _add_reference(entry, reference):
    entry['references'].append(reference)


def _add_definition_reference(entry, reference):
    entry['definition_references'].append(reference)


def _shared_groups(entries):
    by_storage = {}
    by_content = {}
    for entry in entries:
        identity = entry.get('identity')
        if identity:
            by_storage.setdefault(identity, []).append(entry['bindata_id'])
        content = entry.get('content') or {}
        sha256 = content.get('sha256')
        if sha256:
            by_content.setdefault(sha256, []).append(entry['bindata_id'])

    for entry in entries:
        shared = OrderedDict()
        identity = entry.get('identity')
        if identity:
            others = [x for x in by_storage[identity]
                      if x != entry['bindata_id']]
            if others:
                shared['storage_with'] = others
        content = entry.get('content') or {}
        sha256 = content.get('sha256')
        if sha256:
            others = [x for x in by_content[sha256]
                      if x != entry['bindata_id']]
            if others:
                shared['content_with'] = others
        if shared:
            entry['shared'] = shared
        else:
            entry['shared'] = OrderedDict()


def _name_collisions(entries):
    ''' Report same-named entries that point at different storage.

    Two records naming the same storage stream are a shared-storage
    relationship, not a collision.  A collision means the same visible name
    denotes two different storage identities, which must not be conflated.
    '''
    by_name = {}
    for entry in entries:
        name = entry.get('name')
        if name:
            by_name.setdefault(name, []).append(entry)

    collisions = []
    for name in sorted(by_name):
        named = by_name[name]
        identities = []
        for entry in named:
            identity = entry.get('identity')
            if identity not in identities:
                identities.append(identity)
        if len(identities) < 2:
            continue
        collisions.append(OrderedDict([
            ('name', name),
            ('bindata_ids', [entry['bindata_id'] for entry in named]),
            ('identities', identities),
        ]))
        for entry in named:
            entry['name_collision'] = True
    return collisions


def _orphan_streams(storage, used_names, hash_content):
    items = []
    for name in sorted(_storage_names(storage)):
        if name in used_names:
            continue
        item = storage[name]
        orphan = OrderedDict()
        orphan['name'] = name
        orphan['stream'] = 'BinData/' + name
        orphan['declared'] = False
        if _is_storage(item):
            orphan['storage_kind'] = 'storage'
            orphan['content'] = None
        else:
            orphan['storage_kind'] = 'stream'
            if hash_content:
                try:
                    sha256, size = _hash_stream(item)
                except (IOError, OSError) as e:
                    orphan['status'] = 'unreadable'
                    orphan['reason'] = str(e)
                else:
                    orphan['status'] = 'available'
                    orphan['content'] = OrderedDict([
                        ('sha256', sha256),
                        ('size', size),
                    ])
            else:
                orphan['status'] = 'available'
                orphan['content'] = None
        items.append(orphan)

    by_content = {}
    for orphan in items:
        content = orphan.get('content') or {}
        sha256 = content.get('sha256')
        if sha256:
            by_content.setdefault(sha256, []).append(orphan['name'])
    for orphan in items:
        content = orphan.get('content') or {}
        sha256 = content.get('sha256')
        if sha256:
            others = [x for x in by_content[sha256] if x != orphan['name']]
            if others:
                orphan['shared'] = OrderedDict([
                    ('content_with', others),
                ])
    return items


def _document_info(document_path, base_dir, header):
    info = OrderedDict()
    info['path'] = document_path
    info['directory'] = base_dir
    if header is None:
        info['version'] = None
        return info

    version = getattr(header, 'version', None)
    if isinstance(version, tuple):
        version = '.'.join(str(x) for x in version)
    info['version'] = version
    flags = getattr(header, 'flags', None)
    if flags is not None:
        info['flags'] = OrderedDict([
            ('compressed', bool(getattr(flags, 'compressed', False))),
            ('distributable', bool(getattr(flags, 'distributable', False))),
            ('password', bool(getattr(flags, 'password', False))),
        ])
    return info


def _pending_links(entries):
    pending = []
    for entry in entries:
        if entry.get('status') != 'pending':
            continue
        external = entry.get('external') or {}
        pending.append(OrderedDict([
            ('bindata_id', entry['bindata_id']),
            ('name', entry.get('name')),
            ('identity', entry.get('identity')),
            ('declared_abspath', external.get('abspath')),
            ('declared_relpath', external.get('relpath')),
            ('resolved_path', external.get('resolved_path')),
            ('reason', entry.get('reason')),
            ('references', entry.get('references') or []),
        ]))
    return pending


def _summary(entries, unresolved, storage_references, orphans, collisions):
    statuses = {}
    storage_types = {}
    reference_count = 0
    definition_reference_count = 0
    for entry in entries:
        statuses[entry.get('status')] = \
            statuses.get(entry.get('status'), 0) + 1
        storage_type = entry.get('storage_type')
        storage_types[storage_type] = \
            storage_types.get(storage_type, 0) + 1
        reference_count += len(entry.get('references') or ())
        definition_reference_count += len(
            entry.get('definition_references') or ()
        )

    summary = OrderedDict()
    summary['attachments'] = len(entries)
    summary['by_storage_type'] = OrderedDict(
        (key, storage_types[key]) for key in sorted(storage_types)
    )
    summary['by_status'] = OrderedDict(
        (key, statuses[key]) for key in sorted(statuses)
    )
    summary['references'] = reference_count
    summary['definition_references'] = definition_reference_count
    summary['pending_links'] = sum(
        1 for entry in entries if entry.get('status') == 'pending'
    )
    summary['unresolved_references'] = len(unresolved)
    summary['storage_references'] = len(storage_references)
    summary['unreferenced_streams'] = len(orphans)
    summary['name_collisions'] = len(collisions)
    return summary


def build_inventory(bindata_models, sections=None, bindata_storage=None,
                    document_path=None, base_dir=None, hash_content=True,
                    header=None, borderfill_models=None,
                    parashape_models=None, declared_bindata_count=None):
    ''' Build an attachment inventory from parsed models.

    :param bindata_models: ``HWPTAG_BIN_DATA`` models of ``DocInfo``.
    :param sections: iterable of ``(index, models)`` or dictionaries with
                     ``part``, ``index``, ``stream`` and ``models``.
    :param bindata_storage: the ``BinData`` storage from the HWP file.
    :param document_path: path of the ``.hwp`` file.
    :param base_dir: directory used to resolve document-relative links.
    :param hash_content: compute SHA-256 and size of supplied content.
    :param borderfill_models: ``HWPTAG_BORDER_FILL`` models of ``DocInfo``.
                             They let body ``borderfill_id`` references be
                             traced back to a fill-image attachment.
    :param parashape_models: ``HWPTAG_PARA_SHAPE`` models of ``DocInfo``.
    :param declared_bindata_count: the ``bindata`` count from ``IdMappings``
                                   (the document's declared attachment count).
    '''
    if base_dir is None and document_path:
        base_dir = os.path.dirname(os.path.abspath(document_path))

    entries = []
    for ordinal, model in enumerate(bindata_models, 1):
        entry = _attachment_from_model(model, ordinal)
        if entry['storage_type'] == 'link':
            _finish_link_entry(entry, base_dir, hash_content)
        else:
            _finish_embedded_entry(entry, bindata_storage, hash_content)
        entries.append(entry)

    by_bindata_id = dict(
        (entry['bindata_id'], entry) for entry in entries
    )
    by_storage_id = {}
    for entry in entries:
        storage_id = entry.get('storage_id')
        if storage_id:
            by_storage_id.setdefault(storage_id, []).append(entry)

    borderfill_images = {}
    for ordinal, model in enumerate(borderfill_models or (), 1):
        content = model.get('content') or {}
        fill_image = content.get('fill_image') or {}
        bindata_id = fill_image.get('bindata_id')
        if bindata_id is not None:
            borderfill_images[ordinal] = bindata_id

    parashape_borderfills = {}
    for ordinal, model in enumerate(parashape_models or (), 1):
        content = model.get('content') or {}
        borderfill_id = content.get('borderfill_id')
        if borderfill_id:
            parashape_borderfills[ordinal] = borderfill_id

    unresolved = []
    storage_references = []
    used_names = set()

    for part, section, stream, models in _section_models(sections):
        scanner = _ModelReferenceScanner(models)
        for reference in scanner.references():
            reference = _reference_with_context(
                reference, part, section, stream
            )
            if reference.get('kind') == 'storage':
                target = reference['storage_id']
                storage_references.append(OrderedDict([
                    ('storage_id', target),
                    ('attachment_bindata_ids', [
                        entry['bindata_id']
                        for entry in by_storage_id.get(target, ())
                    ]),
                    ('reference', reference),
                ]))
                continue

            if reference.get('kind') == 'parashape':
                borderfill_id = parashape_borderfills.get(
                    reference.get('parashape_id')
                )
                bindata_id = borderfill_images.get(borderfill_id)
                if bindata_id is None:
                    continue
                entry = by_bindata_id.get(bindata_id)
                if entry is not None:
                    indirect = OrderedDict(reference)
                    indirect['via'] = 'parashape'
                    indirect['borderfill_id'] = borderfill_id
                    indirect['bindata_id'] = bindata_id
                    _add_reference(entry, indirect)
                continue

            if reference.get('kind') == 'borderfill':
                bindata_id = borderfill_images.get(
                    reference.get('borderfill_id')
                )
                if bindata_id is None:
                    continue
                entry = by_bindata_id.get(bindata_id)
                if entry is not None:
                    indirect = OrderedDict(reference)
                    indirect['via'] = 'borderfill'
                    indirect['bindata_id'] = bindata_id
                    _add_reference(entry, indirect)
                continue

            bindata_id = reference.get('bindata_id')
            entry = by_bindata_id.get(bindata_id)
            if entry is None:
                unresolved.append(OrderedDict([
                    ('bindata_id', bindata_id),
                    ('reference', reference),
                    ('reason', 'no HWPTAG_BIN_DATA record for this id'),
                ]))
            else:
                _add_reference(entry, reference)

    # Mark storage names that are accounted for by declarations, so that
    # genuine extra streams can be listed without hiding them.
    for entry in entries:
        stream = entry.get('stream')
        if stream and stream.startswith('BinData/'):
            used_names.add(stream[len('BinData/'):])
            continue
        actual_name = _storage_name_for_entry(entry, bindata_storage)
        if actual_name:
            used_names.add(actual_name)

    orphans = _orphan_streams(bindata_storage, used_names, hash_content)
    _shared_groups(entries)
    collisions = _name_collisions(entries)

    result = OrderedDict()
    result['schema_version'] = SCHEMA_VERSION
    result['document'] = _document_info(
        document_path, base_dir, header
    )
    if declared_bindata_count is not None:
        result['document']['declared_bindata_count'] = \
            declared_bindata_count
    result['attachments'] = entries
    result['pending_links'] = _pending_links(entries)
    result['unreferenced_streams'] = orphans
    result['unresolved_references'] = unresolved
    result['storage_references'] = storage_references
    result['name_collisions'] = collisions
    result['summary'] = _summary(
        entries, unresolved, storage_references, orphans, collisions
    )
    if declared_bindata_count is not None:
        result['summary']['declared_bindata_count'] = \
            declared_bindata_count
        result['summary']['bindata_record_count'] = len(entries)
        result['summary']['bindata_count_mismatch'] = \
            declared_bindata_count != len(entries)
    return result


def add_definition_references(inventory, docinfo_models):
    ''' Attach DocInfo-level ``bindata_id`` references to their entries.

    This is separate from :func:`build_inventory` because those references
    describe document resources (for example a BorderFill fill image), not a
    body occurrence.  They therefore never enter ``references``.
    '''
    by_bindata_id = dict(
        (entry['bindata_id'], entry)
        for entry in inventory.get('attachments', ())
    )
    scanner = _ModelReferenceScanner(docinfo_models)
    for reference in scanner.references():
        if reference.get('reference_type') != 'bindata':
            continue
        bindata_id = reference.get('bindata_id')
        entry = by_bindata_id.get(bindata_id)
        if entry is None:
            continue
        reference = _reference_with_context(
            reference, 'DocInfo', None, 'DocInfo'
        )
        _add_definition_reference(entry, reference)

    summary = inventory.get('summary')
    if summary is not None:
        summary['definition_references'] = sum(
            len(entry.get('definition_references') or ())
            for entry in inventory.get('attachments', ())
        )
    return inventory


def inventory_from_hwp5file(hwp5file, document_path=None, base_dir=None,
                            hash_content=True):
    ''' Build an inventory from an opened :class:`hwp5.xmlmodel.Hwp5File`. '''
    docinfo_models = list(hwp5file.docinfo.models())
    bindata_models = [
        model for model in docinfo_models
        if getattr(model.get('type'), '__name__', '') == 'BinData'
    ]
    borderfill_models = [
        model for model in docinfo_models
        if getattr(model.get('type'), '__name__', '') == 'BorderFill'
    ]
    parashape_models = [
        model for model in docinfo_models
        if getattr(model.get('type'), '__name__', '') == 'ParaShape'
    ]

    declared_bindata_count = None
    for model in docinfo_models:
        if getattr(model.get('type'), '__name__', '') == 'IdMappings':
            declared_bindata_count = model['content'].get('bindata')
            break

    sections = []
    text = hwp5file.text
    header = getattr(hwp5file, 'header', None)
    header_flags = getattr(header, 'flags', None)
    distributable = bool(getattr(header_flags, 'distributable', False))
    part = 'ViewText' if distributable else 'BodyText'
    for index in text.section_indexes():
        sections.append({
            'part': part,
            'index': index,
            'stream': '%s/Section%d' % (part, index),
            'models': list(text.section(index).models()),
        })

    bindata_storage = None
    try:
        if 'BinData' in hwp5file:
            bindata_storage = hwp5file['BinData']
    except KeyError:
        bindata_storage = None

    inventory = build_inventory(
        bindata_models,
        sections=sections,
        bindata_storage=bindata_storage,
        document_path=document_path,
        base_dir=base_dir,
        hash_content=hash_content,
        header=header,
        borderfill_models=borderfill_models,
        parashape_models=parashape_models,
        declared_bindata_count=declared_bindata_count,
    )
    add_definition_references(inventory, docinfo_models)
    return inventory
