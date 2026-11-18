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
'''Structured, version-aware import of HWPv5 records.

The ordinary binmodel parser selects a structure using the version declared
in ``FileHeader`` and reads the payload from left to right.  If the declared
version is older than the bytes actually present, that parser can finish
without raising an error while leaving part of the record payload untouched.
This module makes that situation explicit:

* the declared structure version is recorded for every record;
* every interpreted field gets a payload-relative ``[offset, end)`` range;
* the interpreted extent and the unconsumed tail are reported separately;
* when bytes contradict the declared version, alternative structure versions
  are imported as *candidates*, together with the fields they add or remove.

A candidate that consumes the whole payload is deliberately **not** marked
as verified.  Consuming every byte only means that a structure fits the byte
length; it is not independent proof that the structure is the one that wrote
the record.  A candidate therefore remains ``unconfirmed`` unless a caller
supplies additional evidence of its own.
'''
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from binascii import b2a_hex
from io import BytesIO
import copy
import inspect
import json

from .binmodel import UnknownTagModel
from .binmodel._shared import tag_models
from .bintype import ERROREVENT
from .bintype import resolve_type_events
from .bintype import resolve_value_from_stream
from .dataio import FixedArrayType
from .dataio import ParseError
from .dataio import SelectiveType
from .dataio import StructType
from .dataio import VariableLengthArrayType
from .dataio import X_ARRAY
from .treeop import ENDEVENT
from .treeop import STARTEVENT


try:
    binary_type = bytes
except NameError:  # pragma: no cover - Python 2
    binary_type = str
text_type = type(u'')


__all__ = (
    'import_record',
    'interpret_record',
    'diagnose_record',
    'import_records',
    'import_stream',
    'structured_import',
    'stream_import',
    'format_record_report',
    'format_stream_report',
    'report_to_json',
)


ARRAY_TYPES = (X_ARRAY, VariableLengthArrayType, FixedArrayType)


def version_to_string(version):
    return '.'.join(str(x) for x in version)


def version_to_tuple(version):
    if version is None:
        return None
    if isinstance(version, (tuple, list)):
        return tuple(int(x) for x in version)
    parts = str(version).strip().split('.')
    return tuple(int(x) for x in parts)


def _previous_version(version):
    '''Return a version tuple that is strictly smaller than ``version``.

    The exact value does not matter to the parser; the only requirement is
    that every member whose ``version`` equals ``version`` is excluded.
    '''
    version = list(version)
    for idx in range(len(version) - 1, -1, -1):
        if version[idx] > 0:
            version[idx] -= 1
            for tail_idx in range(idx + 1, len(version)):
                version[tail_idx] = 0
            return tuple(version)
    # (0, 0, 0, 0): there is no smaller non-negative version.
    return tuple(version)


def _hexlify(data):
    if not isinstance(data, binary_type):
        data = binary_type(data)
    return b2a_hex(data).decode('ascii')


def _jsonable(value):
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, binary_type):
        return dict(hex=_hexlify(value))
    if isinstance(value, bytearray):
        return dict(hex=_hexlify(bytes(value)))
    if isinstance(value, text_type):
        return value
    if isinstance(value, dict):
        return dict((str(k), _jsonable(v)) for k, v in value.items()
                    if not str(k).startswith('_'))
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return repr(value)


class _Frame(object):

    def __init__(self, node, path, is_array):
        self.node = node
        self.path = path
        self.is_array = is_array
        self.next_index = 0


def _make_node(name, path, item, is_leaf):
    node = dict(
        name=name,
        path=path,
        type=item['type'].__name__,
        is_leaf=is_leaf,
        start=None,
        end=None,
        children=[],
    )
    if is_leaf:
        node['offset'] = item.get('bin_offset')
        node['bin_offset_end'] = item.get('bin_offset_end')
        node['value'] = item.get('value')
    return node


def _collect_nodes(events):
    '''Build a field tree (and a flat list of leaves) from parse events.'''
    stack = []
    roots = []
    leaves = []

    for event, item in events:
        if event is STARTEVENT:
            name = item.get('name')
            item_type = item['type']

            if stack:
                parent = stack[-1]
                if parent.is_array:
                    idx = parent.next_index
                    parent.next_index += 1
                    path = '%s[%d]' % (parent.path, idx)
                    name = '[%d]' % idx
                elif name is None:
                    # A nested anonymous struct: keep the parent's path.
                    path = parent.path
                else:
                    path = '%s.%s' % (parent.path, name)
            else:
                # The root struct of a record has no member name.  Extension
                # models (e.g. Control -> SectionDef) are parsed as several
                # roots; name each with its actual structure.
                name = name or item['type'].__name__
                path = name

            node = _make_node(name, path, item, False)
            node['start'] = item.get('_span_start')
            node['end'] = item.get('_span_end')

            if stack:
                stack[-1].node['children'].append(node)
            else:
                roots.append(node)
            stack.append(
                _Frame(node, path, isinstance(item_type, ARRAY_TYPES)))

        elif event is ENDEVENT:
            if not stack:
                continue
            frame = stack.pop()
            node = frame.node
            if node.get('start') is None:
                node['start'] = item.get('_span_start')
            if node.get('end') is None:
                node['end'] = item.get('_span_end')
            if node.get('start') is None:
                child_starts = [c['start'] for c in node['children']
                                if c.get('start') is not None]
                node['start'] = min(child_starts) if child_starts else 0
            if node.get('end') is None:
                child_ends = [c['end'] for c in node['children']
                              if c.get('end') is not None]
                node['end'] = max(child_ends) if child_ends else node['start']

        elif event is None:
            if not stack:
                continue
            parent = stack[-1]
            name = item.get('name')

            if parent.is_array:
                # Primitive array item (e.g. ARRAY(UINT32, 7)).
                idx = parent.next_index
                parent.next_index += 1
                path = '%s[%d]' % (parent.path, idx)
                name = '[%d]' % idx
            else:
                if name is None:
                    # N_ARRAY's count field and similar bookkeeping values
                    # are not members of the resulting value.
                    continue
                path = '%s.%s' % (parent.path, name)

            node = _make_node(name, path, item, True)
            if node.get('offset') is None:
                node['offset'] = item.get('bin_offset')
            if node.get('bin_offset_end') is None:
                node['bin_offset_end'] = item.get('bin_offset_end')
            node['start'] = node.get('offset')
            node['end'] = node.get('bin_offset_end')
            if node['end'] is None:
                node['end'] = node['start']

            parent.node['children'].append(node)
            leaves.append(node)

    return roots, leaves


def _field_span(field):
    return field.get('start'), field.get('end')


def _public_field(field):
    return dict(
        name=field.get('name'),
        path=field.get('path'),
        type=field.get('type'),
        offset=field.get('start'),
        end=field.get('end'),
        size=(field.get('end') - field.get('start')
              if field.get('start') is not None
              and field.get('end') is not None else None),
        value=_jsonable(field.get('value')),
    )


def _field_map(fields):
    result = {}
    for field in fields:
        result[field['path']] = field
    return result


def _diff_fields(base_fields, candidate_fields):
    base = _field_map(base_fields)
    candidate = _field_map(candidate_fields)

    added = [_public_field(candidate[path])
             for path in sorted(candidate) if path not in base]
    removed = [_public_field(base[path])
               for path in sorted(base) if path not in candidate]
    changed = []
    for path in sorted(set(base) & set(candidate)):
        before = base[path]
        after = candidate[path]
        before_span = _field_span(before)
        after_span = _field_span(after)
        before_value = _jsonable(before.get('value'))
        after_value = _jsonable(after.get('value'))
        if (before_span != after_span or before['type'] != after['type'] or
                before_value != after_value):
            changed.append(dict(
                name=after.get('name'),
                path=path,
                type=after.get('type'),
                before=dict(type=before.get('type'),
                            offset=before_span[0], end=before_span[1],
                            value=before_value),
                after=dict(type=after.get('type'),
                           offset=after_span[0], end=after_span[1],
                           value=after_value),
            ))
    return dict(added=added, removed=removed, changed=changed)


def _affected_names(diff):
    names = []
    for key in ('added', 'removed'):
        for field in diff[key]:
            names.append(field.get('path'))
    for change in diff['changed']:
        names.append(change.get('path'))
    return [name for name in names if name]


def _type_members(struct_type):
    '''Yield members of a struct type, including inherited members.'''
    mro = inspect.getmro(struct_type)
    mro = list(cls for cls in mro if 'members' in cls.__dict__)
    mro = list(reversed(mro))
    for cls in mro:
        for member in cls.members:
            yield member


def _nested_types(member_type):
    if isinstance(member_type, StructType):
        yield member_type
    elif isinstance(member_type, ARRAY_TYPES):
        yield member_type.itemtype
    elif isinstance(member_type, SelectiveType):
        for selection in member_type.selections.values():
            yield selection


def _version_thresholds(model_type):
    '''Collect every member ``version=`` threshold reachable from a model.'''
    thresholds = set()
    seen = set()
    queue = [model_type]

    while queue:
        current = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)

        if isinstance(current, StructType):
            for member in _type_members(current):
                member_version = member.get('version')
                if member_version is not None:
                    thresholds.add(tuple(member_version))
                member_type = member.get('type')
                for nested in _nested_types(member_type):
                    queue.append(nested)

            extension_types = getattr(current, 'extension_types', None)
            if extension_types:
                for extension in extension_types.values():
                    queue.append(extension)

        elif isinstance(current, ARRAY_TYPES):
            queue.append(current.itemtype)
        elif isinstance(current, SelectiveType):
            for selection in current.selections.values():
                queue.append(selection)

    return sorted(thresholds)


def _copy_context_parent(context):
    parent = context.get('parent')
    if not parent:
        return context
    parent_context, parent_model = parent
    parent_model = dict(parent_model)
    parent_model['content'] = copy.deepcopy(parent_model.get('content'))
    context = dict(context)
    context['parent'] = (dict(parent_context), parent_model)
    return context


def _resolve_model_events(model_type, context, model):
    '''The same model-event resolution as binmodel, but with an explicit
    model type (so tests and diagnostics can use synthetic structures).
    '''
    resolve_values = context['resolve_values']

    for event, item in resolve_type_events(model_type, context,
                                           resolve_values):
        yield event, item

    if 'value' in item:
        model['content'] = item['value']
    else:
        model.setdefault('content', dict())

    extension_types = getattr(model_type, 'extension_types', None)
    if extension_types:
        key = model_type.get_extension_key(context, model)
        extension = extension_types.get(key)
        if extension is not None:
            from .binmodel import get_extension_mro
            for cls in get_extension_mro(extension, model_type):
                extension_events = resolve_type_events(
                    cls, context, resolve_values)
                for event, item in extension_events:
                    yield event, item
                if 'value' in item:
                    model['content'].update(item['value'])
            model['type'] = extension

    if 'parent' in context:
        parent = context['parent']
        parent_context, parent_model = parent
        parent_type = parent_model.get('type')
        parent_content = parent_model.get('content')
        on_child = getattr(parent_type, 'on_child', None)
        if on_child:
            on_child(parent_content, parent_context, (context, model))


def _tracked_resolve_values(stream):
    def resolve(events):
        for event, item in events:
            if event is None:
                item['bin_offset'] = stream.tell()
                try:
                    item['value'] = resolve_value_from_stream(item, stream)
                except Exception as e:
                    item['exception'] = e
                    event = ERROREVENT
                item['bin_offset_end'] = stream.tell()
            yield event, item
    return resolve


def _parse_error(message, context, item, stream):
    error = ParseError(message)
    error.cause = item.get('exception')
    error.path = context.get('path')
    error.treegroup = context.get('treegroup')
    error.record = context.get('record')
    error.offset = item.get('bin_offset')
    return error


def _error_dict(error, offset):
    info = dict(
        type=type(error).__name__,
        message=str(error),
        offset=offset,
    )
    cause = getattr(error, 'cause', None)
    if cause is not None:
        info['cause'] = type(cause).__name__
        info['cause_message'] = str(cause)
    return info


def _run_variant(model_type, payload, version, context, isolate_parent=False):
    '''Parse ``payload`` with exactly one structure version.

    The context is shallow-copied so a variant's ``version`` and ``stream``
    do not leak.  Unless ``isolate_parent`` is set, the parent context object
    is shared, which preserves the normal ``on_child`` state transitions
    (e.g. ``seen_table_body``).  Candidate runs isolate the parent so they
    cannot change how later records are parsed.
    '''
    stream = BytesIO(payload)
    context = dict(context)
    if isolate_parent:
        context = _copy_context_parent(context)
    context['version'] = version
    context['stream'] = stream
    context['resolve_values'] = _tracked_resolve_values(stream)

    record = context.get('record')
    model = dict(payload=payload)
    if record is not None:
        model.update(dict((k, v) for k, v in record.items()
                          if k != 'payload'))
    model['type'] = model_type

    events = []
    error = None
    try:
        generator = _resolve_model_events(model_type, context, model)
        while True:
            try:
                event, item = next(generator)
            except StopIteration:
                break
            offset = stream.tell()
            if isinstance(item, dict):
                if event is STARTEVENT:
                    item['_span_start'] = offset
                elif event is ENDEVENT:
                    item['_span_end'] = offset
            events.append((event, item))
            if event is ERROREVENT:
                raise _parse_error('can\'t parse %s' % model_type.__name__,
                                   context, item, stream)
    except ParseError as e:
        error = e
    except Exception as e:
        # A malformed candidate must not abort the whole diagnostic run.
        error = e

    consumed = stream.tell()
    if consumed > len(payload):
        consumed = len(payload)
    remaining = max(0, len(payload) - consumed)

    roots, leaves = _collect_nodes(events)
    return dict(
        version=version,
        model=model,
        roots=roots,
        fields=leaves,
        consumed=consumed,
        remaining=remaining,
        error=error,
        events=events,
    )


def _tail_info(payload, consumed):
    tail = payload[consumed:]
    return dict(
        offset=consumed,
        size=len(tail),
        hex=_hexlify(tail),
        all_zero=bool(tail) and not any(tail),
    )


def _candidate_versions(model_type, declared_version):
    '''Versions worth trying for a record.

    Every versioned member marks a structure boundary.  Both the boundary
    itself (the member exists) and the previous version (the member does not
    exist) are candidates.  Candidates are tried even when the declared
    structure consumes the whole payload: exact consumption is not proof that
    the declared structure is the intended one.
    '''
    thresholds = _version_thresholds(model_type)
    versions = set()

    for threshold in thresholds:
        versions.add(threshold)
        versions.add(_previous_version(threshold))

    # If the declared version is itself a boundary, also try the variant just
    # before it.  This covers "version says new field, bytes say old record".
    following = [v for v in thresholds if v <= declared_version]
    if following:
        versions.add(_previous_version(max(following)))

    versions.discard(declared_version)
    return sorted(versions)


def structure_version_range(model_type, version):
    '''Describe which declared versions select the structure at ``version``.

    The label is derived from the structure's versioned members, so a
    candidate parsed at a synthetic "previous" version is not presented as
    an exact structure version.  Example: ``"<5.0.3.0"`` or
    ``">=5.0.2.5, <5.0.3.4"``.
    '''
    thresholds = _version_thresholds(model_type)
    lower = [t for t in thresholds if t <= version]
    upper = [t for t in thresholds if t > version]

    lower = max(lower) if lower else None
    upper = min(upper) if upper else None

    if lower is None and upper is None:
        return 'any version'
    if lower is None:
        return '<%s' % version_to_string(upper)
    if upper is None:
        return '>=%s' % version_to_string(lower)
    return '>=%s, <%s' % (version_to_string(lower),
                          version_to_string(upper))


def _structure_label(model_type, version):
    '''A stable name for the structure selected at ``version``.

    Synthetic "previous" versions are not real format versions, so callers
    should present the range rather than the raw number.
    '''
    thresholds = _version_thresholds(model_type)
    if version in thresholds or not thresholds:
        return version_to_string(version)
    return structure_version_range(model_type, version)


def _candidate_report(result, baseline, model_type):
    diff = _diff_fields(baseline['fields'], result['fields'])
    affected = _affected_names(diff)

    if result['error'] is not None:
        status = 'does_not_fit'
        assessment = 'rejected'
        confidence = 'none'
    elif result['remaining'] == 0:
        status = 'complete'
        assessment = 'unconfirmed'
        confidence = 'low'
    else:
        status = 'partial'
        assessment = 'partial'
        confidence = 'none'

    candidate = dict(
        structure_version=version_to_string(result['version']),
        structure_version_tuple=list(result['version']),
        structure=_structure_label(model_type, result['version']),
        structure_version_range=structure_version_range(
            model_type, result['version']),
        consumed=result['consumed'],
        extent=dict(start=0, end=result['consumed']),
        remaining=result['remaining'],
        tail=_tail_info(baseline['payload'], result['consumed']),
        complete=(result['error'] is None and result['remaining'] == 0),
        status=status,
        assessment=assessment,
        verified=False,
        confidence=confidence,
        error=(_error_dict(result['error'], result['consumed'])
               if result['error'] is not None else None),
        affected_fields=affected,
        affected_field_details=diff,
        added_fields=diff['added'],
        removed_fields=diff['removed'],
        changed_fields=diff['changed'],
        conflicts_with_declared=bool(affected),
        fields=[_public_field(field) for field in result['fields']],
    )
    if status == 'complete':
        candidate['note'] = (
            'The structure consumes the whole payload, but exact consumption '
            'is not evidence that it is the intended structure. '
            'This interpretation is unconfirmed.'
        )
    else:
        candidate['note'] = (
            'This structure does not explain the whole payload.'
        )
    return candidate


def _baseline_report(result, payload, model_type):
    error = result['error']
    fields = [_public_field(field) for field in result['fields']]
    tail = _tail_info(payload, result['consumed'])

    contradictions = []
    if error is not None:
        contradictions.append(dict(
            kind='truncated_payload',
            severity='error',
            message=(
                'The declared structure failed after %d of %d payload bytes.'
                % (result['consumed'], len(payload))
            ),
            detail=dict(
                error_type=type(error).__name__,
                error_message=str(error),
                offset=result['consumed'],
            ),
        ))
    elif result['remaining'] > 0:
        contradictions.append(dict(
            kind='unconsumed_tail',
            severity='warning',
            message=(
                'The declared structure consumed %d of %d payload bytes; '
                '%d bytes are unexplained.'
                % (result['consumed'], len(payload), result['remaining'])
            ),
            detail=dict(
                offset=result['consumed'],
                size=result['remaining'],
                all_zero=tail['all_zero'],
                note=('the tail is all zero bytes; it may be padding or a '
                      'trailing field, but the structure does not declare it'
                      if tail['all_zero'] else
                      'the tail contains non-zero bytes'),
            ),
        ))

    interpretation = dict(
        structure_version=version_to_string(result['version']),
        structure_version_range=structure_version_range(
            model_type, result['version']),
        structure_version_tuple=list(result['version']),
        consumed=result['consumed'],
        extent=dict(start=0, end=result['consumed']),
        payload_size=len(payload),
        remaining=result['remaining'],
        tail=tail,
        complete=(error is None and result['remaining'] == 0),
        verified=False,
        error=(_error_dict(error, result['consumed'])
               if error is not None else None),
        field_count=len(fields),
        fields=fields,
        field_tree=result['roots'],
    )

    if error is not None:
        interpretation['assessment'] = 'does_not_fit'
    elif result['remaining'] > 0:
        interpretation['assessment'] = 'partial'
    else:
        interpretation['assessment'] = 'unconfirmed'

    return interpretation, contradictions


def import_record(record, version=None, model_type=None, context=None,
                  include_candidates=True):
    '''Import one record according to a declared structure version.

    :param record:
        a record dict containing at least ``payload``.  ``tagid``, ``level``
        and ``tagname`` are used when available.
    :param version:
        the declared HWPv5 format version, as a tuple or ``"5.0.1.7"``.
    :param model_type:
        optional explicit model type.  By default it is selected from
        ``record['tagid']`` (unknown tags use ``UnknownTagModel``).
    :param context:
        optional parser context (``parent``, ``path``, ``treegroup``, ...).
    :param include_candidates:
        when true (the default), neighbouring structure versions are also
        imported.  Complete alternatives are tried even when the declared
        structure consumes everything, so ambiguity can be detected.
    :returns:
        a JSON-serializable report dict.
    '''
    report, _model = _import_record(
        record, version=version, model_type=model_type, context=context,
        include_candidates=include_candidates,
    )
    return report


def _import_record(record, version=None, model_type=None, context=None,
                   include_candidates=True):
    if version is None:
        raise ValueError('a declared version is required')
    declared_version = version_to_tuple(version)

    if model_type is None:
        tagid = record.get('tagid')
        model_type = tag_models.get(tagid, UnknownTagModel)

    payload = record.get('payload', b'')
    context = dict(context or {})
    context['record'] = record

    baseline = _run_variant(model_type, payload, declared_version, context)
    baseline['payload_size'] = len(payload)
    baseline['payload'] = payload

    interpretation, contradictions = _baseline_report(
        baseline, payload, model_type)

    candidates = []
    if include_candidates:
        baseline_conflict = bool(contradictions)
        for candidate_version in _candidate_versions(
                model_type, declared_version):
            result = _run_variant(model_type, payload, candidate_version,
                                  context, isolate_parent=True)
            # Skip candidates that are byte-for-byte the same interpretation.
            # An interpretation that fails is not the same as one that
            # succeeds, even if both stop at the same offset.
            if (result['consumed'] == baseline['consumed'] and
                    result['remaining'] == baseline['remaining'] and
                    (result['error'] is None) ==
                    (baseline['error'] is None) and
                    _field_map(result['fields']).keys() ==
                    _field_map(baseline['fields']).keys()):
                continue
            if result['error'] is not None:
                # A structure that cannot be parsed is not a candidate
                # interpretation.
                continue
            if (not baseline_conflict and result['remaining'] > 0):
                # Without a conflict, a partial alternative adds no
                # information.  Complete alternatives are still checked for
                # ambiguity.
                continue
            if (baseline_conflict and result['remaining'] > 0 and
                    result['consumed'] <= baseline['consumed']):
                continue
            candidates.append(_candidate_report(result, baseline,
                                                model_type))

        candidates.sort(key=lambda c: (
            0 if c['status'] == 'complete' else 1,
            -c['consumed'],
            c['structure_version_tuple'],
        ))

        complete = [c for c in candidates if c['status'] == 'complete']
        signatures = set()
        baseline_complete = (baseline['error'] is None and
                             baseline['remaining'] == 0)
        if baseline_complete:
            # The declared structure is itself a complete interpretation.
            signatures.add(((), (), ()))
        for candidate in complete:
            signature = (
                tuple((f['path'], f['offset'], f['end'], f['type'])
                      for f in candidate['added_fields']),
                tuple((f['path'], f['offset'], f['end'], f['type'])
                      for f in candidate['removed_fields']),
                tuple((c['path'], c['before']['offset'], c['after']['offset'])
                      for c in candidate['changed_fields']),
            )
            signatures.add(signature)
        if len(signatures) > 1:
            for candidate in complete:
                if candidate['affected_fields']:
                    candidate['assessment'] = 'ambiguous'
                    candidate['note'] = (
                        'Several structure versions consume the whole payload '
                        'but interpret different fields.  Byte length alone '
                        'cannot choose between them.'
                    )

        if baseline_complete and complete:
            interpretation['assessment'] = 'ambiguous'
            contradictions.append(dict(
                kind='ambiguous_structure',
                severity='warning',
                message=(
                    'The declared structure consumes the whole payload, but '
                    '%d other structure version(s) also consume it while '
                    'interpreting different fields.'
                    % len(complete)
                ),
                detail=dict(
                    candidate_versions=[
                        c['structure_version'] for c in complete
                    ],
                ),
            ))

    report = dict(
        tagid=record.get('tagid'),
        tagname=record.get('tagname'),
        seqno=record.get('seqno'),
        level=record.get('level'),
        model=model_type.__name__,
        resolved_model=getattr(baseline['model'].get('type'),
                               '__name__', None),
        extension_model=(getattr(baseline['model'].get('type'), '__name__')
                         != model_type.__name__),
        known_model=(model_type is not UnknownTagModel),
        declared_version=version_to_string(declared_version),
        declared_version_tuple=list(declared_version),
        payload_size=len(payload),
        interpretation=interpretation,
        extent=interpretation['extent'],
        interpreted=interpretation['extent'],
        consumed=interpretation['consumed'],
        unconsumed=interpretation['tail'],
        tail=interpretation['tail'],
        contradictions=contradictions,
        version_conflict=bool(contradictions),
        candidates=candidates,
        verified=False,
        assessment_policy=(
            'A structure is not considered correct merely because it consumes '
            'the entire payload.  Candidates remain unconfirmed unless '
            'independent evidence is supplied.'
        ),
    )
    return report, baseline['model']


# Readable aliases for callers that prefer diagnostic terminology.
interpret_record = import_record
diagnose_record = import_record


def import_records(records, version=None, context=None,
                   include_candidates=True):
    '''Import an iterable of records, returning one report per record.'''
    if version is None:
        raise ValueError('a declared version is required')
    return [
        import_record(record, version=version, context=context,
                      include_candidates=include_candidates)
        for record in records
    ]


def import_stream(stream, version=None, records=None,
                  include_candidates=True):
    '''Import all records of a record stream.

    ``stream`` may be a ``ModelStream``/``RecordStream`` or any object with a
    ``records()`` method.  Parent/child levels are honored so that extension
    models (controls, list headers, ...) are selected as usual.
    '''
    if version is None:
        version = getattr(stream, 'version', None)
    if version is None:
        raise ValueError('a declared version is required')

    if records is None:
        records = stream.records()

    reports = []
    stack = []
    for record in records:
        level = record.get('level', 0)
        while stack and stack[-1]['level'] >= level:
            stack.pop()

        context = dict(version=version)
        if stack:
            parent = stack[-1]
            context['parent'] = (parent['context'], parent['model'])

        report, model = _import_record(
            record, version=version, context=context,
            include_candidates=include_candidates,
        )
        reports.append(report)

        # The parsed model is needed as a parent context for child records.
        # It is intentionally not part of the public JSON report.
        parent_context = context
        parent_model = model
        stack.append(dict(level=level, context=parent_context,
                          model=parent_model, report=report))

    summary = dict(
        record_count=len(reports),
        records_with_unconsumed_tail=sum(
            1 for r in reports if r['tail']['size'] > 0),
        records_with_parse_error=sum(
            1 for r in reports if r['interpretation']['error'] is not None),
        records_with_contradictions=sum(
            1 for r in reports if r['contradictions']),
        records_with_candidates=sum(
            1 for r in reports if r['candidates']),
        verified_records=0,
    )
    return dict(
        declared_version=version_to_string(version_to_tuple(version)),
        declared_version_tuple=list(version_to_tuple(version)),
        summary=summary,
        records=reports,
        assessment_policy=(
            'Full payload consumption alone does not verify a structure '
            'interpretation.'
        ),
    )


structured_import = import_stream
stream_import = import_stream


def report_to_json(report, indent=2, sort_keys=True):
    return json.dumps(_jsonable(report), indent=indent,
                      sort_keys=sort_keys, ensure_ascii=False)


def format_record_report(report):
    '''Format one record report as a compact, human-readable text block.'''
    lines = []
    name = report.get('tagname') or report.get('model')
    lines.append('%s %s (declared %s) payload=%d' % (
        _format_seqno(report.get('seqno')),
        name,
        report['declared_version'],
        report['payload_size'],
    ))
    interpretation = report['interpretation']
    lines.append('  interpreted: [0, %d)  unconsumed: [%d, %d)' % (
        interpretation['consumed'],
        interpretation['tail']['offset'],
        interpretation['payload_size'],
    ))
    if interpretation['tail']['size']:
        lines.append('  tail: %s' % (interpretation['tail']['hex'] or ''))
    if interpretation['error']:
        error = interpretation['error']
        lines.append('  error: %s at %s: %s' % (
            error.get('type'), error.get('offset'), error.get('message')))

    for contradiction in report['contradictions']:
        lines.append('  contradiction [%s]: %s' % (
            contradiction['kind'], contradiction['message']))

    candidates = report['candidates']
    if candidates:
        lines.append('  candidate interpretations:')
    for candidate in candidates:
        lines.append('    structure %s (selected by declared versions %s): '
                     'consumed %d, remaining %d, %s, %s' % (
                         candidate['structure'],
                         candidate['structure_version_range'],
                         candidate['consumed'],
                         candidate['remaining'],
                         candidate['status'],
                         candidate['assessment'],
                     ))
        for added in candidate['added_fields']:
            lines.append('      + %s [%s, %s) %s = %r' % (
                added['path'], added['offset'], added['end'], added['type'],
                added['value']))
        for removed in candidate['removed_fields']:
            lines.append('      - %s [%s, %s) %s = %r' % (
                removed['path'], removed['offset'], removed['end'],
                removed['type'], removed['value']))
        for changed in candidate['changed_fields']:
            lines.append('      ~ %s %r -> %r' % (
                changed['path'], changed['before']['value'],
                changed['after']['value']))
        if candidate['status'] == 'complete':
            lines.append('      note: %s' % candidate['note'])
    return '\n'.join(lines)


def format_stream_report(report, declared_version_source=None):
    lines = []
    summary = report['summary']
    if declared_version_source:
        version_text = '%s (from %s)' % (report['declared_version'],
                                         declared_version_source)
    else:
        version_text = report['declared_version']
    lines.append('declared version %s: %d records, %d with tails, '
                 '%d with errors, %d with contradictions, %d candidates' % (
                     version_text,
                     summary['record_count'],
                     summary['records_with_unconsumed_tail'],
                     summary['records_with_parse_error'],
                     summary['records_with_contradictions'],
                     summary['records_with_candidates'],
                 ))
    for record_report in report['records']:
        if (record_report['contradictions'] or
                record_report['interpretation']['remaining']):
            lines.append('')
            lines.append(format_record_report(record_report))
    if len(lines) == 1:
        lines.append('no unconsumed payload or contradictory records')
    lines.append('')
    lines.append('note: %s' % report['assessment_policy'])
    return '\n'.join(lines)


def _format_seqno(seqno):
    if seqno is None:
        return '#?'
    return '#%d' % seqno
