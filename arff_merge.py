'''Merge ARFF partitions whose attributes are declared in different orders.

The merge is driven by attribute *names*, never by old column numbers.  Sparse
rows are expanded according to their own partition's declarations before they
are mapped to the merged schema, so an omitted numeric value becomes zero, an
omitted nominal value becomes that partition's first label, and an omitted
string value becomes ``"0"`` -- the same defaults used by :mod:`arff`.

Each output value keeps a reference to the partition and column it came from.
The returned object therefore contains, in addition to normal ``relation``,
``description``, ``attributes`` and ``data`` keys:

``provenance``
    A row-major matrix of ``SourceRef(partition, column)`` objects, or
    ``None`` where the partition does not declare that field.
``source_columns`` / ``source_partitions``
    The same information as integer matrices.
``sourced_data``
    A row-major matrix of dictionaries with ``value``, ``attribute``,
    ``partition``, ``column`` and ``explicit`` keys.  ``explicit`` is false
    for a sparse cell that was not written and received its implicit default.

``conflicts``
    Fields whose same-name declarations cannot be represented by one lossless
    type.  Each record lists every source partition, column and type.
``conflict_columns``
    The same conflicting source declarations, flattened for easy reporting.

Nominal domains are unified by label.  A label keeps its spelling and is never
reinterpreted through a numeric index from another partition.

When a contract is supplied it fixes the output order and, optionally, the
output types.  Types are otherwise inferred losslessly: nominal domains are
unioned, and ``INTEGER`` can widen to ``REAL`` or ``NUMERIC``.  If sources for
one field cannot be represented by a single lossless type, no column is added
to the merged header just to hold missing values.  The field is reported in
``conflicts`` and omitted from ``attributes`` and ``data``.

The result is a valid ARFF object and can be passed directly to
:func:`arff.dumps`.
'''

import argparse
import json
import numbers
import os
from collections import namedtuple

__all__ = [
    'MergeError', 'TypeConflictError', 'SourceRef',
    'merge_partitions', 'merge_arff', 'merge_arff_partitions',
    'merge_datasets', 'merge', 'combine', 'compose_dataset',
    'reconcile_partitions', 'align_partitions', 'combine_partitions',
    'merge_arff_files', 'merge_files', 'ArffMerger',
]


_SIMPLE_TYPES = ('NUMERIC', 'REAL', 'INTEGER', 'STRING')
_NUMERIC_RANK = {'INTEGER': 0, 'REAL': 1, 'NUMERIC': 2}
_NUMERIC_TYPES = tuple(_NUMERIC_RANK)

SourceRef = namedtuple('SourceRef', ['partition', 'column'])


class MergeError(Exception):
    '''Base exception raised by the merge helpers.'''


class TypeConflictError(MergeError):
    '''Raised when a field cannot be merged without losing type information.'''

    def __init__(self, conflict):
        self.conflict = conflict
        super().__init__(conflict['message'])


# ---------------------------------------------------------------------------
# Type helpers


def _normalise_type(type_):
    if isinstance(type_, (list, tuple)):
        return [str(value) for value in type_]
    if isinstance(type_, str):
        value = type_.strip()
        if value.startswith('{') and value.endswith('}'):
            body = value[1:-1].strip()
            if not body:
                return []
            import arff as _arff
            try:
                labels = _arff._parse_values(body)
            except Exception:
                labels = None
            if isinstance(labels, list):
                return [str(label) for label in labels]
            return [item.strip().strip('"\'')
                    for item in body.split(',') if item.strip()]
        value = value.upper()
        if value in _SIMPLE_TYPES:
            return value
    raise MergeError('Unsupported attribute type: %r' % (type_,))


def _is_nominal(type_):
    return isinstance(type_, (list, tuple))


def _type_signature(type_):
    if _is_nominal(type_):
        return ('NOMINAL', tuple(type_))
    return (type_,)


def _type_text(type_):
    if _is_nominal(type_):
        return '{%s}' % ', '.join(type_)
    return str(type_)


def _unify_types(types):
    '''Return a lossless common type, or ``None`` if one does not exist.'''
    if not types:
        return None

    normalised = [_normalise_type(type_) for type_ in types]
    first = normalised[0]
    if all(_type_signature(type_) == _type_signature(first)
           for type_ in normalised):
        return list(first) if _is_nominal(first) else first

    if all(_is_nominal(type_) for type_ in normalised):
        labels = []
        seen = set()
        for domain in normalised:
            for label in domain:
                if label not in seen:
                    labels.append(label)
                    seen.add(label)
        return labels

    if all(type_ in _NUMERIC_TYPES for type_ in normalised):
        return max(normalised, key=lambda type_: _NUMERIC_RANK[type_])

    return None


def _can_represent(source_type, target_type):
    source_type = _normalise_type(source_type)
    target_type = _normalise_type(target_type)

    if _is_nominal(target_type):
        return (_is_nominal(source_type) and
                set(source_type).issubset(set(target_type)))

    if target_type in _NUMERIC_TYPES:
        return (source_type in _NUMERIC_TYPES and
                _NUMERIC_RANK[source_type] <= _NUMERIC_RANK[target_type])

    if target_type == 'STRING':
        return source_type == 'STRING'

    return source_type == target_type


def _sparse_default(type_):
    if _is_nominal(type_):
        return type_[0] if type_ else None
    if type_ == 'STRING':
        return '0'
    if type_ == 'INTEGER':
        return 0
    return 0.0


# ---------------------------------------------------------------------------
# Input handling


def _is_existing_path(value):
    if not isinstance(value, str):
        return False
    try:
        return os.path.exists(value)
    except (OSError, ValueError):
        return False


def _read_source(source):
    if hasattr(source, 'read'):
        return source.read()
    if isinstance(source, bytes):
        return source.decode('utf-8')
    if isinstance(source, os.PathLike):
        with open(source, 'r', encoding='utf-8') as stream:
            return stream.read()
    if isinstance(source, str):
        if _is_existing_path(source):
            with open(source, 'r', encoding='utf-8') as stream:
                return stream.read()
        return source
    raise MergeError('Cannot read ARFF partition from %r' % (source,))


def _normalise_attributes(attributes):
    result = []
    names = set()
    for attribute in attributes:
        if isinstance(attribute, dict):
            name = attribute.get('name', attribute.get('attribute'))
            type_ = attribute.get('type', attribute.get('datatype'))
            if type_ is None and attribute.get('values') is not None:
                type_ = attribute['values']
        else:
            if not isinstance(attribute, (tuple, list)) or len(attribute) != 2:
                raise MergeError('Invalid attribute declaration: %r' %
                                 (attribute,))
            name, type_ = attribute
        if name is None:
            raise MergeError('Attribute declaration without a name: %r' %
                             (attribute,))
        name = str(name)
        if name in names:
            raise MergeError('Duplicate attribute name: %s' % name)
        names.add(name)
        result.append((name, _normalise_type(type_)))
    return result


def _coerce_value(value, type_):
    if value is None:
        return None

    if _is_nominal(type_):
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, numbers.Integral):
            if 0 <= int(value) < len(type_):
                return type_[int(value)]
            return str(int(value))
        return str(value)

    if type_ == 'INTEGER':
        if isinstance(value, numbers.Integral):
            return int(value)
        return int(float(value))
    if type_ in ('REAL', 'NUMERIC'):
        return float(value)
    if type_ == 'STRING':
        return str(value)
    return value


def _coo_rows(data):
    values, rows, columns = data
    row_count = max(rows) + 1 if len(rows) else 0
    result = [{} for _ in range(row_count)]
    for value, row, column in zip(values, rows, columns):
        result[row][column] = value
    return result


def _is_coo(data):
    if not (isinstance(data, tuple) and len(data) == 3):
        return False
    values, rows, columns = data
    if not (hasattr(values, '__len__') and hasattr(rows, '__len__') and
            hasattr(columns, '__len__')):
        return False
    if not (len(values) == len(rows) == len(columns)):
        return False
    return (all(isinstance(row, numbers.Integral) for row in rows) and
            all(isinstance(column, numbers.Integral) for column in columns))


def _materialise_rows(data, attributes):
    if hasattr(data, 'format'):
        if data.format != 'coo':
            data = data.tocoo()
        data = (data.data, data.row, data.col)
    elif getattr(data, 'tocoo', None) is not None:
        data = data.tocoo()
        data = (data.data, data.row, data.col)

    if _is_coo(data):
        data = _coo_rows(data)
    elif isinstance(data, dict):
        data = [data]

    name_to_index = {name: i for i, (name, _) in enumerate(attributes)}
    rows = []
    for raw_row in data:
        values = [None] * len(attributes)
        present = [False] * len(attributes)

        if isinstance(raw_row, dict):
            for key, value in raw_row.items():
                if isinstance(key, numbers.Integral):
                    column = int(key)
                else:
                    column = name_to_index.get(str(key))
                if column is None or not 0 <= column < len(attributes):
                    raise MergeError('Unknown sparse column %r' % (key,))
                values[column] = _coerce_value(value, attributes[column][1])
                present[column] = True
        elif isinstance(raw_row, (tuple, list)):
            if len(raw_row) != len(attributes):
                raise MergeError(
                    'Instance has %d attributes, expected %d' %
                    (len(raw_row), len(attributes)))
            for column, value in enumerate(raw_row):
                values[column] = _coerce_value(value, attributes[column][1])
                present[column] = True
        else:
            raise MergeError('Invalid data instance: %r' % (raw_row,))

        for column, (_, type_) in enumerate(attributes):
            if not present[column]:
                values[column] = _sparse_default(type_)

        rows.append((values, present))

    return rows


def _presence_masks(text, attributes):
    '''Recover which cells the data section explicitly writes.'''
    import arff as _arff

    lines = text.replace('\r\n', '\n').split('\n')
    data_line = None
    for index, line in enumerate(lines):
        if line.strip().upper().startswith('@DATA'):
            data_line = index
            break
    if data_line is None:
        return None

    masks = []
    for line in lines[data_line + 1:]:
        line = line.strip()
        if not line or line.startswith('%'):
            continue
        values = _arff._parse_values(line)
        mask = [False] * len(attributes)
        if isinstance(values, dict):
            for column in values:
                if 0 <= column < len(attributes):
                    mask[column] = True
        else:
            for column in range(min(len(values), len(attributes))):
                mask[column] = True
        masks.append(mask)
    return masks


class _Partition:
    def __init__(self, index, attributes, rows, relation='', description='',
                 source=None):
        self.index = index
        self.attributes = attributes
        self.rows = rows
        self.relation = relation or ''
        self.description = description or ''
        self.source = source
        self._columns = {name: i for i, (name, _) in enumerate(attributes)}

    def column_of(self, name):
        return self._columns.get(name)


def _load_partition(source, index):
    import arff as _arff

    text = None
    if isinstance(source, dict) and 'attributes' in source:
        obj = source
    else:
        text = _read_source(source)
        obj = _arff.loads(text)

    attributes = _normalise_attributes(obj.get('attributes', []))
    if not attributes:
        raise MergeError('Partition %d has no attributes' % index)

    rows = _materialise_rows(obj.get('data', []), attributes)
    if text is not None:
        masks = _presence_masks(text, attributes)
        if masks is not None and len(masks) == len(rows):
            rows = [(values, mask) for (values, _), mask in zip(rows, masks)]

    return _Partition(
        index=index,
        attributes=attributes,
        rows=rows,
        relation=obj.get('relation', ''),
        description=obj.get('description', ''),
        source=(obj.get('source') or
                (source if isinstance(source, str) and
                 _is_existing_path(source) else None)),
    )


# ---------------------------------------------------------------------------
# Contract handling


def _normalise_contract(contract):
    '''Return an ordered list of ``(name, type_or_None)`` contract fields.'''
    if contract is None:
        return None

    if isinstance(contract, dict):
        if 'attributes' in contract:
            contract = contract['attributes']
        elif 'fields' in contract:
            contract = contract['fields']
        else:
            contract = list(contract.items())
    elif isinstance(contract, str):
        if _is_existing_path(contract):
            with open(contract, 'r', encoding='utf-8') as stream:
                text = stream.read()
            try:
                loaded = json.loads(text)
            except ValueError:
                import arff as _arff
                loaded = _arff.loads(text).get('attributes', [])
        else:
            try:
                loaded = json.loads(contract)
            except ValueError:
                raise MergeError('Invalid contract: %r' % contract)
        return _normalise_contract(loaded)

    if not isinstance(contract, (list, tuple)):
        raise MergeError('Invalid contract: %r' % (contract,))

    result = []
    for entry in contract:
        if isinstance(entry, str):
            result.append((entry, None))
        elif isinstance(entry, dict):
            name = entry.get('name', entry.get('attribute'))
            if name is None:
                raise MergeError('Contract field without a name: %r' % entry)
            type_ = entry.get('type', entry.get('datatype'))
            if type_ is None and entry.get('values') is not None:
                type_ = entry['values']
            result.append((str(name),
                           None if type_ is None
                           else _normalise_type(type_)))
        elif isinstance(entry, (tuple, list)) and len(entry) == 2:
            name, type_ = entry
            result.append((str(name),
                           None if type_ is None
                           else _normalise_type(type_)))
        else:
            raise MergeError('Invalid contract field: %r' % (entry,))

    names = set()
    for name, _ in result:
        if name in names:
            raise MergeError('Duplicate contract field: %s' % name)
        names.add(name)
    return result


# ---------------------------------------------------------------------------
# Merge


def _sources_for(name, partitions):
    sources = []
    for partition in partitions:
        column = partition.column_of(name)
        if column is not None:
            sources.append({
                'partition': partition.index,
                'column': column,
                'type': partition.attributes[column][1],
            })
    return sources


def _conflict(name, sources, reason, message):
    return {
        'field': name,
        'kind': 'type',
        'reason': reason,
        'message': message,
        'resolved': 'omitted',
        'sources': sources,
        'types': sources,
        'partitions': [source['partition'] for source in sources],
    }


def _resolve_field(name, sources, promised):
    '''Return ``(target_type, conflict)`` for one output field.'''
    if promised is None:
        target = _unify_types([source['type'] for source in sources])
    else:
        target = list(promised) if _is_nominal(promised) else promised
        if _is_nominal(target):
            # The contract fixes the order; labels found in the partitions
            # extend it rather than being dropped.
            for source in sources:
                if _is_nominal(source['type']):
                    for label in source['type']:
                        if label not in target:
                            target.append(label)

    if target is None:
        return None, _conflict(
            name, sources, 'incompatible_types',
            'Attribute %s has no lossless common type: %s' %
            (name, ', '.join('%s (partition %d, column %d)' %
                             (_type_text(source['type']),
                              source['partition'], source['column'])
                             for source in sources)))

    incompatible = [source for source in sources
                    if not _can_represent(source['type'], target)]
    if incompatible:
        return None, _conflict(
            name, sources, 'incompatible_types',
            'Attribute %s cannot be represented as %s without losing type '
            'information; conflicting sources: %s' %
            (name, _type_text(target),
             ', '.join('%s (partition %d, column %d)' %
                       (_type_text(source['type']), source['partition'],
                        source['column']) for source in incompatible)))

    return target, None


def _missing_field_conflict(name):
    return {
        'field': name,
        'kind': 'missing',
        'reason': 'not_declared',
        'message': ('Contract field %s is not declared by any partition' %
                    name),
        'resolved': 'omitted',
        'sources': [],
        'types': [],
        'partitions': [],
    }


def _render(value, target_type):
    if value is None:
        return None
    if _is_nominal(target_type):
        return str(value)
    if target_type == 'INTEGER':
        return int(float(value))
    if target_type in ('REAL', 'NUMERIC'):
        return float(value)
    if target_type == 'STRING':
        return str(value)
    return value


def _normalise_partitions(partitions):
    if len(partitions) == 1:
        candidate = partitions[0]
        if (isinstance(candidate, (list, tuple)) and
                not _is_coo(candidate) and
                not (isinstance(candidate, dict) and
                     'attributes' in candidate)):
            partitions = tuple(candidate)
    return list(partitions)


def merge_partitions(*partitions, **options):
    '''Merge ARFF partitions using a field contract.

    Parameters
    ----------
    *partitions:
        ARFF partitions.  A partition may be a path, ARFF text, a file-like
        object, or an already decoded ARFF dictionary.  A single list or tuple
        of partitions is also accepted.
    contract:
        Optional field contract.  It may be an ordered list of ``(name,
        type)`` pairs, a list of field dictionaries, a mapping of names to
        types, or a path/JSON string for either.  A type of ``None`` is
        inferred from the partitions.  The contract determines the output
        attribute order.
    relation, description:
        Override the corresponding metadata in the result.
    on_conflict:
        ``'report'`` (the default) records conflicting fields in
        ``conflicts`` and omits them from the merged header.  ``'raise'``
        raises :class:`TypeConflictError` instead.
    provenance:
        Set to false to omit the provenance matrices.

    The aliases ``field_contract``, ``fields`` and ``schema`` are accepted in
    place of ``contract``.
    '''
    contract = options.pop('contract', None)
    for alias in ('field_contract', 'fields', 'schema'):
        if contract is None and alias in options:
            contract = options.pop(alias)
    relation = options.pop('relation', None)
    description = options.pop('description', None)
    on_conflict = options.pop('on_conflict',
                              options.pop('conflict_policy', 'report'))
    include_provenance = options.pop(
        'provenance', options.pop('include_provenance', True))
    if 'partitions' in options and not partitions:
        partitions = (options.pop('partitions'),)
    if options:
        raise TypeError('Unexpected merge option(s): %s' %
                        ', '.join(sorted(options)))
    if on_conflict not in ('report', 'raise'):
        raise ValueError("on_conflict must be 'report' or 'raise'")

    partitions = _normalise_partitions(partitions)
    if not partitions:
        raise MergeError('At least one partition is required')

    loaded = [_load_partition(partition, index)
              for index, partition in enumerate(partitions)]
    contract = _normalise_contract(contract)

    conflicts = []
    ignored = []

    if contract is None:
        names = []
        seen = set()
        for partition in loaded:
            for name, _ in partition.attributes:
                if name not in seen:
                    names.append(name)
                    seen.add(name)
        contract = [(name, None) for name in names]
    else:
        contract_names = {name for name, _ in contract}
        for partition in loaded:
            for name, type_ in partition.attributes:
                if name not in contract_names:
                    ignored.append({
                        'partition': partition.index,
                        'attribute': name,
                        'column': partition.column_of(name),
                        'type': _type_text(type_),
                    })

    output_fields = []
    for name, promised in contract:
        sources = _sources_for(name, loaded)
        if not sources:
            conflicts.append(_missing_field_conflict(name))
            continue

        target, conflict = _resolve_field(name, sources, promised)
        if conflict is not None:
            conflicts.append(conflict)
            if on_conflict == 'raise':
                raise TypeConflictError(conflict)
            continue

        output_fields.append({
            'name': name,
            'type': target,
            'sources': {source['partition']: source for source in sources},
        })

    data = []
    provenance = []
    sourced_data = []

    for partition in loaded:
        for values, present in partition.rows:
            row_values = []
            row_refs = []
            row_sourced = []
            for field in output_fields:
                source = field['sources'].get(partition.index)
                if source is None:
                    value = None
                    ref = None
                    explicit = False
                else:
                    column = source['column']
                    value = _render(values[column], field['type'])
                    ref = SourceRef(partition.index, column)
                    explicit = present[column]
                row_values.append(value)
                row_refs.append(ref)
                row_sourced.append({
                    'attribute': field['name'],
                    'value': value,
                    'partition': None if ref is None else ref.partition,
                    'column': None if ref is None else ref.column,
                    'explicit': explicit,
                })
            data.append(row_values)
            provenance.append(row_refs)
            sourced_data.append(row_sourced)

    if relation is None:
        relation = next((partition.relation for partition in loaded
                         if partition.relation), 'merged')
    if description is None:
        seen = []
        for partition in loaded:
            if partition.description and partition.description not in seen:
                seen.append(partition.description)
        description = '\n'.join(seen)

    result = {
        'relation': relation,
        'description': description,
        'attributes': [(field['name'], field['type'])
                       for field in output_fields],
        'data': data,
        'conflicts': conflicts,
        'conflict_columns': [
            dict(source, field=conflict['field'])
            for conflict in conflicts
            for source in conflict['sources']
        ],
        'ignored_attributes': ignored,
        'partitions': [
            {
                'index': partition.index,
                'relation': partition.relation,
                'attributes': list(partition.attributes),
                'source': partition.source,
            }
            for partition in loaded
        ],
    }

    if include_provenance:
        result.update({
            'provenance': provenance,
            'sources': provenance,
            'source_columns': [
                [None if ref is None else ref.column for ref in row]
                for row in provenance
            ],
            'source_partitions': [
                [None if ref is None else ref.partition for ref in row]
                for row in provenance
            ],
            'sourced_data': sourced_data,
            'field_sources': {
                field['name']: [
                    {
                        'partition': source['partition'],
                        'column': source['column'],
                        'type': _type_text(source['type']),
                    }
                    for source in field['sources'].values()
                ]
                for field in output_fields
            },
        })

    return result


# Aliases --------------------------------------------------------------------

merge_arff = merge_partitions
merge_arff_partitions = merge_partitions
merge_datasets = merge_partitions
merge = merge_partitions
combine = merge_partitions
compose_dataset = merge_partitions
reconcile_partitions = merge_partitions
align_partitions = merge_partitions
combine_partitions = merge_partitions


class ArffMerger(object):
    '''Convenience wrapper around :func:`merge_partitions`.'''

    def __init__(self, **options):
        self.options = options

    def merge(self, *partitions):
        return merge_partitions(*partitions, **self.options)

    def merge_files(self, paths, output=None):
        return merge_arff_files(paths, output=output, **self.options)


def merge_arff_files(paths, output=None, **options):
    '''Merge ARFF files, optionally writing the result to ``output``.'''
    if isinstance(paths, (str, bytes, os.PathLike)):
        paths = [paths]
    result = merge_partitions(list(paths), **options)
    if output is not None:
        import arff as _arff
        text = _arff.dumps(result)
        if hasattr(output, 'write'):
            output.write(text)
        else:
            with open(output, 'w', encoding='utf-8') as stream:
                stream.write(text)
    return result


merge_files = merge_arff_files


# CLI ------------------------------------------------------------------------


def _json_default(value):
    if isinstance(value, SourceRef):
        return list(value)
    raise TypeError('Object of type %s is not JSON serializable' %
                    type(value).__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Merge ARFF partitions by field name.')
    parser.add_argument('partitions', nargs='+',
                        help='ARFF files to merge')
    parser.add_argument('-c', '--contract',
                        help='JSON contract, or an ARFF file whose header '
                             'defines the contract')
    parser.add_argument('-o', '--output',
                        help='Write merged ARFF here instead of stdout')
    parser.add_argument('--relation', help='Merged relation name')
    parser.add_argument('--description', help='Merged description')
    parser.add_argument('--on-conflict', default='report',
                        choices=('report', 'raise'))
    parser.add_argument('--conflicts',
                        help='Write the conflict report as JSON here')
    parser.add_argument('--provenance',
                        help='Write the provenance matrix as JSON here')
    parser.add_argument('--fail-on-conflict', action='store_true',
                        help='Exit with status 2 when conflicts are found')
    args = parser.parse_args(argv)

    result = merge_arff_files(
        args.partitions,
        output=args.output,
        contract=args.contract,
        relation=args.relation,
        description=args.description,
        on_conflict=args.on_conflict,
    )

    if args.conflicts:
        with open(args.conflicts, 'w', encoding='utf-8') as stream:
            json.dump(result['conflicts'], stream, indent=2,
                      ensure_ascii=False, default=_json_default)
            stream.write('\n')

    if args.provenance:
        with open(args.provenance, 'w', encoding='utf-8') as stream:
            json.dump(result['provenance'], stream, indent=2,
                      ensure_ascii=False, default=_json_default)
            stream.write('\n')

    if args.output is None:
        import arff as _arff
        print(_arff.dumps(result))

    if result['conflicts']:
        import sys
        sys.stderr.write('%d field conflict(s) found\n' %
                         len(result['conflicts']))
        if args.fail_on_conflict:
            return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
