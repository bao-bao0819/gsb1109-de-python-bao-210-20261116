'''
Streaming, mergeable statistics for continuously received ARFF partitions.

The :class:`ArffBaseline` class keeps a keyed registry of the currently valid
samples and derives per-column statistics from it.  Samples may be corrected
(by ingesting the same key again with a newer revision) or retracted.  Repeated
transmission of the same source row is therefore counted once, not twice.

Sparse ARFF rows follow the usual ARFF semantics:

* a column omitted from a sparse row has the value zero, and that zero is a
  valid observation;
* an explicit ``?`` is a missing value and is not included in any statistic.

Statistics are accumulated exactly (using :class:`fractions.Fraction`), so
adding and retracting large, nearly cancelling values does not accumulate
rounding error.  A snapshot is converted to floating point only when it is
created.

Published results should keep the :class:`BaselineSnapshot` they obtained from
:meth:`ArffBaseline.snapshot`.  Snapshots are immutable and do not change when
more partitions are ingested or when earlier samples are corrected/retracted::

    baseline = ArffBaseline.from_arff(shard_one, partition='shard-1')
    published = baseline.snapshot()

    baseline.ingest(shard_two, partition='shard-2')
    baseline.retract(('shard-1', 0))

    # ``published`` still describes the baseline at publication time.
    published['temperature'].mean
'''


import hashlib
import json
import math
import numbers

from collections import Counter
from fractions import Fraction
from types import MappingProxyType

import arff


__all__ = [
    'ArffBaseline',
    'BaselineSnapshot',
    'ColumnSnapshot',
    'ColumnStatistics',
    'SchemaMismatchError',
    'ConflictingRevisionError',
    'ArffStatisticsError',
    'merge_baselines',
]


class ArffStatisticsError(ValueError):
    '''Base error for statistics bookkeeping problems.'''


class SchemaMismatchError(ArffStatisticsError):
    '''Raised when statistics for different attribute layouts are merged.'''


class ConflictingRevisionError(ArffStatisticsError):
    '''Raised when two shards disagree at the same revision.'''

    def __init__(self, key, revision):
        super().__init__(
            'Conflicting values for sample %r at revision %r'
            % (key, revision)
        )
        self.key = key
        self.revision = revision


def _is_missing(value):
    if value is None:
        return True
    try:
        result = value != value
    except Exception:
        return False
    try:
        return bool(result)
    except Exception:
        return False


def _is_number(value):
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _as_fraction(value):
    if isinstance(value, Fraction):
        return value
    try:
        return Fraction(value)
    except (OverflowError, ValueError):
        raise ValueError('Statistics require finite numeric values, got %r'
                         % (value,))


def _as_float(value):
    try:
        return float(value)
    except OverflowError:
        return float('inf') if value > 0 else float('-inf')


def _check_revision(revision):
    if revision is None:
        return None
    if isinstance(revision, bool) or not isinstance(revision, int):
        raise TypeError('revision must be a non-negative integer or None')
    if revision < 0:
        raise ValueError('revision must be a non-negative integer or None')
    return revision


class ColumnSnapshot:
    '''An immutable view of one column's statistics.

    ``count`` is the number of valid (non-missing) values, ``missing`` is the
    number of explicit missing values, and ``sample_count`` is their sum.
    ``variance`` is the population variance.  For an empty column ``sum`` is
    ``0`` while ``mean``, ``variance``, ``stdev``, ``minimum`` and ``maximum``
    are ``None``.

    Non-numeric columns expose ``count``, ``missing`` and ``frequencies`` only;
    their numeric fields are ``None``.
    '''

    __slots__ = (
        'name', 'numeric', 'count', 'missing', 'sum', 'mean', 'variance',
        'stdev', 'minimum', 'maximum', 'frequencies', '_frozen',
    )

    def __init__(self, name, numeric, count, missing, total=None, mean=None,
                 variance=None, stdev=None, minimum=None, maximum=None,
                 frequencies=None):
        self.name = name
        self.numeric = numeric
        self.count = count
        self.missing = missing
        self.sum = total
        self.mean = mean
        self.variance = variance
        self.stdev = stdev
        self.minimum = minimum
        self.maximum = maximum
        if frequencies is None:
            self.frequencies = None
        else:
            self.frequencies = MappingProxyType(dict(frequencies))
        self._frozen = True

    def __setattr__(self, name, value):
        if getattr(self, '_frozen', False):
            raise AttributeError('ColumnSnapshot is immutable')
        object.__setattr__(self, name, value)

    @property
    def n(self):
        '''Alias for the number of valid values (``count``).'''
        return self.count

    @property
    def total(self):
        '''Alias for the sum of valid numeric values.'''
        return self.sum

    @property
    def std(self):
        '''Alias for the population standard deviation (``stdev``).'''
        return self.stdev

    def normalize(self, value):
        '''Return the z-score of ``value`` against this frozen column.

        Missing values and columns without a defined mean return ``None``.
        A constant column returns ``0.0`` for its single value and raises
        :class:`ValueError` for any other value, making the degenerate case
        explicit instead of dividing by zero.
        '''
        if _is_missing(value) or self.mean is None:
            return None
        value = float(value)
        if self.stdev is None:
            return None
        if self.stdev == 0.0:
            if value == self.mean:
                return 0.0
            raise ValueError(
                'Cannot normalize against a constant column %r' % (self.name,)
            )
        return (value - self.mean) / self.stdev

    @property
    def sample_count(self):
        '''Number of samples represented by this column (valid + missing).'''
        return self.count + self.missing

    def __eq__(self, other):
        if not isinstance(other, ColumnSnapshot):
            return NotImplemented
        return self.as_dict() == other.as_dict()

    def __repr__(self):
        return ('ColumnSnapshot(name=%r, count=%r, missing=%r, mean=%r, '
                'variance=%r)' % (self.name, self.count, self.missing,
                                  self.mean, self.variance))

    def as_dict(self):
        '''Return the snapshot as a plain dictionary.'''
        result = {
            'name': self.name,
            'numeric': self.numeric,
            'count': self.count,
            'missing': self.missing,
            'sample_count': self.sample_count,
        }
        if self.numeric:
            result.update({
                'sum': self.sum,
                'mean': self.mean,
                'variance': self.variance,
                'stdev': self.stdev,
                'minimum': self.minimum,
                'maximum': self.maximum,
            })
        else:
            result['frequencies'] = dict(self.frequencies or {})
        return result


class ColumnStatistics:
    '''A mergeable, retractable accumulator for a single column.

    ``add`` adds one observation, ``remove`` retracts one previously added
    observation, and ``merge`` folds another accumulator of the same kind into
    this one.  Missing values (``None`` or NaN) are counted separately and do
    not contribute to numeric statistics.

    The accumulator uses exact rational arithmetic for sums and sums of
    squares.  This makes ``add``/``remove``/``merge`` order independent within
    the precision of the input values, so a final snapshot matches a
    recomputation from the currently valid values even after dramatic
    cancellation.
    '''

    __slots__ = (
        'name', 'numeric', '_count', '_missing', '_sum', '_sum_squares',
        '_values', '_frequencies',
    )

    def __init__(self, name=None, numeric=True):
        self.name = name
        self.numeric = bool(numeric)
        self._count = 0
        self._missing = 0
        self._sum = Fraction(0)
        self._sum_squares = Fraction(0)
        self._values = Counter()
        self._frequencies = None if self.numeric else Counter()

    @property
    def count(self):
        '''Number of valid (non-missing) values.'''
        return self._count

    @property
    def missing(self):
        '''Number of explicit missing values.'''
        return self._missing

    @property
    def n(self):
        '''Alias for the number of valid values (``count``).'''
        return self._count

    @property
    def total(self):
        '''Alias for the exact sum of valid numeric values.'''
        if not self.numeric:
            return None
        return self._sum

    def add(self, value):
        '''Add one observation to the accumulator.'''
        if _is_missing(value):
            self._missing += 1
            return

        if not self.numeric:
            self._count += 1
            self._frequencies[value] += 1
            return

        number = _as_fraction(value)
        self._count += 1
        self._sum += number
        self._sum_squares += number * number
        self._values[value] += 1

    def remove(self, value):
        '''Retract one previously added observation.

        The value must still be present; retracting a value that is not part of
        the current statistics raises :class:`ValueError` instead of silently
        corrupting the accumulator.
        '''
        if _is_missing(value):
            if self._missing <= 0:
                raise ValueError('Cannot remove missing value: count is zero')
            self._missing -= 1
            return

        if not self.numeric:
            if self._frequencies.get(value, 0) <= 0:
                raise ValueError('Cannot remove absent value %r' % (value,))
            self._count -= 1
            self._frequencies[value] -= 1
            if self._frequencies[value] == 0:
                del self._frequencies[value]
            return

        if self._values.get(value, 0) <= 0:
            raise ValueError('Cannot remove absent value %r' % (value,))
        number = _as_fraction(value)
        self._count -= 1
        self._sum -= number
        self._sum_squares -= number * number
        self._values[value] -= 1
        if self._values[value] == 0:
            del self._values[value]

    def merge(self, other):
        '''Merge another accumulator into this one and return ``self``.'''
        if not isinstance(other, ColumnStatistics):
            raise TypeError('Can only merge ColumnStatistics instances')
        if self.numeric != other.numeric:
            raise SchemaMismatchError(
                'Cannot merge numeric and non-numeric column statistics'
            )

        self._count += other._count
        self._missing += other._missing
        if self.numeric:
            self._sum += other._sum
            self._sum_squares += other._sum_squares
            self._values.update(other._values)
        else:
            self._frequencies.update(other._frequencies)
        return self

    def copy(self):
        '''Return an independent copy of this accumulator.'''
        result = ColumnStatistics(self.name, self.numeric)
        result._count = self._count
        result._missing = self._missing
        result._sum = self._sum
        result._sum_squares = self._sum_squares
        result._values = self._values.copy()
        if not self.numeric:
            result._frequencies = self._frequencies.copy()
        return result

    def snapshot(self):
        '''Freeze the current state into a :class:`ColumnSnapshot`.'''
        if not self.numeric:
            return ColumnSnapshot(
                self.name, False, self._count, self._missing,
                frequencies=self._frequencies,
            )

        total = _as_float(self._sum)
        if self._count == 0:
            return ColumnSnapshot(
                self.name, True, 0, self._missing, total=0.0,
            )

        mean = self._sum / self._count
        variance = self._sum_squares / self._count - mean * mean
        if variance < 0:
            # Exact arithmetic makes this impossible, but keep the invariant
            # explicit for alternate numeric inputs.
            variance = Fraction(0)
        variance_float = _as_float(variance)
        if math.isinf(variance_float):
            stdev = float('inf')
        else:
            stdev = math.sqrt(variance_float)

        return ColumnSnapshot(
            name=self.name,
            numeric=True,
            count=self._count,
            missing=self._missing,
            total=total,
            mean=_as_float(mean),
            variance=variance_float,
            stdev=stdev,
            minimum=min(self._values),
            maximum=max(self._values),
        )

    def __repr__(self):
        return ('ColumnStatistics(name=%r, numeric=%r, count=%r, missing=%r)'
                % (self.name, self.numeric, self._count, self._missing))


class BaselineSnapshot:
    '''An immutable statistics snapshot bound to one baseline state.

    Keep the object returned by :meth:`ArffBaseline.snapshot` for as long as a
    published result needs it.  Subsequent ingestion, corrections and
    retractions only affect later snapshots.
    '''

    __slots__ = ('relation', 'attributes', 'sample_count', '_columns',
                 '_frozen')

    def __init__(self, relation, attributes, sample_count, columns):
        self.relation = relation
        self.attributes = tuple(attributes)
        self.sample_count = sample_count
        self._columns = MappingProxyType(dict(columns))
        self._frozen = True

    def __setattr__(self, name, value):
        if getattr(self, '_frozen', False):
            raise AttributeError('BaselineSnapshot is immutable')
        object.__setattr__(self, name, value)

    @property
    def schema(self):
        '''The snapshot's ``(name, type)`` attribute tuple.'''
        return self.attributes

    @property
    def columns(self):
        '''Mapping of column name to :class:`ColumnSnapshot`.'''
        return self._columns

    def __getitem__(self, name):
        return self._columns[name]

    def __contains__(self, name):
        return name in self._columns

    def __iter__(self):
        return iter(self._columns)

    def __len__(self):
        return len(self._columns)

    def __repr__(self):
        return ('BaselineSnapshot(relation=%r, sample_count=%r, columns=%r)'
                % (self.relation, self.sample_count, list(self._columns)))

    def as_dict(self):
        '''Return a plain, recursively copied dictionary representation.'''
        return {
            'relation': self.relation,
            'sample_count': self.sample_count,
            'attributes': [
                [name, list(type_) if isinstance(type_, (list, tuple))
                 else type_]
                for name, type_ in self.attributes
            ],
            'columns': {
                name: column.as_dict()
                for name, column in self._columns.items()
            },
        }

    def normalize(self, values):
        '''Normalize one row (sequence or mapping) against this snapshot.

        Returns a list with one entry per attribute.  Numeric attributes are
        converted to z-scores; missing values and non-numeric attributes are
        passed through unchanged.  The snapshot itself never changes, so a
        published normalization keeps the basis it was created from.
        '''
        if hasattr(values, 'keys'):
            row = [
                values[name] if name in values else None
                for name in self._columns
            ]
        else:
            row = list(values)
            if len(row) != len(self._columns):
                raise ValueError(
                    'Expected %d values, got %d'
                    % (len(self._columns), len(row))
                )

        result = []
        for name, value in zip(self._columns, row):
            column = self._columns[name]
            if column.numeric:
                result.append(column.normalize(value))
            else:
                result.append(value)
        return result

    def normalize_numeric(self, values):
        '''Return only the normalized numeric attributes of one row.

        ``values`` may be a sequence or a mapping keyed by attribute name.
        '''
        if hasattr(values, 'keys'):
            values = [values.get(name) for name in self._columns]
        else:
            values = list(values)
            if len(values) != len(self._columns):
                raise ValueError(
                    'Expected %d values, got %d'
                    % (len(self._columns), len(values))
                )
        return [
            self._columns[name].normalize(value)
            for name, value in zip(self._columns, values)
            if self._columns[name].numeric
        ]


class _Entry:
    __slots__ = ('values', 'revision', 'active')

    def __init__(self, values, revision, active=True):
        # ``active`` is false for a retraction (tombstone).  The withdrawn
        # values are retained when known, so a verbatim retransmission cannot
        # resurrect the row while a genuinely newer value can.
        self.values = values
        self.revision = revision
        self.active = active

    def __repr__(self):
        if not self.active:
            return '_Entry(retracted, values=%r, revision=%r)' % (
                self.values, self.revision)
        return '_Entry(values=%r, revision=%r)' % (self.values, self.revision)


def _normalise_attributes(attributes):
    result = []
    names = set()
    for attribute in attributes:
        if len(attribute) != 2:
            raise ValueError('Invalid attribute declaration %r' % (attribute,))
        name, type_ = attribute
        if name in names:
            raise ValueError('Duplicate attribute name %r' % (name,))
        names.add(name)
        if isinstance(type_, str):
            type_ = type_.upper()
        else:
            type_ = tuple(type_)
        result.append((name, type_))
    return tuple(result)


def _is_numeric_type(type_):
    return isinstance(type_, str) and type_ in ('NUMERIC', 'REAL', 'INTEGER')


class ArffBaseline:
    '''A keyed, continuously updated statistics baseline for ARFF partitions.

    Samples are identified by a caller-supplied key.  Ingesting an existing key
    replaces that sample's values instead of adding another observation, so a
    retransmitted source row is counted once.  Retracting a key removes its
    contribution from every column.

    Revisions order corrections:

    * an upsert without ``revision`` is always accepted and receives the next
      local revision;
    * an upsert with a revision lower than the stored one is ignored;
    * equal revisions with equal values are a no-op;
    * equal revisions with different values raise
      :class:`ConflictingRevisionError`, which makes a cross-shard conflict
      explicit instead of order-dependent.

    ``merge`` follows the same rules, so the result of merging shards does not
    depend on merge order when their revisions are unambiguous.
    '''

    def __init__(self, attributes=None, relation=None):
        self._relation = relation
        self._attributes = None
        self._columns = {}
        self._entries = {}
        self._revision_counter = 0
        self._anonymous_partition = 0
        self._snapshot_cache = None
        if attributes is not None:
            self._set_schema(attributes, relation)

    @property
    def schema(self):
        '''The baseline's ``(name, type)`` attribute tuple.'''
        return self.attributes

    # ------------------------------------------------------------------
    # Schema and basic container behaviour
    @property
    def relation(self):
        return self._relation

    @property
    def attributes(self):
        if self._attributes is None:
            return ()
        return self._attributes

    @property
    def column_names(self):
        return tuple(name for name, _ in self.attributes)

    def __len__(self):
        return sum(1 for entry in self._entries.values() if entry.active)

    def __contains__(self, key):
        entry = self._entries.get(key)
        return entry is not None and entry.active

    def __repr__(self):
        return ('ArffBaseline(relation=%r, samples=%r, columns=%r)'
                % (self._relation, len(self), list(self._columns)))

    def _set_schema(self, attributes, relation=None):
        attributes = _normalise_attributes(attributes)
        self._attributes = attributes
        if relation is not None:
            self._relation = relation
        self._columns = {
            name: ColumnStatistics(name, _is_numeric_type(type_))
            for name, type_ in attributes
        }
        self._snapshot_cache = None

    def _ensure_schema(self, attributes, relation=None):
        if not attributes:
            # A dict source may omit schema metadata; keep the existing
            # schema, and leave an uninitialised baseline uninitialised so a
            # later explicit schema is not rejected.
            return
        attributes = _normalise_attributes(attributes)
        if self._attributes is None:
            self._set_schema(attributes, relation)
            return
        if attributes != self._attributes:
            raise SchemaMismatchError(
                'Attribute schema %r does not match baseline %r'
                % (attributes, self._attributes)
            )

    # ------------------------------------------------------------------
    # Ingesting ARFF and rows
    @classmethod
    def from_arff(cls, source, **kwargs):
        '''Create a baseline from an ARFF document or file-like object.'''
        baseline = cls()
        baseline.ingest(source, **kwargs)
        return baseline

    def ingest(self, source, partition=None, keys=None, revision=None,
               source_key=None):
        '''Ingest one ARFF partition.

        ``source`` may be an ARFF string, a file-like object, or an already
        decoded ARFF dictionary.

        When ``keys`` is omitted, the key is ``(partition, row_index)``.  Pass
        a stable ``partition`` value so retransmitting a growing partition
        keeps each row's identity.  Without one, the partition identity is the
        SHA-256 of the source, so resending an identical document is counted
        once while different shards stay distinct.

        Pass ``source_key=`` with a column name (or index) to key rows by that
        column's value instead: the same source row appearing in different
        documents is then counted once and its later occurrence acts as a
        correction.

        ``revision`` is a non-negative integer applied to every row in the
        partition.

        Returns the tuple of keys used for the ingested rows.
        '''
        partition_token = None
        if keys is None and partition is None and source_key is None:
            partition_token = self._source_identity(source)

        obj = self._decode_source(source)
        rows = obj.get('data', [])
        if rows is None:
            rows = []
        else:
            rows = list(rows)

        self._ensure_schema(obj.get('attributes') or [], obj.get('relation'))

        if source_key is not None:
            keys = self._keys_from_source(rows, source_key)
        elif keys is None:
            if partition is None:
                if partition_token is None:
                    self._anonymous_partition += 1
                    partition = ('anonymous', self._anonymous_partition)
                else:
                    partition = partition_token
            keys = [(partition, index) for index in range(len(rows))]
        else:
            keys = list(keys)
            if len(keys) != len(rows):
                raise ValueError(
                    'Expected %d keys, got %d'
                    % (len(rows), len(keys))
                )

        return self.ingest_rows(rows, keys, revision=revision)

    def _keys_from_source(self, rows, source_key):
        if isinstance(source_key, int) and not isinstance(source_key, bool):
            index = source_key
            if not 0 <= index < len(self._attributes):
                raise ValueError('Invalid source key index %r' % (source_key,))
        else:
            names = [name for name, _ in self._attributes]
            try:
                index = names.index(source_key)
            except ValueError:
                raise ValueError('Unknown source key attribute %r'
                                 % (source_key,))
        keys = []
        for row in rows:
            if isinstance(row, dict):
                row = self._row_from_sparse_dict(row)
            keys.append(('source', row[index]))
        return keys

    def ingest_rows(self, rows, keys, revision=None):
        '''Ingest already-decoded rows with explicit sample keys.

        ``rows`` must expose one value per baseline attribute.  ``keys``
        must be unique per source row and have the same length as ``rows``.
        '''
        rows = list(rows)
        keys = list(keys)
        if len(rows) != len(keys):
            raise ValueError(
                'Expected %d keys, got %d' % (len(rows), len(keys))
            )
        if len(set(keys)) != len(keys):
            raise ValueError('Sample keys must be unique within a partition')
        if self._attributes is None:
            raise SchemaMismatchError(
                'Cannot ingest rows before an attribute schema is known'
            )
        if revision is not None and not isinstance(revision, int):
            raise TypeError('revision must be an integer or None')
        revision = _check_revision(revision)
        normalised_rows = [self._normalise_row(row) for row in rows]
        if revision is not None:
            # Validate the whole partition before mutating anything, so a
            # conflicting shard does not leave a half-applied baseline.
            for key, values in zip(keys, normalised_rows):
                self._check_upsert(revision, key, values)
        for key, values in zip(keys, normalised_rows):
            self._upsert(key, values, revision)
        return tuple(keys)

    def _decode_source(self, source):
        if isinstance(source, dict):
            return source
        if isinstance(source, bytes):
            source = source.decode('utf-8')
        if isinstance(source, str):
            return arff.loads(source)
        if hasattr(source, 'read'):
            return arff.load(source)
        raise TypeError('Unsupported ARFF source %r' % (source,))

    @staticmethod
    def _source_identity(source):
        '''Return a stable default partition token for the given source.'''
        if isinstance(source, bytes):
            payload = source
        elif isinstance(source, str):
            payload = source.encode('utf-8')
        elif isinstance(source, dict):
            payload = json.dumps(
                source, sort_keys=True, default=repr
            ).encode('utf-8')
        elif hasattr(source, 'read'):
            name = getattr(source, 'name', None)
            if not isinstance(name, str):
                return None
            payload = name.encode('utf-8', 'replace')
        else:
            return None
        return ('content', hashlib.sha256(payload).hexdigest())

    def _normalise_row(self, row):
        if isinstance(row, dict):
            row = self._row_from_sparse_dict(row)

        if len(row) != len(self._attributes):
            raise ValueError(
                'Expected %d values, got %d' % (len(self._attributes),
                                                len(row))
            )
        values = []
        for (_, type_), value in zip(self._attributes, row):
            if _is_numeric_type(type_):
                if isinstance(value, str):
                    if value in ('', '?'):
                        value = None
                    elif type_ == 'INTEGER':
                        value = int(float(value))
                    else:
                        value = float(value)
                elif value is not None and not _is_number(value):
                    value = float(value)
                if _is_missing(value):
                    value = None
                else:
                    _as_fraction(value)
            elif _is_missing(value):
                value = None
            values.append(value)
        return tuple(values)

    def _row_from_sparse_dict(self, row):
        '''Expand a decoded sparse (LOD) row into a dense sequence.

        Omitted numeric attributes are the ARFF sparse zero; omitted nominal
        attributes take their first declared value, matching the decoder's
        dense representation.  Explicit ``None`` values stay missing.
        '''
        values = []
        for index, (_, type_) in enumerate(self._attributes):
            if index in row:
                values.append(row[index])
            elif _is_numeric_type(type_):
                values.append(0)
            elif isinstance(type_, (list, tuple)) and type_:
                values.append(type_[0])
            else:
                values.append(0)

        unexpected = [
            key for key in row
            if not isinstance(key, int)
            or isinstance(key, bool)
            or not 0 <= key < len(self._attributes)
        ]
        if unexpected:
            raise ValueError(
                'Sparse row contains invalid attribute index/indices %r'
                % (sorted(unexpected, key=repr),)
            )
        return values
    # ------------------------------------------------------------------
    # Corrections and retractions
    def upsert(self, key, values, revision=None):
        '''Insert or correct one sample.

        Returns ``True`` when the stored sample changed, ``False`` when the
        update was an idempotent retransmission or a stale revision.
        '''
        if self._attributes is None:
            raise SchemaMismatchError(
                'Cannot ingest values before an attribute schema is known'
            )
        return self._upsert(key, self._normalise_row(values), revision)

    def retract(self, key, revision=None):
        '''Retract one sample.

        Returns ``True`` when a live sample was retracted and ``False`` when
        the key is unknown, already retracted, or the supplied ``revision`` is
        stale.  Retracting the last valid value of a column leaves that column
        with ``count == 0``, ``sum == 0`` and no defined mean/variance/minimum/
        maximum.

        The retraction is recorded as a tombstone.  Merging an older shard
        that still contains the row therefore cannot resurrect it, and merging
        a newer revision of the same key restores the row explicitly.
        '''
        entry = self._entries.get(key)

        revision = _check_revision(revision)
        if revision is None:
            revision = self._next_revision()
        elif revision > self._revision_counter:
            self._revision_counter = revision

        if entry is not None:
            if revision < entry.revision:
                return False
            if revision == entry.revision:
                if entry.active:
                    raise ConflictingRevisionError(key, revision)
                return False
            if not entry.active:
                entry.revision = revision
                return False
            for (name, _), value in zip(self._attributes, entry.values):
                self._columns[name].remove(value)

        stored_values = entry.values if entry is not None else None
        self._entries[key] = _Entry(stored_values, revision, active=False)
        self._snapshot_cache = None
        return entry is not None

    def _next_revision(self):
        self._revision_counter += 1
        return self._revision_counter

    def _check_upsert(self, revision, key, values):
        '''Raise if applying this upsert would conflict, without mutating.'''
        existing = self._entries.get(key)
        if existing is None or revision < existing.revision:
            return
        if revision == existing.revision:
            if existing.active and existing.values == values:
                return
            raise ConflictingRevisionError(key, revision)
        if existing.active and existing.values == values:
            return

    def _upsert(self, key, values, revision):
        revision = _check_revision(revision)
        if revision is None:
            revision = self._next_revision()
        elif revision > self._revision_counter:
            self._revision_counter = revision

        existing = self._entries.get(key)
        if existing is not None:
            if revision < existing.revision:
                return False
            if revision == existing.revision:
                if existing.active and existing.values == values:
                    return False
                # Same revision but a different value, or an upsert competing
                # with a retraction, is genuinely ambiguous.
                raise ConflictingRevisionError(key, revision)
            if not existing.active:
                if existing.values == values:
                    # Verbatim retransmission of a withdrawn row: keep the
                    # tombstone but advance its revision.
                    existing.revision = revision
                    return False
                # A genuinely newer, different value resurrects the row.
            elif values == existing.values:
                # Idempotent retransmission: remember the newer revision so a
                # later stale shard is still rejected, but do not touch the
                # accumulated statistics.
                existing.revision = revision
                return False
            else:
                for (name, _), value in zip(self._attributes, existing.values):
                    self._columns[name].remove(value)

        for (name, _), value in zip(self._attributes, values):
            self._columns[name].add(value)
        self._entries[key] = _Entry(values, revision)
        self._snapshot_cache = None
        return True

    # ------------------------------------------------------------------
    # Merging
    def merge(self, other):
        '''Merge another baseline into this one and return ``self``.

        Both baselines must describe the same attribute schema.  Shared sample
        keys are resolved by revision; identical values are never counted
        twice.  Distinct keys are simply combined.
        '''
        self._merge_revisions(other)
        for key, entry in other._entries.items():
            if entry.active:
                self._upsert(key, entry.values, entry.revision)
            else:
                self.retract(key, entry.revision)
        return self

    def _merge_revisions(self, other):
        '''Validate and stage a merge before applying it.

        Validation happens first so a conflicting or schema-mismatched merge
        leaves ``self`` unchanged.
        '''
        if not isinstance(other, ArffBaseline):
            raise TypeError('Can only merge ArffBaseline instances')
        if other._attributes is None:
            return
        if self._attributes is not None:
            if other._attributes != self._attributes:
                raise SchemaMismatchError(
                    'Attribute schema %r does not match baseline %r'
                    % (other._attributes, self._attributes)
                )

        for key, entry in other._entries.items():
            existing = self._entries.get(key)
            if existing is None or entry.revision != existing.revision:
                continue
            if existing.active != entry.active:
                raise ConflictingRevisionError(key, entry.revision)
            if existing.active and entry.values != existing.values:
                raise ConflictingRevisionError(key, entry.revision)

        if self._attributes is None:
            self._set_schema(other._attributes, other._relation)

    def copy(self):
        '''Return an independent copy of the baseline.'''
        result = ArffBaseline()
        result._relation = self._relation
        result._attributes = self._attributes
        result._columns = {
            name: column.copy() for name, column in self._columns.items()
        }
        result._entries = {
            key: _Entry(entry.values, entry.revision, entry.active)
            for key, entry in self._entries.items()
        }
        result._revision_counter = self._revision_counter
        result._anonymous_partition = self._anonymous_partition
        return result

    # ------------------------------------------------------------------
    # Snapshots
    def snapshot(self):
        '''Return an immutable snapshot of the current valid samples.'''
        if self._snapshot_cache is None:
            columns = {
                name: column.snapshot()
                for name, column in self._columns.items()
            }
            self._snapshot_cache = BaselineSnapshot(
                self._relation, self._attributes or (), len(self),
                columns,
            )
        return self._snapshot_cache

    def freeze(self):
        '''Alias for :meth:`snapshot`; returns the frozen baseline state.'''
        return self.snapshot()

    def column(self, name):
        '''Return the live :class:`ColumnStatistics` for one column.'''
        return self._columns[name]


def merge_baselines(baselines):
    '''Merge an iterable of baselines into a new baseline.

    The inputs are not modified.  The result is independent of merge order as
    long as shared keys carry unambiguous revisions.
    '''
    result = None
    for baseline in baselines:
        if not isinstance(baseline, ArffBaseline):
            raise TypeError('Can only merge ArffBaseline instances')
        if result is None:
            result = baseline.copy()
        else:
            result.merge(baseline)
    return ArffBaseline() if result is None else result
