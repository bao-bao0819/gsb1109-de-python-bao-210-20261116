"""
Record views and value-aware comparison of SAS data sets.

Reordering the physical columns of a data set must not make every row
look like a new case.  The tools here compare records by *field
identity* -- the agreed variable name -- rather than by physical column
position, while keeping the original column order and source schema
available for reporting.
"""

# Standard Library
import math
from dataclasses import dataclass, field

# Community Packages
import pandas as pd

# Xport Modules
import xport

__all__ = [
    'FieldMapping',
    'Record',
    'RecordView',
    'ChangeKind',
    'FieldChange',
    'RecordDiff',
    'DatasetDiff',
    'record_view',
    'diff_datasets',
]


class Record:
    """
    One observation as ordered field-name/value pairs.

    Unlike a dictionary, a :class:`Record` preserves repeated field
    names, which is essential for data sets containing duplicate column
    names.  Name lookup returns the first matching field; use
    :meth:`get_all` or :meth:`value_at` to reach every occurrence.
    """

    def __init__(self, pairs):
        """
        Initialize from an iterable of ``(name, value)`` pairs.
        """
        self._pairs = tuple(pairs)

    def __getitem__(self, name):
        """
        Get the value of the first field called ``name``.
        """
        for candidate, value in self._pairs:
            if candidate == name:
                return value
        raise KeyError(name)

    def get_all(self, name):
        """
        Get every value stored under a repeated field name.
        """
        return [value for candidate, value in self._pairs if candidate == name]

    def value_at(self, index):
        """
        Get a value by its position within the record.
        """
        return self._pairs[index][1]

    def items(self):
        """
        The ordered ``(name, value)`` pairs.
        """
        return self._pairs

    def names(self):
        """
        The field names in record order.
        """
        return tuple(name for name, _ in self._pairs)

    def values(self):
        """
        The values in record order.
        """
        return tuple(value for _, value in self._pairs)

    def __iter__(self):
        """
        Iterate over field names.
        """
        return iter(name for name, _ in self._pairs)

    def __len__(self):
        """
        Number of fields.
        """
        return len(self._pairs)

    def __eq__(self, other):
        """
        Equality, including against a plain dict for unique field names.
        """
        if isinstance(other, Record):
            return self._pairs == other._pairs
        if isinstance(other, dict):
            return dict(self._pairs) == other
        return NotImplemented

    def __repr__(self):
        """REPL-format."""
        return f'{type(self).__name__}({list(self._pairs)!r})'


class FieldMapping:
    """
    Explicit mapping between the field names of two data sets.

    Duplicate field names are supported and resolved by *position*, so
    building the mapping never relies on a dictionary silently
    overwriting same-named keys.  When no explicit ``pairs`` are given,
    fields are matched by name, with duplicates matched in the order
    they appear.

        >>> left = xport.Dataset({'A': [1], 'B': [2]})
        >>> right = xport.Dataset({'X': [1], 'B': [2]})
        >>> mapping = FieldMapping.build(left, right, pairs={'A': 'X'})
        >>> mapping.pairs
        (('A', 'X'), ('B', 'B'))
    """

    def __init__(self, pairs=(), index_pairs=None):
        """
        Initialize from ``(left_name, right_name)`` pairs.

        ``index_pairs`` optionally records the physical column position
        of each pair, which disambiguates duplicate field names.
        """
        normalized = tuple(pairs)
        for pair in normalized:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError(f'Field mapping pairs must be 2-tuples, got {pair!r}')
        # Same-named duplicate fields can legitimately repeat; the
        # physical positions (when known) are the true identity.
        identity = tuple(index_pairs) if index_pairs is not None else normalized
        if len(set(identity)) != len(identity):
            raise ValueError(f'Duplicate field mapping pair in {normalized!r}')
        self.pairs = normalized
        self.index_pairs = tuple(index_pairs) if index_pairs is not None else None

    @classmethod
    def build(cls, left, right, pairs=None):
        """
        Build a mapping between two data sets.

        ``pairs`` optionally pairs renamed fields.  Each side of a pair
        is either a field name or a ``(name, occurrence)`` tuple, where
        ``occurrence`` is a zero-based index used to disambiguate
        duplicate field names.  A plain name matches the next
        not-yet-matched occurrence, so duplicate names never rely on
        dictionary overwriting.  Unmapped fields that share a name are
        matched automatically; fields that cannot be paired are reported
        as unmatched and do not take part in record comparison.
        """
        left_names = list(getattr(left, 'columns', ()))
        right_names = list(getattr(right, 'columns', ()))
        explicit = []
        if pairs:
            for item in pairs.items() if isinstance(pairs, dict) else pairs:
                explicit.append((item[0], item[1]))

        # Track occurrences so same-named duplicates are consumed in
        # order instead of one dictionary entry hiding the others.
        left_taken = [False] * len(left_names)
        right_taken = [False] * len(right_names)
        resolved = []
        for left_ref, right_ref in explicit:
            left_index = _resolve_index(left_names, left_taken, left_ref)
            right_index = _resolve_index(right_names, right_taken, right_ref)
            resolved.append((left_index, right_index))
            left_taken[left_index] = True
            right_taken[right_index] = True

        for left_index, left_name in enumerate(left_names):
            if left_taken[left_index]:
                continue
            right_index = _next_index(right_names, right_taken, left_name)
            if right_index is not None:
                resolved.append((left_index, right_index))
                left_taken[left_index] = True
                right_taken[right_index] = True

        resolved.sort()
        return cls(
            [(left_names[i], right_names[j]) for i, j in resolved],
            index_pairs=resolved,
        )

    @property
    def left_fields(self):
        """The left-side field names, in physical column order."""
        return tuple(left for left, _ in self.pairs)

    @property
    def right_fields(self):
        """The paired right-side field names, in left column order."""
        return tuple(right for _, right in self.pairs)

    @property
    def left_positions(self):
        """Physical column positions of the left-side fields."""
        if self.index_pairs is None:
            return tuple(range(len(self.pairs)))
        return tuple(i for i, _ in self.index_pairs)

    @property
    def right_positions(self):
        """Physical column positions of the right-side fields."""
        if self.index_pairs is None:
            return tuple(range(len(self.pairs)))
        return tuple(j for _, j in self.index_pairs)

    def flipped(self):
        """
        Swap the left and right sides of the mapping.
        """
        index_pairs = None
        if self.index_pairs is not None:
            index_pairs = [(j, i) for i, j in self.index_pairs]
        return FieldMapping(
            ((right, left) for left, right in self.pairs),
            index_pairs=index_pairs,
        )

    def __eq__(self, other):
        """Equality."""
        return isinstance(other, FieldMapping) and self.pairs == other.pairs

    def __repr__(self):
        """REPL-format."""
        return f'{type(self).__name__}(pairs={self.pairs!r})'


def _next_index(names, taken, name):
    """
    First not-yet-taken position of ``name`` in ``names``.
    """
    for index, candidate in enumerate(names):
        if not taken[index] and candidate == name:
            return index
    return None


def _resolve_index(names, taken, ref):
    """
    Resolve a field reference (name or ``(name, occurrence)``) to a position.
    """
    if isinstance(ref, tuple):
        name, occurrence = ref
        matches = [index for index, candidate in enumerate(names) if candidate == name]
        try:
            index = matches[occurrence]
        except IndexError as exc:
            raise KeyError(f'Unknown field reference {ref!r}') from exc
    else:
        name = ref
        if name not in names:
            raise KeyError(f'Unknown field {name!r}')
        index = _next_index(names, taken, name)
        if index is None:
            raise KeyError(f'No unmatched occurrence left for field {name!r}')
    if taken[index]:
        raise ValueError(f'Field reference {ref!r} is already used by another mapping')
    return index


def _mapping_with_positions(mapping, left, right):
    """
    Fill in physical column positions for a position-free mapping.

    Occurrences of a repeated name are consumed in order.
    """
    left_names = list(left.columns)
    right_names = list(right.columns)
    left_taken = [False] * len(left_names)
    right_taken = [False] * len(right_names)
    index_pairs = []
    for left_name, right_name in mapping.pairs:
        left_index = _resolve_index(left_names, left_taken, left_name)
        right_index = _resolve_index(right_names, right_taken, right_name)
        left_taken[left_index] = True
        right_taken[right_index] = True
        index_pairs.append((left_index, right_index))
    return FieldMapping(mapping.pairs, index_pairs=index_pairs)


class RecordView:
    """
    Record-oriented view of a data set, keyed by field identity.

    The view presents each observation as a mapping of field name to
    value, regardless of the physical column order.  The source data
    set, its physical column order, and the field mapping are retained
    so they can be cited in comparison reports.
    """

    def __init__(self, dataset, mapping=None, side='left'):
        """
        Build a record view.

        ``side`` selects which half of the :class:`FieldMapping`
        identifies this view's fields: ``'left'`` or ``'right'``.
        """
        self.dataset = dataset
        self.mapping = mapping
        self.side = side
        if mapping is None:
            self.fields = tuple(dataset.columns)
            self.positions = tuple(range(len(self.fields)))
        elif side == 'left':
            self.fields = mapping.left_fields
            self.positions = mapping.left_positions
        elif side == 'right':
            self.fields = mapping.right_fields
            self.positions = mapping.right_positions
        else:
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    @property
    def physical_order(self):
        """
        Field names in the data set's physical column order.
        """
        return tuple(self.dataset.columns)

    def __len__(self):
        """
        Number of observations.
        """
        return len(self.dataset)

    def __iter__(self):
        """
        Iterate over observations as :class:`Record` objects.
        """
        for position in range(len(self.dataset)):
            yield self.record(position)

    def record(self, position):
        """
        Get one observation as an ordered :class:`Record`.
        """
        row = self.dataset.iloc[position]
        return Record(
            (name, row.iloc[column]) for name, column in zip(self.fields, self.positions)
        )

    def key(self, position, key_fields):
        """
        Get the tuple of values that identify an observation.
        """
        row = self.dataset.iloc[position]
        by_name = {}
        for name, column in zip(self.fields, self.positions):
            by_name.setdefault(name, []).append(column)
        positions = [by_name[name].pop(0) for name in key_fields]
        return tuple(row.iloc[column] for column in positions)


def record_view(dataset, mapping=None, side='left'):
    """
    Construct a :class:`RecordView` for a data set.
    """
    return RecordView(dataset, mapping=mapping, side=side)


class ChangeKind:
    """
    How a field value changed between two records.

    ``UNCHANGED``     the value is the same.
    ``FORMATTED``     the value is equal, but the rendered text differs.
    ``VALUE``         the underlying value changed.
    ``MISSING``       present on one side, missing on the other.
    """

    UNCHANGED = 'unchanged'
    FORMATTED = 'formatted'
    VALUE = 'value'
    MISSING = 'missing'


@dataclass(frozen=True)
class FieldChange:
    """
    A single field-level difference between two records.
    """

    field: str
    kind: str
    left: object = None
    right: object = None
    left_text: str = ''
    right_text: str = ''

    def __str__(self):
        """
        One-line human-readable description.
        """
        if self.kind == ChangeKind.FORMATTED:
            return (
                f'{self.field}: format changed {self.left_text!r} -> '
                f'{self.right_text!r} (value {self.left!r})'
            )
        if self.kind == ChangeKind.MISSING:
            return f'{self.field}: missing {self.left_text!r} -> {self.right_text!r}'
        return f'{self.field}: {self.left!r} -> {self.right!r}'


@dataclass(frozen=True)
class RecordDiff:
    """
    The difference between two records sharing an identity.
    """

    key: tuple
    left_position: int
    right_position: int
    changes: tuple = field(default_factory=tuple)

    @property
    def is_identical(self):
        """
        Whether every paired field is unchanged.
        """
        return not self.changes

    @property
    def value_changes(self):
        """
        Changes to underlying values, including missingness.
        """
        return tuple(
            change
            for change in self.changes
            if change.kind in (ChangeKind.VALUE, ChangeKind.MISSING)
        )

    @property
    def format_changes(self):
        """
        Changes that only affect how an equal value is rendered.
        """
        return tuple(change for change in self.changes if change.kind == ChangeKind.FORMATTED)


@dataclass(frozen=True)
class DatasetDiff:
    """
    The result of comparing two data sets by field identity.
    """

    mapping: FieldMapping
    key_fields: tuple
    added: tuple = field(default_factory=tuple)
    removed: tuple = field(default_factory=tuple)
    changed: tuple = field(default_factory=tuple)
    matched: tuple = field(default_factory=tuple)
    unmatched_left: tuple = field(default_factory=tuple)
    unmatched_right: tuple = field(default_factory=tuple)

    @property
    def is_identical(self):
        """
        Whether the data sets contain the same records and values.
        """
        return not (self.added or self.removed or self.changed)

    def __bool__(self):
        """
        A diff is truthy when it reports any difference.
        """
        return not self.is_identical

    def __str__(self):
        """
        Human-readable summary.
        """
        lines = []
        if self.unmatched_left or self.unmatched_right:
            lines.append(
                f'unmatched fields: left={list(self.unmatched_left)!r} '
                f'right={list(self.unmatched_right)!r}'
            )
        for record in self.removed:
            lines.append(f'removed {record}')
        for record in self.added:
            lines.append(f'added {record}')
        for record in self.changed:
            for change in record.changes:
                lines.append(f'{record.key}: {change}')
        if not lines:
            lines.append('no differences')
        return '\n'.join(lines)


def diff_datasets(left, right, key, pairs=None):
    """
    Compare two data sets record-by-record using agreed field identity.

    Column order is irrelevant: records are paired through the explicit
    :class:`FieldMapping` (renamed fields included), then matched within
    each data set by the values of ``key`` (a field name or sequence of
    field names from the left data set).  Format-only changes and true
    value changes are reported separately, and SAS special missing
    categories (``xport.NaN.A`` ... ``.Z``, ``._``) participate in the
    comparison as distinct values.

        >>> left = xport.Dataset({'ID': [1], 'X': [1.0]})
        >>> right = xport.Dataset({'X': [1.0], 'ID': [1]})  # reordered
        >>> result = diff_datasets(left, right, key='ID')
        >>> result.is_identical
        True
    """
    mapping = pairs if isinstance(pairs, FieldMapping) else FieldMapping.build(left, right, pairs)
    if mapping.index_pairs is None:
        mapping = _mapping_with_positions(mapping, left, right)
    key_names = (key,) if isinstance(key, str) else tuple(key)
    for name in key_names:
        if name not in mapping.left_fields:
            raise KeyError(f'Key field {name!r} is not part of the field mapping')

    left_view = RecordView(left, mapping, side='left')
    right_view = RecordView(right, mapping, side='right')
    right_key_fields = _paired_names(mapping.pairs, key_names, side=1)

    left_index = _index_records(left_view, key_names)
    right_index = _index_records(right_view, right_key_fields)

    left_keys = set(left_index)
    right_keys = set(right_index)
    removed = tuple(sorted(left_keys - right_keys, key=_key_sort))
    added = tuple(sorted(right_keys - left_keys, key=_key_sort))

    changed_records = []
    matched_records = []
    for record_key in sorted(left_keys & right_keys, key=_key_sort):
        left_positions = left_index[record_key]
        right_positions = right_index[record_key]
        if len(left_positions) != 1 or len(right_positions) != 1:
            # Duplicate keys cannot be paired one-to-one; treat the
            # surplus observations as removed and added.
            for position in left_positions:
                removed = removed + (left_view.key(position, key_names),)
            for position in right_positions:
                added = added + (right_view.key(position, right_key_fields),)
            continue
        left_position = left_positions[0]
        right_position = right_positions[0]
        record_diff = _diff_record(
            mapping,
            left_view.record(left_position),
            right_view.record(right_position),
            left,
            right,
            left_position,
            right_position,
            record_key,
        )
        if record_diff.is_identical:
            matched_records.append(record_diff)
        else:
            changed_records.append(record_diff)

    matched_left_positions = set(mapping.left_positions)
    matched_right_positions = set(mapping.right_positions)
    unmatched_left = tuple(
        name
        for position, name in enumerate(left.columns)
        if position not in matched_left_positions
    )
    unmatched_right = tuple(
        name
        for position, name in enumerate(right.columns)
        if position not in matched_right_positions
    )
    return DatasetDiff(
        mapping=mapping,
        key_fields=tuple(key_names),
        added=added,
        removed=removed,
        changed=tuple(changed_records),
        matched=tuple(matched_records),
        unmatched_left=unmatched_left,
        unmatched_right=unmatched_right,
    )


def _index_records(view, key_fields):
    """
    Map each record key to the physical positions holding that key.
    """
    index = {}
    for position in range(len(view)):
        record_key = view.key(position, key_fields)
        index.setdefault(record_key, []).append(position)
    return index


def _paired_names(pairs, names, side):
    """
    Names on one side that pair with ``names`` on the other side.

    Occurrences are consumed in order, so repeated field names stay
    unambiguous.
    """
    remaining = list(names)
    result = []
    for pair in pairs:
        wanted = pair[0] if side == 1 else pair[1]
        for index, name in enumerate(remaining):
            if name == wanted:
                result.append(pair[side])
                remaining.pop(index)
                break
    if remaining:
        raise KeyError(f'Unmapped key field {remaining[0]!r}')
    return result


def _diff_record(
    mapping,
    left_record,
    right_record,
    left_dataset,
    right_dataset,
    left_position,
    right_position,
    record_key,
):
    """
    Compare the paired fields of two records.
    """
    changes = []
    for pair_index, (left_name, right_name) in enumerate(mapping.pairs):
        left_value = left_record.value_at(pair_index)
        right_value = right_record.value_at(pair_index)
        kind = _classify(left_value, right_value)
        if kind == ChangeKind.UNCHANGED:
            left_text = _render_at(
                left_dataset, mapping.left_positions[pair_index], left_name, left_position,
                left_value,
            )
            right_text = _render_at(
                right_dataset, mapping.right_positions[pair_index], right_name, right_position,
                right_value,
            )
            if left_text == right_text:
                continue
            kind = ChangeKind.FORMATTED
        else:
            left_text = _render_at(
                left_dataset, mapping.left_positions[pair_index], left_name, left_position,
                left_value,
            )
            right_text = _render_at(
                right_dataset, mapping.right_positions[pair_index], right_name, right_position,
                right_value,
            )
        changes.append(
            FieldChange(
                field=left_name,
                kind=kind,
                left=left_value,
                right=right_value,
                left_text=left_text,
                right_text=right_text,
            )
        )
    return RecordDiff(
        key=record_key,
        left_position=left_position,
        right_position=right_position,
        changes=tuple(changes),
    )


def _classify(left, right):
    """
    Classify the relationship between two field values.
    """
    if _same_value(left, right):
        return ChangeKind.UNCHANGED
    if _is_missing(left) or _is_missing(right):
        return ChangeKind.MISSING
    return ChangeKind.VALUE


def _same_value(left, right):
    """
    Value equality that distinguishes SAS special missing categories.
    """
    if _is_missing(left) and _is_missing(right):
        return _missing_tag(left) == _missing_tag(right)
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def _missing_tag(value):
    """
    Canonical tag for a missing value.

    Special missing values (``xport.NaN``) keep their letter or
    underscore tag; ordinary missing values all share the ``None`` tag,
    so a plain NaN never matches a tagged NaN.
    """
    if isinstance(value, xport.NaN):
        return value.name
    return None


def _is_missing(value):
    """
    Whether a value is any form of missing, including special missing.
    """
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _render_at(dataset, column_position, name, row_position, value):
    """
    Render one value as text using the variable's SAS format, if any.

    Uniquely-named variables are read by name so their ``Variable``
    metadata (format, label, ...) is preserved; duplicate field names
    fall back to the physical column position.
    """
    if list(dataset.columns).count(name) == 1:
        variable = dataset[name]
    else:
        variable = dataset.iloc[:, column_position]
    fmt = getattr(variable, 'format', None)
    if fmt is None or not (fmt.name or fmt.length):
        return '' if _is_missing(value) else str(value)
    return _format_value(value, fmt)


def _format_value(value, fmt):
    """
    Apply a SAS format specification to produce display text.
    """
    if _is_missing(value):
        return ''
    name = fmt.name.upper().lstrip('$')
    if name in ('CHAR', ''):
        text = str(value)
        return text[: fmt.length].ljust(fmt.length)
    if name in ('COMMA', 'DOLLAR'):
        text = f'{float(value):,.{fmt.decimals}f}'
        if name == 'DOLLAR':
            text = '$' + text
        return text.rjust(fmt.length)
    if name in ('PERCENT',):
        return f'{float(value):.{fmt.decimals}%}'.rjust(fmt.length)
    if name in ('DATE',):
        return str(pd.Timestamp('1960-01-01') + pd.to_timedelta(float(value), unit='D'))
    if fmt.decimals:
        return f'{float(value):.{fmt.decimals}f}'.rjust(fmt.length)
    return str(value)


def _key_sort(key):
    """
    Sort key that tolerates unorderable and missing key values.
    """
    return tuple((_is_missing(part), repr(part)) for part in key)
