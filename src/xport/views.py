"""
Row-and-column views of a SAS :class:`~xport.Dataset`, merged by field.

A view is an independent :class:`~xport.Dataset` derived from a source
dataset.  Variable labels, formats, informats, widths, types, and SAS
special missing values are inherited from the source.  Views may be
modified independently, then merged back into the source by field
identity -- variable name and row label.

Merging never mutates the source dataset, the views, or any branch that
does not participate in a change.  The result separates compatible
contributions, which can be applied, from conflicts, where the same
source attribute was changed to different values.  Changing a value in
one branch and a format in another is complementary, not conflicting.
"""

# Standard Library
import copy
import math
from collections import OrderedDict

# Community Packages
import pandas as pd

# Xport Modules
import xport

__all__ = [
    'Conflict',
    'Contribution',
    'MergeResult',
    'derive_view',
    'merge_views',
    'set_metadata',
    'sync_metadata',
]


# Variable-level metadata carried from a source dataset into a view.
VARIABLE_METADATA = (
    'label',
    'width',
    'vtype',
    'format',
    'informat',
)

# Pandas clears the cached Variable objects after writes, dropping their
# SAS metadata.  ``DataFrame.attrs`` survives writes and copies, so it
# holds the durable, per-frame metadata side store.
METADATA_KEY = '_xport_view_metadata'


class Contribution:
    """
    A branch's changes to one variable.

    Attributes:
        branch: Zero-based position of the view among the merged views.
        column: Name of the variable.
        cells: Tuple of ``(row, column)`` keys with changed values.
        values: Mapping of cell key to the branch's value.
        metadata: Mapping of metadata aspect to the branch's value.
    """

    def __init__(self, branch, column, cells=(), values=None, metadata=None):
        self.branch = branch
        self.column = column
        self.cells = tuple(cells)
        self.values = dict(values or {})
        self.metadata = OrderedDict(metadata or {})

    def __repr__(self):
        return (
            f'{type(self).__name__}(branch={self.branch!r}, '
            f'column={self.column!r}, cells={self.cells!r})'
        )

    def __eq__(self, other):
        if not isinstance(other, Contribution):
            return NotImplemented
        return (
            self.branch == other.branch
            and self.column == other.column
            and self.cells == other.cells
            and _same_mapping(self.values, other.values)
            and _same_mapping(self.metadata, other.metadata)
        )


class Conflict:
    """
    Incompatible changes to the same source attribute.

    A value conflict occurs when two branches give a cell different
    values.  A metadata conflict occurs when two branches give the same
    variable aspect different values.
    """

    def __init__(self, kind, column, branches, values, rows=(), aspect=None):
        self.kind = kind
        self.column = column
        self.branches = tuple(branches)
        self.values = dict(values)
        self.rows = tuple(rows)
        self.aspect = aspect

    def __repr__(self):
        if self.kind == 'value':
            target = f'cells={tuple((row, self.column) for row in self.rows)}'
        else:
            target = f'column={self.column!r}, aspect={self.aspect!r}'
        return (
            f'{type(self).__name__}(kind={self.kind!r}, {target}, '
            f'branches={self.branches!r}, values={self.values!r})'
        )

    def __eq__(self, other):
        if not isinstance(other, Conflict):
            return NotImplemented
        return (
            self.kind == other.kind
            and self.column == other.column
            and self.branches == other.branches
            and self.rows == other.rows
            and self.aspect == other.aspect
            and _same_mapping(self.values, other.values)
        )


class MergeResult:
    """
    Outcome of merging views back into a source dataset.

    Attributes:
        merged: A new dataset with the compatible contributions applied.
        compatible: Contributions that were applied to ``merged``.
        conflicts: Incompatible contributions that were not applied.
        originals: Every branch's raw contributions, including conflicts.
    """

    def __init__(self, merged, compatible, conflicts, originals):
        self.merged = merged
        self.compatible = tuple(compatible)
        self.conflicts = tuple(conflicts)
        self.originals = tuple(originals)

    def __repr__(self):
        return (
            f'{type(self).__name__}(compatible={len(self.compatible)}, '
            f'conflicts={len(self.conflicts)})'
        )

    @property
    def ok(self):
        """True when the merge produced no conflicts."""
        return not self.conflicts


def derive_view(dataset, rows=None, columns=None, name=None):
    """
    Create an independent row-and-column view of a :class:`~xport.Dataset`.

    The view copies the selected observations and variables, including
    each variable's label, width, type, format, informat, and SAS special
    missing values.  Later changes to the view or the source cannot
    affect one another.

    Args:
        dataset: The source dataset.
        rows: Row labels to include, or all rows when ``None``.
        columns: Variable names to include, or all variables when
            ``None``.  A single name may be given as a string.
        name: Name for the new view dataset.

    Returns:
        Dataset: A standalone dataset carrying the source metadata.
    """
    if columns is None:
        columns = list(dataset.columns)
    elif isinstance(columns, str):
        columns = [columns]
    else:
        columns = list(columns)
    missing = [column for column in columns if column not in dataset.columns]
    if missing:
        raise KeyError(f'Unknown variable(s) in source: {missing!r}')

    if rows is None:
        selected = dataset.loc[:, columns]
    else:
        if isinstance(rows, str) or not hasattr(rows, '__iter__'):
            rows = [rows]
        rows = list(rows)
        unknown = [row for row in rows if row not in dataset.index]
        if unknown:
            raise KeyError(f'Unknown row label(s) in source: {unknown!r}')
        selected = dataset.loc[rows, columns]

    snapshots = {
        column: _load_metadata(dataset, column) for column in columns
    }
    tagged = {
        column: _tagged_missing(dataset[column]) for column in columns
    }

    view = xport.Dataset(selected, copy=True)
    source_name = getattr(dataset, 'name', None)
    view.name = name if name is not None else source_name
    view.attrs[METADATA_KEY] = {
        column: copy.deepcopy(snapshots[column]) for column in columns
    }
    for column in columns:
        _restore_tagged_missing(view, column, tagged[column])
    _sync_variables(view)
    return view


def merge_views(source, *views, name=None):
    """
    Merge independent views back into ``source`` by field identity.

    Fields are identified by variable name and row label.  Each branch
    is diffed against the source; unchanged cells do nothing.  A cell or
    metadata aspect changed by only one branch, or changed identically
    by several branches, is applied.  Distinct changes to the same cell
    or the same variable aspect are returned as conflicts and left at
    the source value.  A value change in one branch and a metadata
    change in another are complementary and both apply.

    Neither the source nor any view is mutated.  Variables and rows that
    no branch changes are passed through unchanged.

    Args:
        source: The dataset from which the views were derived.
        *views: Datasets returned by :func:`derive_view`.
        name: Name for the merged dataset; defaults to the source name.

    Returns:
        MergeResult: The merged dataset, compatible contributions,
        conflicts, and the original branch contributions.
    """
    views = list(views)
    for branch, view in enumerate(views):
        extra = [column for column in view.columns if column not in source.columns]
        if extra:
            raise KeyError(
                f'View {branch} has variable(s) absent from the source: {extra!r}'
            )
        unknown = [row for row in view.index if row not in source.index]
        if unknown:
            raise KeyError(
                f'View {branch} has row label(s) absent from the source: {unknown!r}'
            )

    # Build a deep copy through the constructor rather than
    # ``DataFrame.copy``, which clears the item cache on the source and
    # would drop the cached Variables that carry SAS metadata.
    merged = xport.Dataset(source, copy=True)
    source_name = getattr(source, 'name', None)
    merged.name = name if name is not None else source_name
    merged.attrs[METADATA_KEY] = {
        column: copy.deepcopy(_load_metadata(source, column))
        for column in source.columns
    }

    changes = _collect_changes(source, views)

    compatible = []
    conflicts = []
    originals = []

    for column in source.columns:
        column_changes = changes.get(column, {})
        if not column_changes:
            continue

        baseline = _load_metadata(source, column)
        branch_snapshots = OrderedDict(
            (branch, record['metadata'])
            for branch, record in column_changes.items()
        )

        originals.extend(
            _build_originals(column, baseline, branch, record)
            for branch, record in column_changes.items()
        )

        value_conflicts, value_updates = _merge_values(column, column_changes)
        conflicts.extend(value_conflicts)

        metadata_conflicts, accepted_metadata, accepted_aspects = _merge_metadata(
            column, baseline, branch_snapshots
        )
        conflicts.extend(metadata_conflicts)

        applied_cells = _apply_value_updates(merged, column, value_updates)
        merged.attrs[METADATA_KEY][column] = accepted_metadata

        compatible.extend(
            _build_compatible(
                column, baseline, column_changes, applied_cells,
                accepted_aspects,
            )
        )

    _sync_variables(merged)
    return MergeResult(
        merged=merged,
        compatible=compatible,
        conflicts=conflicts,
        originals=originals,
    )


def _collect_changes(source, views):
    """Map each column to the branches that changed it."""
    changes = OrderedDict()
    for branch, view in enumerate(views):
        for column in view.columns:
            source_variable = source[column]
            view_variable = view[column]
            metadata = _effective_metadata(view, column)
            values = OrderedDict()
            for row in view.index:
                new_value = view_variable.loc[row]
                old_value = source_variable.loc[row]
                if not _same_value(old_value, new_value):
                    values[row] = copy.deepcopy(new_value)
            if not values and _same_metadata(
                metadata, _load_metadata(source, column)
            ):
                continue
            changes.setdefault(column, OrderedDict())[branch] = {
                'values': values,
                'metadata': metadata,
            }
    return changes


def _merge_values(column, column_changes):
    """Reconcile value changes, returning conflicts and accepted values."""
    conflicts = []
    updates = OrderedDict()
    rows = OrderedDict()
    for branch, record in column_changes.items():
        for row, value in record['values'].items():
            rows.setdefault(row, OrderedDict())[branch] = value
    for row, branch_values in rows.items():
        distinct = []
        for value in branch_values.values():
            if not any(_same_value(value, other) for other in distinct):
                distinct.append(value)
        if len(distinct) > 1:
            conflicts.append(
                Conflict(
                    kind='value',
                    column=column,
                    branches=tuple(branch_values),
                    values=dict(branch_values),
                    rows=(row,),
                )
            )
            continue
        winner = next(iter(branch_values.values()))
        updates[row] = copy.deepcopy(winner)
    return conflicts, updates


def _merge_metadata(column, baseline, branch_snapshots):
    """Reconcile variable metadata, returning conflicts and accepted values."""
    conflicts = []
    accepted = OrderedDict((aspect, copy.deepcopy(value))
                           for aspect, value in baseline.items())
    accepted_aspects = set()
    for aspect in VARIABLE_METADATA:
        distinct = OrderedDict()
        for branch, snapshot in branch_snapshots.items():
            value = snapshot.get(aspect)
            if _same_metadata_value(value, baseline.get(aspect)):
                continue
            if not any(_same_metadata_value(value, other)
                       for other in distinct.values()):
                distinct[branch] = value
        if len(distinct) > 1:
            conflicts.append(
                Conflict(
                    kind='metadata',
                    column=column,
                    branches=tuple(distinct),
                    values=dict(distinct),
                    aspect=aspect,
                )
            )
            continue
        if distinct:
            accepted[aspect] = copy.deepcopy(next(iter(distinct.values())))
            accepted_aspects.add(aspect)
    return conflicts, accepted, accepted_aspects


def _apply_value_updates(dataset, column, updates):
    """Write accepted values into the merged dataset, preserving NaN tags."""
    if not updates:
        return set()
    if any(isinstance(value, xport.SpecialMissingValue)
           for value in updates.values()):
        dataset[column] = dataset[column].astype(object)
    applied = set()
    for row, value in updates.items():
        dataset.loc[row, column] = value
        applied.add((row, column))
    return applied


def _build_originals(column, baseline, branch, record):
    """Preserve a branch's raw contribution, including conflicting ones."""
    cells = tuple((row, column) for row in record['values'])
    metadata = OrderedDict(
        (aspect, copy.deepcopy(value))
        for aspect, value in record['metadata'].items()
        if not _same_metadata_value(value, baseline.get(aspect))
    )
    return Contribution(
        branch=branch,
        column=column,
        cells=cells,
        values=OrderedDict(
            ((row, column), copy.deepcopy(value))
            for row, value in record['values'].items()
        ),
        metadata=metadata,
    )


def _build_compatible(column, baseline, column_changes, applied_cells,
                      accepted_aspects):
    """Build per-branch contributions containing only applied pieces."""
    compatible = []
    for branch, record in column_changes.items():
        values = OrderedDict(
            ((row, column), value)
            for row, value in record['values'].items()
            if (row, column) in applied_cells
        )
        metadata = OrderedDict(
            (aspect, copy.deepcopy(value))
            for aspect, value in record['metadata'].items()
            if aspect in accepted_aspects
            and not _same_metadata_value(value, baseline.get(aspect))
        )
        if not values and not metadata:
            continue
        compatible.append(
            Contribution(
                branch=branch,
                column=column,
                cells=tuple(values),
                values=values,
                metadata=metadata,
            )
        )
    return compatible


def _metadata_store(dataset):
    """Return the durable metadata store, creating an empty one if absent."""
    return dataset.attrs.setdefault(METADATA_KEY, {})


def set_metadata(dataset, column, aspect, value):
    """
    Update one variable metadata aspect in the durable store.

    Use this on a view (or merged dataset) to change metadata such that
    it survives Pandas' internal cache rebuilds and is seen by a merge.
    The corresponding :class:`~xport.Variable` attribute is updated too.
    """
    if aspect not in VARIABLE_METADATA:
        raise KeyError(f'Unknown metadata aspect {aspect!r}')
    if column not in dataset.columns:
        raise KeyError(f'Unknown variable {column!r}')
    store = _metadata_store(dataset)
    if column not in store:
        store[column] = _snapshot_variable(dataset[column])
    setattr(dataset[column], aspect, value)
    # Re-read every aspect: format/informat setters may infer ``vtype``.
    store[column] = _snapshot_variable(dataset[column])


def sync_metadata(dataset):
    """
    Re-apply durable SAS metadata to a frame's cached Variables.

    Call after operations that rebuild Pandas' Variable objects, such as
    writes or slices, to expose label, format, and similar metadata via
    ``dataset[column]`` again.  ``merge_views`` already calls this on its
    result.
    """
    _sync_variables(dataset)


def _load_metadata(dataset, column):
    """Read a column's metadata from the durable store."""
    store = dataset.attrs.get(METADATA_KEY)
    if store is not None and column in store:
        return OrderedDict(
            (aspect, copy.deepcopy(value))
            for aspect, value in store[column].items()
        )
    return _snapshot_variable(dataset[column])


def _effective_metadata(dataset, column):
    """
    Read a view column's current metadata.

    A live, non-null Variable attribute wins, so edits made through
    ``view[column].label = ...`` are honored.  When Pandas has rebuilt
    the cached Variable and cleared an attribute, the durable store
    supplies the inherited value.
    """
    store = _load_metadata(dataset, column)
    variable = dataset[column]
    effective = OrderedDict()
    for aspect in VARIABLE_METADATA:
        live = getattr(variable, aspect, None)
        effective[aspect] = (
            copy.deepcopy(live) if live is not None
            else copy.deepcopy(store.get(aspect))
        )
    return effective


# Order to write metadata onto a Variable.  The format and informat
# setters infer ``vtype`` as a side effect, so ``vtype`` is applied last
# to keep the explicitly stored type authoritative.
SYNC_ORDER = ('label', 'width', 'format', 'informat', 'vtype')


def _sync_variables(dataset):
    """Push the durable metadata store back onto the cached Variables."""
    store = dataset.attrs.get(METADATA_KEY)
    if not store:
        return
    for column, snapshot in store.items():
        if column not in dataset.columns:
            continue
        variable = dataset[column]
        for aspect in SYNC_ORDER:
            if aspect in snapshot:
                setattr(variable, aspect, copy.deepcopy(snapshot[aspect]))


def _snapshot_variable(variable):
    """Deep-copy a variable's metadata so branches cannot share objects."""
    return OrderedDict(
        (aspect, copy.deepcopy(getattr(variable, aspect, None)))
        for aspect in VARIABLE_METADATA
    )


def _tagged_missing(variable):
    """Return SAS special missing values by position, ``None`` elsewhere."""
    return [
        value if isinstance(value, xport.SpecialMissingValue) else None
        for value in variable
    ]


def _restore_tagged_missing(dataset, column, tagged):
    """Force special missing values to survive dtype coercion."""
    if not any(value is not None for value in tagged):
        return
    variable = dataset[column].astype(object)
    for position, value in enumerate(tagged):
        if value is not None:
            variable.iat[position] = value
    dataset[column] = variable


def _same_value(left, right):
    """Compare cell values, treating NaN tags and plain NaN distinctly."""
    left_tagged = isinstance(left, xport.SpecialMissingValue)
    right_tagged = isinstance(right, xport.SpecialMissingValue)
    if left_tagged or right_tagged:
        return left is right
    if _is_missing(left) and _is_missing(right):
        return True
    if _is_missing(left) or _is_missing(right):
        return False
    return bool(left == right)


def _is_missing(value):
    """True for plain missing values, excluding SAS special missing values."""
    if value is None or value is pd.NA:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _same_metadata(left, right):
    """Compare two metadata snapshots."""
    if tuple(left) != tuple(right):
        return False
    return all(
        _same_metadata_value(left[aspect], right[aspect])
        for aspect in left
    )


def _same_metadata_value(left, right):
    """Compare one metadata aspect, tolerating plain missing values."""
    if _is_missing(left) or _is_missing(right):
        return _is_missing(left) and _is_missing(right)
    return bool(left == right)


def _same_mapping(left, right):
    """Compare mappings whose values may include NaN-like objects."""
    if set(left) != set(right):
        return False
    return all(_same_value(left[key], right[key]) for key in left)
