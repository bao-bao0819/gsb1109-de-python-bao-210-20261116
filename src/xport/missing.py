"""
Work with SAS special missing values.

SAS encodes 28 kinds of missing numerics: the plain missing value ``.``
and the special missing values ``.A`` through ``.Z`` and ``._``.  Each
category carries a distinct meaning -- for example, ``.A`` might mean
"not applicable" while a plain ``.`` means "not collected".  Collapsing
them all to ``NaN`` destroys that information.

This module keeps the categories intact end-to-end:

* ``xport.v56.loads`` preserves each special missing value as an
  ``xport.NaN`` member rather than collapsing it to ``float('nan')``.
* ``representation`` and ``from_representation`` translate between
  values and their original SAS text (``.A``, ``.``, ...).
* ``summarize`` computes statistics using the caller's exclusion rules.
* ``keyify`` gives every category a distinct key, so ``groupby``,
  ``join``, and ``reshard`` never lump a missing category together with
  a real value such as ``0``.
* ``xport.v56.dumps`` writes each category back to its original SAS
  representation.
"""

# Standard Library
import logging
import math
import re

# Community Packages
import pandas as pd

# Xport Modules
import xport

__all__ = [
    'is_missing',
    'is_special_missing',
    'tag',
    'representation',
    'from_representation',
    'keyify',
    'keyify_series',
    'summarize',
    'groupby',
    'join',
    'reshard',
]

LOG = logging.getLogger(__name__)

# Key classes.  Sorting real values before missing categories is
# arbitrary; what matters is that the classes never share a key.
_VALUE = 0
_MISSING = 1


def is_special_missing(value):
    """
    Check whether a value is a SAS special missing value (``.A``-``.Z``, ``._``).

        >>> is_special_missing(xport.NaN.A)
        True
        >>> is_special_missing(float('nan'))
        False
    """
    return isinstance(value, xport.SpecialMissingValue)


def is_missing(value):
    """
    Check whether a value is any kind of missing value.

        >>> is_missing(xport.NaN.A)
        True
        >>> is_missing(float('nan'))
        True
        >>> is_missing(None)
        True
        >>> is_missing(0.0)
        False
    """
    if is_special_missing(value):
        return True
    if value is None or value is pd.NA:
        return True
    return isinstance(value, float) and math.isnan(value)


def tag(value):
    """
    Get the missing-value tag: ``'A'``-``'Z'`` or ``'_'`` for special
    missing values, ``''`` for the plain missing value, and ``None``
    for values that aren't missing.

        >>> tag(xport.NaN.A)
        'A'
        >>> tag(float('nan'))
        ''
        >>> tag(0.0) is None
        True
    """
    if is_special_missing(value):
        return value.name
    if is_missing(value):
        return ''
    return None


def representation(value):
    """
    Get the original SAS text representation of a missing value, such
    as ``'.A'`` or ``'.'``, and ``None`` for values that aren't missing.

        >>> representation(xport.NaN.A)
        '.A'
        >>> representation(float('nan'))
        '.'
        >>> representation(0.0) is None
        True
    """
    t = tag(value)
    if t is None:
        return None
    return f'.{t}'


def from_representation(text):
    """
    Parse a SAS missing-value representation such as ``'.A'`` or ``'.'``.

    The plain missing value ``'.'`` becomes ``float('nan')`` and each
    special missing value becomes an ``xport.NaN`` member.

        >>> from_representation('.A') is xport.NaN.A
        True
        >>> from_representation('._') is xport.NaN._
        True
        >>> math.isnan(from_representation('.'))
        True
    """
    mo = re.fullmatch(r'\.(?P<tag>[A-Za-z_]?)', str(text).strip())
    if mo is None:
        raise ValueError(f'Invalid missing-value representation {text!r}')
    t = mo['tag']
    if not t:
        return float('nan')
    return xport.NaN[t.upper()]


def keyify(value):
    """
    Map a value to a hashable, sortable key.

    Every missing category gets its own key, distinct from any real
    value, so grouping, joining, and resharding never let a missing
    state share a key with a real zero -- or with any other category.

        >>> keyify(0.0)
        (0, 0.0)
        >>> keyify(xport.NaN.A)
        (1, '.A')
        >>> keyify(float('nan'))
        (1, '.')
        >>> keyify(0.0) == keyify(xport.NaN.A)
        False
    """
    if is_missing(value):
        return (_MISSING, representation(value))
    return (_VALUE, value)


def keyify_series(series):
    """
    Apply ``keyify`` to each element of a ``pandas.Series``.

        >>> s = xport.Variable([0.0, xport.NaN.A], dtype='object')
        >>> list(keyify_series(s))
        [(0, 0.0), (1, '.A')]
    """
    keys = [keyify(v) for v in series]
    return pd.Series(keys, index=series.index, name=series.name, dtype='object')


def _exclude_predicate(exclude):
    """
    Build a predicate from the caller's exclusion rules.

    The ``exclude`` specification may be:

    * ``None`` or ``'none'`` -- exclude nothing
    * ``'all'`` or ``'missing'`` -- exclude every missing value
    * ``'special'`` -- exclude only special missing values
    * a tag or representation (``'A'``, ``'.A'``, ``'.'``) or an
      iterable of them
    * a callable that returns true for values to exclude
    """
    if exclude is None:
        return lambda v: False
    if callable(exclude):
        return exclude
    if isinstance(exclude, str):
        if exclude in ('all', 'missing'):
            return is_missing
        if exclude == 'special':
            return is_special_missing
        if exclude == 'none':
            return lambda v: False
        exclude = [exclude]
    reps = set()
    for token in exclude:
        if is_special_missing(token):
            reps.add(representation(token))
            continue
        text = str(token).strip()
        if not text.startswith('.'):
            text = '.' + text
        reps.add(representation(from_representation(text)))
    return lambda v: is_missing(v) and representation(v) in reps


def summarize(data, exclude='all'):
    """
    Compute summary statistics using the caller's exclusion rules.

    Values matching ``exclude`` (see ``_exclude_predicate`` for the
    specification) are removed before statistics are computed.  Missing
    values that survive are tallied by category but never enter the
    numeric statistics.

    For a ``pandas.Series``, the result is a dictionary.  For a
    ``pandas.DataFrame`` or ``xport.Dataset``, the result is a
    ``pandas.DataFrame`` with one row per variable.

        >>> s = xport.Variable([0.0, 1.5, xport.NaN.A, float('nan')], dtype='object')
        >>> summarize(s, exclude='all')  # doctest: +ELLIPSIS
        {'count': 2, 'excluded': 2, 'missing': {}, 'n': 2, 'sum': 1.5, ...}
        >>> summarize(s, exclude={'A'})['missing']
        {'.': 1}
        >>> summarize(s, exclude=None)['excluded']
        0
    """
    if isinstance(data, pd.DataFrame):
        rows = {name: summarize(data[name], exclude=exclude) for name in data.columns}
        return pd.DataFrame.from_dict(rows, orient='index')
    predicate = _exclude_predicate(exclude)
    kept = []
    excluded = 0
    for value in data:
        if predicate(value):
            excluded += 1
        else:
            kept.append(value)
    missing_counts = {}
    numbers = []
    for value in kept:
        if is_missing(value):
            rep = representation(value)
            missing_counts[rep] = missing_counts.get(rep, 0) + 1
        elif isinstance(value, (bool, int, float)):
            numbers.append(float(value))
    result = {
        'count': len(kept),
        'excluded': excluded,
        'missing': missing_counts,
        'n': len(numbers),
        'sum': None,
        'mean': None,
        'std': None,
        'min': None,
        'max': None,
    }
    if numbers:
        series = pd.Series(numbers, dtype='float')
        result.update({
            'sum': float(series.sum()),
            'mean': float(series.mean()),
            'std': float(series.std()),
            'min': float(series.min()),
            'max': float(series.max()),
        })
    return result


def groupby(dataset, by, **kwargs):
    """
    Group a dataset, keeping each missing category in its own group.

    Group keys are ``keyify`` tuples, so a missing category never
    shares a group -- or a key -- with a real value such as ``0``.

        >>> ds = xport.Dataset({
        ...     'x': xport.Variable([0.0, xport.NaN.A, 0.0], dtype='object'),
        ...     'y': [1.0, 2.0, 3.0],
        ... })
        >>> {key: len(group) for key, group in groupby(ds, 'x')}
        {(0, 0.0): 2, (1, '.A'): 1}
    """
    if isinstance(by, str):
        by = [by]
    keys = [keyify_series(dataset[name]) for name in by]
    if len(keys) == 1:
        return dataset.groupby(keys[0], **kwargs)
    return dataset.groupby(keys, **kwargs)


def join(left, right, on, how='inner', **kwargs):
    """
    Merge two datasets on key columns, matching missings by category.

    Keys are compared through ``keyify``, so ``.A`` matches ``.A`` and
    ``0`` matches ``0``, but a missing category never matches a real
    value -- or a different missing category.

        >>> left = xport.Dataset({
        ...     'k': xport.Variable([0.0, xport.NaN.A], dtype='object'),
        ...     'v': ['a', 'b'],
        ... })
        >>> right = xport.Dataset({
        ...     'k': xport.Variable([xport.NaN.A, 0.0], dtype='object'),
        ...     'w': ['x', 'y'],
        ... })
        >>> merged = join(left, right, on='k')
        >>> list(merged['w'])
        ['y', 'x']
    """
    if isinstance(on, str):
        on = [on]
    key_names = [f'__missing_key_{i}__' for i in range(len(on))]
    left = left.assign(**dict(zip(key_names, (keyify_series(left[c]) for c in on))))
    right = right.assign(**dict(zip(key_names, (keyify_series(right[c]) for c in on))))
    merged = pd.merge(left, right, left_on=key_names, right_on=key_names, how=how, **kwargs)
    return merged.drop(columns=key_names)


def reshard(dataset, by, **kwargs):
    """
    Split a dataset into shards keyed by distinct values.

    Shard keys are ``keyify`` tuples, so repartitioning never routes a
    missing category and a real zero to the same shard.

        >>> ds = xport.Dataset({
        ...     'x': xport.Variable([0.0, xport.NaN.A, float('nan')], dtype='object'),
        ... })
        >>> sorted(reshard(ds, 'x'))
        [(0, 0.0), (1, '.'), (1, '.A')]
    """
    return {key: shard for key, shard in groupby(dataset, by, **kwargs)}
