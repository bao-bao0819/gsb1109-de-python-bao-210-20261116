"""
Reconcile a SAS delivery that mixes row weights, stratum codes, and control
totals.

A delivery is a table of observations.  Each observation carries:

* a *source occurrence identity*: two rows are repeated imports of one
  sampling occurrence only when they share this identity.  Rows whose
  measured values happen to be equal but whose identities differ are distinct
  samples and are never collapsed into one another;
* a *stratum* code;
* a *row weight*, and optionally a separate already-weighted contribution.

Control totals are supplied per stratum and optionally globally, each with a
rounding tolerance.  A source occurrence contributes to the totals at most
once.  The reconciliation therefore selects the largest set of occurrences
that can be trusted while every stratum total and the global total fall
inside their tolerance windows.

The result separates three kinds of rows:

* settled rows of occurrences that every best selection keeps;
* certain duplicates and exclusions that every best selection removes;
* disputed rows of occurrences that some best selection keeps and another
  excludes.

Several selections can be equally good.  No row order, file order, or other
incidental ordering is used to choose between them: all equally good
selections are reported, and every disputed row appears in the review set.
An occurrence that is excluded by some but not all best selections stays
disputed along with all of its delivered copies.

The core types are small dataclasses, and the solver accepts an iterable of
:class:`Observation` objects or of mappings with the documented keys.
"""

# Standard Library
import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

__all__ = [
    'Observation',
    'Constraint',
    'ConstraintResult',
    'Reconciliation',
    'Tolerance',
    'collapse_duplicates',
    'check_constraints',
    'observations_from_dataset',
    'reconcile',
    'find_review_set',
]

# Relative guard for inclusive floating-point comparisons.  This is not a
# data tolerance: the caller supplies that separately.
_REL_EPSILON = 1e-9


def _within(value, low, high):
    """
    Inclusive comparison with a relative floating-point guard.
    """
    scale = max(1.0, abs(value))
    if math.isfinite(low):
        scale = max(scale, abs(low))
    if math.isfinite(high):
        scale = max(scale, abs(high))
    epsilon = _REL_EPSILON * scale
    return low - epsilon <= value <= high + epsilon


class Tolerance(float):
    """
    A symmetric or asymmetric rounding tolerance.

    ``Tolerance(0.5)`` accepts ``target - 0.5 <= total <= target + 0.5``.
    ``Tolerance(lower=0.4, upper=0.6)`` allows a different margin on each
    side.  A margin of ``None`` means that side is unbounded.
    """

    def __new__(cls, lower=0.0, upper=None):
        if upper is None:
            upper = lower
        if lower is not None and lower < 0:
            raise ValueError(f'Tolerance lower bound {lower!r} is negative')
        if upper is not None and upper < 0:
            raise ValueError(f'Tolerance upper bound {upper!r} is negative')
        self = super().__new__(cls, 0.0 if lower is None else lower)
        self.lower = lower
        self.upper = upper
        return self

    def __repr__(self):
        return f'{type(self).__name__}(lower={self.lower!r}, upper={self.upper!r})'

    def window(self, target):
        """
        The closed interval ``(low, high)`` accepted around ``target``.
        """
        low = -math.inf if self.lower is None else target - self.lower
        high = math.inf if self.upper is None else target + self.upper
        return low, high

    def contains(self, value, target):
        """
        Whether ``value`` is within tolerance of ``target``.
        """
        low, high = self.window(target)
        return _within(value, low, high)


@dataclass(frozen=True)
class Observation:
    """
    One delivered row.

    Parameters
    ----------
    identity:
        Source occurrence identity.  Repeated imports of one occurrence share
        this value.  It is the only thing that makes two rows copies of one
        another.
    stratum:
        Stratum code, or ``None`` for a row outside every stratum total.
    weight:
        Row weight.  Must be finite.  Negative weights are rejected by
        :func:`reconcile` unless explicitly allowed.
    value:
        Optional already-weighted contribution.  When ``None``, ``weight``
        is the contribution.
    row:
        Optional source label used only in reports.  It never influences the
        result.
    """

    identity: Any
    stratum: Any = None
    weight: float = 1.0
    value: Optional[float] = None
    row: Any = None

    def __post_init__(self):
        weight = float(self.weight)
        if not math.isfinite(weight):
            raise ValueError(f'Row weight {self.weight!r} is not finite')
        object.__setattr__(self, 'weight', weight)
        if self.value is not None:
            value = float(self.value)
            if not math.isfinite(value):
                raise ValueError(f'Row value {self.value!r} is not finite')
            object.__setattr__(self, 'value', value)

    @property
    def contribution(self):
        """
        The amount this row adds to its stratum and global totals.
        """
        return self.weight if self.value is None else self.value

    @property
    def key(self):
        """
        A stable, hashable form of the identity.

        ``row`` is excluded: it is only a display label, so differing labels
        cannot split one occurrence into two.
        """
        return _hashable(self.identity)


def _hashable(value):
    """
    Return a stable, hashable representation of a value.

    Unhashable values are converted structurally, so equal identities (for
    example equal lists) still collapse together.
    """
    if isinstance(value, Mapping):
        return tuple(sorted((_hashable(k), _hashable(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_hashable(v) for v in value)
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _coerce(row):
    """
    Turn a mapping or an :class:`Observation` into an observation.
    """
    if isinstance(row, Observation):
        return row
    if isinstance(row, Mapping):
        data = dict(row)
        try:
            identity = data.pop('identity')
        except KeyError:
            raise ValueError('Row mapping must contain an "identity" key') from None
        return Observation(identity=identity, **data)
    raise TypeError(f'Expected an Observation or mapping, got {type(row).__name__}')


def _coerce_rows(rows):
    return [_coerce(row) for row in rows]


@dataclass(frozen=True)
class Constraint:
    """
    A control total for one stratum, or for the whole delivery.

    ``stratum=None`` denotes the global constraint.  ``total`` is the control
    total, and ``tolerance`` accepts a non-negative number (symmetric) or a
    :class:`Tolerance`.
    """

    total: float
    stratum: Any = None
    tolerance: Any = 0.0
    name: Optional[str] = None

    def __post_init__(self):
        total = float(self.total)
        if not math.isfinite(total):
            raise ValueError(f'Control total {self.total!r} is not finite')
        object.__setattr__(self, 'total', total)
        object.__setattr__(self, 'tolerance', _as_tolerance(self.tolerance))

    @property
    def label(self):
        """
        Display name for reports.
        """
        if self.name:
            return self.name
        return 'ALL' if self.stratum is None else str(self.stratum)


def _as_tolerance(value):
    if isinstance(value, Tolerance):
        return value
    return Tolerance(value)


@dataclass(frozen=True)
class ConstraintResult:
    """
    The measured total for one constraint and whether it passes.
    """

    constraint: Constraint
    total: float
    pass_: bool
    low: float
    high: float

    def __str__(self):
        status = 'pass' if self.pass_ else 'FAIL'
        return (
            f'{self.constraint.label}: {self.total:.6g} '
            f'in [{self.low:.6g}, {self.high:.6g}] {status}'
        )

    def as_dict(self):
        """
        Plain-data representation.
        """
        return {
            'constraint': self.constraint.label,
            'stratum': self.constraint.stratum,
            'total': self.total,
            'low': self.low,
            'high': self.high,
            'pass': self.pass_,
        }


@dataclass
class Reconciliation:
    """
    The outcome of :func:`reconcile`.

    Attributes
    ----------
    rows:
        The observations, in the order supplied.
    review:
        Row indices needing human review: every copy of an occurrence that
        some best selection excludes and another keeps, plus every copy of an
        occurrence whose copies disagree.  This is the minimal disputed set,
        not every row that happens to be left out of one explanation.
    kept:
        Row indices settled as trustworthy: rows of single-copy occurrences
        trusted by every best selection.  Additional copies of a kept
        occurrence are certain duplicates and appear in ``certain_duplicates``
        rather than here.
    removed:
        Row indices left out by every best selection, including repeated
        imports of excluded occurrences and inconsistent occurrences.
    certain_duplicates:
        Repeated imports of an occurrence that every best selection keeps.
        They contribute nothing to the totals but need no review: the
        identity already establishes that they are copies.
    disputed:
        ``review`` minus ``removed``: the genuinely contested rows.  All of
        them are retained in the report; no row order picks a winner.
    solutions:
        Every best selection, as a tuple of one representative row index per
        kept occurrence, sorted ascending.  More than one means an exact tie.
        When an occurrence has several identical copies, which copy is the
        representative is immaterial; the remaining copies are in ``review``.
    cost:
        Minimum number of delivered rows left out by one best selection, or
        ``None`` if no selection satisfies every constraint.
    constraints:
        Measurements for a representative best selection.
    collapsed:
        Groups of row indices sharing a source occurrence identity.
    ignored:
        Groups excluded from the totals because their copies disagree.
        These rows always need review.
    feasible:
        Whether any selection satisfies every constraint.
    message:
        Explanation when ``feasible`` is false, otherwise an empty string.
    complete:
        False when a safety limit cut the search short.
    """

    rows: list
    review: tuple
    kept: tuple
    removed: tuple
    certain_duplicates: tuple
    disputed: tuple
    solutions: list
    cost: Optional[int]
    constraints: list
    collapsed: list
    ignored: list
    feasible: bool
    message: str = ''
    complete: bool = True

    @property
    def tied(self):
        """
        Whether more than one best selection exists.
        """
        return len(self.solutions) > 1

    @property
    def review_rows(self):
        """
        The observations at the review indices.
        """
        return [self.rows[i] for i in self.review]

    @property
    def review_labels(self):
        """
        The ``row`` labels of the review observations.
        """
        return [getattr(row, 'row', None) for row in self.review_rows]

    def as_dict(self):
        """
        Plain-data summary suitable for JSON serialization.
        """
        return {
            'feasible': self.feasible,
            'complete': self.complete,
            'message': self.message,
            'tied': self.tied,
            'cost': self.cost,
            'review': list(self.review),
            'kept': list(self.kept),
            'removed': list(self.removed),
            'certain_duplicates': list(self.certain_duplicates),
            'disputed': list(self.disputed),
            'solutions': [list(s) for s in self.solutions],
            'constraints': [r.as_dict() for r in self.constraints],
            'collapsed': [list(g) for g in self.collapsed],
            'ignored': [list(g) for g in self.ignored],
        }


def _stratum_keys(rows):
    """
    Return ``(strata, index)`` with distinct non-``None`` stratum codes in
    first-seen order and a mapping from hashable code to position.
    """
    strata = []
    index = {}
    for row in rows:
        if row.stratum is None:
            continue
        key = _hashable(row.stratum)
        if key not in index:
            index[key] = len(strata)
            strata.append(row.stratum)
    return strata, index


def _prepare_constraints(constraints, index):
    """
    Validate constraints and return them in a stable order.

    A constraint naming a stratum absent from the data is an error: silently
    treating a typo as a zero total would hide it.
    """
    prepared = []
    seen = set()
    for constraint in constraints:
        if not isinstance(constraint, Constraint):
            constraint = Constraint(**constraint)
        key = None if constraint.stratum is None else _hashable(constraint.stratum)
        if key is not None and key not in index:
            raise ValueError(f'Constraint stratum {constraint.stratum!r} not in any row')
        if key in seen:
            label = 'global' if key is None else repr(constraint.stratum)
            raise ValueError(f'Duplicate {label} constraint')
        seen.add(key)
        prepared.append(constraint)
    return prepared


def collapse_duplicates(rows):
    """
    Group rows by source occurrence identity.

    Returns a list of groups, each a list of row indices in input order.
    Rows in one group are repeated imports of a single occurrence.  Rows with
    different identities are never grouped, even when every measured value is
    equal.
    """
    rows = _coerce_rows(rows)
    groups = {}
    order = []
    for i, row in enumerate(rows):
        key = row.key
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(i)
    return [groups[key] for key in order]


def _totals(rows, strata_index):
    """
    Compute per-stratum totals and the global total.
    """
    strata = [0.0] * len(strata_index)
    total = 0.0
    for row in rows:
        total += row.contribution
        if row.stratum is not None:
            strata[strata_index[_hashable(row.stratum)]] += row.contribution
    return strata, total


def check_constraints(rows, constraints):
    """
    Measure every constraint against ``rows`` without removing anything.

    Returns a list of :class:`ConstraintResult`.  Stratum totals use rows
    with that stratum code; the global total uses every row.  Repeated
    imports are still counted here, because this function only reports the
    raw measurement.  Use :func:`reconcile` to choose a consistent set.
    """
    rows = _coerce_rows(rows)
    constraints = [c if isinstance(c, Constraint) else Constraint(**c) for c in constraints]
    _, index = _stratum_keys(rows)
    constraints = _prepare_constraints(constraints, index)
    strata, total = _totals(rows, index)
    results = []
    for constraint in constraints:
        if constraint.stratum is None:
            value = total
        else:
            value = strata[index[_hashable(constraint.stratum)]]
        low, high = constraint.tolerance.window(constraint.total)
        results.append(
            ConstraintResult(
                constraint=constraint,
                total=value,
                pass_=constraint.tolerance.contains(value, constraint.total),
                low=low,
                high=high,
            )
        )
    return results


def _measure(groups_by_index, selection, strata_index, constraints):
    """
    Measure every constraint for a selection of source occurrences.

    Unlike :func:`check_constraints`, this works from a selection rather than
    from rows, so a constraint on a stratum whose occurrences were all
    excluded is measured as zero instead of being rejected as unknown.
    """
    values = [0.0] * len(strata_index)
    total = 0.0
    for index in selection:
        group = groups_by_index[index]
        if group.stratum is not None:
            values[strata_index[group.stratum_key]] += group.contribution
        total += group.contribution
    results = []
    for constraint in constraints:
        if constraint.stratum is None:
            value = total
        else:
            value = values[strata_index[_hashable(constraint.stratum)]]
        low, high = constraint.tolerance.window(constraint.total)
        results.append(
            ConstraintResult(
                constraint=constraint,
                total=value,
                pass_=constraint.tolerance.contains(value, constraint.total),
                low=low,
                high=high,
            )
        )
    return results


def observations_from_dataset(
    dataset,
    identity='IDENTITY',
    stratum=None,
    weight='WEIGHT',
    value=None,
    row=None,
):
    """
    Build observations from an ``xport.Dataset`` or Pandas ``DataFrame``.

    ``identity`` is the column with the source occurrence identity.
    ``stratum`` names the stratum column (``None`` means every row is
    unstratified), ``weight`` the row-weight column, and ``value`` an optional
    already-weighted contribution column.  ``row`` names an optional label
    column; when omitted, the DataFrame index is used.
    """
    columns = set(dataset.columns)
    for label, name in [('identity', identity), ('weight', weight)]:
        if name not in columns:
            raise ValueError(f'Dataset has no {label} column {name!r}')
    if stratum is not None and stratum not in columns:
        raise ValueError(f'Dataset has no stratum column {stratum!r}')
    if value is not None and value not in columns:
        raise ValueError(f'Dataset has no value column {value!r}')
    if row is not None and row not in columns:
        raise ValueError(f'Dataset has no row column {row!r}')

    observations = []
    for position, (_, record) in enumerate(dataset.iterrows()):
        observations.append(
            Observation(
                identity=record[identity],
                stratum=record[stratum] if stratum is not None else None,
                weight=float(record[weight]),
                value=float(record[value]) if value is not None else None,
                row=record[row] if row is not None else dataset.index[position],
            )
        )
    return observations


@dataclass
class _Group:
    """
    One source occurrence and the delivered copies of it.
    """

    index: int
    rows: tuple
    contribution: float
    stratum: Any
    stratum_key: Any
    consistent: bool


def _build_groups(rows, strata_index):
    """
    Group rows by identity and validate that copies agree.

    Copies of one occurrence must share stratum and contribution.  Copies
    that disagree are marked inconsistent; the caller decides whether that is
    an error or a forced review.
    """
    groups = []
    for index, row_indices in enumerate(collapse_duplicates(rows)):
        members = [rows[i] for i in row_indices]
        first = members[0]
        consistent = all(
            _hashable(m.stratum) == _hashable(first.stratum)
            and m.contribution == first.contribution
            for m in members[1:]
        )
        groups.append(
            _Group(
                index=index,
                rows=tuple(row_indices),
                contribution=first.contribution,
                stratum=first.stratum,
                stratum_key=_hashable(first.stratum),
                consistent=consistent,
            )
        )
    return groups


def _contribution_vector(group, strata_index):
    """
    Per-stratum and global contribution of one kept occurrence.
    """
    vector = [0.0] * len(strata_index)
    if group.stratum is not None:
        vector[strata_index[group.stratum_key]] += group.contribution
    return vector, group.contribution


def reconcile(
    rows,
    constraints=(),
    *,
    allow_negative_weights=False,
    inconsistent='review',
    max_nodes=200000,
    max_solutions=10000,
):
    """
    Find the largest trusted set of source occurrences and the review set.

    Parameters
    ----------
    rows:
        Iterable of :class:`Observation` objects or mappings with an
        ``identity`` key and optional ``stratum``, ``weight``, ``value``, and
        ``row`` keys.
    constraints:
        Iterable of :class:`Constraint` objects or mappings with ``total``
        and optional ``stratum`` and ``tolerance``.
    allow_negative_weights:
        Negative weights are rejected by default because they make control
        totals ill-defined.  When allowed, the search remains exhaustive but
        skips the monotone pruning that assumes dropping a row lowers totals.
    inconsistent:
        What to do when copies of one identity disagree about stratum or
        contribution.  ``'review'`` (default) excludes that occurrence from
        the totals and puts all its rows in the review set; ``'error'``
        raises ``ValueError``.
    max_nodes, max_solutions:
        Safety limits for the exhaustive search.  If either is hit,
        ``complete`` is false and the result is the best found so far.

    Returns
    -------
    Reconciliation

    Notes
    -----
    A source occurrence contributes to the totals at most once, no matter how
    many copies were imported.  Selecting an occurrence therefore means
    keeping exactly one of its copies; the other copies are repeated imports
    and need review (they may be deleted, but which physical copy is the
    original is not a sampling decision).

    The search maximizes the number of trusted occurrences.  Every selection
    achieving that maximum is retained.  ``review`` is the union of the rows
    left out by those selections, so a tie leaves every disputed member in
    the report.  Sorting, input order, and identity ordering affect reporting
    order only, never which rows are selected.
    """
    rows = _coerce_rows(rows)
    if not allow_negative_weights:
        for i, row in enumerate(rows):
            if row.weight < 0:
                raise ValueError(f'Row {i} has negative weight {row.weight!r}')
            if row.contribution < 0:
                raise ValueError(
                    f'Row {i} has negative contribution {row.contribution!r}; '
                    'pass allow_negative_weights=True to accept it'
                )

    strata, strata_index = _stratum_keys(rows)
    constraints = _prepare_constraints(constraints, strata_index)
    groups = _build_groups(rows, strata_index)

    if inconsistent not in ('review', 'error'):
        raise ValueError(f'Unknown inconsistent policy {inconsistent!r}')
    if inconsistent == 'error':
        for group in groups:
            if not group.consistent:
                raise ValueError(
                    f'Occurrence {rows[group.rows[0]].identity!r} has copies '
                    f'that disagree about stratum or contribution'
                )

    usable = [g for g in groups if g.consistent]
    ignored = [g for g in groups if not g.consistent]

    windows = [
        (c, *c.tolerance.window(c.total))
        for c in constraints
    ]

    def passes(values, total):
        for constraint, low, high in windows:
            value = total if constraint.stratum is None else values[
                strata_index[_hashable(constraint.stratum)]]
            if not _within(value, low, high):
                return False
        return True

    # Deterministic search order.  It cannot affect the optimum because all
    # optimal selections are collected.
    order = sorted(usable, key=lambda g: (repr(rows[g.rows[0]].identity), g.index))
    vectors = [_contribution_vector(g, strata_index) for g in order]

    # Suffix sums of positive contributions, used only for pruning.  Unlike a
    # greedy choice, these bounds never discard an optimal branch.
    suffix_strata = [[0.0] * len(strata_index) for _ in range(len(order) + 1)]
    suffix_total = [0.0] * (len(order) + 1)
    for i in range(len(order) - 1, -1, -1):
        vector, delta = vectors[i]
        suffix_strata[i] = [
            suffix_strata[i + 1][j] + max(0.0, vector[j])
            for j in range(len(strata_index))
        ]
        suffix_total[i] = suffix_total[i + 1] + max(0.0, delta)

    nodes = 0
    complete = True
    best_count = -1
    best_selections = []

    def upper_reachable(position, values, total):
        """
        Check the optimistic reachable window from this node.
        """
        for constraint, low, high in windows:
            if constraint.stratum is None:
                value, gain = total, suffix_total[position]
            else:
                j = strata_index[_hashable(constraint.stratum)]
                value, gain = values[j], suffix_strata[position][j]
            if not allow_negative_weights and value + gain < low:
                return False
        return True

    def dfs(position, selected, values, total):
        nonlocal nodes, complete, best_count, best_selections
        if nodes >= max_nodes:
            complete = False
            return
        if len(best_selections) >= max_solutions:
            complete = False
            return
        nodes += 1

        # Even keeping every remaining occurrence cannot beat or tie the
        # current best, so this branch cannot add another best selection.
        if len(selected) + (len(order) - position) < best_count:
            return
        # A constrained total already above its upper bound cannot be lowered
        # by keeping more non-negative contributions.
        if not allow_negative_weights:
            for constraint, low, high in windows:
                value = total if constraint.stratum is None else values[
                    strata_index[_hashable(constraint.stratum)]]
                if value > high and not _within(value, high, high):
                    return
        if not allow_negative_weights and not upper_reachable(position, values, total):
            return

        if position == len(order):
            # Reconciling a delivery never means deleting all of it.  An
            # empty selection is acceptable only when there is nothing to
            # keep; otherwise a zero control total must be explained some
            # other way and the caller needs to review the delivery.
            if order and not selected:
                return
            if passes(values, total):
                count = len(selected)
                if count > best_count:
                    best_count = count
                    best_selections = [tuple(selected)]
                elif count == best_count:
                    best_selections.append(tuple(selected))
            return

        group = order[position]
        vector, delta = vectors[position]

        # Include first: a greedy high count tightens the bound early.
        new_values = [a + b for a, b in zip(values, vector)]
        new_total = total + delta
        # With non-negative contributions a total above its upper bound can
        # only get worse, so this branch is dead.
        over = False
        if not allow_negative_weights:
            for constraint, low, high in windows:
                value = new_total if constraint.stratum is None else new_values[
                    strata_index[_hashable(constraint.stratum)]]
                if value > high and not _within(value, high, high):
                    over = True
                    break
        if not over:
            selected.append(group.index)
            dfs(position + 1, selected, new_values, new_total)
            selected.pop()
        if not complete:
            return

        dfs(position + 1, selected, values, total)

    dfs(0, [], [0.0] * len(strata_index), 0.0)

    if best_count < 0:
        message = (
            'No selection of source occurrences satisfies every constraint'
            if complete else
            'Search stopped at a safety limit before finding a feasible selection'
        )
        return Reconciliation(
            rows=rows,
            review=tuple(range(len(rows))),
            kept=(),
            removed=(),
            certain_duplicates=(),
            disputed=tuple(range(len(rows))),
            solutions=[],
            cost=None,
            constraints=check_constraints(rows, constraints),
            collapsed=[list(g.rows) for g in groups],
            ignored=[list(g.rows) for g in ignored],
            feasible=False,
            message=message,
            complete=complete,
        )

    # Group-level coverage of the optimal selections.
    selections = [frozenset(s) for s in best_selections]
    kept_anywhere = set().union(*selections) if selections else set()
    kept_everywhere = set(selections[0]) if selections else set()
    for selection in selections[1:]:
        kept_everywhere &= selection

    review = set()
    removed = set()

    for group in groups:
        members = set(group.rows)
        if not group.consistent:
            # Its contribution is unknowable, so the occurrence cannot be
            # trusted in any selection.  Every copy needs a human decision.
            review |= members
            continue
        if group.index not in kept_anywhere:
            # No best selection trusts this occurrence: excluding it is
            # certain, not a matter of judgment.
            removed |= members
            continue
        if group.index not in kept_everywhere:
            # Some best selection keeps it and some excludes it.  All of its
            # copies are disputed; no row order may choose between them.
            review |= members

    disputed = review - removed
    review = tuple(sorted(review))
    removed = tuple(sorted(removed))
    disputed = tuple(sorted(disputed))

    # Additional copies of occurrences trusted by every best selection.
    certain_duplicates = set()
    for group in groups:
        if group.consistent and group.index in kept_everywhere:
            certain_duplicates.update(group.rows[1:])

    kept = tuple(
        i for i in range(len(rows))
        if i not in set(review) and i not in set(removed)
        and i not in certain_duplicates
    )

    # Canonical representative per selection: one row per trusted occurrence.
    # Copy choice is immaterial to every total and is covered by the review
    # set rather than by enumerating equivalent row orders.
    representatives = []
    for selection in best_selections:
        rows_kept = [
            group.rows[0]
            for group in order
            if group.index in selection
        ]
        representatives.append(tuple(sorted(rows_kept)))
    representatives = sorted(set(representatives))

    # Measure the constraints on a representative optimal selection.  Every
    # best selection produces the same totals within tolerance; the
    # representative is reported only for display.
    groups_by_index = {g.index: g for g in groups}
    representative_selection = frozenset(best_selections[0]) if best_selections else frozenset()
    measured = _measure(groups_by_index, representative_selection, strata_index, constraints)

    # ``cost`` is the minimum number of delivered rows that must be left out
    # of a single best explanation: every kept occurrence contributes exactly
    # one row, so all other copies and all untrusted occurrences are excluded.
    cost = len(rows) - best_count

    return Reconciliation(
        rows=rows,
        review=review,
        kept=kept,
        removed=removed,
        certain_duplicates=tuple(sorted(certain_duplicates)),
        disputed=disputed,
        solutions=representatives,
        cost=cost,
        constraints=measured,
        collapsed=[list(g.rows) for g in groups],
        ignored=[list(g.rows) for g in ignored],
        feasible=True,
        complete=complete,
    )


def find_review_set(rows, constraints=(), **kwargs):
    """
    Convenience wrapper returning just the review row indices.

    See :func:`reconcile` for parameters and semantics.  Use
    ``reconcile(...)`` when labels, tied selections, or constraint
    measurements are needed.
    """
    return reconcile(rows, constraints, **kwargs).review
