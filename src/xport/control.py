"""
Rehearse field-level substitutions for control-total compliance.

Writing a dataset to SAS Transport (XPT) format encodes numbers as IBM
hexadecimal floating-point.  Only a limited range of magnitudes fits that
format: zero, or ``2 ** -260 <= abs(x) < 2 ** 249`` (see
``xport.v56.ieee_to_ibm``).  Values outside that range cannot be written
-- overflow for huge magnitudes, underflow for tiny ones, and infinity
not at all.  SAS "special missing" values (``xport.NaN``) *can* be
written, but they carry no magnitude, so they are reported separately and
never substituted.

When a few rows of the numeric columns that jointly participate in a
control total fall outside the representable range, this module
*rehearses* the allowed field-level substitution plans without modifying
any data:

*   Every problem field is classified as overflow, underflow, or special
    missing.  The categories are planned independently: rounding a huge
    value can never offset the loss of a tiny one.

*   A problem field may be replaced only by a representable value within
    ``row_tolerance`` of the original.  Values are never clamped to an
    endpoint of the representable range; if no representable value lies
    within the tolerance, the field is reported as a conflict.

*   Within each category, the signed errors of the chosen replacements
    must sum to a control error no larger than ``control_tolerance``.
    Arithmetic is exact (``fractions.Fraction``), and fields that do not
    need substitution are never adjusted to make the total fit.

*   All plans satisfying both bounds are equally acceptable.  Fields
    whose replacement varies across those plans are reported as
    "disputed" and need adjudication.

*   If no plan satisfies the bounds, the rehearsal reports the conflict
    together with the closest achievable error as evidence.

For example, two columns participate in one control total, and one row
of each column overflows the IBM-format range:

>>> import xport
>>> import xport.control
>>> data = {
...     'DEBIT': [1e75, 100.0, xport.NaN.A],
...     'CREDIT': [-1e75, -100.0, 0.0],
... }
>>> rehearsal = xport.control.rehearse(
...     data, columns=('DEBIT', 'CREDIT'),
...     row_tolerance=1e74, control_tolerance=1e58,
... )
>>> rehearsal
<Rehearsal feasible=True groups=1 special_missing=1>
>>> group = rehearsal.groups[0]
>>> group
<CategoryRehearsal OVERFLOW problems=2 plans=4>
>>> group.suggested
<Plan OVERFLOW substitutions=2 control_error=0.0 max_row_error=9.537430283346725e+73>

All four plans satisfy both bounds, and the two fields take different
replacements across them, so both are disputed:

>>> [field.column for field in group.disputed]
['DEBIT', 'CREDIT']

The special missing value is representable as-is; it is listed
separately and plays no role in the control total:

>>> rehearsal.special_missing[0].field
Field(column='DEBIT', row=2)

If the per-row tolerance cannot be met, the rehearsal refuses to clamp
the value to the nearest representable endpoint and instead reports the
conflict with the closest achievable error as evidence:

>>> rehearsal = xport.control.rehearse({'A': [1e75]}, ['A'], 1e70, 1e58)
>>> rehearsal.feasible
False
>>> rehearsal.conflicts[0]
<Conflict ROW OVERFLOW fields=1 achieved=9.537430283346725e+73 tolerance=1e+70>

Likewise, if the per-row bounds can be met but the control total cannot,
the best-effort plan is returned as evidence:

>>> rehearsal = xport.control.rehearse({'A': [1e75, 1e75]}, ['A'], 1e74, 1e50)
>>> rehearsal.conflicts[0]
<Conflict CONTROL OVERFLOW fields=2 achieved=1.907486056669345e+74 tolerance=1e+50>
"""

# Standard Library
import enum
import itertools
import logging
import math
import struct
from fractions import Fraction
from typing import NamedTuple, Optional

# Xport Modules
import xport
from xport.v56 import Overflow, Underflow, ieee_to_ibm

__all__ = [
    'rehearse',
    'find_problems',
    'classify',
    'Category',
    'ConflictKind',
    'Field',
    'Problem',
    'Substitution',
    'Plan',
    'Conflict',
    'CategoryRehearsal',
    'Rehearsal',
    'MAX_REPRESENTABLE',
    'MIN_REPRESENTABLE',
]

LOG = logging.getLogger(__name__)

# The largest and smallest nonzero magnitudes that ``ieee_to_ibm``
# accepts.  ``ieee_to_ibm`` raises ``Overflow`` for binary exponents
# greater than 248 and ``Underflow`` for exponents less than -260.
MAX_REPRESENTABLE = math.nextafter(2.0**249, 0.0)
MIN_REPRESENTABLE = 2.0**-260


class Category(enum.Enum):
    """
    The kind of representability problem a field value has.
    """

    OVERFLOW = 'overflow'
    UNDERFLOW = 'underflow'
    SPECIAL_MISSING = 'special_missing'


class ConflictKind(enum.Enum):
    """
    Which bound a ``Conflict`` evidence refers to.
    """

    ROW = 'row'
    CONTROL = 'control'


class Field(NamedTuple):
    """
    Identify one cell of a dataset: a column and a row label.
    """

    column: str
    row: object


class Problem(NamedTuple):
    """
    A field whose value cannot be written to XPT format as-is.
    """

    field: Field
    value: object
    category: Category


class Substitution(NamedTuple):
    """
    Replace one field's value with a representable value.

    ``error`` is the signed, exact difference ``replacement - original``.
    """

    field: Field
    original: float
    replacement: float
    error: Fraction

    def __repr__(self):
        """
        Format for the REPL.
        """
        return (
            f'<{type(self).__name__} {self.field.column}[{self.field.row}]'
            f' {self.original!r} -> {self.replacement!r} error={float(self.error)}>'
        )


class Plan(NamedTuple):
    """
    One allowed substitution plan for a category of problems.

    A plan is allowed when every substitution's error is within the
    per-row tolerance (by construction) and the summed ``control_error``
    is within the control tolerance.
    """

    category: Category
    substitutions: tuple
    control_error: Fraction
    max_row_error: Fraction

    def __repr__(self):
        """
        Format for the REPL.
        """
        return (
            f'<{type(self).__name__} {self.category.name}'
            f' substitutions={len(self.substitutions)}'
            f' control_error={float(self.control_error)}'
            f' max_row_error={float(self.max_row_error)}>'
        )


class Conflict(NamedTuple):
    """
    Evidence that no legal substitution plan exists for a category.

    For a ``ROW`` conflict, ``fields`` have no representable replacement
    within ``tolerance``; ``achieved`` is the closest per-row error any
    representable value can offer (``None`` for infinite values).  For a
    ``CONTROL`` conflict, every combination of allowed replacements
    misses the control ``tolerance``; ``achieved`` is the closest
    attainable control error and ``best_effort`` the plan attaining it.
    """

    kind: ConflictKind
    category: Category
    fields: tuple
    tolerance: Fraction
    achieved: Optional[Fraction]
    best_effort: Optional[Plan]
    detail: str

    def __repr__(self):
        """
        Format for the REPL.
        """
        achieved = None if self.achieved is None else float(self.achieved)
        return (
            f'<{type(self).__name__} {self.kind.name} {self.category.name}'
            f' fields={len(self.fields)} achieved={achieved}'
            f' tolerance={float(self.tolerance)}>'
        )


class CategoryRehearsal(NamedTuple):
    """
    Rehearsal outcome for one ``Category`` of problems.
    """

    category: Category
    problems: tuple
    plans: tuple
    conflict: Optional[Conflict]

    def __repr__(self):
        """
        Format for the REPL.
        """
        if self.feasible:
            status = f'plans={len(self.plans)}'
        else:
            status = f'conflict={self.conflict.kind.name}'
        return (
            f'<{type(self).__name__} {self.category.name}'
            f' problems={len(self.problems)} {status}>'
        )

    @property
    def feasible(self):
        """
        Whether at least one allowed plan exists for this category.
        """
        return self.conflict is None

    @property
    def suggested(self):
        """
        The plan with the smallest per-row error, then control error.
        """
        return self.plans[0] if self.plans else None

    @property
    def disputed(self):
        """
        Fields whose replacement varies across the allowed plans.
        """
        if len(self.plans) < 2:
            return ()
        disputed = []
        for i, problem in enumerate(self.problems):
            replacements = {plan.substitutions[i].replacement for plan in self.plans}
            if len(replacements) > 1:
                disputed.append(problem.field)
        return tuple(disputed)


class Rehearsal(NamedTuple):
    """
    Dry-run of field-level substitutions for a control total.

    Categories are independent, so any choice of one plan per group
    composes into a legal overall substitution.  ``special_missing``
    lists the fields holding ``xport.NaN`` special missing values; they
    are representable as-is and excluded from the control total.
    """

    groups: tuple
    special_missing: tuple
    row_tolerance: Fraction
    control_tolerance: Fraction

    def __repr__(self):
        """
        Format for the REPL.
        """
        return (
            f'<{type(self).__name__} feasible={self.feasible}'
            f' groups={len(self.groups)} special_missing={len(self.special_missing)}>'
        )

    @property
    def feasible(self):
        """
        Whether every category has at least one allowed plan.
        """
        return all(group.feasible for group in self.groups)

    @property
    def problems(self):
        """
        All problem fields, including special missing values.
        """
        problems = (problem for group in self.groups for problem in group.problems)
        return tuple(itertools.chain(problems, self.special_missing))

    @property
    def conflicts(self):
        """
        Conflict evidence for each infeasible category.
        """
        return tuple(group.conflict for group in self.groups if group.conflict is not None)

    @property
    def disputed(self):
        """
        Fields whose replacement varies across any category's plans.
        """
        return tuple(field for group in self.groups for field in group.disputed)


def _neighbor(x, toward):
    """
    Get the closest double to ``x`` in the direction of ``toward``.

    Equivalent to ``math.nextafter`` on Python 3.9+, implemented here
    with ``struct`` so the module works on older versions.
    """
    if x == toward:
        return x
    if x == 0.0:
        return math.copysign(2.0**-1074, toward)
    bits, = struct.unpack('>q', struct.pack('>d', x))
    bits += 1 if (x > 0.0) == (toward > x) else -1
    return struct.unpack('>d', struct.pack('>q', bits))[0]


def classify(value):
    """
    Classify a value's representability problem, if any.

    Returns ``None`` for values that ``xport.v56.ieee_to_ibm`` accepts,
    including ordinary missing values (``None``, ``float('nan')``).
    """
    if isinstance(value, xport.NaN):
        return Category.SPECIAL_MISSING
    if value is None:
        return None
    try:
        if math.isnan(value):
            return None
    except TypeError:
        return None  # pd.NA and friends
    except OverflowError:
        return Category.OVERFLOW  # Integer too large to convert to float.
    try:
        ieee_to_ibm(value)
    except Underflow:
        return Category.UNDERFLOW
    except (Overflow, NotImplementedError, OverflowError):
        return Category.OVERFLOW
    return None


def find_problems(dataset, columns):
    """
    Find field values that cannot be written to XPT format as-is.

    ``dataset`` may be a Pandas data frame or a mapping of column names
    to sequences; ``columns`` names the numeric columns that participate
    in the control total.  Rows are identified by the data frame index,
    or by position for plain sequences.
    """
    problems = []
    for column in columns:
        values = dataset[column]
        items = values.items() if hasattr(values, 'items') else enumerate(values)
        for row, value in items:
            category = classify(value)
            if category is None:
                continue
            if category is not Category.SPECIAL_MISSING:
                try:
                    value = float(value)
                except OverflowError:
                    value = math.copysign(math.inf, value)
            problems.append(Problem(Field(column, row), value, category))
    return tuple(problems)


def _candidates(value, tolerance, limit):
    """
    Get representable replacements for ``value``, nearest first.

    Only values that ``ieee_to_ibm`` accepts and that lie within
    ``tolerance`` of ``value`` are returned -- at most ``limit`` of
    them.  An empty tuple means no legal replacement exists; the value
    is never clamped to an endpoint of the representable range.
    """
    if math.isnan(value) or math.isinf(value):
        return ()
    target = Fraction(value)
    found = []
    if value > MAX_REPRESENTABLE:
        replacement = MAX_REPRESENTABLE
        while len(found) < limit and target - Fraction(replacement) <= tolerance:
            found.append(replacement)
            replacement = _neighbor(replacement, 0.0)
    elif value < -MAX_REPRESENTABLE:
        replacement = -MAX_REPRESENTABLE
        while len(found) < limit and Fraction(replacement) - target <= tolerance:
            found.append(replacement)
            replacement = _neighbor(replacement, 0.0)
    elif 0.0 < value < MIN_REPRESENTABLE:
        if target <= tolerance:
            found.append(0.0)
        replacement = MIN_REPRESENTABLE
        while len(found) < limit and Fraction(replacement) - target <= tolerance:
            found.append(replacement)
            replacement = _neighbor(replacement, math.inf)
    elif -MIN_REPRESENTABLE < value < 0.0:
        if -target <= tolerance:
            found.append(0.0)
        replacement = -MIN_REPRESENTABLE
        while len(found) < limit and target - Fraction(replacement) <= tolerance:
            found.append(replacement)
            replacement = _neighbor(replacement, -math.inf)
    found.sort(key=lambda r: (abs(Fraction(r) - target), r))
    return tuple(found[:limit])


def _nearest(value):
    """
    Get the representable value closest to ``value``.
    """
    if math.isinf(value):
        return math.copysign(MAX_REPRESENTABLE, value)
    if value > MAX_REPRESENTABLE:
        return MAX_REPRESENTABLE
    if value < -MAX_REPRESENTABLE:
        return -MAX_REPRESENTABLE
    if 0.0 < value < MIN_REPRESENTABLE:
        return 0.0 if 2.0 * value <= MIN_REPRESENTABLE else MIN_REPRESENTABLE
    if -MIN_REPRESENTABLE < value < 0.0:
        return 0.0 if 2.0 * value >= -MIN_REPRESENTABLE else -MIN_REPRESENTABLE
    return value


def _tolerance(value, name):
    """
    Validate a tolerance and convert it to an exact ``Fraction``.
    """
    try:
        tolerance = Fraction(value)
    except TypeError:
        raise TypeError(f'{name} must be a non-negative number, got {value!r}')
    except (ValueError, OverflowError):
        raise ValueError(f'{name} must be a non-negative number, got {value!r}')
    if tolerance < 0:
        raise ValueError(f'{name} must be non-negative, got {value!r}')
    return tolerance


def _row_conflict(category, problems, tolerance):
    """
    Build conflict evidence for fields with no legal replacement.
    """
    fields = tuple(problem.field for problem in problems)
    try:
        achieved = max(
            abs(Fraction(_nearest(problem.value)) - Fraction(problem.value))
            for problem in problems
        )
    except OverflowError:  # Fraction(infinity) is not defined.
        achieved = None
    detail = '; '.join(
        f'{problem.field.column}[{problem.field.row}]={problem.value!r} has no'
        f' representable replacement within row tolerance {float(tolerance)}'
        for problem in problems
    )
    return Conflict(
        kind=ConflictKind.ROW,
        category=category,
        fields=fields,
        tolerance=tolerance,
        achieved=achieved,
        best_effort=None,
        detail=detail,
    )


def _rehearse_group(
    category,
    problems,
    row_tolerance,
    control_tolerance,
    max_candidates,
    max_enumerations,
):
    """
    Rehearse the allowed substitution plans for one category.
    """
    options = []
    blocked = []
    for problem in problems:
        candidates = _candidates(problem.value, row_tolerance, max_candidates)
        if candidates:
            options.append((problem, candidates))
        else:
            blocked.append(problem)
    if blocked:
        conflict = _row_conflict(category, blocked, row_tolerance)
        return CategoryRehearsal(category, problems, (), conflict)

    combinations = 1
    for _, candidates in options:
        combinations *= len(candidates)
    if combinations > max_enumerations:
        raise ValueError(
            f'Enumerating {combinations} candidate combinations exceeds the limit'
            f' of {max_enumerations}; reduce `max_candidates`'
        )

    feasible = []
    best = None
    for combo in itertools.product(*(candidates for _, candidates in options)):
        substitutions = tuple(
            Substitution(
                field=problem.field,
                original=problem.value,
                replacement=replacement,
                error=Fraction(replacement) - Fraction(problem.value),
            ) for (problem, _), replacement in zip(options, combo)
        )
        control_error = sum((s.error for s in substitutions), Fraction(0))
        plan = Plan(
            category=category,
            substitutions=substitutions,
            control_error=control_error,
            max_row_error=max(abs(s.error) for s in substitutions),
        )
        if best is None or abs(control_error) < abs(best.control_error):
            best = plan
        if abs(control_error) <= control_tolerance:
            feasible.append(plan)
    feasible.sort(key=lambda plan: (plan.max_row_error, abs(plan.control_error)))

    if feasible:
        LOG.debug(f'{category.name}: {len(feasible)} allowed plans')
        return CategoryRehearsal(category, problems, tuple(feasible), None)

    conflict = Conflict(
        kind=ConflictKind.CONTROL,
        category=category,
        fields=tuple(problem.field for problem in problems),
        tolerance=control_tolerance,
        achieved=abs(best.control_error),
        best_effort=best,
        detail=(
            f'closest achievable control error {float(abs(best.control_error))}'
            f' exceeds control tolerance {float(control_tolerance)}'
        ),
    )
    return CategoryRehearsal(category, problems, (), conflict)


def rehearse(
    dataset,
    columns,
    row_tolerance,
    control_tolerance,
    *,
    max_candidates=4,
    max_enumerations=2**20,
):
    """
    Rehearse field-level substitutions for a control total.

    ``dataset`` may be a Pandas data frame or a mapping of column names
    to sequences.  ``columns`` names the numeric columns that jointly
    participate in the control total.  ``row_tolerance`` bounds each
    substitution's error and ``control_tolerance`` bounds the summed
    signed error within each problem category; both must be
    non-negative numbers.

    At most ``max_candidates`` replacement values are considered per
    problem field, and at most ``max_enumerations`` combinations are
    evaluated per category.

    Returns a ``Rehearsal``; no data is modified.
    """
    row_tolerance = _tolerance(row_tolerance, 'row_tolerance')
    control_tolerance = _tolerance(control_tolerance, 'control_tolerance')
    columns = tuple(columns)
    problems = find_problems(dataset, columns)
    LOG.info(f'Found {len(problems)} problem fields in {len(columns)} columns')
    special_missing = tuple(p for p in problems if p.category is Category.SPECIAL_MISSING)
    groups = []
    for category in (Category.OVERFLOW, Category.UNDERFLOW):
        subset = tuple(p for p in problems if p.category is category)
        if subset:
            groups.append(
                _rehearse_group(
                    category,
                    subset,
                    row_tolerance,
                    control_tolerance,
                    max_candidates,
                    max_enumerations,
                )
            )
    return Rehearsal(
        groups=tuple(groups),
        special_missing=special_missing,
        row_tolerance=row_tolerance,
        control_tolerance=control_tolerance,
    )
