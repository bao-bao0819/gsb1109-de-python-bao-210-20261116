"""
Audit whether a displayed value and a stored value can still agree.

SAS variable formats such as ``PERCENT7.1`` round the stored value for
display, so the displayed string ``12.3%`` does not identify a unique
underlying number -- it identifies an interval of possible originals.
This module derives that interval exactly, preserving whether each
boundary is open or closed, then compares it with the interval implied
by the stored value.

    >>> from xport.audit import audit, DisplayInterpretation
    >>> result = audit('12.3%', 0.12304,
    ...                interpretation=DisplayInterpretation.PERCENT)
    >>> result.conclusion.name
    'COMPATIBLE'
    >>> str(result.left)
    '(0.1225, 0.1235)'

All arithmetic uses :class:`decimal.Decimal`, so the boundary openness
is exact and never depends on binary floating-point behavior.  Equal
display strings are never treated as equal originals: ``12.3%`` and
``12.30%`` produce intervals of different widths.
"""

# Standard Library
import enum
import re
import typing
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_EVEN, ROUND_HALF_UP, localcontext

__all__ = [
    'AuditConclusion',
    'AuditResult',
    'DisplayInterpretation',
    'FormatHint',
    'IntervalComparison',
    'ParsedDisplay',
    'RoundingRule',
    'ValueInterval',
    'audit',
    'compare_intervals',
    'compatible_with_display',
    'display_interval',
    'parse_display',
    'parse_format',
    'stored_interval',
]


class RoundingRule(enum.Enum):
    """
    Rounding rule that may have produced the displayed string.

    SAS display formats round to nearest; the tie behavior is not always
    documented in an audit trail, so callers can evaluate every candidate.
    """

    ROUND_HALF_EVEN = ROUND_HALF_EVEN
    ROUND_HALF_UP = ROUND_HALF_UP
    ROUND_DOWN = ROUND_DOWN


class DisplayInterpretation(enum.Enum):
    """
    How the displayed text maps onto the stored numeric scale.
    """

    NUMBER = Decimal(1)
    PERCENT = Decimal('0.01')
    PERMILLE = Decimal('0.001')


class AuditConclusion(enum.Enum):
    """
    Whether the stored value is consistent with the displayed value.
    """

    COMPATIBLE = 'compatible'
    INCOMPATIBLE = 'incompatible'
    INDETERMINATE = 'indeterminate'


class ParsedDisplay(typing.NamedTuple):
    """
    Components parsed from a displayed numeric string.
    """

    coefficient: Decimal
    decimals: int
    interpretation: DisplayInterpretation


class FormatHint(typing.NamedTuple):
    """
    Audit-relevant parts of a SAS format such as ``PERCENT7.1``.

    ``decimals`` is the format's ``d`` component; ``interpretation`` is
    inferred only for formats with a documented scale (``PERCENT``),
    otherwise it is ``None`` and must be confirmed from the variable's
    documented dimensions.
    """

    name: str
    decimals: typing.Optional[int]
    interpretation: typing.Optional[DisplayInterpretation]


class ValueInterval(typing.NamedTuple):
    """
    Closed, open, or half-open interval of possible original values.

    ``lower_inclusive`` and ``upper_inclusive`` record whether each
    boundary itself rounds to the displayed value.
    """

    lower: Decimal
    lower_inclusive: bool
    upper: Decimal
    upper_inclusive: bool
    unit: typing.Optional[str] = None
    decimals: typing.Optional[int] = None

    @property
    def is_singleton(self):
        """
        Whether the interval contains exactly one value.
        """
        return self.lower == self.upper and self.lower_inclusive and self.upper_inclusive

    def contains(self, value):
        """
        Whether ``value`` falls inside the interval, honoring open edges.
        """
        value = _to_decimal(value)
        if self.lower_inclusive:
            above_lower = value >= self.lower
        else:
            above_lower = value > self.lower
        if self.upper_inclusive:
            below_upper = value <= self.upper
        else:
            below_upper = value < self.upper
        return above_lower and below_upper

    def overlaps(self, other):
        """
        Whether the two intervals share at least one point.

        Touching at a single point counts only when both intervals
        include that boundary.
        """
        if self.upper < other.lower or other.upper < self.lower:
            return False
        if self.upper == other.lower:
            return self.upper_inclusive and other.lower_inclusive
        if other.upper == self.lower:
            return other.upper_inclusive and self.lower_inclusive
        return True

    def intersection(self, other):
        """
        Return the overlapping interval, or ``None`` if they are disjoint.
        """
        if not self.overlaps(other):
            return None
        if self.lower >= other.lower:
            lower = self.lower
            lower_inclusive = self.lower_inclusive and (
                other.lower_inclusive if self.lower == other.lower else True
            )
        else:
            lower = other.lower
            lower_inclusive = other.lower_inclusive
        if self.upper <= other.upper:
            upper = self.upper
            upper_inclusive = self.upper_inclusive and (
                other.upper_inclusive if self.upper == other.upper else True
            )
        else:
            upper = other.upper
            upper_inclusive = other.upper_inclusive
        return ValueInterval(
            lower,
            lower_inclusive,
            upper,
            upper_inclusive,
            unit=self.unit or other.unit,
            decimals=self.decimals if self.decimals is not None else other.decimals,
        )

    def __str__(self):
        """
        Mathematical interval notation, such as ``(0.1225, 0.1235) %``.
        """
        left_bracket = '[' if self.lower_inclusive else '('
        right_bracket = ']' if self.upper_inclusive else ')'
        suffix = f' {self.unit}' if self.unit else ''
        return f'{left_bracket}{self.lower}, {self.upper}{right_bracket}{suffix}'


class IntervalComparison(typing.NamedTuple):
    """
    Result of comparing two value intervals.
    """

    overlap: typing.Optional[ValueInterval]
    conclusion: AuditConclusion
    explanation: str


class AuditResult(typing.NamedTuple):
    """
    Full audit verdict for a displayed string against a stored value.
    """

    left: ValueInterval
    right: ValueInterval
    overlap: typing.Optional[ValueInterval]
    conclusion: AuditConclusion
    explanation: str
    candidates: typing.Tuple[typing.Tuple[RoundingRule, IntervalComparison], ...]


_DISPLAY_PATTERN = re.compile(r'^([+-]?)(\d+(?:\.\d+)?|\.\d+)([%‰]?)$')

_SUFFIX_INTERPRETATION = {
    '': DisplayInterpretation.NUMBER,
    '%': DisplayInterpretation.PERCENT,
    '‰': DisplayInterpretation.PERMILLE,
}


def _to_decimal(value):
    """
    Convert a numeric input to an exact, finite ``Decimal``.

    Floats go through their shortest string representation, so the
    recorded token ``0.12304`` stays ``Decimal('0.12304')`` rather than
    expanding to its full binary representation.
    """
    if isinstance(value, bool):
        raise TypeError('Boolean values are not supported')
    if isinstance(value, Decimal):
        decimal_value = value
    elif isinstance(value, (int, float, str)):
        decimal_value = Decimal(str(value))
    else:
        raise TypeError(f'Cannot interpret {value!r} as a decimal number')
    if not decimal_value.is_finite():
        raise ValueError(f'Only finite values are supported, got {value!r}')
    return decimal_value


def parse_display(text):
    """
    Split a displayed string into its coefficient, decimals, and scale.

        >>> parse_display('12.3%')  # doctest: +NORMALIZE_WHITESPACE
        ParsedDisplay(coefficient=Decimal('12.3'), decimals=1,
                      interpretation=<DisplayInterpretation.PERCENT: Decimal('0.01')>)
    """
    if not isinstance(text, str):
        raise TypeError(f'Expected a display string, got {type(text).__name__}')
    cleaned = text.strip().replace(',', '').replace('_', '').replace(' ', '')
    match = _DISPLAY_PATTERN.match(cleaned)
    if match is None:
        raise ValueError(f'Cannot parse displayed value {text!r}')
    sign, body, suffix = match.groups()
    coefficient = Decimal(sign + body)
    decimals = len(body.split('.')[1]) if '.' in body else 0
    interpretation = _SUFFIX_INTERPRETATION[suffix]
    return ParsedDisplay(coefficient, decimals, interpretation)


_FORMAT_PATTERN = re.compile(
    r'^\s*(?P<name>\$?[A-Za-z0-9]*?)(?P<width>\d+)\.(?P<decimals>\d+)?\s*$'
)

_PERCENT_FORMATS = frozenset({'PERCENT', 'PERCENTN'})


def parse_format(spec):
    """
    Extract the audit-relevant hint from a SAS format specification.

        >>> parse_format('PERCENT7.1')  # doctest: +NORMALIZE_WHITESPACE
        FormatHint(name='PERCENT', decimals=1,
                   interpretation=<DisplayInterpretation.PERCENT: Decimal('0.01')>)
        >>> parse_format('8.2')
        FormatHint(name='', decimals=2, interpretation=None)

    Only the decimal places and (for ``PERCENT``) the percent scale are
    inferable; generic numeric formats return ``interpretation=None`` so
    the dimension has to be confirmed rather than guessed.
    """
    if not isinstance(spec, str):
        raise TypeError(f'Expected a format specification, got {type(spec).__name__}')
    match = _FORMAT_PATTERN.match(spec)
    if match is None:
        raise ValueError(f'Cannot parse SAS format {spec!r}')
    name = match.group('name').upper()
    decimals_text = match.group('decimals')
    decimals = int(decimals_text) if decimals_text is not None else None
    interpretation = (
        DisplayInterpretation.PERCENT if name in _PERCENT_FORMATS else None
    )
    return FormatHint(name=name, decimals=decimals, interpretation=interpretation)


def _rounding_preimage(mantissa, decimals, rounding):
    """
    Interval of values that round to ``mantissa`` under ``rounding``.
    """
    step = Decimal(1).scaleb(-decimals)
    if rounding is RoundingRule.ROUND_DOWN:
        # Truncation toward zero maps [mantissa, mantissa + step) to
        # mantissa.  This holds for negative mantissas as well.
        return mantissa, True, mantissa + step, False
    half = step / 2
    lower = mantissa - half
    upper = mantissa + half
    needed_precision = max(40, len(mantissa.as_tuple().digits) + decimals + 10)
    with localcontext() as context:
        context.prec = max(context.prec, needed_precision)
        lower_inclusive = lower.quantize(step, rounding=rounding.value) == mantissa
        upper_inclusive = upper.quantize(step, rounding=rounding.value) == mantissa
    return lower, lower_inclusive, upper, upper_inclusive


def display_interval(text, decimals=None, interpretation=None,
                     rounding=RoundingRule.ROUND_HALF_EVEN, unit=None,
                     format_spec=None):
    """
    Interval of stored values that could display as ``text``.

    Pass ``rounding=None`` to get the union of every candidate rule;
    use :func:`audit` when you need the per-rule verdicts.  A SAS
    ``format_spec`` such as ``PERCENT7.1`` can supply the decimal
    places and percent scale.
    """
    if format_spec is not None:
        hint = parse_format(format_spec)
        if decimals is None:
            decimals = hint.decimals
        if interpretation is None:
            interpretation = hint.interpretation
    parsed = parse_display(text)
    if interpretation is None:
        interpretation = parsed.interpretation
    elif not isinstance(interpretation, DisplayInterpretation):
        raise TypeError(f'Invalid display interpretation {interpretation!r}')
    if decimals is None:
        decimals = parsed.decimals
    elif decimals < parsed.decimals:
        raise ValueError(
            f'Explicit decimals={decimals} cannot express {text!r}, '
            f'which needs {parsed.decimals}'
        )
    if rounding is None:
        intervals = [
            display_interval(text, decimals, interpretation, rule, unit)
            for rule in RoundingRule
        ]
        return _union(intervals)
    if not isinstance(rounding, RoundingRule):
        raise TypeError(f'Invalid rounding rule {rounding!r}')
    lower, lower_inclusive, upper, upper_inclusive = _rounding_preimage(
        parsed.coefficient, decimals, rounding
    )
    scale = interpretation.value
    return ValueInterval(
        lower * scale,
        lower_inclusive,
        upper * scale,
        upper_inclusive,
        unit,
        decimals,
    )


def stored_interval(value, unit=None):
    """
    Singleton interval for an exact stored numeric value.
    """
    decimal_value = _to_decimal(value)
    return ValueInterval(decimal_value, True, decimal_value, True, unit, None)


def _union(intervals):
    """
    Smallest interval containing every candidate interval.
    """
    lower = min(interval.lower for interval in intervals)
    upper = max(interval.upper for interval in intervals)
    lower_inclusive = any(
        interval.lower == lower and interval.lower_inclusive for interval in intervals
    )
    upper_inclusive = any(
        interval.upper == upper and interval.upper_inclusive for interval in intervals
    )
    unit = next((interval.unit for interval in intervals if interval.unit), None)
    decimals = next(
        (interval.decimals for interval in intervals if interval.decimals is not None), None
    )
    return ValueInterval(lower, lower_inclusive, upper, upper_inclusive, unit, decimals)


def _check_units(left, right):
    """
    Reject comparisons across unconfirmed, mismatched dimensions.
    """
    if left.unit is not None and right.unit is not None and left.unit != right.unit:
        raise ValueError(
            f'Refusing to compare different units: {left.unit!r} vs {right.unit!r}; '
            'convert to a common unit before auditing'
        )


def compare_intervals(left, right):
    """
    Compare two intervals, treating open boundaries as exclusive.
    """
    _check_units(left, right)
    if not left.overlaps(right):
        explanation = (
            f'No overlap between {left} and {right}; the difference cannot be '
            'explained by display precision.'
        )
        return IntervalComparison(None, AuditConclusion.INCOMPATIBLE, explanation)
    overlap = left.intersection(right)
    explanation = (
        f'Intervals overlap at {overlap}; display precision can explain the '
        'difference, but it does not prove the originals are equal.'
    )
    return IntervalComparison(overlap, AuditConclusion.COMPATIBLE, explanation)


def compatible_with_display(value, text, decimals=None, interpretation=None,
                            rounding=RoundingRule.ROUND_HALF_EVEN, unit=None):
    """
    Whether a concrete value could have displayed as ``text``.
    """
    return display_interval(
        text, decimals, interpretation, rounding, unit
    ).contains(_to_decimal(value))


def audit(display, stored, interpretation=None, decimals=None,
          rounding=RoundingRule.ROUND_HALF_EVEN, unit=None, stored_unit=None,
          format_spec=None):
    """
    Audit a displayed string against a stored value.

    A SAS format such as ``PERCENT7.1`` can be passed as ``format_spec``
    to supply the decimal places and the percent scale from the variable
    metadata; explicit ``decimals`` or ``interpretation`` arguments take
    precedence over the format hint.

    When ``rounding`` is ``None``, every candidate rounding rule is
    evaluated.  If their verdicts disagree, the conclusion is
    :attr:`AuditConclusion.INDETERMINATE` and the per-rule evidence is
    retained in ``candidates``.
    """
    if format_spec is not None:
        hint = parse_format(format_spec)
        if decimals is None:
            decimals = hint.decimals
        if interpretation is None:
            interpretation = hint.interpretation
    if unit is not None and stored_unit is not None and unit != stored_unit:
        raise ValueError(
            f'Refusing to compare different units: {unit!r} vs {stored_unit!r}'
        )
    right_unit = stored_unit or unit
    right = stored_interval(stored, right_unit)
    rules = (rounding,) if rounding is not None else tuple(RoundingRule)
    candidates = []
    for rule in rules:
        left = display_interval(display, decimals, interpretation, rule, unit)
        candidates.append((rule, compare_intervals(left, right)))
    conclusions = {comparison.conclusion for _, comparison in candidates}
    if rounding is not None:
        rule, comparison = candidates[0]
        return AuditResult(
            left=display_interval(display, decimals, interpretation, rule, unit),
            right=right,
            overlap=comparison.overlap,
            conclusion=comparison.conclusion,
            explanation=comparison.explanation,
            candidates=tuple(candidates),
        )
    if len(conclusions) == 1:
        conclusion = next(iter(conclusions))
        overlap = candidates[0][1].overlap
        explanation = (
            f'All {len(candidates)} candidate rounding rules give the same '
            f'verdict: {conclusion.value}.'
        )
    else:
        conclusion = AuditConclusion.INDETERMINATE
        overlap = None
        detail = ', '.join(
            f'{rule.name}: {comparison.conclusion.value}' for rule, comparison in candidates
        )
        explanation = (
            'The rounding rule is undocumented and the candidate rules '
            f'disagree ({detail}); collect the format definition before '
            'concluding.'
        )
    return AuditResult(
        left=display_interval(display, decimals, interpretation, None, unit),
        right=right,
        overlap=overlap,
        conclusion=conclusion,
        explanation=explanation,
        candidates=tuple(candidates),
    )
