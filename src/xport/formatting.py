"""
SAS numeric formats and the information they preserve.

The SAS Transport formats store a numeric value and, separately, the
format used to display it.  Once a value has been reduced to its display
string, many different numeric values can look identical.  This module
models both directions:

* :func:`format_value` renders a number the way a SAS format would.
* :func:`precision_interval` recovers the interval of numeric values that
  could have produced a display string.

The interval is the useful part when reconciling historical summaries.
It says what was lost by displaying the value and, just as importantly,
what was *not* lost.
"""

# Standard Library
import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Optional

__all__ = [
    'FormatSpec',
    'PrecisionInterval',
    'coerce_format',
    'format_value',
    'parse_display',
    'precision_interval',
    'value_interval',
]

MISSING = '.'
_SPECIAL_MISSING = re.compile(r'^\.(?P<tag>[A-Z_])$')
_SCIENTIFIC = re.compile(
    r'^(?P<sign>[+-]?)(?P<integer>\d+)'
    r'(?:\.(?P<fraction>\d*))?'
    r'[Ee](?P<exponent>[+-]?\d+)$'
)
_FIXED = re.compile(r'^(?P<sign>[+-]?)(?P<integer>\d*)(?:\.(?P<fraction>\d*))?$')
_FORMAT_SPEC = re.compile(
    r'^(?P<name>\$?[A-Za-z0-9_]*?)'
    r'(?P<width>\d*)'
    r'\.?(?P<decimals>\d*)$'
)


@dataclass(frozen=True)
class FormatSpec:
    """
    A parsed numeric format, such as ``BEST12.`` or ``E12.4``.
    """

    name: str = ''
    width: Optional[int] = None
    decimals: Optional[int] = None
    justify: Optional[int] = None

    def __str__(self):
        """SAS-style display value."""
        if not (self.name or self.width or self.decimals):
            return ''
        decimals = '' if not self.decimals else str(self.decimals)
        width = '' if self.width is None else str(self.width)
        return f'{self.name}{width}.{decimals}'

    @property
    def is_character(self):
        """Whether this is a character (``$``) format."""
        return self.name.startswith('$')

    @property
    def is_numeric(self):
        """Whether this format describes a numeric value."""
        return not self.is_character


@dataclass(frozen=True)
class PrecisionInterval:
    """
    The set of numeric values consistent with a display string.

    Bounds are half-open by default: ``[low, high)``.  SAS rounds a
    displayed value to the requested number of decimal places, so the
    upper bound is normally exclusive.  ``None`` means the corresponding
    side is unbounded.
    """

    low: Optional[Decimal] = None
    high: Optional[Decimal] = None
    low_inclusive: bool = True
    high_inclusive: bool = False

    @property
    def is_bounded(self):
        """Whether both ends are known."""
        return self.low is not None and self.high is not None

    @property
    def is_point(self):
        """Whether the interval contains exactly one value."""
        return self.is_bounded and self.low == self.high

    @property
    def width(self):
        """Interval width, or ``None`` when unbounded."""
        if not self.is_bounded:
            return None
        return self.high - self.low

    def contains(self, value) -> bool:
        """
        Whether ``value`` lies in the interval.
        """
        if value is None:
            return False
        number = _as_decimal(value)
        if number is None:
            return False
        if self.low is not None:
            if number < self.low:
                return False
            if number == self.low and not self.low_inclusive:
                return False
        if self.high is not None:
            if number > self.high:
                return False
            if number == self.high and not self.high_inclusive:
                return False
        return True

    @classmethod
    def point(cls, value):
        """An interval containing exactly one value."""
        number = _as_decimal(value)
        return cls(number, number, low_inclusive=True, high_inclusive=True)

    def intersection(self, other):
        """
        Intersect two intervals, or return ``None`` if they are disjoint.
        """
        if other is None:
            return None
        low = None
        low_inclusive = True
        if self.low is not None and other.low is not None:
            if self.low > other.low:
                low, low_inclusive = self.low, self.low_inclusive
            elif other.low > self.low:
                low, low_inclusive = other.low, other.low_inclusive
            else:
                low = self.low
                low_inclusive = self.low_inclusive and other.low_inclusive
        else:
            low = self.low if self.low is not None else other.low
            low_inclusive = (
                self.low_inclusive if self.low is not None else other.low_inclusive
            )

        high = None
        high_inclusive = False
        if self.high is not None and other.high is not None:
            if self.high < other.high:
                high, high_inclusive = self.high, self.high_inclusive
            elif other.high < self.high:
                high, high_inclusive = other.high, other.high_inclusive
            else:
                high = self.high
                high_inclusive = self.high_inclusive and other.high_inclusive
        else:
            high = self.high if self.high is not None else other.high
            high_inclusive = (
                self.high_inclusive if self.high is not None else other.high_inclusive
            )

        if low is not None and high is not None:
            if low > high:
                return None
            if low == high and not (low_inclusive and high_inclusive):
                return None
        return PrecisionInterval(low, high, low_inclusive, high_inclusive)

    def __and__(self, other):
        """Intersect two intervals."""
        return self.intersection(other)

    def union(self, other):
        """
        Smallest interval covering both intervals.
        """
        if other is None:
            return self
        if self.low is None or other.low is None:
            low, low_inclusive = None, True
        elif self.low < other.low:
            low, low_inclusive = self.low, self.low_inclusive
        elif other.low < self.low:
            low, low_inclusive = other.low, other.low_inclusive
        else:
            low, low_inclusive = self.low, self.low_inclusive or other.low_inclusive

        if self.high is None or other.high is None:
            high, high_inclusive = None, False
        elif self.high > other.high:
            high, high_inclusive = self.high, self.high_inclusive
        elif other.high > self.high:
            high, high_inclusive = other.high, other.high_inclusive
        else:
            high, high_inclusive = self.high, self.high_inclusive or other.high_inclusive
        return PrecisionInterval(low, high, low_inclusive, high_inclusive)

    def __or__(self, other):
        """Union two intervals."""
        return self.union(other)

    def as_dict(self):
        """JSON-friendly representation."""
        return {
            'low': _display_decimal(self.low),
            'high': _display_decimal(self.high),
            'low_inclusive': self.low_inclusive,
            'high_inclusive': self.high_inclusive,
            'width': _display_decimal(self.width),
        }


def _as_decimal(value) -> Optional[Decimal]:
    """
    Convert a numeric value to ``Decimal`` without binary-float noise.
    """
    if value is None:
        return None
    if isinstance(value, Decimal):
        number = value
    elif isinstance(value, bool):
        number = Decimal(int(value))
    elif isinstance(value, int):
        number = Decimal(value)
    else:
        try:
            number = Decimal(str(value))
        except (InvalidOperation, ValueError):
            return None
    if not number.is_finite():
        return None
    return number


def _display_decimal(value) -> Optional[str]:
    """
    Display a ``Decimal`` without exponent notation when practical.
    """
    if value is None:
        return None
    if not isinstance(value, Decimal):
        value = Decimal(str(value))
    if value == value.to_integral_value():
        return str(value.quantize(Decimal(1)))
    text = format(value.normalize(), 'f')
    if '.' in text:
        text = text.rstrip('0').rstrip('.')
    return text or '0'


def coerce_format(value) -> FormatSpec:
    """
    Coerce a format object, string, or tuple to a :class:`FormatSpec`.
    """
    if value is None:
        return FormatSpec()
    if isinstance(value, FormatSpec):
        return value
    if hasattr(value, 'name') and hasattr(value, 'length') and hasattr(value, 'decimals'):
        return FormatSpec(
            name=str(value.name or '').upper(),
            width=getattr(value, 'length', None),
            decimals=getattr(value, 'decimals', None),
            justify=getattr(value, 'justify', None),
        )
    if isinstance(value, (tuple, list)):
        name = str(value[0] or '') if len(value) > 0 else ''
        width = value[1] if len(value) > 1 else None
        decimals = value[2] if len(value) > 2 else None
        return FormatSpec(name.upper(), width, decimals)
    text = str(value).strip()
    if not text:
        return FormatSpec()
    match = _FORMAT_SPEC.fullmatch(text)
    if match is None:
        raise ValueError(f'Invalid numeric format {value!r}')
    name = match.group('name').upper()
    width = int(match.group('width')) if match.group('width') else None
    decimals = int(match.group('decimals')) if match.group('decimals') else None
    return FormatSpec(name=name, width=width, decimals=decimals)


def _is_missing(value) -> bool:
    """Whether ``value`` is a SAS missing value."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, Decimal) and not value.is_finite():
        return True
    return False


def _missing_text(value) -> str:
    """SAS display text for a missing value."""
    name = getattr(value, 'name', None)
    if name and name in '_ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        return f'.{name}'
    return MISSING


def _asterisks(width: Optional[int]) -> str:
    """SAS overflow marker."""
    return '*' * (width or 1)


def _fit(text: str, width: Optional[int]) -> str:
    """Return ``text`` or SAS-style overflow asterisks."""
    if width and len(text) > width:
        return _asterisks(width)
    return text


def _round(value: Decimal, decimals: int) -> Decimal:
    """Round half away from zero, as SAS does for display."""
    quantum = Decimal(1).scaleb(-decimals)
    return value.quantize(quantum, rounding=ROUND_HALF_UP)


def _fixed(
    value: Decimal,
    width: Optional[int],
    decimals: int,
    *,
    grouping: bool = False,
    prefix: str = '',
    suffix: str = '',
) -> str:
    """Render a fixed-point numeric format."""
    rounded = _round(value, decimals)
    if rounded == 0:
        rounded = abs(rounded)
    template = f'{{:,.{decimals}f}}' if grouping else f'{{:.{decimals}f}}'
    text = prefix + template.format(rounded) + suffix
    return _fit(text, width)


def _scientific(value: Decimal, width: Optional[int], decimals: int) -> str:
    """Render a scientific-notation numeric format."""
    if value == 0:
        mantissa = Decimal(0)
        exponent = 0
    else:
        exponent = value.copy_abs().adjusted()
        mantissa = value.scaleb(-exponent)
        mantissa = _round(mantissa, decimals)
        if abs(mantissa) >= 10:
            mantissa /= 10
            exponent += 1
    digits = f'{mantissa:.{decimals}f}'
    text = f'{digits}E{exponent:+03d}'
    return _fit(text, width)


def _general(value: Decimal, width: Optional[int]) -> str:
    """Render a value without a fixed number of decimals."""
    if value == value.to_integral_value():
        text = str(value.quantize(Decimal(1)))
    else:
        text = format(value.normalize(), 'f')
        if '.' in text:
            text = text.rstrip('0').rstrip('.')
    if width and len(text) > width:
        return ''
    return text


def _best(value: Decimal, width: Optional[int], decimals: Optional[int]) -> str:
    """
    Approximate SAS ``BESTw.``: the shortest form that fits the width.

    Integers use fixed notation when they fit.  Anything that would lose
    its leading significant digits, or that rounds to zero, falls back to
    scientific notation with as many mantissa digits as the width allows.
    The exact SAS algorithm varies by version; this implementation keeps
    the useful property that its output is deterministic and that
    :func:`precision_interval` recovers the information the display
    actually preserved.
    """
    if not width:
        general = _general(value, None)
        if general:
            return general
        return _scientific(value, None, decimals or 0)

    if decimals:
        rounded = _round(value, decimals)
        if rounded != 0 or value == 0:
            fixed = _fixed(value, width, decimals)
            if fixed != _asterisks(width) and _preserves_magnitude(fixed, value):
                return fixed
    else:
        general = _general(value, width)
        if general and _preserves_magnitude(general, value):
            return general

    exponent = 0 if value == 0 else value.copy_abs().adjusted()
    exponent_width = len(str(abs(exponent))) + 1
    overhead = 1 + 1 + 1 + exponent_width  # digit, dot, E, signed exponent
    available = max(0, width - overhead)
    if decimals:
        available = min(available, decimals)
    for digits in range(available, -1, -1):
        text = _scientific(value, width, digits)
        if text != _asterisks(width):
            return text
    return _asterisks(width)


def _preserves_magnitude(text: str, value: Decimal) -> bool:
    """
    Whether a fixed-point rendering keeps the leading significant digits.

    ``BEST`` does not turn a large number into a smaller one that merely
    rounds to zero; it switches to scientific notation instead.
    """
    if value == 0:
        return True
    parsed = parse_display(text)
    if parsed is None:
        return False
    if parsed == 0:
        return False
    return abs(parsed - value) < abs(value) / 10


def format_value(value, format=None):
    """
    Render ``value`` using a SAS numeric format.

    ``format`` may be a :class:`xport.Format`, a format string such as
    ``'BEST12.'``, a :class:`FormatSpec`, or ``None``.  The returned text
    is *not* blank-padded; SAS right-justifies numeric fields within the
    format width, but padding carries no information for reconciliation.
    """
    spec = coerce_format(format)
    if _is_missing(value):
        return _missing_text(value)

    if isinstance(value, str):
        return value

    number = _as_decimal(value)
    if number is None:
        return str(value)

    if spec.is_character:
        return str(value)

    name = spec.name.upper()
    width = spec.width
    decimals = spec.decimals or 0

    if name in ('BEST', ''):
        if name == '' and spec.width is not None:
            return _fixed(number, width, decimals)
        # ``BEST12.`` means "no fixed decimal count"; parsers that store
        # the omitted decimal count as 0 must not be read as ``BEST12.0``.
        return _best(number, width, spec.decimals or None)

    if name in ('E', 'SCIENTIFIC'):
        return _scientific(number, width, decimals)

    if name in ('F', 'NUMERIC', 'COMMA'):
        return _fixed(number, width, decimals, grouping=(name == 'COMMA'))

    if name == 'DOLLAR':
        return _fixed(number, width, decimals, grouping=True, prefix='$')

    if name in ('PERCENT', 'PCT'):
        return _fixed(number * 100, width, decimals, suffix='%')

    if name in ('Z', 'HEX'):
        try:
            integer = int(number)
        except (ValueError, OverflowError):
            return _fit(str(number), width)
        text = f'{integer:X}'
        return _fit(text, width)

    return _fit(str(value), width)


def parse_display(display) -> Optional[Decimal]:
    """
    Parse the numeric value shown in a display string.

    Returns ``None`` for missing values, overflow markers, and text that
    does not look like a SAS numeric display.
    """
    if display is None:
        return None
    text = str(display).strip()
    if not text or text in ('NaN', 'nan', 'NA', 'N/A'):
        return None
    if _SPECIAL_MISSING.match(text) or text == MISSING:
        return None
    if set(text) == {'*'}:
        return None

    percent = text.endswith('%')
    if percent:
        text = text[:-1].strip()
    text = text.replace(',', '').replace('$', '').strip()
    if text.startswith('(') and text.endswith(')'):
        text = '-' + text[1:-1].strip()

    match = _SCIENTIFIC.fullmatch(text)
    if match:
        sign = -1 if match.group('sign') == '-' else 1
        fraction = match.group('fraction') or ''
        number = Decimal(match.group('integer') + ('.' + fraction if fraction else ''))
        number *= sign
        exponent = int(match.group('exponent'))
        number *= Decimal(10) ** exponent
        return number / 100 if percent else number

    match = _FIXED.fullmatch(text)
    if match and (match.group('integer') or match.group('fraction')):
        sign = -1 if match.group('sign') == '-' else 1
        integer = match.group('integer') or '0'
        fraction = match.group('fraction') or ''
        number = Decimal(integer + ('.' + fraction if fraction else ''))
        number *= sign
        return number / 100 if percent else number

    return None


def precision_interval(display, format=None) -> Optional[PrecisionInterval]:
    """
    Recover the interval of values that could display as ``display``.

    The number of digits in the display determines the rounding step.  A
    format may be supplied for context, but the display string is the
    evidence: a truncated display does not become more precise merely
    because the original format allowed more digits.

    Returns ``None`` when the display contains no numeric information
    (for example ``.``, ``.A``, or ``****``).
    """
    if display is None:
        return None
    text = str(display).strip()
    if not text or text in ('NaN', 'nan', 'NA', 'N/A'):
        return None
    if _SPECIAL_MISSING.match(text) or text == MISSING:
        return None
    if set(text) == {'*'}:
        return None

    percent = text.endswith('%')
    if percent:
        text = text[:-1].strip()
    text = text.replace(',', '').replace('$', '').strip()
    if text.startswith('(') and text.endswith(')'):
        text = '-' + text[1:-1].strip()

    match = _SCIENTIFIC.fullmatch(text)
    if match:
        sign = -1 if match.group('sign') == '-' else 1
        fraction = match.group('fraction') or ''
        decimals = len(fraction)
        mantissa = Decimal(match.group('integer') + ('.' + fraction if fraction else ''))
        mantissa *= sign
        exponent = int(match.group('exponent'))
        step = Decimal(5).scaleb(exponent - decimals - 1)
        value = mantissa.scaleb(exponent)
        low, high = value - step, value + step
    else:
        match = _FIXED.fullmatch(text)
        if match is None or not (match.group('integer') or match.group('fraction')):
            return None
        sign = -1 if match.group('sign') == '-' else 1
        integer = match.group('integer') or '0'
        fraction = match.group('fraction') or ''
        decimals = len(fraction)
        value = Decimal(integer + ('.' + fraction if fraction else '')) * sign
        step = Decimal(5).scaleb(-decimals - 1)
        low, high = value - step, value + step

    if percent:
        low, high = low / 100, high / 100
        value = value / 100

    # SAS rounds half away from zero.  A positive value at the lower
    # boundary rounds up into the display; a negative value at its lower
    # boundary rounds away from zero and out of the display.
    if value > 0:
        low_inclusive, high_inclusive = True, False
    elif value < 0:
        low_inclusive, high_inclusive = False, True
    else:
        low_inclusive, high_inclusive = False, False
    return PrecisionInterval(low, high, low_inclusive, high_inclusive)


def value_interval(value, format=None, exact=True) -> Optional[PrecisionInterval]:
    """
    Interval for a retained numeric value.

    By default a retained value is treated as exact.  The format is
    display metadata, not evidence that the stored value was rounded.
    Pass ``exact=False`` when the value came from a display that was
    itself rounded to the format's decimals.
    """
    number = _as_decimal(value)
    if number is None:
        return None

    spec = coerce_format(format)
    if not exact and spec.is_numeric and spec.decimals is not None:
        step = Decimal(5).scaleb(-spec.decimals - 1)
        return PrecisionInterval(number - step, number + step, True, False)
    return PrecisionInterval.point(number)
