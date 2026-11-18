"""
Audit IBM floating point representations and their IEEE numeric views.

A SAS Transport (XPORT) numeric field is stored in IBM hexadecimal floating
point.  Reading it yields a Python ``float``, whose IEEE-754 representation
has a narrower significand.  The two numeric values can be very close while
the original 56-bit IBM fraction is not reproducible from the float.  Numeric
equality alone is therefore not evidence that a value can be re-encoded
without loss.

This module compares the original bytes with a fresh numeric conversion and
reports which stage accounts for any difference:

* ``missing``: SAS missing values, including tagged values such as ``.A``
* ``negative_zero``: IBM and IEEE both distinguish ``-0.0`` from ``0.0``
* ``underflow``: the numeric view is below the smallest IBM value
* ``overflow``: the numeric view is above the largest IBM value
* ``precision``: the IBM fraction had bits that IEEE-754 cannot hold
* ``zero``: a non-canonical IBM zero encoding
* ``exact``: the original encoding was reproduced byte-for-byte
"""

# Standard Library
import dataclasses
import math
from collections.abc import Mapping
from fractions import Fraction

# Xport Modules
import xport

__all__ = [
    'AuditReport',
    'AuditResult',
    'EXACT',
    'MISSING',
    'NEGATIVE_ZERO',
    'OVERFLOW',
    'PRECISION',
    'PRECISION_ONLY',
    'Tolerance',
    'UNDERFLOW',
    'ZERO',
    'audit_array',
    'audit_dataset',
    'audit_float',
    'audit_library',
    'audit_series',
    'audit_value',
    'can_reencode',
    'exact_ibm_value',
    'field_tolerance',
    'reencode',
]

EXACT = 'exact'
MISSING = 'missing'
NEGATIVE_ZERO = 'negative_zero'
OVERFLOW = 'overflow'
PRECISION = 'precision'
PRECISION_ONLY = 'precision_only'
UNDERFLOW = 'underflow'
ZERO = 'zero'
ENCODING = 'encoding_mismatch'

# An IBM value is ``sign * fraction / 2 ** 56 * 16 ** (exponent - 64)``.
FRACTION_BITS = 56

_TAG_BYTES = frozenset(b'_ABCDEFGHIJKLMNOPQRSTUVWXYZ')
_ZERO_BITS = b'\x00' * 8


@dataclasses.dataclass(frozen=True)
class Tolerance:
    """
    Field-level error requirement.

    ``rtol`` is a relative error bound.  ``atol`` is an absolute error bound.
    ``decimals`` is a SAS format width: a value is acceptable when it agrees
    to the displayed decimal place, allowing half a unit in the last place.
    """

    rtol: float = None
    atol: float = None
    decimals: int = None

    def __post_init__(self):
        if self.decimals is not None:
            if self.decimals < 0:
                raise ValueError('decimals must be non-negative')
            if self.atol is None:
                object.__setattr__(self, 'atol', 0.5 * 10.0**(-self.decimals))
        if self.rtol is not None and self.rtol < 0:
            raise ValueError('rtol must be non-negative')
        if self.atol is not None and self.atol < 0:
            raise ValueError('atol must be non-negative')

    def allows(self, original, candidate):
        """
        Whether ``candidate`` is within the error requirement of ``original``.
        """
        original = Fraction(original)
        candidate = Fraction(candidate)
        error = abs(candidate - original)
        if self.atol is not None and error <= Fraction(self.atol):
            return True
        if self.rtol is not None:
            if original == 0:
                return error == 0
            return error <= abs(original) * Fraction(self.rtol)
        return error == 0

    def __bool__(self):
        return self.rtol is not None or self.atol is not None

    @classmethod
    def coerce(cls, tolerance=None, **kwds):
        """
        Normalize a tolerance specification.

        A number is interpreted as a relative tolerance.  A mapping may
        contain ``rtol``, ``atol``, ``rel_tol``, ``abs_tol``, or ``decimals``.
        """
        aliases = {
            'rel_tol': 'rtol',
            'relative': 'rtol',
            'abs_tol': 'atol',
            'absolute': 'atol',
        }
        options = {aliases.get(k, k): v for k, v in kwds.items() if v is not None}
        if isinstance(tolerance, Tolerance):
            if options:
                values = dataclasses.asdict(tolerance)
                values.update(options)
                return cls(**values)
            return tolerance
        if isinstance(tolerance, dict):
            values = {
                aliases.get(k, k): v for k, v in tolerance.items() if v is not None
            }
            values.update(options)
            return cls(**values)
        if tolerance is not None and not isinstance(tolerance, (int, float)):
            raise TypeError(f'Unsupported tolerance {tolerance!r}')
        if tolerance is not None:
            options.setdefault('rtol', tolerance)
        if not options:
            return None
        return cls(**options)


@dataclasses.dataclass
class AuditResult:
    """
    Result of auditing one numeric field value.

    ``raw`` is the original IBM representation, possibly shorter than eight
    bytes.  ``encoded`` is a fresh conversion of the numeric view, padded to
    the field width.  ``byte_equal`` compares those representations;
    ``float_equal`` compares numeric views.  The latter can be true while the
    former is false, which is exactly why this audit exists.
    """

    raw: bytes = None
    width: int = 8
    value: float = None
    encoded: bytes = None
    status: str = EXACT
    reason: str = ''
    tolerance: Tolerance = None
    absolute_error: Fraction = None
    relative_error: Fraction = None
    byte_equal: bool = None
    float_equal: bool = None
    lossless: bool = False
    reencodable: bool = False
    missing_tag: str = None

    @property
    def ok(self):
        """Whether the value satisfies the audit."""
        return self.reencodable

    @property
    def exact(self):
        """Whether the original representation was reproduced exactly."""
        return bool(self.byte_equal)

    @property
    def error(self):
        """Relative error when meaningful, otherwise absolute error."""
        if self.relative_error is not None:
            return float(self.relative_error)
        if self.absolute_error is not None:
            return float(self.absolute_error)
        return None

    @property
    def explanation(self):
        """Human-readable explanation."""
        return self.reason

    @property
    def numeric(self):
        """
        Whether the result has an ordinary numeric value.

        Missing values and unencodable overflow/underflow fields are not
        numeric, which prevents accidental comparisons that would treat a NaN
        as equal to another NaN.
        """
        return (
            self.value is not None and math.isfinite(self.value) and
            self.status not in {MISSING, UNDERFLOW, OVERFLOW, ENCODING}
        )

    def __bool__(self):
        return self.reencodable

    def as_dict(self):
        """Return a plain mapping suitable for reporting."""
        return {
            'raw': self.raw.hex() if self.raw is not None else None,
            'width': self.width,
            'value': self.value,
            'encoded': self.encoded.hex() if self.encoded is not None else None,
            'status': self.status,
            'reason': self.reason,
            'byte_equal': self.byte_equal,
            'float_equal': self.float_equal,
            'lossless': self.lossless,
            'reencodable': self.reencodable,
            'absolute_error': None if self.absolute_error is None else float(
                self.absolute_error
            ),
            'relative_error': None if self.relative_error is None else float(
                self.relative_error
            ),
            'missing_tag': self.missing_tag,
        }


def exact_ibm_value(raw):
    """
    Return the exact rational value of IBM-format bytes.

    Unlike ``ibm_to_ieee``, this never rounds to IEEE precision.  Missing
    values and non-canonical zeros with a zero fraction return ``None`` or
    zero as appropriate.
    """
    raw = bytes(raw).ljust(8, b'\x00')
    bits = int.from_bytes(raw, 'big')
    sign = -1 if bits & 0x8000000000000000 else 1
    exponent = (bits >> 56) & 0x7f
    fraction = bits & ((1 << FRACTION_BITS) - 1)
    if fraction == 0:
        if raw[:1] == b'.' or raw[:1] in _TAG_BYTES:
            return None
        return Fraction(0)
    value = Fraction(sign * fraction, 1 << FRACTION_BITS)
    return value * Fraction(16)**(exponent - 64)


def _field_width(raw, width):
    if width is None:
        return len(raw)
    return int(width)


def _fresh_encode(value):
    """
    Encode a numeric view, translating arithmetic failures for reporting.
    """
    from xport.v56 import Overflow, Underflow, ieee_to_ibm
    try:
        return ieee_to_ibm(value), None
    except Underflow:
        return None, UNDERFLOW
    except Overflow:
        return None, OVERFLOW
    except (NotImplementedError, ValueError, TypeError):
        return None, ENCODING


def _missing_tag(raw):
    raw = bytes(raw).ljust(8, b'\x00')
    if raw[0] in _TAG_BYTES and not any(raw[1:]):
        return chr(raw[0])
    return None


def _is_dot_missing(raw):
    raw = bytes(raw).ljust(8, b'\x00')
    return raw[:1] == b'.' and not any(raw[1:])


def _is_negative_zero(value):
    return value == 0.0 and math.copysign(1.0, value) < 0


def _float_fraction(value):
    if value is None or not math.isfinite(value):
        return None
    return Fraction(*value.as_integer_ratio())


def _compare(original, candidate):
    """
    Absolute and relative error of exact rational ``candidate``.
    """
    if original is None or candidate is None:
        return None, None
    error = abs(candidate - original)
    relative = None if original == 0 else error / abs(original)
    return error, relative


def _within(result, original, candidate):
    if result.tolerance is None:
        return bool(result.byte_equal)
    if original is None or candidate is None:
        return False
    return result.tolerance.allows(original, candidate)


def audit_value(raw, value=None, width=None, tolerance=None, decimals=None,
                format=None, **kwds):
    """
    Audit whether an IBM-format field can be re-encoded from its numeric view.

    ``raw`` is the original field bytes.  ``value`` is the numeric view; when
    omitted, ``ibm_to_ieee`` is used.  ``tolerance`` may be a relative number,
    a ``Tolerance`` instance, or a mapping with ``rtol`` and/or ``atol``.
    ``decimals`` provides a field error requirement based on the variable's
    displayed decimal places.

    The result is re-encodable when either the bytes are identical or the
    fresh conversion satisfies the supplied tolerance.  ``lossless`` is true
    only for byte-identical encodings.  In particular, a matching float never
    by itself makes ``lossless`` true.
    """
    from xport.v56 import ibm_to_ieee

    raw = bytes(raw)
    width = _field_width(raw, width)
    if decimals is None and format is not None:
        decimals = getattr(format, 'decimals', None)
    tol = Tolerance.coerce(tolerance, decimals=decimals, **kwds)
    if value is None:
        value = float(ibm_to_ieee(raw))
    else:
        value = float(value)

    result = AuditResult(
        raw=raw,
        width=width,
        value=value,
        tolerance=tol,
    )

    # Special missing values carry a letter in an otherwise-zero fraction.
    tag = _missing_tag(raw)
    if tag is not None:
        encoded = (tag.encode('ascii') + b'\x00' * 7)[:width].ljust(width, b'\x00')
        original = raw[:width].ljust(width, b'\x00')
        result.encoded = encoded
        result.missing_tag = tag
        result.status = MISSING
        result.byte_equal = encoded == original
        result.float_equal = False
        result.lossless = result.byte_equal
        result.reencodable = result.byte_equal
        result.reason = (
            f'Special missing value .{tag} is encoded as its tag byte; '
            f'it has no numeric value and is never equal to another NaN.'
        )
        return result

    if _is_dot_missing(raw):
        encoded = b'.' + b'\x00' * 7
        result.encoded = encoded[:width].ljust(width, b'\x00')
        result.status = MISSING
        result.missing_tag = '.'
        result.byte_equal = result.encoded == raw[:width].ljust(width, b'\x00')
        result.float_equal = False
        result.lossless = result.byte_equal
        result.reencodable = result.byte_equal
        result.reason = (
            'Ordinary SAS missing value (.) is preserved as a missing marker, '
            'not as an IEEE NaN payload.'
        )
        return result

    original = exact_ibm_value(raw)

    # A zero fraction is a zero regardless of exponent.  Sign is significant,
    # so -0.0 and 0.0 have different encodings and different explanations.
    if original == 0:
        encoded = _fresh_encode(value)[0]
        if encoded is None:
            encoded = _ZERO_BITS
        result.encoded = encoded[:width].ljust(width, b'\x00')
        result.byte_equal = result.encoded == raw[:width].ljust(width, b'\x00')
        result.float_equal = True
        result.lossless = result.byte_equal
        result.reencodable = result.byte_equal or bool(
            result.tolerance and result.tolerance.allows(0, 0)
        )
        padded = raw.ljust(8, b'\x00')
        raw_negative = bool(padded[0] & 0x80)
        canonical = not any(padded)
        if raw_negative:
            result.status = NEGATIVE_ZERO
            result.reason = (
                'Negative zero is a distinct IBM encoding (sign bit set, zero '
                'fraction).  Preserve the sign; float equality to +0.0 is not '
                'enough to call the encoding lossless.'
            )
            if not _is_negative_zero(value):
                result.reason += ' The numeric view has lost the negative sign.'
        elif canonical and result.byte_equal:
            result.status = EXACT
            result.reason = 'The canonical positive IBM zero was reproduced byte-for-byte.'
        else:
            result.status = ZERO
            result.reason = (
                'IBM permits alternative zero encodings, but the canonical '
                'positive zero is all zero bits.  Numeric zero does not show '
                'whether the original sign or exponent bits were canonical.'
            )
            if _is_negative_zero(value):
                result.reason += ' The numeric view has a negative sign.'
        return result

    encoded, failure = _fresh_encode(value)
    if encoded is None:
        result.encoded = None
        result.status = failure
        result.byte_equal = False
        result.float_equal = False
        result.lossless = False
        result.reencodable = False
        if failure == UNDERFLOW:
            result.reason = (
                f'Underflow: {value!r} is smaller in magnitude than the '
                f'smallest normalized IBM value (~16 ** -65); re-encoding '
                f'would round to zero instead of reproducing the source.'
            )
        elif failure == OVERFLOW:
            result.reason = (
                f'Overflow: {value!r} is larger in magnitude than the largest '
                f'IBM value (~16 ** 63); it cannot be re-encoded.'
            )
        else:
            result.reason = 'The numeric view cannot be encoded as IBM floating point.'
        return result

    field = encoded[:width].ljust(width, b'\x00')
    original_field = raw[:width].ljust(width, b'\x00')
    result.encoded = field
    result.byte_equal = field == original_field
    try:
        result.float_equal = float(ibm_to_ieee(field)) == value
    except (ValueError, TypeError):
        result.float_equal = False

    # Compare exact rational values, not rounded floats.  This exposes IBM's
    # extra fraction bits even when the two floats compare equal.
    candidate = exact_ibm_value(field)
    absolute, relative = _compare(original, candidate)
    result.absolute_error = absolute
    result.relative_error = relative
    result.lossless = result.byte_equal

    if result.byte_equal:
        result.status = EXACT
        result.reencodable = True
        result.reason = 'The original IBM encoding was reproduced byte-for-byte.'
        return result

    if result.tolerance is not None and candidate is not None:
        result.reencodable = result.tolerance.allows(original, candidate)
    else:
        result.reencodable = False

    if result.float_equal:
        result.status = PRECISION_ONLY
        result.reason = (
            'The float values compare equal, but the original 56-bit IBM '
            'fraction cannot be reproduced from the IEEE-754 double.  Float '
            'equality alone is not lossless.'
        )
    else:
        result.status = PRECISION
        result.reason = (
            'The IEEE numeric view and the original IBM fraction differ; '
            're-encoding changes the field.'
        )
    if result.reencodable:
        result.reason += ' The difference is within the field error requirement.'
    return result


def audit_float(value, tolerance=None, **kwds):
    """
    Audit a numeric value for IBM encodability without an original encoding.

    This checks overflow and underflow and reports the exact error introduced
    by the IBM round trip.  It cannot report byte-level losslessness because
    no source representation exists.
    """
    tol = Tolerance.coerce(tolerance, **kwds)
    result = AuditResult(value=float(value), tolerance=tol, status=ENCODING)
    encoded, failure = _fresh_encode(float(value))
    if encoded is None:
        result.status = failure
        result.byte_equal = False
        result.float_equal = False
        result.reencodable = False
        result.lossless = False
        if failure == UNDERFLOW:
            result.reason = (
                f'Underflow: {value!r} is below the smallest normalized IBM '
                f'value (~16 ** -65); writing it would round to zero.'
            )
        elif failure == OVERFLOW:
            result.reason = (
                f'Overflow: {value!r} exceeds the largest IBM value (~16 ** 63).'
            )
        else:
            result.reason = 'The value cannot be represented as IBM floating point.'
        return result

    result.encoded = encoded
    decoded = float(xport.v56.ibm_to_ieee(encoded))
    original = _float_fraction(float(value))
    candidate = _float_fraction(decoded)
    result.absolute_error, result.relative_error = _compare(original, candidate)
    result.float_equal = decoded == float(value)
    result.byte_equal = None
    result.reencodable = result.float_equal or (
        tol is not None and original is not None and candidate is not None
        and tol.allows(original, candidate)
    )
    result.lossless = False
    if result.float_equal:
        result.status = EXACT
        result.reason = (
            'The value can be encoded and decoded to the same IEEE float; '
            'there is no original IBM encoding to compare byte-for-byte.'
        )
    else:
        result.status = PRECISION
        result.reason = 'The IBM round trip changes the numeric value.'
        if result.reencodable:
            result.reason += ' It is within the field error requirement.'
    return result


def reencode(raw, value=None, width=None, preserve_raw=False):
    """
    Re-encode one field, returning IBM-format bytes.

    When ``value`` is given, a fresh numeric conversion is used.  Otherwise
    the original bytes are returned only if ``preserve_raw`` is true;
    this function does not silently claim a conversion happened.
    """
    from xport.v56 import encode_value
    if value is None:
        if raw is None:
            raise ValueError('Either raw bytes or a numeric value is required')
        if preserve_raw:
            return bytes(raw).ljust(8, b'\x00')
        value = float(xport.v56.ibm_to_ieee(raw))
    encoded = encode_value(value, preserve_raw=preserve_raw)
    if width is not None:
        encoded = encoded[:width].ljust(width, b'\x00')
    return encoded


def audit_series(values, raws=None, tolerance=None, name=None, **kwds):
    """
    Audit a sequence of values and optional original encodings.

    ``raws`` may be a sequence parallel to ``values`` or a single value to be
    applied to all entries.  Results are returned as a list.
    """
    values = list(values)
    if raws is None:
        raws = [None] * len(values)
    elif isinstance(raws, (bytes, bytearray, memoryview)):
        raws = [bytes(raws)] * len(values)
    else:
        raws = list(raws)
        if len(raws) != len(values):
            raise ValueError('raws and values must have the same length')
    tol = Tolerance.coerce(tolerance, **kwds)
    return [
        audit_value(raw, value=value, tolerance=tol) if raw is not None
        else audit_float(value, tolerance=tol)
        for value, raw in zip(values, raws)
    ]


audit_array = audit_series


class AuditReport:
    """
    Collection of audit results for a dataset or library.

    Iteration yields results.  ``reencodable`` is true when every result
    satisfies the field error requirement; ``lossless`` additionally requires
    every original encoding to be reproduced byte-for-byte.
    """

    def __init__(self, results=(), name=None, tolerance=None):
        self.results = list(results)
        self.name = name
        self.tolerance = tolerance

    def __iter__(self):
        return iter(self.results)

    def __len__(self):
        return len(self.results)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            column, index = key
            return next(r for k, r in self.items() if k == (column, index))
        if isinstance(key, str):
            return [r for k, r in self.items() if k[0] == key]
        return self.results[key]

    def __contains__(self, key):
        if isinstance(key, tuple):
            return any(k == key for k, _ in self.items())
        if isinstance(key, str):
            return any(k[0] == key for k, _ in self.items())
        return key in self.results

    def items(self):
        """Yield ``(column, index)`` keys and results in input order."""
        for result in self.results:
            yield (getattr(result, 'column', None), getattr(result, 'index', None)), result

    @property
    def ok(self):
        """Whether every value is re-encodable."""
        return self.reencodable

    @property
    def reencodable(self):
        """Whether every value is re-encodable."""
        return all(r.reencodable for r in self.results)

    @property
    def all_reencodable(self):
        """Alias for ``reencodable``."""
        return self.reencodable

    @property
    def lossless(self):
        """Whether every value was reproduced byte-for-byte."""
        return all(r.lossless for r in self.results)

    @property
    def all_lossless(self):
        """Alias for ``lossless``."""
        return self.lossless

    @property
    def failures(self):
        """Results that are not re-encodable."""
        return [r for r in self.results if not r.reencodable]

    @property
    def counts(self):
        """Counts by status."""
        counts = {}
        for r in self.results:
            counts[r.status] = counts.get(r.status, 0) + 1
        return counts

    @property
    def explanations(self):
        """Distinct explanations, useful for audit logs."""
        return sorted({r.reason for r in self.results if r.reason})

    def to_frame(self):
        """Return a Pandas DataFrame of results."""
        import pandas as pd
        rows = []
        for (column, index), result in self.items():
            row = result.as_dict()
            row['variable'] = column
            row['index'] = index
            rows.append(row)
        return pd.DataFrame(rows)

    def __bool__(self):
        return self.reencodable

    def __repr__(self):
        return (
            f'{type(self).__name__}(name={self.name!r}, '
            f'results={len(self.results)}, reencodable={self.reencodable}, '
            f'lossless={self.lossless})'
        )


def _raw_columns(data):
    raw_columns = getattr(data, '_raw_columns', None)
    if raw_columns:
        return raw_columns
    try:
        raw = data.raw
    except (AttributeError, ValueError, TypeError):
        return {}
    if raw is None:
        return {}
    return {k: list(v) for k, v in raw.items()}


def _aligned_raws(raw, index):
    """
    Convert a raw series or sequence to a list aligned with ``index``.
    """
    import pandas as pd
    if isinstance(raw, pd.Series):
        return list(raw.reindex(index))
    raw = list(raw)
    if len(raw) != len(index):
        raise ValueError('Raw encodings and values have different lengths')
    return raw


def _numeric_columns(dataset, raw_columns):
    """
    Names of numeric columns, using raw encodings to identify IBM fields.

    Character columns are excluded even when they happen to have a numeric
    Pandas dtype, because encoding them as IBM floats would be meaningless.
    """
    import pandas as pd
    columns = []
    for name in dataset.columns:
        if name in raw_columns:
            columns.append(name)
            continue
        dtype = getattr(dataset[name], 'dtype', None)
        if (
            dtype is not None and pd.api.types.is_numeric_dtype(dtype) and
            not pd.api.types.is_bool_dtype(dtype)
        ):
            columns.append(name)
    return columns


def _column_tolerance(dataset, column, tolerance, options):
    """
    Resolve per-column field error requirements.

    Precedence is an explicit mapping entry, then explicit keyword options,
    then the variable's SAS format decimals, then a global tolerance mapping
    or numeric tolerance.
    """
    global_mapping = None
    if isinstance(tolerance, Mapping):
        spec = tolerance.get(column)
        if spec is not None:
            return Tolerance.coerce(spec)
        keys = ('rtol', 'atol', 'rel_tol', 'abs_tol', 'decimals')
        if any(k in tolerance for k in keys):
            global_mapping = tolerance
    if options:
        return Tolerance.coerce(None, **options)
    variable = dataset[column]
    fmt = getattr(variable, 'format', None)
    decimals = getattr(fmt, 'decimals', None)
    if decimals is not None:
        return Tolerance(decimals=decimals)
    if global_mapping is not None:
        return Tolerance.coerce(global_mapping)
    if not isinstance(tolerance, Mapping):
        return Tolerance.coerce(tolerance)
    return None


def audit_dataset(dataset, tolerance=None, **kwds):
    """
    Audit every numeric value in a dataset.

    Original encodings are taken from the raw columns retained by the reader.
    A column without raw bytes is checked only for IBM encodability.  A
    mapping ``tolerance`` may provide a per-variable requirement, such as
    ``{'TEMP': {'decimals': 1}}``; otherwise a variable's format decimals
    define its field error requirement.  An explicit ``decimals`` argument
    applies to every variable.
    """
    raw_columns = _raw_columns(dataset)
    columns = _numeric_columns(dataset, raw_columns)
    options = dict(kwds)
    global_tolerance = None
    if options.get('decimals') is not None or options.get('rtol') is not None or options.get(
        'atol'
    ) is not None:
        global_tolerance = Tolerance.coerce(None, **options)
    elif tolerance is not None and not isinstance(tolerance, Mapping):
        global_tolerance = Tolerance.coerce(tolerance)
    results = []
    for column in columns:
        values = list(dataset[column])
        raws = raw_columns.get(column)
        if raws is not None:
            raws = _aligned_raws(raws, dataset.index)
        else:
            raws = [None] * len(values)
        tol = (
            global_tolerance if global_tolerance is not None else
            _column_tolerance(dataset, column, tolerance, options)
        )
        for index, (value, raw) in enumerate(zip(values, raws)):
            if raw is None:
                result = audit_float(value, tolerance=tol)
            else:
                result = audit_value(raw, value=value, tolerance=tol)
            result.column = column
            result.index = index
            results.append(result)
    return AuditReport(results, name=getattr(dataset, 'name', None), tolerance=tolerance)


def audit_library(library, tolerance=None, **kwds):
    """
    Audit every dataset in a library, returning a mapping of reports.
    """
    return {
        name: audit_dataset(dataset, tolerance=tolerance, **kwds)
        for name, dataset in library.items()
    }


def can_reencode(value, tolerance=None, **kwds):
    """
    Convenience predicate for a float, IBMFloat, or raw bytes value.
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        return audit_value(value, tolerance=tolerance, **kwds).reencodable
    if isinstance(value, xport.IBMFloat):
        return value.can_reencode(tolerance=tolerance, **kwds)
    return audit_float(value, tolerance=tolerance, **kwds).reencodable


def field_tolerance(fmt=None, decimals=None):
    """
    Build a ``Tolerance`` from a SAS format or explicit decimal places.
    """
    if fmt is not None and decimals is None:
        decimals = getattr(fmt, 'decimals', None)
    return Tolerance(decimals=decimals) if decimals is not None else None
