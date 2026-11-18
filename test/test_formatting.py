"""
Tests for SAS numeric formatting and precision recovery.
"""

# Standard Library
from decimal import Decimal

# Community Packages
import pytest

# Xport Modules
import xport
import xport.formatting


class TestFormatSpec:
    """Parse format strings."""

    def test_parse(self):
        spec = xport.FormatSpec(name='BEST', width=12, decimals=4)
        assert str(spec) == 'BEST12.4'

    def test_parse_string(self):
        assert xport.formatting.coerce_format('E12.4') == xport.FormatSpec('E', 12, 4)
        assert xport.formatting.coerce_format('BEST12.') == xport.FormatSpec('BEST', 12, None)
        assert xport.formatting.coerce_format('F0.') == xport.FormatSpec('F', 0, None)

    def test_parse_object(self):
        fmt = xport.Format.from_spec('BEST12.')
        spec = xport.formatting.coerce_format(fmt)
        assert spec.name == 'BEST'
        assert spec.width == 12

    def test_invalid(self):
        with pytest.raises(ValueError):
            xport.formatting.coerce_format('not a format')


class TestFormatValue:
    """Render numeric values the way a SAS format would."""

    def test_fixed(self):
        assert xport.format_value(1234.5, 'F10.1') == '1234.5'
        assert xport.format_value(1234.5, 'F10.0') == '1235'
        assert xport.format_value(-1234.5, 'F10.0') == '-1235'

    def test_scientific(self):
        assert xport.format_value(1234567890123, 'E12.4') == '1.2346E+12'
        assert xport.format_value(0.000012345, 'E12.4') == '1.2345E-05'

    def test_best_integer(self):
        assert xport.format_value(1234567890, 'BEST12.') == '1234567890'

    def test_best_large(self):
        text = xport.format_value(1234567890123, 'BEST12.')
        assert text == '1.234568E+12'

    def test_best_format_object(self):
        """A parsed ``Format`` stores an omitted decimals as zero."""
        fmt = xport.Format.from_spec('BEST12.')
        assert xport.format_value(1234567890123, fmt) == '1.234568E+12'

    def test_best_zero_width(self):
        """``BEST0.`` means no width limit, not a one-character field."""
        assert xport.format_value(1234567890123, 'BEST0.') == '1234567890123'

    def test_best_explicit_decimals(self):
        assert xport.format_value(1234567890123, 'BEST12.4') == '1.2346E+12'

    def test_overflow(self):
        assert xport.format_value(1234567890123, 'F12.0') == '************'

    def test_missing(self):
        assert xport.format_value(None, 'BEST12.') == '.'
        assert xport.format_value(xport.NaN.A, 'BEST12.') == '.A'

    def test_character(self):
        assert xport.format_value('abc', '$3.') == 'abc'


class TestPrecisionInterval:
    """Recover the set of values a display could represent."""

    def test_integer_display(self):
        interval = xport.precision_interval('1234567890')
        assert interval.low == Decimal('1234567889.5')
        assert interval.high == Decimal('1234567890.5')
        assert not interval.high_inclusive

    def test_scientific_display(self):
        interval = xport.precision_interval('1.23457E+12')
        assert interval.low == Decimal('1234565000000')
        assert interval.high == Decimal('1234575000000')

    def test_fixed_decimals(self):
        interval = xport.precision_interval('1.23')
        assert interval.low == Decimal('1.225')
        assert interval.high == Decimal('1.235')

    def test_negative(self):
        interval = xport.precision_interval('-1.5')
        assert interval.low == Decimal('-1.55')
        assert interval.high == Decimal('-1.45')

    def test_missing(self):
        assert xport.precision_interval('.') is None
        assert xport.precision_interval('.A') is None
        assert xport.precision_interval('****') is None

    def test_contains(self):
        interval = xport.precision_interval('1.23457E+12')
        assert interval.contains(1234567890124)
        assert not interval.contains(1234575000000)
        assert interval.contains(1234565000000)

    def test_intersection(self):
        left = xport.precision_interval('1.2346E+12')
        right = xport.precision_interval('1.23457E+12')
        overlap = left & right
        assert overlap.low == Decimal('1234565000000')
        assert overlap.high == Decimal('1234575000000')

    def test_disjoint(self):
        left = xport.precision_interval('1.0')
        right = xport.precision_interval('2.0')
        assert left & right is None

    def test_point(self):
        interval = xport.PrecisionInterval.point(1234567890123)
        assert interval.is_point
        assert interval.contains(1234567890123)

    def test_display_roundtrip_contains_value(self):
        format_value = xport.format_value
        interval = xport.precision_interval(format_value(1234567890124, 'BEST12.'))
        assert interval.contains(1234567890124)

    def test_format_object_roundtrip(self):
        fmt = xport.Format.from_spec('BEST12.')
        display = xport.format_value(1234567890124, fmt)
        assert xport.precision_interval(display, fmt).contains(1234567890124)


class TestValueInterval:
    """Retained numeric values are exact evidence."""

    def test_exact_by_default(self):
        interval = xport.value_interval(1234567890123, 'F12.0')
        assert interval.is_point

    def test_rounded_display(self):
        interval = xport.value_interval(1234567890123, 'F12.0', exact=False)
        assert not interval.is_point
        assert interval.low == Decimal('1234567890122.5')
