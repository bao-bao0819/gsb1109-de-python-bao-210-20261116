"""
Tests for auditing IBM floating point encodings against numeric views.
"""

# Standard Library
import math
from datetime import datetime

# Community Packages
import pytest

# Xport Modules
import xport
import xport.audit
import xport.v56


EXACT_ONE = bytes.fromhex('4110000000000000')
EXACT_TWO = bytes.fromhex('4120000000000000')
NEGATIVE_ZERO = bytes.fromhex('8000000000000000')
POSITIVE_ZERO = bytes.fromhex('0000000000000000')
MISSING = bytes.fromhex('2e00000000000000')
MISSING_A = bytes.fromhex('4100000000000000')
PRECISION_ONLY = bytes.fromhex('48ef0ee51f7637fb')
UNDERFLOW = bytes.fromhex('0001000000000000')


class TestAuditValue:
    """
    Verify per-value audit results.
    """

    def test_exact(self):
        result = xport.audit.audit_value(EXACT_ONE)
        assert result.status == xport.audit.EXACT
        assert result.byte_equal
        assert result.float_equal
        assert result.lossless
        assert result.reencodable
        assert result.encoded == EXACT_ONE
        assert result.absolute_error == 0

    def test_value_argument_is_used(self):
        result = xport.audit.audit_value(EXACT_ONE, value=1.0)
        assert result.byte_equal
        result = xport.audit.audit_value(EXACT_ONE, value=2.0)
        assert not result.byte_equal
        assert not result.reencodable

    def test_missing_dot(self):
        result = xport.audit.audit_value(MISSING)
        assert result.status == xport.audit.MISSING
        assert result.missing_tag == '.'
        assert result.encoded == MISSING
        assert result.reencodable
        assert not result.float_equal
        assert 'missing' in result.reason

    def test_special_missing_value(self):
        result = xport.audit.audit_value(MISSING_A)
        assert result.status == xport.audit.MISSING
        assert result.missing_tag == 'A'
        assert result.encoded == MISSING_A
        assert result.reencodable
        assert not result.numeric

    def test_special_missing_unknown_tag(self):
        result = xport.audit.audit_value(EXACT_ONE)
        assert result.status == xport.audit.EXACT
        assert result.missing_tag is None

    def test_negative_zero(self):
        result = xport.audit.audit_value(NEGATIVE_ZERO)
        assert result.status == xport.audit.NEGATIVE_ZERO
        assert result.byte_equal
        assert result.lossless
        assert result.reencodable
        assert 'sign' in result.reason

    def test_positive_zero(self):
        result = xport.audit.audit_value(POSITIVE_ZERO)
        assert result.status == xport.audit.EXACT
        assert result.byte_equal
        assert result.lossless

    def test_zero_sign_is_not_float_equality(self):
        positive = xport.audit.audit_value(POSITIVE_ZERO)
        negative = xport.audit.audit_value(NEGATIVE_ZERO)
        assert positive.value == negative.value
        assert positive.encoded != negative.encoded
        assert positive.status != negative.status

    def test_underflow(self):
        result = xport.audit.audit_value(UNDERFLOW)
        assert result.status == xport.audit.UNDERFLOW
        assert not result.reencodable
        assert not result.lossless
        assert result.encoded is None
        assert 'Underflow' in result.reason

    def test_precision_is_not_called_lossless(self):
        result = xport.audit.audit_value(PRECISION_ONLY)
        assert result.status == xport.audit.PRECISION_ONLY
        assert result.float_equal
        assert not result.byte_equal
        assert not result.lossless
        assert not result.reencodable
        assert 'Float equality alone is not lossless' in result.reason
        assert result.relative_error > 0

    def test_precision_within_relative_tolerance(self):
        result = xport.audit.audit_value(PRECISION_ONLY, tolerance={'rtol': 1e-15})
        assert result.reencodable
        assert not result.lossless
        assert result.status == xport.audit.PRECISION_ONLY

    def test_precision_outside_relative_tolerance(self):
        result = xport.audit.audit_value(PRECISION_ONLY, tolerance={'rtol': 1e-30})
        assert not result.reencodable

    def test_decimal_tolerance(self):
        result = xport.audit.audit_value(PRECISION_ONLY, decimals=6)
        assert result.reencodable
        result = xport.audit.audit_value(PRECISION_ONLY, decimals=12)
        assert not result.reencodable

    def test_field_tolerance_from_format(self):
        fmt = xport.Format.from_spec('8.1')
        tolerance = xport.audit.field_tolerance(fmt)
        assert tolerance.decimals == 1
        assert tolerance.atol == pytest.approx(0.05)

    def test_report_rows(self):
        result = xport.audit.audit_value(PRECISION_ONLY)
        row = result.as_dict()
        assert row['raw'] == PRECISION_ONLY.hex()
        assert row['status'] == xport.audit.PRECISION_ONLY
        assert row['byte_equal'] is False
        assert row['float_equal'] is True
        assert row['lossless'] is False


class TestAuditFloat:
    """
    Verify auditing values that have no original encoding.
    """

    def test_exact_roundtrip(self):
        result = xport.audit.audit_float(1.0)
        assert result.reencodable
        assert not result.lossless  # No source bytes to be lossless about.

    def test_underflow(self):
        result = xport.audit.audit_float(1e-100)
        assert result.status == xport.audit.UNDERFLOW
        assert not result.reencodable

    def test_overflow(self):
        result = xport.audit.audit_float(1e100)
        assert result.status == xport.audit.OVERFLOW
        assert not result.reencodable


class TestIBMToIEEE:
    """
    Verify numeric conversion and retained representations.
    """

    def test_raw_bytes(self):
        value = xport.v56.ibm_to_ieee(EXACT_ONE)
        assert isinstance(value, xport.IBMFloat)
        assert value == 1.0
        assert value.raw == EXACT_ONE
        assert value.encoded == EXACT_ONE
        assert value.audit().lossless

    def test_short_field_is_padded(self):
        value = xport.v56.ibm_to_ieee(b'\x41\x10')
        assert value == 1.0
        assert value.raw == b'\x41\x10'
        assert value.width == 2
        assert value.encoded == EXACT_ONE

    def test_negative_zero(self):
        value = xport.v56.ibm_to_ieee(NEGATIVE_ZERO)
        assert value == 0.0
        assert math.copysign(1.0, value) < 0
        assert xport.v56.ieee_to_ibm(value) == NEGATIVE_ZERO

    def test_alternative_zero(self):
        raw = bytes.fromhex('4000000000000000')
        value = xport.v56.ibm_to_ieee(raw)
        assert value == 0.0
        result = value.audit()
        assert result.status == xport.audit.ZERO
        assert not result.byte_equal

    def test_precision_loss_is_exposed(self):
        value = xport.v56.ibm_to_ieee(PRECISION_ONLY)
        assert value.reencode() != PRECISION_ONLY
        assert value.reencode(preserve_raw=True) == PRECISION_ONLY
        assert not value.can_reencode()

    def test_special_missing(self):
        value = xport.v56.ibm_to_ieee(MISSING_A)
        assert value is xport.NaN.A
        assert value.audit().reencodable


class TestEncoderRange:
    """
    Verify IBM range boundaries.
    """

    def test_overflow(self):
        with pytest.raises(xport.v56.Overflow):
            xport.v56.ieee_to_ibm(16**63)

    def test_underflow(self):
        with pytest.raises(xport.v56.Underflow):
            xport.v56.ieee_to_ibm(16**-66)

    def test_large_values_below_16_to_63(self):
        for value in [2.0**249, 2.0**250, 2.0**251, 1e75]:
            encoded = xport.v56.ieee_to_ibm(value)
            assert xport.v56.ibm_to_ieee(encoded) == value

    def test_roundtrip_preserves_value(self):
        for value in [0.0, -0.0, 1.0, -1.5, 98.6, 1e-75, 5.397605346934028e-79]:
            encoded = xport.v56.ieee_to_ibm(value)
            decoded = xport.v56.ibm_to_ieee(encoded)
            assert decoded == value
            assert math.copysign(1.0, decoded) == math.copysign(1.0, value)


@pytest.fixture(scope='module')
def dataset():
    """
    Dataset with exact values, precision loss, missing, and signed zero.
    """
    raw = EXACT_ONE + PRECISION_ONLY + MISSING_A + NEGATIVE_ZERO
    ds = xport.v56.Member({'X': [1.0, 1.0, 1.0, 1.0]}, name='AUDIT')
    ds.created = ds.modified = datetime(2020, 1, 1)
    # Retain the encodings just as the reader would.
    return ds, raw


class TestDatasetAudit:
    """
    Verify raw representations travel with a dataset read result.
    """

    def make_library(self):
        ds = xport.v56.Member({'X': [0.0, 1.0], 'TEXT': ['a', 'b']}, name='AUDIT')
        ds.created = ds.modified = datetime(2020, 1, 1)
        return xport.v56.Library([ds])

    def test_roundtrip_retains_raw(self):
        library = xport.v56.loads(xport.v56.dumps(self.make_library()))
        ds = library['AUDIT']
        assert list(ds.raw) == ['X']
        assert len(ds.raw['X']) == 2
        assert ds['X'].raw is not None

    def test_dataset_audit(self):
        library = xport.v56.loads(xport.v56.dumps(self.make_library()))
        ds = library['AUDIT']
        report = ds.audit()
        assert report.reencodable
        assert report.lossless
        assert report.counts == {'exact': 2}
        assert len(report.to_frame()) == 2

    def test_character_columns_are_skipped(self):
        library = xport.v56.loads(xport.v56.dumps(self.make_library()))
        report = library['AUDIT'].audit()
        assert all(result.column == 'X' for result in report)

    def test_audit_library(self):
        library = xport.v56.loads(xport.v56.dumps(self.make_library()))
        reports = xport.audit.audit_library(library)
        assert set(reports) == {'AUDIT'}
        assert reports['AUDIT'].reencodable

    def test_raw_follows_filtering(self):
        library = xport.v56.loads(xport.v56.dumps(self.make_library()))
        ds = library['AUDIT']
        filtered = ds[ds['X'] > 0]
        assert list(filtered.index) == [1]
        assert list(filtered.raw['X']) == [ds.raw['X'].iloc[1]]
        assert filtered.audit().counts == {'exact': 1}

    def test_format_decimals_are_field_requirement(self):
        raw = PRECISION_ONLY
        ds = xport.Dataset({
            'X': [float(xport.v56.ibm_to_ieee(raw))]
        }, name='AUDIT')
        object.__setattr__(ds, '_raw_columns', {'X': [raw]})
        ds['X'].format = '10.3'
        result, = ds.audit()
        assert result.reencodable
        assert not result.lossless
        assert result.tolerance.decimals == 3
        ds['X'].format = '10.15'
        result, = ds.audit()
        assert not result.reencodable

    def test_write_reuses_matching_raw(self):
        ds = xport.v56.Member({'X': [0.0, 1.0]}, name='AUDIT')
        ds.created = ds.modified = datetime(2020, 1, 1)
        xport.v56._set_raw_columns(ds, {
            'X': [NEGATIVE_ZERO, bytes.fromhex('4110000000000001')]
        })
        library = xport.v56.loads(xport.v56.dumps(xport.v56.Library([ds])))
        result = library['AUDIT'].audit()
        assert result.lossless
        assert result.counts == {
            'negative_zero': 1,
            'exact': 1,
        }

    def test_write_does_not_reuse_stale_raw(self):
        ds = xport.v56.Member({'X': [2.0]}, name='AUDIT')
        ds.created = ds.modified = datetime(2020, 1, 1)
        xport.v56._set_raw_columns(ds, {'X': [EXACT_ONE]})
        library = xport.v56.loads(xport.v56.dumps(xport.v56.Library([ds])))
        result, = library['AUDIT'].audit()
        assert result.value == 2.0
        assert result.raw == bytes.fromhex('4120000000000000')

    def test_write_preserves_missing_tag(self):
        ds = xport.v56.Member({'X': [xport.NaN.A, float('nan')]}, name='AUDIT')
        ds.created = ds.modified = datetime(2020, 1, 1)
        xport.v56._set_raw_columns(ds, {'X': [MISSING_A, MISSING]})
        library = xport.v56.loads(xport.v56.dumps(xport.v56.Library([ds])))
        result = library['AUDIT'].audit()
        assert result.lossless
        assert [r.raw for r in result] == [MISSING_A, MISSING]
