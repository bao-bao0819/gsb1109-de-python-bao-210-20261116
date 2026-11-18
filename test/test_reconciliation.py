"""
Tests for reconciling case numbers across display and value evidence.
"""

# Standard Library
from decimal import Decimal

# Community Packages
import pandas as pd
import pytest

# Xport Modules
import xport


def record(source, record_id, display=None, value=None, format=None, batch=None):
    """Build one input record."""
    return {
        'source': source,
        'record_id': record_id,
        'display': display,
        'value': value,
        'format': format,
        'batch': batch,
    }


class TestDisplayGroups:
    """Same text is a collision until identity is confirmed."""

    def test_same_display_different_sources_not_merged(self):
        records = [
            record('A', 1, display='1.23457E+12'),
            record('B', 1, display='1.23457E+12'),
        ]
        result = xport.reconcile(records)
        assert len(result.clusters) == 2
        assert len(result.groups) == 1
        assert result.groups[0].is_ambiguous
        assert not result.unique
        assert result.possible

    def test_same_display_same_source_stays_separate(self):
        records = [
            record('A', 1, display='1.23457E+12'),
            record('A', 2, display='1.23457E+12'),
        ]
        result = xport.reconcile(records)
        assert len(result.clusters) == 2
        assert not result.unique
        assert not result.possible

    def test_same_source_different_batches_are_compared(self):
        records = [
            record('H', 1, display='1.23457E+12', batch='2021'),
            record('H', 2, display='1.23457E+12', batch='2022'),
        ]
        result = xport.reconcile(records)
        assert result.possible

    def test_same_source_same_batch_is_not_a_match(self):
        records = [
            record('H', 1, display='1.23457E+12', batch='2021'),
            record('H', 2, display='1.23457E+12', batch='2021'),
        ]
        result = xport.reconcile(records)
        assert result.groups[0].is_ambiguous
        assert not result.correspondences

    def test_confirmed_link_merges_identity(self):
        records = [
            record('A', 1, value=1234567890124),
            record('B', 'x', display='1.23457E+12'),
        ]
        links = [{'left': ('A', 1), 'right': ('B', 'x'), 'confirmed': True}]
        result = xport.reconcile(records, links=links)
        assert len(result.clusters) == 1
        cluster = result.clusters[0]
        assert set(cluster.sources) == {'A', 'B'}
        assert cluster.interval.is_point

    def test_unconfirmed_link_does_not_merge(self):
        records = [
            record('A', 1, value=1234567890124),
            record('B', 'x', value=1234567890124),
        ]
        links = [{'left': ('A', 1), 'right': ('B', 'x'), 'confirmed': False}]
        result = xport.reconcile(records, links=links)
        assert len(result.clusters) == 2

    def test_link_with_ambiguous_bare_id_is_reported(self):
        records = [
            record('A', 1, value=1),
            record('B', 1, value=1),
        ]
        links = [{'left': 1, 'right': ('A', 1), 'confirmed': True}]
        result = xport.reconcile(records, links=links)
        assert any('ambiguous' in warning for warning in result.warnings)


class TestCorrespondences:
    """Unique recovery requires an exact value and no rival candidate."""

    def test_identical_display_only_is_possible_not_unique(self):
        records = [
            record('A', 1, display='1.23457E+12'),
            record('B', 'x', display='1.23457E+12'),
        ]
        result = xport.reconcile(records)
        assert not result.unique
        assert len(result.possible) == 1

    def test_exact_value_recovers_unique_connection(self):
        records = [
            record('A', 1, display='1.23457E+12', format='E12.5'),
            record('B', 'x', display='1.23457E+12', format='E12.5'),
        ]
        original = xport.reconcile(records, formats={'A': 'E12.5', 'B': 'E12.5'})
        assert not original.unique

        narrowed = original.with_original('B', 'x', 1234567890124, 'E12.5')
        assert len(narrowed.unique) == 1
        correspondence = narrowed.unique[0]
        assert correspondence.interval.is_point
        assert correspondence.interval.low == Decimal('1234567890124')

    def test_value_outside_precision_is_rejected(self):
        records = [
            record('A', 1, display='1.23457E+12', format='E12.5'),
            record('B', 'x', display='1.23457E+12', format='E12.5'),
        ]
        result = xport.reconcile(records, formats={'A': 'E12.5', 'B': 'E12.5'})
        with pytest.raises(ValueError):
            result.with_original('B', 'x', 1234560000000, 'E12.5')

    def test_display_only_does_not_gain_precision(self):
        records = [
            record('A', 1, display='1.23457E+12', format='E12.5'),
            record('B', 'x', display='1.23457E+12', format='E12.5'),
        ]
        result = xport.reconcile(records, formats={'A': 'E12.5', 'B': 'E12.5'})
        narrowed = result.with_original('B', 'x', 1234567890124, 'E12.5')
        assert narrowed.unique[0].interval.is_point
        assert narrowed.unique[0].interval.low == Decimal('1234567890124')

    def test_two_values_two_display_records(self):
        records = [
            record('A', 1, display='1.23457E+12', format='E12.5'),
            record('A', 2, display='1.23457E+12', format='E12.5'),
            record('B', 'x', value=1234567890124),
            record('B', 'y', value=1234567890999),
        ]
        result = xport.reconcile(records, formats={'A': 'E12.5', 'B': 'BEST12.'})
        assert not result.unique
        assert result.possible

    def test_value_and_format_recovers_display(self):
        records = [
            record('A', 1, value=1234567890124, format='BEST12.'),
            record('B', 'x', value=1234567890124, format='BEST12.'),
        ]
        result = xport.reconcile(records, formats={'A': 'BEST12.', 'B': 'BEST12.'})
        assert len(result.unique) == 1
        assert result.unique[0].interval.is_point

    def test_different_values_do_not_match(self):
        records = [
            record('A', 1, value=1234567890124, format='BEST12.'),
            record('B', 'x', value=9999999999999, format='BEST12.'),
        ]
        result = xport.reconcile(records, formats={'A': 'BEST12.', 'B': 'BEST12.'})
        assert not result.correspondences


class TestPrecisionIntervals:
    """Every recovered connection carries its precision."""

    def test_possible_match_has_interval(self):
        records = [
            record('A', 1, display='1.23457E+12'),
            record('B', 'x', display='1.23457E+12'),
        ]
        result = xport.reconcile(records)
        correspondence = result.possible[0]
        assert correspondence.interval.low == Decimal('1234565000000')
        assert correspondence.interval.high == Decimal('1234575000000')

    def test_confirmed_link_intersects_intervals(self):
        records = [
            record('A', 1, value=1234567890124),
            record('B', 'x', display='1.23457E+12'),
        ]
        links = [{'left': ('A', 1), 'right': ('B', 'x')}]
        result = xport.reconcile(records, links=links)
        assert result.clusters[0].interval.is_point
        assert result.clusters[0].interval.low == Decimal('1234567890124')

    def test_conflicting_link_reports_conflict(self):
        records = [
            record('A', 1, value=1234567890124),
            record('B', 'x', value=9999999999999),
        ]
        links = [{'left': ('A', 1), 'right': ('B', 'x')}]
        result = xport.reconcile(records, links=links)
        reasons = {conflict.reason for conflict in result.conflicts}
        assert 'Identity retains conflicting values' in reasons
        assert result.clusters[0].interval is None


class TestWithOriginal:
    """Supplying one number narrows only evidenced candidates."""

    def test_unrelated_value_does_not_create_matches(self):
        records = [
            record('A', 1, display='1.23457E+12'),
            record('B', 'x', display='1.23457E+12'),
            record('C', 'q', display='9.87654E+12'),
        ]
        result = xport.reconcile(records)
        narrowed = result.with_original('C', 'q', 9876543210987)
        labels = {(c.left.label, c.right.label) for c in narrowed.unique}
        assert ('A:1', 'B:x') not in labels

    def test_missing_record_raises(self):
        records = [record('A', 1, display='1')]
        result = xport.reconcile(records)
        try:
            result.with_original('Z', 9, 1)
        except KeyError:
            pass
        else:
            raise AssertionError('expected KeyError')


class TestInputForms:
    """Records may be mappings, tuples, or a DataFrame."""

    def test_tuple_input(self):
        records = [
            ('A', 1, '1.23457E+12'),
            ('B', 'x', '1.23457E+12'),
        ]
        result = xport.reconcile(records)
        assert len(result.records) == 2

    def test_dataframe_input(self):
        frame = pd.DataFrame([
            {'source': 'A', 'record_id': 1, 'display': '1.2'},
            {'source': 'B', 'record_id': 2, 'display': '1.2'},
        ])
        result = xport.reconcile(frame)
        assert len(result.records) == 2

    def test_batch_format_lookup(self):
        records = [
            record('A', 1, display='1.23457E+12', batch='2021'),
            record('B', 'x', display='1.23457E+12', batch='2022'),
        ]
        result = xport.reconcile(records, formats={'2021': 'E12.5', '2022': 'E12.5'})
        assert result.records[0].format.name == 'E'

    def test_as_dict(self):
        records = [
            record('A', 1, display='1.23457E+12'),
            record('B', 'x', display='1.23457E+12'),
        ]
        payload = xport.reconcile(records).as_dict()
        assert set(payload) >= {'records', 'clusters', 'groups', 'unique', 'possible'}

    def test_summary_is_readable(self):
        records = [
            record('A', 1, display='1.23457E+12'),
            record('B', 'x', display='1.23457E+12'),
        ]
        text = xport.reconcile(records).summary()
        assert 'collision' in text
        assert 'possible correspondences: 1' in text
