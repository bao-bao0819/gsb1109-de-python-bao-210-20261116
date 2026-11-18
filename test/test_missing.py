"""
Tests for preserving and analyzing SAS special missing values.
"""

# Standard Library
import math

# Community Packages
import pandas as pd
import pytest

# Xport Modules
import xport
import xport.v56
from xport import missing


@pytest.fixture
def dataset():
    """
    A dataset with real zeros, plain missings, and special missings.
    """
    return xport.Dataset(
        {
            'id': [1.0, 2.0, 3.0, 4.0, 5.0],
            'x': xport.Variable(
                [0.0, 1.5, xport.NaN.A, xport.NaN._, float('nan')],
                dtype='object',
            ),
        },
        name='TEST',
    )


class TestPreservation:
    """
    Verify special missing values survive loading and export.
    """

    def test_load_preserves_categories(self, dataset):
        library = xport.v56.loads(xport.v56.dumps(dataset))
        x = library['TEST']['x']
        assert x.iloc[0] == 0.0
        assert x.iloc[1] == 1.5
        assert x.iloc[2] is xport.NaN.A
        assert x.iloc[3] is xport.NaN._
        assert math.isnan(x.iloc[4]) and not missing.is_special_missing(x.iloc[4])

    def test_export_restores_categories(self, dataset):
        bytestring = xport.v56.dumps(dataset)
        assert b'A' + b'\x00' * 7 in bytestring
        assert b'_' + b'\x00' * 7 in bytestring
        assert b'.' + b'\x00' * 7 in bytestring
        library = xport.v56.loads(bytestring)
        got = [missing.representation(v) for v in library['TEST']['x']]
        assert got == [None, None, '.A', '._', '.']

    def test_zero_not_confused_with_missing(self, dataset):
        library = xport.v56.loads(xport.v56.dumps(dataset))
        x = library['TEST']['x']
        assert x.iloc[0] == 0.0
        assert not missing.is_missing(x.iloc[0])
        assert type(x.iloc[0]) is float

    def test_constructor_preserves_categories(self):
        v = xport.Variable([1.0, xport.NaN.A])
        assert v.dtype == object
        assert v.iloc[1] is xport.NaN.A
        ds = xport.Dataset({'x': [0.0, xport.NaN.B]})
        assert ds['x'].dtype == object
        assert ds['x'].iloc[1] is xport.NaN.B

    def test_none_encodes_as_plain_missing(self):
        assert xport.v56.ieee_to_ibm(None) == b'.' + b'\x00' * 7
        assert xport.v56.ieee_to_ibm(pd.NA) == b'.' + b'\x00' * 7


class TestRepresentation:
    """
    Verify translation to and from the original SAS text.
    """

    def test_representation(self):
        assert xport.NaN.A.representation == '.A'
        assert xport.NaN._.representation == '._'
        assert missing.representation(float('nan')) == '.'
        assert missing.representation(0.0) is None
        assert missing.representation('text') is None

    def test_from_representation(self):
        assert missing.from_representation('.A') is xport.NaN.A
        assert missing.from_representation('.z') is xport.NaN.Z
        assert missing.from_representation('._') is xport.NaN._
        assert math.isnan(missing.from_representation('.'))
        with pytest.raises(ValueError):
            missing.from_representation('nope')

    def test_tag(self):
        assert missing.tag(xport.NaN.B) == 'B'
        assert missing.tag(float('nan')) == ''
        assert missing.tag(0.0) is None

    def test_roundtrip(self):
        for c in '_' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            value = getattr(xport.NaN, c)
            assert missing.from_representation(missing.representation(value)) is value


class TestSummarize:
    """
    Verify statistics follow the caller's exclusion rules.
    """

    def test_exclude_all(self, dataset):
        stats = missing.summarize(dataset['x'], exclude='all')
        assert stats['count'] == 2
        assert stats['excluded'] == 3
        assert stats['missing'] == {}
        assert stats['n'] == 2
        assert stats['sum'] == 1.5
        assert stats['mean'] == 0.75
        assert stats['min'] == 0.0
        assert stats['max'] == 1.5

    def test_exclude_selected_categories(self, dataset):
        stats = missing.summarize(dataset['x'], exclude={'A'})
        assert stats['excluded'] == 1
        assert stats['count'] == 4
        assert stats['missing'] == {'._': 1, '.': 1}
        assert stats['n'] == 2
        stats = missing.summarize(dataset['x'], exclude=['.A', '._'])
        assert stats['excluded'] == 2
        assert stats['missing'] == {'.': 1}

    def test_exclude_nothing(self, dataset):
        stats = missing.summarize(dataset['x'], exclude=None)
        assert stats['excluded'] == 0
        assert stats['count'] == 5
        assert stats['missing'] == {'.A': 1, '._': 1, '.': 1}

    def test_exclude_special(self, dataset):
        stats = missing.summarize(dataset['x'], exclude='special')
        assert stats['excluded'] == 2
        assert stats['missing'] == {'.': 1}

    def test_exclude_callable(self, dataset):
        stats = missing.summarize(
            dataset['x'],
            exclude=lambda v: missing.representation(v) == '._',
        )
        assert stats['excluded'] == 1
        assert stats['missing'] == {'.A': 1, '.': 1}

    def test_dataset(self, dataset):
        stats = missing.summarize(dataset, exclude='all')
        assert isinstance(stats, pd.DataFrame)
        assert stats.loc['x', 'excluded'] == 3
        assert stats.loc['id', 'n'] == 5


class TestKeys:
    """
    Verify missing categories never share keys with real values.
    """

    def test_keyify(self):
        assert missing.keyify(0.0) != missing.keyify(xport.NaN.A)
        assert missing.keyify(0.0) != missing.keyify(float('nan'))
        assert missing.keyify(xport.NaN.A) != missing.keyify(xport.NaN.B)
        assert missing.keyify(xport.NaN.A) != missing.keyify(float('nan'))
        assert missing.keyify(0.0) == missing.keyify(0.0)
        assert len({missing.keyify(v) for v in [0.0, float('nan'), xport.NaN.A, xport.NaN._]}) == 4

    def test_groupby(self, dataset):
        groups = {key: len(group) for key, group in missing.groupby(dataset, 'x')}
        assert groups == {
            (0, 0.0): 1,
            (0, 1.5): 1,
            (1, '.'): 1,
            (1, '.A'): 1,
            (1, '._'): 1,
        }

    def test_groupby_zero_not_merged_with_missing(self):
        ds = xport.Dataset({'x': [0.0, xport.NaN.A, float('nan'), 0.0]})
        groups = {key: group for key, group in missing.groupby(ds, 'x')}
        assert len(groups[(0, 0.0)]) == 2
        assert len(groups[(1, '.A')]) == 1
        assert len(groups[(1, '.')]) == 1

    def test_join(self):
        left = xport.Dataset({
            'k': xport.Variable([0.0, xport.NaN.A, float('nan')], dtype='object'),
            'v': ['a', 'b', 'c'],
        })
        right = xport.Dataset({
            'k': xport.Variable([xport.NaN.A, 0.0, float('nan')], dtype='object'),
            'w': ['x', 'y', 'z'],
        })
        merged = missing.join(left, right, on='k')
        assert list(merged['v']) == ['a', 'b', 'c']
        assert list(merged['w']) == ['y', 'x', 'z']

    def test_join_missing_never_matches_zero(self):
        left = xport.Dataset({'k': xport.Variable([xport.NaN.A], dtype='object'), 'v': ['a']})
        right = xport.Dataset({'k': [0.0], 'w': ['b']})
        merged = missing.join(left, right, on='k')
        assert len(merged) == 0

    def test_reshard(self, dataset):
        shards = missing.reshard(dataset, 'x')
        assert set(shards) == {(0, 0.0), (0, 1.5), (1, '.'), (1, '.A'), (1, '._')}
        assert sum(len(shard) for shard in shards.values()) == len(dataset)
        assert shards[(1, '.A')]['x'].iloc[0] is xport.NaN.A
        assert shards[(0, 0.0)]['x'].iloc[0] == 0.0
