"""
Tests for row-and-column views and field-identity merging.
"""

# Standard Library
import math
from datetime import datetime

# Community Packages
import pandas as pd
import pytest

# Xport Modules
import xport
from xport.views import (
    METADATA_KEY,
    Conflict,
    Contribution,
    MergeResult,
    derive_view,
    merge_views,
    set_metadata,
    sync_metadata,
)


def make_dataset():
    """Build a dataset with full variable metadata and tagged missing values."""
    ds = xport.Dataset(
        {
            'n': [1, 2, 3],
            'c': ['a', 'b', 'c'],
            'm': pd.Series(
                [xport.NaN.A, xport.NaN.B, float('nan')], dtype=object
            ),
        },
        name='SOURCE',
        label='Source dataset',
    )
    ds['n'].label = 'Number'
    ds['n'].width = 8
    ds['n'].vtype = xport.VariableType.NUMERIC
    ds['n'].format = 'comma8.0'
    ds['n'].informat = '8.0'
    ds['c'].label = 'Letter'
    ds['c'].width = 4
    ds['c'].vtype = xport.VariableType.CHARACTER
    ds['c'].format = xport.Format(
        '$CHAR', 4, 0, xport.FormatAlignment.RIGHT
    )
    return ds


class TestDeriveView:
    """Verify view selection and metadata inheritance."""

    def test_default_includes_everything(self):
        ds = make_dataset()
        view = derive_view(ds)
        assert list(view.columns) == ['n', 'c', 'm']
        assert list(view.index) == [0, 1, 2]
        assert view.name == 'SOURCE'

    def test_row_and_column_selection(self):
        ds = make_dataset()
        view = derive_view(ds, rows=[0, 2], columns=['n', 'm'], name='V')
        assert list(view.columns) == ['n', 'm']
        assert list(view.index) == [0, 2]
        assert view.name == 'V'
        assert list(view['n']) == [1, 3]

    def test_scalar_selectors(self):
        ds = make_dataset()
        view = derive_view(ds, rows=0, columns='n')
        assert list(view.columns) == ['n']
        assert list(view['n']) == [1]

    def test_inherits_variable_metadata(self):
        ds = make_dataset()
        view = derive_view(ds, columns=['n', 'c'])
        for aspect in ('label', 'width', 'vtype', 'format', 'informat'):
            assert getattr(view['n'], aspect) == getattr(ds['n'], aspect)
            assert getattr(view['c'], aspect) == getattr(ds['c'], aspect)

    def test_inherits_special_missing_strategy(self):
        ds = make_dataset()
        view = derive_view(ds, columns=['m'])
        assert view['m'].iloc[0] is xport.NaN.A
        assert view['m'].iloc[1] is xport.NaN.B
        assert math.isnan(view['m'].iloc[2])
        assert not isinstance(view['m'].iloc[2], xport.SpecialMissingValue)

    def test_unknown_columns_and_rows_raise(self):
        ds = make_dataset()
        with pytest.raises(KeyError):
            derive_view(ds, columns=['absent'])
        with pytest.raises(KeyError):
            derive_view(ds, rows=[99])

    def test_is_a_dataset(self):
        ds = make_dataset()
        assert isinstance(derive_view(ds), xport.Dataset)

    def test_accepts_plain_dataframe(self):
        frame = pd.DataFrame({'a': [1, 2, 3], 's': ['x', 'y', 'z']})
        view = derive_view(frame, rows=[0, 1], columns=['a', 's'])
        assert isinstance(view, xport.Dataset)
        view.loc[0, 'a'] = 10
        result = merge_views(frame, view)
        assert result.ok
        assert list(result.merged['a']) == [10, 2, 3]
        assert list(frame['a']) == [1, 2, 3]


class TestViewIndependence:
    """Verify views and source cannot affect one another."""

    def test_value_changes_are_independent(self):
        ds = make_dataset()
        view = derive_view(ds, columns=['n'])
        view.loc[0, 'n'] = 99
        assert ds['n'].iloc[0] == 1
        ds.loc[1, 'n'] = 77
        assert view['n'].iloc[1] == 2

    def test_metadata_objects_are_not_shared(self):
        ds = make_dataset()
        view = derive_view(ds, columns=['n'])
        set_metadata(view, 'n', 'format', xport.Format.from_spec('dollar10.2'))
        assert str(ds['n'].format) == 'COMMA8.'
        assert view['n'].format is not ds['n'].format

    def test_later_source_metadata_change_leaves_view(self):
        ds = make_dataset()
        view = derive_view(ds, columns=['n'])
        ds['n'].label = 'Changed on source'
        assert view.attrs[METADATA_KEY]['n']['label'] == 'Number'


class TestMergeValues:
    """Verify value contributions and value conflicts."""

    def test_no_change_is_noop(self):
        ds = make_dataset()
        result = merge_views(ds, derive_view(ds))
        assert result.ok
        assert result.compatible == ()
        assert result.conflicts == ()
        assert list(result.merged['n']) == [1, 2, 3]

    def test_single_branch_value_change_applies(self):
        ds = make_dataset()
        view = derive_view(ds, rows=[1], columns=['n'])
        view.loc[1, 'n'] = 50
        result = merge_views(ds, view)
        assert result.ok
        assert list(result.merged['n']) == [1, 50, 3]
        contribution = result.compatible[0]
        assert contribution.branch == 0
        assert contribution.column == 'n'
        assert contribution.values == {(1, 'n'): 50}

    def test_disjoint_cells_both_apply(self):
        ds = make_dataset()
        first = derive_view(ds, rows=[0], columns=['n'])
        second = derive_view(ds, rows=[2], columns=['n'])
        first.loc[0, 'n'] = 10
        second.loc[2, 'n'] = 30
        result = merge_views(ds, first, second)
        assert result.ok
        assert list(result.merged['n']) == [10, 2, 30]

    def test_merges_by_row_label_identity(self):
        ds = xport.Dataset(
            {'n': [10, 20, 30]},
            index=pd.Index(['r0', 'r1', 'r2'], name='obs'),
        )
        view = derive_view(ds, rows=['r2', 'r0'])
        view.loc['r2', 'n'] = 300
        result = merge_views(ds, view)
        assert list(result.merged['n']) == [10, 20, 300]
        assert result.merged.index.tolist() == ['r0', 'r1', 'r2']

    def test_identical_change_is_not_conflict(self):
        ds = make_dataset()
        first = derive_view(ds, columns=['n'])
        second = derive_view(ds, columns=['n'])
        first.loc[0, 'n'] = 5
        second.loc[0, 'n'] = 5
        result = merge_views(ds, first, second)
        assert result.ok
        assert result.merged['n'].iloc[0] == 5

    def test_distinct_cell_changes_conflict(self):
        ds = make_dataset()
        first = derive_view(ds, columns=['n'])
        second = derive_view(ds, columns=['n'])
        first.loc[0, 'n'] = 10
        second.loc[0, 'n'] = 20
        result = merge_views(ds, first, second)
        assert not result.ok
        conflict = result.conflicts[0]
        assert conflict.kind == 'value'
        assert conflict.column == 'n'
        assert conflict.rows == (0,)
        assert conflict.branches == (0, 1)
        # Conflicting cell stays at the source value.
        assert result.merged['n'].iloc[0] == 1
        # Both raw contributions survive, but neither counts as compatible.
        assert len(result.originals) == 2
        assert result.compatible == ()

    def test_special_missing_tags_conflict_distinctly(self):
        ds = make_dataset()
        first = derive_view(ds, columns=['m'])
        second = derive_view(ds, columns=['m'])
        first.loc[0, 'm'] = xport.NaN.X
        second.loc[0, 'm'] = xport.NaN.Y
        result = merge_views(ds, first, second)
        assert len(result.conflicts) == 1
        assert result.merged['m'].iloc[0] is xport.NaN.A

    def test_special_missing_value_applies(self):
        ds = make_dataset()
        view = derive_view(ds, columns=['m'])
        view.loc[2, 'm'] = xport.NaN.Z
        result = merge_views(ds, view)
        assert result.ok
        assert result.merged['m'].iloc[2] is xport.NaN.Z


class TestMergeMetadata:
    """Verify metadata contributions and metadata conflicts."""

    def test_single_branch_metadata_change_applies(self):
        ds = make_dataset()
        view = derive_view(ds, columns=['n'])
        set_metadata(view, 'n', 'label', 'Renamed')
        result = merge_views(ds, view)
        assert result.ok
        assert result.merged['n'].label == 'Renamed'
        assert result.compatible[0].metadata == {'label': 'Renamed'}

    def test_identical_metadata_change_is_not_conflict(self):
        ds = make_dataset()
        first = derive_view(ds, columns=['n'])
        second = derive_view(ds, columns=['n'])
        set_metadata(first, 'n', 'width', 10)
        set_metadata(second, 'n', 'width', 10)
        result = merge_views(ds, first, second)
        assert result.ok
        assert result.merged['n'].width == 10

    def test_distinct_metadata_changes_conflict(self):
        ds = make_dataset()
        first = derive_view(ds, columns=['n'])
        second = derive_view(ds, columns=['n'])
        set_metadata(first, 'n', 'width', 10)
        set_metadata(second, 'n', 'width', 20)
        result = merge_views(ds, first, second)
        conflict = result.conflicts[0]
        assert conflict.kind == 'metadata'
        assert conflict.column == 'n'
        assert conflict.aspect == 'width'
        # Baseline retained for the conflicting aspect.
        assert result.merged['n'].width == 8

    def test_value_and_format_changes_are_complementary(self):
        ds = make_dataset()
        first = derive_view(ds, columns=['n'])
        second = derive_view(ds, columns=['n'])
        first.loc[0, 'n'] = 10
        set_metadata(second, 'n', 'format', 'dollar10.2')
        result = merge_views(ds, first, second)
        assert result.ok
        assert result.merged['n'].iloc[0] == 10
        assert str(result.merged['n'].format) == 'DOLLAR10.2'
        aspects = {
            aspect
            for contribution in result.compatible
            for aspect in contribution.metadata
        }
        assert 'format' in aspects

    def test_distinct_labels_conflict_but_other_aspect_applies(self):
        ds = make_dataset()
        first = derive_view(ds, columns=['n'])
        second = derive_view(ds, columns=['n'])
        set_metadata(first, 'n', 'label', 'From first')
        set_metadata(second, 'n', 'label', 'From second')
        set_metadata(second, 'n', 'width', 12)
        result = merge_views(ds, first, second)
        assert result.merged['n'].label == 'Number'
        assert result.merged['n'].width == 12

    def test_clearing_metadata_applies(self):
        ds = make_dataset()
        view = derive_view(ds, columns=['n'])
        set_metadata(view, 'n', 'label', None)
        result = merge_views(ds, view)
        assert result.ok
        assert result.merged['n'].label is None
        assert ds['n'].label == 'Number'

    def test_direct_variable_attribute_edit_is_honored(self):
        ds = make_dataset()
        view = derive_view(ds, columns=['n'])
        view['n'].label = 'Live edit'
        result = merge_views(ds, view)
        assert result.ok
        assert result.merged['n'].label == 'Live edit'

    def test_clearing_versus_setting_same_aspect_conflicts(self):
        ds = make_dataset()
        first = derive_view(ds, columns=['n'])
        second = derive_view(ds, columns=['n'])
        set_metadata(first, 'n', 'label', None)
        set_metadata(second, 'n', 'label', 'New')
        result = merge_views(ds, first, second)
        assert result.conflicts[0].aspect == 'label'
        assert result.merged['n'].label == 'Number'


class TestMergeIsolation:
    """Verify the source and idle branches stay untouched."""

    def test_source_values_and_metadata_unchanged(self):
        ds = make_dataset()
        view = derive_view(ds, columns=['n'])
        view.loc[0, 'n'] = 10
        set_metadata(view, 'n', 'label', 'View label')
        merge_views(ds, view)
        assert list(ds['n']) == [1, 2, 3]
        assert ds['n'].label == 'Number'
        assert str(ds['n'].format) == 'COMMA8.'

    def test_merged_writes_do_not_reach_source(self):
        ds = make_dataset()
        result = merge_views(ds, derive_view(ds, columns=['n']))
        result.merged.loc[0, 'n'] = 999
        assert ds['n'].iloc[0] == 1

    def test_unparticipated_branch_unchanged(self):
        ds = make_dataset()
        active = derive_view(ds, columns=['n'])
        idle = derive_view(ds)
        active.loc[0, 'n'] = 7
        merge_views(ds, active, idle)
        assert list(idle['n']) == [1, 2, 3]
        assert idle['c'].label == 'Letter'

    def test_unparticipated_variables_pass_through(self):
        ds = make_dataset()
        active = derive_view(ds, columns=['n'])
        active.loc[0, 'n'] = 7
        result = merge_views(ds, active)
        assert list(result.merged['c']) == ['a', 'b', 'c']
        assert result.merged['c'].label == 'Letter'
        assert result.merged['c'].vtype == xport.VariableType.CHARACTER

    def test_views_unchanged_by_merge(self):
        ds = make_dataset()
        view = derive_view(ds, columns=['n'])
        view.loc[0, 'n'] = 10
        merge_views(ds, view)
        assert list(view['n']) == [10, 2, 3]

    def test_dataset_level_metadata_passes_through(self):
        ds = make_dataset()
        ds.created = ds.modified = datetime(2020, 1, 1)
        ds.sas_os = 'LINUX'
        ds.sas_version = '9.4'
        view = derive_view(ds, columns=['n'])
        view.loc[0, 'n'] = 10
        merged = merge_views(ds, view).merged
        assert merged.name == 'SOURCE'
        assert merged.dataset_label == 'Source dataset'
        assert merged.created == datetime(2020, 1, 1)
        assert merged.sas_os == 'LINUX'
        assert merged.sas_version == '9.4'
        assert ds.sas_os == 'LINUX'


class TestResultObjects:
    """Verify result containers and helper functions."""

    def test_merge_result_reports_counts(self):
        ds = make_dataset()
        view = derive_view(ds, columns=['n'])
        view.loc[0, 'n'] = 10
        result = merge_views(ds, view)
        assert isinstance(result, MergeResult)
        assert isinstance(result.merged, xport.Dataset)
        assert result.ok
        assert repr(result) == 'MergeResult(compatible=1, conflicts=0)'

    def test_contribution_equality(self):
        left = Contribution(0, 'n', cells=((0, 'n'),), values={(0, 'n'): 1})
        right = Contribution(0, 'n', cells=((0, 'n'),), values={(0, 'n'): 1})
        different = Contribution(1, 'n', cells=((0, 'n'),), values={(0, 'n'): 2})
        assert left == right
        assert left != different
        assert left != 'not a contribution'

    def test_conflict_equality(self):
        left = Conflict('value', 'n', (0, 1), {0: 1, 1: 2}, rows=(0,))
        right = Conflict('value', 'n', (0, 1), {0: 1, 1: 2}, rows=(0,))
        assert left == right
        assert left != Conflict('metadata', 'n', (0, 1), {}, aspect='label')

    def test_originals_keep_conflicting_contributions(self):
        ds = make_dataset()
        first = derive_view(ds, columns=['n'])
        second = derive_view(ds, columns=['n'])
        first.loc[0, 'n'] = 10
        second.loc[0, 'n'] = 20
        result = merge_views(ds, first, second)
        assert len(result.originals) == 2
        assert {c.branch for c in result.originals} == {0, 1}
        assert all(isinstance(c, Contribution) for c in result.originals)

    def test_set_metadata_validates_inputs(self):
        ds = make_dataset()
        view = derive_view(ds, columns=['n'])
        with pytest.raises(KeyError):
            set_metadata(view, 'n', 'unknown_aspect', 1)
        with pytest.raises(KeyError):
            set_metadata(view, 'absent', 'label', 'x')

    def test_sync_metadata_restores_cached_variables(self):
        ds = make_dataset()
        view = derive_view(ds, columns=['n'])
        view.loc[0, 'n'] = 9  # Pandas rebuilds the cached Variable.
        sync_metadata(view)
        assert view['n'].label == 'Number'
        assert str(view['n'].format) == 'COMMA8.'

    def test_merge_rejects_fields_absent_from_source(self):
        ds = make_dataset()
        view = derive_view(ds, columns=['n'])
        view['extra'] = [1, 2, 3]
        with pytest.raises(KeyError):
            merge_views(ds, view)
