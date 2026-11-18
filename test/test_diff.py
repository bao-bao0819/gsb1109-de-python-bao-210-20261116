"""
Tests for record views and field-identity comparison.
"""

# Community Packages
import pandas as pd
import pytest

# Xport Modules
import xport


def make_special(values):
    """
    Build an object-dtype column that preserves special missing tags.
    """
    return pd.Series(values, dtype=object)


class TestFieldMapping:
    """
    Verify explicit field identity and duplicate-name handling.
    """

    def test_match_by_name_ignores_order(self):
        left = xport.Dataset({'ID': [1], 'X': [2], 'C': ['a']})
        right = xport.Dataset({'C': ['a'], 'ID': [1], 'X': [2]})
        mapping = xport.FieldMapping.build(left, right)
        assert mapping.pairs == (('ID', 'ID'), ('X', 'X'), ('C', 'C'))
        assert mapping.left_positions == (0, 1, 2)
        assert mapping.right_positions == (1, 2, 0)

    def test_explicit_rename_mapping(self):
        left = xport.Dataset({'ID': [1], 'X': [2]})
        right = xport.Dataset({'ID': [1], 'Y': [2]})
        mapping = xport.FieldMapping.build(left, right, pairs={'X': 'Y'})
        assert mapping.pairs == (('ID', 'ID'), ('X', 'Y'))

    def test_explicit_rename_mapping_accepts_pairs(self):
        left = xport.Dataset({'ID': [1], 'X': [2]})
        right = xport.Dataset({'ID': [1], 'Y': [2]})
        mapping = xport.FieldMapping.build(left, right, pairs=[('X', 'Y')])
        assert mapping.pairs == (('ID', 'ID'), ('X', 'Y'))

    def test_unknown_field_raises(self):
        left = xport.Dataset({'ID': [1]})
        right = xport.Dataset({'ID': [1]})
        with pytest.raises(KeyError):
            xport.FieldMapping.build(left, right, pairs={'NOPE': 'ID'})
        with pytest.raises(KeyError):
            xport.FieldMapping.build(left, right, pairs={'ID': 'NOPE'})

    def test_duplicate_names_resolved_by_occurrence(self):
        left = xport.Dataset(pd.DataFrame([[1, 2]], columns=['DUP', 'DUP']))
        right = xport.Dataset(pd.DataFrame([[2, 1]], columns=['DUP', 'DUP']))
        mapping = xport.FieldMapping.build(left, right)
        assert mapping.pairs == (('DUP', 'DUP'), ('DUP', 'DUP'))
        assert mapping.index_pairs == ((0, 0), (1, 1))

    def test_duplicate_names_explicit_occurrence(self):
        left = xport.Dataset(pd.DataFrame([[1, 2]], columns=['DUP', 'DUP']))
        right = xport.Dataset(pd.DataFrame([[2, 1]], columns=['DUP', 'DUP']))
        mapping = xport.FieldMapping.build(
            left, right, pairs={('DUP', 0): ('DUP', 1), ('DUP', 1): ('DUP', 0)}
        )
        assert mapping.index_pairs == ((0, 1), (1, 0))

    def test_unmatched_fields_left_out(self):
        left = xport.Dataset({'ID': [1], 'A': [1], 'EXTRA': [9]})
        right = xport.Dataset({'ID': [1], 'A': [1], 'OTHER': [8]})
        mapping = xport.FieldMapping.build(left, right)
        assert mapping.pairs == (('ID', 'ID'), ('A', 'A'))

    def test_flipped(self):
        mapping = xport.FieldMapping(
            [('ID', 'ID'), ('X', 'Y')], index_pairs=((0, 1), (1, 0))
        )
        flipped = mapping.flipped()
        assert flipped.pairs == (('ID', 'ID'), ('Y', 'X'))
        assert flipped.index_pairs == ((1, 0), (0, 1))

    def test_invalid_pair_raises(self):
        with pytest.raises(ValueError):
            xport.FieldMapping([('ID',)])


class TestRecord:
    """
    Verify records preserve repeated field names.
    """

    def test_repeated_names_are_preserved(self):
        record = xport.Record([('ID', 1), ('DUP', 2), ('DUP', 3)])
        assert record.names() == ('ID', 'DUP', 'DUP')
        assert record.get_all('DUP') == [2, 3]
        assert record.value_at(2) == 3
        assert record['DUP'] == 2

    def test_view_keeps_duplicate_columns_distinct(self):
        dataset = xport.Dataset(pd.DataFrame([[1, 2, 3]], columns=['ID', 'DUP', 'DUP']))
        view = xport.record_view(dataset)
        record = view.record(0)
        assert record.get_all('DUP') == [2, 3]
        assert record.values() == (1, 2, 3)

    def test_equality_with_dict_for_unique_names(self):
        assert xport.Record([('ID', 1), ('X', 2)]) == {'ID': 1, 'X': 2}


class TestRecordView:
    """
    Verify the record view keeps identity while preserving physical order.
    """

    def test_physical_order_preserved(self):
        dataset = xport.Dataset({'C': ['a'], 'X': [1.0], 'ID': [1]})
        view = xport.record_view(dataset)
        assert view.physical_order == ('C', 'X', 'ID')

    def test_record_keyed_by_field_identity(self):
        left = xport.Dataset({'ID': [1], 'X': [1.0], 'C': ['a']})
        right = xport.Dataset({'C': ['a'], 'X': [1.0], 'ID': [1]})
        mapping = xport.FieldMapping.build(left, right)
        view = xport.record_view(right, mapping, side='right')
        assert view.fields == ('ID', 'X', 'C')
        assert view.physical_order == ('C', 'X', 'ID')
        assert view.record(0) == {'ID': 1, 'X': 1.0, 'C': 'a'}

    def test_iteration_and_length(self):
        dataset = xport.Dataset({'ID': [1, 2]})
        view = xport.record_view(dataset)
        assert len(view) == 2
        assert list(view) == [{'ID': 1}, {'ID': 2}]

    def test_invalid_side_raises(self):
        dataset = xport.Dataset({'ID': [1]})
        with pytest.raises(ValueError):
            xport.record_view(dataset, xport.FieldMapping(), side='middle')


class TestDiffDatasets:
    """
    Verify record comparison by agreed field identity.
    """

    def test_reordered_columns_are_not_new_cases(self):
        left = xport.Dataset({'ID': [1, 2], 'X': [1.0, 2.0], 'C': ['a', 'b']})
        right = xport.Dataset({'C': ['a', 'b'], 'X': [1.0, 2.0], 'ID': [1, 2]})
        result = xport.diff_datasets(left, right, key='ID')
        assert result.is_identical
        assert len(result.matched) == 2
        assert not result.changed
        assert not result.added
        assert not result.removed

    def test_renamed_field_requires_mapping_and_matches(self):
        left = xport.Dataset({'ID': [1, 2], 'X': [1.0, 2.0]})
        right = xport.Dataset({'ID': [1, 2], 'Y': [1.0, 2.0]})
        result = xport.diff_datasets(left, right, key='ID', pairs={'X': 'Y'})
        assert result.is_identical

    def test_hand_built_mapping_resolves_positions(self):
        left = xport.Dataset({'ID': [1, 2], 'X': [1.0, 2.0]})
        right = xport.Dataset({'X': [1.0, 2.0], 'ID': [1, 2]})
        mapping = xport.FieldMapping([('ID', 'ID'), ('X', 'X')])
        assert mapping.index_pairs is None
        result = xport.diff_datasets(left, right, key='ID', pairs=mapping)
        assert result.is_identical
        assert result.mapping.index_pairs == ((0, 1), (1, 0))

    def test_value_change_reported(self):
        left = xport.Dataset({'ID': [1], 'AMT': [1000.0]})
        right = xport.Dataset({'ID': [1], 'AMT': [2000.0]})
        result = xport.diff_datasets(left, right, key='ID')
        assert not result.is_identical
        change = result.changed[0].changes[0]
        assert change.field == 'AMT'
        assert change.kind == xport.ChangeKind.VALUE
        assert change.left == 1000.0
        assert change.right == 2000.0
        assert len(result.changed[0].value_changes) == 1
        assert not result.changed[0].format_changes

    def test_format_only_change_distinct_from_value_change(self):
        left = xport.Dataset({'ID': [1], 'AMT': [1000.0]})
        left['AMT'].format = 'comma8.0'
        right = xport.Dataset({'ID': [1], 'AMT': [1000.0]})
        right['AMT'].format = 'dollar10.0'
        result = xport.diff_datasets(left, right, key='ID')
        assert not result.is_identical
        change = result.changed[0].changes[0]
        assert change.kind == xport.ChangeKind.FORMATTED
        assert change.left == change.right == 1000.0
        assert change.left_text != change.right_text
        assert not result.changed[0].value_changes
        assert len(result.changed[0].format_changes) == 1

    def test_added_and_removed_records(self):
        left = xport.Dataset({'ID': [1, 2], 'X': [1, 2]})
        right = xport.Dataset({'ID': [2, 3], 'X': [2, 3]})
        result = xport.diff_datasets(left, right, key='ID')
        assert result.removed == ((1,),)
        assert result.added == ((3,),)
        assert len(result.matched) == 1

    def test_missing_versus_present(self):
        left = xport.Dataset({'ID': [1], 'M': make_special([1.0])})
        right = xport.Dataset({'ID': [1], 'M': make_special([float('nan')])})
        result = xport.diff_datasets(left, right, key='ID')
        change = result.changed[0].changes[0]
        assert change.kind == xport.ChangeKind.MISSING

    def test_special_missing_categories_participate(self):
        left = xport.Dataset(
            {'ID': [1, 2, 3], 'M': make_special([xport.NaN.A, xport.NaN.A, xport.NaN.B])}
        )
        right = xport.Dataset(
            {'ID': [1, 2, 3], 'M': make_special([xport.NaN.A, xport.NaN.B, xport.NaN.B])}
        )
        result = xport.diff_datasets(left, right, key='ID')
        changed = {record.key[0]: record for record in result.changed}
        assert 1 not in changed
        assert changed[2].changes[0].kind == xport.ChangeKind.MISSING
        assert 3 not in changed

    def test_plain_missing_differs_from_tagged_missing(self):
        left = xport.Dataset({'ID': [1], 'M': make_special([float('nan')])})
        right = xport.Dataset({'ID': [1], 'M': make_special([xport.NaN.A])})
        result = xport.diff_datasets(left, right, key='ID')
        assert result.changed[0].changes[0].kind == xport.ChangeKind.MISSING

    def test_unmatched_fields_reported(self):
        left = xport.Dataset({'ID': [1], 'A': [1], 'EXTRA': [9]})
        right = xport.Dataset({'ID': [1], 'A': [1], 'OTHER': [8]})
        result = xport.diff_datasets(left, right, key='ID')
        assert result.unmatched_left == ('EXTRA',)
        assert result.unmatched_right == ('OTHER',)
        assert result.is_identical

    def test_composite_key(self):
        left = xport.Dataset({'A': [1, 1], 'B': [1, 2], 'V': ['x', 'y']})
        right = xport.Dataset({'B': [2, 1], 'A': [1, 1], 'V': ['y', 'x']})
        result = xport.diff_datasets(left, right, key=('A', 'B'))
        assert result.is_identical
        assert len(result.matched) == 2

    def test_duplicate_field_names_compare_positionally(self):
        left = xport.Dataset(pd.DataFrame([[1, 2, 3]], columns=['ID', 'DUP', 'DUP']))
        right = xport.Dataset(pd.DataFrame([[3, 2, 1]], columns=['DUP', 'DUP', 'ID']))
        mapping = xport.FieldMapping.build(
            left, right, pairs={('DUP', 0): ('DUP', 1), ('DUP', 1): ('DUP', 0)}
        )
        result = xport.diff_datasets(left, right, key='ID', pairs=mapping)
        assert result.is_identical

    def test_duplicate_field_names_detect_real_change(self):
        left = xport.Dataset(pd.DataFrame([[1, 2, 3]], columns=['ID', 'DUP', 'DUP']))
        right = xport.Dataset(pd.DataFrame([[2, 3, 1]], columns=['DUP', 'DUP', 'ID']))
        mapping = xport.FieldMapping.build(
            left, right, pairs={('DUP', 0): ('DUP', 1), ('DUP', 1): ('DUP', 0)}
        )
        result = xport.diff_datasets(left, right, key='ID', pairs=mapping)
        # Right order is DUP=2, DUP=3; crossed mapping pairs 2->3 and 3->2.
        assert {change.kind for record in result.changed for change in record.changes} == {
            xport.ChangeKind.VALUE
        }

    def test_key_must_be_mapped(self):
        left = xport.Dataset({'ID': [1]})
        right = xport.Dataset({'OTHER': [1]})
        with pytest.raises(KeyError):
            xport.diff_datasets(left, right, key='ID')

    def test_bool_and_str(self):
        identical = xport.diff_datasets(
            xport.Dataset({'ID': [1], 'X': [1]}),
            xport.Dataset({'X': [1], 'ID': [1]}),
            key='ID',
        )
        assert not bool(identical)
        assert 'no differences' in str(identical)
        different = xport.diff_datasets(
            xport.Dataset({'ID': [1], 'X': [1]}),
            xport.Dataset({'ID': [1], 'X': [2]}),
            key='ID',
        )
        assert bool(different)
        assert 'X' in str(different)
