from decimal import Decimal

from pytest import raises

from wikitextparser import Table, record_view
from wikitextparser._records import ColumnError


SORTABLE = '''{| class="wikitable sortable"
! 名称 !! 报价 !! 数量
|-
| A || data-sort-value="29800" | 约三万 || 1
|-
| B || 4.5万 || 2
|-
| C || 12000 || 3
|-
| D || 1.2万 || 4
|}'''


def test_display_sort_and_source():
    view = Table(SORTABLE).records()
    assert view.headers == ('名称', '报价', '数量')
    first = view.records[0]
    value = first.value(1)
    assert value is not None
    assert value.display == '约三万'
    assert value.sort == 29800
    assert value.sort_kind == 'number'
    assert value.sort_source == 'data-sort-value'
    assert value.has_data_sort_value
    assert value.source is not None
    assert value.source.wikitext.rstrip() == (
        '|| data-sort-value="29800" | 约三万'
    )
    assert value.source.attrs == {'data-sort-value': '29800'}
    assert value.source.row == 1
    assert value.source.column == 1
    assert value.source.is_anchor


def test_sorting_does_not_rewrite_display_value():
    table = Table(SORTABLE)
    before = table.string
    view = table.records()
    result = view.sort_by('报价')
    assert [record.display(0) for record in result.records] == [
        'C',
        'D',
        'A',
        'B',
    ]
    assert [record.display(1) for record in result.records] == [
        '12000',
        '1.2万',
        '约三万',
        '4.5万',
    ]
    assert result.records[2].display(1) == '约三万'
    assert result.records[2].sort_value(1) == 29800
    assert table.string == before


def test_rowspan_group_is_one_record():
    table = Table(
        '{|\n'
        '! 名称 !! 报价\n'
        '|-\n'
        '| rowspan=2 | A || 100\n'
        '|-\n'
        '| 200\n'
        '|-\n'
        '| B || 300\n'
        '|}'
    )
    view = table.records()
    assert len(view.records) == 2
    record = view.records[0]
    assert record.rows == (1, 2)
    assert record.value(0) is not None
    assert record.value(0).display == 'A'
    assert record.value(0).source is not None
    assert record.value(0).source.rowspan == 2
    assert record.displays(1) == ('100', '200')
    assert record.value(0).source.anchor_row == 1


def test_sort_keeps_merged_record_together():
    table = Table(
        '{|\n'
        '! 名称 !! 数量\n'
        '|-\n'
        '| rowspan=2 | A || 2\n'
        '|-\n'
        '| 1\n'
        '|-\n'
        '| B || 3\n'
        '|}'
    )
    result = table.records().sort_by('数量')
    labels = [[record.display(0) for _ in record.rows]
              for record in result.records]
    assert labels == [['A', 'A'], ['B']]
    assert [record.rows for record in result.records] == [(1, 2), (3,)]


def test_colspan_repeats_value_but_not_source():
    table = Table(
        '{|\n'
        '! A !! B !! C\n'
        '|-\n'
        '| colspan=2 | M || 1\n'
        '|-\n'
        '| N || 2 || 3\n'
        '|}'
    )
    view = table.records()
    first = view.records[0]
    assert first.displays(0) == ('M',)
    assert first.displays(1) == ('M',)
    assert first.display(2) == '1'
    assert first.source(0) is not None
    assert first.source(1) is not None
    assert first.source(0).colspan == 2
    assert first.source(0).anchor_column == 0
    assert first.source(1).anchor_column == 0
    assert first.source(1).is_anchor is False


def test_missing_key_is_not_zero():
    table = Table(
        '{|\n'
        '! 名称 !! 分数\n'
        '|-\n'
        '| A || -\n'
        '|-\n'
        '| B ||\n'
        '|-\n'
        '| C || 0\n'
        '|}'
    )
    result = table.records().sort_by('分数')
    assert [record.display(0) for record in result.records] == ['C']
    reasons = {entry.reason for entry in result.unsortable}
    assert reasons == {'invalid', 'missing'}
    assert len(result.records) + len(result.unsortable) == 3


def test_invalid_and_incompatible_units_are_separated():
    table = Table(
        '{|\n'
        '! 名称 !! 长度\n'
        '|-\n'
        '| A || 1200 m\n'
        '|-\n'
        '| B || 1.5 km\n'
        '|-\n'
        '| C || 2 英里\n'
        '|-\n'
        '| D || 3 cm\n'
        '|}'
    )
    result = table.records().sort_by('长度')
    assert result.kind == 'number'
    assert result.unit == 'length'
    assert [record.display(0) for record in result.records] == [
        'D',
        'A',
        'B',
        'C',
    ]
    assert result.records[-1].display(1) == '2 英里'
    assert result.unsortable == ()


def test_separate_unit_column():
    table = Table(
        '{|\n'
        '! 名称 !! 金额 !! 单位\n'
        '|-\n'
        '| A || 2 || 千元\n'
        '|-\n'
        '| B || 3 || 万元\n'
        '|-\n'
        '| C || 500 || 元\n'
        '|-\n'
        '| D || 4 || 美元\n'
        '|}'
    )
    result = table.records().sort_by('金额')
    assert [record.display(0) for record in result.records] == [
        'C',
        'A',
        'B',
    ]
    assert [entry.display for entry in result.unsortable] == ['4']


def test_unknown_inline_unit_is_text_not_magic_number():
    table = Table(
        '{|\n'
        '! 名称 !! 值\n'
        '|-\n'
        '| A || 12 apples\n'
        '|-\n'
        '| B || 100\n'
        '|}'
    )
    result = table.records().sort_by('值')
    assert [record.display(0) for record in result.records] == ['B']
    assert [entry.reason for entry in result.unsortable] == ['unit']


def test_sort_by_column_index_and_negative_index():
    view = Table(SORTABLE).records()
    assert view.column_index('报价') == 1
    assert view.column_index(-1) == 2
    with raises(ColumnError):
        view.column_index('不存在')
    with raises(ColumnError):
        view.column_index(9)


def test_header_lookup_is_case_insensitive():
    table = Table(
        '{|\n! Name !! Amount\n|-\n| A || 2\n|-\n| B || 1\n|}'
    )
    view = table.records()
    assert view.column_index('amount') == 1
    result = view.sort_by('AMOUNT')
    assert [record.display(0) for record in result.records] == ['B', 'A']


def test_span_data_sort_value_is_supported():
    table = Table(
        '{|\n'
        '! 名称 !! 报价\n'
        '|-\n'
        '| A || <span data-sort-value="29800">约三万</span>\n'
        '|-\n'
        '| B || 4.5万\n'
        '|-\n'
        '| C || <span data-sort-value="45000">4.5万</span>\n'
        '|}'
    )
    view = table.records()
    value = view.records[0].value(1)
    assert value is not None
    assert value.display == '约三万'
    assert value.sort == 29800
    assert value.sort_source == 'data-sort-value'
    result = view.sort_by('报价')
    assert [record.display(0) for record in result.records] == [
        'A',
        'B',
        'C',
    ]


def test_uppercase_span_data_sort_value_is_supported():
    table = Table(
        '{|\n'
        '! 名称 !! 报价\n'
        '|-\n'
        '| A || <SPAN data-sort-value="29800">约三万</SPAN>\n'
        '|-\n'
        '| B || 12000\n'
        '|}'
    )
    result = table.records().sort_by('报价')
    assert [record.display(0) for record in result.records] == ['B', 'A']
    assert result.records[1].display(1) == '约三万'
    assert result.records[1].sort_value(1) == 29800


def test_data_sort_value_key_can_be_textual():
    table = Table(
        '{|\n'
        '! 名称 !! 评级\n'
        '|-\n'
        '| A || data-sort-value="b" | 乙\n'
        '|-\n'
        '| B || data-sort-value="a" | 甲\n'
        '|}'
    )
    result = table.records().sort_by('评级')
    assert result.kind == 'text'
    assert [record.display(0) for record in result.records] == ['B', 'A']
    assert result.records[0].display(1) == '甲'


def test_reverse_keeps_whole_records_and_separates_bad_keys():
    table = Table(
        '{|\n'
        '! 名称 !! 值\n'
        '|-\n'
        '| A || 1\n'
        '|-\n'
        '| B || 3\n'
        '|-\n'
        '| C || -\n'
        '|-\n'
        '| D || 2\n'
        '|}'
    )
    result = table.records().sort_by('值', reverse=True)
    assert [record.display(0) for record in result.records] == [
        'B',
        'D',
        'A',
    ]
    assert [entry.display for entry in result.unsortable] == ['-']


def test_empty_table():
    view = Table('{|class=wikitable\n|}').records()
    assert view.records == ()
    assert view.headers == ()
    assert view.width == 0
    assert view.sort_by(0).records == ()


def test_provenance_dicts():
    view = Table(SORTABLE).records()
    count = sum(
        1
        for record in view.records
        if record.value(1) is not None
        and record.value(1).source is not None
        and record.value(1).source.anchor_row == record.row
    )
    assert count == len(view.records)
    row = view.as_dicts(include_sources=True)[0]
    assert row['名称']['display'] == 'A'
    assert row['报价']['sort'] == Decimal('29800')
    assert row['报价']['source']['attrs'] == {
        'data-sort-value': '29800'
    }


def test_module_level_record_view():
    view = record_view(Table(SORTABLE))
    assert len(view) == 4
    assert view.as_lists()[0] == ['A', '约三万', '1']


def test_approximate_and_range_values_are_separated():
    table = Table(
        '{|\n'
        '! 名称 !! 预算\n'
        '|-\n'
        '| A || 约 2,000\n'
        '|-\n'
        '| B || 1,000–1,500\n'
        '|-\n'
        '| C || 1200\n'
        '|}'
    )
    result = table.records().sort_by('预算')
    assert [record.display(0) for record in result.records] == ['C']
    reasons = {entry.reason for entry in result.unsortable}
    assert 'approximate' in reasons
    assert 'invalid' in reasons


def test_currency_prefix_and_suffix_are_comparable():
    table = Table(
        '{|\n'
        '! 名称 !! 价格\n'
        '|-\n'
        '| A || RMB 900\n'
        '|-\n'
        '| B || 800元\n'
        '|-\n'
        '| C || US$ 1\n'
        '|}'
    )
    result = table.records().sort_by('价格')
    assert result.kind == 'number'
    assert result.unit == 'CNY'
    assert [record.display(0) for record in result.records] == ['B', 'A']
    assert [entry.reason for entry in result.unsortable] == ['unit']


def test_strip_false_preserves_spacing():
    table = Table('{|\n! A\n|-\n|  x  \n|}')
    assert table.records(strip=False).records[0].display(0) == '  x  '
    assert table.records().records[0].display(0) == 'x'


def test_header_only_table_has_no_records():
    view = Table('{|\n! A !! B\n|}').records()
    assert view.headers == ('A', 'B')
    assert view.records == ()
    assert view.sort_by(0).records == ()


def test_sort_by_numeric_column_keeps_merged_group_whole():
    table = Table(
        '{|\n'
        '! 名称 !! 数量\n'
        '|-\n'
        '| rowspan=2 | A || 2\n'
        '|-\n'
        '| 3\n'
        '|-\n'
        '| B || 1\n'
        '|}'
    )
    result = table.records().sort_by('数量')
    assert [record.rows for record in result.records] == [(3,), (1, 2)]
    assert result.records[1].display(0) == 'A'
    assert result.records[1].displays(1) == ('2', '3')


def test_values_compare_as_display_strings():
    view = Table(SORTABLE).records()
    first = view.records[0].value(1)
    assert first is not None
    assert first == '约三万'
    assert str(first) == '约三万'
    assert first[0] == '约'
    assert first.display == str(first)


def test_duplicate_header_can_be_disambiguated():
    view = Table('{|\n! A !! B !! A\n|-\n| 1 || 2 || 3\n|}').records()
    assert view.column_index('B') == 1
    assert view.column_index('A#1') == 0
    assert view.column_index('a#2') == 2
    with raises(ColumnError):
        view.column_index('A')
    with raises(ColumnError):
        view.column_index('A#3')


def test_record_lookup_by_header_and_aliases():
    view = Table(SORTABLE).records()
    record = view.rows[0]
    assert record['名称'] == 'A'
    assert record['报价'] == '约三万'
    with raises(KeyError):
        record['bogus']
    result = Table(SORTABLE).sorted_records('报价')
    assert [item.display(0) for item in result.records] == [
        'C',
        'D',
        'A',
        'B',
    ]


def test_regression_existing_data_is_unchanged():
    table = Table(SORTABLE)
    assert table.data() == [
        ['名称', '报价', '数量'],
        ['A', '约三万', '1'],
        ['B', '4.5万', '2'],
        ['C', '12000', '3'],
        ['D', '1.2万', '4'],
    ]
