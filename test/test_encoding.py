"""
Tests for character-field encoding previews.
"""

# Standard Library
import json

# Community Packages
import pandas as pd
import pytest

# Xport Modules
import xport
import xport.v56
import xport.v89


def test_character_count_unchanged_but_bytes_overflow():
    """
    Two-byte characters keep the count but exceed a byte-width field.
    """
    values = ['ab', 'éé', 'ééé']
    report = xport.encoding_report(values, encoding='utf-8', width=4)
    assert [len(v.encoded_bytes) for v in report] == [2, 4, 6]
    assert [v.truncated for v in report] == [False, False, True]
    assert report.values[2].truncated_value == 'éé'
    assert report.values[2].truncated_bytes == b'\xc3\xa9\xc3\xa9'
    assert report.values[1].character_only_overflow is False
    report = xport.encoding_report(['ééé'], encoding='utf-8', width=5)
    entry = report.values[0]
    assert entry.char_count == 3
    assert entry.fits_by_characters
    assert not entry.fits_by_bytes
    assert entry.character_only_overflow
    assert entry.overflow_bytes == 1


def test_unrepresentable_reported_per_value():
    """
    Each value reports whether the target codec can represent it.
    """
    report = xport.encoding_report(['ok', '☃'], encoding='ascii', width=8)
    assert report.representability == {'ok': True, '☃': False}
    assert report.has_unrepresentable
    assert report.values[1].error
    assert report.values[1].encoded_bytes is None
    assert not report.lossless


def test_original_bytes_preserved():
    """
    The report retains the original bytes and the target encoding.
    """
    report = xport.encoding_report(['ñ'], encoding='utf-8', width=8)
    entry = report.values[0]
    assert entry.original_bytes == 'ñ'.encode('ISO-8859-1')
    assert entry.original_bytes != entry.encoded_bytes
    assert entry.encoding == report.encoding == 'utf-8'
    assert entry.source_encoding == 'ISO-8859-1'
    json.dumps(report.as_json())


def test_truncation_stops_at_character_boundary():
    """
    Truncation never leaves a partial multibyte character.
    """
    report = xport.encoding_report(['ééé'], encoding='utf-8', width=5)
    entry = report.values[0]
    assert entry.truncated
    assert entry.truncated_bytes == b'\xc3\xa9\xc3\xa9'
    assert entry.truncated_value == 'éé'
    assert entry.dropped_bytes == b'\xc3\xa9'
    assert entry.removed_characters == 1
    entry.truncated_bytes.decode('utf-8')


def test_truncation_stops_at_boundary_for_utf16():
    """
    A code unit is two bytes; an odd width must not split a character.
    """
    text = 'abé'
    encoded = text.encode('utf-16-le')
    truncated = xport.truncate_bytes(encoded, 5, 'utf-16-le')
    assert truncated == 'ab'.encode('utf-16-le')
    assert truncated.decode('utf-16-le') == 'ab'


def test_no_complete_character_fits():
    """
    When even one character exceeds the width, the output is empty.
    """
    report = xport.encoding_report(['☃'], encoding='utf-8', width=2)
    entry = report.values[0]
    assert entry.truncated
    assert entry.truncated_bytes == b''
    assert entry.truncated_value == ''


def test_collisions_listed_for_distinct_values():
    """
    Distinct values that converge after truncation are collisions.
    """
    report = xport.encoding_report(['abcdX', 'abcdY', 'abcdX'], encoding='utf-8', width=4)
    assert len(report.collisions) == 1
    collision = report.collisions[0]
    assert collision.values == ('abcdX', 'abcdY')
    assert collision.target_value == 'abcd'
    assert collision.target_bytes == b'abcd'
    assert collision.count == 2
    assert collision.as_dict()['positions'] == [0, 1]
    assert all(v.collision == 0 for v in collision.reports)


def test_no_collision_when_values_stay_distinct():
    """
    Truncation alone is not a collision only when the values match.
    """
    report = xport.encoding_report(['abcde'], encoding='utf-8', width=4)
    assert not report.collisions
    assert report.has_truncation
    assert not report.lossless


def test_collision_after_reencoding():
    """
    Replacement characters may converge distinct source values.
    """
    report = xport.encoding_report(['☃', '☄'], encoding='ascii', width=4, errors='replace')
    assert not report.representable
    assert len(report.collisions) == 1
    assert report.collisions[0].target_bytes == b'?'
    assert report.collisions[0].values == ('☃', '☄')


def test_dataset_report_per_field():
    """
    A dataset report covers each character field and its width.
    """
    ds = xport.Dataset({
        'a': xport.Variable(['abcd'], width=4),
        'b': xport.Variable(['abcd'], width=2),
        'c': xport.Variable([1, 2], width=8),
    }, name='D')
    report = ds.encoding_report(encoding='utf-8')
    assert list(report) == ['a', 'b']
    assert not report['a'].has_truncation
    assert report['b'].has_truncation
    assert report['b'].values[0].truncated_value == 'ab'
    json.dumps(report.as_json())


def test_dataset_report_can_include_numeric():
    """
    Numeric columns are skipped unless explicitly requested.
    """
    ds = xport.Dataset({'n': [1], 's': ['x']}, name='D')
    assert list(ds.encoding_report(encoding='utf-8')) == ['s']
    report = xport.dataset_encoding_report(ds, encoding='utf-8', include_numeric=True)
    assert list(report) == ['n', 's']


def test_explicit_widths_override_variable_width():
    """
    Widths may be supplied per field.
    """
    ds = xport.Dataset({'a': ['abcd']}, name='D')
    report = k = ds.encoding_report(encoding='utf-8', widths={'a': 2})
    assert k['a'].width == 2
    assert report['a'].values[0].truncated_value == 'ab'


def test_series_index_is_retained():
    """
    Per-value reports identify rows by their index.
    """
    series = xport.Variable(['abcd', 'ab'], index=['x', 'y'], width=2)
    report = series.encoding_report(encoding='utf-8')
    assert [v.index for v in report] == ['x', 'y']
    assert [v.position for v in report] == [0, 1]


def test_missing_values_are_reported():
    """
    Missing values write blanks without error or collision noise.
    """
    report = xport.encoding_report([None, None], encoding='ascii', width=4)
    assert all(v.missing for v in report)
    assert all(v.truncated_bytes == b'' for v in report)
    assert not report.collisions


def test_missing_and_empty_string_collide():
    """
    Missing and empty values converge on the same stored bytes.
    """
    report = xport.encoding_report([None, ''], encoding='ascii', width=4)
    assert report.collisions
    descriptor = report.collisions[0]
    assert descriptor.target_bytes == b''
    assert set(descriptor.values) == {None, ''}


def test_preflight_aliases():
    """
    Each module exposes the preview helpers.
    """
    assert xport.preflight_encoding is xport.encoding_report
    assert xport.analyze_encoding is xport.encoding_report
    assert xport.v56.preview_encoding is xport.v56.encoding_report
    assert xport.v89.encoding_report is xport.v56.encoding_report
    assert xport.v89.analyze_encoding is xport.v56.encoding_report


def test_v56_report_uses_current_text_encoding():
    """
    The v56 preview follows the module's configurable data encoding.
    """
    values = ['é']
    assert xport.v56.encoding_report(values, width=8).encoding == xport.v56.TEXT_DATA_ENCODING
    with xport.v56._encoding(data='utf-8'):
        report = xport.v56.encoding_report(values, width=8)
        assert report.encoding == 'utf-8'
        assert report.values[0].representable


def test_variable_and_dataset_methods():
    """
    ``Variable`` and ``Dataset`` expose the preview directly.
    """
    variable = xport.Variable(['éé'], width=3)
    assert variable.encoding_report(encoding='utf-8').values[0].truncated_value == 'é'
    dataset = xport.Dataset({'v': variable}, name='D')
    assert dataset.encoding_report(encoding='utf-8')['v'].width == 3


def test_report_require_rejects_lossy_conversion():
    """
    ``require`` refuses collisions and truncation by default.
    """
    report = xport.encoding_report(['abcdX', 'abcdY'], encoding='utf-8', width=4)
    with pytest.raises(ValueError, match='collision'):
        report.require()
    report.require(
        allow_truncation=True,
        allow_unrepresentable=True,
        allow_collisions=True,
    )


def test_dataset_report_require_names_the_field():
    """
    A rejected dataset report identifies the offending field.
    """
    ds = xport.Dataset({'a': xport.Variable(['abcd'], width=2)}, name='D')
    report = ds.encoding_report(encoding='utf-8')
    with pytest.raises(ValueError, match=r"Field 'a'"):
        report.require()


def test_stored_value_reflects_reader_semantics():
    """
    Stored bytes include padding; readers discard trailing blanks.
    """
    report = xport.encoding_report(['ab'], encoding='ascii', width=4)
    entry = report.values[0]
    assert entry.stored_bytes == b'ab  '
    assert entry.stored_value == 'ab'
    assert entry.used_encoding == 'ascii'
    assert entry.original_encoding == report.source_encoding


def test_report_text_mentions_bytes_and_collisions():
    """
    The human-readable preview retains bytes and flags collisions.
    """
    report = xport.encoding_report(['abcdX', 'abcdY'], encoding='utf-8', width=4)
    text = str(report)
    assert 'abcd' in text
    assert '61626364' in text
    assert 'Collisions' in text


def test_dump_truncates_at_boundary():
    """
    Writing a character field truncates on a character boundary.
    """
    ds = xport.Dataset(
        {'v': xport.Variable(['ééé'], width=5)},
        name='D',
    )
    with xport.v56._encoding(data='utf-8'):
        bytestring = xport.v56.dumps(ds)
        library = xport.v56.loads(bytestring)
    assert library['D']['v'][0] == 'éé'


def test_encoding_report_single_value_and_scalar():
    """
    A single value and a Pandas series are both accepted.
    """
    assert len(xport.encoding_report('é', encoding='utf-8', width=1)) == 1
    series = pd.Series(['é'], dtype='string')
    assert xport.encoding_report(series, encoding='utf-8', width=1).values[0].truncated


def test_v89_metadata_truncation_keeps_characters_whole():
    """
    v8 short names are cut at metadata character boundaries.
    """
    name = 'a' + 'é' * 10
    example = xport.Dataset({'v': ['x']}, name=name)
    with xport.v89._encoding(metadata='utf-8'):
        bytestring = xport.v89.dumps(example)
        library = xport.v89.loads(bytestring)
    assert library[name]['v'][0] == 'x'
    assert bytestring.count(b'\xc3\xa9') > 0


def test_library_report_and_require():
    """
    A library preview covers each dataset and reports collisions by field.
    """
    first = xport.Dataset({'v': xport.Variable(['abcdX', 'abcdY'], width=4)}, name='A')
    second = xport.Dataset({'s': xport.Variable(['ok'], width=2)}, name='B')
    library = xport.Library([first, second])
    report = library.encoding_report(encoding='utf-8')
    assert list(report) == ['A', 'B']
    assert report['A'].has_collisions
    assert report.all_collisions[0][:2] == ('A', 'v')
    assert not report.lossless
    with pytest.raises(ValueError, match=r"Dataset 'A'"):
        report.require()
    json.dumps(report.as_json())


def test_library_report_with_v56_defaults():
    """
    The v56 library preview follows the module's text encoding.
    """
    library = xport.v56.Library(xport.v56.Member({'v': ['é']}))
    with xport.v56._encoding(data='utf-8'):
        report = library.encoding_report()
    assert report.encoding == 'utf-8'
    assert report[None]['v'].encoding == 'utf-8'


def test_source_unrepresentable_has_no_original_bytes():
    """
    When the source codec cannot represent a value, no bytes are invented.
    """
    report = xport.encoding_report(['☃'], encoding='utf-8', width=8)
    entry = report.values[0]
    assert entry.original_bytes == b''
    assert not entry.original_bytes_complete
    assert not entry.source_representable
    assert entry.source_encoding == 'ISO-8859-1'
    assert entry.representable
    assert not entry.ok


def test_v56_member_report_uses_current_text_encoding():
    """
    A v56 Member preview follows the module's data encoding.
    """
    ds = xport.v56.Member({'v': xport.Variable(['中文'], width=4)}, name='D')
    with xport.v56._encoding(data='gb18030'):
        report = ds.encoding_report()
    assert report.encoding == 'gb18030'
    assert report['v'].values[0].original_bytes == '中文'.encode('gb18030')
    assert not report['v'].values[0].truncated


@pytest.mark.parametrize('encoding,text,width,expected', [
    ('utf-8', 'éé', 3, 'é'),
    ('utf-8', 'éé', 4, 'éé'),
    ('utf-16-le', 'abé', 5, 'ab'),
    ('gb18030', '中文', 3, '中'),
])
def test_boundary_cases(encoding, text, width, expected):
    """
    Boundary-safe truncation for variable-width codecs.
    """
    report = xport.encoding_report([text], encoding=encoding, width=width)
    assert report.values[0].truncated_value == expected
