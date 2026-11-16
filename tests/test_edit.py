from itertools import permutations

import pytest

from wikitextparser import (
    EditConflictError,
    EditError,
    EditSuggestion,
    apply_edit_suggestions,
    parse,
)

TEXT = (
    '<!-- c1 -->\n'
    'See [https://old.example.com old] and bare https://bare.test/x.\n'
    '{| class="wikitable"\n'
    '! H1 !! H2\n'
    '|-\n'
    '| a || b\n'
    '|}\n'
    '<!-- tail -->'
)

EXPECTED = (
    '\n'
    'See [https://new.example.org/path old] and bare '
    'https://bare.test/x.\n'
    '{| class="wikitable"\n'
    '! H1 !! H2\n'
    '|-\n'
    '|alpha|| b\n'
    '|}\n'
    '<!-- tail -->'
)


def _suggestions():
    parsed = parse(TEXT)
    comment_span = parsed.comments[0].span
    link_span = parsed.external_links[0].span
    cell = parsed.tables[0].cells(row=1, column=0)
    return [
        EditSuggestion.delete_comment(comment_span),
        EditSuggestion.replace_external_link(
            link_span, 'https://new.example.org/path'
        ),
        EditSuggestion.replace_cell(cell.span, 'alpha'),
    ]


def test_merges_all_three_suggestion_kinds():
    parsed = parse(TEXT)
    suggestions = _suggestions()
    result = apply_edit_suggestions(parsed, suggestions)
    assert result is parsed
    assert parsed.string == EXPECTED
    # Bytes outside every selected span survive verbatim.
    untouched = [
        'See ',
        ' old] and bare https://bare.test/x.\n',
        '{| class="wikitable"\n',
        '! H1 !! H2\n|-\n|',
        '|| b\n|}\n',
        '<!-- tail -->',
    ]
    for fragment in untouched:
        assert fragment in parsed.string
    assert '<!-- c1 -->' not in parsed.string
    assert 'https://new.example.org/path' in parsed.string


def test_result_is_independent_of_input_order():
    outputs = set()
    for order in permutations(_suggestions()):
        parsed = parse(TEXT)
        apply_edit_suggestions(parsed, order)
        outputs.add(parsed.string)
    assert len(outputs) == 1


def test_empty_suggestion_list_is_a_noop():
    parsed = parse(TEXT)
    assert apply_edit_suggestions(parsed, []) is parsed
    assert parsed.string == TEXT


def test_touching_but_disjoint_spans_do_not_conflict():
    parsed = parse('<!--a-->[https://b c]')
    comment_span = parsed.comments[0].span
    link_span = parsed.external_links[0].span
    assert comment_span[1] == link_span[0]
    apply_edit_suggestions(
        parsed,
        [
            EditSuggestion.delete_comment(comment_span),
            EditSuggestion.replace_external_link(link_span, 'https://d'),
        ],
    )
    assert parsed.string == '[https://d c]'


def test_duplicate_spans_conflict_and_change_nothing():
    parsed = parse(TEXT)
    span = parsed.comments[0].span
    with pytest.raises(EditConflictError) as exc_info:
        apply_edit_suggestions(
            parsed,
            [
                EditSuggestion.delete_comment(span),
                EditSuggestion.delete_comment(span),
            ],
        )
    assert 'overlapping' in str(exc_info.value)
    assert parsed.string == TEXT


def test_nested_comment_and_cell_edits_conflict():
    text = '{|\n| a <!-- in cell --> b\n|}'
    parsed = parse(text)
    cell = parsed.tables[0].cells(row=0, column=0)
    comment = parsed.comments[0]
    assert cell.span[0] < comment.span[0]
    assert comment.span[1] < cell.span[1]
    with pytest.raises(EditConflictError):
        apply_edit_suggestions(
            parsed,
            [
                EditSuggestion.replace_cell(cell.span, 'x'),
                EditSuggestion.delete_comment(comment.span),
            ],
        )
    assert parsed.string == text


def test_cell_and_its_nested_link_conflict_in_either_order():
    text = '{|\n| visit [https://old.example.com site] now\n|}'
    cell_edit = EditSuggestion.replace_cell(
        parse(text).tables[0].cells(row=0, column=0).span, 'x'
    )
    link_edit = EditSuggestion.replace_external_link(
        parse(text).external_links[0].span, 'https://new.example.com'
    )
    for order in (cell_edit, link_edit), (link_edit, cell_edit):
        parsed = parse(text)
        with pytest.raises(EditConflictError):
            apply_edit_suggestions(parsed, order)
        assert parsed.string == text


def test_url_with_whitespace_is_a_syntax_conflict():
    parsed = parse(TEXT)
    span = parsed.external_links[0].span
    with pytest.raises(EditConflictError) as exc_info:
        apply_edit_suggestions(
            parsed,
            [EditSuggestion.replace_external_link(span, 'https://x.test/a b')],
        )
    assert 'break the link' in str(exc_info.value)
    assert parsed.string == TEXT


def test_url_with_bracket_is_a_syntax_conflict():
    parsed = parse(TEXT)
    span = parsed.external_links[0].span
    with pytest.raises(EditConflictError):
        apply_edit_suggestions(
            parsed, [EditSuggestion.replace_external_link(span, 'https://x/[')]
        )
    assert parsed.string == TEXT


@pytest.mark.parametrize(
    'bad_value',
    ['x || y', 'x|y', 'x !! y', 'x\n| y', 'a\r\n!b', 'a\r!b'],
)
def test_cell_value_with_separator_is_a_syntax_conflict(bad_value):
    parsed = parse(TEXT)
    cell = parsed.tables[0].cells(row=1, column=0)
    with pytest.raises(EditConflictError):
        apply_edit_suggestions(
            parsed, [EditSuggestion.replace_cell(cell.span, bad_value)]
        )
    assert parsed.string == TEXT


@pytest.mark.parametrize('good_value', ['line1\nline2', 'a!b', 'plain'])
def test_cell_value_without_structure_break_is_allowed(good_value):
    parsed = parse(TEXT)
    cell = parsed.tables[0].cells(row=1, column=0)
    apply_edit_suggestions(
        parsed, [EditSuggestion.replace_cell(cell.span, good_value)]
    )
    assert good_value in parsed.string


def test_stale_span_points_to_wrong_element_type():
    parsed = parse(TEXT)
    comment_span = parsed.comments[0].span
    with pytest.raises(EditError):
        apply_edit_suggestions(
            parsed, [EditSuggestion.replace_cell(comment_span, 'x')]
        )
    assert parsed.string == TEXT


def test_unknown_target_raises_value_error():
    parsed = parse(TEXT)
    with pytest.raises(ValueError):
        apply_edit_suggestions(parsed, [EditSuggestion('bogus', (0, 1))])


@pytest.mark.parametrize('bad_span', [(-1, 1), (2, 1), (0, 10**9), (0,)])
def test_malformed_span_is_rejected(bad_span):
    parsed = parse(TEXT)
    with pytest.raises((EditError, ValueError)):
        apply_edit_suggestions(
            parsed, [EditSuggestion.delete_comment(bad_span)]
        )


def test_convenience_constructors():
    assert EditSuggestion.delete_comment((1, 2)) == EditSuggestion(
        'comment', (1, 2), None
    )
    assert EditSuggestion.replace_external_link(
        (3, 4), 'https://x'
    ) == EditSuggestion('external_link', (3, 4), 'https://x')
    assert EditSuggestion.replace_cell((5, 6), 'v') == EditSuggestion(
        'cell', (5, 6), 'v'
    )


def test_bare_external_link_url_can_be_replaced():
    parsed = parse('see https://old.test/p page')
    span = parsed.external_links[0].span
    apply_edit_suggestions(
        parsed,
        [EditSuggestion.replace_external_link(span, 'https://new.test')],
    )
    assert parsed.string == 'see https://new.test page'


def test_multiple_disjoint_cells_in_one_table():
    parsed = parse(TEXT)
    table = parsed.tables[0]
    first = table.cells(row=1, column=0)
    second = table.cells(row=1, column=1)
    apply_edit_suggestions(
        parsed,
        [
            EditSuggestion.replace_cell(second.span, 'beta'),
            EditSuggestion.replace_cell(first.span, 'alpha'),
        ],
    )
    assert table.data(row=1) == ['alpha', 'beta']


def test_cell_in_nested_table_is_reachable():
    text = '{|class=outer\n|a\n|\n{|class=inner\n|inner-value\n|}\n|}'
    parsed = parse(text)
    inner = parsed.get_tables(recursive=True)[1]
    cell = inner.cells(row=0, column=0)
    apply_edit_suggestions(
        parsed, [EditSuggestion.replace_cell(cell.span, 'INNER')]
    )
    assert parsed.string == text.replace('inner-value', 'INNER')
