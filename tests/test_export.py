from json import loads

from pytest import raises

import wikitextparser as wtp
from wikitextparser import (
    PageExport,
    apply_revision,
    export_page,
    export_tables,
)

THREE_TABLES = '''\
{| class="wikitable"
|-
! Name !! Value
|-
| {{Card|title={{T|x=1}}}} || x=1
|}
{| class="wikitable"
|-
| {{Card|title={{T|x=1}}}}
|}
{| class="wikitable"
|-
| {{Card|title={{T|x=1}}}}
|}
'''


def test_values_match_table_data():
    text = (
        '{| class=wikitable\n'
        '|-\n'
        '! a !! b\n'
        '|-\n'
        '|  one  || {{t|x=1}}  \n'
        '|}\n'
    )
    export = export_page(text)
    assert export.values == [table.data() for table in wtp.parse(text).tables]
    assert export.values == [[['a', 'b'], ['one', '{{t|x=1}}']]]


def test_span_false_exports_physical_cells():
    text = '{| class=wikitable\n|colspan=2| a || b\n|}'
    assert export_page(text, span=False).values == [[['a', 'b']]]
    assert export_page(text).values == [[['a', 'a', 'b']]]


def test_export_tables_is_an_alias():
    assert export_tables('{|\n|a\n|}').values == [[['a']]]


def test_cell_dependency_is_syntactic():
    text = '{|\n| {{T|x=1}} literal x=1\n|}'
    field = export_page(text).field(0, 0, 0)
    assert field is not None
    assert [
        (d.kind, d.name, d.argument, d.value) for d in field.dependencies
    ] == [('template', 'T', 'x', '1')]


def test_same_text_outside_template_is_not_a_dependency():
    text = '{|\n| no template here: x=1 {{not_arg}}\n|}'
    field = export_page(text).field(0, 0, 0)
    assert field is not None
    assert field.dependencies == ()


def test_nested_and_parser_function_dependencies():
    text = '{|\n| {{A|x={{#if:1|yes|no}}}}\n|}'
    field = export_page(text).field(0, 0, 0)
    assert field is not None
    names = [
        (d.kind, d.name, d.argument) for d in field.dependencies
    ]
    assert names == [
        ('template', 'A', 'x'),
        ('parser_function', '#if', '1'),
        ('parser_function', '#if', '2'),
        ('parser_function', '#if', '3'),
    ]


def test_dependencies_are_only_template_arguments():
    text = (
        '{|\n'
        "| {{T|keep=x=1}} literal x=1 and '{{not_a_call}}'\n"
        '|-\n'
        '| plain x=1\n'
        '|}'
    )
    export = export_page(text)
    first = export.field(0, 0, 0)
    second = export.field(0, 1, 0)
    assert first is not None and second is not None
    assert [
        (d.scope, d.name, d.argument) for d in first.dependencies
    ] == [('cell', 'T', 'keep')]
    assert second.dependencies == ()


def test_one_parameter_only_recomputes_the_affected_cells():
    old = export_page(THREE_TABLES)
    new_text = THREE_TABLES.replace(
        '| {{Card|title={{T|x=1}}}} || x=1',
        '| {{Card|title={{T|x=9}}}} || x=1',
    )
    result = old.apply_revision(new_text, verify=True)
    assert result.recomputed == ((0, 1, 0),)
    assert len(result.reused) == 5
    assert len(result.changes) == 1
    change = result.changes[0]
    assert change.address == (0, 1, 0)
    assert change.old_value == '{{Card|title={{T|x=1}}}}'
    assert change.new_value == '{{Card|title={{T|x=9}}}}'
    assert {c.action for c in change.dependency_changes} == {'changed'}
    arguments = {
        c.new.argument
        for c in change.dependency_changes
        if c.new is not None
    }
    assert arguments == {'title', 'x'}
    explanation = change.describe()
    assert "'1' -> '9'" in explanation
    assert 'Card' in explanation and 'T' in explanation


def test_unrelated_edit_does_not_recompute_cells():
    old = export_page(THREE_TABLES)
    new_text = THREE_TABLES.replace(
        '{| class="wikitable"', '{| class="wikitable" id="new"', 1
    )
    result = old.apply_revision(new_text, verify=True)
    assert result.recomputed == ()
    assert result.changes == ()
    assert len(result.reused) == len(old.fields)


def test_literal_text_edit_recomputes_without_dependency_claim():
    old = export_page('{| class=wikitable\n| {{t|x=1}} old\n|}')
    new_text = 'a prefix\n{| class=wikitable\n| {{t|x=1}} new\n|}'
    result = old.apply_revision(new_text, verify=True)
    assert len(result.changes) == 1
    change = result.changes[0]
    assert change.dependency_changes == ()
    assert 'literal cell markup changed' in change.describe()


def test_moving_a_parameter_into_a_nested_template():
    old_text = '{| class=wikitable\n| before {{T|x=1}} after\n|}'
    new_text = '{| class=wikitable\n| before {{T|x={{N|y=1}}}} after\n|}'
    old = export_page(old_text)
    result = old.apply_revision(new_text, verify=True)
    assert result.recomputed == ((0, 0, 0),)
    change = result.changes[0]
    actions = {
        (c.action, c.old.argument if c.old else None,
         c.new.argument if c.new else None)
        for c in change.dependency_changes
    }
    assert ('added', None, 'y') in actions
    assert ('changed', 'x', 'x') in actions
    # The new nested dependency path records the move.
    added = next(
        c for c in change.dependency_changes if c.action == 'added'
    )
    assert added.new is not None
    assert added.new.path == (
        ('template', 'T', 0),
        ('template', 'N', 0),
    )


def test_moved_argument_is_reported_as_moved():
    old_text = '{| class=wikitable\n| {{A|x=1}}{{B|z=1}}\n|}'
    new_text = '{| class=wikitable\n| {{A}}{{B|x=1|z=1}}\n|}'
    result = export_page(old_text).apply_revision(new_text, verify=True)
    actions = [
        c.action for c in result.changes[0].dependency_changes
    ]
    assert 'moved' in actions


def test_incremental_result_equals_full_extraction():
    old = export_page(THREE_TABLES)
    new_text = THREE_TABLES.replace(
        '| {{Card|title={{T|x=1}}}}', '| {{Card|title={{T|x=2}}}}'
    )
    result = old.apply_revision(new_text)
    full = export_page(new_text)
    assert result.export.to_dict() == full.to_dict()
    assert result.matches_full(new_text)


def test_incremental_result_equals_full_extraction_when_moved():
    old_text = '{| class=wikitable\n| {{Card|title={{T|x=1}}}}\n|}'
    new_text = '{| class=wikitable\n| {{Card|title={{W|t={{T|x=1}}}}}}\n|}'
    result = export_page(old_text).apply_revision(new_text, verify=True)
    assert result.export.values == export_page(new_text).values


def test_verify_raises_on_mismatch():
    old = export_page('{| class=wikitable\n| a\n|}')
    result = old.apply_revision('{| class=wikitable\n| b\n|}')
    assert not result.matches_full('{| class=wikitable\n| c\n|}')


def test_json_round_trip():
    old = export_page(THREE_TABLES)
    json_text = old.dumps()
    loaded = PageExport.loads(json_text)
    assert loaded == old
    assert loads(json_text)['version'] == 1
    result = loaded.apply_revision(
        THREE_TABLES.replace('x=1', 'x=2', 1), verify=True
    )
    assert result.recomputed == ((0, 1, 0),)


def test_file_round_trip(tmp_path):
    old = export_page(THREE_TABLES)
    path = tmp_path / 'export.json'
    old.save(path)
    loaded = PageExport.load(path)
    assert loaded == old
    assert loaded.values == old.values


def test_custom_extractor_must_be_passed_again():
    def extractor(cell):
        return cell.value.upper()

    text = '{|\n| a\n|}'
    old = export_page(text, extractor=extractor)
    assert old.values == [[[' A']]]
    with raises(ValueError):
        PageExport.loads(old.dumps()).apply_revision(text)
    loaded = PageExport.loads(old.dumps(), extractor=extractor)
    assert loaded.apply_revision(text).export.values == [[[' A']]]


def test_apply_revision_accepts_old_wikitext():
    old_text = '{| class=wikitable\n| a\n|}'
    new_text = '{| class=wikitable\n| b\n|}'
    result = apply_revision(old_text, new_text, verify=True)
    assert result.export.values == [[['b']]]


def test_explain_lists_changed_cells():
    old = export_page(THREE_TABLES)
    new_text = THREE_TABLES.replace(
        '| {{Card|title={{T|x=1}}}}', '| {{Card|title={{T|x=5}}}}'
    )
    text = old.apply_revision(new_text).explain()
    assert '3 exported cell(s) changed' in text
    assert 'table 0 cell' in text
    assert 'table 1 cell' in text
    assert 'table 2 cell' in text


def test_move_then_modify_matches_full_extraction_and_explains():
    old_text = '{|\n| {{A|x=1}}\n|}'
    moved_text = '{|\n| {{A|z={{B|x=1}}}}\n|}'
    modified_text = '{|\n| {{A|z={{B|x=2}}}}\n|}'

    old = export_page(old_text)
    moved = old.apply_revision(moved_text, verify=True)
    assert moved.recomputed == ((0, 0, 0),)
    move_actions = [
        c.action for c in moved.changes[0].dependency_changes
    ]
    assert 'moved' in move_actions
    assert 'added' in move_actions

    modified = moved.export.apply_revision(modified_text, verify=True)
    assert modified.recomputed == ((0, 0, 0),)
    assert modified.export == export_page(modified_text)
    nested = [
        c for c in modified.changes[0].dependency_changes
        if c.action == 'changed' and c.new is not None
        and c.new.name == 'B'
    ]
    assert len(nested) == 1
    assert nested[0].old.value == '1'
    assert nested[0].new.value == '2'
    assert "'1' -> '2'" in modified.explain()


def test_cell_dependencies_property():
    cell = wtp.parse('{|\n| {{T|x=1}} plain x=1\n|}').tables[0].cells(0, 0)
    assert cell is not None
    assert [
        (d.kind, d.name, d.argument)
        for d in cell.dependencies
    ] == [('template', 'T', 'x')]


def test_parameter_dependencies():
    text = '{|\n| {{{p|d}}} and {{{q}}}\n|}'
    export = export_page(text)
    field = export.field(0, 0, 0)
    assert field is not None
    assert {
        (d.kind, d.name, d.argument, d.value)
        for d in field.dependencies
    } == {
        ('parameter', 'p', 'name', 'p'),
        ('parameter', 'p', 'default', 'd'),
        ('parameter', 'q', 'name', 'q'),
    }
    result = export.apply_revision(
        text.replace('{{{p|d}}}', '{{{p|z}}}'), verify=True
    )
    assert result.recomputed == ((0, 0, 0),)
    change = result.changes[0]
    assert any(
        c.action == 'changed' and c.old is not None
        and c.old.name == 'p' and c.old.argument == 'default'
        for c in change.dependency_changes
    )


def test_parameter_moved_into_nested_parameter():
    old_text = '{|\n| {{{p|d}}}\n|}'
    new_text = '{|\n| {{{p|{{{r|d}}}}}}\n|}'
    result = export_page(old_text).apply_revision(new_text, verify=True)
    assert result.recomputed == ((0, 0, 0),)
    added = [
        c.new
        for c in result.changes[0].dependency_changes
        if c.action == 'added'
    ]
    assert added and all(d is not None for d in added)
    assert {d.name for d in added} == {'r'}


def test_nested_tables_are_exported_separately():
    text = (
        '{|\n'
        '| outer\n'
        '{| class=inner\n'
        '| {{T|x=1}}\n'
        '|}\n'
        '|}'
    )
    export = export_page(text)
    assert len(export.tables) == 2
    assert export.values == [table.data() for table in wtp.parse(text).tables]
    new_text = text.replace('x=1', 'x=2')
    result = export.apply_revision(new_text, verify=True)
    # The outer cell contains the nested table markup, so both changed.
    assert result.recomputed == ((0, 0, 0), (1, 0, 0))
    assert result.reused == ()


def test_inserting_unrelated_markup_does_not_recompute():
    old_text = '{|\n| a\n|}'
    new_text = '{{Unrelated|z=1}}\n' + old_text
    result = export_page(old_text).apply_revision(new_text, verify=True)
    assert result.recomputed == ()
    assert result.changes == ()


def test_identical_cell_text_at_two_addresses_is_independent():
    old_text = '{|\n| {{T|x=1}}\n|-\n| {{T|x=1}}\n|}'
    new_text = '{|\n| {{T|x=1}}\n|-\n| {{T|x=2}}\n|}'
    result = export_page(old_text).apply_revision(new_text, verify=True)
    assert result.recomputed == ((0, 1, 0),)
    assert len(result.reused) == 1


def test_inserting_a_table_does_not_invalidate_later_tables():
    old_text = (
        '{| class=a\n| {{T|x=1}}\n|}\n'
        '{| class=b\n| {{T|x=1}}\n|}\n'
        '{| class=c\n| {{T|x=1}}\n|}\n'
    )
    new_text = (
        '{| class=a\n| {{T|x=1}}\n|}\n'
        '{| class=x\n| inserted\n|}\n'
        '{| class=b\n| {{T|x=1}}\n|}\n'
        '{| class=c\n| {{T|x=1}}\n|}\n'
    )
    result = export_page(old_text).apply_revision(new_text, verify=True)
    assert result.recomputed == ((1, 0, 0),)
    assert len(result.reused) == 3
    assert result.export.values[2] == [['{{T|x=1}}']]
    assert result.table_map == {0: 0, 1: 2, 2: 3}


def test_deleting_a_table_keeps_other_values():
    old_text = (
        '{| class=a\n| A\n|}\n'
        '{| class=b\n| B\n|}\n'
        '{| class=c\n| C\n|}\n'
    )
    new_text = '{| class=b\n| B\n|}\n{| class=c\n| C\n|}\n'
    result = export_page(old_text).apply_revision(new_text, verify=True)
    assert result.recomputed == ()
    assert result.changes == ()
    assert result.table_map == {1: 0, 2: 1}


def test_added_and_removed_rows_are_explained():
    old_text = '{| class=a\n| A || B\n|-\n| C || D\n|}'
    new_text = '{| class=a\n| A || B\n|-\n| C || D\n|-\n| E || F\n|}'
    result = export_page(old_text).apply_revision(new_text, verify=True)
    assert result.recomputed == ((0, 2, 0), (0, 2, 1))
    assert len(result.changes) == 2
    assert 'added with value' in result.explain()

    removed = result.export.apply_revision(old_text, verify=True)
    assert removed.removed == ((0, 2, 0), (0, 2, 1))
    assert 'were removed' in removed.explain()


def test_empty_page_export():
    export = export_page('no tables')
    assert export.values == []
    assert export.fields == ()
    assert len(export) == 0


def test_caption_is_exported():
    text = '{| class="wikitable"\n|+ Food\n|-\n| a\n|}'
    export = export_page(text)
    assert export.table(0).caption == ' Food'
