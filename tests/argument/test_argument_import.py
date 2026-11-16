from __future__ import annotations

from wikitextparser import (
    ArgumentDelete,
    ArgumentInsert,
    ArgumentSet,
    Template,
    import_arguments,
)


def test_occurrences_keep_explicit_and_positional_one_apart():
    t = Template('{{t|1=a|x}}')
    report = t.import_arguments()
    one, two = report.occurrences_before
    assert (one.name, one.value, one.positional) == ('1', 'a', False)
    assert (two.name, two.value, two.positional) == ('1', 'x', True)
    assert report.effective_before['1'].value == 'x'
    assert report.effective_before['1'].occurrence == 2
    assert report.conflicts_before[0].values == ('a', 'x')
    assert report.string_before == '{{t|1=a|x}}'
    assert not report.changed


def test_named_identity_ignores_surrounding_whitespace():
    report = Template('{{t| a = 1 |a=2|  a  =3}}').import_arguments()
    assert [o.name for o in report.occurrences_before] == ['a', 'a', 'a']
    assert [o.raw_name for o in report.occurrences_before] == [
        ' a ',
        'a',
        '  a  ',
    ]
    assert [o.value for o in report.occurrences_before] == ['1', '2', '3']
    assert report.effective_before['a'].value == '3'


def test_positional_identity_counts_only_positional_occurrences():
    report = Template('{{t|1=named|first|second}}').import_arguments()
    assert [o.name for o in report.occurrences_before] == ['1', '1', '2']
    assert [o.positional for o in report.occurrences_before] == [
        False,
        True,
        True,
    ]


def test_positional_value_keeps_whitespace_but_named_value_trims():
    report = Template('{{t| v |n= w }}').import_arguments()
    first, second = report.occurrences_before
    assert (first.value, first.raw_value) == (' v ', ' v ')
    assert (second.value, second.raw_value) == ('w', ' w ')


def test_delete_does_not_shift_the_position_of_later_requests():
    t = Template('{{t|a|b|c|d}}')
    report = t.import_arguments([ArgumentDelete(1), ArgumentSet(3, 'C')])
    assert report.string_after == '{{t|b|C|d}}'
    assert report.positional_shifts[0].before_name == '2'
    assert report.positional_shifts[0].after_name == '1'
    assert [o.origin for o in report.occurrences_after] == [2, 3, 4]


def test_delete_reveals_earlier_value_for_the_same_name():
    t = Template('{{t|a=1|a=2}}')
    report = t.import_arguments([ArgumentDelete(2)])
    assert report.string_after == '{{t|a=1}}'
    assert report.effective_after['a'].value == '1'
    (change,) = report.effective_changes
    assert change.kind == 'revealed'
    assert change.before.value == '2'
    assert change.after.value == '1'
    assert 'revealed' in report.describe()


def test_delete_explicit_one_reveals_the_positional_first_argument():
    t = Template('{{t|1=a|x|b}}')
    report = t.import_arguments([ArgumentDelete(2)])
    assert report.string_after == '{{t|1=a|b}}'
    # The surviving positional |b is now argument 1, behind |1=a.
    assert report.effective_after['1'].value == 'b'
    assert report.positional_shifts[0].before_name == '2'
    assert report.positional_shifts[0].after_name == '1'


def test_inserting_a_positional_renumbers_later_positional_occurrences():
    t = Template('{{t|a|b}}')
    report = t.import_arguments([ArgumentInsert(0, 'z')])
    assert report.string_after == '{{t|z|a|b}}'
    assert [
        (s.original, s.before_name, s.after_name)
        for s in report.positional_shifts
    ] == [(1, '1', '2'), (2, '2', '3')]
    assert report.effective_after['1'].value == 'z'
    assert report.effective_after['2'].value == 'a'
    assert report.effective_after['3'].value == 'b'


def test_insert_anchor_refers_to_the_original_list():
    t = Template('{{t|a|b|c}}')
    report = t.import_arguments(
        [ArgumentDelete(2), ArgumentInsert(3, 'Z', where='before')]
    )
    assert report.string_after == '{{t|a|Z|c}}'


def test_inserted_positional_does_not_change_a_delete_target():
    t = Template('{{t|a|b}}')
    report = t.import_arguments([ArgumentInsert(0, 'Z'), ArgumentDelete(1)])
    assert report.string_after == '{{t|Z|b}}'


def test_delete_and_set_on_the_same_occurrence_conflict():
    t = Template('{{t|a|b}}')
    report = t.import_arguments([ArgumentDelete(1), ArgumentSet(1, 'X')])
    assert report.string_after == '{{t|a|b}}'
    assert report.request_conflicts[0].occurrence == 1
    assert report.request_conflicts[0].requests == (1, 2)


def test_unknown_position_is_reported_without_aborting_the_batch():
    t = Template('{{t|a|b}}')
    report = t.import_arguments([ArgumentSet(9, 'X'), ArgumentSet(2, 'B')])
    assert report.string_after == '{{t|a|B}}'
    assert report.errors[0].request == 1
    assert 'no occurrence at original index 9' in report.errors[0].reason


def test_named_set_requires_conversion_to_keep_its_name():
    t = Template('{{t|a|b}}')
    report = t.import_arguments([ArgumentSet(2, 'X', positional=True)])
    assert report.string_after == '{{t|a|X}}'
    assert report.effective_after['2'].value == 'X'

    t = Template('{{t|1=a|2=b}}')
    report = t.import_arguments([ArgumentSet(2, 'X', positional=True)])
    assert report.string_after == '{{t|1=a|2=b}}'
    assert report.errors[0].request == 1
    assert 'renumber' in report.errors[0].reason


def test_set_name_converts_positional_to_named():
    t = Template('{{t|a|b}}')
    report = t.import_arguments([ArgumentSet(1, 'v', name='n')])
    assert report.string_after == '{{t|n=v|b}}'
    assert [
        (s.original, s.before_name, s.after_name)
        for s in report.positional_shifts
    ] == [(2, '2', '1')]


def test_insert_named_and_positional():
    t = Template('{{t|a}}')
    report = t.import_arguments(
        [ArgumentInsert(1, 'v', name='n'), ArgumentInsert(1, 'z')]
    )
    assert report.string_after == '{{t|a|n=v|z}}'
    assert [o.name for o in report.occurrences_after] == ['1', 'n', '2']


def test_mapping_requests_are_accepted():
    t = Template('{{t|a|b}}')
    report = import_arguments(
        t,
        [
            {'action': 'delete', 'position': 1},
            {'action': 'insert', 'at': 2, 'value': 'Z'},
        ],
    )
    assert report.string_after == '{{t|b|Z}}'


def test_top_level_pipe_is_rejected_but_nested_pipe_is_allowed():
    t = Template('{{t|a}}')
    report = t.import_arguments([ArgumentInsert(1, 'x|y')])
    assert report.string_after == '{{t|a}}'
    assert 'template arguments' in report.errors[0].reason

    report = t.import_arguments([ArgumentInsert(1, '{{x|y}}')])
    assert report.string_after == '{{t|a|{{x|y}}}}'


def test_delete_all_arguments_keeps_a_well_formed_template():
    t = Template('{{t|a|b}}')
    report = t.import_arguments([ArgumentDelete(1), ArgumentDelete(2)])
    assert report.string_after == '{{t}}'
    assert report.occurrences_after == ()


def test_insert_at_end_and_before_first():
    t = Template('{{t|a|b}}')
    assert t.import_arguments([ArgumentInsert(2, 'z')]).string_after == (
        '{{t|a|b|z}}'
    )
    t = Template('{{t|a|b}}')
    assert (
        t.import_arguments(
            [ArgumentInsert(1, 'z', where='before')]
        ).string_after
        == '{{t|z|a|b}}'
    )


def test_bad_insert_position_is_reported():
    t = Template('{{t|a|b}}')
    report = t.import_arguments([ArgumentInsert(3, 'z')])
    assert report.string_after == '{{t|a|b}}'
    assert report.errors[0].request == 1

    t = Template('{{t|a|b}}')
    report = t.import_arguments([ArgumentInsert(0, 'z', where='before')])
    assert report.errors[0].request == 1


def test_conflicts_are_reported_per_name_not_per_occurrence():
    t = Template('{{t|a=1|a=2|a=2|b=1|b=1}}')
    report = t.import_arguments()
    assert [c.name for c in report.conflicts_before] == ['a']
    assert report.conflicts_before[0].occurrences == (1, 2, 3)
    assert report.conflicts_before[0].values == ('1', '2', '2')


def test_report_is_json_serializable():
    report = Template('{{t|a|b}}').import_arguments([ArgumentDelete(1)])
    data = report.to_dict()
    assert data['string_after'] == '{{t|b}}'
    assert data['occurrences_after'][0]['origin'] == 2
    assert data['positional_shifts'][0]['after_name'] == '1'


def test_template_helper_and_module_helper_agree():
    t = Template('{{t|a|b}}')
    first = t.import_arguments([ArgumentDelete(1)])
    second = Template('{{t|a|b}}').apply_argument_changes([ArgumentDelete(1)])
    assert first.to_dict() == second.to_dict()


def test_delete_duplicate_named_occurrence_updates_the_effective_mapping():
    t = Template('{{t|a=1|a=2|a=3}}')
    report = t.import_arguments([ArgumentDelete(2)])
    assert report.string_after == '{{t|a=1|a=3}}'
    assert report.effective_after['a'].value == '3'
    assert [c.kind for c in report.effective_changes] == ['occurrences']


def test_deleting_the_winning_occurrence_of_a_duplicate():
    t = Template('{{t|a=1|a=2}}')
    report = t.import_arguments([ArgumentDelete(2)])
    assert (
        report.resolved_conflicts and report.resolved_conflicts[0].name == 'a'
    )
    assert report.new_conflicts == ()


def test_occurrence_shift_keeps_original_anchor():
    t = Template('{{t|a|b|c}}')
    report = t.import_arguments([ArgumentInsert(0, 'z')])
    by_origin = {
        o.origin: o for o in report.occurrences_after if o.origin is not None
    }
    assert by_origin[1].name == '2'
    assert by_origin[1].value == 'a'


def test_invalid_request_type_is_recorded():
    t = Template('{{t|a}}')
    report = t.import_arguments(['nope'])
    assert report.string_after == '{{t|a}}'
    assert report.errors[0].request == 1


def test_empty_report_has_no_changes():
    report = Template('{{t|a|b}}').import_arguments()
    assert not report.changed
    assert report.describe() == 'No argument changes.'


def test_bool_is_not_accepted_as_an_index():
    report = Template('{{t|a}}').import_arguments(
        [{'action': 'delete', 'index': True}]
    )
    assert report.string_after == '{{t|a}}'
    assert report.errors[0].request == 1


def test_template_occurrence_properties():
    t = Template('{{t| a = 1 |x| 1 = 2 }}')
    assert [(o.name, o.value) for o in t.argument_occurrences] == [
        ('a', '1'),
        ('1', 'x'),
        ('1', '2'),
    ]
    assert t.effective_arguments['1'].value == '2'
    assert t.argument_conflicts[0].name == '1'


def test_report_is_a_sequence_of_original_occurrences():
    report = Template('{{t|a|b}}').import_arguments([ArgumentDelete(1)])
    assert len(report) == 2
    assert report[0].name == '1'
    assert [o.value for o in report] == ['a', 'b']
    assert [o.value for o in report[:1]] == ['a']
