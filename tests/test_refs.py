from wikitextparser import (
    RefDefinition,
    RefOccurrence,
    analyze_refs,
    parse,
)


def test_occurrence_and_definition_tables():
    text = 'a<ref name="r">body</ref>b<ref name="r" />c<ref>anonymous</ref>'
    report = parse(text).refs()
    assert len(report.occurrences) == 3
    assert all(isinstance(o, RefOccurrence) for o in report.occurrences)
    assert len(report.definitions) == 2
    assert all(isinstance(d, RefDefinition) for d in report.definitions)
    definition, reuse, anonymous = report.occurrences
    assert definition.is_definition and not reuse.is_definition
    assert anonymous.is_definition and anonymous.name is None
    first_definition = report.definitions[0]
    assert reuse.definition is first_definition
    assert [o.ordinal for o in first_definition.reuses] == [1]


def test_original_positions_are_recorded():
    text = 'line one\n  <ref name="r">body</ref>'
    occurrence = parse(text).refs().occurrences[0]
    start = text.index('<ref')
    assert (occurrence.start, occurrence.end) == (start, len(text))
    assert (occurrence.line, occurrence.column) == (2, 3)
    assert text[occurrence.start : occurrence.end] == occurrence.tag.string


def test_reference_before_definition_is_matched():
    report = parse('a<ref name="r" />b<ref name="r">body</ref>').refs()
    reuse, definition = report.occurrences
    first_definition = report.definitions[0]
    assert reuse.is_forward_reference is True
    assert reuse.definition is first_definition
    assert definition.is_forward_reference is False
    assert [o.ordinal for o in first_definition.reuses] == [0]


def test_unmatched_reuse_has_no_definition():
    occurrence = parse('<ref name="ghost" />').refs().occurrences[0]
    assert occurrence.definition is None
    assert occurrence.is_forward_reference is False
    assert occurrence.is_definition is False


def test_groups_are_isolated():
    report = parse(
        '<ref group="g" name="r">group body</ref>'
        '<ref group="h" name="r">other body</ref>'
        '<ref group="g" name="r" />'
    ).refs()
    g_def, h_def, g_reuse = report.occurrences
    g_definition, h_definition = report.definitions
    assert g_reuse.definition is g_definition
    assert g_def.definition is g_definition
    assert g_def.group == 'g' and h_def.group == 'h'
    assert report.definitions_for('r', 'g') == [g_definition]
    assert report.definitions_for('r', 'h') == [h_definition]
    assert report.conflicts == {}


def test_conflicting_bodies_are_all_retained():
    report = parse(
        '<ref name="r">first</ref><ref name="r" /><ref name="r">second</ref>'
    ).refs()
    first, reuse, second = report.occurrences
    first_definition, second_definition = report.definitions
    assert len(report.definitions) == 2
    bodies = [d.body for d in report.definitions]
    assert bodies == ['first', 'second']
    (key, conflicting) = next(iter(report.conflicts.items()))
    assert key == ('', 'r')
    assert [d.ordinal for d in conflicting] == [0, 2]
    assert [d.ordinal for d in first_definition.conflicts] == [2]
    assert [d.ordinal for d in second_definition.conflicts] == [0]
    # The reuse still resolves to the first definition, never the last body.
    assert reuse.definition is first_definition
    assert [o.ordinal for o in first_definition.reuses] == [1]
    # Each occurrence keeps its own body rather than sharing a single value.
    assert [o.body for o in report.occurrences] == [
        'first',
        None,
        'second',
    ]


def test_identical_duplicate_bodies_are_not_conflicts():
    report = parse('<ref name="r">same</ref><ref name="r">same</ref>').refs()
    assert len(report.definitions) == 2
    assert report.conflicts == {}
    assert all(d.conflicts == () for d in report.definitions)


def test_self_closing_and_empty_paired_tags():
    report = parse('<ref name="a" /><ref name="a"></ref>').refs()
    reuse, definition = report.occurrences
    first_definition = report.definitions[0]
    assert reuse.is_definition is False and reuse.body is None
    assert definition.is_definition is True and definition.body == ''
    assert reuse.definition is first_definition
    assert report.conflicts == {}


def test_references_tag_group_is_inherited():
    report = parse(
        '<ref group="g" name="r" />'
        '<references group="g">'
        '<ref name="r">defined inside</ref>'
        '</references>'
    ).refs()
    reuse, definition = report.occurrences
    first_definition = report.definitions[0]
    assert reuse.in_references is False
    assert definition.in_references is True
    assert definition.group == 'g'
    assert reuse.definition is first_definition


def test_analyze_refs_function_matches_method():
    parsed = parse('<ref name="r">body</ref><ref name="r" />')
    assert analyze_refs(parsed).as_dict() == parsed.refs().as_dict()


def test_as_dict_and_str_tables():
    report = parse('a<ref name="r">body</ref>b<ref name="r" />').refs()
    tables = report.as_dict()
    assert tables['occurrences'][0] == {
        'ordinal': 0,
        'start': 1,
        'end': 25,
        'line': 1,
        'column': 2,
        'name': 'r',
        'group': '',
        'body': 'body',
        'in_references': False,
        'is_definition': True,
        'is_forward_reference': False,
        'definition_ordinal': 0,
    }
    assert tables['occurrences'][1]['definition_ordinal'] == 0
    assert tables['definitions'][0]['reuse_ordinals'] == [1]
    assert tables['definitions'][0]['conflict_ordinals'] == []
    rendered = str(report)
    assert 'Occurrences:' in rendered and 'Definitions:' in rendered
    assert 'definition' in rendered and 'reference -> def#0' in rendered


def test_ten_occurrences_with_another_groups_definition():
    text = (
        ''.join('<ref name="r">same body</ref>' for _ in range(10))
        + '<ref group="g" name="r">different</ref>'
    )
    report = parse(text).refs()
    default_group = report.occurrences_for('r')
    assert len(default_group) == 10
    assert all(o.group == '' for o in default_group)
    assert len(report.definitions_for('r')) == 10
    assert [d.body for d in report.definitions_for('r', 'g')] == ['different']
    assert report.conflicts == {}


def test_empty_name_is_anonymous():
    report = parse('<ref name="">one</ref><ref name="">two</ref>').refs()
    assert all(o.name is None for o in report.occurrences)
    assert report.conflicts == {}
    assert [d.body for d in report.definitions] == ['one', 'two']


def test_positions_are_global_when_analyzing_a_section():
    text = (
        'intro line\n'
        '== heading ==\n'
        'section body <ref name="r">cite</ref><ref name="r" />\n'
    )
    section = parse(text).sections[1]
    report = section.refs()
    definition, reuse = report.occurrences
    start = text.index('<ref')
    assert definition.start == start
    assert (definition.line, definition.column) == (3, 14)
    assert text[definition.start : definition.end] == definition.tag.string
    assert reuse.definition is report.definitions[0]
