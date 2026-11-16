from __future__ import annotations

from json import loads

from pytest import raises

from wikitextparser import (
    FieldSpec,
    FieldStatus,
    TemplateSchema,
    extract_fields,
    parse,
    resolve_value,
)

SCHEMA = TemplateSchema(
    {
        'Infobox person': {
            'name': ('name', '1'),
            'birth_date': ('birth_date', 'birthdate'),
            'website': 'website',
            'occupation': 'occupation',
        },
    }
)


def test_literal_and_expansion_dependent_are_separated():
    source = (
        '{{Infobox person'
        '|name=Alice'
        '|birth_date={{Birth date|1990|1|1}}'
        '|website=[https://example.org Alice]'
        '}}'
    )
    report = extract_fields(source, SCHEMA)

    assert report.literal_values == {
        'name': 'Alice',
        'website': '[https://example.org Alice]',
    }
    assert [f.field for f in report.expansion_dependent] == ['birth_date']
    assert report.expansion_dependent[0].value == '{{Birth date|1990|1|1}}'
    assert report.expansion_dependent[0].constructs[0].kind == 'template'
    assert report.expansion_dependent[0].constructs[0].detail == 'Birth date'


def test_absent_fields_are_not_silently_literal():
    report = extract_fields('{{Infobox person|name=Alice}}', SCHEMA)
    assert {f.field for f in report.absent} == {
        'birth_date',
        'website',
        'occupation',
    }
    assert report.by_field()['occupation'].value is None


def test_declared_default_is_used_when_argument_is_missing():
    schema = TemplateSchema(
        {
            'T': {'f': FieldSpec(('f',), default='fallback')},
        }
    )
    field = extract_fields('{{T}}', schema).by_field()['f']
    assert field.status is FieldStatus.LITERAL
    assert field.value == 'fallback'
    assert field.default_used is False
    assert field.schema_default_used is True
    assert 'used the declared default' in field.basis


def test_missing_argument_without_default_is_absent_not_empty():
    schema = TemplateSchema({'T': {'f': 'f'}})
    field = extract_fields('{{T}}', schema).by_field()['f']
    assert field.status is FieldStatus.ABSENT
    assert field.value is None


def test_empty_argument_is_literal_empty_string():
    schema = TemplateSchema({'T': {'f': 'f'}})
    field = extract_fields('{{T|f=}}', schema).by_field()['f']
    assert field.status is FieldStatus.LITERAL
    assert field.value == ''
    assert field.argument_present is True
    assert 'empty string' in field.basis


def test_whitespace_only_keyword_argument_counts_as_empty():
    schema = TemplateSchema({'T': {'f': 'f'}})
    field = extract_fields('{{T|f=   }}', schema).by_field()['f']
    assert field.status is FieldStatus.LITERAL
    assert field.value == ''


def test_positional_and_keyword_parameters_are_tried_in_order():
    schema = TemplateSchema(
        {
            'T': {'f': ('2', 'f')},
        }
    )
    assert extract_fields('{{T|x|y}}', schema).by_field()['f'].value == 'y'
    assert extract_fields('{{T|f=z}}', schema).by_field()['f'].value == 'z'
    assert extract_fields('{{T|2=w}}', schema).by_field()['f'].parameter == '2'


def test_comments_are_removed_from_the_stored_value():
    schema = TemplateSchema({'T': {'f': 'f'}})
    field = extract_fields('{{T|f=Al<!--x-->ice}}', schema).by_field()['f']
    assert field.status is FieldStatus.LITERAL
    assert field.value == 'Alice'
    assert field.raw == 'Al<!--x-->ice'
    assert 'comments removed' in field.basis


def test_nowiki_contents_are_literal_but_other_tags_are_not():
    schema = TemplateSchema({'T': {'f': 'f', 'g': 'g'}})
    report = extract_fields(
        '{{T|f=<nowiki>{{U}}</nowiki>|g=<ref>{{U}}</ref>}}', schema
    )
    nowiki = report.by_field()['f']
    ref = report.by_field()['g']
    assert nowiki.status is FieldStatus.LITERAL
    assert nowiki.value == '{{U}}'
    assert ref.status is FieldStatus.EXPANSION_DEPENDENT
    assert ref.constructs[0].kind == 'extension-tag'
    assert ref.constructs[0].detail == 'ref'


def test_parser_functions_magic_words_and_signatures_need_expansion():
    schema = TemplateSchema(
        {
            'T': {'pf': 'pf', 'mw': 'mw', 'sig': 'sig'},
        }
    )
    source = '{{T|pf={{#if:1|y|n}}|mw={{PAGENAME}}|sig=~~~}}'
    report = extract_fields(source, schema)
    kinds = {
        f.field: [c.kind for c in f.constructs]
        for f in report.expansion_dependent
    }
    assert kinds == {
        'pf': ['parser-function'],
        'mw': ['parser-function'],
        'sig': ['signature'],
    }


def test_evidence_spans_point_at_the_construct_in_the_source():
    schema = TemplateSchema({'T': {'f': 'f'}})
    source = 'lead {{T|f=pre {{U|x}} post}} tail'
    field = extract_fields(source, schema).by_field()['f']
    construct = field.constructs[0]
    assert construct.text == '{{U|x}}'
    assert source[slice(*construct.span)] == construct.text
    assert field.span is not None
    assert source[slice(*field.span)] == field.raw


def test_unbound_parameter_without_default_is_not_a_literal():
    schema = TemplateSchema({'T': {'f': 'f'}})
    field = extract_fields('{{T|f={{{x}}}}}', schema).by_field()['f']
    assert field.status is FieldStatus.UNBOUND
    assert field.value is None
    assert 'unbound and has no default' in field.basis


def test_binding_value_is_substituted():
    schema = TemplateSchema({'T': {'f': 'f'}})
    field = extract_fields(
        '{{T|f={{{x|d}}}}}', schema, bindings={'x': 'v'}
    ).by_field()['f']
    assert field.status is FieldStatus.LITERAL
    assert field.value == 'v'
    assert field.bindings == (('x', 'bound'),)


def test_empty_binding_and_no_binding_have_different_effects():
    schema = TemplateSchema({'T': {'f': 'f'}})
    source = '{{T|f={{{x|DEFAULT}}}}}'

    unbound = extract_fields(source, schema).by_field()['f']
    empty = extract_fields(source, schema, bindings={'x': ''}).by_field()['f']

    assert unbound.status is empty.status is FieldStatus.LITERAL
    assert unbound.value == 'DEFAULT'
    assert empty.value == ''
    assert unbound.parameter_defaults_used == ('x',)
    assert empty.parameter_defaults_used == ()
    assert unbound.bindings == (('x', 'unbound-default'),)
    assert empty.bindings == (('x', 'bound-empty'),)
    assert 'default is not used' in empty.basis


def test_nested_defaults_require_the_binding_condition():
    # {{{x|{{{y|Y}}}}}}: Y may only be used when y is unbound.
    source = '{{T|f={{{x|{{{y|Y}}}}}}}}'
    schema = TemplateSchema({'T': {'f': 'f'}})

    assert extract_fields(source, schema).by_field()['f'].value == 'Y'
    assert (
        extract_fields(source, schema, bindings={'y': ''})
        .by_field()['f']
        .value
        == ''
    )
    assert (
        extract_fields(source, schema, bindings={'y': 'z'})
        .by_field()['f']
        .value
        == 'z'
    )
    # Binding x to empty prevents both nested defaults from being used.
    assert (
        extract_fields(source, schema, bindings={'x': ''})
        .by_field()['f']
        .value
        == ''
    )


def test_schema_default_may_itself_be_binding_dependent():
    schema = TemplateSchema(
        {
            'T': {'f': FieldSpec(('f',), default='{{{x|FALLBACK}}}')},
        }
    )
    unbound = extract_fields('{{T}}', schema).by_field()['f']
    bound = extract_fields('{{T}}', schema, bindings={'x': ''}).by_field()['f']
    assert unbound.value == 'FALLBACK'
    assert unbound.schema_default_used is True
    assert unbound.parameter_defaults_used == ('x',)
    assert bound.value == ''
    assert bound.parameter_defaults_used == ()


def test_binding_condition_applies_to_argument_values_too():
    schema = TemplateSchema({'T': {'f': 'f'}})
    source = '{{T|f={{{flag|on}}}}}'
    assert extract_fields(source, schema).by_field()['f'].value == 'on'
    assert (
        extract_fields(source, schema, bindings={'flag': ''})
        .by_field()['f']
        .value
        == ''
    )


def test_template_names_are_normalized():
    schema = TemplateSchema({'template:Infobox_person': {'name': 'name'}})
    for source in (
        '{{Infobox person|name=A}}',
        '{{Template:Infobox_person|name=A}}',
        '{{infobox person|name=A}}',
    ):
        assert extract_fields(source, schema).by_field()['name'].value == 'A'


def test_last_invocation_wins_and_the_basis_says_so():
    schema = TemplateSchema({'T': {'f': 'f'}})
    field = extract_fields('{{T|f=a}}{{T|f=b}}', schema).by_field()['f']
    assert field.value == 'b'
    assert 'occurs 2 times' in field.basis


def test_template_filter_limits_the_report():
    schema = TemplateSchema(
        {
            'A': {'x': 'x'},
            'B': {'y': 'y'},
        }
    )
    report = extract_fields('{{A|x=1}}{{B|y=2}}', schema, template='B')
    assert [f.field for f in report] == ['y']


def test_report_serializes_to_json():
    report = extract_fields(
        '{{Infobox person|name=Alice|birth_date={{D|1}}}}', SCHEMA
    )
    data = loads(report.to_json())
    assert data['literal_values'] == {'name': 'Alice'}
    assert data['expansion_dependent'][0]['field'] == 'birth_date'
    assert data['expansion_dependent'][0]['basis']


def test_character_references_are_flagged():
    schema = TemplateSchema({'T': {'f': 'f'}})
    field = extract_fields('{{T|f=a&amp;b}}', schema).by_field()['f']
    assert field.status is FieldStatus.LITERAL
    assert field.value == 'a&amp;b'
    assert 'character references' in field.basis


def test_resolve_value_reports_binding_state():
    assert resolve_value('{{{a|A}}}').value == 'A'
    assert resolve_value('{{{a|A}}}').binding == 'unbound'
    assert resolve_value('{{{a|A}}}', bindings={'a': ''}).value == ''
    assert (
        resolve_value('{{{a|A}}}', bindings={'a': ''}).binding == 'bound-empty'
    )
    assert resolve_value('{{{a|A}}}', bindings={'a': 'Z'}).binding == 'bound'
    assert resolve_value('{{{a}}}').status is FieldStatus.UNBOUND


def test_resolve_value_treats_nowiki_as_literal():
    resolution = resolve_value('<nowiki>{{U}}</nowiki>')
    assert resolution.status is FieldStatus.LITERAL
    assert resolution.value == '{{U}}'


def test_schema_rejects_unknown_field_spec_keys():
    with raises(ValueError, match='unknown FieldSpec keys'):
        TemplateSchema({'T': {'f': {'param': 'f'}}})


def test_schema_rejects_empty_field_names():
    with raises(ValueError, match='non-empty'):
        TemplateSchema({'T': {'': 'f'}})


def test_bindings_must_be_strings():
    schema = TemplateSchema({'T': {'f': 'f'}})
    with raises(TypeError, match='must be a string'):
        extract_fields('{{T}}', schema, bindings={'x': 1})


def test_unknown_template_filter_raises():
    with raises(ValueError, match='not in the schema'):
        extract_fields('{{T}}', SCHEMA, template='Other')


def test_unbound_parameter_wins_over_a_same_level_template():
    schema = TemplateSchema({'T': {'f': 'f'}})
    field = extract_fields('{{T|f={{{x}}} {{U}}}}', schema).by_field()['f']
    assert field.status is FieldStatus.UNBOUND
    assert field.value is None
    assert [c.kind for c in field.constructs] == ['template']
    assert 'without a binding' in field.basis


def test_parameter_nested_inside_a_template_does_not_downgrade_it():
    schema = TemplateSchema({'T': {'f': 'f'}})
    field = extract_fields('{{T|f={{U|{{{x}}}}}}}', schema).by_field()['f']
    assert field.status is FieldStatus.EXPANSION_DEPENDENT
    assert field.value == '{{U|{{{x}}}}}'


def test_wikitext_method_matches_function():
    parsed = parse('{{T|f=v}}')
    schema = TemplateSchema({'T': {'f': 'f'}})
    assert parsed.extract_fields(schema).literal_values == {'f': 'v'}
    assert (
        parsed.extract_fields(schema, template='T').literal_values
        == extract_fields('{{T|f=v}}', schema).literal_values
    )
