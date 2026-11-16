from categorygraph.extract import extract_category_references

ALIASES = frozenset({'category', 'kategorie'})


def refs(source):
    return [ref.target for ref in extract_category_references(source, ALIASES)]


def test_plain_category_links_become_references():
    assert refs('[[Category:Animals]] [[Category:Dogs|sortkey]]') == [
        'Animals',
        'Dogs',
    ]


def test_leading_colon_link_is_not_membership():
    assert refs('[[:Category:Animals]] [[Category:Dogs]]') == ['Dogs']


def test_ordinary_links_to_other_namespaces_are_ignored():
    source = '[[Main Page]] [[Help:Editing]] [[File:Dog.jpg]] text'
    assert refs(source) == []


def test_protected_text_does_not_create_membership():
    source = (
        '<nowiki>[[Category:HiddenNowiki]]</nowiki>'
        '<!-- [[Category:HiddenComment]] -->'
        '<pre>[[Category:HiddenPre]]</pre>'
        '[[Category:Visible]]'
    )
    assert refs(source) == ['Visible']


def test_links_inside_templates_are_not_direct_declarations():
    source = '{{Navbox|[[Category:InsideTemplate]]}} [[Category:Direct]]'
    assert refs(source) == ['Direct']


def test_links_inside_parser_functions_are_not_direct_declarations():
    source = '{{#if:x|[[Category:InsideFunction]]}} [[Category:Direct]]'
    assert refs(source) == ['Direct']


def test_links_nested_in_another_link_are_not_direct_declarations():
    source = '[[Article|see [[Category:Nested]]]] [[Category:Direct]]'
    assert refs(source) == ['Direct']


def test_namespace_aliases_are_case_insensitive():
    assert refs('[[category:lower]] [[KATEGORIE:Alias]]') == [
        'lower',
        'Alias',
    ]


def test_underscores_and_surrounding_spaces_are_normalized():
    assert refs('[[Category: Red _Fox ]]') == ['Red  Fox']


def test_duplicate_links_count_once():
    source = '[[Category:Cats]] [[Category:Cats]] [[Category:_Cats_]]'
    assert refs(source) == ['Cats']


def test_fragment_is_dropped_from_target_name():
    assert refs('[[Category:Books#Chapter]]') == ['Books']


def test_code_tag_is_not_protected_text():
    # MediaWiki only protects nowiki/pre-like content; <code> is inline.
    assert refs('<code>[[Category:Inline]]</code>') == ['Inline']
