from wikitextparser import WikiList, parse, parse_list_items


def test_lists():
    assert repr(WikiList('# a\n## b', '#').get_lists()) == "[WikiList('## b')]"


def test_subitem_are_part_of_item():
    """A few basic examples from [[mw:Help:Lists]]."""
    ul = WikiList(
        '* Lists are easy to do:\n'
        '** start every line\n'
        '* with a star\n'
        '** more stars mean\n'
        '*** deeper levels',
        pattern=r'\*',
    )
    items = ul.items
    assert items == [' Lists are easy to do:', ' with a star']
    fullitems = ul.fullitems
    assert fullitems == [
        '* Lists are easy to do:\n** start every line\n',
        '* with a star\n** more stars mean\n*** deeper levels',
    ]


def test_commented_list_item():
    """One of the list items is commented through the wikitext."""
    wl = WikiList(
        '#1\n##1-1\n##1-2<!-- \n##1-3\n -->\n#2\n',
        pattern=r'\#',
    )
    assert wl.items[1] == '2'


def test_dont_return_shadow():
    wl = WikiList('#1 {{t}}', pattern=r'\#')
    assert wl.items[0] == '1 {{t}}'


def test_subitems_for_the_first_item():
    wl = WikiList('# 0\n## 0.0\n## 0.1\n#* 0.0\n# 2\n', pattern=r'\#')
    items = wl.items[0]
    assert items == ' 0'
    subitems0 = wl.sublists(0, r'\#')[0]
    assert subitems0.items == [' 0.0', ' 0.1']
    # Test to see that arguments are optional.
    sublists = wl.sublists()
    assert sublists[1].fullitems == ['#* 0.0\n']
    assert sublists[0].items == [' 0.0', ' 0.1']


def test_subitems_for_the_second_item():
    parsed = parse(
        'text\n'
        '* list item a\n'
        '* list item b\n'
        '** sub-list of b\n'
        '* list item c\n'
        'text'
    )
    wikilist = parsed.get_lists(pattern=r'\*')[0]
    assert wikilist.items == [' list item a', ' list item b', ' list item c']
    sublist = wikilist.sublists(1, r'\*')[0]
    assert sublist.items == [' sub-list of b']


def test_link_in_definition_list():
    wl = WikiList('; https://github.com : definition', pattern=r'[:;]\s*')
    assert wl.items == ['https://github.com ', ' definition']
    wl = WikiList('; https://a.b : c https://d.e : f', pattern=r'[:;]\s*')
    assert wl.items == ['https://a.b ', ' c https://d.e : f']


def test_mixed_definition_lists():
    wl = WikiList(
        '; Mixed definition lists\n'
        '; item 1 : definition\n'
        ':; sub-item 1 plus term\n'
        ':: two colons plus definition\n'
        ':; sub-item 2 : colon plus definition\n'
        '; item 2 \n'
        ': back to the main list\n',
        pattern=r'[:;]\s*',
    )
    assert wl.items == [
        'Mixed definition lists',
        'item 1 ',
        ' definition',
        'item 2 ',
        'back to the main list',
    ]


def test_order_definition_lists():
    wl = WikiList('; Item 1 : definition 1\n', pattern=r'[:;]\s*')
    assert wl.items == ['Item 1 ', ' definition 1']
    assert wl.fullitems == ['; Item 1 : definition 1\n', ': definition 1']


def test_travese_mixed_list_completely():
    wl = WikiList(
        '* Or create mixed lists\n'
        '*# and nest them\n'
        '*#* like this\n'
        '*#*; definitions\n'
        '*#*: work:\n'
        '*#*; apple\n'
        '*#*; banana\n'
        '*#*: fruits',
        pattern=r'\*',
    )
    assert wl.items == [' Or create mixed lists']
    swl = wl.sublists(0, r'\#')[0]
    assert swl.items == [' and nest them']
    sswl = swl.sublists(0, r'\*')[0]
    assert sswl.items == [' like this']
    ssswl = sswl.sublists(0, '[;:]')[0]
    assert ssswl.items == [
        ' definitions',
        ' work:',
        ' apple',
        ' banana',
        ' fruits',
    ]


def test_convert():
    wl = WikiList(
        ':*A1\n:*#B1\n:*#B2\n:*:continuing A1\n:*A2',
        pattern=r':\*',
    )
    assert wl.level == 2
    wl.convert('#')
    assert wl.string == ('#A1\n##B1\n##B2\n#:continuing A1\n#A2')
    assert wl.pattern == r'\#'
    assert wl.level == 1


def test_cache_update():
    wl = WikiList('*a {{t}}', pattern=r'\*')
    wl.templates[0].name = 'ttt'
    assert wl.string == '*a {{ttt}}'


# todo: check if ref tags can contain lists and add a test for it.


def test_external_link_colon_shadow():  # 91
    assert parse('===abc===\n; https://github.com : definition\n').get_lists(
        r'[:;]'
    )[0]._list_shadow == bytearray(b'; __________________ : definition\n')


def test_sublist_for_missing_level():
    assert parse('**1\n**2\n').get_lists()[0].sublists(0)[0].items == [
        '1',
        '2',
    ]


def test_multiline_with_carriage_return():
    parsed = parse(
        'text\n'
        '* list item a\r\n'
        '* list item b\r'
        '** sub-list of b\n'
        '* list item c\r'
        'text'
    )
    wikilist = parsed.get_lists(pattern=r'\*')[0]
    assert wikilist.items == [' list item a', ' list item b', ' list item c']
    sublist = wikilist.sublists(1, r'\*')[0]
    assert sublist.items == [' sub-list of b']

    parsed = parse(
        'text\r'
        '# list item a\r'
        '# list item b\r'
        '## sub-list of b\n'
        '# list item c\r'
        'text'
    )
    wikilist = parsed.get_lists(pattern=r'#')[0]
    assert wikilist.items == [' list item a', ' list item b', ' list item c']
    sublist = wikilist.sublists(1, r'#')[0]
    assert sublist.items == [' sub-list of b']


def test_parse_list_items_mixed_markers():
    text = '* a\n** b\n*# c\n; term : def\n: def2\n# num\n'
    items = parse_list_items(text)
    assert [i.marker for i in items] == ['*', '**', '*#', ';', ':', '#']
    assert [i.level for i in items] == [1, 2, 2, 1, 1, 1]
    assert [i.body for i in items] == [
        ' a',
        ' b',
        ' c',
        ' term : def',
        ' def2',
        ' num',
    ]
    # spans are in source order and map back to the source text
    spans = [i.span for i in items]
    assert spans == sorted(spans)
    for item, (s, e) in zip(items, spans):
        assert text[s:e] == item.marker + item.body


def test_parse_list_items_parents():
    text = '* a\n** b\n*# c\n*** d\n* e\n'
    a, b, c, d, e = parse_list_items(text)
    assert a.parent is None
    assert b.parent is a
    assert c.parent is a
    # '*#' is not a prefix of '***', so d is re-parented to a
    assert d.parent is a
    assert e.parent is None
    assert a.children == [b, c, d]
    assert b.children == []
    assert c.children == []
    # a single indentation change re-parents the following items
    a, b, c = parse_list_items('* a\n** b\n** c\n')
    assert b.parent is a
    assert c.parent is a
    a, b, c = parse_list_items('* a\n** b\n*# c\n')
    assert b.parent is a
    assert c.parent is a


def test_parse_list_items_continuation_lines():
    items = parse_list_items('* a\nwrapped a\n** b\nwrapped b\n* c\n')
    assert [i.body for i in items] == [
        ' a\nwrapped a',
        ' b\nwrapped b',
        ' c',
    ]
    a, b, c = items
    assert b.parent is a
    assert c.parent is None
    text = '* a\nwrapped a\n** b\nwrapped b\n* c\n'
    for item in items:
        s, e = item.span
        assert text[s:e] == item.marker + item.body


def test_parse_list_items_orphan_continuations():
    # orphan continuations do not form new items
    items = parse_list_items('orphan\n* a\n\nanother orphan\n* b\n')
    assert [(i.marker, i.body) for i in items] == [('*', ' a'), ('*', ' b')]
    assert parse_list_items('only an orphan\n') == []
    assert parse_list_items('') == []


def test_parse_list_items_definitions():
    term, definition = parse_list_items('; term : def\n: def2\n')
    assert term.name == ' term '
    assert term.definition == ' def'
    assert definition.name is None
    assert definition.definition == ' def2'
    # a term without an inline definition
    (item,) = parse_list_items('; term only\n')
    assert item.name == ' term only'
    assert item.definition is None
    # regular items have neither name nor definition
    (item,) = parse_list_items('* regular\n')
    assert item.name is None
    assert item.definition is None
    # colons inside external links do not split the definition
    (item,) = parse_list_items('; https://a.b : c https://d.e : f\n')
    assert item.name == ' https://a.b '
    assert item.definition == ' c https://d.e : f'
    # name and definition spans map back to the source
    text = '; term : def\n'
    (item,) = parse_list_items(text)
    s, e = item.name_span
    assert text[s:e] == ' term '
    s, e = item.definition_span
    assert text[s:e] == ' def'


def test_parse_list_items_ignores_markup():
    # markers inside templates are not list entries
    items = parse_list_items('{{t|\n* fake\n}}\n* real\n')
    assert [(i.marker, i.body) for i in items] == [('*', ' real')]
    # colons inside templates do not split definitions
    (item,) = parse_list_items('; {{t|a:b}} : def\n')
    assert item.name == ' {{t|a:b}} '
    assert item.definition == ' def'


def test_wikilist_get_items():
    wl = WikiList('* a\n** b\n* c\n', pattern=r'\*')
    a, b, c = wl.get_items()
    assert (a.marker, a.body) == ('*', ' a')
    assert b.parent is a
    assert c.parent is None
    # spans are relative to the list string
    s, e = b.span
    assert wl.string[s:e] == '** b'
    # works for mixed sub-lists as well
    wl = WikiList('# a\n#* b\n#; t : d\n', pattern=r'\#')
    a, b, t = wl.get_items()
    assert b.parent is a
    assert t.parent is a
    assert t.name == ' t '
    assert t.definition == ' d'
