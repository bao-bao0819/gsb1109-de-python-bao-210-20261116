from wikitextparser import ExternalLink


def test_externallinks():
    assert ExternalLink('http://example.org').external_links == []


def test_repr():
    assert (
        repr(ExternalLink('HTTP://mediawiki.org'))
        == "ExternalLink('HTTP://mediawiki.org')"
    )


def test_numberedmailto_change_none_to_empty():
    s = (
        '[mailto:'
        'info@example.org?Subject=URL%20Encoded%20Subject&body='
        'Body%20Textinfo]'
    )
    el = ExternalLink(s)
    assert s[1:-1] == el.url
    assert el.text is None
    assert el.in_brackets
    el.text = ''
    assert el.string == s[:-1] + ' ]'


def test_bare_link():
    el = ExternalLink('HTTP://mediawiki.org')
    assert 'HTTP://mediawiki.org' == el.url
    assert el.text is None
    assert not el.in_brackets


def test_inbracket_with_text():
    el = ExternalLink('[ftp://mediawiki.org mediawiki ftp]')
    assert 'ftp://mediawiki.org' == el.url
    assert 'mediawiki ftp' == el.text
    assert el.in_brackets is True


def test_text_setter():
    el = ExternalLink('[ftp://mediawiki.org mediawiki ftp]')
    el.text = 'a'
    assert '[ftp://mediawiki.org a]' == el.string

    del el.text
    el.text = 'b'
    assert '[ftp://mediawiki.org b]' == el.string

    el = ExternalLink('ftp://mediawiki.org')
    el.text = 'c'
    assert '[ftp://mediawiki.org c]' == el.string


def test_text_delter():
    el = ExternalLink('[ftp://mediawiki.org mediawiki ftp]')
    del el.text
    assert '[ftp://mediawiki.org]' == el.string

    del el.text
    assert '[ftp://mediawiki.org]' == el.string

    el = ExternalLink('ftp://mediawiki.org')
    del el.text
    assert 'ftp://mediawiki.org' == el.string


def test_url_setter():
    el = ExternalLink('[ftp://mediawiki.org mw]')
    el.url = 'https://www.mediawiki.org/'
    assert '[https://www.mediawiki.org/ mw]' == el.string

    el = ExternalLink('ftp://mediawiki.org')
    el.url = 'https://www.mediawiki.org/'
    assert 'https://www.mediawiki.org/' == el.string

    el = ExternalLink('[ftp://mediawiki.org]')
    el.url = 'https://www.mediawiki.org/'
    assert '[https://www.mediawiki.org/]' == el.string


def test_ending_with_less_than_sign():
    el = ExternalLink('[https://www.google.<com]')
    assert el.url == 'https://www.google.'
    assert el.text == '<com'


def test_external_link_case():  # 99
    assert ExternalLink('[Https://a b]').text == 'b'


def test_free_link_sentence_punctuation_is_outside_boundary():
    el = ExternalLink('https://en.wikipedia.org/wiki/Foo_(bar).')
    assert el.target == 'https://en.wikipedia.org/wiki/Foo_(bar)'
    assert el.string == el.target
    assert el.span == (0, 39)
    assert el.link_span == (0, 39)
    assert el.target_span == (0, 39)
    assert el.display is None
    assert el.analyze().boundary_proven is False
    assert el.analyze().trailing_text == '.'


def test_free_link_balanced_parentheses_are_kept():
    url = 'https://example.com/a_(b)_(c)'
    el = ExternalLink(url)
    assert el.target == url
    assert el.string == url


def test_free_link_unbalanced_closing_parentheses_are_candidates():
    el = ExternalLink('https://example.com/a(b)).')
    assert el.target == 'https://example.com/a(b)'
    assert el.string == 'https://example.com/a(b)'
    targets = [c.target for c in el.candidates]
    assert targets == [
        'https://example.com/a(b)',
        'https://example.com/a(b))',
        'https://example.com/a(b)).',
    ]
    assert [c.is_canonical for c in el.candidates] == [True, False, False]
    assert [c.reason for c in el.candidates] == [
        'ambiguous_trim_kept_as_candidate',
        'ambiguous_closing_parenthesis',
        'ambiguous_sentence_punctuation',
    ]


def test_free_link_comma_can_be_url_data():
    el = ExternalLink('https://example.com/a(b)c,')
    assert el.target == 'https://example.com/a(b)c'
    assert el.analyze().trailing_text == ','


def test_free_link_punctuation_then_unbalanced_parenthesis():
    # makeFreeExternalLink trims punctuation first, so the period in "a.)"
    # stays in the target and only the unmatched closing parenthesis goes out.
    el = ExternalLink('https://example.com/a.)')
    assert el.target == 'https://example.com/a.'
    assert el.string == 'https://example.com/a.'
    assert el.analyze().trailing_text == ')'
    el = ExternalLink('https://example.com/a()).')
    assert el.target == 'https://example.com/a()'
    assert el.analyze().trailing_text == ').'


def test_bracket_link_spans_and_duplicate_display_target():
    from wikitextparser import parse

    s = '[https://example.com/foo_(bar) https://example.com/foo_(bar)]'
    (el,) = parse(s).external_links
    info = el.analyze()
    assert info.boundary_proven is True
    assert info.link_span == (0, 61)
    assert info.target_span == (1, 30)
    assert info.display_span == (31, 60)
    assert info.target == 'https://example.com/foo_(bar)'
    assert info.display == 'https://example.com/foo_(bar)'
    assert info.display_equals_target is True
    assert info.target_in_display is True
    assert len(info.candidates) == 1


def test_info_dict_shape():
    from wikitextparser import parse

    info = parse('[https://a.example/x t]').external_links[0].analyze()
    d = info.as_dict()
    assert d['link_span'] == (0, 23)
    assert d['target_span'] == (1, 20)
    assert d['display_span'] == (21, 22)
    assert d['target'] == 'https://a.example/x'
    assert d['display'] == 't'
    assert d['candidates'][0]['reason'] == 'bracket_delimiter'


def test_analyze_external_links_helper():
    from wikitextparser import analyze_external_links, parse

    infos = analyze_external_links(parse('https://a.example/x.'))
    assert len(infos) == 1
    assert infos[0].target == 'https://a.example/x'
    assert len(infos[0].candidates) == 2


def test_replace_targets_only_changes_target_bytes():
    from wikitextparser import parse

    s = (
        'old https://old.example/a_(b) '
        '[https://old.example/a_(b) https://old.example/a_(b)]'
    )
    p = parse(s)
    p.replace_external_link_targets(
        {'https://old.example/a_(b)': 'https://new.example/x_(y)'}
    )
    assert p.string == (
        'old https://new.example/x_(y) '
        '[https://new.example/x_(y) https://old.example/a_(b)]'
    )
    links = p.external_links
    assert [link.target for link in links] == [
        'https://new.example/x_(y)',
        'https://new.example/x_(y)',
    ]
    assert links[1].display == 'https://old.example/a_(b)'


def test_replace_targets_keeps_sentence_punctuation_and_parentheses():
    from wikitextparser import parse

    s = 'See https://old.example/a_(b). (https://old.example/a_(b)).'
    p = parse(s)
    p.replace_external_link_targets(
        {'https://old.example/a_(b)': 'https://new.example/a_(b)'}
    )
    assert p.string == (
        'See https://new.example/a_(b). (https://new.example/a_(b)).'
    )


def test_replace_targets_can_return_report_in_document_order():
    from wikitextparser import replace_external_link_targets

    s = '[https://old/a x] https://old/a.'
    out, report = replace_external_link_targets(
        s, {'https://old/a': 'https://new/b'}, with_report=True
    )
    assert out == '[https://new/b x] https://new/b.'
    assert [r.target_span_before[0] for r in report] == [1, 18]
    assert all(r.display_unchanged for r in report)
    assert all(r.reparsed for r in report)
    assert report.unmatched == []


def test_replace_targets_reports_unmatched_keys():
    from wikitextparser import replace_external_link_targets

    out, report = replace_external_link_targets(
        'https://a.example/x',
        {'https://missing.example/y': 'https://b/z'},
        with_report=True,
    )
    assert out == 'https://a.example/x'
    assert report.unmatched == ['https://missing.example/y']


def test_replace_targets_rejects_unrecognized_target_before_mutation():
    from pytest import raises

    from wikitextparser import parse

    s = '[https://old.example/a x] https://old.example/a'
    p = parse(s)
    with raises(ValueError):
        p.replace_external_link_targets({'https://old.example/a': 'not a url'})
    assert p.string == s


def test_replace_targets_rejects_display_text_breakage():
    from pytest import raises

    from wikitextparser import parse

    p = parse('[https://old.example/a x]')
    with raises(ValueError):
        p.replace_external_link_targets(
            {'https://old.example/a': 'https://new.example/b y'}
        )
    assert p.string == '[https://old.example/a x]'


def test_replace_targets_do_not_touch_prose_equal_to_target():
    from wikitextparser import replace_external_link_targets

    s = 'https://old.example/a in prose, [https://old.example/a text]'
    assert (
        replace_external_link_targets(
            s, {'https://old.example/a': 'https://new.example/b'}
        )
        == 'https://new.example/b in prose, [https://new.example/b text]'
    )


def test_replace_targets_in_place_keeps_other_links():
    from wikitextparser import parse

    p = parse('https://old.example/a https://keep.example/a')
    p.replace_external_link_targets(
        {'https://old.example/a': 'https://new.example/a'}
    )
    assert [link.target for link in p.external_links] == [
        'https://new.example/a',
        'https://keep.example/a',
    ]


def test_retarget_and_replace_target_methods():
    from wikitextparser import parse

    p = parse('[https://old.example/a text]')
    link = p.external_links[0]
    link.replace_target('https://new.example/b')
    assert p.string == '[https://new.example/b text]'
    link.retarget('https://third.example/c')
    assert p.string == '[https://third.example/c text]'
    assert link.display == 'text'


def test_display_repeating_only_target_fragment():
    from wikitextparser import parse

    info = parse('[https://a.example/x a.example]').external_links[0].analyze()
    assert info.display_equals_target is False
    assert info.display_in_target is True
    assert info.display_repeats_target is True

    info = parse('[https://a.example/x y]').external_links[0].analyze()
    assert info.display_repeats_target is False


def test_replacement_does_not_rewrite_fragment_display():
    from wikitextparser import parse

    p = parse('[https://old.example/a old.example/a]')
    p.replace_external_link_targets(
        {'https://old.example/a': 'https://new.example/b'}
    )
    assert p.string == '[https://new.example/b old.example/a]'
    assert p.external_links[0].display == 'old.example/a'


def test_free_link_full_width_sentence_punctuation():
    el = ExternalLink('https://example.com/路径。')
    assert el.target == 'https://example.com/路径'
    assert el.analyze().trailing_text == '。'
