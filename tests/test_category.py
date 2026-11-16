from __future__ import annotations

import wikitextparser as wtp
from wikitextparser import (
    CategoryDeclaration,
    CategoryReport,
    TransclusionRegion,
    category_declarations,
    parse,
)
from wikitextparser._category import _control_regions, _protected_ranges


def _names(report: CategoryReport, view: str) -> list[str]:
    return [declaration.link.string for declaration in getattr(report, view)]


def test_plain_categories_belong_to_both_views():
    report = category_declarations(parse('text [[Category:A]] [[Category:B]]'))
    assert _names(report, 'page') == [
        '[[Category:A]]',
        '[[Category:B]]',
    ]
    assert _names(report, 'transclusion') == [
        '[[Category:A]]',
        '[[Category:B]]',
    ]
    assert all(declaration.region is None for declaration in report.page)


def test_noinclude_hides_from_transclusion_only():
    report = category_declarations(
        parse(
            '[[Category:A]] <noinclude>[[Category:B]]</noinclude> '
            '[[Category:C]]'
        )
    )
    assert _names(report, 'page') == [
        '[[Category:A]]',
        '[[Category:B]]',
        '[[Category:C]]',
    ]
    assert _names(report, 'transclusion') == [
        '[[Category:A]]',
        '[[Category:C]]',
    ]
    (declaration,) = [
        d for d in report.page if d.link.string == '[[Category:B]]'
    ]
    assert isinstance(declaration.region, TransclusionRegion)
    assert declaration.region.name == 'noinclude'
    assert declaration.region.complete is True
    assert declaration.region.tag is not None
    assert declaration.region.tag.name == 'noinclude'


def test_includeonly_hides_from_page_view_only():
    report = category_declarations(
        parse(
            '[[Category:A]] <includeonly>[[Category:B]]</includeonly> '
            '[[Category:C]]'
        )
    )
    assert _names(report, 'page') == [
        '[[Category:A]]',
        '[[Category:C]]',
    ]
    assert _names(report, 'transclusion') == [
        '[[Category:A]]',
        '[[Category:B]]',
        '[[Category:C]]',
    ]
    (declaration,) = [
        d for d in report.transclusion if d.link.string == '[[Category:B]]'
    ]
    assert declaration.region is not None
    assert declaration.region.name == 'includeonly'
    assert declaration.region.complete is True


def test_onlyinclude_restricts_transclusion_view():
    report = category_declarations(
        parse(
            '[[Category:A]] '
            '<onlyinclude>[[Category:B]] [[Category:C]]</onlyinclude> '
            '[[Category:D]]'
        )
    )
    assert _names(report, 'page') == [
        '[[Category:A]]',
        '[[Category:B]]',
        '[[Category:C]]',
        '[[Category:D]]',
    ]
    assert _names(report, 'transclusion') == [
        '[[Category:B]]',
        '[[Category:C]]',
    ]


def test_multiple_onlyinclude_regions_are_all_emitted():
    report = category_declarations(
        parse(
            '<onlyinclude>[[Category:O1]]</onlyinclude> x '
            '<onlyinclude>[[Category:O2]]</onlyinclude> [[Category:O3]]'
        )
    )
    assert _names(report, 'page') == [
        '[[Category:O1]]',
        '[[Category:O2]]',
        '[[Category:O3]]',
    ]
    assert _names(report, 'transclusion') == [
        '[[Category:O1]]',
        '[[Category:O2]]',
    ]


def test_includeonly_outside_onlyinclude_is_not_transcluded():
    report = category_declarations(
        parse(
            '<includeonly>[[Category:I]]</includeonly>'
            '<onlyinclude>[[Category:O]]</onlyinclude>'
        )
    )
    assert _names(report, 'page') == ['[[Category:O]]']
    assert _names(report, 'transclusion') == ['[[Category:O]]']


def test_nested_noinclude_and_includeonly():
    # includeonly inside noinclude reaches neither view.
    assert (
        _names(
            category_declarations(
                parse(
                    '<noinclude>a <includeonly>[[Category:X]]</includeonly> b'
                    '</noinclude>'
                )
            ),
            'page',
        )
        == []
    )
    # noinclude inside includeonly reaches neither view as well.
    report = category_declarations(
        parse(
            '<includeonly>a <noinclude>[[Category:Y]]</noinclude> b'
            '</includeonly>'
        )
    )
    assert _names(report, 'page') == []
    assert _names(report, 'transclusion') == []


def test_nested_control_tags_inside_onlyinclude():
    includeonly = category_declarations(
        parse(
            '<onlyinclude>a<includeonly>[[Category:IO]]</includeonly>b'
            '</onlyinclude> [[Category:OUT]]'
        )
    )
    assert _names(includeonly, 'transclusion') == ['[[Category:IO]]']
    assert _names(includeonly, 'page') == ['[[Category:OUT]]']

    noinclude = category_declarations(
        parse(
            '<onlyinclude>a<noinclude>[[Category:NI]]</noinclude>b'
            '</onlyinclude>'
        )
    )
    assert _names(noinclude, 'transclusion') == []
    assert _names(noinclude, 'page') == ['[[Category:NI]]']


def test_unclosed_includeonly_keeps_affected_range():
    text = 'a <includeonly>[[Category:Z]] tail'
    parsed = parse(text)
    masked = _protected_ranges(parsed.comments, parsed.get_tags())
    regions = _control_regions(text, masked)
    assert regions == [
        TransclusionRegion('includeonly', 15, len(text), False, None)
    ]
    report = category_declarations(parsed)
    assert _names(report, 'page') == []
    assert _names(report, 'transclusion') == ['[[Category:Z]]']
    declaration = report.transclusion[0]
    assert declaration.region is not None
    assert declaration.region.complete is False
    assert declaration.region.tag is None


def test_unclosed_noinclude_keeps_affected_range():
    text = 'a <noinclude>[[Category:Y]] tail'
    report = category_declarations(parse(text))
    assert _names(report, 'page') == ['[[Category:Y]]']
    assert _names(report, 'transclusion') == []
    assert report.page[0].region is not None
    assert report.page[0].region.complete is False


def test_unclosed_onlyinclude_keeps_affected_range():
    text = 'a <onlyinclude>[[Category:O]] tail'
    report = category_declarations(parse(text))
    assert _names(report, 'page') == ['[[Category:O]]']
    assert _names(report, 'transclusion') == ['[[Category:O]]']
    assert report.page[0].region is not None
    assert report.page[0].region.complete is False


def test_mismatched_tags_are_treated_as_incomplete():
    text = '<includeonly><noinclude>[[Category:M]]</includeonly></noinclude>'
    parsed = parse(text)
    masked = _protected_ranges(parsed.comments, parsed.get_tags())
    regions = {(r.name, r.complete) for r in _control_regions(text, masked)}
    assert regions == {
        ('noinclude', False),
        ('includeonly', True),
    }
    report = category_declarations(parsed)
    assert _names(report, 'page') == []
    assert _names(report, 'transclusion') == []


def test_stray_end_tag_is_ignored():
    report = category_declarations(parse('</noinclude>[[Category:S]]'))
    assert _names(report, 'page') == ['[[Category:S]]']
    assert _names(report, 'transclusion') == ['[[Category:S]]']
    assert report.page[0].region is None


def test_self_closing_control_tag_has_no_region():
    report = category_declarations(parse('<includeonly/>[[Category:SC]]'))
    assert _names(report, 'page') == ['[[Category:SC]]']
    assert _names(report, 'transclusion') == ['[[Category:SC]]']


def test_comment_pseudo_categories_are_ignored():
    report = category_declarations(
        parse('<!-- [[Category:C]] --> [[Category:R]]')
    )
    assert _names(report, 'page') == ['[[Category:R]]']
    assert _names(report, 'transclusion') == ['[[Category:R]]']


def test_protected_text_pseudo_categories_are_ignored():
    for tag in ('nowiki', 'pre', 'source'):
        report = category_declarations(
            parse(f'<{tag}>[[Category:P]]</{tag}> [[Category:R]]')
        )
        assert _names(report, 'page') == ['[[Category:R]]']
        assert _names(report, 'transclusion') == ['[[Category:R]]']


def test_control_tags_inside_comments_are_ignored():
    text = '<!-- <noinclude> --> [[Category:A]] <!-- </noinclude> -->'
    report = category_declarations(parse(text))
    assert _names(report, 'page') == ['[[Category:A]]']
    assert _names(report, 'transclusion') == ['[[Category:A]]']
    assert report.page[0].region is None


def test_control_tags_inside_protected_text_are_ignored():
    text = (
        '<nowiki><includeonly></nowiki> [[Category:A]] '
        '<nowiki></includeonly></nowiki>'
    )
    report = category_declarations(parse(text))
    assert _names(report, 'page') == ['[[Category:A]]']
    assert _names(report, 'transclusion') == ['[[Category:A]]']


def test_leading_colon_is_not_a_category_declaration():
    report = category_declarations(
        parse('[[:Category:Colon]] [[Category:Real]]')
    )
    assert _names(report, 'page') == ['[[Category:Real]]']
    assert _names(report, 'transclusion') == ['[[Category:Real]]']


def test_categories_inside_templates_are_not_direct_declarations():
    report = category_declarations(
        parse('{{navbox|[[Category:T]]}} [[Category:D]]')
    )
    assert _names(report, 'page') == ['[[Category:D]]']
    assert _names(report, 'transclusion') == ['[[Category:D]]']


def test_categories_inside_parser_functions_are_not_direct():
    report = category_declarations(
        parse('{{#if:x|[[Category:P]]}} [[Category:D]]')
    )
    assert _names(report, 'page') == ['[[Category:D]]']


def test_categories_inside_parameters_are_not_direct():
    report = category_declarations(
        parse('{{{a|[[Category:Param]]}}} [[Category:D]]')
    )
    assert _names(report, 'page') == ['[[Category:D]]']


def test_direct_category_inside_includeonly_is_still_reported():
    report = category_declarations(
        parse('<includeonly>{{t|[[Category:T]]}} [[Category:C]]</includeonly>')
    )
    assert _names(report, 'transclusion') == ['[[Category:C]]']
    assert _names(report, 'page') == []


def test_localized_category_namespaces():
    report = category_declarations(
        parse('[[Κατηγορία:Greek]] [[Category:English]]'),
        category_namespaces=('category', 'κατηγορία'),
    )
    assert _names(report, 'page') == [
        '[[Κατηγορία:Greek]]',
        '[[Category:English]]',
    ]


def test_namespace_matching_is_case_and_space_insensitive():
    report = category_declarations(
        parse('[[CATEGORY:Upper]] [[Category :Spaced]] [[Foo:Bar]]')
    )
    assert _names(report, 'page') == [
        '[[CATEGORY:Upper]]',
        '[[Category :Spaced]]',
    ]


def test_non_category_wikilinks_are_ignored():
    report = category_declarations(
        parse('[[Article]] [[File:a.jpg]] [[Category:Ok]]')
    )
    assert _names(report, 'page') == ['[[Category:Ok]]']


def test_declaration_is_shared_between_views():
    report = category_declarations(
        parse('<noinclude></noinclude> [[Category:Shared]]')
    )
    assert report.page[0] is report.transclusion[0]
    assert isinstance(report.page[0], CategoryDeclaration)


def test_report_keeps_document_order():
    report = category_declarations(
        parse(
            '<includeonly>[[Category:I]]</includeonly>'
            '[[Category:A]]'
            '<noinclude>[[Category:N]]</noinclude>'
        )
    )
    assert _names(report, 'page') == [
        '[[Category:A]]',
        '[[Category:N]]',
    ]
    assert _names(report, 'transclusion') == [
        '[[Category:I]]',
        '[[Category:A]]',
    ]


def test_empty_wikitext():
    report = category_declarations(parse(''))
    assert isinstance(report, CategoryReport)
    assert report.page == []
    assert report.transclusion == []
    assert report.regions == []


def test_report_regions_keep_nested_and_incomplete_ranges():
    text = (
        '<onlyinclude>[[Category:O]]</onlyinclude>'
        '<noinclude><includeonly></includeonly></noinclude>'
        '<includeonly>[[Category:I]]'
    )
    report = category_declarations(parse(text))
    regions = {(r.name, r.complete) for r in report.regions}
    assert regions == {
        ('onlyinclude', True),
        ('noinclude', True),
        ('includeonly', True),
        ('includeonly', False),
    }
    # The unclosed includeonly keeps its affected range to the end.
    unclosed = next(r for r in report.regions if not r.complete)
    assert unclosed.name == 'includeonly'
    assert unclosed.end == len(text)
    assert unclosed.tag is None
    # Empty regions are still reported so the affected range is preserved.
    empty_includeonly = next(
        r
        for r in report.regions
        if r.complete and r.name == 'includeonly' and r.start == r.end
    )
    assert empty_includeonly.tag is not None


def test_case_insensitive_control_tag_names():
    report = category_declarations(
        parse('<NOINCLUDE>[[Category:B]]</NOINCLUDE> [[Category:A]]')
    )
    assert _names(report, 'page') == [
        '[[Category:B]]',
        '[[Category:A]]',
    ]
    assert _names(report, 'transclusion') == ['[[Category:A]]']


def test_returned_links_are_live_wikilink_objects():
    report = category_declarations(parse('[[Category:Foo|bar]]'))
    link = report.page[0].link
    assert isinstance(link, wtp.WikiLink)
    assert link.title == 'Category:Foo'
    assert link.text == 'bar'
