from __future__ import annotations

from pytest import raises

from wikitextparser import (
    KNOWN_VARIANTS,
    VariantProjection,
    parse,
    project_variant,
    project_variant_segments,
)
from wikitextparser._variant import (
    EMPTY,
    HIDDEN,
    LITERAL,
    NO_CONVERT,
    PROTECTED,
    RAW,
    REPLACE,
    TEMPLATE,
    TITLE,
    UNKNOWN,
    VARIANT,
    VARIANT_FALLBACK,
)


def test_returned_value_is_a_string():
    projection = project_variant(
        '-{zh-hans:简体;zh-hant:繁體}-', 'zh-hans'
    )
    assert isinstance(projection, str)
    assert isinstance(projection, VariantProjection)
    assert projection == '简体'
    assert projection.text == '简体'
    assert projection.variant == 'zh-hans'


def test_branch_selection_and_source_ranges():
    source = '甲-{zh-hans:简体;zh-hant:繁體}-乙'
    simplified = project_variant(source, 'zh-hans')
    traditional = project_variant(source, 'zh-hant')

    assert simplified == '甲简体乙'
    assert traditional == '甲繁體乙'

    (literal,) = simplified.segments[:1]
    assert literal.text == '甲'
    assert literal.span == (0, 1)
    assert literal.rule == LITERAL

    selected = simplified.segments[1]
    assert selected.text == '简体'
    assert source[selected.start : selected.end] == '简体'
    assert selected.rule == VARIANT
    assert selected.branch == 'zh-hans'
    assert selected.construct_span == (1, 26)

    other = traditional.segments[1]
    assert source[other.start : other.end] == '繁體'
    assert other.start != selected.start
    assert other.branch == 'zh-hant'


def test_the_same_character_can_come_from_different_branches():
    source = '-{zh-hans:发;zh-hant:發}-'
    simplified = project_variant(source, 'zh-hans')
    traditional = project_variant(source, 'zh-hant')
    assert simplified == '发'
    assert traditional == '發'
    assert simplified.segments[0].span == (10, 11)
    assert traditional.segments[0].span == (20, 21)


def test_regional_variants_fall_back_to_their_script_variant():
    source = '甲-{zh-hans:简体;zh-hant:繁體}-乙'
    assert project_variant(source, 'zh-cn') == '甲简体乙'
    assert project_variant(source, 'zh-sg') == '甲简体乙'
    assert project_variant(source, 'zh-tw') == '甲繁體乙'
    assert project_variant(source, 'zh-hk') == '甲繁體乙'

    segment = project_variant(source, 'zh-cn').segments[1]
    assert segment.rule == VARIANT_FALLBACK
    assert segment.branch == 'zh-hans'


def test_default_branch_is_used_when_no_selector_matches():
    source = '-{默认;zh-hans:简体;zh-hant:繁體}-'
    projection = project_variant(source, 'en')
    assert projection == '默认'
    assert projection.segments[0].rule == 'default'
    assert projection.segments[0].branch is None
    assert projection.segments[0].text_span == (2, 4)


def test_unmatched_variant_uses_the_first_branch():
    source = '甲-{zh-hans:简体;zh-hant:繁體}-乙'
    projection = project_variant(source, 'en')
    assert projection == '甲简体乙'
    assert projection.segments[1].rule == VARIANT_FALLBACK
    assert projection.segments[1].branch == 'zh-hans'


def test_base_language_fallback():
    source = '-{zh:中文;zh-hant:繁體}-'
    projection = project_variant(source, 'zh-cn')
    assert projection == '中文'
    assert projection.segments[0].rule == VARIANT_FALLBACK
    assert projection.segments[0].branch == 'zh'


def test_local_replacement_only_changes_its_branch():
    source = '-{zh-hans:简体=>简;zh-hant:繁體}-后文简体'
    simplified = project_variant(source, 'zh-hans')
    traditional = project_variant(source, 'zh-hant')

    assert simplified == '简后文简体'
    assert traditional == '繁體后文简体'
    assert simplified.segments[0].rule == REPLACE
    assert simplified.segments[0].span == (14, 15)
    assert simplified.segments[1].rule == LITERAL
    assert simplified.segments[1].text == '后文简体'


def test_hidden_rules_do_not_apply_to_later_text():
    source = '-{H|zh-hans:电脑;zh-hant:電腦}-电脑'
    projection = project_variant(source, 'zh-hant')
    assert projection == '电脑'
    assert projection.segments[0].rule == HIDDEN
    assert projection.segments[0].text == ''
    assert projection.segments[1].rule == LITERAL


def test_unknown_variant_selector_is_unexplained_source():
    source = '-{zh-unknown:甲;zh-hant:乙}-后'
    projection = project_variant(source, 'zh-hant')
    assert projection == source
    assert projection.segments[0].rule == UNKNOWN
    assert projection.segments[0].text == '-{zh-unknown:甲;zh-hant:乙}-'


def test_unbalanced_construct_is_unexplained_source():
    source = '前-{zh-hans:甲;zh-hant:乙'
    projection = project_variant(source, 'zh-hans')
    assert projection == source
    assert projection.segments[-1].rule == UNKNOWN


def test_explicit_no_conversion_form():
    projection = project_variant('-{明文}-', 'zh-hans')
    assert projection == '明文'
    assert projection.segments[0].rule == NO_CONVERT
    assert projection.segments[0].text == '明文'


def test_raw_flag_does_not_convert():
    source = '-{R|-{zh-hans:甲;zh-hant:乙}-}-'
    projection = project_variant(source, 'zh-hant')
    assert projection == '-{zh-hans:甲;zh-hant:乙}-'
    assert projection.segments[0].rule == RAW


def test_description_flag_does_not_convert():
    source = '-{D|-{zh-hans:甲;zh-hant:乙}-}-'
    projection = project_variant(source, 'zh-hans')
    assert projection == '-{zh-hans:甲;zh-hant:乙}-'
    assert projection.segments[0].rule == NO_CONVERT


def test_unknown_flag_stays_unexplained():
    source = '-{A|-{zh-hans:甲;zh-hant:乙}-}-'
    projection = project_variant(source, 'zh-hans')
    assert projection == source
    assert projection.segments[0].rule == UNKNOWN
    assert projection.segments[0].explained is False


def test_bare_default_replacement_is_local():
    source = '-{甲=>乙}-后文甲'
    projection = project_variant(source, 'zh-hans')
    assert projection == '乙后文甲'
    assert projection.segments[0].rule == 'default-replace'
    assert projection.segments[0].span == (5, 6)


def test_title_flag_emits_no_body_text():
    projection = project_variant('-{T|标题}-正文', 'zh-hans')
    assert projection == '正文'
    assert projection.segments[0].rule == TITLE
    assert projection.segments[0].text == ''


def test_empty_conversion_construct():
    projection = project_variant('甲-{}-乙', 'zh-hans')
    assert projection == '甲乙'
    assert [segment.rule for segment in projection.segments] == [
        LITERAL,
        EMPTY,
        LITERAL,
    ]


def test_comments_are_protected_from_conversion():
    source = '甲<!-- -{zh-hans:简体;zh-hant:繁體}- -->乙'
    projection = project_variant(source, 'zh-hant')
    assert projection == source
    assert projection.segments[1].rule == 'comment'


def test_nowiki_is_protected_from_conversion():
    source = '甲<nowiki>-{zh-hans:简体;zh-hant:繁體}-</nowiki>乙'
    projection = project_variant(source, 'zh-hant')
    assert projection == source
    segment = projection.segments[1]
    assert segment.rule == PROTECTED
    assert segment.tag == 'nowiki'


def test_pre_and_syntaxhighlight_are_protected():
    for tag in ('pre', 'syntaxhighlight', 'math', 'source'):
        source = f'<{tag}>-{{zh-hans:简体;zh-hant:繁體}}-</{tag}>'
        projection = project_variant(source, 'zh-hant')
        assert projection == source
        assert projection.segments[0].rule == PROTECTED
        assert projection.segments[0].tag == tag


def test_templates_are_not_executed_or_converted():
    source = '甲{{t|-{zh-hans:简体;zh-hant:繁體}-}}乙'
    projection = project_variant(source, 'zh-hant')
    assert projection == source
    assert projection.segments[1].rule == TEMPLATE
    assert projection.segments[1].text == (
        '{{t|-{zh-hans:简体;zh-hant:繁體}-}}'
    )


def test_parser_functions_are_not_executed_or_converted():
    source = '{{#if:x|-{zh-hans:简体;zh-hant:繁體}-}}'
    projection = project_variant(source, 'zh-hant')
    assert projection == source
    assert projection.segments[0].rule == TEMPLATE


def test_conversion_marker_in_an_html_attribute_is_not_converted():
    source = (
        '<span title="-{zh-hans:甲;zh-hant:乙}-">'
        '-{zh-hans:甲;zh-hant:乙}-</span>'
    )
    projection = project_variant(source, 'zh-hant')
    assert projection == '<span title="-{zh-hans:甲;zh-hant:乙}-">乙</span>'
    assert projection.segments[0].text.startswith('<span title="')


def test_nested_conversion_uses_the_outer_selection():
    source = '-{zh-hans:外-{zh-hans:内层;zh-hant:內層}-;zh-hant:外層}-'
    simplified = project_variant(source, 'zh-hans')
    traditional = project_variant(source, 'zh-hant')
    assert simplified == '外内层'
    assert traditional == '外層'


def test_nested_conversion_ranges_point_at_the_selected_text():
    source = '-{zh-hans:外-{zh-hans:内层;zh-hant:內層}-;zh-hant:外層}-'
    projection = project_variant(source, 'zh-hans')
    inner = projection.segments[1]
    assert inner.text == '内层'
    assert source[inner.start : inner.end] == '内层'
    assert inner.construct_span is not None


def test_source_spans_keep_gaps_for_unexplained_regions():
    source = 'a-{H|zh-hans:甲;zh-hant:乙}-b'
    projection = project_variant(source, 'zh-hans')
    assert projection == 'ab'
    assert projection.source_spans() == [(0, 1), (26, 27)]
    assert projection.source_spans(0, 2) == [(0, 1), (26, 27)]


def test_source_spans_do_not_merge_across_discarded_branches():
    source = '-{zh-hans:甲;zh-hant:乙}--{zh-hans:丙;zh-hant:丁}-'
    projection = project_variant(source, 'zh-hans')
    assert projection == '甲丙'
    assert projection.source_spans() == [(10, 11), (33, 34)]


def test_char_spans_and_range_map():
    source = '甲-{zh-hans:简体;zh-hant:繁體}-乙'
    projection = project_variant(source, 'zh-hans')
    assert projection.char_spans[0] == ((0, 1),)
    assert projection.char_spans[1] == ((11, 12),)
    assert projection.range_map[0][0] == 0
    assert projection.range_map[0][2].text == '甲'


def test_segments_are_json_serializable():
    source = '甲-{zh-hans:简体;zh-hant:繁體}-乙'
    data = project_variant(source, 'zh-hant').to_dict()
    assert data['variant'] == 'zh-hant'
    assert data['text'] == '甲繁體乙'
    assert data['segments'][1]['rule'] == VARIANT
    assert data['segments'][1]['start'] == 22


def test_subnode_offsets_are_relative_to_the_root_document():
    source = '前-{zh-hans:甲;zh-hant:乙}-后'
    section = parse(source).sections[0]
    assert section.span == (0, len(source))
    projection = section.project_variant('zh-hans')
    assert projection == '前甲后'
    selected = projection.segments[1]
    assert source[selected.start : selected.end] == '甲'
    assert selected.construct_span == (1, 24)


def test_wikitext_method_and_module_function_agree():
    source = '甲-{zh-hans:简体;zh-hant:繁體}-乙'
    parsed = parse(source)
    assert parsed.project_variant('zh-hant') == project_variant(
        source, 'zh-hant'
    )
    assert project_variant_segments(source, 'zh-hant') == (
        parsed.project_variant('zh-hant').segments
    )


def test_segment_description_matches_rule():
    projection = project_variant('-{zh-hans:甲;zh-hant:乙}-', 'zh-hans')
    segment = projection.segments[0]
    assert segment.description == 'selected variant branch'
    assert segment.as_tuple() == ('甲', 10, 11, VARIANT)


def test_module_function_can_project_a_source_region():
    source = '前-{zh-hans:甲;zh-hant:乙}-后'
    projection = project_variant(source, 'zh-hans', start=1, end=22)
    assert projection == '甲'
    assert projection.source_start == 1
    assert projection.source_end == 22
    assert projection.segments[0].span == (11, 12)


def test_variant_is_required():
    with raises(ValueError):
        project_variant('x', '')
    with raises(TypeError):
        project_variant('x', None)  # type: ignore[arg-type]


def test_known_variants_contains_chinese_codes():
    assert {'zh', 'zh-hans', 'zh-hant', 'zh-cn', 'zh-tw'} <= KNOWN_VARIANTS
