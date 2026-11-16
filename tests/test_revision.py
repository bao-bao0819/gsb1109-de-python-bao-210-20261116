from wikitextparser import Section, WikiText, parse
from wikitextparser._revision import LinkResolution, compare_revisions
from wikitextparser._section import (
    normalize_section_anchor,
    normalize_section_title,
)

DUP = """\
== 沿革 ==
早期历史，1890年建立。
=== 沿革 ===
子章节甲。
== 其他 ==
其他内容。
=== 沿革 ===
子章节乙，1999年扩建。
"""


def test_normalize_section_title():
    assert normalize_section_title(' 沿革 ') == '沿革'
    assert normalize_section_title('沿 革') == '沿_革'
    assert normalize_section_title('沿革<!-- note -->') == '沿革'
    assert normalize_section_title("'''沿革'''") == '沿革'
    assert normalize_section_title('A &amp; B') == 'A_&_B'


def test_normalize_section_anchor():
    assert normalize_section_anchor('%E6%B2%BF%E9%9D%A9') == '沿革'


def test_section_anchor_duplicate_ordinals():
    sections = parse(DUP).get_sections(include_subsections=False)
    assert [section.anchor for section in sections] == [
        None,
        '沿革',
        '沿革_2',
        '其他',
        '沿革_3',
    ]
    assert [section.anchor_ordinal for section in sections] == [
        1,
        1,
        2,
        1,
        3,
    ]


def test_section_parent_and_path():
    sections = parse(DUP).get_sections(include_subsections=False)
    child = sections[2]
    assert child.parent_section.span == sections[1].span
    assert child.parent_section.string == sections[1].string
    assert child.section_path == ('沿革', '沿革')
    assert sections[4].section_path == ('其他', '沿革')
    assert [s.span for s in sections[1].subsections] == [sections[2].span]
    assert sections[0].anchor is None
    assert sections[0].base_anchor is None


def test_direct_section_anchor():
    section = Section('== 沿革 ==\nA')
    assert section.anchor == '沿革'
    assert section.base_anchor == '沿革'


def test_compare_renamed_middle_heading():
    new = """\
== 沿革 ==
早期历史，1890年建立。
=== 历史沿革 ===
子章节甲。
== 其他 ==
其他内容。
=== 沿革 ===
子章节乙，1999年扩建。
"""
    comparison = compare_revisions(DUP, new)
    assert comparison.added == []
    assert comparison.removed == []
    by_anchor = {pair.old.anchor: pair for pair in comparison.section_pairs}
    assert by_anchor['沿革'].status == 'matched'
    assert by_anchor['沿革'].new_anchor == '沿革'
    renamed = by_anchor['沿革_2']
    assert renamed.status == 'renamed'
    assert renamed.new_anchor == '历史沿革'
    # The later duplicate is not allowed to become the renamed one.
    assert by_anchor['沿革_3'].new_anchor == '沿革_2'


def test_link_resolution_uses_identity_not_first_title_match():
    new = """\
== 沿革 ==
早期历史，1890年建立。
=== 历史沿革 ===
子章节甲。
== 其他 ==
其他内容。
=== 沿革 ===
子章节乙，1999年扩建。
[[条目#沿革_2]]
"""
    old = DUP + '[[条目#沿革]]\n[[条目#沿革_2]]\n[[条目#沿革_3]]\n'
    comparison = compare_revisions(old, new)
    assert comparison.resolve_section_link('沿革').new_fragment == '沿革'
    assert comparison.resolve_section_link('沿革_2').new_fragment == '历史沿革'
    assert comparison.resolve_section_link('沿革_3').new_fragment == '沿革_2'
    links = comparison.resolve_links()
    assert links['沿革'].status == 'unchanged'
    assert links['沿革_2'].status == 'resolved'
    assert links['沿革_3'].status == 'resolved'


def test_ambiguous_swapped_sections_keep_candidates():
    old = """\
== A ==
=== 沿革 ===
相同正文。
=== 沿革 ===
相同正文。
== B ==
=== 沿革 ===
相同正文。
"""
    new = """\
== A ==
=== 甲 ===
相同正文。
=== 沿革 ===
相同正文。
== B ==
=== 沿革 ===
相同正文。
"""
    comparison = compare_revisions(old, new)
    ambiguous = {pair.old.anchor: pair for pair in comparison.ambiguous}
    assert set(ambiguous) == {'沿革', '沿革_2'}
    assert set(ambiguous['沿革'].candidate_anchors) == {'甲', '沿革'}
    assert set(ambiguous['沿革_2'].candidate_anchors) == {'甲', '沿革'}
    resolution = comparison.resolve_section_link('沿革')
    assert resolution.status == 'ambiguous'
    assert resolution.new_fragment is None
    assert set(resolution.candidates) == {'甲', '沿革'}
    # The unrelated third heading is still uniquely matched.
    assert comparison.resolve_section_link('沿革_3').new_fragment == '沿革_2'


def test_resolution_distinguishes_parents():
    old = """\
== A ==
=== 沿革 ===
X内容，1890年。
== B ==
=== 沿革 ===
Y内容，1990年。
"""
    new = """\
== A ==
=== 沿革 ===
Y内容，1990年。
== B ==
=== 沿革 ===
X内容，1890年。
"""
    comparison = compare_revisions(old, new)
    # Body evidence is stronger than "the first one with this title".
    assert comparison.resolve_section_link('沿革').new_fragment == '沿革_2'
    assert comparison.resolve_section_link('沿革_2').new_fragment == '沿革'


def test_removed_and_added_sections():
    comparison = compare_revisions(
        '== a ==\nx\n== old ==\ny\n',
        '== a ==\nx\n== new ==\nz\n',
    )
    assert [pair.old.anchor for pair in comparison.removed] == ['old']
    assert [info.anchor for info in comparison.added] == ['new']
    resolution = comparison.resolve_section_link('old')
    assert resolution.status == 'unresolved'
    assert resolution.candidates == []


def test_link_without_fragment_is_unresolved():
    comparison = compare_revisions('== a ==\n[[b]]\n', '== a ==\n')
    link = comparison.old.wikilinks[0]
    resolution = comparison.resolve_link(link)
    assert isinstance(resolution, LinkResolution)
    assert resolution.status == 'unresolved'


def test_comparison_accepts_parsed_wikitext():
    comparison = compare_revisions(WikiText(DUP), parse(DUP))
    assert comparison.removed == []
    assert comparison.added == []
    assert all(pair.status == 'matched' for pair in comparison.section_pairs)


def test_three_sibling_headings_only_middle_renamed():
    old = """\
== 沿革 ==
第一节，1890年。
== 沿革 ==
第二节，1950年。
== 沿革 ==
第三节，1990年。
"""
    new = """\
== 沿革 ==
第一节，1890年。
== 发展沿革 ==
第二节，1950年。
== 沿革 ==
第三节，1990年。
"""
    comparison = compare_revisions(old, new)
    anchors = [pair.old.anchor for pair in comparison.section_pairs]
    assert anchors == ['沿革', '沿革_2', '沿革_3']
    assert comparison.section_pairs[0].new_anchor == '沿革'
    assert comparison.section_pairs[1].new_anchor == '发展沿革'
    assert comparison.section_pairs[2].new_anchor == '沿革_2'
    assert comparison.resolve_section_link('沿革_2').new_fragment == (
        '发展沿革'
    )


def test_percent_encoded_fragment():
    comparison = compare_revisions('== 沿革 ==\nold\n', '== 沿革 ==\nnew\n')
    resolution = comparison.resolve_section_link('%E6%B2%BF%E9%9D%A9')
    assert resolution.status == 'unchanged'
    assert resolution.new_fragment == '沿革'


def test_links_do_not_shift_when_first_duplicate_is_renamed():
    old = """\
== 沿革 ==
第一节
== 沿革 ==
第二节
== 沿革 ==
第三节
[[p#沿革]]
[[p#沿革_2]]
"""
    new = """\
== 早期沿革 ==
第一节
== 沿革 ==
第二节
== 沿革 ==
第三节
"""
    comparison = compare_revisions(old, new)
    assert comparison.resolve_section_link('沿革').new_fragment == '早期沿革'
    assert comparison.resolve_section_link('沿革_2').new_fragment == '沿革'
    assert comparison.resolve_section_link('沿革_3').new_fragment == '沿革_2'


def test_same_new_heading_is_not_proven_for_two_old_sections():
    comparison = compare_revisions(
        '== x ==\n=== 沿革 ===\nA正文。\n=== 沿革 ===\nA正文。\n',
        '== x ==\n=== 沿革 ===\nA正文。\n',
    )
    for pair in comparison.section_pairs:
        if pair.old.anchor in ('沿革', '沿革_2'):
            assert pair.status == 'ambiguous'
            assert pair.candidate_anchors == ['沿革']
