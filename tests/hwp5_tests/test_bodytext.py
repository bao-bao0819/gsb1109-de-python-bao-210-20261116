# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from unittest import TestCase

from hwp5.binmodel import Paragraph
from hwp5.binmodel import ParaText
from hwp5.binmodel import ParaRangeTag
from hwp5.bodytext import edit_paragraph
from hwp5.bodytext import merge_bodytext
from hwp5.bodytext import paragraphs_from_modelevents
from hwp5.bodytext import split_by_markers
from hwp5.treeop import STARTEVENT
from hwp5.treeop import ENDEVENT


KIND_HYPERLINK = 3
KIND_FIELD = 0


def range_tag(kind, data, start, end):
    return dict(start=start, end=end, tag=(kind << 24) | data)


def marker_keys(markers):
    return [marker.key for marker in markers]


class TestMergeBodyText(TestCase):

    def test_hyperlink_crosses_paragraph_with_nested_field(self):
        # the hyperlink starts at the beginning of paragraph 1 and ends in
        # paragraph 2; a field is nested in between, fully inside paragraph 1
        paragraphs = [
            ('ABCDE', [
                range_tag(KIND_HYPERLINK, 1, 0, 10),
                range_tag(KIND_FIELD, 2, 2, 4),
            ]),
            ('FGHIJ', [
                range_tag(KIND_HYPERLINK, 1, 0, 3),
            ]),
        ]

        bodytext = merge_bodytext(paragraphs)

        self.assertEqual('ABCDEFGHIJ', bodytext.text)
        self.assertEqual([0, 5, 10], bodytext.paragraph_offsets)

        link, field = bodytext.markers
        self.assertEqual((0, 8), (link.start, link.end))
        self.assertTrue(link.closed)
        self.assertEqual((KIND_FIELD, 2), field.key)
        self.assertEqual((2, 4), (field.start, field.end))
        self.assertTrue(field.closed)

        # every character still knows which link/field it belongs to
        ownership = [(piece_range, text, marker_keys(active))
                     for piece_range, text, active
                     in split_by_markers(bodytext)]
        self.assertEqual([
            ((0, 2), 'AB', [(KIND_HYPERLINK, 1)]),
            ((2, 4), 'CD', [(KIND_HYPERLINK, 1), (KIND_FIELD, 2)]),
            ((4, 8), 'EFGH', [(KIND_HYPERLINK, 1)]),
            ((8, 10), 'IJ', []),
        ], ownership)

    def test_marker_ending_at_paragraph_boundary_is_closed(self):
        paragraphs = [
            ('ABCDE', [range_tag(KIND_HYPERLINK, 1, 0, 5)]),
            ('FGHIJ', []),
        ]
        bodytext = merge_bodytext(paragraphs)
        marker = bodytext.markers[0]
        self.assertEqual((0, 5), (marker.start, marker.end))
        self.assertTrue(marker.closed)
        self.assertEqual([], bodytext.markers_at(7))

    def test_unpaired_marker_without_end_is_kept(self):
        # the link opens but no following paragraph carries its end marker
        paragraphs = [
            ('ABCDE', [range_tag(KIND_HYPERLINK, 1, 0, 10)]),
            ('FGHIJ', []),
        ]
        bodytext = merge_bodytext(paragraphs)

        self.assertEqual(1, len(bodytext.markers))
        marker = bodytext.markers[0]
        self.assertFalse(marker.closed)
        # the unpaired marker runs to the end of the merged text
        self.assertEqual((0, 10), (marker.start, marker.end))
        self.assertEqual(
            [(KIND_HYPERLINK, 1)],
            marker_keys(bodytext.markers_at(9)))

    def test_inner_end_does_not_clear_outer_relationship(self):
        paragraphs = [
            ('0123456789', [
                range_tag(KIND_HYPERLINK, 1, 0, 10),
                range_tag(KIND_FIELD, 2, 2, 6),
            ]),
        ]
        bodytext = merge_bodytext(paragraphs)

        inside = bodytext.markers_at(3)
        self.assertEqual(
            [(KIND_HYPERLINK, 1), (KIND_FIELD, 2)], marker_keys(inside))

        # after the inner field ends, the outer hyperlink is still active
        after_inner_end = bodytext.markers_at(7)
        self.assertEqual(
            [(KIND_HYPERLINK, 1)], marker_keys(after_inner_end))

    def test_multiple_crossing_markers_close_independently(self):
        paragraphs = [
            ('ABCDE', [
                range_tag(KIND_HYPERLINK, 1, 0, 9),
                range_tag(KIND_FIELD, 2, 1, 12),
            ]),
            ('FGHIJ', [
                range_tag(KIND_HYPERLINK, 1, 0, 4),
                range_tag(KIND_FIELD, 2, 0, 5),
            ]),
        ]
        bodytext = merge_bodytext(paragraphs)

        link, field = bodytext.markers
        self.assertEqual((0, 9), (link.start, link.end))
        self.assertTrue(link.closed)
        self.assertEqual((1, 10), (field.start, field.end))
        self.assertTrue(field.closed)


class TestEditParagraph(TestCase):

    def _bodytext(self):
        # link opens in paragraph 1 with an end past the paragraph length, so
        # it stays unpaired across the boundary; field lives in paragraph 2
        paragraphs = [
            ('ABCDE', [range_tag(KIND_HYPERLINK, 1, 0, 13)]),
            ('FGHIJKL', [
                range_tag(KIND_FIELD, 2, 4, 6),
            ]),
        ]
        return merge_bodytext(paragraphs)

    def test_insert_in_first_paragraph_shifts_later_nodes(self):
        bodytext = self._bodytext()
        self.assertFalse(bodytext.markers[0].closed)
        edited = edit_paragraph(bodytext, 0, 1, 1, 'XY')

        self.assertEqual('AXYBCDEFGHIJKL', edited.text)
        self.assertEqual([0, 7, 14], edited.paragraph_offsets)

        link, field = edited.markers
        # the link encloses the edit: its start is kept, end moves by delta
        self.assertEqual((0, 14), (link.start, link.end))
        self.assertFalse(link.closed)
        # the field is fully after the edit point: shifted by the real delta
        self.assertEqual((11, 13), (field.start, field.end))

    def test_edit_second_paragraph_keeps_earlier_offsets(self):
        bodytext = self._bodytext()
        edited = edit_paragraph(bodytext, 1, 0, 1, 'XYZ')

        self.assertEqual('ABCDEXYZGHIJKL', edited.text)
        # paragraph 0 offset is untouched; only later offsets move
        self.assertEqual([0, 5, 14], edited.paragraph_offsets)

        link, field = edited.markers
        # the link encloses the edit: start kept, end extended by real delta
        self.assertEqual((0, 14), (link.start, link.end))
        # the field starts after the edited region: shifted, not rebuilt
        self.assertEqual((11, 13), (field.start, field.end))

    def test_delete_inside_enclosing_link_keeps_outer_relation(self):
        bodytext = self._bodytext()
        edited = edit_paragraph(bodytext, 0, 1, 4, '')

        self.assertEqual('AEFGHIJKL', edited.text)
        link, field = edited.markers
        # inner deletion must not tear down the outer hyperlink
        self.assertEqual((0, 9), (link.start, link.end))
        self.assertTrue(all(link.start <= pos < link.end
                            for pos in range(0, 9)))
        # field was after the edit: shifted by the real deletion delta
        self.assertEqual((6, 8), (field.start, field.end))

    def test_no_change_edit_leaves_every_offset_intact(self):
        bodytext = self._bodytext()
        edited = edit_paragraph(bodytext, 1, 2, 4, 'HI')
        self.assertEqual(bodytext.text, edited.text)
        self.assertEqual(bodytext.paragraph_offsets, edited.paragraph_offsets)
        self.assertEqual(
            [(m.start, m.end) for m in bodytext.markers],
            [(m.start, m.end) for m in edited.markers])

    def test_unpaired_marker_kept_after_edit(self):
        paragraphs = [
            ('ABCDE', [range_tag(KIND_HYPERLINK, 1, 0, 99)]),
            ('FGHIJ', []),
        ]
        bodytext = merge_bodytext(paragraphs)
        self.assertFalse(bodytext.markers[0].closed)

        edited = edit_paragraph(bodytext, 0, 0, 0, 'ZZ')
        marker = edited.markers[0]
        self.assertFalse(marker.closed)
        self.assertEqual((0, 12), (marker.start, marker.end))
        self.assertEqual('ZZABCDEFGHIJ', edited.text)


class TestParagraphsFromModelEvents(TestCase):

    def test_collects_text_and_range_tags(self):
        context = dict()

        def paragraph_events(text, tags):
            events = [(STARTEVENT, (Paragraph, dict(), context))]
            chunks = [((0, len(text)), text)]
            events.append((STARTEVENT, (ParaText, dict(chunks=chunks),
                                        context)))
            events.append((ENDEVENT, (ParaText, dict(chunks=chunks),
                                      context)))
            events.append((STARTEVENT, (ParaRangeTag,
                                        dict(range_tags=tags), context)))
            events.append((ENDEVENT, (ParaRangeTag,
                                      dict(range_tags=tags), context)))
            events.append((ENDEVENT, (Paragraph, dict(), context)))
            return events

        tags = [range_tag(KIND_HYPERLINK, 1, 0, 5)]
        events = paragraph_events('ABCDE', tags)
        events += paragraph_events('FG', [])

        paragraphs = paragraphs_from_modelevents(iter(events))
        self.assertEqual([
            ('ABCDE', tags),
            ('FG', []),
        ], paragraphs)

        bodytext = merge_bodytext(paragraphs)
        self.assertEqual('ABCDEFG', bodytext.text)
        marker = bodytext.markers[0]
        self.assertEqual((0, 5), (marker.start, marker.end))

    def test_control_char_keeps_one_character_slot(self):
        context = dict()
        chunks = [((0, 2), 'AB'), ((2, 3), dict(code=9)),
                  ((3, 5), 'CD')]
        events = [
            (STARTEVENT, (Paragraph, dict(), context)),
            (STARTEVENT, (ParaText, dict(chunks=chunks), context)),
            (ENDEVENT, (ParaText, dict(chunks=chunks), context)),
            (ENDEVENT, (Paragraph, dict(), context)),
        ]
        paragraphs = paragraphs_from_modelevents(iter(events))
        self.assertEqual('AB\tCD', paragraphs[0][0])
