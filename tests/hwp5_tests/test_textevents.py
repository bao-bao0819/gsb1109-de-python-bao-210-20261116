# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from unittest import TestCase

from hwp5.binmodel import Footer
from hwp5.binmodel import Header
from hwp5.binmodel import Paragraph
from hwp5.binmodel import Text
from hwp5.binmodel.controls import SectionDef
from hwp5.storage.ole import OleStorage
from hwp5.textevents import IDENTITY_BODY_TEXT
from hwp5.textevents import IDENTITY_PAGE_DECORATION
from hwp5.textevents import REGION_BODY
from hwp5.textevents import REGION_FOOTER
from hwp5.textevents import REGION_HEADER
from hwp5.textevents import build_report
from hwp5.textevents import classify_matches
from hwp5.textevents import collect_header_footer_decorations
from hwp5.textevents import collect_text_occurrences
from hwp5.textevents import find_in_paragraphs
from hwp5.textevents import group_occurrences_by_paragraph
from hwp5.treeop import STARTEVENT
from hwp5.treeop import ENDEVENT
from hwp5.utils import cached_property
from hwp5.xmlmodel import Hwp5File

from . import test_binmodel
from .fixtures import get_fixture_path


NEEDLE = u'表二'


def start(model, **attrs):
    return STARTEVENT, (model, attrs, {})


def end(model):
    return ENDEVENT, (model, {}, {})


def paragraph(paragraph_id, text):
    return [
        start(Paragraph, paragraph_id=paragraph_id),
        start(Text, text=text),
        end(Text),
        end(Paragraph),
    ]


def header(paragraph_id, text):
    return [
        start(Header, chid='head', flags=0),
    ] + paragraph(paragraph_id, text) + [
        end(Header),
    ]


class TestBase(test_binmodel.TestBase):

    @cached_property
    def hwp5file_xml(self):
        return Hwp5File(self.olestg)

    hwp5file = hwp5file_xml


class HeaderFooterFixtureTest(TestBase):

    hwp5file_name = 'headerfooter.hwp'

    def test_collect_occurrences_keeps_region_evidence(self):
        events = list(self.hwp5file.events())
        occurrences = collect_text_occurrences(iter(events))

        regions = [(o['region'], o['paragraph_id']) for o in occurrences]
        self.assertIn((REGION_BODY, 0), regions)
        self.assertIn((REGION_HEADER, 2), regions)
        self.assertIn((REGION_FOOTER, 3), regions)
        self.assertTrue(
            all(o['section_index'] == 0 for o in occurrences)
        )

    def test_header_and_body_are_separate_regions(self):
        events = list(self.hwp5file.events())
        paragraphs = group_occurrences_by_paragraph(
            collect_text_occurrences(iter(events))
        )
        by_region = {}
        for para in paragraphs:
            by_region.setdefault(para['region'], []).append(para['text'])

        self.assertEqual([u'첫 페이지'], by_region[REGION_BODY])
        self.assertEqual(
            [u'Header 이것은 머리말입니다.'],
            by_region[REGION_HEADER],
        )
        self.assertEqual(
            [u'Footer 이것은 꼬리말입니다.'],
            by_region[REGION_FOOTER],
        )

    def test_decorations_recorded_once_with_rendering_note(self):
        events = list(self.hwp5file.events())
        decorations = collect_header_footer_decorations(iter(events))

        self.assertEqual(2, len(decorations))
        header_deco = next(
            d for d in decorations if d['control'] == 'Header'
        )
        self.assertEqual(REGION_HEADER, header_deco['region'])
        self.assertEqual(1, header_deco['model_occurrences'])
        self.assertEqual('both_pages', header_deco['places'])
        self.assertEqual([2], header_deco['paragraph_ids'])
        self.assertIn('repeats it on', header_deco['rendering_note'])

        footer_deco = next(
            d for d in decorations if d['control'] == 'Footer'
        )
        self.assertEqual('odd_page', footer_deco['places'])

    def test_report_classifies_header_match_as_decoration(self):
        report = build_report(self.hwp5file, u'머리말')

        self.assertEqual(1, report['total_paragraph_matches'])
        match = report['matches'][0]
        self.assertEqual(REGION_HEADER, match['region'])
        self.assertEqual(IDENTITY_PAGE_DECORATION, match['identity'])
        self.assertEqual(
            [IDENTITY_PAGE_DECORATION],
            match['candidate_identities'],
        )
        self.assertEqual(
            1,
            report['identity_summary'][IDENTITY_PAGE_DECORATION]['min_count'],
        )
        self.assertEqual(
            1,
            report['identity_summary'][IDENTITY_PAGE_DECORATION]['max_count'],
        )
        self.assertEqual(
            0,
            report['identity_summary'][IDENTITY_BODY_TEXT]['definite_count'],
        )

    def test_report_classifies_body_match_as_body_text(self):
        report = build_report(self.hwp5file, u'첫 페이지')

        self.assertEqual(1, report['total_paragraph_matches'])
        self.assertEqual(REGION_BODY, report['matches'][0]['region'])
        self.assertEqual(
            IDENTITY_BODY_TEXT,
            report['matches'][0]['identity'],
        )

    def test_report_lists_decoration_containing_needle(self):
        report = build_report(self.hwp5file, u'머리말')
        hits = report['page_decorations_containing_needle']
        self.assertEqual(1, len(hits))
        self.assertEqual('Header', hits[0]['control'])
        self.assertEqual('both_pages', hits[0]['places'])


class SyntheticOccurrenceTest(TestCase):

    def test_identical_text_is_not_deduplicated_by_content(self):
        events = []
        events += [start(SectionDef, section_id=0), end(SectionDef)]
        events += paragraph(0, NEEDLE)
        events += header(1, NEEDLE)
        events += paragraph(2, NEEDLE)

        paragraphs = group_occurrences_by_paragraph(
            collect_text_occurrences(iter(events))
        )
        matches = find_in_paragraphs(paragraphs, NEEDLE)

        self.assertEqual(3, len(matches))
        self.assertEqual(
            [REGION_BODY, REGION_BODY, REGION_HEADER],
            sorted(m['region'] for m in matches),
        )

    def test_identical_text_keeps_separate_positions_per_region(self):
        events = []
        events += [start(SectionDef, section_id=0), end(SectionDef)]
        events += paragraph(0, NEEDLE)
        events += header(1, NEEDLE)

        paragraphs = group_occurrences_by_paragraph(
            collect_text_occurrences(iter(events))
        )
        matches = find_in_paragraphs(paragraphs, NEEDLE)
        self.assertEqual([0], matches[0]['positions'])
        self.assertEqual([0], matches[1]['positions'])
        self.assertNotEqual(
            matches[0]['paragraph_id'],
            matches[1]['paragraph_id'],
        )

    def test_repeated_needle_in_one_paragraph_records_all_positions(self):
        events = paragraph(0, NEEDLE + u' x ' + NEEDLE)
        paragraphs = group_occurrences_by_paragraph(
            collect_text_occurrences(iter(events))
        )
        matches = find_in_paragraphs(paragraphs, NEEDLE)
        self.assertEqual(1, len(matches))
        self.assertEqual([0, 5], matches[0]['positions'])

    def test_each_section_header_is_separate_model_occurrence(self):
        events = []
        events += [start(SectionDef, section_id=0), end(SectionDef)]
        events += header(0, NEEDLE)
        events += [start(SectionDef, section_id=1), end(SectionDef)]
        events += header(1, NEEDLE)

        decorations = collect_header_footer_decorations(iter(events))
        header_decors = [
            d for d in decorations if d['control'] == 'Header'
        ]
        self.assertEqual(2, len(header_decors))
        self.assertEqual(
            [0, 1],
            [d['section_index'] for d in header_decors],
        )
        self.assertTrue(
            all(d['model_occurrences'] == 1 for d in header_decors)
        )

    def test_unknown_region_yields_candidates_and_count_ranges(self):
        events = []
        events += [start(SectionDef, section_id=0), end(SectionDef)]
        events += paragraph(0, NEEDLE)
        paragraphs = group_occurrences_by_paragraph(
            collect_text_occurrences(iter(events))
        )
        matches = find_in_paragraphs(paragraphs, NEEDLE)
        matches = [dict(m, region='mystery') for m in matches]

        classified, summary = classify_matches(matches)
        self.assertEqual('unknown', classified[0]['identity'])
        self.assertEqual(
            [IDENTITY_BODY_TEXT, IDENTITY_PAGE_DECORATION],
            classified[0]['candidate_identities'],
        )
        for identity in (IDENTITY_BODY_TEXT, IDENTITY_PAGE_DECORATION):
            self.assertEqual(0, summary[identity]['min_count'])
            self.assertEqual(1, summary[identity]['max_count'])
            self.assertEqual(1, summary[identity]['uncertain_count'])

    def test_mixed_known_and_unknown_combine_ranges(self):
        events = []
        events += [start(SectionDef, section_id=0), end(SectionDef)]
        events += paragraph(0, NEEDLE)
        events += header(1, NEEDLE)
        paragraphs = group_occurrences_by_paragraph(
            collect_text_occurrences(iter(events))
        )
        matches = find_in_paragraphs(paragraphs, NEEDLE)
        # corrupt the region of one match to simulate missing evidence
        matches[0] = dict(matches[0], region=None)

        _, summary = classify_matches(matches)
        body = summary[IDENTITY_BODY_TEXT]
        decoration = summary[IDENTITY_PAGE_DECORATION]
        self.assertEqual((body['min_count'], body['max_count']), (0, 1))
        self.assertEqual(
            (decoration['min_count'], decoration['max_count']), (1, 2)
        )

    def test_fragmented_text_is_joined_before_matching(self):
        events = [
            start(Paragraph, paragraph_id=0),
            start(Text, text=u'表'),
            end(Text),
            start(Text, text=u'二'),
            end(Text),
            end(Paragraph),
        ]
        paragraphs = group_occurrences_by_paragraph(
            collect_text_occurrences(iter(events))
        )
        matches = find_in_paragraphs(paragraphs, NEEDLE)
        self.assertEqual(1, len(matches))
        self.assertEqual(2, matches[0]['fragment_count'])


class FixturePathSanityTest(TestCase):

    def test_headerfooter_fixture_available(self):
        self.assertTrue(get_fixture_path('headerfooter.hwp'))

    def test_ole_storage_loads(self):
        path = get_fixture_path('headerfooter.hwp')
        storage = OleStorage(path)
        self.assertTrue(storage is not None)
