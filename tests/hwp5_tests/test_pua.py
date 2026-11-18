# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from io import BytesIO
from unittest import TestCase
import json
import sys

from hwp5.binmodel import Hwp5File
from hwp5.binmodel import ParaText
from hwp5.hypua_mappings import sample_table_v1
from hwp5.hypua_mappings import sample_table_v2
from hwp5.pua import HANGUL_PUA_END
from hwp5.pua import HANGUL_PUA_START
from hwp5.pua import MappingEntry
from hwp5.pua import PuaMappingTable
from hwp5.pua import SearchView
from hwp5.pua import apply_mapping
from hwp5.pua import build_search_view
from hwp5.pua import codepoint_of_unknown
from hwp5.pua import codepoint_text
from hwp5.pua import coalesce_ranges
from hwp5.pua import diff_views
from hwp5.pua import format_ranges
from hwp5.pua import is_pua_codepoint
from hwp5.pua import iter_paratext_views
from hwp5.pua import iter_pua_chars
from hwp5.pua import load_mapping_table
from hwp5.pua import remap
from hwp5.pua import search_hit_changes
from hwp5.pua import split_pua_ranges
from hwp5.pua import unknown_char

from .fixtures import get_fixture_path


PY3 = sys.version_info.major == 3
if PY3:
    unichr = chr


class TestPuaCodepoints(TestCase):

    def test_is_pua_codepoint(self):
        self.assertTrue(is_pua_codepoint(HANGUL_PUA_START))
        self.assertTrue(is_pua_codepoint(HANGUL_PUA_END))
        self.assertTrue(is_pua_codepoint(0xF53A))
        self.assertFalse(is_pua_codepoint(0xAC00))
        self.assertFalse(is_pua_codepoint(0xE0000))

    def test_unknown_char_is_distinct_and_recoverable(self):
        # two unknown codepoints must never become the same business char
        a = unknown_char(0xEBD4)
        b = unknown_char(0xEBD7)
        self.assertNotEqual(a, b)
        self.assertEqual(0xEBD4, codepoint_of_unknown(a))
        self.assertEqual(0xEBD7, codepoint_of_unknown(b))
        # the sentinel itself lives outside the BMP PUA block
        self.assertFalse(is_pua_codepoint(ord(a)))

    def test_unknown_char_rejects_non_pua(self):
        self.assertRaises(ValueError, unknown_char, 0xAC00)
        self.assertRaises(ValueError, codepoint_of_unknown, 'x')

    def test_coalesce_ranges(self):
        self.assertEqual([(1, 1)], coalesce_ranges([1]))
        self.assertEqual([(1, 3)], coalesce_ranges([1, 2, 3]))
        self.assertEqual([(1, 2), (5, 5)],
                         coalesce_ranges([2, 5, 1]))
        self.assertEqual(['U+0001..U+0002', 'U+0005..U+0005'],
                         format_ranges([(1, 2), (5, 5)]))

    def test_codepoint_text(self):
        self.assertEqual('U+EBD4', codepoint_text(0xEBD4))


class TestMappingTable(TestCase):

    def test_one_to_many_relation(self):
        table = sample_table_v1()
        sources = table.sources_for_target(u'나')
        self.assertEqual([0xEBD4, 0xEBD7], sources)
        self.assertEqual([(0xEBD4, 0xEBD4), (0xEBD7, 0xEBD7)],
                         table.ranges_for_target(u'나'))
        self.assertTrue(table.is_known(0xEBD4))
        self.assertFalse(table.is_known(0xF100))

    def test_version_is_kept(self):
        self.assertEqual('sample-hypua/1', sample_table_v1().version)
        self.assertEqual('sample-hypua/2', sample_table_v2().version)

    def test_duplicate_source_rejected(self):
        entries = [MappingEntry(0xEBD4, u'나'),
                   MappingEntry(0xEBD4, u'가')]
        self.assertRaises(ValueError, PuaMappingTable, entries)

    def test_invalid_entry(self):
        self.assertRaises(ValueError, MappingEntry, 0xAC00, u'나')
        self.assertRaises(ValueError, MappingEntry, 0xEBD4, u'나라')

    def test_json_roundtrip(self):
        table = sample_table_v1()
        stream = BytesIO(table.dumps().encode('utf-8'))
        loaded = load_mapping_table(stream)
        self.assertEqual(table.version, loaded.version)
        self.assertEqual(table.sources_for_target(u'나'),
                         loaded.sources_for_target(u'나'))
        self.assertEqual(table.ranges_for_target(u'●'),
                         loaded.ranges_for_target(u'●'))


class TestApplyMapping(TestCase):

    def test_mapped_and_plain_chars(self):
        table = sample_table_v1()
        text = u'A' + unichr(0xEBD4) + u'B'
        normalized, records = apply_mapping(text, table)
        self.assertEqual(u'A' + u'나' + u'B', normalized)
        self.assertEqual(3, len(records))
        self.assertEqual((0xEBD4,), records[1].source_codepoints)
        self.assertEqual('sample-hypua/1', records[1].mapping_version)
        self.assertFalse(records[1].unknown)
        self.assertEqual((), records[0].source_codepoints)

    def test_unknown_codepoint_kept_distinct(self):
        table = sample_table_v1()
        text = unichr(0xF100) + unichr(0xF101)
        normalized, records = apply_mapping(text, table)
        self.assertTrue(records[0].unknown and records[1].unknown)
        # unknown glyphs are never unified into one business char
        self.assertNotEqual(normalized[0], normalized[1])
        self.assertEqual(0xF100,
                         codepoint_of_unknown(normalized[0]))
        self.assertEqual(0xF101,
                         codepoint_of_unknown(normalized[1]))

    def test_iter_pua_chars_and_split(self):
        text = u'A' + unichr(0xEBD4) + u'B' + unichr(0xF100)
        pua = [cp for _, _, cp in iter_pua_chars(text)]
        self.assertEqual([0xEBD4, 0xF100], pua)
        runs = split_pua_ranges(text)
        self.assertEqual('text', runs[0][0])
        self.assertEqual('pua', runs[1][0])
        self.assertEqual((0xEBD4,), runs[1][1])


class TestSearchView(TestCase):

    def setUp(self):
        self.table = sample_table_v1()
        self.source = (unichr(0xEBD4) + unichr(0xEBD7) + unichr(0xF100) + u'나')
        self.view = build_search_view(self.source, self.table)

    def test_normalized_without_unifying_unknown(self):
        self.assertEqual('sample-hypua/1', self.view.mapping_version)
        self.assertEqual(self.source, self.view.source_text)
        self.assertEqual([0xF100], self.view.unknown_codepoints)
        # one-to-many: both source glyphs normalize to the same syllable
        self.assertEqual(u'나', self.view.normalized_text[0])
        self.assertEqual(u'나', self.view.normalized_text[1])

    def test_search_hits_trace_to_source_glyphs(self):
        hits = self.view.search(u'나')
        # glyph 0 and 1 both match, plus the native 나 at index 3
        self.assertEqual(3, len(hits))
        pua_hits = [h for h in hits if h.source_codepoints]
        self.assertEqual([0xEBD4], list(pua_hits[0].source_codepoints))
        self.assertEqual([0xEBD7], list(pua_hits[1].source_codepoints))
        native = [h for h in hits if not h.source_codepoints]
        self.assertEqual(1, len(native))
        # each hit carries the one-to-many source range
        self.assertEqual([(0xEBD4, 0xEBD4)], pua_hits[0].pua_ranges)

    def test_search_empty_query(self):
        self.assertEqual([], self.view.search(u''))

    def test_view_serialization(self):
        data = json.loads(self.view.dumps())
        self.assertEqual(self.source, data['source_text'])
        self.assertEqual('sample-hypua/1', data['mapping_version'])
        self.assertIn('U+F100', data['unknown_codepoints'])
        sources = [c['source_codepoints'] for c in data['characters']
                   if c['source_codepoints']]
        self.assertIn(['U+EBD4'], sources)

    def test_explain_hit_mentions_source(self):
        hit = self.view.search(u'나')[0]
        explanation = self.view.explain_hit(hit)
        self.assertIn('U+EBD4', explanation)
        self.assertIn('sample-hypua/1', explanation)


class TestRemap(TestCase):

    def setUp(self):
        self.source = (unichr(0xEBD4) + unichr(0xEBD7) + unichr(0xF100))
        self.v1 = build_search_view(self.source, sample_table_v1())
        self.v2 = remap(self.v1, sample_table_v2())

    def test_remap_preserves_source_and_switches_version(self):
        self.assertEqual(self.v1.source_text, self.v2.source_text)
        self.assertEqual('sample-hypua/1', self.v1.mapping_version)
        self.assertEqual('sample-hypua/2', self.v2.mapping_version)

    def test_diff_reports_changed_ranges(self):
        diff = diff_views(self.v1, self.v2)
        changed_sources = [c['source_codepoints']
                           for c in diff['changed_characters']]
        self.assertIn(['U+EBD7'], changed_sources)
        self.assertIn(['U+F100'], changed_sources)
        # EBD4 mapping is unchanged, so it must not appear
        flat = [s for pair in changed_sources for s in pair]
        self.assertNotIn('U+EBD4', flat)
        self.assertIn('U+EBD7..U+EBD7', diff['affected_pua_ranges'])

    def test_diff_detects_unknown_to_mapped(self):
        diff = diff_views(self.v1, self.v2)
        f100 = next(c for c in diff['changed_characters']
                    if c['source_codepoints'] == ['U+F100'])
        self.assertTrue(f100['old']['unknown'])
        self.assertFalse(f100['new']['unknown'])
        self.assertEqual(u'가', f100['new']['normalized'])

    def test_search_hit_changes(self):
        old_hits, new_hits, changed = search_hit_changes(
            self.v1, self.v2, u'나')
        # EBD7 moves from 나 to 낙: one fewer 나 hit after remap
        self.assertEqual(2, len(old_hits))
        self.assertEqual(1, len(new_hits))
        self.assertIn('U+EBD7..U+EBD7', changed)

    def test_diff_rejects_different_sources(self):
        other = build_search_view(u'다른글자', sample_table_v2())
        self.assertRaises(ValueError, diff_views, self.v1, other)


class TestHwp5Integration(TestCase):

    def test_views_trace_real_pua_glyphs(self):
        path = get_fixture_path('sample-5017.hwp')
        hwp5file = Hwp5File(path)
        try:
            views = list(iter_paratext_views(hwp5file, sample_table_v1()))
        finally:
            hwp5file.close()
        self.assertTrue(views)
        # every view carries its document location and mapping version
        for view in views:
            self.assertEqual(2, len(view.location))
            self.assertEqual('sample-hypua/1', view.mapping_version)
            self.assertEqual(len(view.normalized_text), len(view.records))
        # the fixture contains the U+F53A PUA glyph, unmapped in v1
        unknown = [cp for view in views for cp in view.unknown_codepoints]
        self.assertIn(0xF53A, unknown)

    def test_remap_real_hwp_changes_explained(self):
        path = get_fixture_path('sample-5017.hwp')
        table = PuaMappingTable(
            [MappingEntry(0xF53A, u'가')], version='t/1')
        revised = PuaMappingTable(
            [MappingEntry(0xF53A, u'나')], version='t/2')
        hwp5file = Hwp5File(path)
        try:
            view = next(v for v in iter_paratext_views(hwp5file, table)
                        if 0xF53A in v.unknown_codepoints
                        or any(0xF53A in r.source_codepoints
                               for r in v.records))
        finally:
            hwp5file.close()
        self.assertEqual([], view.unknown_codepoints)
        new_view = remap(view, revised)
        diff = diff_views(view, new_view)
        self.assertIn('U+F53A..U+F53A', diff['affected_pua_ranges'])
        old_hits, new_hits, _ = search_hit_changes(
            view, new_view, u'가')
        self.assertTrue(old_hits and not new_hits)

    def test_iter_views_match_paratext_models(self):
        path = get_fixture_path('sample-5017.hwp')
        hwp5file = Hwp5File(path)
        try:
            paratexts = [
                m for section in hwp5file.bodytext
                for m in hwp5file.bodytext[section].models()
                if m['type'] is ParaText
                and any(isinstance(c, str)
                        for _, c in m['content']['chunks'])]
            views = list(iter_paratext_views(
                hwp5file, sample_table_v1()))
        finally:
            hwp5file.close()
        self.assertEqual(len(paratexts), len(views))
        self.assertIsInstance(views[0], SearchView)
