# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from unittest import TestCase

from hwp5 import xmlmodel
from hwp5.binmodel import Paragraph
from hwp5.binmodel import TableControl
from hwp5.binmodel import Text
from hwp5.stableid import CORRESPONDENCE_FORMAT
from hwp5.stableid import INDEX_FORMAT
from hwp5.stableid import StableIdAssigner
from hwp5.stableid import compute_fingerprint
from hwp5.stableid import give_elements_stable_id
from hwp5.stableid import index_from_jsonable
from hwp5.stableid import index_to_jsonable
from hwp5.stableid import reconcile
from hwp5.treeop import STARTEVENT, ENDEVENT

from .fixtures import get_fixture_path


def make_paragraph_events(text, attrs=None):
    paragraph_attrs = dict(attrs) if attrs else dict()
    text_attrs = dict(text=text)
    return [
        (STARTEVENT, (Paragraph, paragraph_attrs, dict())),
        (STARTEVENT, (Text, text_attrs, dict())),
        (ENDEVENT, (Text, text_attrs, dict())),
        (ENDEVENT, (Paragraph, paragraph_attrs, dict())),
    ]


def make_document_events(texts):
    events = []
    for text in texts:
        events.extend(make_paragraph_events(text))
    return events


def collect_paragraph_ids(events, assigner=None):
    assigner = assigner or StableIdAssigner()
    ids = []
    for event, item in give_elements_stable_id(iter(events), assigner):
        model, attributes, context = item
        if event is STARTEVENT and model is Paragraph:
            ids.append(attributes['paragraph_id'])
    return ids


class TestStableIdAssignment(TestCase):

    def test_ids_survive_prepend(self):
        ids_before = collect_paragraph_ids(
            make_document_events(['one', 'two', 'three']))
        ids_after = collect_paragraph_ids(
            make_document_events(['zero', 'one', 'two', 'three']))
        self.assertEqual(ids_before, ids_after[1:])
        self.assertEqual(4, len(set(ids_after)))

    def test_ids_are_deterministic(self):
        ids1 = collect_paragraph_ids(make_document_events(['a', 'b']))
        ids2 = collect_paragraph_ids(make_document_events(['a', 'b']))
        self.assertEqual(ids1, ids2)

    def test_duplicate_paragraphs_get_occurrence_suffix(self):
        ids = collect_paragraph_ids(
            make_document_events(['dup', 'other', 'dup']))
        self.assertEqual(3, len(set(ids)))
        self.assertEqual(ids[0] + '#2', ids[2])

    def test_attributes_take_part_in_identity(self):
        ids = collect_paragraph_ids(
            make_paragraph_events('same', dict(parashape_id=0)) +
            make_paragraph_events('same', dict(parashape_id=1)))
        self.assertNotEqual(ids[0], ids[1])

    def test_nested_elements_get_ids(self):
        paragraph_attrs = dict()
        text_attrs = dict(text='cell text')
        table_attrs = dict()
        events = [
            (STARTEVENT, (TableControl, table_attrs, dict())),
            (STARTEVENT, (Paragraph, paragraph_attrs, dict())),
            (STARTEVENT, (Text, text_attrs, dict())),
            (ENDEVENT, (Text, text_attrs, dict())),
            (ENDEVENT, (Paragraph, paragraph_attrs, dict())),
            (ENDEVENT, (TableControl, table_attrs, dict())),
        ]
        assigner = StableIdAssigner()
        events = list(give_elements_stable_id(iter(events), assigner))
        self.assertTrue('paragraph_id' in paragraph_attrs)
        self.assertTrue('table_id' in table_attrs)
        self.assertTrue(table_attrs['table_id'].startswith('t'))
        self.assertTrue(paragraph_attrs['paragraph_id'].startswith('p'))
        kinds = set(record['kind'] for record in assigner.records)
        self.assertEqual(set(['paragraph', 'table']), kinds)

    def test_position_is_recorded_as_evidence(self):
        assigner = StableIdAssigner()
        collect_paragraph_ids(make_document_events(['a', 'b']), assigner)
        positions = [record['position'] for record in assigner.records]
        self.assertEqual(2, len(positions))
        self.assertNotEqual(positions[0], positions[1])
        self.assertEqual('Paragraph', positions[0][-1][0])

    def test_baseline_ids_are_continued(self):
        baseline = StableIdAssigner()
        baseline_ids = collect_paragraph_ids(
            make_document_events(['one', 'two']), baseline)

        assigner = StableIdAssigner(baseline.records)
        ids = collect_paragraph_ids(
            make_document_events(['zero', 'one', 'two']), assigner)
        self.assertEqual(baseline_ids, ids[1:])
        self.assertNotIn(ids[0], baseline_ids)

    def test_retired_ids_are_not_reused(self):
        baseline = StableIdAssigner()
        baseline_ids = collect_paragraph_ids(
            make_document_events(['gone', 'stays']), baseline)

        assigner = StableIdAssigner(baseline.records)
        ids = collect_paragraph_ids(
            make_document_events(['brand new', 'stays']), assigner)
        self.assertEqual(baseline_ids[1], ids[1])
        self.assertNotIn(ids[0], baseline_ids)


class TestReconcile(TestCase):

    def collect(self, texts):
        assigner = StableIdAssigner()
        collect_paragraph_ids(make_document_events(texts), assigner)
        return assigner.records

    def collect_events(self, events):
        assigner = StableIdAssigner()
        list(give_elements_stable_id(iter(events), assigner))
        return assigner.records

    def test_format(self):
        correspondence = reconcile(self.collect(['a']), self.collect(['a']))
        self.assertEqual(CORRESPONDENCE_FORMAT, correspondence['format'])

    def test_matched_elements_keep_identity_with_evidence(self):
        baseline = self.collect(['one', 'two'])
        current = self.collect(['zero', 'one', 'two'])
        correspondence = reconcile(baseline, current)

        matched = correspondence['matched']
        self.assertEqual(2, len(matched))
        self.assertEqual([r['id'] for r in baseline],
                         [m['id'] for m in matched])
        for entry in matched:
            self.assertIn('baseline_position', entry)
            self.assertIn('current_position', entry)
            self.assertNotEqual(entry['baseline_position'],
                                entry['current_position'])
            self.assertTrue(entry['text_excerpt'])

        self.assertEqual(1, len(correspondence['new']))
        self.assertEqual(0, len(correspondence['retired']))

    def test_split_leaves_candidates_with_contribution_ranges(self):
        baseline = self.collect(['Hello world'])
        current = self.collect(['Hello ', 'world'])
        correspondence = reconcile(baseline, current)

        self.assertEqual(0, len(correspondence['matched']))
        retired = correspondence['retired']
        new = correspondence['new']
        self.assertEqual(1, len(retired))
        self.assertEqual(2, len(new))

        source_id = baseline[0]['id']
        by_excerpt = dict((entry['text_excerpt'], entry) for entry in new)
        hello = by_excerpt['Hello']
        world = by_excerpt['world']

        self.assertEqual(1, len(hello['candidates']))
        candidate = hello['candidates'][0]
        self.assertEqual(source_id, candidate['source_id'])
        self.assertEqual('split', candidate['relation'])
        self.assertEqual([0, 6], candidate['source_range'])
        self.assertEqual([0, 6], candidate['target_range'])

        candidate = world['candidates'][0]
        self.assertEqual(source_id, candidate['source_id'])
        self.assertEqual('split', candidate['relation'])
        self.assertEqual([6, 11], candidate['source_range'])
        self.assertEqual([0, 5], candidate['target_range'])

        # the retired element points to the same candidates
        self.assertEqual(2, len(retired[0]['candidates']))

    def test_merge_leaves_candidates_with_contribution_ranges(self):
        baseline = self.collect(['Hello ', 'world'])
        current = self.collect(['Hello world'])
        correspondence = reconcile(baseline, current)

        self.assertEqual(0, len(correspondence['matched']))
        new = correspondence['new']
        retired = correspondence['retired']
        self.assertEqual(1, len(new))
        self.assertEqual(2, len(retired))

        candidates = new[0]['candidates']
        self.assertEqual(2, len(candidates))
        self.assertEqual(set(['merge']),
                         set(c['relation'] for c in candidates))
        by_source = dict((c['source_id'], c) for c in candidates)
        hello = by_source[baseline[0]['id']]
        world = by_source[baseline[1]['id']]
        self.assertEqual([0, 6], hello['source_range'])
        self.assertEqual([0, 6], hello['target_range'])
        self.assertEqual([0, 5], world['source_range'])
        self.assertEqual([6, 11], world['target_range'])

    def test_duplicate_growth_allocates_fresh_id(self):
        baseline = self.collect(['note', 'note'])
        current = self.collect(['note', 'note', 'note'])
        correspondence = reconcile(baseline, current)

        self.assertEqual(2, len(correspondence['matched']))
        self.assertEqual(1, len(correspondence['new']))
        self.assertEqual(0, len(correspondence['retired']))
        # the extra occurrence gets a fresh id which does not
        # overwrite any index of the baseline
        new_id = correspondence['new'][0]['id']
        self.assertNotIn(new_id, [r['id'] for r in baseline])

    def test_restyled_duplicate_leaves_duplicate_candidates(self):
        baseline = self.collect_events(
            make_paragraph_events('note', dict(parashape_id=0)) +
            make_paragraph_events('note', dict(parashape_id=0)))
        current = self.collect_events(
            make_paragraph_events('note', dict(parashape_id=1)))
        correspondence = reconcile(baseline, current)

        self.assertEqual(0, len(correspondence['matched']))
        retired = correspondence['retired']
        new = correspondence['new']
        self.assertEqual(2, len(retired))
        self.assertEqual(1, len(new))
        # both retired duplicates remain as candidates of the new
        # element, each contributing its whole text
        candidates = new[0]['candidates']
        self.assertEqual(2, len(candidates))
        self.assertEqual(set(['duplicate']),
                         set(c['relation'] for c in candidates))
        self.assertEqual(set(r['id'] for r in baseline),
                         set(c['source_id'] for c in candidates))


class TestIndexSerialization(TestCase):

    def test_roundtrip(self):
        assigner = StableIdAssigner()
        collect_paragraph_ids(make_document_events(['a', 'b']), assigner)
        document = index_to_jsonable(assigner.records)
        self.assertEqual(INDEX_FORMAT, document['format'])
        records = index_from_jsonable(document)
        self.assertEqual(assigner.records, records)

    def test_rejects_foreign_format(self):
        self.assertRaises(ValueError, index_from_jsonable,
                          dict(format='something-else', records=[]))


class TestStableIdExport(TestCase):

    hwp5file_name = 'sample-5017.hwp'

    def export_paragraph_ids(self, **kwargs):
        path = get_fixture_path(self.hwp5file_name)
        hwp5file = xmlmodel.Hwp5File(path)
        assigner = kwargs.pop('id_assigner', None) or StableIdAssigner()
        ids = []
        for event, item in hwp5file.events(id_assigner=assigner):
            model, attributes, context = item
            if event is STARTEVENT and model is Paragraph:
                ids.append(attributes['paragraph_id'])
        return ids, assigner

    def test_export_ids_are_stable_and_unique(self):
        ids1, assigner1 = self.export_paragraph_ids()
        ids2, assigner2 = self.export_paragraph_ids()
        self.assertEqual(ids1, ids2)
        self.assertEqual(len(ids1), len(set(ids1)))
        for stable_id in ids1:
            self.assertTrue(stable_id.startswith('p'))
        paragraph_records = [r for r in assigner1.records
                             if r['kind'] == 'paragraph']
        self.assertEqual(len(ids1), len(paragraph_records))
        all_ids = [r['id'] for r in assigner1.records]
        self.assertEqual(len(all_ids), len(set(all_ids)))

    def test_export_with_baseline_continues_ids(self):
        ids, assigner = self.export_paragraph_ids()
        assigner2 = StableIdAssigner(assigner.records)
        ids2, assigner2 = self.export_paragraph_ids(id_assigner=assigner2)
        self.assertEqual(ids, ids2)
        correspondence = reconcile(assigner.records, assigner2.records)
        self.assertEqual(len(assigner.records),
                         len(correspondence['matched']))
        self.assertEqual(0, len(correspondence['new']))
        self.assertEqual(0, len(correspondence['retired']))

    def test_fingerprint_is_content_derived(self):
        fp1 = compute_fingerprint('paragraph', '', 'hello')
        fp2 = compute_fingerprint('paragraph', '', 'hello')
        fp3 = compute_fingerprint('paragraph', '', 'world')
        self.assertEqual(fp1, fp2)
        self.assertNotEqual(fp1, fp3)
