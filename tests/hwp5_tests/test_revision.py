# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from unittest import TestCase
import copy

from hwp5.binmodel import Hwp5File
from hwp5.revision.compare import (
    STATUS_REMAPPED,
    STATUS_REORDER_AFFECTED,
    STATUS_RESOURCE_CHANGE,
    STATUS_TEXT_CHANGE,
    STATUS_UNCHANGED,
    RevisionComparison,
    RevisionDocument,
)
from hwp5.revision.paragraphs import SectionParagraphs
from hwp5.revision.resources import (
    FONT,
    LANGUAGES,
    Resource,
    ResourceTable,
)
from hwp5.utils import cached_property

from .fixtures import get_fixture_path


class RevisionTestBase(TestCase):

    hwp5file_name = 'facename.hwp'

    @cached_property
    def hwp5file(self):
        return Hwp5File(get_fixture_path(self.hwp5file_name))

    @cached_property
    def old_document(self):
        return RevisionDocument.open(self.hwp5file)


class Helpers(object):

    @staticmethod
    def clone_table(table, font_transform=None, charshape_transform=None):
        new = ResourceTable()
        new.resources = {}
        fonts = []
        for lang in LANGUAGES:
            items = [r for r in table.resources[FONT] if r.lang == lang]
            items = font_transform(lang, items) if font_transform else items
            for index, resource in enumerate(items):
                content = copy.deepcopy(resource.content)
                fonts.append(Resource(FONT, index, content, lang=lang))
        new.resources[FONT] = fonts

        for kind, items in table.resources.items():
            if kind == FONT:
                continue
            cloned = []
            for resource in items:
                content = copy.deepcopy(resource.content)
                if charshape_transform and kind == 'charshape':
                    content = charshape_transform(resource, content)
                cloned.append(Resource(kind, resource.index, content))
            new.resources[kind] = cloned
        return new

    @staticmethod
    def recompute_font_pointers(old_table, new_table):
        '''Rewrite charshape font ids so reversed font tables keep meaning.'''
        remap = {}
        for lang in LANGUAGES:
            old_items = [r for r in old_table.resources[FONT]
                         if r.lang == lang]
            new_items = [r for r in new_table.resources[FONT]
                         if r.lang == lang]
            by_content = {}
            for resource in new_items:
                by_content.setdefault(str(resource.content), []).append(
                    resource.index)
            for resource in old_items:
                indexes = by_content[str(resource.content)]
                remap[(lang, resource.index)] = indexes.pop(0)

        def transform(resource, content):
            content = dict(content)
            content['font_face'] = dict(
                (lang, remap[(lang, index)])
                for lang, index in content['font_face'].items())
            return content

        return transform

    @classmethod
    def rebuild_sections(cls, hwp5file, document, table,
                         paragraph_transform=None):
        sections = []
        for section in document.sections:
            models = list(hwp5file.bodytext.section(
                section.section_index).models())
            if paragraph_transform:
                models = [paragraph_transform(model) for model in models]
            sections.append(SectionParagraphs.from_models(
                section.section_index, models, table))
        return sections


class ResourceReorderTest(RevisionTestBase, Helpers):

    def new_document(self):
        old = self.old_document
        new_table = self.clone_table(old.table)

        # Reverse each language's font table; ids move, content does not.
        reordered = ResourceTable()
        reordered.resources = {}
        reordered.resources[FONT] = []
        for lang in LANGUAGES:
            items = [r for r in new_table.resources[FONT] if r.lang == lang]
            for index, resource in enumerate(reversed(items)):
                reordered.resources[FONT].append(
                    Resource(FONT, index, resource.content, lang=lang))
        for kind, items in new_table.resources.items():
            if kind != FONT:
                reordered.resources[kind] = items

        # Rewrite every charshape font pointer to the moved position.
        transform = self.recompute_font_pointers(old.table, reordered)
        for resource in reordered.resources['charshape']:
            resource.content = transform(resource, resource.content)

        sections = self.rebuild_sections(self.hwp5file, old, reordered)
        return RevisionDocument(reordered, sections)

    def test_reorder_does_not_touch_paragraphs(self):
        comparison = RevisionComparison(self.old_document,
                                        self.new_document()).compare()
        summary = comparison.summary()

        self.assertTrue(summary['categories']['resource_reorder'] > 0)
        self.assertEqual(0, summary['categories']['resource_content_change'])
        self.assertEqual(0, summary['categories']['body_change'])
        self.assertEqual(0, summary['categories']['affected_paragraphs'])
        for result in comparison.paragraphs:
            self.assertEqual(STATUS_UNCHANGED, result.status)
            self.assertFalse(result.rebuilt)

    def test_reorder_understands_rewritten_ids(self):
        comparison = RevisionComparison(self.old_document,
                                        self.new_document()).compare()
        report = comparison.to_dict()
        # Font ids moved, so charshapes now point at different font indexes.
        reordered_kinds = set(item['kind']
                              for item in report['resource_reorder'])
        self.assertIn(FONT, reordered_kinds)

        old_charshapes = self.old_document.table.resources['charshape']
        new_charshapes = comparison.new.table.resources['charshape']
        rewritten = [old for old, new in zip(old_charshapes, new_charshapes)
                     if old.content['font_face'] != new.content['font_face']]
        self.assertTrue(rewritten)

        # The resolved meaning must still be identical for every paragraph.
        for result in comparison.paragraphs:
            self.assertEqual(result.old.semantic, result.new.semantic)


class ResourceContentChangeTest(RevisionTestBase, Helpers):

    def new_document(self, lang='ko', index=3, name=u'새글꼴'):
        old = self.old_document

        def font_transform(lang_name, items):
            if lang_name != lang:
                return items
            result = []
            for resource in items:
                content = copy.deepcopy(resource.content)
                if resource.index == index:
                    content['name'] = name
                result.append(Resource(FONT, resource.index, content,
                                       lang=lang_name))
            return result

        new_table = self.clone_table(old.table, font_transform=font_transform)
        sections = self.rebuild_sections(self.hwp5file, old, new_table)
        return RevisionDocument(new_table, sections)

    def test_font_edit_is_resource_change(self):
        comparison = RevisionComparison(self.old_document,
                                        self.new_document()).compare()
        summary = comparison.summary()
        self.assertEqual(1,
                         summary['categories']['resource_content_change'])
        self.assertTrue(summary['categories']['affected_paragraphs'] > 0)

        affected = [r for r in comparison.paragraphs if r.affected]
        unaffected = [r for r in comparison.paragraphs if not r.affected]
        self.assertTrue(affected)
        self.assertTrue(unaffected)
        for result in affected:
            self.assertEqual(STATUS_RESOURCE_CHANGE, result.status)
            self.assertTrue(result.rebuilt)
        for result in unaffected:
            self.assertEqual(STATUS_UNCHANGED, result.status)
            self.assertFalse(result.rebuilt)

    def test_unchanged_paragraphs_keep_old_index(self):
        comparison = RevisionComparison(self.old_document,
                                        self.new_document()).compare()
        report = comparison.to_dict(include_indices=True)
        for item in report['paragraphs']:
            if item['affected']:
                self.assertEqual('new', item['language_index_source'])
            else:
                self.assertEqual('old', item['language_index_source'])

    def test_charshape_edit_marks_only_its_users(self):
        old = self.old_document

        def charshape_transform(resource, content):
            if resource.kind == 'charshape' and resource.index == 7:
                content = dict(content)
                content['basesize'] = 1200
            return content

        new_table = self.clone_table(
            old.table, charshape_transform=charshape_transform)
        new = RevisionDocument(
            new_table,
            self.rebuild_sections(self.hwp5file, old, new_table))
        comparison = RevisionComparison(old, new).compare()
        self.assertEqual(1,
                         comparison.summary()['categories'][
                             'resource_content_change'])
        for result in comparison.paragraphs:
            if result.affected:
                self.assertIn(7, [span.charshape_id
                                  for span in result.old.spans])


class DuplicateResourceTest(RevisionTestBase, Helpers):

    hwp5file_name = 'lists.hwp'

    def test_duplicates_are_not_paired_by_position(self):
        old = self.old_document
        comparison = RevisionComparison(old, old).compare()
        duplicates = comparison.match.duplicate_classes
        self.assertTrue(duplicates)
        kinds = set(duplicate['kind'] for duplicate in duplicates)
        self.assertIn('numbering', kinds)
        for duplicate in duplicates:
            # Duplicate groups are reported as an equivalence class; they are
            # never split into positional one-to-one pairs.
            self.assertTrue(duplicate['old_count'] >= 2)
            self.assertEqual(duplicate['old_count'],
                             duplicate['new_count'])
        # Resources inside a duplicate class are not reported as
        # removed/added just because they could not be paired.
        self.assertEqual(0, len(comparison.match.unmatched_old()))
        self.assertEqual(0, len(comparison.match.unmatched_new()))

    def test_duplicate_group_is_not_split_into_pairs(self):
        old = self.old_document
        comparison = RevisionComparison(old, old).compare()
        counts = comparison.summary()['resources']
        duplicate = [item for item in comparison.match.duplicate_classes
                     if item['kind'] == 'numbering'][0]
        duplicate_indexes = set(r.index for r in duplicate['old'])

        # The duplicated numberings are neither paired nor reported as
        # removed/added; they stay an interchangeable class.
        for pair in comparison.match.pairs:
            if pair.kind == 'numbering':
                self.assertNotIn(pair.old.index, duplicate_indexes)
        self.assertEqual(0, counts['removed'])
        self.assertEqual(0, counts['added'])
        self.assertEqual(0, len(comparison.match.unmatched_old()))
        self.assertEqual(0, len(comparison.match.unmatched_new()))


class BodyChangeTest(RevisionTestBase, Helpers):

    def new_document_with_text(self, replace):
        old = self.old_document

        def transform(model):
            if model['type'].__name__ != 'ParaText':
                return model
            content = dict(model['content'])
            chunks = []
            for range_, chunk in content['chunks']:
                if isinstance(chunk, str):
                    for old_text, new_text in replace.items():
                        chunk = chunk.replace(old_text, new_text)
                chunks.append((range_, chunk))
            content['chunks'] = chunks
            return dict(model, content=content)

        sections = self.rebuild_sections(self.hwp5file, old, old.table,
                                         paragraph_transform=transform)
        return RevisionDocument(old.table, sections)

    def test_text_change_is_body_change(self):
        old = self.old_document
        new = self.new_document_with_text({u'ko바탕': u'ko새바탕'})
        comparison = RevisionComparison(old, new).compare()
        summary = comparison.summary()
        self.assertTrue(summary['categories']['body_change'] > 0)
        self.assertEqual(0, summary['categories']['resource_content_change'])
        statuses = [r.status for r in comparison.paragraphs]
        self.assertIn(STATUS_TEXT_CHANGE, statuses)
        affected = [r for r in comparison.paragraphs if r.affected]
        self.assertTrue(all(r.rebuilt for r in affected))


class ReorderAttributionTest(RevisionTestBase, Helpers):

    def _document(self, reorder=True, edit=None):
        old = self.old_document
        new_table = ResourceTable()
        new_table.resources = {}
        new_table.resources[FONT] = []
        for lang in LANGUAGES:
            items = [r for r in old.table.resources[FONT] if r.lang == lang]
            if reorder and lang == 'ko':
                items = list(reversed(items))
            for index, resource in enumerate(items):
                content = copy.deepcopy(resource.content)
                if edit and edit[0] == lang and edit[1] == resource.index:
                    content['name'] = edit[2]
                new_table.resources[FONT].append(
                    Resource(FONT, index, content, lang=lang))
        for kind, items in old.table.resources.items():
            if kind != FONT:
                new_table.resources[kind] = [
                    Resource(kind, r.index, copy.deepcopy(r.content))
                    for r in items]
        sections = self.rebuild_sections(self.hwp5file, old, new_table)
        return RevisionDocument(new_table, sections)

    def test_pure_reorder_is_attributed_to_reorder(self):
        comparison = RevisionComparison(self.old_document,
                                        self._document()).compare()
        summary = comparison.summary()['categories']
        self.assertGreaterEqual(summary['resource_reorder'], 1)
        self.assertEqual(0, summary['resource_content_change'])
        self.assertEqual(0, summary['body_change'])
        self.assertEqual(summary['resource_reorder_affected'],
                         summary['affected_paragraphs'])
        for result in comparison.paragraphs:
            if result.affected:
                self.assertEqual(STATUS_REORDER_AFFECTED, result.status)

    def test_reorder_plus_content_edit_keeps_causes_separate(self):
        comparison = RevisionComparison(
            self.old_document,
            self._document(edit=('ko', 3, u'새글꼴'))).compare()
        summary = comparison.summary()['categories']
        self.assertEqual(1, summary['resource_content_change'])
        self.assertGreaterEqual(summary['resource_reorder_affected'], 1)
        statuses = set(result.status for result in comparison.paragraphs
                       if result.affected)
        self.assertEqual(set([STATUS_REORDER_AFFECTED,
                              STATUS_RESOURCE_CHANGE]), statuses)

        for result in comparison.paragraphs:
            if result.status == STATUS_RESOURCE_CHANGE:
                texts = [span.text for span in result.old.spans]
                self.assertTrue(any(u'휴먼명조' in text for text in texts))


class RemappedIdTest(RevisionTestBase, Helpers):

    hwp5file_name = 'parashape.hwp'

    def _swap(self, value):
        return {14: 15, 15: 14}.get(value, value)

    def new_document(self):
        old = self.old_document
        new_table = self.clone_table(old.table)

        items = new_table.resources['parashape']
        items[14].content, items[15].content = \
            items[15].content, items[14].content

        def transform(model):
            if model['type'].__name__ != 'Paragraph':
                return model
            content = dict(model['content'])
            content['parashape_id'] = self._swap(content['parashape_id'])
            return dict(model, content=content)

        sections = self.rebuild_sections(self.hwp5file, old, new_table,
                                         paragraph_transform=transform)
        return RevisionDocument(new_table, sections)

    def test_rewritten_ids_keep_semantics(self):
        comparison = RevisionComparison(self.old_document,
                                        self.new_document()).compare()
        summary = comparison.summary()['categories']
        self.assertGreaterEqual(summary['resource_reorder'], 2)
        self.assertEqual(0, summary['resource_content_change'])
        self.assertEqual(0, summary['body_change'])
        self.assertEqual(0, summary['affected_paragraphs'])

        remapped = [r for r in comparison.paragraphs
                    if r.status == STATUS_REMAPPED]
        self.assertTrue(remapped)
        for result in remapped:
            self.assertFalse(result.rebuilt)
            self.assertEqual(result.old.semantic, result.new.semantic)
            self.assertNotEqual(result.old.numeric_references(),
                                result.new.numeric_references())


class DiffCommandTest(RevisionTestBase, Helpers):

    def run_diff(self, *extra):
        from argparse import Namespace
        from hwp5.hwp5diff import run
        import os
        import tempfile
        fd, path = tempfile.mkstemp(suffix='.txt')
        os.close(fd)
        try:
            args = Namespace(
                old_hwp5file=get_fixture_path('facename.hwp'),
                new_hwp5file=get_fixture_path('facename.hwp'),
                output=path, format='text', indices=False, verbose=False)
            for key, value in zip(extra[::2], extra[1::2]):
                setattr(args, key, value)
            self.assertEqual(0, run(args))
            with open(path, 'rb') as f:
                return f.read().decode('utf-8')
        finally:
            os.unlink(path)

    def test_text_output_reports_categories(self):
        output = self.run_diff()
        self.assertIn('resources: reordered=0 content-changed=0', output)
        self.assertIn('body: changed=0', output)

    def test_json_output_is_valid(self):
        import json
        output = self.run_diff('format', 'json')
        report = json.loads(output)
        self.assertEqual(
            ['body_change', 'paragraphs', 'reorder_affected',
             'resource_content_change', 'resource_dependency_change',
             'resource_reorder', 'resources', 'summary'],
            sorted(report))
        self.assertEqual(0, report['summary']['categories']['body_change'])
