# -*- coding: utf-8 -*-
#
#   pyhwp : hwp file format parser in python
#   Copyright (C) 2010-2023 mete0r <https://github.com/mete0r>
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Affero General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Affero General Public License for more details.
#
#   You should have received a copy of the GNU Affero General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from unittest import TestCase
import json

from hwp5.binmodel import Hwp5File
from hwp5.recordstream import Record
from hwp5.structured_import import format_record_report
from hwp5.structured_import import format_stream_report
from hwp5.structured_import import import_record
from hwp5.structured_import import import_stream
from hwp5.structured_import import report_to_json
from hwp5.tagids import HWPTAG_BEGIN

from .fixtures import get_fixture_path


def numbering_payload(unknowns=()):
    '''A Numbering payload.  ``unknowns`` adds the versioned UINT32 array.'''
    payload = b''
    for _ in range(7):
        payload += (0).to_bytes(4, 'little')       # flags
        payload += (0).to_bytes(2, 'little')       # width_correction
        payload += (0).to_bytes(2, 'little')       # space
        payload += (0).to_bytes(4, 'little', signed=True)  # charshape_id
        payload += (0).to_bytes(2, 'little')       # empty BSTR
    payload += (1).to_bytes(2, 'little')           # starting_number
    for value in unknowns:
        payload += value.to_bytes(4, 'little')
    return payload


def numbering_record(payload, seqno=0):
    return Record(HWPTAG_BEGIN + 7, 0, payload, seqno=seqno)


class ImportRecordTest(TestCase):

    def test_declared_old_structure_with_new_bytes_reports_tail(self):
        payload = numbering_payload(range(1, 8))
        report = import_record(numbering_record(payload), version='5.0.1.7')

        self.assertEqual('5.0.1.7', report['declared_version'])
        self.assertEqual([5, 0, 1, 7], report['declared_version_tuple'])
        self.assertEqual(128, report['payload_size'])
        self.assertEqual(dict(start=0, end=100), report['extent'])
        self.assertEqual(100, report['tail']['offset'])
        self.assertEqual(28, report['tail']['size'])
        self.assertEqual(
            ''.join('%02x000000' % i for i in range(1, 8)),
            report['tail']['hex'],
        )
        self.assertTrue(report['version_conflict'])
        self.assertEqual(['unconsumed_tail'],
                         [c['kind'] for c in report['contradictions']])
        detail = report['contradictions'][0]['detail']
        self.assertEqual(100, detail['offset'])
        self.assertEqual(28, detail['size'])
        self.assertFalse(detail['all_zero'])
        self.assertFalse(report['unconsumed']['all_zero'])
        self.assertFalse(report['verified'])

    def test_fields_have_payload_relative_ranges(self):
        report = import_record(
            numbering_record(numbering_payload()), version='5.0.1.7')
        fields = report['interpretation']['fields']
        by_path = dict((f['path'], f) for f in fields)

        flags = by_path['Numbering.levels[0].flags']
        self.assertEqual((0, 4), (flags['offset'], flags['end']))
        self.assertEqual(4, flags['size'])

        bstr = by_path['Numbering.levels[0].numbering_format']
        self.assertEqual((12, 14), (bstr['offset'], bstr['end']))

        starting = by_path['Numbering.starting_number']
        self.assertEqual((98, 100), (starting['offset'], starting['end']))

        self.assertEqual(100, report['interpretation']['field_tree'][0]['end'])

    def test_candidate_structure_records_affected_fields(self):
        payload = numbering_payload(range(1, 8))
        report = import_record(numbering_record(payload), version='5.0.1.7')

        self.assertEqual(1, len(report['candidates']))
        candidate = report['candidates'][0]
        self.assertEqual('5.0.3.0', candidate['structure_version'])
        self.assertEqual('>=5.0.3.0', candidate['structure_version_range'])
        self.assertEqual(128, candidate['consumed'])
        self.assertEqual(0, candidate['remaining'])
        self.assertTrue(candidate['complete'])
        self.assertEqual('complete', candidate['status'])
        self.assertEqual('unconfirmed', candidate['assessment'])
        self.assertFalse(candidate['verified'])

        added = candidate['added_fields']
        self.assertEqual(7, len(added))
        self.assertEqual('Numbering.unknown[0]', added[0]['path'])
        self.assertEqual((100, 104), (added[0]['offset'], added[0]['end']))
        self.assertEqual(1, added[0]['value'])
        self.assertEqual([], candidate['removed_fields'])
        self.assertEqual([], candidate['changed_fields'])

    def test_declared_new_with_old_bytes_finds_legacy(self):
        payload = numbering_payload()
        report = import_record(numbering_record(payload), version='5.0.3.0')

        self.assertEqual(0, report['tail']['size'])
        self.assertEqual('truncated_payload',
                         report['contradictions'][0]['kind'])
        self.assertEqual('does_not_fit',
                         report['interpretation']['assessment'])
        self.assertIsNotNone(report['interpretation']['error'])

        self.assertEqual(1, len(report['candidates']))
        candidate = report['candidates'][0]
        self.assertEqual('<5.0.3.0', candidate['structure'])
        self.assertEqual('complete', candidate['status'])
        self.assertEqual('unconfirmed', candidate['assessment'])

    def test_agreement_is_still_unconfirmed(self):
        payload = numbering_payload()
        report = import_record(numbering_record(payload), version='5.0.1.7')

        self.assertEqual([], report['contradictions'])
        self.assertEqual([], report['candidates'])
        self.assertEqual(0, report['tail']['size'])
        self.assertEqual('unconfirmed',
                         report['interpretation']['assessment'])
        self.assertFalse(report['verified'])

    def test_a_candidate_that_fits_is_not_verified(self):
        '''The central rule: exact consumption does not prove correctness.'''
        payload = numbering_payload(range(1, 8))
        report = import_record(numbering_record(payload), version='5.0.1.7')

        for candidate in report['candidates']:
            self.assertFalse(candidate['verified'])
            self.assertIn(candidate['assessment'],
                          ('unconfirmed', 'ambiguous', 'partial'))

        self.assertIn('not considered correct merely because it consumes',
                      report['assessment_policy'])
        self.assertFalse(report['verified'])

    def test_no_candidates_option(self):
        payload = numbering_payload(range(1, 8))
        report = import_record(numbering_record(payload), version='5.0.1.7',
                               include_candidates=False)
        self.assertEqual([], report['candidates'])
        self.assertEqual(28, report['tail']['size'])

    def test_version_is_required(self):
        self.assertRaises(ValueError, import_record,
                          numbering_record(numbering_payload()))

    def test_report_is_json_serializable(self):
        payload = numbering_payload(range(1, 8))
        report = import_record(numbering_record(payload), version='5.0.1.7')
        restored = json.loads(report_to_json(report))

        self.assertEqual('5.0.1.7', restored['declared_version'])
        self.assertEqual(1, len(restored['candidates']))
        self.assertEqual(7, len(restored['candidates'][0]['added_fields']))
        self.assertEqual('assessment_policy', sorted(restored)[0])

    def test_text_report_names_extent_tail_and_candidate(self):
        payload = numbering_payload(range(1, 8))
        report = import_record(numbering_record(payload), version='5.0.1.7')
        text = format_record_report(report)

        self.assertIn('interpreted: [0, 100)', text)
        self.assertIn('unconsumed: [100, 128)', text)
        self.assertIn('unconsumed_tail', text)
        self.assertIn('structure 5.0.3.0', text)
        self.assertIn('unconfirmed', text)

    def test_record_level_extent_aliases(self):
        payload = numbering_payload(range(1, 8))
        report = import_record(numbering_record(payload), version='5.0.1.7')

        self.assertEqual(report['extent'], report['interpreted'])
        self.assertEqual(100, report['consumed'])
        self.assertEqual(report['tail'], report['unconsumed'])
        self.assertEqual(28, report['unconsumed']['size'])
        self.assertEqual(7, len(report['candidates'][0]['fields']) -
                         len(report['interpretation']['fields']))


class AmbiguityTest(TestCase):
    '''Two structures can fit the same bytes while reading different fields.
    Exact consumption must not settle which one is right.
    '''

    def ambiguous_type(self):
        from six import add_metaclass
        from hwp5.dataio import StructType
        from hwp5.dataio import UINT16
        from hwp5.dataio import UINT32

        def before_v2(context, values):
            return context['version'] < (5, 0, 2, 0)

        def from_v2(context, values):
            return context['version'] >= (5, 0, 2, 0)

        def never(context, values):
            return False

        @add_metaclass(StructType)
        class Ambiguous(object):
            @classmethod
            def attributes(cls):
                # Before 5.0.2.0: one 4-byte field.
                yield dict(type=UINT32, name='legacy',
                           condition=before_v2)
                # From 5.0.2.0: two 2-byte fields.  The never-taken member
                # only declares the version boundary.
                yield dict(type=UINT16, name='modern_hi',
                           condition=from_v2, version=(5, 0, 2, 0))
                yield dict(type=UINT16, name='modern_lo',
                           condition=from_v2, version=(5, 0, 2, 0))
                yield dict(type=UINT16, name='marker',
                           condition=never, version=(5, 0, 2, 0))

        return Ambiguous

    def test_declared_structure_complete_but_alternative_differs(self):
        payload = b'\x01\x00\x02\x00'
        report = import_record(
            numbering_record(payload), version='5.0.1.7',
            model_type=self.ambiguous_type())

        self.assertEqual(0, report['tail']['size'])
        self.assertEqual('ambiguous', report['interpretation']['assessment'])
        self.assertFalse(report['verified'])
        self.assertIn('ambiguous_structure',
                      [c['kind'] for c in report['contradictions']])

        self.assertEqual(1, len(report['candidates']))
        candidate = report['candidates'][0]
        self.assertEqual('complete', candidate['status'])
        self.assertEqual('ambiguous', candidate['assessment'])
        self.assertTrue(candidate['conflicts_with_declared'])
        self.assertEqual(['Ambiguous.modern_hi', 'Ambiguous.modern_lo'],
                         [f['path'] for f in candidate['added_fields']])
        self.assertEqual(['Ambiguous.legacy'],
                         [f['path'] for f in candidate['removed_fields']])


class ImportStreamTest(TestCase):

    def test_uses_fileheader_version_and_keeps_extensions(self):
        hwp5file = Hwp5File(get_fixture_path('sample-5017.hwp'))
        report = import_stream(hwp5file.bodytext.section(0))

        self.assertEqual('5.0.1.7', report['declared_version'])
        self.assertEqual(128, report['summary']['record_count'])
        self.assertEqual(0, report['summary']['records_with_unconsumed_tail'])
        self.assertEqual(0, report['summary']['records_with_parse_error'])
        self.assertEqual(0, report['summary']['verified_records'])

        resolved = set(r['resolved_model'] for r in report['records'])
        self.assertIn('SectionDef', resolved)
        self.assertIn('TableControl', resolved)
        self.assertIn('TableCell', resolved)

        text = format_stream_report(report)
        self.assertIn('declared version 5.0.1.7', text)
        self.assertIn('no unconsumed payload', text)

    def test_summary_counts_tails_in_docinfo(self):
        hwp5file = Hwp5File(get_fixture_path('facename.hwp'))
        report = import_stream(hwp5file.docinfo)

        tailed = [r for r in report['records'] if r['tail']['size'] > 0]
        self.assertTrue(tailed)
        self.assertEqual(len(tailed),
                         report['summary']['records_with_unconsumed_tail'])
        self.assertEqual(len(tailed),
                         report['summary']['records_with_contradictions'])
        self.assertEqual(0, report['summary']['records_with_candidates'])

        text = format_stream_report(report, 'FileHeader')
        self.assertIn('(from FileHeader)', text)

    def test_old_structure_can_consume_new_bytes_without_error(self):
        '''This is the failure mode the feature makes visible.

        The binmodel parser imports a 5.0.3.0-shaped Numbering record as a
        5.0.1.7 record without raising; only the leftover exposes it.
        '''
        from hwp5.binmodel import parse_models

        payload = numbering_payload(range(1, 8))
        record = numbering_record(payload)

        models = list(parse_models(dict(version=(5, 0, 1, 7)), [record]))
        model = models[0]
        self.assertEqual(28, len(model['unparsed']))
        self.assertNotIn('unknown', model['content'])

        report = import_record(record, version='5.0.1.7')
        self.assertEqual(28, report['tail']['size'])
        self.assertEqual('5.0.3.0',
                         report['candidates'][0]['structure_version'])
