import unittest

import arff
import arff_merge


HEADER_A = '''@RELATION left
@ATTRIBUTE id INTEGER
@ATTRIBUTE color {red, green, blue}
@ATTRIBUTE score REAL
@DATA
'''

HEADER_B = '''@RELATION right
@ATTRIBUTE score REAL
@ATTRIBUTE color {blue, red, green}
@ATTRIBUTE id INTEGER
@DATA
'''


class TestMergePartitions(unittest.TestCase):
    def test_aligns_by_name_not_position(self):
        left = HEADER_A + '1,red,10\n'
        right = HEADER_B + '20,blue,2\n'

        result = arff_merge.merge_partitions(left, right)

        self.assertEqual(
            result['attributes'],
            [('id', 'INTEGER'),
             ('color', ['red', 'green', 'blue']),
             ('score', 'REAL')])
        self.assertEqual(result['data'], [[1, 'red', 10.0],
                                          [2, 'blue', 20.0]])
        self.assertEqual(result['conflicts'], [])

    def test_sparse_rows_use_partition_local_columns(self):
        left = HEADER_A + '{0 7,2 2.5}\n'
        right = HEADER_B + '{0 4.5}\n'

        result = arff_merge.merge_partitions(left, right)

        self.assertEqual(result['data'], [
            [7, 'red', 2.5],
            [0, 'blue', 4.5],
        ])
        self.assertEqual(result['provenance'], [
            [arff_merge.SourceRef(0, 0),
             arff_merge.SourceRef(0, 1),
             arff_merge.SourceRef(0, 2)],
            [arff_merge.SourceRef(1, 2),
             arff_merge.SourceRef(1, 1),
             arff_merge.SourceRef(1, 0)],
        ])
        self.assertEqual(
            [cell['explicit'] for cell in result['sourced_data'][0]],
            [True, False, True])
        self.assertEqual(
            [cell['explicit'] for cell in result['sourced_data'][1]],
            [False, False, True])

    def test_nominal_domains_are_remapped_by_label(self):
        left = HEADER_A + '1,blue,0\n'
        right = HEADER_B + '0,green,2\n'

        result = arff_merge.merge_partitions(left, right, relation='joined')

        color = dict(result['attributes'])['color']
        self.assertEqual(color, ['red', 'green', 'blue'])
        self.assertEqual([row[1] for row in result['data']],
                         ['blue', 'green'])
        self.assertEqual(arff.loads(arff.dumps(result))['data'],
                         result['data'])

    def test_provenance_records_source_column_per_value(self):
        left = HEADER_A + '1,red,10\n'
        right = HEADER_B + '20,blue,2\n'

        result = arff_merge.merge_partitions(left, right)

        self.assertEqual(result['provenance'], [
            [arff_merge.SourceRef(0, 0),
             arff_merge.SourceRef(0, 1),
             arff_merge.SourceRef(0, 2)],
            [arff_merge.SourceRef(1, 2),
             arff_merge.SourceRef(1, 1),
             arff_merge.SourceRef(1, 0)],
        ])
        self.assertEqual(result['source_columns'], [[0, 1, 2], [2, 1, 0]])
        self.assertEqual(result['source_partitions'], [[0, 0, 0], [1, 1, 1]])
        self.assertEqual(result['sourced_data'][1][0],
                         {'attribute': 'id', 'value': 2,
                          'partition': 1, 'column': 2, 'explicit': True})
        self.assertEqual(
            [(entry['partition'], entry['column'],
              entry['type']) for entry in result['field_sources']['id']],
            [(0, 0, 'INTEGER'), (1, 2, 'INTEGER')])

    def test_missing_partition_column_keeps_null_provenance(self):
        left = HEADER_A + '1,red,10\n'
        right = '''@RELATION right
@ATTRIBUTE score REAL
@DATA
20
'''

        result = arff_merge.merge_partitions(left, right)

        self.assertEqual(result['data'], [[1, 'red', 10.0],
                                          [None, None, 20.0]])
        self.assertEqual(result['provenance'][1],
                         [None, None, arff_merge.SourceRef(1, 0)])
        self.assertIsNone(result['sourced_data'][1][0]['column'])

    def test_incompatible_types_are_reported_not_padded(self):
        left = HEADER_A + '1,red,10\n'
        right = '''@RELATION right
@ATTRIBUTE score STRING
@ATTRIBUTE id INTEGER
@DATA
high,2
'''

        result = arff_merge.merge_partitions(left, right)

        self.assertNotIn('score', dict(result['attributes']))
        self.assertEqual(len(result['conflicts']), 1)
        conflict = result['conflicts'][0]
        self.assertEqual(conflict['field'], 'score')
        self.assertEqual(conflict['reason'], 'incompatible_types')
        self.assertEqual(conflict['resolved'], 'omitted')
        self.assertEqual(conflict['partitions'], [0, 1])
        self.assertEqual(
            [(source['partition'], source['column'], source['type'])
             for source in conflict['sources']],
            [(0, 2, 'REAL'), (1, 0, 'STRING')])
        self.assertEqual(
            [(entry['field'], entry['partition'], entry['column'])
             for entry in result['conflict_columns']],
            [('score', 0, 2), ('score', 1, 0)])
        self.assertEqual(arff.loads(arff.dumps(result))['data'],
                         [[1, 'red'], [2, None]])

    def test_conflicts_can_raise(self):
        left = HEADER_A + '1,red,10\n'
        right = '''@RELATION right
@ATTRIBUTE score STRING
@DATA
high
'''

        with self.assertRaises(arff_merge.TypeConflictError) as caught:
            arff_merge.merge_partitions(left, right, on_conflict='raise')

        self.assertEqual(caught.exception.conflict['field'], 'score')

    def test_integer_and_real_are_losslessly_unified(self):
        left = '''@RELATION a
@ATTRIBUTE value INTEGER
@DATA
1
'''
        right = '''@RELATION b
@ATTRIBUTE value REAL
@DATA
1.5
'''

        result = arff_merge.merge_partitions(left, right)

        self.assertEqual(result['attributes'], [('value', 'REAL')])
        self.assertEqual(result['data'], [[1.0], [1.5]])
        self.assertEqual(result['conflicts'], [])

    def test_contract_controls_order_without_copying_old_columns(self):
        left = HEADER_A + '1,red,10\n'
        right = HEADER_B + '20,blue,2\n'

        result = arff_merge.merge_partitions(
            left, right,
            contract=[('score', 'REAL'), ('id', 'INTEGER'),
                      ('color', ['red', 'green', 'blue'])])

        self.assertEqual(result['attributes'],
                         [('score', 'REAL'), ('id', 'INTEGER'),
                          ('color', ['red', 'green', 'blue'])])
        self.assertEqual(result['data'], [[10.0, 1, 'red'],
                                          [20.0, 2, 'blue']])
        self.assertEqual(result['conflicts'], [])

    def test_contract_nominal_domain_is_extended_by_labels(self):
        left = HEADER_A + '1,red,10\n'
        right = HEADER_B + '20,blue,2\n'

        result = arff_merge.merge_partitions(
            left, right, contract=[('id', 'INTEGER'),
                                   ('color', ['red', 'green'])])

        self.assertEqual(
            dict(result['attributes'])['color'],
            ['red', 'green', 'blue'])
        self.assertEqual([row[1] for row in result['data']],
                         ['red', 'blue'])
        self.assertEqual(result['conflicts'], [])

    def test_contract_type_mismatch_is_a_conflict(self):
        left = HEADER_A + '1,red,10\n'
        right = HEADER_B + '20,blue,2\n'

        result = arff_merge.merge_partitions(
            left, right, contract=[('score', 'INTEGER'), ('id', 'INTEGER')])

        self.assertEqual(result['attributes'], [('id', 'INTEGER')])
        self.assertEqual([row[0] for row in result['data']], [1, 2])
        self.assertEqual(result['conflicts'][0]['field'], 'score')
        self.assertEqual(result['conflicts'][0]['resolved'], 'omitted')

    def test_contract_field_absent_everywhere_is_a_conflict(self):
        left = HEADER_A + '1,red,10\n'

        result = arff_merge.merge_partitions(
            left, contract=[('id', 'INTEGER'), ('missing', 'REAL')])

        self.assertEqual(result['attributes'], [('id', 'INTEGER')])
        self.assertEqual(result['data'], [[1]])
        self.assertEqual(result['conflicts'][0]['field'], 'missing')
        self.assertEqual(result['conflicts'][0]['kind'], 'missing')
        self.assertEqual(result['conflicts'][0]['resolved'], 'omitted')

    def test_contract_lists_uncontracted_fields_as_ignored(self):
        left = HEADER_A + '1,red,10\n'

        result = arff_merge.merge_partitions(
            left, contract=[('id', 'INTEGER')])

        self.assertEqual(
            [(entry['attribute'], entry['column'])
             for entry in result['ignored_attributes']],
            [('color', 1), ('score', 2)])

    def test_accepts_decoded_and_sparse_list_of_dicts(self):
        left = arff.loads(HEADER_A + '1,red,10\n')
        right = {
            'relation': 'right',
            'attributes': [('score', 'REAL'), ('color', ['blue', 'red']),
                           ('id', 'INTEGER')],
            'data': [{0: 3.0, 1: 'red'}, {2: 4}],
        }

        result = arff_merge.merge_partitions(left, right)

        self.assertEqual(result['data'], [[1, 'red', 10.0],
                                          [0, 'red', 3.0],
                                          [4, 'blue', 0.0]])

    def test_accepts_a_single_list_of_partitions(self):
        left = HEADER_A + '1,red,10\n'
        right = HEADER_B + '20,blue,2\n'

        result = arff_merge.merge_partitions([left, right])

        self.assertEqual(result['data'], [[1, 'red', 10.0],
                                          [2, 'blue', 20.0]])

    def test_string_sparse_default_is_zero_label(self):
        left = '''@RELATION a
@ATTRIBUTE note STRING
@ATTRIBUTE value REAL
@DATA
{1 2.5}
'''

        result = arff_merge.merge_partitions(left)

        self.assertEqual(result['data'], [['0', 2.5]])

    def test_nominal_default_follows_source_declaration_order(self):
        left = '''@RELATION a
@ATTRIBUTE color {green, blue}
@ATTRIBUTE value REAL
@DATA
{1 1.0}
'''

        result = arff_merge.merge_partitions(left)

        self.assertEqual(result['data'], [['green', 1.0]])
        self.assertEqual(result['provenance'][0],
                         [arff_merge.SourceRef(0, 0),
                          arff_merge.SourceRef(0, 1)])
        self.assertFalse(result['sourced_data'][0][0]['explicit'])

    def test_module_entry_points_are_shared(self):
        left = HEADER_A + '1,red,10\n'
        right = HEADER_B + '20,blue,2\n'

        result = arff.merge_partitions(left, right)

        self.assertEqual(result['data'], [[1, 'red', 10.0],
                                          [2, 'blue', 20.0]])

    def test_header_only_partition_contributes_schema_not_rows(self):
        left = HEADER_A + '1,red,10\n'
        right = HEADER_B

        result = arff_merge.merge_partitions(left, right)

        self.assertEqual(len(result['data']), 1)
        self.assertEqual(result['data'][0], [1, 'red', 10.0])
        self.assertEqual(
            [(entry['partition'], entry['column'])
             for entry in result['field_sources']['score']],
            [(0, 2), (1, 0)])

    def test_unknown_option_is_rejected(self):
        with self.assertRaises(TypeError):
            arff_merge.merge_partitions(HEADER_A + '1,red,1\n',
                                        unknown_option=True)


if __name__ == '__main__':
    unittest.main()
