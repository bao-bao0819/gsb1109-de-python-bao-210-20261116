import unittest

import arff


WEATHER = '''% Weather data
% Collected for testing
@RELATION weather

@ATTRIBUTE outlook {sunny, overcast, rainy}
@ATTRIBUTE temperature REAL
@ATTRIBUTE humidity REAL
@ATTRIBUTE windy {TRUE, FALSE}
@ATTRIBUTE play {yes, no}

@DATA
sunny,85.0,85.0,FALSE,no
sunny,80.0,90.0,TRUE,no
overcast,83.0,86.0,FALSE,yes
rainy,70.0,96.0,FALSE,yes
'''


def load(text, **kwargs):
    return arff.loads(text, **kwargs)


class TestMetadataVersusData(unittest.TestCase):
    def test_relation_and_comment_changes_do_not_require_reimport(self):
        previous = load(WEATHER)
        current = load(WEATHER
                       .replace('% Weather data', '% Renamed dataset')
                       .replace('% Collected for testing', '% New comment')
                       .replace('@RELATION weather', '@RELATION weather_v2'))

        self.assertTrue(arff.data_equivalent(previous, current))
        self.assertFalse(arff.needs_reimport(previous, current))

        result = arff.compare_imports(previous, current)
        self.assertFalse(result.needs_reimport)
        self.assertTrue(result.data_equivalent)
        self.assertTrue(result.metadata_changed)
        self.assertTrue(result.relation_changed)
        self.assertTrue(result.description_changed)

    def test_attribute_name_change_is_metadata_only(self):
        previous = load(WEATHER)
        current = load(WEATHER.replace('@ATTRIBUTE humidity REAL',
                                       '@ATTRIBUTE relative_humidity REAL'))

        self.assertTrue(arff.data_equivalent(previous, current))
        self.assertTrue(arff.compare_imports(previous, current).metadata_changed)
        self.assertFalse(arff.compare_imports(previous, current).needs_reimport)

    def test_exchanging_unrelated_comments_keeps_data_version(self):
        previous = load(WEATHER)
        current = load(WEATHER
                       .replace('% Weather data', '% Collected for testing')
                       .replace('% Collected for testing', '% Weather data'))

        self.assertTrue(arff.data_equivalent(previous, current))
        self.assertEqual(arff.data_version(previous), arff.data_version(current))
        self.assertFalse(arff.compare_imports(previous, current).needs_reimport)

    def test_comments_are_ignored_for_reimport_but_versioned(self):
        previous = load(WEATHER)
        current = load(WEATHER.replace('% Weather data', '% Other comment'))

        self.assertTrue(arff.data_equivalent(previous, current))
        self.assertFalse(arff.needs_reimport(previous, current))
        self.assertNotEqual(arff.metadata_version(previous),
                            arff.metadata_version(current))

    def test_comments_can_be_excluded_from_metadata_version(self):
        previous = load(WEATHER)
        current = load(WEATHER.replace('% Weather data', '% Other comment'))

        self.assertEqual(
            arff.metadata_version(previous, include_comments=False),
            arff.metadata_version(current, include_comments=False))

    def test_descriptions_are_not_data(self):
        previous = load(WEATHER)
        current = load(WEATHER.replace('% Weather data', '% Changed comment'))
        self.assertEqual(arff.data_version(previous), arff.data_version(current))


class TestNominalComparison(unittest.TestCase):
    def test_declaration_order_is_metadata_for_symbolic_values(self):
        previous = load(WEATHER)
        current = load(WEATHER.replace(
            '@ATTRIBUTE outlook {sunny, overcast, rainy}',
            '@ATTRIBUTE outlook {rainy, sunny, overcast}'))

        self.assertTrue(arff.data_equivalent(previous, current))
        self.assertNotEqual(arff.metadata_version(previous),
                            arff.metadata_version(current))
        self.assertFalse(arff.compare_imports(previous, current).needs_reimport)

    def test_declaration_order_is_not_metadata_when_labels_are_encoded(self):
        previous = {
            'relation': 'r',
            'attributes': [('outlook', ['sunny', 'overcast', 'rainy'])],
            'data': [[0]],
        }
        reordered = {
            'relation': 'r',
            'attributes': [('outlook', ['rainy', 'sunny', 'overcast'])],
            'data': [[0]],
        }

        # Position 0 means "sunny" before and "rainy" after the reorder.
        self.assertFalse(arff.data_equivalent(previous, reordered))
        self.assertTrue(arff.needs_reimport(previous, reordered))

    def test_renamed_label_is_a_data_change_for_symbolic_values(self):
        previous = load(WEATHER)
        current = load(WEATHER
                       .replace('@ATTRIBUTE outlook {sunny, overcast, rainy}',
                                '@ATTRIBUTE outlook {bright, overcast, rainy}')
                       .replace('sunny,', 'bright,'))

        self.assertFalse(arff.data_equivalent(previous, current))
        self.assertTrue(arff.compare_imports(previous, current).needs_reimport)

    def test_renamed_label_is_detected_even_when_encoded_positions_match(self):
        previous = load(WEATHER, encode_nominal=True)
        current = load(WEATHER
                       .replace('@ATTRIBUTE outlook {sunny, overcast, rainy}',
                                '@ATTRIBUTE outlook {bright, overcast, rainy}')
                       .replace('sunny,', 'bright,'),
                       encode_nominal=True)

        # Both files use positions 0/1/2 for the same rows, but position 0 no
        # longer denotes the same category.
        self.assertEqual([row[0] for row in previous['data']],
                         [row[0] for row in current['data']])
        self.assertFalse(arff.data_equivalent(previous, current))
        self.assertTrue(arff.needs_reimport(previous, current))

    def test_label_reorder_with_matching_encoded_values_is_not_reimport(self):
        previous = {
            'relation': 'r',
            'attributes': [('outlook', ['sunny', 'overcast', 'rainy'])],
            # 0 = sunny, 1 = overcast, 2 = rainy
            'data': [[0], [0], [1], [2]],
        }
        reordered = {
            'relation': 'r',
            'attributes': [('outlook', ['rainy', 'sunny', 'overcast'])],
            # Same rows, but positions now resolve to the same labels.
            'data': [[1], [1], [2], [0]],
        }
        self.assertTrue(arff.data_equivalent(previous, reordered))
        self.assertNotEqual(arff.metadata_version(previous),
                            arff.metadata_version(reordered))
        self.assertFalse(arff.needs_reimport(previous, reordered))


class TestMissingAndRowOrder(unittest.TestCase):
    def test_all_missing_spellings_are_equivalent(self):
        attributes = [('a', 'REAL'), ('b', 'REAL')]
        left = {'relation': 'r', 'attributes': attributes,
                'data': [[None, 1.0], [2.0, None]]}
        right = {'relation': 'r', 'attributes': attributes,
                 'data': [['?', 1.0], [2.0, float('nan')]]}
        empty = {'relation': 'r', 'attributes': attributes,
                 'data': [['', 1.0], [2.0, '?']]}

        self.assertTrue(arff.data_equivalent(left, right))
        self.assertTrue(arff.data_equivalent(left, empty))
        self.assertEqual(arff.data_version(left), arff.data_version(right))

    def test_missing_is_distinct_from_present_zero(self):
        left = {'relation': 'r', 'attributes': [('a', 'REAL')],
                'data': [[None]]}
        right = {'relation': 'r', 'attributes': [('a', 'REAL')],
                 'data': [[0.0]]}
        self.assertFalse(arff.data_equivalent(left, right))

    def test_quoted_question_mark_string_is_not_missing(self):
        left = {'relation': 'r', 'attributes': [('a', 'STRING')],
                'data': [['?']]}
        right = {'relation': 'r', 'attributes': [('a', 'STRING')],
                 'data': [[None]]}
        self.assertFalse(arff.data_equivalent(left, right))

        decoded = arff.loads('''@RELATION r
@ATTRIBUTE a STRING
@DATA
'?'
?
''')
        self.assertEqual(decoded['data'], [['?'], [None]])

    def test_sparse_omitted_values_equal_explicit_defaults(self):
        dense = load('''@RELATION r
@ATTRIBUTE a {x, y}
@ATTRIBUTE b REAL
@DATA
x,0
y,2
''')
        sparse = arff.loads('''@RELATION r
@ATTRIBUTE a {x, y}
@ATTRIBUTE b REAL
@DATA
{}
{0 y, 1 2}
''', return_type=arff.LOD)
        self.assertTrue(arff.data_equivalent(dense, sparse))
        self.assertTrue(arff.data_equivalent(sparse, dense))

    def test_sparse_string_defaults_agree_across_return_types(self):
        sparse_text = '''@RELATION r
@ATTRIBUTE a STRING
@ATTRIBUTE b REAL
@DATA
{}
{0 x, 1 2}
'''
        dense = load(sparse_text)
        lod = arff.loads(sparse_text, return_type=arff.LOD)
        # Omitted STRING slots decode to '0' in both representations.
        self.assertEqual(dense['data'], [['0', 0.0], ['x', 2.0]])
        self.assertEqual(lod['data'], [{}, {0: 'x', 1: 2.0}])
        self.assertTrue(arff.data_equivalent(dense, lod))

    def test_shape_mismatch_is_not_equivalent(self):
        full = {'relation': 'r', 'attributes': [('a', 'REAL'), ('b', 'REAL')],
                'data': [[1.0, 2.0]]}
        short = {'relation': 'r', 'attributes': [('a', 'REAL'), ('b', 'REAL')],
                 'data': [[1.0]]}
        wide = {'relation': 'r', 'attributes': [('a', 'REAL'), ('b', 'REAL')],
                'data': [[1.0, 2.0, 3.0]]}
        self.assertFalse(arff.data_equivalent(full, short))
        self.assertFalse(arff.data_equivalent(full, wide))

    def test_row_order_is_ignored_by_default(self):
        previous = load(WEATHER)
        current = load(WEATHER)
        current['data'] = list(reversed(current['data']))

        self.assertTrue(arff.data_equivalent(previous, current))
        self.assertEqual(arff.data_version(previous), arff.data_version(current))
        self.assertFalse(arff.needs_reimport(previous, current))

    def test_row_order_can_be_compared_strictly(self):
        previous = load(WEATHER)
        current = load(WEATHER)
        current['data'] = list(reversed(current['data']))

        self.assertFalse(arff.data_equivalent(previous, current,
                                              row_order='ordered'))
        self.assertFalse(arff.data_equivalent(previous, current, ordered=True))
        self.assertNotEqual(arff.data_version(previous, row_order='ordered'),
                            arff.data_version(current, row_order='ordered'))

    def test_duplicate_rows_are_counted(self):
        previous = {
            'relation': 'r',
            'attributes': [('a', 'REAL')],
            'data': [[1.0], [1.0]],
        }
        current = {
            'relation': 'r',
            'attributes': [('a', 'REAL')],
            'data': [[1.0]],
        }
        self.assertFalse(arff.data_equivalent(previous, current))
        self.assertTrue(arff.needs_reimport(previous, current))

    def test_progressive_data_must_be_materialised(self):
        previous = arff.loads('''@RELATION r
@ATTRIBUTE x REAL
@DATA
1
2
''', return_type=arff.DENSE_GEN)
        current = arff.loads('''@RELATION r
@ATTRIBUTE x REAL
@DATA
1
2
''', return_type=arff.DENSE_GEN)

        with self.assertRaises(ValueError):
            arff.data_equivalent(previous, current)

        previous['data'] = list(previous['data'])
        current['data'] = list(current['data'])
        self.assertTrue(arff.data_equivalent(previous, current))

    def test_invalid_row_order_is_rejected(self):
        previous = load(WEATHER)
        with self.assertRaises(ValueError):
            arff.data_equivalent(previous, previous, row_order='sometimes')


class TestImportSignature(unittest.TestCase):
    def test_signature_separates_versions(self):
        previous = load(WEATHER)
        current = load(WEATHER.replace('@RELATION weather',
                                       '@RELATION weather_renamed'))

        old = arff.import_signature(previous)
        new = arff.import_signature(current)
        self.assertEqual(old.data_version, new.data_version)
        self.assertNotEqual(old.metadata_version, new.metadata_version)

    def test_signature_values_round_trip_through_json_values(self):
        signature = arff.import_signature(load(WEATHER))
        self.assertEqual(signature['data_version'], signature.data_version)
        self.assertEqual(len(signature.data_version), 64)
        self.assertEqual(len(signature.metadata_version), 64)
