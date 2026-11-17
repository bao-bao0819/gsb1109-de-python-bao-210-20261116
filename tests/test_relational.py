import textwrap
import unittest

import arff


def _dedent(text):
    return textwrap.dedent(text).lstrip('\n')


RELATIONAL_HEADER = _dedent('''\
    @RELATION mi

    @ATTRIBUTE id INTEGER
    @ATTRIBUTE bag RELATIONAL
      @ATTRIBUTE x REAL
      @ATTRIBUTE s STRING
    @END
    @ATTRIBUTE label {yes, no}
    ''')

# The ``\\n`` inside the quoted cell is the on-disk escape separating the
# sub-rows of a RELATIONAL value.
DENSE_DATA = _dedent('''\
    @DATA
    1,"1.0,a\\n2.0,b",yes
    2,?,no
    3,"",yes
    ''')

DENSE_ARFF = RELATIONAL_HEADER + '\n' + DENSE_DATA

DENSE_OBJ = {
    'relation': 'mi',
    'attributes': [
        ('id', 'INTEGER'),
        ('bag', ('RELATIONAL', (
            ('x', 'REAL'),
            ('s', 'STRING'),
        ))),
        ('label', ['yes', 'no']),
    ],
    'data': [
        [1, [[1.0, 'a'], [2.0, 'b']], 'yes'],
        [2, None, 'no'],
        [3, [], 'yes'],
    ],
}

DENSE_DUMPED = _dedent('''\
    @RELATION mi

    @ATTRIBUTE id INTEGER
    @ATTRIBUTE bag RELATIONAL
    @ATTRIBUTE x REAL
    @ATTRIBUTE s STRING
    @END
    @ATTRIBUTE label {yes, no}

    @DATA
    1,"1.0,a\\n2.0,b",yes
    2,?,no
    3,"",yes
    ''')


class TestRelationalHeader(unittest.TestCase):
    def test_decode_relational_attribute(self):
        obj = arff.loads(DENSE_ARFF)
        self.assertEqual(
            obj['attributes'][1],
            ('bag', ('RELATIONAL', (('x', 'REAL'), ('s', 'STRING')))))

    def test_decode_is_case_insensitive_and_tolerates_comments(self):
        src = _dedent('''\
            @RELATION t
            @ATTRIBUTE "my bag" relational
              % a comment inside the block
              @ATTRIBUTE n INTEGER
            @end
            @DATA
            ?
            ''')
        obj = arff.loads(src)
        self.assertEqual(
            obj['attributes'][0],
            ('my bag', ('RELATIONAL', (('n', 'INTEGER'),))))

    def test_decode_nested_relational(self):
        src = _dedent('''\
            @RELATION nested
            @ATTRIBUTE outer RELATIONAL
              @ATTRIBUTE o REAL
              @ATTRIBUTE inner RELATIONAL
                @ATTRIBUTE z INTEGER
              @END
            @END
            @DATA
            ?
            ''')
        obj = arff.loads(src)
        self.assertEqual(obj['attributes'][0][1], (
            'RELATIONAL',
            (
                ('o', 'REAL'),
                ('inner', ('RELATIONAL', (('z', 'INTEGER'),))),
            )))

    def test_single_line_decode_rejects_relational(self):
        decoder = arff.ArffDecoder()
        self.assertRaises(
            arff.BadAttributeType,
            decoder._decode_attribute,
            '@ATTRIBUTE bag RELATIONAL')

    def test_missing_end(self):
        src = _dedent('''\
            @RELATION t
            @ATTRIBUTE bag RELATIONAL
              @ATTRIBUTE x REAL

            @DATA
            1
            ''')
        self.assertRaises(arff.BadRelationalFormat, arff.loads, src)

    def test_unexpected_line_in_block(self):
        src = _dedent('''\
            @RELATION t
            @ATTRIBUTE bag RELATIONAL
              @ATTRIBUTE x REAL
              @nonsense
            @END
            @DATA
            1
            ''')
        self.assertRaises(arff.BadRelationalFormat, arff.loads, src)

    def test_duplicate_sub_attribute(self):
        src = _dedent('''\
            @RELATION t
            @ATTRIBUTE bag RELATIONAL
              @ATTRIBUTE x REAL
              @ATTRIBUTE x INTEGER
            @END
            @DATA
            1
            ''')
        self.assertRaises(arff.BadAttributeName, arff.loads, src)

    def test_bad_sub_attribute_type(self):
        src = _dedent('''\
            @RELATION t
            @ATTRIBUTE bag RELATIONAL
              @ATTRIBUTE x BOGUS
            @END
            @DATA
            1
            ''')
        self.assertRaises(arff.BadAttributeType, arff.loads, src)


class TestRelationalDecode(unittest.TestCase):
    def test_dense_decode(self):
        obj = arff.loads(DENSE_ARFF)
        self.assertEqual(obj['data'], DENSE_OBJ['data'])

    def test_missing_vs_empty_subtable(self):
        obj = arff.loads(DENSE_ARFF)
        # An unquoted '?' is an explicit missing value.
        self.assertIsNone(obj['data'][1][1])
        # A quoted empty string is an empty sub-table instead.
        self.assertEqual(obj['data'][2][1], [])

    def test_sub_row_types_are_converted(self):
        src = _dedent('''\
            @RELATION t
            @ATTRIBUTE bag RELATIONAL
              @ATTRIBUTE i INTEGER
              @ATTRIBUTE r REAL
              @ATTRIBUTE s STRING
            @END
            @DATA
            "1,2.0,text"
            ''')
        sub, = arff.loads(src)['data'][0]
        self.assertEqual(sub, [[1, 2.0, 'text']])
        self.assertIsInstance(sub[0][0], int)
        self.assertIsInstance(sub[0][1], float)
        self.assertIsInstance(sub[0][2], str)

    def test_sparse_outer_relational_cell(self):
        src = _dedent('''\
            @RELATION t
            @ATTRIBUTE id INTEGER
            @ATTRIBUTE bag RELATIONAL
              @ATTRIBUTE x REAL
            @END
            @DATA
            {0 1,1 "1.0\\n2.0"}
            {0 2}
            {1 "3.0"}
            ''')
        obj = arff.loads(src, return_type=arff.LOD)
        self.assertEqual(obj['data'], [
            {0: 1, 1: [[1.0], [2.0]]},
            {0: 2},
            {1: [[3.0]]},
        ])

    def test_coo_is_not_supported(self):
        self.assertRaises(
            arff.BadRelationalFormat,
            arff.loads, DENSE_ARFF, return_type=arff.COO)

    def test_subrow_separator_distinct_from_escaped_newline(self):
        # The embedded newline lives inside a sub-row string value and must
        # not be confused with the sub-row separator.
        obj = {
            'relation': 't',
            'attributes': [
                ('id', 'INTEGER'),
                ('bag', ('RELATIONAL', (('s', 'STRING'),))),
            ],
            'data': [
                [1, [['line1\nline2'], ['second row']]],
            ],
        }
        encoded = arff.dumps(obj)
        # The cell contains both the escaped value newline and the real
        # separator newline.
        cell = encoded.split('@DATA\n', 1)[1].strip()
        self.assertIn('\\n', cell)
        self.assertEqual(arff.loads(encoded)['data'], obj['data'])


class TestRelationalErrors(unittest.TestCase):
    def _arff(self, bad_cell):
        return _dedent('''\
            @RELATION t
            @ATTRIBUTE id INTEGER
            @ATTRIBUTE bag RELATIONAL
              @ATTRIBUTE x REAL
              @ATTRIBUTE y REAL
            @END
            @DATA
            1,"1.0,2.0\\n3.0,4.0"
            2,"%s"
            3,"5.0,6.0"
            ''' % bad_cell)

    def test_arity_error_reports_parent_subrow_and_path(self):
        with self.assertRaises(arff.BadDataFormat) as ctx:
            arff.loads(self._arff('1.0,2.0,9.9'))
        exc = ctx.exception
        self.assertEqual(exc.row, 2)
        self.assertEqual(exc.sub_row, 1)
        self.assertEqual(exc.attribute_path, 'bag')

    def test_numeric_error_reports_deep_attribute_path(self):
        with self.assertRaises(arff.BadNumericalValue) as ctx:
            arff.loads(self._arff('oops,2.0'))
        exc = ctx.exception
        self.assertEqual(exc.row, 2)
        self.assertEqual(exc.sub_row, 1)
        self.assertEqual(exc.attribute_path, 'bag.x')

    def test_corrupt_subtable_does_not_swallow_next_record(self):
        good = _dedent('''\
            @RELATION t
            @ATTRIBUTE id INTEGER
            @ATTRIBUTE bag RELATIONAL
              @ATTRIBUTE x REAL
            @END
            @DATA
            1,"1.0\\n2.0"
            3,"4.0"
            ''')
        self.assertEqual(
            arff.loads(good)['data'],
            [[1, [[1.0], [2.0]]], [3, [[4.0]]]])

    def test_generator_isolates_failure_to_its_record(self):
        generator = arff.loads(
            self._arff('oops,2.0'),
            return_type=arff.DENSE_GEN)['data']
        self.assertEqual(next(generator)[0], 1)
        with self.assertRaises(arff.BadNumericalValue) as ctx:
            next(generator)
        self.assertEqual(ctx.exception.row, 2)
        self.assertEqual(ctx.exception.sub_row, 1)


class TestRelationalRoundTrip(unittest.TestCase):
    def test_encode_matches_canonical_text(self):
        self.assertEqual(arff.dumps(DENSE_OBJ), DENSE_DUMPED)

    def test_round_trip_preserves_levels_and_types(self):
        encoded = arff.dumps(DENSE_OBJ)
        decoded = arff.loads(encoded)
        self.assertEqual(decoded['attributes'], DENSE_OBJ['attributes'])
        self.assertEqual(decoded['data'], DENSE_OBJ['data'])

    def test_repeated_round_trip_is_stable(self):
        text = arff.dumps(DENSE_OBJ)
        for _ in range(10):
            text = arff.dumps(arff.loads(text))
        self.assertEqual(text, DENSE_DUMPED)

    def test_nested_round_trip(self):
        obj = {
            'relation': 'nested',
            'attributes': [
                ('id', 'INTEGER'),
                ('outer', ('RELATIONAL', (
                    ('o', 'REAL'),
                    ('inner', ('RELATIONAL', (('z', 'INTEGER'),))),
                ))),
            ],
            'data': [
                [1, [[1.0, [[1], [2]]], [2.0, []]]],
                [2, None],
            ],
        }
        encoded = arff.dumps(obj)
        decoded = arff.loads(encoded)
        self.assertEqual(decoded['attributes'], obj['attributes'])
        self.assertEqual(decoded['data'], obj['data'])
        self.assertEqual(arff.dumps(decoded), encoded)

    def test_sparse_round_trip(self):
        src = _dedent('''\
            @RELATION t
            @ATTRIBUTE id INTEGER
            @ATTRIBUTE bag RELATIONAL
              @ATTRIBUTE x REAL
            @END
            @DATA
            {0 1,1 "1.0\\n2.0"}
            {0 2}
            {1 "3.0"}
            ''')
        obj = arff.loads(src, return_type=arff.LOD)
        self.assertEqual(
            arff.loads(arff.dumps(obj), return_type=arff.LOD), obj)


class TestRelationalEncodingValidation(unittest.TestCase):
    def _encoder(self):
        return arff.ArffEncoder()

    def test_relational_value_must_be_list_or_none(self):
        obj = {
            'relation': 't',
            'attributes': [
                ('bag', ('RELATIONAL', (('x', 'REAL'),))),
            ],
            'data': [['not-a-list']],
        }
        self.assertRaises(arff.BadObject, self._encoder().encode, obj)

    def test_relational_type_requires_sub_attributes(self):
        obj = {
            'relation': 't',
            'attributes': [('bag', ('RELATIONAL', ()))],
            'data': [[]],
        }
        self.assertRaises(arff.BadObject, self._encoder().encode, obj)


if __name__ == '__main__':
    unittest.main()
