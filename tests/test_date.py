import datetime
import unittest

import arff


EPOCH = datetime.datetime(1970, 1, 1)


class DateAdapter:
    '''Minimal date adapter used by the tests.

    Translates Weka-style format strings to strptime/strftime ones and
    interprets numeric values as days since the epoch (numeric zero is the
    epoch itself).
    '''

    _FORMAT_MAP = [('yyyy', '%Y'), ('MM', '%m'), ('dd', '%d'),
                   ('HH', '%H'), ('mm', '%M'), ('ss', '%S')]

    def __init__(self, tz='UTC', version='1.0'):
        self.tz = tz
        self.version = version

    @classmethod
    def _to_strftime(cls, fmt):
        if not fmt:
            return '%Y-%m-%d'
        for weka, py in cls._FORMAT_MAP:
            fmt = fmt.replace(weka, py)
        return fmt

    def parse(self, value, fmt):
        if isinstance(value, (int, float)):
            return EPOCH + datetime.timedelta(days=value)
        return datetime.datetime.strptime(value, self._to_strftime(fmt))

    def format(self, value, fmt):
        return value.strftime(self._to_strftime(fmt))


ARFF = '''@RELATION dates

@ATTRIBUTE when DATE "yyyy-MM-dd"
@ATTRIBUTE value REAL

@DATA
"2020-01-02",1.0
?,2.0
{1 3.0}
'''

SPARSE_ARFF = '''@RELATION dates

@ATTRIBUTE when DATE yyyy-MM-dd
@ATTRIBUTE value REAL

@DATA
{0 "2020-01-02",1 1.0}
{0 ?,1 2.0}
{1 3.0}
'''


class TestDecodeDateAttribute(unittest.TestCase):
    def get_decoder(self):
        return arff.ArffDecoder()

    def test_bare_date(self):
        decoder = self.get_decoder()
        result = decoder._decode_attribute('@ATTRIBUTE when DATE')
        self.assertEqual(result, ('when', 'DATE'))

    def test_date_case_insensitive(self):
        decoder = self.get_decoder()
        result = decoder._decode_attribute('@ATTRIBUTE when DaTe')
        self.assertEqual(result, ('when', 'DATE'))

    def test_date_with_quoted_format(self):
        decoder = self.get_decoder()
        result = decoder._decode_attribute(
            '@ATTRIBUTE when DATE "yyyy-MM-dd HH:mm"')
        self.assertEqual(result, ('when', 'DATE yyyy-MM-dd HH:mm'))

    def test_date_with_unquoted_format(self):
        decoder = self.get_decoder()
        result = decoder._decode_attribute('@ATTRIBUTE when DATE yyyy-MM-dd')
        self.assertEqual(result, ('when', 'DATE yyyy-MM-dd'))

    def test_date_format_is_case_sensitive(self):
        decoder = self.get_decoder()
        result = decoder._decode_attribute('@ATTRIBUTE when DATE "YYYY-ww"')
        self.assertEqual(result, ('when', 'DATE YYYY-ww'))

    def test_date_with_unbalanced_quote(self):
        decoder = self.get_decoder()
        self.assertRaises(
            arff.BadAttributeType,
            decoder._decode_attribute,
            '@ATTRIBUTE when DATE "yyyy-MM-dd'
        )


class TestDecodeDateData(unittest.TestCase):
    def test_dense(self):
        obj = arff.loads(ARFF, date_adapter=DateAdapter())

        self.assertEqual(obj['attributes'],
                         [('when', 'DATE yyyy-MM-dd'), ('value', 'REAL')])

        # Explicit date: parsed by the adapter into a real date value
        self.assertIsInstance(obj['data'][0][0], datetime.datetime)
        self.assertEqual(obj['data'][0][0], datetime.datetime(2020, 1, 2))
        # Question mark: missing value
        self.assertIsNone(obj['data'][1][0])
        # Omitted sparse entry: epoch of the date type, not missing
        self.assertEqual(obj['data'][2][0], EPOCH)
        self.assertIsNotNone(obj['data'][2][0])

    def test_dense_gen(self):
        obj = arff.loads(ARFF, return_type=arff.DENSE_GEN,
                         date_adapter=DateAdapter())
        data = list(obj['data'])
        self.assertEqual(data[0][0], datetime.datetime(2020, 1, 2))
        self.assertIsNone(data[1][0])
        self.assertEqual(data[2][0], EPOCH)

    def test_lod(self):
        obj = arff.loads(SPARSE_ARFF, return_type=arff.LOD,
                         date_adapter=DateAdapter())

        # Explicit date
        self.assertEqual(obj['data'][0][0], datetime.datetime(2020, 1, 2))
        # Question mark: present but missing
        self.assertIn(0, obj['data'][1])
        self.assertIsNone(obj['data'][1][0])
        # Omitted sparse entry: absent from the row, i.e. implicit zero
        self.assertNotIn(0, obj['data'][2])

    def test_lod_gen(self):
        obj = arff.loads(SPARSE_ARFF, return_type=arff.LOD_GEN,
                         date_adapter=DateAdapter())
        data = list(obj['data'])
        self.assertEqual(data[0][0], datetime.datetime(2020, 1, 2))
        self.assertIsNone(data[1][0])
        self.assertNotIn(0, data[2])

    def test_coo(self):
        obj = arff.loads(SPARSE_ARFF, return_type=arff.COO,
                         date_adapter=DateAdapter())
        data, rows, cols = obj['data']
        self.assertEqual(
            data,
            [datetime.datetime(2020, 1, 2), 1.0, None, 2.0, 3.0])
        self.assertEqual(rows, [0, 0, 1, 1, 2])
        self.assertEqual(cols, [0, 1, 0, 1, 1])

    def test_explicit_zero_dense(self):
        fixture = ('@RELATION dates\n'
                   '@ATTRIBUTE when DATE yyyy-MM-dd\n'
                   '@DATA\n'
                   '{ }\n')
        obj = arff.loads(fixture, date_adapter=DateAdapter())
        # The omitted entry decodes as the epoch, not as a missing value
        self.assertEqual(obj['data'][0][0], EPOCH)

    def test_bad_date_value(self):
        fixture = ('@RELATION dates\n'
                   '@ATTRIBUTE when DATE yyyy-MM-dd\n'
                   '@DATA\n'
                   'not-a-date\n')
        self.assertRaises(arff.BadDateValue, arff.loads, fixture,
                          date_adapter=DateAdapter())

    def test_missing_adapter(self):
        self.assertRaises(arff.BadDateAdapter, arff.loads, ARFF)


class TestEncodeDateData(unittest.TestCase):
    def get_obj(self):
        return {
            'description': '',
            'relation': 'dates',
            'attributes': [('when', 'DATE yyyy-MM-dd'), ('value', 'REAL')],
            'data': [
                [datetime.datetime(2020, 1, 2), 1.0],
                [None, 2.0],
                [EPOCH, 3.0],
            ],
        }

    def test_dense(self):
        result = arff.dumps(self.get_obj(), date_adapter=DateAdapter())

        self.assertIn('@ATTRIBUTE when DATE yyyy-MM-dd', result)
        self.assertIn('2020-01-02,1.0', result)
        # Missing values are encoded as '?'
        self.assertIn('?,2.0', result)
        # The epoch is a regular date value, encoded as a date
        self.assertIn('1970-01-01,3.0', result)

    def test_format_with_spaces_is_quoted(self):
        obj = self.get_obj()
        obj['attributes'] = [('when', 'DATE yyyy-MM-dd HH:mm'),
                             ('value', 'REAL')]
        result = arff.dumps(obj, date_adapter=DateAdapter())
        self.assertIn("@ATTRIBUTE when DATE 'yyyy-MM-dd HH:mm'", result)

    def test_lod(self):
        obj = self.get_obj()
        obj['data'] = [
            {0: datetime.datetime(2020, 1, 2), 1: 1.0},
            {0: None, 1: 2.0},
            {1: 3.0},
        ]
        result = arff.dumps(obj, date_adapter=DateAdapter())

        self.assertIn('{ 0 2020-01-02,1 1.0 }', result)
        self.assertIn('{ 0 ?,1 2.0 }', result)
        # Omitted entries stay omitted: occurrence is preserved
        self.assertIn('{ 1 3.0 }', result)

    def test_coo(self):
        class COOStub:
            format = 'coo'

            def __init__(self, data, row, col):
                self.data = data
                self.row = row
                self.col = col

        obj = self.get_obj()
        obj['data'] = COOStub(
            [datetime.datetime(2020, 1, 2), 1.0, None, 2.0, 3.0],
            [0, 0, 1, 1, 2],
            [0, 1, 0, 1, 1],
        )
        result = arff.dumps(obj, date_adapter=DateAdapter())

        self.assertIn('{ 0 2020-01-02,1 1.0 }', result)
        self.assertIn('{ 0 ?,1 2.0 }', result)
        self.assertIn('{ 1 3.0 }', result)

    def test_encode_uses_adapter_format(self):
        class SpyAdapter(DateAdapter):
            def __init__(self):
                super().__init__()
                self.formatted = []

            def format(self, value, fmt):
                self.formatted.append((value, fmt))
                return super().format(value, fmt)

        adapter = SpyAdapter()
        arff.dumps(self.get_obj(), date_adapter=adapter)
        self.assertEqual(
            adapter.formatted,
            [(datetime.datetime(2020, 1, 2), 'yyyy-MM-dd'),
             (EPOCH, 'yyyy-MM-dd')])

    def test_missing_adapter(self):
        self.assertRaises(arff.BadObject, arff.dumps, self.get_obj())


class TestDateRoundTrip(unittest.TestCase):
    def test_dense_round_trip(self):
        adapter = DateAdapter()
        obj1 = arff.loads(ARFF, date_adapter=adapter)
        s = arff.dumps(obj1, date_adapter=adapter)
        obj2 = arff.loads(s, date_adapter=adapter)

        self.assertEqual(obj1['attributes'], obj2['attributes'])
        self.assertEqual(obj1['data'], obj2['data'])

    def test_lod_round_trip(self):
        adapter = DateAdapter()
        obj1 = arff.loads(SPARSE_ARFF, return_type=arff.LOD,
                          date_adapter=adapter)
        s = arff.dumps(obj1, date_adapter=adapter)
        obj2 = arff.loads(s, return_type=arff.LOD, date_adapter=adapter)

        self.assertEqual(obj1['attributes'], obj2['attributes'])
        # Dict equality reconciles values, missing state and occurrence
        self.assertEqual(obj1['data'], obj2['data'])

    def test_epoch_survives_dense_round_trip(self):
        adapter = DateAdapter()
        obj1 = arff.loads(ARFF, date_adapter=adapter)
        obj2 = arff.loads(arff.dumps(obj1, date_adapter=adapter),
                          date_adapter=adapter)
        self.assertEqual(obj2['data'][2][0], EPOCH)
        self.assertIsNone(obj2['data'][1][0])


class TestDateAdapterFreezing(unittest.TestCase):
    def test_adapter_change_during_read(self):
        adapter = DateAdapter()
        obj = arff.loads(ARFF, return_type=arff.DENSE_GEN,
                         date_adapter=adapter)
        # The adapter configuration changes in the middle of the task
        adapter.tz = 'America/Sao_Paulo'
        self.assertRaises(arff.BadDateAdapter, list, obj['data'])

    def test_adapter_change_during_write(self):
        adapter = DateAdapter()
        encoder = arff.ArffEncoder()
        obj = {
            'description': '',
            'relation': 'dates',
            'attributes': [('when', 'DATE yyyy-MM-dd')],
            'data': [[datetime.datetime(2020, 1, 2)],
                     [datetime.datetime(2020, 1, 3)]],
        }
        rows = encoder.iter_encode(obj, date_adapter=adapter)
        with self.assertRaises(arff.BadDateAdapter):
            for row in rows:
                if row == '2020-01-02':
                    # The adapter configuration changes in the middle of
                    # the task
                    adapter.version = '2.0'

    def test_adapter_change_between_tasks(self):
        adapter = DateAdapter()
        arff.loads(ARFF, date_adapter=adapter)
        # Changing the adapter between tasks is fine: each task freezes its
        # own configuration
        adapter.tz = 'America/Sao_Paulo'
        obj = arff.loads(ARFF, date_adapter=adapter)
        self.assertEqual(obj['data'][0][0], datetime.datetime(2020, 1, 2))

    def test_tasks_do_not_share_adapters(self):
        class EpochAdapter(DateAdapter):
            def __init__(self, epoch):
                super().__init__()
                self.epoch = epoch

            def parse(self, value, fmt):
                if isinstance(value, (int, float)):
                    return self.epoch
                return super().parse(value, fmt)

        decoder = arff.ArffDecoder()
        obj1 = decoder.decode(ARFF, date_adapter=EpochAdapter('epoch-1'))
        obj2 = decoder.decode(ARFF, date_adapter=EpochAdapter('epoch-2'))

        # Each decoding task used its own adapter for the omitted zero
        self.assertEqual(obj1['data'][2][0], 'epoch-1')
        self.assertEqual(obj2['data'][2][0], 'epoch-2')

    def test_decoder_reuse_resets_conversors(self):
        decoder = arff.ArffDecoder()
        decoder.decode(ARFF, date_adapter=DateAdapter())
        # A second, narrower decoding task must not see stale conversors
        fixture = ('@RELATION dates\n'
                   '@ATTRIBUTE when DATE yyyy-MM-dd\n'
                   '@DATA\n'
                   '2020-01-02\n')
        obj = decoder.decode(fixture, date_adapter=DateAdapter())
        self.assertEqual(obj['data'], [[datetime.datetime(2020, 1, 2)]])


if __name__ == '__main__':
    unittest.main()
