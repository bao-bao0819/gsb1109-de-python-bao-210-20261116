import io
import json
import os
import tempfile
import unittest

import arff


HEADER = '''@RELATION weather

@ATTRIBUTE outlook {sunny, overcast}
@ATTRIBUTE temperature REAL
@ATTRIBUTE play {yes, no}

@DATA
'''

SHARD_ONE = HEADER + '''sunny,85.0,yes
rainy,70.0,no
'''

SHARD_TWO = '''@RELATION weather

@ATTRIBUTE outlook {sunny, overcast, rainy, snowy}
@ATTRIBUTE temperature REAL
@ATTRIBUTE play {yes, no, maybe}

@DATA
overcast,83.0,yes
snowy,20.0,maybe
'''

FULL = '''@RELATION weather

@ATTRIBUTE outlook {sunny, overcast, rainy, snowy}
@ATTRIBUTE temperature REAL
@ATTRIBUTE play {yes, no, maybe}

@DATA
sunny,85.0,yes
rainy,70.0,no
overcast,83.0,yes
snowy,20.0,maybe
'''


class TestShardAccumulator(unittest.TestCase):
    def make_accumulator(self, **kwargs):
        kwargs.setdefault('expected_shards', 2)
        return arff.ArffShardAccumulator(**kwargs)

    def test_incomplete_dataset_cannot_be_published(self):
        accumulator = self.make_accumulator()
        accumulator.add_shard(SHARD_ONE, shard_id='one')

        self.assertFalse(accumulator.complete)
        self.assertEqual(accumulator.accepted_rows, 2)
        with self.assertRaises(arff.IncompleteShards):
            accumulator.publish()
        with self.assertRaises(arff.IncompleteShards):
            accumulator.final_attributes()
        with self.assertRaises(arff.IncompleteShards):
            accumulator.finalize()

    def test_publish_refuses_to_write_partial_output(self):
        accumulator = self.make_accumulator()
        accumulator.add_shard(SHARD_ONE, shard_id='one')

        target = io.StringIO()
        with self.assertRaises(arff.IncompleteShards):
            accumulator.publish(target)
        self.assertEqual(target.getvalue(), '')

    def test_incomplete_publish_does_not_touch_target_file(self):
        accumulator = self.make_accumulator()
        accumulator.add_shard(SHARD_ONE, shard_id='one')

        directory = tempfile.mkdtemp()
        try:
            path = os.path.join(directory, 'data.arff')
            with open(path, 'w', encoding='utf-8') as stream:
                stream.write('previous\n')

            with self.assertRaises(arff.IncompleteShards):
                accumulator.publish(path)

            with open(path, 'r', encoding='utf-8') as stream:
                self.assertEqual(stream.read(), 'previous\n')
            self.assertEqual(os.listdir(directory), ['data.arff'])
        finally:
            for name in os.listdir(directory):
                os.unlink(os.path.join(directory, name))
            os.rmdir(directory)

    def test_completed_publish_roundtrips(self):
        accumulator = self.make_accumulator()
        accumulator.add_shard(SHARD_ONE, shard_id='one')
        accumulator.add_shard(SHARD_TWO, shard_id='two')

        target = io.StringIO()
        accumulator.publish(target)
        self.assertEqual(arff.loads(target.getvalue()),
                         arff.loads(FULL))


        accumulator = self.make_accumulator()
        accumulator.add_shard(SHARD_ONE, shard_id='one')

        self.assertEqual(accumulator.attributes,
                         [('outlook', ['sunny', 'overcast']),
                          ('temperature', 'REAL'),
                          ('play', ['yes', 'no'])])
        self.assertEqual(accumulator.pending_nominals,
                         {'outlook': ['rainy']})
        self.assertEqual(accumulator.accepted_rows, 2)

    def test_completed_dataset_matches_full_import(self):
        accumulator = self.make_accumulator()
        accumulator.add_shard(SHARD_ONE, shard_id='one')
        accumulator.add_shard(SHARD_TWO, shard_id='two')

        self.assertTrue(accumulator.complete)
        self.assertEqual(accumulator.finalize(), arff.loads(FULL))

    def test_encoded_nominals_match_full_import(self):
        accumulator = self.make_accumulator()
        accumulator.add_shard(SHARD_ONE, shard_id='one')
        accumulator.add_shard(SHARD_TWO, shard_id='two')

        result = accumulator.finalize(encode_nominal=True)
        expected = arff.loads(FULL, encode_nominal=True)
        self.assertEqual(result, expected)

    def test_all_return_types_match_full_import(self):
        sparse_one = ('@RELATION r\n@ATTRIBUTE a {x,y}\n@ATTRIBUTE b REAL\n'
                      '@DATA\n{0 x,1 1}\n')
        sparse_two = ('@RELATION r\n@ATTRIBUTE a {x,y,z}\n'
                      '@ATTRIBUTE b REAL\n@DATA\n{0 z,1 2}\n')
        full = ('@RELATION r\n@ATTRIBUTE a {x,y,z}\n@ATTRIBUTE b REAL\n'
                '@DATA\n{0 x,1 1}\n{0 z,1 2}\n')

        for return_type in (arff.DENSE, arff.LOD, arff.COO):
            accumulator = self.make_accumulator()
            accumulator.add_shard(sparse_one, shard_id='one')
            accumulator.add_shard(sparse_two, shard_id='two')
            self.assertEqual(
                accumulator.finalize(return_type=return_type),
                arff.loads(full, return_type=return_type),
                'return type %r' % return_type)

    def test_generator_return_type_is_lazy(self):
        accumulator = self.make_accumulator()
        accumulator.add_shard(SHARD_ONE, shard_id='one')
        accumulator.add_shard(SHARD_TWO, shard_id='two')

        result = accumulator.finalize(return_type=arff.DENSE_GEN)
        self.assertFalse(isinstance(result['data'], list))
        self.assertEqual(list(result['data']),
                         arff.loads(FULL)['data'])

    def test_publish_matches_dumps_of_full_import(self):
        accumulator = self.make_accumulator()
        accumulator.add_shard(SHARD_ONE, shard_id='one')
        accumulator.add_shard(SHARD_TWO, shard_id='two')

        self.assertEqual(arff.dumps(accumulator.publish()), arff.dumps(
            arff.loads(FULL)))

    def test_redeclared_nominal_order_controls_interpretation(self):
        reordered = '''@RELATION weather

@ATTRIBUTE outlook {overcast, snowy, rainy, sunny}
@ATTRIBUTE temperature REAL
@ATTRIBUTE play {maybe, yes, no}

@DATA
overcast,83.0,yes
snowy,20.0,maybe
'''
        full = '''@RELATION weather

@ATTRIBUTE outlook {overcast, snowy, rainy, sunny}
@ATTRIBUTE temperature REAL
@ATTRIBUTE play {maybe, yes, no}

@DATA
sunny,85.0,yes
rainy,70.0,no
overcast,83.0,yes
snowy,20.0,maybe
'''

        accumulator = self.make_accumulator()
        accumulator.add_shard(SHARD_ONE, shard_id='one')
        accumulator.add_shard(reordered, shard_id='two')

        self.assertEqual(accumulator.finalize(encode_nominal=True),
                         arff.loads(full, encode_nominal=True))
        self.assertEqual(accumulator.final_attributes()[0][1],
                         ['overcast', 'snowy', 'rainy', 'sunny'])

    def test_headerless_shard_needs_explicit_domain_completion(self):
        accumulator = self.make_accumulator()
        accumulator.add_shard(SHARD_ONE, shard_id='one')
        accumulator.add_shard('overcast,83.0,yes\nsnowy,20.0,maybe\n',
                              shard_id='two')

        self.assertFalse(accumulator.complete)
        with self.assertRaises(arff.IncompleteNominalDomain):
            accumulator.publish()

        accumulator.declare_attributes([
            ('outlook', ['sunny', 'overcast', 'rainy', 'snowy']),
            ('temperature', 'REAL'),
            ('play', ['yes', 'no', 'maybe']),
        ])
        self.assertTrue(accumulator.complete)
        self.assertEqual(accumulator.finalize(), arff.loads(FULL))

    def test_declare_attributes_cannot_drop_existing_categories(self):
        accumulator = self.make_accumulator()
        accumulator.add_shard(SHARD_ONE, shard_id='one')
        accumulator.add_shard('overcast,83.0,yes\n', shard_id='two')

        with self.assertRaises(arff.ShardConflict):
            accumulator.declare_attributes([
                ('outlook', ['sunny']),
                ('temperature', 'REAL'),
                ('play', ['yes', 'no']),
            ])
        self.assertEqual(accumulator.pending_nominals,
                         {'outlook': ['rainy']})

    def test_declare_attributes_cannot_guess_pending_categories(self):
        accumulator = self.make_accumulator()
        accumulator.add_shard(SHARD_ONE, shard_id='one')
        accumulator.add_shard('overcast,83.0,yes\nsnowy,20.0,maybe\n',
                              shard_id='two')

        with self.assertRaises(arff.IncompleteNominalDomain) as raised:
            accumulator.declare_attributes([
                ('outlook', ['sunny', 'overcast', 'rainy', 'snowy']),
                ('temperature', 'REAL'),
                ('play', ['yes', 'no']),
            ])
        self.assertEqual(raised.exception.pending, {'play': ['maybe']})

    def test_shard_without_prior_header_is_rejected(self):
        accumulator = self.make_accumulator()
        with self.assertRaises(arff.BadLayout):
            accumulator.add_shard('sunny,85.0,yes\n')
        self.assertEqual(accumulator.accepted_rows, 0)

    def test_rejected_shard_is_atomic(self):
        accumulator = self.make_accumulator()
        accumulator.add_shard(SHARD_ONE, shard_id='one')
        state = accumulator.state()

        with self.assertRaises(arff.BadDataFormat):
            accumulator.add_shard(HEADER + 'sunny,85.0\n', shard_id='bad')

        self.assertEqual(accumulator.state(), state)

    def test_invalid_nominal_is_not_rejected_early(self):
        accumulator = self.make_accumulator()
        accumulator.add_shard(SHARD_ONE, shard_id='one')
        with self.assertRaises(arff.BadNumericalValue):
            accumulator.add_shard(
                HEADER + 'sunny,not-a-number,yes\n', shard_id='bad')
        self.assertEqual(accumulator.accepted_rows, 2)

    def test_conflicting_attribute_declaration_is_rejected(self):
        changed_type = '''@RELATION weather

@ATTRIBUTE outlook {sunny, overcast}
@ATTRIBUTE temperature STRING
@ATTRIBUTE play {yes, no}

@DATA
overcast,83.0,yes
'''
        accumulator = self.make_accumulator()
        accumulator.add_shard(SHARD_ONE, shard_id='one')
        with self.assertRaises(arff.ShardConflict):
            accumulator.add_shard(changed_type, shard_id='two')
        self.assertEqual(accumulator.accepted_rows, 2)

    def test_shrinking_nominal_domain_is_rejected(self):
        shrunk = '''@RELATION weather

@ATTRIBUTE outlook {sunny}
@ATTRIBUTE temperature REAL
@ATTRIBUTE play {yes, no}

@DATA
sunny,85.0,yes
'''
        accumulator = self.make_accumulator()
        accumulator.add_shard(SHARD_ONE, shard_id='one')
        with self.assertRaises(arff.ShardConflict):
            accumulator.add_shard(shrunk, shard_id='two')

    def test_relation_conflict_is_rejected(self):
        other_relation = SHARD_TWO.replace('weather', 'other')
        accumulator = self.make_accumulator()
        accumulator.add_shard(SHARD_ONE, shard_id='one')
        with self.assertRaises(arff.ShardConflict):
            accumulator.add_shard(other_relation, shard_id='two')

    def test_replaying_a_shard_is_idempotent(self):
        accumulator = self.make_accumulator()
        self.assertTrue(accumulator.add_shard(SHARD_ONE, shard_id='one'))
        self.assertFalse(accumulator.add_shard(SHARD_ONE, shard_id='one'))
        self.assertEqual(accumulator.accepted_rows, 2)

    def test_replaying_a_shard_with_new_content_is_rejected(self):
        accumulator = self.make_accumulator()
        accumulator.add_shard(SHARD_ONE, shard_id='one')
        with self.assertRaises(arff.ShardConflict):
            accumulator.add_shard(SHARD_ONE + 'rainy,1.0,no\n',
                                  shard_id='one')

    def test_extra_shard_is_rejected(self):
        accumulator = self.make_accumulator()
        accumulator.add_shard(SHARD_ONE, shard_id='one')
        accumulator.add_shard(SHARD_TWO, shard_id='two')
        with self.assertRaises(arff.ShardConflict):
            accumulator.add_shard(HEADER, shard_id='three')
        self.assertEqual(accumulator.accepted_shards[-1]['id'], 'two')

    def test_open_ended_accumulator_requires_seal(self):
        accumulator = arff.ArffShardAccumulator()
        accumulator.add_shard(SHARD_ONE, shard_id='one')
        accumulator.add_shard(SHARD_TWO, shard_id='two')

        self.assertFalse(accumulator.complete)
        with self.assertRaises(arff.IncompleteShards):
            accumulator.publish()

        self.assertTrue(accumulator.seal())
        self.assertEqual(accumulator.finalize(), arff.loads(FULL))


class TestLateCategory(unittest.TestCase):
    '''The motivating case: a new category appears only on the last page.'''

    EARLY = '''@RELATION labels

@ATTRIBUTE sample INTEGER
@ATTRIBUTE label {alpha, beta}

@DATA
1,alpha
2,beta
'''

    LATE = '''@RELATION labels

@ATTRIBUTE sample INTEGER
@ATTRIBUTE label {alpha, beta, gamma}

@DATA
3,gamma
4,alpha
'''

    FULL = '''@RELATION labels

@ATTRIBUTE sample INTEGER
@ATTRIBUTE label {alpha, beta, gamma}

@DATA
1,alpha
2,beta
3,gamma
4,alpha
'''

    def test_new_category_in_last_shard_matches_full_import(self):
        accumulator = arff.ArffShardAccumulator(expected_shards=2)
        accumulator.add_shard(self.EARLY, shard_id='page-1')

        self.assertEqual(accumulator.attributes,
                         [('sample', 'INTEGER'),
                          ('label', ['alpha', 'beta'])])
        self.assertEqual(accumulator.pending_nominals, {})

        accumulator.add_shard(self.LATE, shard_id='page-2')
        self.assertTrue(accumulator.complete)
        self.assertEqual(accumulator.finalize(), arff.loads(self.FULL))
        self.assertEqual(accumulator.finalize(encode_nominal=True),
                         arff.loads(self.FULL, encode_nominal=True))

    def test_early_rows_are_reinterpreted_against_final_declaration(self):
        accumulator = arff.ArffShardAccumulator(expected_shards=2)
        accumulator.add_shard(self.EARLY, shard_id='page-1')
        accumulator.add_shard(self.LATE, shard_id='page-2')

        data = accumulator.finalize(encode_nominal=True)['data']
        self.assertEqual(data, [[1, 0], [2, 1], [3, 2], [4, 0]])

    def test_early_publish_cannot_freeze_the_old_domain(self):
        accumulator = arff.ArffShardAccumulator(expected_shards=2)
        accumulator.add_shard(self.EARLY, shard_id='page-1')

        with self.assertRaises(arff.IncompleteShards):
            accumulator.publish()
        self.assertEqual(len(accumulator.attributes[1][1]), 2)
        self.assertNotIn('gamma', accumulator.attributes[1][1])

    def test_late_category_without_header_blocks_publication(self):
        accumulator = arff.ArffShardAccumulator(expected_shards=2)
        accumulator.add_shard(self.EARLY, shard_id='page-1')
        accumulator.add_shard('3,gamma\n4,alpha\n', shard_id='page-2')

        self.assertFalse(accumulator.complete)
        with self.assertRaises(arff.IncompleteNominalDomain) as raised:
            accumulator.publish()
        self.assertEqual(raised.exception.pending, {'label': ['gamma']})

        accumulator.declare_attributes([
            ('sample', 'INTEGER'),
            ('label', ['alpha', 'beta', 'gamma']),
        ])
        self.assertEqual(accumulator.finalize(), arff.loads(self.FULL))


class TestShardCheckpoint(unittest.TestCase):
    def test_checkpoint_roundtrip_resumes_dataset(self):
        accumulator = arff.ArffShardAccumulator(expected_shards=2)
        accumulator.add_shard(SHARD_ONE, shard_id='one')

        checkpoint = io.StringIO()
        accumulator.checkpoint(checkpoint)
        checkpoint.seek(0)
        restored = arff.ArffShardAccumulator.from_checkpoint(checkpoint)

        self.assertFalse(restored.complete)
        self.assertEqual(restored.accepted_rows, 2)
        self.assertEqual(restored.pending_nominals,
                         {'outlook': ['rainy']})

        restored.add_shard(SHARD_TWO, shard_id='two')
        self.assertEqual(restored.finalize(), arff.loads(FULL))

    def test_resume_after_crash_replays_last_shard_safely(self):
        accumulator = arff.ArffShardAccumulator(expected_shards=2)
        accumulator.add_shard(SHARD_ONE, shard_id='one')

        checkpoint = io.StringIO()
        accumulator.checkpoint(checkpoint)
        checkpoint.seek(0)
        restored = arff.ArffShardAccumulator.from_checkpoint(checkpoint)

        restored.add_shard(SHARD_TWO, shard_id='two')
        # Simulate a crash before the selector recorded progress, then replay.
        self.assertFalse(restored.add_shard(SHARD_TWO, shard_id='two'))
        self.assertEqual(restored.accepted_rows, 4)
        self.assertEqual(restored.finalize(), arff.loads(FULL))

    def test_tampered_checkpoint_is_rejected(self):
        accumulator = arff.ArffShardAccumulator(expected_shards=2)
        accumulator.add_shard(SHARD_ONE, shard_id='one')

        state = accumulator.state()
        state['data'] = state['data'][:1]
        with self.assertRaises(ValueError):
            arff.ArffShardAccumulator.from_state(state)

        state = accumulator.state()
        state['version'] = 999
        with self.assertRaises(ValueError):
            arff.ArffShardAccumulator.from_state(state)

    def test_complete_checkpoint_survives_roundtrip(self):
        accumulator = arff.ArffShardAccumulator(expected_shards=2)
        accumulator.add_shard(SHARD_ONE, shard_id='one')
        accumulator.add_shard(SHARD_TWO, shard_id='two')

        restored = arff.ArffShardAccumulator.from_state(accumulator.state())
        self.assertTrue(restored.complete)
        self.assertEqual(restored.finalize(), arff.loads(FULL))

        accumulator = arff.ArffShardAccumulator(expected_shards=2)
        accumulator.add_shard(SHARD_ONE, shard_id='one')

        state = accumulator.state()
        self.assertEqual(state['status'], 'partial')
        self.assertFalse(state['complete'])
        self.assertNotIn('@DATA', json.dumps(state))

        with self.assertRaises(arff.BadLayout):
            arff.loads(json.dumps(state))

    def test_checkpoint_file_is_written_atomically(self):
        accumulator = arff.ArffShardAccumulator(expected_shards=2)
        accumulator.add_shard(SHARD_ONE, shard_id='one')

        directory = tempfile.mkdtemp()
        try:
            path = os.path.join(directory, 'state.json')
            accumulator.checkpoint(path)
            with open(path, 'r', encoding='utf-8') as stream:
                state = json.load(stream)
            self.assertEqual(state['status'], 'partial')
            self.assertEqual(os.listdir(directory), ['state.json'])
        finally:
            for name in os.listdir(directory):
                os.unlink(os.path.join(directory, name))
            os.rmdir(directory)

    def test_incomplete_checkpoint_cannot_be_mistaken_for_final(self):
        accumulator = arff.ArffShardAccumulator(expected_shards=2)
        accumulator.add_shard(SHARD_ONE, shard_id='one')

        restored = arff.ArffShardAccumulator.from_state(accumulator.state())
        with self.assertRaises(arff.IncompleteShards):
            restored.publish()

    def test_state_has_clear_data_pending_boundary(self):
        accumulator = arff.ArffShardAccumulator(expected_shards=2)
        accumulator.add_shard(SHARD_ONE, shard_id='one')

        state = accumulator.state()
        self.assertEqual(state['data'],
                         ['sunny,85.0,yes', 'rainy,70.0,no'])
        self.assertEqual(state['pending_nominals'],
                         {'outlook': ['rainy']})
        self.assertTrue(all('rainy' not in declaration
                            for _, declaration in state['attributes']))


class TestLoadShards(unittest.TestCase):
    def test_load_shards_matches_full_import(self):
        result = arff.load_shards([SHARD_ONE, SHARD_TWO],
                                  expected_shards=2)
        self.assertEqual(result, arff.loads(FULL))

    def test_load_shards_requires_all_shards(self):
        with self.assertRaises(arff.IncompleteShards):
            arff.load_shards([SHARD_ONE], expected_shards=2)
