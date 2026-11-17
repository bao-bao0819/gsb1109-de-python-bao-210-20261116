# -*- coding: utf-8 -*-
import math

import pytest

import arff
from jackknife_arff import jackknife_report, load_report


BASIC_ARFF = u'''@relation basic
@attribute g string
@attribute x numeric
@data
a,1
a,3
b,5
b,?
c,7
c,9
'''


def _report(text, **kwargs):
    dataset = arff.loads(text)
    return jackknife_report(dataset['data'], dataset['attributes'], **kwargs)


def test_weighted_mean_ignores_missing_and_uses_weight_denominator():
    report = _report(BASIC_ARFF, group='g')
    mean = report['means']['x']
    assert mean['estimate'] == pytest.approx(5.0)
    assert mean['weight_sum'] == pytest.approx(5.0)
    assert mean['n_obs'] == 5
    assert mean['source_rows'] == [0, 1, 2, 4, 5]


def test_undeclared_weights_count_as_one():
    report = _report(BASIC_ARFF, group='g')
    assert all(row['weight'] == 1.0 for row in report['rows'])
    assert all(not row['weight_declared'] for row in report['rows'])
    assert report['control_total'] == pytest.approx(6.0)


def test_jackknife_standard_error_matches_hand_computation():
    report = _report(BASIC_ARFF, group='g')
    # replicate means: drop a -> 7, drop b -> 5, drop c -> 3
    se = report['standard_errors']['x']
    assert se['estimate'] == pytest.approx(math.sqrt(16.0 / 3.0))
    assert se['replicates_used'] == 3
    assert report['failed_replicates'] == []


def test_replicates_recalibrate_weights_to_control_total():
    report = _report(BASIC_ARFF, group='g')
    factors = {rep['dropped_group']: rep['calibration_factor']
               for rep in report['replicates']}
    # control total 6; every group holds two unit-weight rows, so the
    # remaining weight sum is always 4
    assert factors['a'] == pytest.approx(1.5)
    assert factors['b'] == pytest.approx(1.5)
    assert factors['c'] == pytest.approx(1.5)
    means = {rep['dropped_group']: rep['columns']['x']['mean']
             for rep in report['replicates']}
    assert means == {'a': 7.0, 'b': 5.0, 'c': 3.0}


def test_groups_and_replicates_trace_back_to_source_rows():
    report = _report(BASIC_ARFF, group='g')
    groups = {g['value']: g['source_rows'] for g in report['groups']}
    assert groups == {'a': [0, 1], 'b': [2, 3], 'c': [4, 5]}
    for rep in report['replicates']:
        assert rep['dropped_source_rows'] == groups[rep['dropped_group']]


COMPLETE_ARFF = u'''@relation complete
@attribute g string
@attribute x numeric
@attribute y numeric
@data
a,1,2
b,3,4
c,5,6
'''


def test_weighted_covariance_pairwise_complete():
    report = _report(COMPLETE_ARFF, group='g')
    cov = report['covariance']
    # denominator V1 - V2/V1 = 3 - 1 = 2
    assert cov['x']['y']['estimate'] == pytest.approx(4.0)
    assert cov['x']['x']['estimate'] == pytest.approx(4.0)
    assert cov['x']['y']['denominator'] == pytest.approx(2.0)
    assert report['mean_covariance']['x']['y']['estimate'] == \
        pytest.approx(4.0 / 3.0)
    assert report['standard_errors']['x']['estimate'] == \
        pytest.approx(math.sqrt(4.0 / 3.0))


def test_missing_values_give_each_pair_its_own_denominator():
    text = u'''@relation miss
@attribute g string
@attribute x numeric
@attribute y numeric
@data
a,1,10
a,?,20
b,3,?
b,5,50
c,7,70
c,9,?
'''
    report = _report(text, group='g')
    assert report['means']['x']['n_obs'] == 5
    assert report['means']['y']['n_obs'] == 4
    pair = report['covariance']['x']['y']
    assert pair['n_obs'] == 3  # rows 0, 3 and 4 are complete


def test_unsolvable_calibration_is_listed_not_zero_error():
    text = u'''@relation fail
@attribute g string
@attribute w numeric
@attribute x numeric
@data
a,1,10
a,1,20
b,0,30
b,0,40
'''
    report = _report(text, group='g', weight='w')
    # dropping a leaves only zero-weight rows: no calibration solution
    assert len(report['failed_replicates']) == 1
    failed = report['failed_replicates'][0]
    assert failed['dropped_group'] == 'a'
    assert failed['dropped_source_rows'] == [0, 1]
    assert 'no solution' in failed['reason']
    # a single usable replicate must not become a fake zero error
    assert len(report['replicates']) == 1
    assert report['standard_errors']['x']['estimate'] is None
    assert report['standard_errors']['x']['replicates_used'] == 1


def test_missing_weight_value_counts_as_one():
    text = u'''@relation wmiss
@attribute g string
@attribute w numeric
@attribute x numeric
@data
a,2,10
b,?,20
c,3,30
'''
    report = _report(text, group='g', weight='w')
    weights = {row['index']: row['weight'] for row in report['rows']}
    assert weights == {0: 2.0, 1: 1.0, 2: 3.0}
    assert report['control_total'] == pytest.approx(6.0)
    mean = report['means']['x']
    assert mean['estimate'] == pytest.approx((20 + 20 + 90) / 6.0)


def test_group_attribute_must_be_complete():
    text = u'''@relation nogroup
@attribute g string
@attribute x numeric
@data
a,1
?,2
'''
    with pytest.raises(ValueError):
        _report(text, group='g')


def test_load_report_from_file(tmp_path):
    path = tmp_path / 'data.arff'
    path.write_text(BASIC_ARFF)
    report = load_report(str(path), group='g')
    assert report['source'] == str(path)
    assert report['means']['x']['estimate'] == pytest.approx(5.0)
