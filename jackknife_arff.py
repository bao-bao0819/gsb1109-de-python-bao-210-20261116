# =============================================================================
# Delete-one-group jackknife with calibrated row weights for ARFF data.
#
# Rows that belong to the same sampling unit (group) are *not* independent
# replicates, so row weights must not be read as replication counts.  This
# module reads an independent-group column and an optional row-weight column
# from an ARFF file and produces, for every numeric analysis column:
#
#   * the weighted mean (full sample),
#   * the jackknife standard error of that mean (delete-one-group),
#   * the weighted covariance matrix of the columns,
#   * the jackknife covariance matrix of the estimated means.
#
# Every jackknife replicate drops one group and *recalibrates* the remaining
# weights so they sum to a given control total (ratio calibration).  A
# replicate whose remaining weights cannot be calibrated (remaining weight
# sum is zero) has no solution: it is listed explicitly in
# ``failed_replicates`` and never silently treated as zero error.
#
# Missing values ('?' in the ARFF) give every column and every column pair
# its own denominator (pairwise-complete weights).  Rows without a declared
# weight are treated as weight 1.  Original weights, the groups actually
# used, and every replicate result carry the source row indices so each
# number can be traced back to the ARFF data rows.
# =============================================================================

import argparse
import json
import math
import sys

import arff

NUMERIC_TYPES = {'NUMERIC', 'REAL', 'INTEGER'}

DEFAULT_WEIGHT = 1.0


def _is_numeric_type(type_decl):
    return isinstance(type_decl, str) and type_decl.upper() in NUMERIC_TYPES


def _attribute_index(attributes, name, role):
    for index, (attr_name, _) in enumerate(attributes):
        if attr_name == name:
            return index
    raise ValueError('%s attribute %r not found among %r'
                     % (role, name, [a[0] for a in attributes]))


def _weighted_mean(pairs):
    """Weighted mean of (weight, value) pairs; None when no weight remains."""
    weight_sum = sum(w for w, _ in pairs)
    if weight_sum == 0:
        return None, 0.0
    return sum(w * x for w, x in pairs) / weight_sum, weight_sum


def _column_pairs(data, weights, rows, col):
    return [(weights[r], data[r][col]) for r in rows
            if data[r][col] is not None and weights[r] != 0]


def _pair_pairs(data, weights, rows, col_a, col_b):
    return [(weights[r], data[r][col_a], data[r][col_b]) for r in rows
            if data[r][col_a] is not None and data[r][col_b] is not None
            and weights[r] != 0]


def _weighted_covariance(triples):
    """Reliability-weight covariance of (weight, x, y) triples.

    Uses the pair-specific means and the unbiased reliability-weight
    denominator V1 - V2/V1.  Returns (estimate, denominator); the estimate
    is None when the denominator is not positive.
    """
    v1 = sum(w for w, _, _ in triples)
    v2 = sum(w * w for w, _, _ in triples)
    denominator = v1 - v2 / v1 if v1 else 0.0
    if denominator <= 0:
        return None, denominator
    mean_x = sum(w * x for w, x, _ in triples) / v1
    mean_y = sum(w * y for w, _, y in triples) / v1
    cov = sum(w * (x - mean_x) * (y - mean_y) for w, x, y in triples)
    return cov / denominator, denominator


def _jackknife_combine(replicate_values):
    """Delete-one jackknife variance of a list of replicate estimates.

    Returns (variance, mean_of_replicates) or (None, None) when fewer than
    two usable replicates remain -- a single replicate would fake a zero
    error, so the result is reported as undefined instead.
    """
    m = len(replicate_values)
    if m < 2:
        return None, None
    mean = sum(replicate_values) / m
    variance = (m - 1.0) / m * sum((v - mean) ** 2 for v in replicate_values)
    return variance, mean


def jackknife_report(data, attributes, group, weight=None, columns=None,
                     control_total=None, source=None):
    """Build the full jackknife report from decoded ARFF content.

    ``data`` is a sequence of rows (as in ``arff.load(...)['data']``) and
    ``attributes`` the matching ``(name, type)`` declarations.  ``group``
    names the independent sampling-unit attribute; ``weight`` names the row
    weight attribute (rows with missing/undeclared weight count as 1).
    ``columns`` restricts the analysis columns (default: every numeric
    attribute other than ``group``/``weight``).  ``control_total`` is the
    calibration target for the replicate weights (default: the full-sample
    weight sum).
    """
    group_idx = _attribute_index(attributes, group, 'group')
    weight_idx = (_attribute_index(attributes, weight, 'weight')
                  if weight is not None else None)

    if columns is None:
        columns = [name for i, (name, decl) in enumerate(attributes)
                   if i not in (group_idx, weight_idx)
                   and _is_numeric_type(decl)]
    col_idx = {}
    for name in columns:
        idx = _attribute_index(attributes, name, 'analysis')
        if idx in (group_idx, weight_idx):
            raise ValueError('analysis column %r collides with the group or '
                             'weight attribute' % (name,))
        if not _is_numeric_type(attributes[idx][1]):
            raise ValueError('analysis column %r is not numeric: %r'
                             % (name, attributes[idx][1]))
        col_idx[name] = idx

    data = [list(row) for row in data]
    n_rows = len(data)

    base_weights = []
    for r, row in enumerate(data):
        if weight_idx is None or row[weight_idx] is None:
            base_weights.append(DEFAULT_WEIGHT)
        else:
            base_weights.append(float(row[weight_idx]))

    groups = {}
    missing_group_rows = []
    for r, row in enumerate(data):
        value = row[group_idx]
        if value is None:
            missing_group_rows.append(r)
            continue
        groups.setdefault(value, []).append(r)
    if missing_group_rows:
        raise ValueError('rows %r have no group value; every row must belong '
                         'to an independent group' % (missing_group_rows,))
    if not groups:
        raise ValueError('no groups found in group attribute %r' % (group,))

    if control_total is None:
        control_total = sum(base_weights)
    control_total = float(control_total)

    all_rows = list(range(n_rows))

    # -- full-sample weighted means and covariance (pairwise denominators) --
    means = {}
    for name in columns:
        pairs = _column_pairs(data, base_weights, all_rows, col_idx[name])
        estimate, weight_sum = _weighted_mean(pairs)
        means[name] = {
            'estimate': estimate,
            'weight_sum': weight_sum,
            'n_obs': len(pairs),
            'source_rows': [r for r in all_rows
                            if data[r][col_idx[name]] is not None
                            and base_weights[r] != 0],
        }

    covariance = {}
    for name_a in columns:
        covariance[name_a] = {}
        for name_b in columns:
            triples = _pair_pairs(data, base_weights, all_rows,
                                  col_idx[name_a], col_idx[name_b])
            estimate, denominator = _weighted_covariance(triples)
            covariance[name_a][name_b] = {
                'estimate': estimate,
                'denominator': denominator,
                'n_obs': len(triples),
            }

    # -- delete-one-group replicates with recalibrated weights --
    replicates = []
    failed_replicates = []
    for group_value, group_rows in groups.items():
        dropped = set(group_rows)
        remaining_rows = [r for r in all_rows if r not in dropped]
        remaining_weight = sum(base_weights[r] for r in remaining_rows)
        record = {
            'dropped_group': group_value,
            'dropped_source_rows': list(group_rows),
        }
        if remaining_weight == 0:
            record['reason'] = (
                'calibration has no solution: remaining weight sum is 0 '
                'after dropping group %r' % (group_value,))
            failed_replicates.append(record)
            continue
        factor = control_total / remaining_weight
        record['calibration_factor'] = factor
        record['remaining_weight_sum'] = remaining_weight
        record['columns'] = {}
        for name in columns:
            idx = col_idx[name]
            pairs = [(base_weights[r] * factor, data[r][idx])
                     for r in remaining_rows
                     if data[r][idx] is not None
                     and base_weights[r] * factor != 0]
            estimate, weight_sum = _weighted_mean(pairs)
            record['columns'][name] = {
                'mean': estimate,
                'weight_sum': weight_sum,
                'n_obs': len(pairs),
                'source_rows': [r for r in remaining_rows
                                if data[r][idx] is not None
                                and base_weights[r] * factor != 0],
            }
        replicates.append(record)

    # -- jackknife standard errors and mean covariance (pairwise) --
    standard_errors = {}
    mean_covariance = {}
    for name_a in columns:
        values_a = [rep['columns'][name_a]['mean'] for rep in replicates
                    if rep['columns'][name_a]['mean'] is not None]
        variance, _ = _jackknife_combine(values_a)
        standard_errors[name_a] = {
            'estimate': math.sqrt(variance) if variance is not None else None,
            'variance': variance,
            'replicates_used': len(values_a),
        }
        mean_covariance[name_a] = {}
        for name_b in columns:
            pairs = [(rep['columns'][name_a]['mean'],
                      rep['columns'][name_b]['mean'])
                     for rep in replicates
                     if rep['columns'][name_a]['mean'] is not None
                     and rep['columns'][name_b]['mean'] is not None]
            m = len(pairs)
            if m < 2:
                estimate = None
            else:
                mean_a = sum(p[0] for p in pairs) / m
                mean_b = sum(p[1] for p in pairs) / m
                estimate = ((m - 1.0) / m *
                            sum((a - mean_a) * (b - mean_b)
                                for a, b in pairs))
            mean_covariance[name_a][name_b] = {
                'estimate': estimate,
                'replicates_used': m,
            }

    return {
        'source': source,
        'group_attribute': group,
        'weight_attribute': weight,
        'control_total': control_total,
        'columns': list(columns),
        'rows': [{'index': r,
                  'group': data[r][group_idx],
                  'weight': base_weights[r],
                  'weight_declared': weight_idx is not None
                  and data[r][weight_idx] is not None}
                 for r in all_rows],
        'groups': [{'value': value, 'source_rows': rows}
                   for value, rows in groups.items()],
        'means': means,
        'covariance': covariance,
        'standard_errors': standard_errors,
        'mean_covariance': mean_covariance,
        'replicates': replicates,
        'failed_replicates': failed_replicates,
    }


def load_report(path, group, weight=None, columns=None, control_total=None):
    """Load an ARFF file and build the jackknife report for it."""
    with open(path, 'r') as fp:
        dataset = arff.load(fp)
    return jackknife_report(dataset['data'], dataset['attributes'],
                            group=group, weight=weight, columns=columns,
                            control_total=control_total, source=path)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Delete-one-group jackknife with calibrated row weights '
                    'for ARFF data.')
    parser.add_argument('arff_file', help='path to the ARFF file')
    parser.add_argument('--group', required=True,
                        help='attribute identifying the independent sampling '
                             'unit (rows sharing a value are one group)')
    parser.add_argument('--weight', default=None,
                        help='row weight attribute (missing/undeclared '
                             'weights count as 1)')
    parser.add_argument('--columns', nargs='*', default=None,
                        help='analysis columns (default: all numeric columns '
                             'except group/weight)')
    parser.add_argument('--control-total', type=float, default=None,
                        help='calibration target for replicate weights '
                             '(default: full-sample weight sum)')
    args = parser.parse_args(argv)

    report = load_report(args.arff_file, group=args.group, weight=args.weight,
                         columns=args.columns,
                         control_total=args.control_total)
    json.dump(report, sys.stdout, indent=2, default=str)
    sys.stdout.write('\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
