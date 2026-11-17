# -*- coding: utf-8 -*-
"""Select a minimum-cost manual review sample from an ARFF file.

Given a set of required rare label combinations, pick the cheapest set of
observations (rows) such that every coverable requirement is represented.

Cost model
----------
- Each observation costs its weight (``--weight`` attribute, default 1).
- An observation selected for several requirements is paid only once
  (set-cover semantics; duplicate selection is never double-counted).

Tie policy
----------
File row order is NOT used as a priority signal.  If several subsets reach
the same minimum cost, all of them (up to ``--limit``) are reported as
interchangeable alternatives, so downstream reviewers can pick freely.

Uncoverable requirements (no row matches them) are reported separately and
excluded from the optimisation target.

Usage
-----
    python review_sample.py data.arff \
        --weight review_weight \
        --require "label_a=fraud,channel=web" \
        --require "label_b=1" \
        --limit 50 --json
"""
import argparse
import json
import sys
from fractions import Fraction
from itertools import combinations

import arff


def parse_requirement(text):
    """Parse 'attr1=val1,attr2=val2' into a dict (AND semantics)."""
    req = {}
    for part in text.split(','):
        part = part.strip()
        if not part:
            continue
        if '=' not in part:
            raise ValueError('invalid requirement term %r (expected attr=value)' % part)
        attr, _, value = part.partition('=')
        attr = attr.strip()
        value = value.strip()
        if not attr:
            raise ValueError('empty attribute name in %r' % part)
        req[attr] = value
    if not req:
        raise ValueError('empty requirement')
    return req


def value_matches(actual, wanted):
    """Compare an ARFF-decoded value against a required string."""
    if actual is None:
        return False
    if isinstance(actual, bool):
        return str(actual).lower() == wanted.lower()
    if isinstance(actual, (int, float)):
        try:
            return float(actual) == float(wanted)
        except ValueError:
            return False
    return str(actual) == wanted


def row_covers(row, columns, requirement):
    for attr, wanted in requirement.items():
        idx = columns.get(attr)
        if idx is None or not value_matches(row[idx], wanted):
            return False
    return True


def solve(rows, weights, masks, n_req):
    """Exact min-cost set cover via DP over requirement bitmask.

    Returns (best_cost, list_of_optimal_subsets_capped).
    Weights are Fractions so equal-cost ties are detected exactly.
    """
    target = (1 << n_req) - 1
    # forward cost table: cost[i][mask] = min cost using rows[:i]
    cost = [{0: Fraction(0)}]
    for i in range(len(rows)):
        prev = cost[-1]
        cur = dict(prev)
        w = weights[i]
        m = masks[i]
        if m:
            for mask, c in prev.items():
                nm = mask | m
                nc = c + w
                if nm not in cur or nc < cur[nm]:
                    cur[nm] = nc
        cost.append(cur)

    best = cost[-1].get(target)
    if best is None:
        return None, []

    solutions = []
    cap = [None]  # mutable limit flag

    def subsets(bits):
        out = []
        b = bits
        while b:
            low = b & (-b)
            out.append(low)
            b ^= low
        return out

    def enum(i, mask, chosen):
        if len(solutions) >= enum.limit:
            return
        if mask == 0:
            solutions.append(tuple(sorted(chosen)))
            return
        if i == 0:
            return
        # option 1: skip row i-1
        if cost[i - 1].get(mask) == cost[i].get(mask):
            enum(i - 1, mask, chosen)
        # option 2: take row i-1
        m = masks[i - 1]
        w = weights[i - 1]
        if m:
            base = mask & ~m
            extra_bits = subsets(m & mask)
            for r in range(len(extra_bits) + 1):
                for combo in combinations(extra_bits, r):
                    prev_mask = base
                    for bit in combo:
                        prev_mask |= bit
                    if cost[i - 1].get(prev_mask) is not None and \
                            cost[i - 1][prev_mask] + w == cost[i][mask]:
                        enum(i - 1, prev_mask, chosen + (i - 1,))

    enum.limit = 10000
    enum(len(rows), target, ())
    return best, sorted(set(solutions))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Minimum-cost review sample covering required rare '
                    'label combinations (ties reported, row order ignored).')
    parser.add_argument('arff_file', help='input ARFF file')
    parser.add_argument('--weight', default=None,
                        help='name of the observation weight attribute '
                             '(default: every row weighs 1)')
    parser.add_argument('--require', action='append', required=True,
                        metavar='ATTR=VAL[,ATTR=VAL...]',
                        help='required label combination; repeatable')
    parser.add_argument('--limit', type=int, default=100,
                        help='max interchangeable optima to print '
                             '(default 100)')
    parser.add_argument('--json', action='store_true',
                        help='emit machine-readable JSON')
    args = parser.parse_args(argv)

    with open(args.arff_file, 'r') as fh:
        dataset = arff.load(fh)

    attributes = [name for name, _ in dataset['attributes']]
    columns = {name: i for i, name in enumerate(attributes)}
    rows = dataset['data']

    requirements = []
    for text in args.require:
        try:
            requirements.append(parse_requirement(text))
        except ValueError as exc:
            parser.error(str(exc))

    unknown = sorted({a for req in requirements for a in req
                      if a not in columns})
    if unknown:
        parser.error('unknown attribute(s) in requirements: %s'
                     % ', '.join(unknown))

    if args.weight is not None and args.weight not in columns:
        parser.error('weight attribute %r not found' % args.weight)
    widx = columns[args.weight] if args.weight else None

    weights = []
    for lineno, row in enumerate(rows, start=1):
        if widx is None:
            weights.append(Fraction(1))
        else:
            raw = row[widx]
            if raw is None:
                parser.error('missing weight at data row %d' % lineno)
            w = Fraction(str(raw))
            if w < 0:
                parser.error('negative weight at data row %d' % lineno)
            weights.append(w)

    # which requirement bits each row covers
    masks = []
    for row in rows:
        mask = 0
        for j, req in enumerate(requirements):
            if row_covers(row, columns, req):
                mask |= 1 << j
        masks.append(mask)

    coverable = 0
    for mask in masks:
        coverable |= mask
    uncoverable = [j for j in range(len(requirements))
                   if not (coverable >> j) & 1]

    cov_reqs = [requirements[j] for j in range(len(requirements))
                if j not in uncoverable]
    # re-index masks onto coverable-only bits
    remap = {}
    bit = 0
    for j in range(len(requirements)):
        if j not in uncoverable:
            remap[j] = bit
            bit += 1
    cov_masks = []
    for mask in masks:
        nm = 0
        for j in remap:
            if (mask >> j) & 1:
                nm |= 1 << remap[j]
        cov_masks.append(nm)

    best, solutions = solve(rows, weights, cov_masks, len(cov_reqs))
    solutions = solutions[:args.limit]

    def fmt_req(req):
        return ','.join('%s=%s' % kv for kv in sorted(req.items()))

    result = {
        'min_cost': str(best) if best is not None else None,
        'n_interchangeable_optima': len(solutions),
        'solutions': [
            {
                'rows': [i + 1 for i in sol],  # 1-based data row numbers
                'cost': str(sum(weights[i] for i in sol)),
            } for sol in solutions
        ],
        'uncoverable_requirements': [fmt_req(requirements[j])
                                     for j in uncoverable],
    }

    if args.json:
        json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write('\n')
        return 0

    if uncoverable:
        print('Uncoverable requirements (no matching observation):')
        for j in uncoverable:
            print('  - %s' % fmt_req(requirements[j]))
    if best is None:
        print('No feasible cover exists.')
        return 1
    print('Minimum review budget (sum of weights, rows counted once): %s'
          % best)
    if len(solutions) == 1:
        print('Unique minimum-cost selection:')
    else:
        print('%d interchangeable minimum-cost selections '
              '(row order carries no priority):' % len(solutions))
    for k, sol in enumerate(solutions, 1):
        rows_str = ', '.join('row %d (w=%s)' % (i + 1, weights[i])
                             for i in sol)
        print('  option %d: %s' % (k, rows_str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
