"""
Command-line interface for :mod:`xport.reconcile`.

Reads an XPT (or CSV) delivery plus a JSON specification and writes the
minimum review set as JSON.

Example specification::

    {
      "identity": "RESP_ID",
      "stratum": "STRATUM",
      "weight": "WEIGHT",
      "value": null,
      "row": "LINE_NO",
      "constraints": [
        {"stratum": "NORTH", "total": 1200, "tolerance": 0.5},
        {"stratum": "SOUTH", "total": 800,  "tolerance": 0.5},
        {"total": 2000, "tolerance": 1.0}
      ]
    }
"""

# Standard Library
import argparse
import json
import sys

# Community Packages
import pandas as pd

# Xport Modules
import xport
import xport.v56
import xport.v89
from xport.reconcile import Constraint, observations_from_dataset, reconcile

__all__ = [
    'main',
]


def _read_delivery(path):
    """
    Read a delivery table from an XPT, CSV, or JSON file.
    """
    if path == '-':
        bytestring = sys.stdin.buffer.read()
        if bytestring[:1] in (b'H', b'{', b'['):
            if bytestring[:1] == b'{' or bytestring[:1] == b'[':
                return pd.DataFrame(json.loads(bytestring))
            if xport.v89.Library.pattern.match(bytestring):
                return next(iter(xport.v89.loads(bytestring).values()))
            return next(iter(xport.v56.loads(bytestring).values()))
        raise ValueError('Standard input must contain XPT or JSON data')

    with open(path, 'rb') as file:
        bytestring = file.read()
    if path.lower().endswith(('.json',)):
        return pd.DataFrame(json.loads(bytestring))
    if path.lower().endswith(('.csv', '.txt')):
        return pd.read_csv(path)
    if xport.v89.Library.pattern.match(bytestring):
        return next(iter(xport.v89.loads(bytestring).values()))
    return next(iter(xport.v56.loads(bytestring).values()))


def _load_spec(path):
    """
    Load the JSON specification.
    """
    with open(path) as file:
        return json.load(file)


def build_parser():
    """
    Create the argument parser.
    """
    parser = argparse.ArgumentParser(
        prog='xport-reconcile',
        description=(
            'Find the minimum set of delivered records that needs review '
            'when a SAS delivery has row weights, strata, and control totals.'
        ),
    )
    parser.add_argument('delivery', help='XPT, CSV, or JSON delivery table')
    parser.add_argument('--spec', required=True, help='JSON specification file')
    parser.add_argument(
        '--output', '-o', default='-',
        help='Write JSON here (default: standard output)',
    )
    parser.add_argument(
        '--allow-negative-weights',
        action='store_true',
        help='Accept negative row weights',
    )
    parser.add_argument(
        '--inconsistent',
        choices=['review', 'error'],
        default='review',
        help='Policy when copies of one identity disagree',
    )
    return parser


def main(argv=None):
    """
    Run the command-line interface.
    """
    args = build_parser().parse_args(argv)
    spec = _load_spec(args.spec)
    dataset = _read_delivery(args.delivery)

    observations = observations_from_dataset(
        dataset,
        identity=spec['identity'],
        stratum=spec.get('stratum'),
        weight=spec['weight'],
        value=spec.get('value'),
        row=spec.get('row'),
    )
    constraints = [
        Constraint(
            total=item['total'],
            stratum=item.get('stratum'),
            tolerance=item.get('tolerance', 0.0),
            name=item.get('name'),
        )
        for item in spec.get('constraints', [])
    ]
    result = reconcile(
        observations,
        constraints,
        allow_negative_weights=args.allow_negative_weights,
        inconsistent=args.inconsistent,
    )

    payload = result.as_dict()
    payload['review_labels'] = result.review_labels
    payload['rows'] = [
        {
            'identity': getattr(row, 'identity', None),
            'stratum': getattr(row, 'stratum', None),
            'weight': getattr(row, 'weight', None),
            'value': getattr(row, 'value', None),
            'row': getattr(row, 'row', None),
        }
        for row in result.review_rows
    ]

    text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    if args.output == '-':
        print(text)
    else:
        with open(args.output, 'w') as file:
            file.write(text + '\n')
    return 0 if result.feasible else 1


if __name__ == '__main__':
    sys.exit(main())
