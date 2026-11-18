"""
Tests for the reconciliation command-line interface.
"""

# Standard Library
import json

# Community Packages
import pytest

# Xport Modules
import xport
import xport.v56
from xport.reconcile_cli import main


@pytest.fixture
def delivery(tmp_path):
    dataset = xport.Dataset(
        {
            'RESP_ID': ['a', 'a', 'b', 'c'],
            'STRATUM': ['NORTH', 'NORTH', 'NORTH', 'SOUTH'],
            'WEIGHT': [2.0, 2.0, 3.0, 4.0],
            'LINE_NO': [1, 2, 3, 4],
        },
        name='DELIVERY',
    )
    path = tmp_path / 'delivery.xpt'
    with open(path, 'wb') as file:
        xport.v56.dump(dataset, file)
    return path


@pytest.fixture
def spec(tmp_path):
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps({
        'identity': 'RESP_ID',
        'stratum': 'STRATUM',
        'weight': 'WEIGHT',
        'row': 'LINE_NO',
        'constraints': [
            {'stratum': 'NORTH', 'total': 5.0},
            {'stratum': 'SOUTH', 'total': 4.0},
            {'total': 9.0},
        ],
    }))
    return path


def test_cli_writes_json(delivery, spec, tmp_path, capsys):
    output = tmp_path / 'result.json'
    code = main([str(delivery), '--spec', str(spec), '--output', str(output)])
    assert code == 0
    payload = json.loads(output.read_text())
    assert payload['feasible'] is True
    assert payload['cost'] == 1
    # The second NORTH copy is an unequivocal duplicate import, so no row
    # needs review; all four rows belong to trusted occurrences.
    assert payload['review'] == []
    assert payload['review_labels'] == []


def test_cli_stdout(delivery, spec, capsys):
    assert main([str(delivery), '--spec', str(spec)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['feasible']


def test_cli_failure_exit_code(tmp_path):
    dataset = xport.Dataset(
        {'ID': ['a'], 'W': [3.0], 'LINE': [1]},
        name='D',
    )
    delivery = tmp_path / 'delivery.xpt'
    with open(delivery, 'wb') as file:
        xport.v56.dump(dataset, file)
    spec = tmp_path / 'spec.json'
    spec.write_text(json.dumps({
        'identity': 'ID',
        'weight': 'W',
        'constraints': [{'total': 0.0}],
    }))
    assert main([str(delivery), '--spec', str(spec)]) == 1


def test_cli_csv(tmp_path):
    path = tmp_path / 'delivery.csv'
    path.write_text('ID,STRATUM,WEIGHT\na,S,2\na,S,2\nb,S,3\n')
    spec = tmp_path / 'spec.json'
    spec.write_text(json.dumps({
        'identity': 'ID',
        'stratum': 'STRATUM',
        'weight': 'WEIGHT',
        'constraints': [{'stratum': 'S', 'total': 5.0}],
    }))
    assert main([str(path), '--spec', str(spec)]) == 0
