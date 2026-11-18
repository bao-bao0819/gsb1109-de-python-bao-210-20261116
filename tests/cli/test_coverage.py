import json

import numpy as np

import laspy

from . import skip_if_cli_deps_are_not_installed

skip_if_cli_deps_are_not_installed()

from typer.testing import CliRunner

from laspy.cli.core import app

runner = CliRunner()


def write_las(path, x, y, gps_time, classification=None):
    las = laspy.create(point_format=1)
    n = len(x)
    las.x = np.asarray(x, dtype=float)
    las.y = np.asarray(y, dtype=float)
    las.z = np.zeros(n)
    las.return_number = np.ones(n, dtype=np.int64)
    las.number_of_returns = np.ones(n, dtype=np.int64)
    las.gps_time = np.asarray(gps_time, dtype=float)
    if classification is not None:
        las.classification = np.asarray(classification, dtype=np.int64)
    las.write(path)
    return path


def test_review_outputs_json(tmp_path):
    strip = write_las(
        tmp_path / "strip.las",
        x=[0.5, 0.6, 1.5],
        y=[0.5, 0.5, 0.5],
        gps_time=[10.0, 11.0, 12.0],
    )
    output = tmp_path / "review.json"

    result = runner.invoke(
        app,
        [
            "coverage",
            "review",
            str(strip),
            "--cell-size",
            "1",
            "--grid-bounds",
            "0 0 2 1",
            "--min-pulses",
            "2",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Coverage summary" in result.output
    document = json.loads(output.read_text())
    assert document["summary"]["cells"]["covered"] == 1
    assert document["summary"]["cells"]["low_density"] == 1
    assert document["summary"]["cells"]["gap"] == 0
    assert document["summary"]["points"]["input"] == 3


def test_review_multireturn_pulse_deduplication(tmp_path):
    las = laspy.create(point_format=1)
    las.x = np.array([0.5, 0.6, 0.7])
    las.y = np.zeros(3)
    las.z = np.zeros(3)
    las.return_number = np.array([1, 2, 3])
    las.number_of_returns = np.array([3, 3, 3])
    las.gps_time = np.array([10.0, 10.0, 10.0])
    path = tmp_path / "multi.las"
    las.write(path)
    output = tmp_path / "review.json"

    result = runner.invoke(
        app,
        [
            "coverage",
            "review",
            str(path),
            "--cell-size",
            "1",
            "--grid-bounds",
            "0 0 1 1",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    document = json.loads(output.read_text())
    assert document["summary"]["pulses"]["input"] == 1
    assert document["summary"]["points"]["input"] == 3
    assert document["summary"]["cells"]["covered"] == 1


def test_review_applies_filter(tmp_path):
    strip = write_las(
        tmp_path / "strip.las",
        x=[0.5, 0.6],
        y=[0.5, 0.5],
        gps_time=[10.0, 11.0],
        classification=[1, 2],
    )
    output = tmp_path / "review.json"

    result = runner.invoke(
        app,
        [
            "coverage",
            "review",
            str(strip),
            "--cell-size",
            "1",
            "--grid-bounds",
            "0 0 1 1",
            "--filter",
            "classification == 2",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    document = json.loads(output.read_text())
    assert document["summary"]["points"]["retained"] == 1
    assert document["summary"]["points"]["filtered_out"] == 1


def test_review_csv_output(tmp_path):
    strip = write_las(
        tmp_path / "strip.las",
        x=[0.5],
        y=[0.5],
        gps_time=[10.0],
    )
    output = tmp_path / "review.csv"

    result = runner.invoke(
        app,
        [
            "coverage",
            "review",
            str(strip),
            "--cell-size",
            "1",
            "--grid-bounds",
            "0 0 1 1",
            "--min-pulses",
            "2",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    lines = output.read_text().strip().splitlines()
    assert lines[0].startswith("cell_id,")
    assert lines[1].startswith("r0c0,")


def test_review_rejects_invalid_filter(tmp_path):
    strip = write_las(
        tmp_path / "strip.las",
        x=[0.5],
        y=[0.5],
        gps_time=[10.0],
    )

    result = runner.invoke(
        app,
        [
            "coverage",
            "review",
            str(strip),
            "--cell-size",
            "1",
            "--filter",
            "classification >>> 2",
        ],
    )

    assert result.exit_code != 0


def test_compare_distinguishes_new_and_repeat(tmp_path):
    baseline = write_las(
        tmp_path / "baseline.las",
        x=[0.5],
        y=[0.5],
        gps_time=[10.0],
    )
    added = write_las(
        tmp_path / "added.las",
        x=[0.6, 1.5],
        y=[0.5, 0.5],
        gps_time=[11.0, 12.0],
    )
    output = tmp_path / "delta.json"

    result = runner.invoke(
        app,
        [
            "coverage",
            "compare",
            str(baseline),
            "--added",
            str(added),
            "--cell-size",
            "1",
            "--grid-bounds",
            "0 0 2 1",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    document = json.loads(output.read_text())
    delta = document["summary"]["delta"]
    assert delta["newly_covered_cells"] == 1
    assert delta["repeat_observation_cells"] == 1
    classifications = {row["classification"] for row in document["cells"]}
    assert classifications == {"new_coverage", "repeat_observation"}
