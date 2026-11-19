# -*- coding: utf-8 -*-
"""CLI tests for the ordered SHACL Rules closure report."""

import json
import os
import platform
import subprocess
import sys
from os import getenv, path

import pytest

here_dir = path.abspath(path.dirname(__file__))
project_dir = path.abspath(path.join(here_dir, os.pardir))
cli_rules_script = path.join(project_dir, "pyshacl", "cli_rules.py")


def _run_cli(tmp_path, *extra):
    data_file = tmp_path / "data.ttl"
    shacl_file = tmp_path / "shapes.ttl"
    report_file = tmp_path / "report.json"
    data_file.write_text(
        """
@prefix ex: <http://example.com/rules-closure#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
ex:a a ex:Thing .
""",
        encoding="utf-8",
    )
    shacl_file.write_text(
        """
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.com/rules-closure#> .

ex:NeedsBShape
    a sh:NodeShape ;
    sh:targetClass ex:Thing ;
    sh:order 0 ;
    sh:rule [
        a sh:TripleRule ;
        sh:subject sh:this ;
        sh:predicate ex:c ;
        sh:object true ;
        sh:condition [ sh:property [ sh:path ex:b ; sh:hasValue true ] ] ;
    ] .

ex:ProducesBShape
    a sh:NodeShape ;
    sh:targetClass ex:Thing ;
    sh:order 1 ;
    sh:rule [
        a sh:TripleRule ;
        sh:subject sh:this ;
        sh:predicate ex:b ;
        sh:object true ;
    ] .
""",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((project_dir, env.get("PYTHONPATH", "")))
    cmd = [
        sys.executable,
        cli_rules_script,
        str(data_file),
        "-s",
        str(shacl_file),
        "-it",
        "--rules-report",
        str(report_file),
    ]
    cmd.extend(extra)
    return subprocess.run(cmd, capture_output=True, text=True, env=env), report_file


@pytest.mark.skipif(platform.system() == "Windows", reason="Commandline tests use POSIX paths")
def test_cli_rules_report_converged(tmp_path):
    result, report_file = _run_cli(tmp_path)
    assert result.returncode == 0, result.stderr
    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert report["status"] == "converged"
    assert report["round_count"] == 3
    assert report["total_added"] == 2
    assert "<http://example.com/rules-closure#c>" in " ".join(report["closure_triples"])


@pytest.mark.skipif(platform.system() == "Windows", reason="Commandline tests use POSIX paths")
def test_cli_rules_report_incomplete_exit_code(tmp_path):
    result, report_file = _run_cli(tmp_path, "--max-rule-rounds", "1")
    assert result.returncode == 4, result.stderr
    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert report["status"] == "incomplete"
    assert report["incomplete"] is True
    assert report["reason"] == "max_rounds_exhausted"
    assert "incomplete" in result.stderr.lower()


@pytest.mark.skipif(platform.system() == "Windows", reason="Commandline tests use POSIX paths")
def test_main_cli_rules_delegation_forwards_closure_options(tmp_path):
    """`pyshacl --rules` must forward the closure options to the rules CLI."""
    data_file = tmp_path / "data.ttl"
    shacl_file = tmp_path / "shapes.ttl"
    report_file = tmp_path / "report.json"
    data_file.write_text(
        "@prefix ex: <http://example.com/rules-closure#> .\n"
        "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n"
        "ex:a a ex:Thing .\n",
        encoding="utf-8",
    )
    shacl_file.write_text(
        """
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.com/rules-closure#> .
ex:NeedsBShape a sh:NodeShape ; sh:targetClass ex:Thing ; sh:order 0 ;
    sh:rule [ a sh:TripleRule ; sh:subject sh:this ; sh:predicate ex:c ; sh:object true ;
              sh:condition [ sh:property [ sh:path ex:b ; sh:hasValue true ] ] ] .
ex:ProducesBShape a sh:NodeShape ; sh:targetClass ex:Thing ; sh:order 1 ;
    sh:rule [ a sh:TripleRule ; sh:subject sh:this ; sh:predicate ex:b ; sh:object true ] .
""",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((project_dir, env.get("PYTHONPATH", "")))
    cmd = [
        sys.executable,
        path.join(project_dir, "pyshacl", "cli.py"),
        str(data_file),
        "-s",
        str(shacl_file),
        "--rules",
        "-it",
        "--max-rule-rounds",
        "1",
        "--rules-report",
        str(report_file),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert result.returncode == 4, result.stderr
    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert report["status"] == "incomplete"
    assert report["round_count"] == 1
