# -*- coding: utf-8 -*-
"""Tests for the multi-tenant SHACL validation layer.

These tests exercise the isolation guarantees required when several tenants
share RDF datasets and a shape library: per-tenant authorization, pinned
dictionary versions, focus-URI scoping, independence of failed/concurrent
runs, and report reproducibility from the tenant's own materials.
"""
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from rdflib import URIRef

from pyshacl.multitenant import (
    MaterialManifest,
    TenantScopeError,
    TenantValidationError,
    TenantValidationReport,
    TenantValidationRunner,
    TenantValidationSpec,
    graph_fingerprint,
)
from pyshacl.multitenant.manifest import load_material_graph
from pyshacl.rdfutil import clone_graph

PUBLIC_SHAPES = """\
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.com/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

ex:ThingShape a sh:NodeShape ;
    sh:targetClass ex:Thing ;
    sh:property [
        sh:path ex:prop ;
        sh:datatype xsd:string ;
        sh:minCount 1 ;
    ] .
"""

TENANT_A_SHAPES = """\
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.com/tenant-a/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

ex:TenantThingShape a sh:NodeShape ;
    sh:targetClass ex:TenantThing ;
    sh:property [ sh:path ex:tenantProp ; sh:minCount 1 ; sh:datatype xsd:string ] .
"""

TENANT_B_SHAPES = """\
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.com/tenant-b/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

ex:TenantThingShape a sh:NodeShape ;
    sh:targetClass ex:TenantThing ;
    sh:property [ sh:path ex:tenantProp ; sh:minCount 1 ; sh:datatype xsd:integer ] .
"""

MALFORMED_SHAPES = """\
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.com/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

ex:MalformedShape a sh:NodeShape ;
    sh:targetClass ex:Thing ;
    sh:property [ sh:path ex:prop ; sh:minCount "not-an-integer"^^xsd:string ] .
"""

DATA_A_OK = """\
@prefix ex: <http://example.com/> .

ex:node1 a ex:Thing ;
    ex:prop "ok" .
"""

DATA_A_BAD = """\
@prefix ex: <http://example.com/> .

ex:node2 a ex:Thing .
"""

DATA_A_TWO_THINGS = """\
@prefix ex: <http://example.com/> .

ex:good a ex:Thing ; ex:prop "ok" .
ex:bad a ex:Thing .
"""

# Same focus URI as tenant B uses, but with tenant-A specific content.
DATA_A_SHARED_FOCUS = """\
@prefix ex: <http://example.com/> .

ex:shared-focus a ex:Thing ;
    ex:prop "tenant-a value" .
"""

DATA_B_SHARED_FOCUS = """\
@prefix ex: <http://example.com/> .
@prefix tb: <http://example.com/tenant-b/> .

ex:shared-focus a ex:Thing ;
    tb:tenantProp "not-an-integer" .
"""

DATA_B_TENANT_THING = """\
@prefix tb: <http://example.com/tenant-b/> .

tb:thing1 a tb:TenantThing ;
    tb:tenantProp 42 .
"""

DICTIONARY_V1 = """\
@prefix ex: <http://example.com/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

ex:SpecialThing rdfs:subClassOf ex:Thing .
ex:code1 rdfs:label "code-one-v1" .
"""

DICTIONARY_V2 = """\
@prefix ex: <http://example.com/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

ex:SpecialThing rdfs:subClassOf ex:Thing .
ex:code1 rdfs:label "code-one-v2" .
ex:code2 rdfs:label "code-two-v2" .
"""

DATA_SPECIAL = """\
@prefix ex: <http://example.com/> .

ex:special1 a ex:SpecialThing .
"""


def build_manifest_a():
    manifest = MaterialManifest("tenant-a")
    manifest.add_public_shapes("public-shapes", PUBLIC_SHAPES, version="2026.1")
    manifest.add_tenant_shapes("tenant-a-shapes", TENANT_A_SHAPES, version="a-7")
    manifest.add_data_graph("a-ok", DATA_A_OK)
    manifest.add_data_graph("a-bad", DATA_A_BAD)
    manifest.add_data_graph("a-two-things", DATA_A_TWO_THINGS)
    manifest.add_data_graph("a-shared-focus", DATA_A_SHARED_FOCUS)
    manifest.add_shared_dictionary("dict-v1", DICTIONARY_V1, version="1.0.0")
    manifest.add_shared_dictionary("dict-v2", DICTIONARY_V2, version="2.0.0")
    manifest.add_data_graph("a-special", DATA_SPECIAL)
    return manifest


def build_manifest_b():
    manifest = MaterialManifest("tenant-b")
    manifest.add_public_shapes("public-shapes", PUBLIC_SHAPES, version="2026.1")
    manifest.add_tenant_shapes("tenant-b-shapes", TENANT_B_SHAPES, version="b-3")
    manifest.add_data_graph("b-shared-focus", DATA_B_SHARED_FOCUS)
    manifest.add_data_graph("b-tenant-thing", DATA_B_TENANT_THING)
    return manifest


def test_graph_fingerprint_is_stable_under_clone():
    source = load_material_graph(DATA_A_OK)
    cloned = clone_graph(source)
    assert graph_fingerprint(source) == graph_fingerprint(cloned)
    other = load_material_graph(DATA_A_BAD)
    assert graph_fingerprint(source) != graph_fingerprint(other)


def test_tenant_cannot_reference_unauthorized_material():
    manifest_a = build_manifest_a()
    manifest_b = build_manifest_b()
    runner_a = TenantValidationRunner(manifest_a)

    with pytest.raises(TenantScopeError) as exc:
        runner_a.validate(TenantValidationSpec(data_graph_ids=["b-shared-focus"]))
    assert exc.value.tenant_id == "tenant-a"
    assert exc.value.reference == "b-shared-focus"

    with pytest.raises(TenantScopeError):
        runner_a.validate(
            TenantValidationSpec(data_graph_ids=["a-ok"], shapes_ids=["tenant-b-shapes"])
        )

    with pytest.raises(TenantScopeError):
        runner_a.validate(
            TenantValidationSpec(data_graph_ids=["a-ok"], dictionary_ids=["dict-v9"])
        )

    # A report belonging to tenant B cannot be replayed inside tenant A.
    runner_b = TenantValidationRunner(manifest_b)
    report_b = runner_b.validate(TenantValidationSpec(data_graph_ids=["b-tenant-thing"]))
    with pytest.raises(TenantScopeError):
        runner_a.replay(report_b)


def test_report_only_contains_authorized_materials():
    manifest = build_manifest_a()
    runner = TenantValidationRunner(manifest)
    report = runner.validate(
        TenantValidationSpec(
            data_graph_ids=["a-ok"],
            shapes_ids=["public-shapes"],
            dictionary_ids=["dict-v1"],
            dictionary_versions={"dict-v1": "1.0.0"},
        )
    )
    assert report.conforms is True
    assert report.error is None
    assert report.used_material_ids["data"] == ["a-ok"]
    assert report.used_material_ids["shapes"] == ["public-shapes"]
    assert report.used_material_ids["dictionaries"] == ["dict-v1"]
    # The full tenant snapshot is embedded, including materials not used in
    # this run, so the report is self-contained for this tenant.
    assert set(report.material_snapshot.keys()) == set(manifest.material_ids())
    assert report.material_snapshot["dict-v1"]["version"] == "1.0.0"
    assert report.material_snapshot["dict-v1"]["visibility"] == "public"
    assert report.material_snapshot["a-ok"]["visibility"] == "tenant"


def test_default_selection_combines_public_and_tenant_shapes_and_dicts():
    manifest = build_manifest_a()
    runner = TenantValidationRunner(manifest)
    report = runner.validate(TenantValidationSpec(data_graph_ids=["a-ok"]))
    assert sorted(report.used_material_ids["shapes"]) == ["public-shapes", "tenant-a-shapes"]
    assert sorted(report.used_material_ids["dictionaries"]) == ["dict-v1", "dict-v2"]


def test_same_focus_uri_is_resolved_per_tenant():
    manifest_a = build_manifest_a()
    manifest_b = build_manifest_b()
    focus = "http://example.com/shared-focus"

    report_a = TenantValidationRunner(manifest_a).validate(
        TenantValidationSpec(data_graph_ids=["a-shared-focus"], focus_nodes=[focus])
    )
    report_b = TenantValidationRunner(manifest_b).validate(
        TenantValidationSpec(
            data_graph_ids=["b-shared-focus"],
            shapes_ids=["public-shapes", "tenant-b-shapes"],
            focus_nodes=[focus],
        )
    )

    # Same URI, different tenant graphs and shapes: tenant A conforms while
    # tenant B's integer-datatype tenant shape fails.
    assert report_a.conforms is True
    assert report_b.conforms is False
    assert report_a.focus_node_visibility[focus] == ["a-shared-focus"]
    assert report_b.focus_node_visibility[focus] == ["b-shared-focus"]
    # The reports must not leak each other's material ids.
    assert "tenant-b-shapes" not in report_a.material_snapshot
    assert "a-shared-focus" not in report_b.material_snapshot


def test_focus_node_outside_visible_graphs_is_rejected():
    manifest = build_manifest_a()
    runner = TenantValidationRunner(manifest)
    with pytest.raises(TenantScopeError) as exc:
        runner.validate(
            TenantValidationSpec(
                data_graph_ids=["a-ok"],
                focus_nodes=["http://example.com/does-not-exist"],
            )
        )
    assert exc.value.reference == "http://example.com/does-not-exist"


def test_focus_nodes_restrict_target_discovery():
    manifest = build_manifest_a()
    runner = TenantValidationRunner(manifest)
    full = runner.validate(TenantValidationSpec(data_graph_ids=["a-two-things"]))
    assert full.conforms is False
    focused = runner.validate(
        TenantValidationSpec(
            data_graph_ids=["a-two-things"],
            focus_nodes=[URIRef("http://example.com/good")],
        )
    )
    assert focused.conforms is True


def test_dictionary_version_pin_and_participation_in_inference():
    manifest = build_manifest_a()
    runner = TenantValidationRunner(manifest)

    # Without the dictionary, RDFS inference never types ex:special1 as
    # ex:Thing, so the public shape has no target and validation conforms.
    no_dict = runner.validate(
        TenantValidationSpec(
            data_graph_ids=["a-special"],
            shapes_ids=["public-shapes"],
            dictionary_ids=[],
            inference="rdfs",
        )
    )
    assert no_dict.conforms is True

    # Mixing in the versioned dictionary and running RDFS inference adds the
    # ex:Thing type, so the shape now applies and finds the missing property.
    with_dict = runner.validate(
        TenantValidationSpec(
            data_graph_ids=["a-special"],
            shapes_ids=["public-shapes"],
            dictionary_ids=["dict-v1"],
            dictionary_versions={"dict-v1": "1.0.0"},
            inference="rdfs",
        )
    )
    assert with_dict.conforms is False

    # A wrong version pin is refused before validation runs.
    with pytest.raises(TenantValidationError):
        runner.validate(
            TenantValidationSpec(
                data_graph_ids=["a-special"],
                shapes_ids=["public-shapes"],
                dictionary_ids=["dict-v1"],
                dictionary_versions={"dict-v1": "9.9.9"},
                inference="rdfs",
            )
        )

    # v1 and v2 fingerprints differ, proving the pinned version is the one used.
    assert (
        manifest.get_material("dict-v1").fingerprint
        != manifest.get_material("dict-v2").fingerprint
    )


def test_failed_run_does_not_pollute_following_run_or_manifest_graphs():
    manifest = build_manifest_a()
    manifest.add_tenant_shapes("malformed-shapes", MALFORMED_SHAPES)
    runner = TenantValidationRunner(manifest)

    before = {mid: len(material.graph) for mid, material in manifest._materials.items()}

    failed = runner.validate(
        TenantValidationSpec(data_graph_ids=["a-ok"], shapes_ids=["malformed-shapes"])
    )
    assert failed.conforms is False
    assert failed.error is not None and "ConstraintLoadError" in failed.error

    # The next, unrelated run is unaffected by the failed neighbour.
    healthy = runner.validate(
        TenantValidationSpec(data_graph_ids=["a-ok"], shapes_ids=["public-shapes"])
    )
    assert healthy.conforms is True
    assert healthy.error is None

    after = {mid: len(material.graph) for mid, material in manifest._materials.items()}
    # Inference/cloning never writes back into the manifest-owned source graphs.
    assert before == after


def test_concurrent_runs_keep_tenants_and_state_independent():
    manifest_a = build_manifest_a()
    manifest_b = build_manifest_b()
    runner_a = TenantValidationRunner(manifest_a)
    runner_b = TenantValidationRunner(manifest_b)

    jobs = []
    for i in range(12):
        jobs.append(
            (
                "a",
                TenantValidationSpec(
                    data_graph_ids=["a-ok"],
                    shapes_ids=["public-shapes"],
                    focus_nodes=["http://example.com/node1"],
                    inference="rdfs",
                    label=f"a-ok-{i}",
                ),
                True,
            )
        )
        jobs.append(
            (
                "a",
                TenantValidationSpec(data_graph_ids=["a-bad"], shapes_ids=["public-shapes"]),
                False,
            )
        )
        jobs.append(
            (
                "b",
                TenantValidationSpec(
                    data_graph_ids=["b-tenant-thing"], shapes_ids=["tenant-b-shapes"]
                ),
                True,
            )
        )

    def run_job(job):
        tenant, spec, _expected = job
        runner = runner_a if tenant == "a" else runner_b
        return tenant, runner.validate(spec)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(run_job, jobs))

    assert len(results) == len(jobs)
    for (tenant, report), (_t, _spec, expected_conforms) in zip(results, jobs):
        expected_tenant_id = "tenant-a" if tenant == "a" else "tenant-b"
        assert report.tenant_id == expected_tenant_id
        assert report.conforms is expected_conforms
        assert report.error is None
        if tenant == "a":
            assert all(mid.startswith(("a-", "public-", "dict-", "tenant-a")) for mid in report.material_snapshot)
        else:
            assert "tenant-a-shapes" not in report.material_snapshot
            assert "a-ok" not in report.material_snapshot

    # Equivalent concurrent specs must hash to identical per-run input graphs.
    a_ok_reports = [
        r for _tenant, r in results if (r.spec.get("label") or "").startswith("a-ok-")
    ]
    fingerprints = {r.input_fingerprints["data_graph"] for r in a_ok_reports}
    assert len(fingerprints) == 1


def test_report_is_json_roundtrippable_and_replayable():
    manifest = build_manifest_a()
    runner = TenantValidationRunner(manifest)
    report = runner.validate(
        TenantValidationSpec(
            data_graph_ids=["a-ok"],
            shapes_ids=["public-shapes"],
            dictionary_ids=["dict-v2"],
            dictionary_versions={"dict-v2": "2.0.0"},
        )
    )

    encoded = report.to_json()
    restored = TenantValidationReport.from_dict(json.loads(encoded))
    assert restored.run_id == report.run_id
    assert restored.input_fingerprints == report.input_fingerprints

    replayed = runner.replay(restored)
    assert replayed.conforms == report.conforms
    assert replayed.input_fingerprints == report.input_fingerprints
    assert replayed.material_snapshot == report.material_snapshot


def test_replay_detects_tampered_material():
    manifest = build_manifest_a()
    runner = TenantValidationRunner(manifest)
    report = runner.validate(
        TenantValidationSpec(data_graph_ids=["a-ok"], shapes_ids=["public-shapes"])
    )

    tampered = MaterialManifest("tenant-a")
    tampered.add_public_shapes("public-shapes", PUBLIC_SHAPES, version="2026.1")
    tampered.add_data_graph("a-ok", DATA_A_BAD)  # same id, different triples
    tampered_runner = TenantValidationRunner(tampered)
    with pytest.raises(TenantValidationError):
        tampered_runner.replay(report)


def test_replay_detects_version_drift():
    manifest = build_manifest_a()
    runner = TenantValidationRunner(manifest)
    report = runner.validate(
        TenantValidationSpec(
            data_graph_ids=["a-ok"],
            shapes_ids=["public-shapes"],
            dictionary_ids=["dict-v1"],
        )
    )
    drifted = TenantValidationReport.from_dict(json.loads(report.to_json()))
    drifted.material_snapshot["dict-v1"]["version"] = "0.0.1-drifted"
    with pytest.raises(TenantValidationError):
        runner.replay(drifted)


def test_report_reproducible_from_rebuilt_tenant_manifest():
    # First process: build manifest, validate, persist only the report JSON.
    runner = TenantValidationRunner(build_manifest_a())
    original = runner.validate(
        TenantValidationSpec(
            data_graph_ids=["a-ok"],
            shapes_ids=["public-shapes", "tenant-a-shapes"],
            dictionary_ids=["dict-v2"],
            dictionary_versions={"dict-v2": "2.0.0"},
        )
    )
    persisted = original.to_json()

    # Second process: the tenant rebuilds a manifest from its own authorized
    # materials. No other tenant's state is available or consulted.
    rebuilt_runner = TenantValidationRunner(build_manifest_a())
    restored = TenantValidationReport.from_dict(json.loads(persisted))
    replayed = rebuilt_runner.replay(restored)

    assert replayed.conforms == original.conforms
    assert replayed.input_fingerprints == original.input_fingerprints
    assert replayed.used_material_ids == original.used_material_ids


def test_replay_fails_when_rebuilt_manifest_is_missing_a_material():
    runner = TenantValidationRunner(build_manifest_a())
    original = runner.validate(
        TenantValidationSpec(data_graph_ids=["a-ok"], shapes_ids=["public-shapes"])
    )

    incomplete = MaterialManifest("tenant-a")
    incomplete.add_public_shapes("public-shapes", PUBLIC_SHAPES, version="2026.1")
    # a-ok is intentionally not registered.
    with pytest.raises(TenantValidationError):
        TenantValidationRunner(incomplete).replay(
            TenantValidationReport.from_dict(json.loads(original.to_json()))
        )


def test_duplicate_registration_and_bad_specs_are_rejected():
    manifest = build_manifest_a()
    with pytest.raises(TenantValidationError):
        manifest.add_data_graph("a-ok", DATA_A_OK)
    runner = TenantValidationRunner(manifest)
    with pytest.raises(TenantValidationError):
        runner.validate(TenantValidationSpec(data_graph_ids=[]))
    with pytest.raises(TenantValidationError):
        runner.validate(
            TenantValidationSpec(data_graph_ids=["a-ok"], shapes_ids=["nonexistent-shape"])
        )


def test_validate_batch_keeps_each_task_isolated_and_ordered():
    manifest = build_manifest_a()
    runner = TenantValidationRunner(manifest)
    specs = [
        TenantValidationSpec(data_graph_ids=["a-ok"], shapes_ids=["public-shapes"]),
        TenantValidationSpec(data_graph_ids=["a-bad"], shapes_ids=["public-shapes"]),
        TenantValidationSpec(data_graph_ids=["not-authorized"], shapes_ids=["public-shapes"]),
        TenantValidationSpec(
            data_graph_ids=["a-ok"],
            shapes_ids=["public-shapes"],
            focus_nodes=["http://example.com/node1"],
        ),
    ]

    reports = runner.validate_batch(specs)
    assert [r.used_material_ids["data"] for r in reports] == [
        ["a-ok"],
        ["a-bad"],
        [],
        ["a-ok"],
    ]
    assert reports[0].conforms is True
    assert reports[1].conforms is False
    # The authorization failure is contained to its own task's report and
    # does not stop the remaining tasks from running.
    assert reports[2].conforms is False
    assert "TenantScopeError" in reports[2].error
    assert reports[3].conforms is True
    # Independent runs of the same spec must build identical input graphs.
    assert (
        reports[0].input_fingerprints["data_graph"]
        == reports[3].input_fingerprints["data_graph"]
    )


def test_validate_batch_runs_concurrently_without_state_cross_talk():
    manifest = build_manifest_a()
    runner = TenantValidationRunner(manifest)
    specs = []
    for index in range(20):
        specs.append(
            TenantValidationSpec(
                data_graph_ids=["a-ok"] if index % 2 == 0 else ["a-bad"],
                shapes_ids=["public-shapes"],
                label=f"batch-{index}",
            )
        )

    reports = runner.validate_batch(specs, max_workers=8)
    assert len(reports) == len(specs)
    for index, report in enumerate(reports):
        assert report.spec["label"] == f"batch-{index}"  # submission order preserved
        assert report.tenant_id == "tenant-a"
        assert report.error is None
        assert report.conforms is (index % 2 == 0)

    ok_fingerprints = {r.input_fingerprints["data_graph"] for r in reports[::2]}
    bad_fingerprints = {r.input_fingerprints["data_graph"] for r in reports[1::2]}
    assert len(ok_fingerprints) == 1
    assert len(bad_fingerprints) == 1
    assert ok_fingerprints != bad_fingerprints


def test_validate_batch_fail_fast_stops_after_first_error():
    manifest = build_manifest_a()
    runner = TenantValidationRunner(manifest)
    specs = [
        TenantValidationSpec(data_graph_ids=["a-ok"], shapes_ids=["public-shapes"]),
        TenantValidationSpec(data_graph_ids=["a-bad"], shapes_ids=["public-shapes"]),
        TenantValidationSpec(data_graph_ids=["a-ok"], shapes_ids=["public-shapes"]),
    ]
    # A non-conforming but valid run is not an error, so fail_fast continues.
    reports = runner.validate_batch(specs, fail_fast=True)
    assert len(reports) == 3
    assert [r.conforms for r in reports] == [True, False, True]

    error_specs = [
        TenantValidationSpec(data_graph_ids=["a-ok"], shapes_ids=["public-shapes"]),
        TenantValidationSpec(data_graph_ids=["a-ok"], shapes_ids=["missing-shape"]),
        TenantValidationSpec(data_graph_ids=["a-ok"], shapes_ids=["public-shapes"]),
    ]
    fast_reports = runner.validate_batch(error_specs, fail_fast=True)
    assert len(fast_reports) == 2
    assert fast_reports[0].error is None
    assert "TenantScopeError" in fast_reports[1].error


def test_validate_batch_rejects_empty_and_typed_input():
    manifest = build_manifest_a()
    runner = TenantValidationRunner(manifest)
    assert runner.validate_batch([]) == []
    with pytest.raises(TenantValidationError):
        runner.validate_batch(["not-a-spec"])
