# -*- coding: utf-8 -*-
#
"""End-to-end tests for offline validation archive capture and replay.

The fixture mirrors an intake batch with two import trees (a SHACL shapes tree
and an ontology tree). Both trees import a byte-identical ``common.ttl`` from
different directories, each document carries relative IRIs and a relative
``owl:imports``, and a later import loops back to an already visited document.
"""
import json
import shutil
import urllib.request
from pathlib import Path

import pytest

from pyshacl import validate
from pyshacl.archive import (
    ArchiveStore,
    IncompleteReplayError,
    capture_validation,
    detect_import_cycles,
    find_missing_imports,
    replay_validation,
)

COMMON_TTL = """\
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.org/model#> .

<> a owl:Ontology ;
    owl:imports <dep.ttl> .

<LocalTerm> rdfs:label "relative IRI resolved against this document base" .

ex:DeviceShape a sh:NodeShape ;
    sh:targetClass ex:Device ;
    sh:property [
        sh:path ex:name ;
        sh:minCount 1 ;
    ] .
"""

SHAPE_ROOT_TTL = """\
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.org/model#> .

<> a owl:Ontology ;
    owl:imports <common.ttl> .

<ShapeLocalTerm> rdfs:label "relative IRI in the shapes entry document" .

ex:Widget a sh:NodeShape ;
    sh:property [ sh:path ex:name ; sh:minCount 1 ] .

# Same names as the ontology classes, but this triple lives in the shapes
# graph. It must never be treated as ontology vocabulary during RDFS
# inference, otherwise ex:redHerring would wrongly become an ex:Device.
ex:RedHerring rdfs:subClassOf ex:Device .
"""

SHAPE_DEP_TTL = """\
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.org/model#> .

<> a owl:Ontology ;
    owl:imports <common.ttl> .

ex:WidgetShape a sh:NodeShape ;
    sh:targetClass ex:Widget .
"""

ONTOLOGY_ROOT_TTL = """\
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

<> a owl:Ontology ;
    owl:imports <common.ttl> .

<OntologyLocalTerm> rdfs:label "relative IRI in the ontology entry document" .
"""

ONTOLOGY_DEP_TTL = """\
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex: <http://example.org/model#> .

<> a owl:Ontology ;
    owl:imports <common.ttl> .

ex:Widget rdfs:subClassOf ex:Device .
ex:SpecialWidget rdfs:subClassOf ex:Widget .
"""

DATA_TTL = """\
@prefix ex: <http://example.org/model#> .

ex:thing1 a ex:SpecialWidget .
ex:thing2 a ex:RedHerring .
"""


def _build_batch(base_dir: Path) -> dict:
    shape_tree = base_dir / "shape_tree"
    ontology_tree = base_dir / "ontology_tree"
    shape_tree.mkdir(parents=True)
    ontology_tree.mkdir(parents=True)
    documents = {
        shape_tree / "root.ttl": SHAPE_ROOT_TTL,
        shape_tree / "common.ttl": COMMON_TTL,
        shape_tree / "dep.ttl": SHAPE_DEP_TTL,
        ontology_tree / "root.ttl": ONTOLOGY_ROOT_TTL,
        ontology_tree / "common.ttl": COMMON_TTL,
        ontology_tree / "dep.ttl": ONTOLOGY_DEP_TTL,
        base_dir / "data.ttl": DATA_TTL,
    }
    for path, text in documents.items():
        path.write_text(text, encoding="utf-8")
    return {
        "data": base_dir / "data.ttl",
        "shapes": shape_tree / "root.ttl",
        "ontology": ontology_tree / "root.ttl",
        "shape_common": shape_tree / "common.ttl",
        "ontology_common": ontology_tree / "common.ttl",
        "shape_dep": shape_tree / "dep.ttl",
        "ontology_dep": ontology_tree / "dep.ttl",
    }


def test_identical_common_bytes_stored_once_with_distinct_bases(tmp_path):
    paths = _build_batch(tmp_path / "batch")
    assert paths["shape_common"].read_bytes() == paths["ontology_common"].read_bytes()

    archive_dir = tmp_path / "archive"
    result = capture_validation(
        archive_dir,
        paths["data"],
        shacl_graph=paths["shapes"],
        ont_graph=paths["ontology"],
        inference="rdfs",
        do_owl_imports=True,
    )

    manifest = result.manifest
    common_docs = [doc for doc in manifest.documents if doc.base_iri.endswith("/common.ttl")]
    assert len(common_docs) == 2
    blob_paths = {doc.blob_path for doc in common_docs}
    assert len(blob_paths) == 1, "identical bytes must be stored once"
    roles = {role for doc in common_docs for role, _occ in doc.role_occurrences}
    assert roles == {"shapes", "ontology"}

    shape_doc = next(doc for doc in common_docs if "shape_tree" in doc.base_iri)
    ontology_doc = next(doc for doc in common_docs if "ontology_tree" in doc.base_iri)
    assert shape_doc.resolved_imports != ontology_doc.resolved_imports
    assert shape_doc.resolved_imports[0].endswith("/shape_tree/dep.ttl")
    assert ontology_doc.resolved_imports[0].endswith("/ontology_tree/dep.ttl")
    assert shape_doc.relative_imports == ["dep.ttl"]
    assert shape_doc.relative_term_count >= 1

    blob_files = list((archive_dir / "blobs").rglob("*"))
    blob_files = [item for item in blob_files if item.is_file()]
    assert len(blob_files) == result.unique_blobs
    manifest_json = json.loads((archive_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_json["validation"]["inference"] == "rdfs"


def test_import_cycle_is_recorded_but_not_re_fetched(tmp_path):
    paths = _build_batch(tmp_path / "batch")
    archive_dir = tmp_path / "archive"
    result = capture_validation(
        archive_dir,
        paths["data"],
        shacl_graph=paths["shapes"],
        ont_graph=paths["ontology"],
        inference="rdfs",
        do_owl_imports=True,
    )

    cycles = detect_import_cycles(result.manifest)
    flattened = {node for cycle in cycles for node in cycle}
    assert any(node.endswith("/common.ttl") for node in flattened)
    assert any(node.endswith("/dep.ttl") for node in flattened)

    common_bases = [doc.base_iri for doc in result.manifest.documents if doc.base_iri.endswith("/common.ttl")]
    common_events = [
        event for event in result.manifest.load_events if event.effective_base in common_bases
    ]
    # The loop-back re-enters an already visited address but is bounded by the
    # import depth limit: it can never keep fetching/copying material forever.
    assert 2 <= len(common_events) < 6
    revisited = {event.effective_base for event in common_events}
    assert revisited == set(common_bases)
    # Triples are a set, so revisiting adds no new document record or blob.
    assert len(common_bases) == 2


def test_rdfs_focus_comes_from_ontology_not_shapes_tree(tmp_path):
    paths = _build_batch(tmp_path / "batch")
    result = capture_validation(
        tmp_path / "archive",
        paths["data"],
        shacl_graph=paths["shapes"],
        ont_graph=paths["ontology"],
        inference="rdfs",
        do_owl_imports=True,
    )
    report = result.report
    assert report.conforms is False
    assert report.focus_nodes == ["http://example.org/model#thing1"]
    assert len(report.violations) == 1
    violation = report.violations[0]
    assert violation.focus_node == "http://example.org/model#thing1"
    assert violation.constraint_component == "http://www.w3.org/ns/shacl#MinCountConstraintComponent"


def test_replay_in_new_directory_offline_matches_archive(tmp_path):
    paths = _build_batch(tmp_path / "batch")
    archive_dir = tmp_path / "archive"
    captured = capture_validation(
        archive_dir,
        paths["data"],
        shacl_graph=paths["shapes"],
        ont_graph=paths["ontology"],
        inference="rdfs",
        do_owl_imports=True,
    )

    moved_dir = tmp_path / "relocated_archive"
    shutil.copytree(archive_dir, moved_dir)

    def deny_network(*args, **kwargs):
        raise AssertionError("offline replay must not perform any network access")

    original_urlopen = urllib.request.urlopen
    urllib.request.urlopen = deny_network
    try:
        replay = replay_validation(moved_dir)
    finally:
        urllib.request.urlopen = original_urlopen

    assert replay.complete is True
    assert replay.report_matches is True
    assert replay.relative_expansions_match is True
    assert replay.event_sequence_match is True
    assert replay.conforms == captured.report.conforms is False
    assert replay.report.focus_nodes == captured.report.focus_nodes
    assert [v.to_dict() for v in replay.report.violations] == [
        v.to_dict() for v in captured.report.violations
    ]


def test_removing_ontology_only_dependency_blocks_green_replay(tmp_path):
    paths = _build_batch(tmp_path / "batch")
    archive_dir = tmp_path / "archive"
    capture_validation(
        archive_dir,
        paths["data"],
        shacl_graph=paths["shapes"],
        ont_graph=paths["ontology"],
        inference="rdfs",
        do_owl_imports=True,
    )

    store = ArchiveStore(archive_dir)
    ontology_dep_id = next(
        doc.document_id
        for doc in store.read_manifest().documents
        if doc.base_iri.endswith("/ontology_tree/dep.ttl")
    )
    missing = find_missing_imports(store.read_manifest(), store, excluded_ids={ontology_dep_id})
    edge = next(item for item in missing if item["missing_document"] == ontology_dep_id)
    assert edge["raw_import"] == "dep.ttl"
    assert edge["importer_base"].endswith("/ontology_tree/common.ttl")
    assert edge["import_iri"].endswith("/ontology_tree/dep.ttl")

    with pytest.raises(IncompleteReplayError) as exc_info:
        replay_validation(archive_dir, exclude_document_ids=[ontology_dep_id])
    assert ontology_dep_id in {item["missing_document"] for item in exc_info.value.missing_imports}

    # Without the ontology-only dependency the RDFS subclass chain is gone, the
    # focus disappears and the run would go green; that is exactly why the
    # incomplete archive must be rejected instead of being silently passed.
    conforms, _graph, _text = validate(
        str(paths["data"]),
        shacl_graph=str(paths["shapes"]),
        ont_graph=None,
        inference="rdfs",
        do_owl_imports=True,
    )
    assert conforms is True
