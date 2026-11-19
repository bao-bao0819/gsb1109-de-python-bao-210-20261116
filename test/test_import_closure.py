# -*- coding: utf-8 -*-
#
# Tests for pyshacl.rdfutil.import_closure: capturing the actual owl:imports
# closure (with version identity) for offline, reproducible validation.
import json

import pytest
import rdflib
from rdflib.compare import to_isomorphic

from pyshacl.errors import ReportableRuntimeError
from pyshacl.rdfutil import capture_import_closure, load_from_source, replay_import_closure
from pyshacl.rdfutil.import_closure import MANIFEST_FILENAME, ImportClosure

ONT_A = """\
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

<http://example.com/ontA> a owl:Ontology ;
    owl:imports <{ont_b_uri}> .

<http://example.com/ontA#Widget> a rdfs:Class .
"""

ONT_B = """\
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

<http://example.com/ontB> a owl:Ontology ;
    owl:versionIRI <http://example.com/ontB/1.2.0> ;
    owl:versionInfo "1.2.0" .

<http://example.com/ontB#Gadget> a rdfs:Class .
"""

ROOT = """\
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix sh: <http://www.w3.org/ns/shacl#> .

<http://example.com/root> a owl:Ontology ;
    owl:imports <{ont_a_uri}> .

<http://example.com/root#WidgetShape> a sh:NodeShape .
"""


def _write(path, text):
    path.write_text(text, encoding='utf-8')
    return path


def _basic_tree(tmp_path):
    ont_b = _write(tmp_path / "ontB.ttl", ONT_B)
    ont_a = _write(tmp_path / "ontA.ttl", ONT_A.format(ont_b_uri=ont_b.as_uri()))
    root = _write(tmp_path / "root.ttl", ROOT.format(ont_a_uri=ont_a.as_uri()))
    return root, ont_a, ont_b


def test_capture_records_closure_and_version_identity(tmp_path):
    root, ont_a, ont_b = _basic_tree(tmp_path)
    closure_dir = tmp_path / "closure"
    closure = capture_import_closure(str(root), closure_dir)

    assert (closure_dir / MANIFEST_FILENAME).is_file()
    assert closure.root_record is not None
    assert closure.root_record.verified
    assert closure.unverified == []

    imports = {r.uri: r for r in closure.records if r.kind == "import"}
    assert set(imports) == {ont_a.as_uri(), ont_b.as_uri()}

    rec_a = imports[ont_a.as_uri()]
    rec_b = imports[ont_b.as_uri()]
    assert rec_a.verified and rec_a.status == "resolved"
    assert rec_a.depth == 1
    assert rec_a.digest.startswith("sha256:")
    assert rec_a.imported_by == ["http://example.com/root"]
    assert rec_b.depth == 2
    assert rec_b.imported_by == ["http://example.com/ontA"]

    # Version identity of the imported ontology is captured
    assert rec_b.ontology_uri == "http://example.com/ontB"
    assert rec_b.version_iri == "http://example.com/ontB/1.2.0"
    assert rec_b.version_info == "1.2.0"

    # Stored content matches the recorded digest
    import hashlib

    stored = (closure_dir / rec_a.content_file).read_bytes()
    assert rec_a.digest == "sha256:" + hashlib.sha256(stored).hexdigest()


def test_replay_rebuilds_same_graph_as_live_imports(tmp_path):
    root, ont_a, ont_b = _basic_tree(tmp_path)
    closure_dir = tmp_path / "closure"
    closure = capture_import_closure(str(root), closure_dir)

    replayed, loaded_closure = replay_import_closure(closure_dir)
    assert isinstance(loaded_closure, ImportClosure)
    assert loaded_closure.unverified == []

    replayed_graph = rdflib.Graph()
    for triple in replayed.default_graph:
        replayed_graph.add(triple)
    live = load_from_source(str(root), do_owl_imports=True)
    assert to_isomorphic(replayed_graph) == to_isomorphic(live)
    assert closure.root_uri == loaded_closure.root_uri


def test_duplicate_imports_and_cycles_are_recorded_not_refollowed(tmp_path):
    ont_c = _write(
        tmp_path / "ontC.ttl",
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
        "<http://example.com/ontC> a owl:Ontology .\n",
    )
    # B imports C and also its importer A (a cycle back)
    ont_a_uri = (tmp_path / "ontA.ttl").as_uri()
    ont_b = _write(
        tmp_path / "ontB.ttl",
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
        "<http://example.com/ontB> a owl:Ontology ;\n"
        "    owl:imports <{}>, <{}> .\n".format(ont_c.as_uri(), ont_a_uri),
    )
    ont_a = _write(
        tmp_path / "ontA.ttl",
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
        "<http://example.com/ontA> a owl:Ontology ;\n"
        "    owl:imports <{}>, <{}> .\n".format(ont_b.as_uri(), ont_c.as_uri()),
    )
    root = _write(
        tmp_path / "root.ttl",
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
        "<http://example.com/root> a owl:Ontology ;\n"
        "    owl:imports <{}> .\n".format(ont_a.as_uri()),
    )

    closure = capture_import_closure(str(root), tmp_path / "closure")
    imports = {r.uri: r for r in closure.records if r.kind == "import"}
    # Each unique dependency is stored exactly once
    assert set(imports) == {ont_a.as_uri(), ont_b.as_uri(), ont_c.as_uri()}

    rec_c = imports[ont_c.as_uri()]
    # C is imported by both A and B: one canonical record, duplicate counted
    assert rec_c.duplicate_refs == 1
    assert sorted(rec_c.imported_by) == ["http://example.com/ontA", "http://example.com/ontB"]

    # B's import of A closes a cycle; it is recorded, not followed forever
    rec_a = imports[ont_a.as_uri()]
    assert rec_a.cycle_refs == ["http://example.com/ontB"]
    assert rec_a.verified

    # And the closure still replays cleanly
    replayed, _ = replay_import_closure(tmp_path / "closure")
    assert len(replayed) > 0


def test_unavailable_dependency_is_unverified_not_silently_cached(tmp_path):
    missing = tmp_path / "does-not-exist.ttl"
    root = _write(
        tmp_path / "root.ttl",
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
        "<http://example.com/root> a owl:Ontology ;\n"
        "    owl:imports <{}> .\n".format(missing.as_uri()),
    )
    closure_dir = tmp_path / "closure"
    closure = capture_import_closure(str(root), closure_dir)

    unverified = closure.unverified
    assert len(unverified) == 1
    rec = unverified[0]
    assert rec.uri == missing.as_uri()
    assert rec.status == "unresolved"
    assert rec.error  # the reason is recorded on the manifest

    # Rebuild refuses to proceed silently while a dependency is unattested
    with pytest.raises(ReportableRuntimeError):
        replay_import_closure(closure_dir)

    # Callers may explicitly opt in, and the gap stays visible on the closure
    replayed, loaded = replay_import_closure(closure_dir, allow_unverified=True)
    assert [r.uri for r in loaded.unverified] == [missing.as_uri()]
    assert len(replayed) == len(load_from_source(str(root)))


def test_replay_detects_tampered_or_missing_materials(tmp_path):
    root, ont_a, ont_b = _basic_tree(tmp_path)
    closure_dir = tmp_path / "closure"
    closure = capture_import_closure(str(root), closure_dir)
    rec_a = next(r for r in closure.records if r.uri == ont_a.as_uri())

    content_path = closure_dir / rec_a.content_file
    original = content_path.read_bytes()
    try:
        content_path.write_bytes(b"# replaced upstream, no longer the captured material\n")
        with pytest.raises(ReportableRuntimeError):
            replay_import_closure(closure_dir)
    finally:
        content_path.write_bytes(original)

    content_path.unlink()
    with pytest.raises(ReportableRuntimeError):
        replay_import_closure(closure_dir)


def test_manifest_is_stable_json_with_version(tmp_path):
    root, ont_a, ont_b = _basic_tree(tmp_path)
    closure_dir = tmp_path / "closure"
    capture_import_closure(str(root), closure_dir)
    with open(closure_dir / MANIFEST_FILENAME, 'r', encoding='utf-8') as fh:
        manifest = json.load(fh)
    assert manifest['manifest_version'] == 1
    assert manifest['created_at']
    assert manifest['root_uri']
    assert isinstance(manifest['records'], list)
    # Round-trips through ImportClosure.load
    loaded = ImportClosure.load(closure_dir)
    assert loaded.to_dict() == manifest
