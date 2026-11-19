# -*- coding: utf-8 -*-
"""Multi-tenant SHACL validation example.

Scenario: two tenants share an RDF dataset and a shape library. The library
holds public shapes plus tenant-specific shapes, and a shared dictionary is
versioned. Each tenant may only see its own materials, the same focus URI may
exist in both tenants without meaning the same entity, and every report must
be reproducible from that tenant's authorized materials alone.

Run with:  python examples/multitenant_validation.py
"""
from rdflib import URIRef

from pyshacl.multitenant import (
    MaterialManifest,
    TenantValidationRunner,
    TenantValidationSpec,
)

PUBLIC_SHAPES = """\
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.com/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

ex:ThingShape a sh:NodeShape ;
    sh:targetClass ex:Thing ;
    sh:property [ sh:path ex:prop ; sh:datatype xsd:string ; sh:minCount 1 ] .
"""

TENANT_A_SHAPES = """\
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.com/tenant-a/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

ex:CodeShape a sh:NodeShape ;
    sh:targetClass ex:Record ;
    sh:property [ sh:path ex:code ; sh:datatype xsd:string ; sh:minCount 1 ] .
"""

TENANT_B_SHAPES = """\
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.com/tenant-b/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

ex:CodeShape a sh:NodeShape ;
    sh:targetClass ex:Record ;
    sh:property [ sh:path ex:code ; sh:datatype xsd:integer ; sh:minCount 1 ] .
"""

SHARED_DICTIONARY_V3 = """\
@prefix ex: <http://example.com/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

ex:Record rdfs:subClassOf ex:Thing .
ex:alpha rdfs:label "alpha-v3" .
"""

# The exact same focus URI appears in both tenants, with different triples.
TENANT_A_DATA = """\
@prefix ex: <http://example.com/> .

ex:shared-focus a ex:Record ; ex:code "A-001" ; ex:prop "ok" .
"""

TENANT_B_DATA = """\
@prefix ex: <http://example.com/> .
@prefix tb: <http://example.com/tenant-b/> .

ex:shared-focus a tb:Record ; tb:code "B-not-an-integer" .
"""

FOCUS_URI = URIRef("http://example.com/shared-focus")


def build_tenant(tenant_id, tenant_shapes, data):
    manifest = MaterialManifest(tenant_id)
    manifest.add_public_shapes("public-shapes", PUBLIC_SHAPES, version="2026.11")
    manifest.add_tenant_shapes("tenant-shapes", tenant_shapes)
    manifest.add_shared_dictionary("shared-dict", SHARED_DICTIONARY_V3, version="3.0.0")
    manifest.add_data_graph("tenant-data", data)
    return TenantValidationRunner(manifest)


def main():
    runner_a = build_tenant("tenant-a", TENANT_A_SHAPES, TENANT_A_DATA)
    runner_b = build_tenant("tenant-b", TENANT_B_SHAPES, TENANT_B_DATA)

    spec = TenantValidationSpec(
        data_graph_ids=["tenant-data"],
        focus_nodes=[FOCUS_URI],
        dictionary_versions={"shared-dict": "3.0.0"},
        inference="rdfs",
    )

    report_a = runner_a.validate(spec)
    report_b = runner_b.validate(spec)

    print("tenant-a conforms:", report_a.conforms)
    print("tenant-b conforms:", report_b.conforms)
    print("same focus URI, visible graphs differ:")
    print("  tenant-a:", report_a.focus_node_visibility)
    print("  tenant-b:", report_b.focus_node_visibility)
    print("tenant-a report only names its own materials:",
          sorted(report_a.used_material_ids["shapes"]))

    # Batch validation: every task gets fresh, disjoint graph clones.
    batch = runner_a.validate_batch([spec, spec, spec], max_workers=4)
    print("batch size:", len(batch), "all conform:", all(r.conforms for r in batch))

    # Reproduce the report from tenant-a's materials alone.
    replayed = runner_a.replay(report_a)
    print("replay matches original fingerprints:",
          replayed.input_fingerprints == report_a.input_fingerprints)


if __name__ == "__main__":
    main()
