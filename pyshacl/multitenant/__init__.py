# -*- coding: utf-8 -*-
"""Multi-tenant SHACL validation support.

This package provides tenant-scoped validation on top of :func:`pyshacl.validate`:

* :class:`MaterialManifest` holds the per-tenant materials a tenant is allowed to
  see: tenant-visible data graphs, public and tenant-specific shapes, and
  versioned shared/tenant dictionaries.
* :class:`TenantValidationRunner` performs every validation run against fresh,
  disjoint clones of those materials, so concurrent or failed neighbouring tasks
  can never share shapes or inference state.
* :class:`TenantValidationReport` records the material fingerprints and the full
  validation spec used, allowing a report to be reproduced using only the
  tenant's own authorized materials.
"""
from .errors import TenantScopeError, TenantValidationError
from .manifest import (
    MaterialManifest,
    TenantMaterial,
    TenantValidationReport,
    TenantValidationSpec,
    graph_fingerprint,
)
from .runner import TenantValidationRunner

__all__ = [
    'MaterialManifest',
    'TenantMaterial',
    'TenantValidationReport',
    'TenantValidationRunner',
    'TenantValidationSpec',
    'TenantScopeError',
    'TenantValidationError',
    'graph_fingerprint',
]
