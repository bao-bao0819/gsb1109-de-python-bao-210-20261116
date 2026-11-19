# -*- coding: utf-8 -*-
"""Errors raised by the multi-tenant validation layer."""


class TenantValidationError(RuntimeError):
    """Base class for failures of a tenant-scoped validation run."""


class TenantScopeError(TenantValidationError):
    """Raised when a request references materials outside the tenant's manifest.

    This is an authorization failure: the tenant is not allowed to see the
    requested graph, shape, dictionary, focus node or ontology. The offending
    reference is kept in :attr:`reference` and the tenant id in :attr:`tenant_id`
    so callers can audit the attempt without exposing other tenants' data.
    """

    def __init__(self, message: str, tenant_id: str, reference: str) -> None:
        self.tenant_id = tenant_id
        self.reference = reference
        super().__init__(message)

    def __str__(self) -> str:
        return "{} (tenant={!r}, reference={!r})".format(
            super().__str__(), self.tenant_id, self.reference
        )
