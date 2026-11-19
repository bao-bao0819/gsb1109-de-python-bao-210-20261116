# -*- coding: utf-8 -*-
"""
Cross-shard SHACL validation.

Validate ``sh:minCount``, ``sh:maxCount`` and ``sh:uniqueLang`` constraints
on direct property paths across a set of versioned data shards, without
assembling the merged source graph. See
:class:`pyshacl.sharded.coordinator.ShardCoordinator` for the entry point.
"""

from pyshacl.errors import UnsupportedShapeError
from pyshacl.sharded.coordinator import (
    ShardCoordinator,
    ShardedValidationResult,
    ViolationEvidence,
)
from pyshacl.sharded.identity import (
    BNodeConvention,
    default_bnode_convention,
    identity_digest,
    term_from_identity,
    term_identity,
)
from pyshacl.sharded.shard import LocalShard, ShardProtocol, TargetSummary


__all__ = [
    'BNodeConvention',
    'LocalShard',
    'ShardCoordinator',
    'ShardProtocol',
    'ShardedValidationResult',
    'TargetSummary',
    'UnsupportedShapeError',
    'ViolationEvidence',
    'default_bnode_convention',
    'identity_digest',
    'term_from_identity',
    'term_identity',
]
