#!/bin/python

#    This file is part of python-registry.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""
Compatibility import for :mod:`Registry.LayeredConfig`.

The implementation lives in ``Registry.LayeredConfig``; this module simply
re-exports its public names so callers may use either import path.
"""
from __future__ import absolute_import

from .LayeredConfig import (  # noqa: F401
    DELETE,
    EMPTY,
    POLICY,
    SYSTEM,
    USER,
    Conflict,
    ConfigEntry,
    DeleteTree,
    EffectiveConfig,
    EffectiveValue,
    EntryKind,
    Layer,
    LayeredConfig,
    Source,
    State,
    compute_effective_config,
    resolve_config,
    resolve_layers,
)

Config = LayeredConfig
ConfigResolver = LayeredConfig
EffectiveConfigResolver = LayeredConfig
resolve_effective_config = compute_effective_config

__all__ = [
    "Layer",
    "State",
    "EntryKind",
    "SYSTEM",
    "USER",
    "POLICY",
    "DELETE",
    "EMPTY",
    "DeleteTree",
    "ConfigEntry",
    "Source",
    "EffectiveValue",
    "Conflict",
    "EffectiveConfig",
    "LayeredConfig",
    "Config",
    "ConfigResolver",
    "EffectiveConfigResolver",
    "compute_effective_config",
    "resolve_effective_config",
    "resolve_config",
    "resolve_layers",
]
