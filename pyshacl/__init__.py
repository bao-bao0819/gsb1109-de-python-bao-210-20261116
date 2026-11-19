# -*- coding: latin-1 -*-
#
from .entrypoints import shacl_rules, validate, validate_each
from .release_gate import (
    BlockingEvidence,
    Exemption,
    ReleaseDecision,
    ReleaseGate,
    Severity,
    build_release_gate,
)
from .rule_expand_runner import RuleExpandRunner
from .shape import Shape
from .shapes_graph import ShapesGraph
from .validator import Validator

# version compliant with https://www.python.org/dev/peps/pep-0440/
__version__ = '0.40.2'
# Don't forget to change the version number in pyproject.toml, Dockerfile, and CITATION.cff along with this one

__all__ = [
    'validate',
    'validate_each',
    'shacl_rules',
    'Validator',
    'RuleExpandRunner',
    '__version__',
    'Shape',
    'ShapesGraph',
    'ReleaseGate',
    'ReleaseDecision',
    'BlockingEvidence',
    'Exemption',
    'Severity',
    'build_release_gate',
]
