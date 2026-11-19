# -*- coding: latin-1 -*-
#
from .entrypoints import shacl_rules, validate, validate_each
from .rule_expand_runner import RuleExpandRunner
from .rules import RuleClosureReport, apply_rules_closure
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
    'RuleClosureReport',
    'apply_rules_closure',
    '__version__',
    'Shape',
    'ShapesGraph',
]
