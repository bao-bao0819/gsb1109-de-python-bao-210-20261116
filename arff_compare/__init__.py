"""Stratified class-share comparison for two weighted ARFF periods."""

from .core import (
    ClassMapping,
    ComparisonError,
    compare,
    compare_files,
    load_dataset,
)

__all__ = [
    "ClassMapping",
    "ComparisonError",
    "compare",
    "compare_files",
    "load_dataset",
]
__version__ = "1.0.0"
