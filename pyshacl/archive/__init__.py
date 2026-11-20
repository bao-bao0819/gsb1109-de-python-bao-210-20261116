# -*- coding: utf-8 -*-
"""Offline archive and replay of RDF/SHACL validation runs.

This package captures every document loaded during a validation (including
``owl:imports`` chains), stores identical bytes once in a content-addressed
layout, and re-runs the same validation fully offline against the archived
base IRIs so relative references and import cycles reproduce identically.
"""
from .archive import CaptureResult, capture_validation, replay_comparison
from .replay import (
    IncompleteReplayError,
    ReplayResult,
    detect_import_cycles,
    find_missing_imports,
    replay_validation,
)
from .report import QualityReport, Violation, extract_report, reports_equivalent
from .store import ArchiveStore, Manifest

__all__ = [
    "capture_validation",
    "replay_validation",
    "replay_comparison",
    "detect_import_cycles",
    "find_missing_imports",
    "extract_report",
    "reports_equivalent",
    "CaptureResult",
    "ReplayResult",
    "IncompleteReplayError",
    "QualityReport",
    "Violation",
    "ArchiveStore",
    "Manifest",
]
