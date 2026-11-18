#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backward and forward compatible import location for fragment matching.

The implementation lives in :mod:`pymzml.fragment`; this module re-exports it
so that ``from pymzml.fragment_match import match_fragments`` also works.
"""

from pymzml.fragment import (
    Assignment,
    CandidateMatch,
    FragmentCandidate,
    FragmentMatchResult,
    FragmentMatcher,
    GlobalAssignment,
    GlobalExplanation,
    Ion,
    MatchResult,
    NeutralLoss,
    Peak,
    match_candidates,
    match_fragments,
    match_spectrum,
)

__all__ = [
    "Assignment",
    "CandidateMatch",
    "FragmentCandidate",
    "FragmentMatcher",
    "FragmentMatchResult",
    "GlobalAssignment",
    "GlobalExplanation",
    "Ion",
    "MatchResult",
    "NeutralLoss",
    "Peak",
    "match_candidates",
    "match_fragments",
    "match_spectrum",
]
