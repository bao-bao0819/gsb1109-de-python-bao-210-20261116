.. pymzml_fragment:

Fragment matching
=================

.. automodule:: pymzml.fragment

The matcher compares a measured MS2 spectrum against one or more candidate
molecules. Every candidate contributes its theoretical fragments and, when
requested, every allowed neutral loss. Matching is solved as a weighted
bipartite assignment, not as a greedy peak-by-peak walk.

Rules guaranteed by the implementation:

* A measured peak scores a candidate at most once, even when several
  theoretical ions of that candidate fall into the same tolerance window.
* Candidates are never favoured because they were supplied first. With
  ``peak_usage="exclusive"`` all candidates are solved together and every
  globally optimal explanation is kept, so a peak that fits two candidates is
  reported for both.
* Measured peaks that fit no theoretical ion remain in
  :attr:`pymzml.fragment.MatchResult.unmatched_peaks`.
* Equally good assignments are retained as
  :attr:`pymzml.fragment.CandidateMatch.explanations` instead of being
  collapsed into an arbitrary winner.

Basic usage
-----------

::

    import pymzml
    from pymzml.fragment import match_fragments

    run = pymzml.run.Reader("tests/data/BSA1.mzML.gz", MS_precisions={2: 20e-6})
    spectrum = run[3542]

    candidates = [
        {
            "name": "HLVDEPQNLIK",
            "fragments": {"b2": 251.1503, "y2": 260.1969},
            "neutral_losses": [("H2O", 18.0106)],
        },
        {"name": "decoy", "fragments": {"b2": 311.1714}},
    ]
    result = match_fragments(
        spectrum,
        candidates,
        tolerance=20,
        tolerance_unit="ppm",
        weights={"HLVDEPQNLIK": {"b2": 1.0, "y2": 2.0}},
        peak_usage="exclusive",
    )
    for candidate in result.ranking:
        print(candidate.name, candidate.score, candidate.n_explanations)
    print(result.unmatched_peaks)

API
---

.. autoclass:: pymzml.fragment.MatchResult
    :members:

.. autoclass:: pymzml.fragment.CandidateMatch
    :members:

.. autoclass:: pymzml.fragment.Assignment
    :members:

.. autoclass:: pymzml.fragment.GlobalExplanation
    :members:

.. autoclass:: pymzml.fragment.FragmentMatcher
    :members:

.. autofunction:: pymzml.fragment.match_fragments
