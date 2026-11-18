Classification cleaning
=======================

This module cleans point classifications without destroying the classification
that came from the device.  It keeps a separate, versioned result and records
why every point changed.

Why a separate result
---------------------

The standard ``las.classification`` dimension is treated as the device
original and is never rewritten by this module.  The cleaned class is stored
either:

- in an extra dimension (``cleaned_classification`` by default) through
  :meth:`laspy.cleaning.CleaningResult.apply_to_las`, or
- in your own array by reading
  :attr:`laspy.cleaning.CleaningResult.classification`.

A manual correction is applied as an overlay on top of the original.  The
original is still available for every point, and the correction itself is
recorded as a change with its own reason.

Rules
-----

Rules are grouped in a :class:`laspy.cleaning.ClassificationRuleSet` that has a
version string.  Two rule types are built in:

- :class:`laspy.cleaning.ExtentRule` reclassifies points in an explicit,
  inclusive, axis-aligned extent.
- :class:`laspy.cleaning.NeighborRule` reclassifies a point when enough of its
  neighbours within a radius have the requested classes.  With the default
  ``neighbor_source="current"`` a correction propagates through as many layers
  of neighbours as needed until the result stops changing.

When several rules propose a different class for the same point, the rule with
the highest ``priority`` wins.  Ties are broken by ``rule_id``, so the order in
which rules were declared never changes the result.

Determinism and convergence
---------------------------

Every round evaluates a snapshot and applies all of its changes at once.  A
repeated state or the iteration limit is reported as ``cycle`` or
``max_iterations``.  In that case ``converged`` is ``False``, no version is
published, and :meth:`~laspy.cleaning.CleaningResult.apply_to_las` refuses to
write anything.

Only affected points are recomputed.  After the first round the engine
restricts work to changed points and their neighbourhood.  A
:class:`laspy.cleaning.ClassificationSession` does the same for manual
corrections: it rebuilds the neighbourhood closure of the changed seeds and
keeps independent clusters untouched.

Provenance
----------

Each run produces a :class:`laspy.cleaning.ClassificationVersion`.  It stores
the original and result digests, the rule-set version, the effective
corrections, and one :class:`laspy.cleaning.ClassificationChange` per decision
with the old class, new class, rule id and human-readable reason.  The version
id is content-addressed, so recomputing the same inputs yields the same id.

The version id identifies the run, including its parent.  Different update
orders therefore keep different lineage rows, but they share the same
``published_id`` when they converge to the same rule version and final
classification.  The published content, rather than the update order,
determines what a consumer should treat as the final result.

:class:`laspy.cleaning.ClassificationLedger` is an append-only history.
:func:`laspy.cleaning.store_classification_history` and
:func:`laspy.cleaning.load_classification_history` persist it in LAS VLRs, so
the original classification and every version can be inspected later.

Example
-------

.. code-block:: python

    import laspy

    las = laspy.read("input.las")
    rules = laspy.ClassificationRuleSet(
        "2026-11-18",
        [
            laspy.ExtentRule(
                "yard",
                2,
                extent=(xmin, xmax, ymin, ymax),
                match_classes=[1],
            ),
            laspy.NeighborRule(
                "fill-ground",
                2,
                radius=0.5,
                neighbor_classes=[2],
                min_count=2,
            ),
        ],
    )
    result = laspy.clean_classification(las, rules)
    result.raise_for_status()
    print(result.summary())
    for change in result.rule_changes[:10]:
        print(change.point_index, change.reason)
    result.apply_to_las(las)
    las.write("cleaned.las")
