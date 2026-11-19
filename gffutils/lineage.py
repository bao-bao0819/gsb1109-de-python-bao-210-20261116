"""
Identity-preserving parent edges and coverage accounting.

A GFF3 feature may carry multiple ``Parent=`` values (for example an exon
shared by two transcripts). That feature is still a *single* source line and
a single row in the ``features`` table; the ``relations`` table stores one
row per parent edge. The objects in this module keep that distinction
explicit:

- :class:`FeatureLineage` pairs one feature identity with every stored
  parent edge, together with a resolution status. Shared parentage is
  therefore represented as several edges of one feature rather than several
  copies of the feature.

- :class:`CoverageResult` / :class:`CoverageReport` provide the two
  supported coverage conventions:

    * ``"transcript"`` scope counts each parent edge (per-transcript
      statistics, where a shared exon contributes once to each transcript);
    * ``"gene"`` scope takes the union of feature identities / coordinates
      (joint gene coverage, where a shared exon contributes only once).

Parentage that is missing or conflicting is never repaired by silently
picking a parent: the lineage stays unresolved and is handed back to the
caller via the report's ``unresolved`` list (or raises
:class:`gffutils.exceptions.ParentResolutionError` when requested).
"""

import collections
import warnings


# A feature whose parent edges all resolve to features within a single gene.
RESOLVED = "resolved"

# At least one direct parent edge points at an ID that is not present as a
# feature. A feature with no parent edges at all is reported RESOLVED by
# FeatureDB.lineage() (it is a valid root), but FeatureDB.coverage() parks it
# as pending under this status because it cannot be attached to any group.
MISSING_PARENT = "missing_parent"

# Parent edges resolve to features belonging to different genes, so no single
# parent can be chosen without inventing information.
CONFLICTING_PARENTS = "conflicting_parents"

_UNRESOLVED_STATUSES = (MISSING_PARENT, CONFLICTING_PARENTS)

ParentEdge = collections.namedtuple("ParentEdge", ["parent", "level"])


class FeatureLineage(object):
    """
    One feature identity together with every parent edge stored for it.

    Attributes
    ----------
    feature : gffutils.feature.Feature
        The single database feature these edges belong to.

    parent_edges : list of ParentEdge
        Direct parent edges (``relations.level == 1``), one per value of the
        feature's GFF ``Parent`` attribute / GTF parent key. A feature shared
        by two parents therefore has two edges but still references one
        feature.

    ancestor_edges : list of ParentEdge
        Every edge row stored for the feature, including the transitive
        closure (e.g. the level-2 gene above a transcript). Kept for full
        fidelity; grouping and parent identity use the direct
        :attr:`parent_edges`.

    status : str
        One of :data:`RESOLVED`, :data:`MISSING_PARENT`, or
        :data:`CONFLICTING_PARENTS`.

    missing_parent_ids : list of str
        Parent IDs referenced by a direct edge but absent from the database.

    conflicting_gene_ids : list of str
        The distinct gene IDs the parent edges resolve to when more than one
        gene is implicated. Empty otherwise.
    """

    def __init__(
        self,
        feature,
        parent_edges,
        status=RESOLVED,
        missing_parent_ids=None,
        conflicting_gene_ids=None,
        ancestor_edges=None,
    ):
        self.feature = feature
        self.feature_id = feature.id
        self.parent_edges = list(parent_edges)
        # Default to the direct edges when no explicit closure is supplied.
        self.ancestor_edges = (
            list(ancestor_edges) if ancestor_edges is not None else list(parent_edges)
        )
        self.status = status
        self.missing_parent_ids = list(missing_parent_ids or [])
        self.conflicting_gene_ids = list(conflicting_gene_ids or [])

    @property
    def parent_ids(self):
        """Distinct direct parent IDs, in the order the edges were encountered."""
        seen = set()
        ordered = []
        for edge in self.parent_edges:
            if edge.parent not in seen:
                seen.add(edge.parent)
                ordered.append(edge.parent)
        return ordered

    @property
    def ancestor_ids(self):
        """Distinct IDs across every stored edge, including transitive ones."""
        seen = set()
        ordered = []
        for edge in self.ancestor_edges:
            if edge.parent not in seen:
                seen.add(edge.parent)
                ordered.append(edge.parent)
        return ordered

    @property
    def is_resolved(self):
        """True when no parent edge is missing or conflicting."""
        return self.status not in _UNRESOLVED_STATUSES

    def __repr__(self):
        return (
            "<FeatureLineage {fid} status={status} parents={parents}>".format(
                fid=self.feature_id,
                status=self.status,
                parents=self.parent_ids,
            )
        )


class CoverageResult(object):
    """
    Coverage computed for one grouping key (a transcript or a gene).

    Attributes
    ----------
    key : str
        ID of the transcript (``scope="transcript"``) or gene
        (``scope="gene"``) the coverage was computed for.

    scope : str
        ``"transcript"`` or ``"gene"``.

    bp : int
        Number of covered base pairs. In transcript scope a shared exon
        contributes to each transcript; in gene scope feature identities are
        unioned so a shared exon contributes once.

    features : list of gffutils.feature.Feature
        Distinct child features contributing to this result, each appearing
        once regardless of how many parent edges reach it.

    lineages : list of FeatureLineage
        Resolved lineages of the contributing features; inspect these for
        the individual parent edges.
    """

    def __init__(self, key, scope, bp, features, lineages):
        self.key = key
        self.scope = scope
        self.bp = bp
        self.features = list(features)
        self.lineages = list(lineages)

    def __repr__(self):
        return "<CoverageResult {scope}:{key} bp={bp}>".format(
            scope=self.scope, key=self.key, bp=self.bp
        )

    def __eq__(self, other):
        if not isinstance(other, CoverageResult):
            return NotImplemented
        return (
            self.key == other.key
            and self.scope == other.scope
            and self.bp == other.bp
            and [f.id for f in self.features]
            == [f.id for f in other.features]
        )

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result


class CoverageReport(object):
    """
    A set of :class:`CoverageResult` objects plus any unresolved lineages.

    Behaves as a read-only mapping of grouping-key ID to
    :class:`CoverageResult`. Lineages with missing or conflicting parentage
    are never forced into a result group; they are collected in
    :attr:`unresolved` so that no parent is ever picked on the caller's
    behalf.
    """

    def __init__(self, scope, results, unresolved):
        self.scope = scope
        if isinstance(results, dict):
            self.results = dict(results)
        else:
            self.results = {result.key: result for result in results}
        self.unresolved = list(unresolved)

    @property
    def total_bp(self):
        """Sum of ``bp`` across all results in this report's scope."""
        return sum(result.bp for result in self.results.values())

    def __getitem__(self, key):
        return self.results[key]

    def __contains__(self, key):
        return key in self.results

    def __iter__(self):
        return iter(self.results)

    def __len__(self):
        return len(self.results)

    def keys(self):
        return self.results.keys()

    def items(self):
        return self.results.items()

    def values(self):
        return self.results.values()

    def get(self, key, default=None):
        return self.results.get(key, default)

    def __repr__(self):
        return (
            "<CoverageReport scope={scope} groups={n} unresolved={u}>".format(
                scope=self.scope, n=len(self.results), u=len(self.unresolved)
            )
        )


def check_on_unresolved(on_unresolved, unresolved):
    """
    Apply the requested policy to a list of unresolved lineages.

    ``"keep"`` returns the list untouched, ``"warn"`` additionally emits a
    warning, and ``"error"`` raises
    :class:`gffutils.exceptions.ParentResolutionError`.
    """
    if on_unresolved not in ("keep", "warn", "error"):
        raise ValueError(
            "on_unresolved must be one of 'keep', 'warn', 'error'; got %r"
            % (on_unresolved,)
        )
    if not unresolved:
        return unresolved
    if on_unresolved == "error":
        from gffutils.exceptions import ParentResolutionError

        raise ParentResolutionError(unresolved)
    if on_unresolved == "warn":
        ids = ", ".join(sorted(lineage.feature_id for lineage in unresolved))
        warnings.warn(
            "Unresolved parentage retained for feature(s): %s" % ids,
            stacklevel=2,
        )
    return unresolved
