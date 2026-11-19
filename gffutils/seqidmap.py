"""
Resolve ambiguous annotation seqids against one or more reference catalogs.

When the same annotation seqid has several candidate reference sequences
(e.g. similarly-named sequences in two assembly directories), a name match
alone is not enough to pick one.  This module combines, for every catalog:

* sequence lengths (required),
* declared summary regions (e.g. ``##sequence-region`` directives),
* explicit aliases (source seqid -> reference name),
* parent edges that are allowed to cross references,

with the feature ranges and parent/child relations of the annotation
records, and computes -- for every source seqid -- the set of candidate
references it can still correspond to in *some* globally consistent
interpretation.  Records related through parent/child links are forced to
use a single consistent mapping; candidates may not be cherry-picked per
line.  When no consistent interpretation exists, the conflicting evidence
is reported.  Only uniquely provable mappings are exposed through
:meth:`ResolutionResult.unified_index`; everything else keeps its original
seqid and ranges.
"""

from collections import defaultdict
from collections import namedtuple


class Candidate(namedtuple("Candidate", ["catalog", "reference"])):
    """
    A reference sequence in a catalog that a source seqid may map to.
    """

    __slots__ = ()

    def __str__(self):
        return "%s:%s" % (self.catalog, self.reference)


class SourceRecord(namedtuple("SourceRecord", ["seqid", "start", "end",
                                               "feature_id", "parents"])):
    """
    One annotation record to be mapped.

    Parameters
    ----------
    seqid : str
        Source seqid as it appears in the annotation.

    start, end : int
        1-based feature coordinates on the source seqid.

    feature_id : str or None
        ID of the feature, used to resolve parent/child relations.

    parents : iterable of str
        IDs of parent features.
    """

    __slots__ = ()

    def __new__(cls, seqid, start, end, feature_id=None, parents=()):
        return super(SourceRecord, cls).__new__(
            cls, seqid, int(start), int(end), feature_id, tuple(parents)
        )


class _ParentConstraint(
    namedtuple(
        "_ParentConstraint",
        ["parent_seqid", "child_seqid", "parent_feature_id",
         "child_feature_id"],
    )
):
    __slots__ = ()

    def other_side(self, seqid):
        if seqid == self.parent_seqid:
            return self.child_seqid
        return self.parent_seqid


class ReferenceCatalog(object):
    """
    Known facts about one assembly directory.

    Parameters
    ----------
    name : str
        Identifier for this catalog (e.g. the assembly directory).

    lengths : dict
        Mapping of reference name -> sequence length.  Required; defines
        the reference sequences that exist in this catalog.

    aliases : dict
        Explicit mapping of source seqid -> reference name.  If a source
        seqid has an alias in this catalog, only the aliased reference is
        considered for that seqid within this catalog.

    summaries : dict
        Mapping of reference name -> (start, end) declared region (e.g.
        from ``##sequence-region`` directives).  Features mapped to such a
        reference must fall inside the declared region.

    parent_edges : iterable of (parent_reference, child_reference)
        Parent/child relations that are allowed to cross references within
        this catalog.  Without a declared edge, a child feature must map
        to the same reference as its parent.
    """

    def __init__(self, name, lengths, aliases=None, summaries=None,
                 parent_edges=None):
        self.name = name
        self.lengths = dict(lengths)
        self.aliases = dict(aliases or {})
        self.summaries = {}
        for ref, region in (summaries or {}).items():
            start, end = region
            self.summaries[ref] = (int(start), int(end))
        self.parent_edges = set()
        for edge in parent_edges or ():
            parent_ref, child_ref = edge
            self.parent_edges.add((parent_ref, child_ref))

    def range_violation(self, reference, min_start, max_end):
        """
        Return None if a source span [min_start, max_end] fits on
        `reference`, otherwise a human-readable reason.
        """
        length = self.lengths[reference]
        if max_end > length:
            return (
                "source range end %d exceeds length %d of reference %r"
                % (max_end, length, reference)
            )
        region = self.summaries.get(reference)
        if region is not None:
            lo, hi = region
            if min_start < lo or max_end > hi:
                return (
                    "source range %d-%d falls outside declared region "
                    "%d-%d of reference %r"
                    % (min_start, max_end, lo, hi, reference)
                )
        return None


class MappingConflict(object):
    """
    Evidence collected for a group of seqids that cannot be mapped
    consistently.

    Attributes
    ----------
    seqids : list of str
        Source seqids of the connected component with no legal solution.

    domains : dict
        Surviving candidates per seqid after unary (range/alias/summary)
        checks.

    eliminations : dict
        seqid -> list of (Candidate, reason) for candidates removed by
        unary checks.

    constraints : list of _ParentConstraint
        Parent/child relations linking the seqids.

    inconsistent_edges : list of _ParentConstraint
        Parent/child relations for which no pair of surviving candidates
        is compatible.
    """

    def __init__(self, seqids, domains, eliminations, constraints,
                 inconsistent_edges):
        self.seqids = sorted(seqids)
        self.domains = domains
        self.eliminations = eliminations
        self.constraints = list(constraints)
        self.inconsistent_edges = list(inconsistent_edges)

    def __str__(self):
        lines = [
            "no consistent mapping for seqid(s): %s" % ", ".join(self.seqids)
        ]
        for seqid in self.seqids:
            surviving = self.domains.get(seqid, [])
            if surviving:
                lines.append(
                    "  %s: surviving candidates: %s"
                    % (seqid, ", ".join(str(c) for c in surviving))
                )
            else:
                lines.append("  %s: no surviving candidates" % seqid)
            for cand, reason in self.eliminations.get(seqid, []):
                lines.append("    eliminated %s: %s" % (cand, reason))
        for edge in self.inconsistent_edges:
            lines.append(
                "  unsatisfiable parent relation: feature %r (seqid %s) -> "
                "feature %r (seqid %s): no compatible candidate pair"
                % (
                    edge.parent_feature_id,
                    edge.parent_seqid,
                    edge.child_feature_id,
                    edge.child_seqid,
                )
            )
        return "\n".join(lines)


class ResolutionResult(object):
    """
    Outcome of :meth:`SeqidResolver.resolve`.

    Attributes
    ----------
    possible : dict
        seqid -> frozenset of Candidate; every reference the seqid can
        still correspond to in some globally consistent interpretation.

    conflicts : list of MappingConflict
        Evidence for groups of seqids with no consistent interpretation.

    unresolved_parents : list of (child_feature_id, parent_id)
        Parent references pointing at feature IDs not present in the
        input records; these impose no constraint.
    """

    def __init__(self, possible, conflicts, records, unresolved_parents):
        self.possible = possible
        self.conflicts = list(conflicts)
        self.records = list(records)
        self.unresolved_parents = list(unresolved_parents)

    @property
    def unique(self):
        """
        dict of seqid -> Candidate for seqids with exactly one possible
        mapping across all legal interpretations.
        """
        return {
            seqid: next(iter(cands))
            for seqid, cands in self.possible.items()
            if len(cands) == 1
        }

    @property
    def ambiguous(self):
        """
        dict of seqid -> frozenset of Candidate for seqids with more than
        one possible mapping.
        """
        return {
            seqid: cands
            for seqid, cands in self.possible.items()
            if len(cands) > 1
        }

    def unified_index(self, records=None):
        """
        Build a unified coordinate index over `records` (defaults to the
        records the resolver was run on).

        Returns a dict whose keys are :class:`Candidate` tuples for
        uniquely resolved seqids -- only these enter the unified
        coordinate system -- and plain source seqid strings for everything
        else.  Values are lists of :class:`SourceRecord` with their
        original coordinates, unchanged.
        """
        if records is None:
            records = self.records
        unique = self.unique
        index = {}
        for rec in records:
            target = unique.get(rec.seqid)
            key = target if target is not None else rec.seqid
            index.setdefault(key, []).append(rec)
        return index


class SeqidResolver(object):
    """
    Jointly resolve source seqids against one or more reference catalogs.

    Parameters
    ----------
    catalogs : iterable of ReferenceCatalog, or dict of name -> ReferenceCatalog
        The candidate assembly directories.
    """

    def __init__(self, catalogs):
        self.catalogs = {}
        if isinstance(catalogs, dict):
            items = catalogs.values()
        else:
            items = catalogs
        for catalog in items:
            self.catalogs[catalog.name] = catalog
        if not self.catalogs:
            raise ValueError("at least one reference catalog is required")

    def resolve(self, records):
        """
        Compute, for every source seqid in `records`, the set of candidate
        references consistent with all legal interpretations.

        Parameters
        ----------
        records : iterable of SourceRecord

        Returns
        -------
        ResolutionResult
        """
        records = list(records)
        stats = self._seqid_stats(records)
        domains, eliminations = self._domains(stats)
        constraints, unresolved_parents = self._constraints(records)
        components = self._components(sorted(stats), constraints)

        possible = {}
        conflicts = []
        for component in components:
            comp_domains = {s: domains[s] for s in component}
            comp_constraints = [
                c for c in constraints if c.parent_seqid in comp_domains
            ]
            solutions = self._solve(comp_domains, comp_constraints)
            if not solutions:
                conflicts.append(
                    self._conflict(
                        component, comp_domains, eliminations,
                        comp_constraints,
                    )
                )
                continue
            for seqid in component:
                possible[seqid] = frozenset(
                    solution[seqid] for solution in solutions
                )
        return ResolutionResult(possible, conflicts, records,
                                unresolved_parents)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _seqid_stats(records):
        """
        Group records by seqid, tracking the coordinate extremes and the
        records responsible for them (used as conflict evidence).
        """
        stats = {}
        for rec in records:
            entry = stats.get(rec.seqid)
            if entry is None:
                entry = stats[rec.seqid] = {
                    "min_start": rec.start,
                    "max_end": rec.end,
                    "min_record": rec,
                    "max_record": rec,
                    "records": [],
                }
            entry["records"].append(rec)
            if rec.start < entry["min_start"]:
                entry["min_start"] = rec.start
                entry["min_record"] = rec
            if rec.end > entry["max_end"]:
                entry["max_end"] = rec.end
                entry["max_record"] = rec
        return stats

    def _domains(self, stats):
        """
        Apply unary constraints (aliases, lengths, summaries) to compute
        the candidate domain of every seqid, recording why each rejected
        candidate was eliminated.
        """
        domains = {}
        eliminations = {}
        for seqid, info in stats.items():
            cands = []
            elim = []
            for catalog in self.catalogs.values():
                if seqid in catalog.aliases:
                    ref = catalog.aliases[seqid]
                    if ref not in catalog.lengths:
                        elim.append(
                            (
                                Candidate(catalog.name, ref),
                                "explicit alias target %r is not present "
                                "in catalog %r" % (ref, catalog.name),
                            )
                        )
                        continue
                    refs = [ref]
                else:
                    refs = sorted(catalog.lengths)
                for ref in refs:
                    cand = Candidate(catalog.name, ref)
                    reason = catalog.range_violation(
                        ref, info["min_start"], info["max_end"]
                    )
                    if reason is None:
                        cands.append(cand)
                    else:
                        max_rec = info["max_record"]
                        min_rec = info["min_record"]
                        evidence = []
                        if max_rec.feature_id is not None:
                            evidence.append(
                                "record %r end=%d"
                                % (max_rec.feature_id, max_rec.end)
                            )
                        if (
                            min_rec is not max_rec
                            and min_rec.feature_id is not None
                        ):
                            evidence.append(
                                "record %r start=%d"
                                % (min_rec.feature_id, min_rec.start)
                            )
                        if evidence:
                            reason = "%s (%s)" % (reason, "; ".join(evidence))
                        elim.append((cand, reason))
            domains[seqid] = cands
            eliminations[seqid] = elim
        return domains, eliminations

    @staticmethod
    def _constraints(records):
        """
        Build parent/child constraints between records living on different
        seqids.  Records on the same seqid share a mapping by construction.
        """
        by_id = {}
        for rec in records:
            if rec.feature_id is not None:
                by_id[rec.feature_id] = rec
        constraints = []
        unresolved = []
        seen = set()
        for rec in records:
            for parent_id in rec.parents:
                parent = by_id.get(parent_id)
                if parent is None:
                    unresolved.append((rec.feature_id, parent_id))
                    continue
                if parent.seqid == rec.seqid:
                    continue
                key = (parent.seqid, rec.seqid,
                       parent.feature_id, rec.feature_id)
                if key in seen:
                    continue
                seen.add(key)
                constraints.append(
                    _ParentConstraint(
                        parent.seqid, rec.seqid,
                        parent.feature_id, rec.feature_id,
                    )
                )
        return constraints, unresolved

    @staticmethod
    def _components(seqids, constraints):
        """
        Connected components of seqids joined by parent/child constraints.
        """
        root = {s: s for s in seqids}

        def find(s):
            while root[s] != s:
                root[s] = root[root[s]]
                s = root[s]
            return s

        for c in constraints:
            root[find(c.parent_seqid)] = find(c.child_seqid)
        groups = defaultdict(list)
        for s in seqids:
            groups[find(s)].append(s)
        return [sorted(members) for members in groups.values()]

    def _pair_ok(self, constraint, parent_cand, child_cand):
        """
        Check one parent/child constraint for a concrete pair of
        candidates.  Parent and child must land in the same catalog, on
        the same reference, unless the catalog declares a cross-reference
        parent edge.
        """
        if parent_cand.catalog != child_cand.catalog:
            return False
        if parent_cand.reference == child_cand.reference:
            return True
        edge = (parent_cand.reference, child_cand.reference)
        return edge in self.catalogs[parent_cand.catalog].parent_edges

    def _check(self, constraint, seqid, cand, other_cand):
        if constraint.parent_seqid == seqid:
            return self._pair_ok(constraint, cand, other_cand)
        return self._pair_ok(constraint, other_cand, cand)

    def _solve(self, domains, constraints):
        """
        Enumerate all solutions of one connected component using
        backtracking with forward checking.
        """
        adjacency = defaultdict(list)
        for c in constraints:
            adjacency[c.parent_seqid].append(c)
            adjacency[c.child_seqid].append(c)

        order = sorted(domains, key=lambda s: (len(domains[s]), s))
        current = {s: list(d) for s, d in domains.items()}
        assignment = {}
        solutions = []

        def backtrack(index):
            if index == len(order):
                solutions.append(dict(assignment))
                return
            seqid = order[index]
            for cand in list(current[seqid]):
                assignment[seqid] = cand
                pruned = []
                ok = True
                for constraint in adjacency[seqid]:
                    other = constraint.other_side(seqid)
                    if other in assignment:
                        if not self._check(
                            constraint, seqid, cand, assignment[other]
                        ):
                            ok = False
                            break
                        continue
                    kept = [
                        oc
                        for oc in current[other]
                        if self._check(constraint, seqid, cand, oc)
                    ]
                    if len(kept) != len(current[other]):
                        pruned.append((other, current[other]))
                        current[other] = kept
                    if not kept:
                        ok = False
                        break
                if ok:
                    backtrack(index + 1)
                for other, saved in pruned:
                    current[other] = saved
                del assignment[seqid]

        backtrack(0)
        return solutions

    def _conflict(self, component, comp_domains, eliminations,
                  comp_constraints):
        inconsistent = []
        for constraint in comp_constraints:
            found = False
            for parent_cand in comp_domains[constraint.parent_seqid]:
                for child_cand in comp_domains[constraint.child_seqid]:
                    if self._pair_ok(constraint, parent_cand, child_cand):
                        found = True
                        break
                if found:
                    break
            if not found:
                inconsistent.append(constraint)
        return MappingConflict(
            component,
            comp_domains,
            {s: eliminations.get(s, []) for s in component},
            comp_constraints,
            inconsistent,
        )


def records_from_features(features, id_key="ID", parent_key="Parent"):
    """
    Build :class:`SourceRecord` objects from gffutils features (or any
    objects with ``seqid``, ``start``, ``end`` and ``attributes``).

    Parameters
    ----------
    features : iterable of Feature

    id_key, parent_key : str
        Attribute keys holding the feature ID and parent IDs.
    """
    records = []
    for f in features:
        ids = f.attributes.get(id_key, [])
        feature_id = ids[0] if ids else None
        parents = tuple(f.attributes.get(parent_key, []))
        records.append(
            SourceRecord(f.seqid, f.start, f.end, feature_id, parents)
        )
    return records
