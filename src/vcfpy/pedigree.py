# -*- coding: utf-8 -*-
"""Pedigree consistency screening

This module screens genotype calls for consistency with a pedigree and an
inheritance mode.  It distinguishes three outcomes instead of reducing
everything to a boolean:

:py:data:`CONSISTENT`
    **Provably consistent.**  Every completion of the missing alleles is
    compatible with the pedigree.  For complete calls this simply means the
    observed genotypes are compatible.

:py:data:`CONFLICT`
    **Provably conflicting.**  No completion of the missing alleles is
    compatible with the pedigree.

:py:data:`INSUFFICIENT`
    **Insufficient evidence.**  Some completions are compatible and some are
    not, or the calls do not contain enough information to decide.

The two failure modes called out in the task are therefore avoided by
construction:

- A missing allele is never reported as a conflict merely because it is
  missing.  Missing values are completed from the known allele set and the
  verdict is quantified over all completions.
- A single compatible completion is never reported as proven consistency.
  Consistency is only claimed when *all* completions are compatible.

Every result keeps the conditions under which it holds in
:py:attr:`ScreeningResult.assumptions`, plus a supporting and a refuting
witness when those exist.  Where a definitive verdict is impossible for
structural reasons (for example a parent absent from the pedigree), the result
is :py:data:`INSUFFICIENT` rather than an optimistic ``CONSISTENT``.

Scope and model
---------------

- Alleles are compared by VCF allele index.  A record's allele universe is
  ``0 .. len(record.ALT)``, so this is exact within one site.  For calls from
  different sites use :py:func:`screen_record` / :py:func:`screen_records`,
  which screen each site independently.
- Autosomal modes use Mendelian gamete transmission.  ``autosomal_dominant``
  and ``autosomal_recessive`` differ in phenotype interpretation, not in
  which genotypes can be transmitted; without phenotype input both therefore
  perform the same genotype-level check.
- X-linked modes model male children as receiving X alleles from the mother
  only and female children as receiving one X allele from the father and the
  remainder from the mother.  If the child sex is unknown, both models are
  tried and the assumption is recorded.
- ``de_novo`` is compatible only if the child carries at least one allele that
  is absent from both parents.
- Polyploid children are supported.  For an even child ploidy ``p`` both
  parents contribute ``p / 2`` alleles.  For an odd ploidy one parent
  contributes ``floor(p / 2)`` and the other ``ceil(p / 2)``; both
  parent-of-origin assignments are considered.  Each transmitted gamete must
  be a sub-multiset of the parent's called genotype, so a parent cannot
  transmit an allele it does not carry.  When a polyploid gamete is required,
  the result records that assumption.

Example
-------

.. code-block:: python

    pedigree = Pedigree.from_trio("child", father="father", mother="mother")
    for result in screen_record(record, pedigree, "autosomal_recessive"):
        if result.conflict:
            print(result.explanation)
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Literal, Sequence

from vcfpy.record import Call, Record

__all__ = [
    "CONSISTENT",
    "CONFLICT",
    "INSUFFICIENT",
    "INSUFFICIENT_EVIDENCE",
    "PROVEN_CONSISTENT",
    "PROVEN_CONFLICT",
    "INHERITANCE_MODES",
    "PedigreeMember",
    "Pedigree",
    "ScreeningResult",
    "screen_calls",
    "screen_record",
    "screen_records",
    "screen_reader",
]


#: Outcome: every completion of the missing alleles is compatible
CONSISTENT = "consistent"
#: Outcome: no completion of the missing alleles is compatible
CONFLICT = "conflict"
#: Outcome: the available calls do not decide between the two
INSUFFICIENT = "insufficient_evidence"

#: Verbose aliases, useful in reports
PROVEN_CONSISTENT = CONSISTENT
PROVEN_CONFLICT = CONFLICT
INSUFFICIENT_EVIDENCE = INSUFFICIENT

#: Supported inheritance modes
INHERITANCE_MODES = (
    "autosomal_dominant",
    "autosomal_recessive",
    "x_linked_dominant",
    "x_linked_recessive",
    "de_novo",
)

Status = Literal["consistent", "conflict", "insufficient_evidence"]
AllelePolicy = Literal["required", "optional", "ignore"]


def _normalize_sex(value: Any) -> Literal["male", "female", "unknown"] | None:
    """Normalize a VCF ``Sex`` value to ``"male"``, ``"female"``, or ``"unknown"``"""
    if value is None or value == "":
        return None
    if isinstance(value, int):
        value = str(value)
    lowered = str(value).strip().lower()
    if lowered in ("1", "m", "male"):
        return "male"
    if lowered in ("2", "f", "female"):
        return "female"
    if lowered in ("0", "u", "unknown", "."):
        return "unknown"
    raise ValueError(f"Invalid pedigree sex {value!r}")


# ---------------------------------------------------------------------------
# Pedigree model
# ---------------------------------------------------------------------------


@dataclass
class PedigreeMember:
    """One member of a pedigree

    :param name: sample name as used in the VCF
    :param father: sample name of the father or ``None``
    :param mother: sample name of the mother or ``None``
    :param sex: ``"male"``, ``"female"``, ``"unknown"``, or ``None``; needed
        for X-linked inheritance
    """

    name: str
    father: str | None = None
    mother: str | None = None
    sex: Literal["male", "female", "unknown"] | None = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("Pedigree member must have a non-empty name")
        if self.sex not in (None, "male", "female", "unknown"):
            raise ValueError(f'Invalid sex {self.sex!r}, must be "male", "female", "unknown", or None')
        if self.father == self.name or self.mother == self.name:
            raise ValueError(f"Pedigree member {self.name!r} cannot be its own parent")


class Pedigree:
    """A collection of pedigree members

    :param members: iterable of :py:class:`PedigreeMember`
    :raises ValueError: on duplicate names or references to unknown parents
    """

    def __init__(self, members: Iterable[PedigreeMember]):
        self.members: dict[str, PedigreeMember] = {}
        for member in members:
            if member.name in self.members:
                raise ValueError(f"Duplicate pedigree member {member.name!r}")
            self.members[member.name] = member
        for member in self.members.values():
            for parent in (member.father, member.mother):
                if parent is not None and parent not in self.members:
                    raise ValueError(f"Unknown parent {parent!r} of pedigree member {member.name!r}")

    @classmethod
    def from_trio(
        cls,
        child: str,
        father: str | None = None,
        mother: str | None = None,
        sex: Literal["male", "female", "unknown"] | None = None,
    ) -> "Pedigree":
        """Build a pedigree from one trio

        :param child: sample name of the child
        :param father: sample name of the father or ``None``
        :param mother: sample name of the mother or ``None``
        :param sex: sex of the child, needed for X-linked screening
        :raises ValueError: if both parents are ``None``
        """
        if father is None and mother is None:
            raise ValueError("A trio requires at least one parent")
        members = [PedigreeMember(child, father=father, mother=mother, sex=sex)]
        if father is not None:
            members.append(PedigreeMember(father, sex="male"))
        if mother is not None:
            members.append(PedigreeMember(mother, sex="female"))
        return cls(members)

    def __len__(self) -> int:
        return len(self.members)

    def __bool__(self) -> bool:
        return bool(self.members)

    @classmethod
    def from_header(cls, header: Any) -> "Pedigree":
        """Build a pedigree from ``##PEDIGREE`` header lines

        The VCF specification uses ``ID``, ``Father``, ``Mother``, and
        ``Sex`` (``1`` male, ``2`` female, ``0`` unknown).  Members that are
        only referenced as parents are created implicitly.

        :param header: a :py:class:`vcfpy.Header`
        :raises ValueError: on duplicate or inconsistent entries
        """
        members: dict[str, PedigreeMember] = {}
        for line in header.get_lines("PEDIGREE"):
            mapping = line.mapping
            name = mapping["ID"]
            members[name] = PedigreeMember(
                name,
                father=mapping.get("Father") or None,
                mother=mapping.get("Mother") or None,
                sex=_normalize_sex(mapping.get("Sex")),
            )
        for member in list(members.values()):
            for parent, sex in ((member.father, "male"), (member.mother, "female")):
                if parent is not None and parent not in members:
                    members[parent] = PedigreeMember(parent, sex=sex)
        return cls(members.values())

    def __iter__(self) -> Iterator[PedigreeMember]:
        return iter(self.members.values())

    def __contains__(self, name: object) -> bool:
        return name in self.members

    def get(self, name: str) -> PedigreeMember:
        """Return the member with the given name

        :raises KeyError: if the member is unknown
        """
        return self.members[name]


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScreeningResult:
    """Outcome of screening one child in one family at one site

    :param status: :py:data:`CONSISTENT`, :py:data:`CONFLICT`, or
        :py:data:`INSUFFICIENT`
    :param sample: name of the screened child
    :param inheritance: inheritance mode that was assumed
    :param pedigree: human-readable description of the relationship
    :param explanation: why the status was assigned
    :param assumptions: conditions on which the result depends; this includes
        the known-allele restriction and every completed missing call
    :param witnesses: a concrete compatible completion, if one exists
    :param counter_witnesses: a concrete incompatible completion, if one
        exists; this makes an ``INSUFFICIENT`` verdict auditable
    :param compatible_completions: number of compatible completions tested
    :param total_completions: total number of completions tested
    :param informative: whether any involved call carried an observed allele
    :param allele_universe: allele indices used to complete missing alleles
    :param sex_models: child-sex models considered; one entry for autosomal
        modes and for a known child sex, two for an unknown-sex X-linked screen
    """

    status: Status
    sample: str
    inheritance: str
    pedigree: str
    explanation: str
    assumptions: tuple[str, ...] = field(default_factory=tuple)
    witnesses: tuple[str, ...] = field(default_factory=tuple)
    counter_witnesses: tuple[str, ...] = field(default_factory=tuple)
    compatible_completions: int = 0
    total_completions: int = 0
    informative: bool = True
    allele_universe: tuple[int, ...] = field(default_factory=tuple)
    sex_models: tuple[str, ...] = field(default_factory=tuple)

    @property
    def consistent(self) -> bool:
        """Return ``True`` for :py:data:`CONSISTENT`"""
        return self.status == CONSISTENT

    @property
    def conflict(self) -> bool:
        """Return ``True`` for :py:data:`CONFLICT`"""
        return self.status == CONFLICT

    @property
    def insufficient(self) -> bool:
        """Return ``True`` for :py:data:`INSUFFICIENT`"""
        return self.status == INSUFFICIENT

    @property
    def proven(self) -> bool:
        """Return ``True`` if the result is decided (consistent or conflict)"""
        return self.status in (CONSISTENT, CONFLICT)

    def __bool__(self) -> bool:
        """Return ``True`` only for provably consistent relationships"""
        return self.consistent

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation"""
        return {
            "status": self.status,
            "sample": self.sample,
            "inheritance": self.inheritance,
            "pedigree": self.pedigree,
            "explanation": self.explanation,
            "assumptions": list(self.assumptions),
            "witnesses": list(self.witnesses),
            "counter_witnesses": list(self.counter_witnesses),
            "compatible_completions": self.compatible_completions,
            "total_completions": self.total_completions,
            "informative": self.informative,
            "allele_universe": list(self.allele_universe),
            "sex_models": list(self.sex_models),
        }


# ---------------------------------------------------------------------------
# Call state and completion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CallState:
    """A call normalized for screening"""

    sample: str
    has_call: bool
    has_gt: bool
    alleles: tuple[int | None, ...]
    observed: tuple[int, ...]
    ploidy: int | None
    fully_called: bool
    policy: AllelePolicy

    @property
    def partial(self) -> bool:
        return self.has_gt and not self.fully_called


def _resolve_call(call: Call | None, sample: str, policy: AllelePolicy) -> _CallState:
    """Normalize a :py:class:`vcfpy.Call` into a :py:class:`_CallState`"""
    if policy not in ("required", "optional", "ignore"):
        raise ValueError(f'Invalid allele policy {policy!r}, must be "required", "optional", or "ignore"')
    if call is None or not hasattr(call, "gt_alleles"):
        return _CallState(sample, has_call=False, has_gt=False, alleles=(), observed=(), ploidy=None,
                          fully_called=False, policy=policy)
    alleles = getattr(call, "gt_alleles", None)
    if alleles is None:
        return _CallState(sample, has_call=True, has_gt=False, alleles=(), observed=(), ploidy=None,
                          fully_called=False, policy=policy)
    raw = tuple(alleles)
    observed = tuple(a for a in raw if a is not None)
    return _CallState(
        sample,
        has_call=True,
        has_gt=True,
        alleles=raw,
        observed=observed,
        ploidy=len(raw),
        fully_called=getattr(call, "called", None) is True and len(observed) == len(raw),
        policy=policy,
    )


def _genotype_options(
    state: _CallState,
    universe: tuple[int, ...],
    model_ploidy: int,
) -> set[tuple[int, ...]]:
    """Return all genotypes the call may have

    Fully called genotypes yield one option.  Partially missing genotypes are
    completed position by position from ``universe``.  A call without any
    observed allele is completely unknown; its arity is taken from the GT
    field, except for a single ``"."`` which carries no reliable ploidy and
    therefore uses ``model_ploidy``.
    """
    if not state.has_gt or not state.observed:
        # A single "." carries no reliable ploidy information, so the model
        # ploidy is used.  "./." and "0/." keep their stated arity.
        ploidy = len(state.alleles) if len(state.alleles) > 1 else model_ploidy
        return set(itertools.combinations_with_replacement(universe, ploidy))
    if state.fully_called:
        return {tuple(sorted(state.observed))}
    missing_indices = [index for index, allele in enumerate(state.alleles) if allele is None]
    options: set[tuple[int, ...]] = set()
    for fill in itertools.product(universe, repeat=len(missing_indices)):
        completed = list(state.alleles)
        for index, allele in zip(missing_indices, fill):
            completed[index] = allele
        options.add(tuple(sorted(a for a in completed if a is not None)))
    return options


def _format_gt(alleles: Sequence[int | None]) -> str:
    return "/".join("." if allele is None else str(allele) for allele in alleles)


def _format_completed(alleles: Sequence[int]) -> str:
    return "/".join(str(allele) for allele in alleles)


# ---------------------------------------------------------------------------
# Mendelian transmission
# ---------------------------------------------------------------------------


def _gametes(genotype: Sequence[int], count: int) -> set[tuple[int, ...]]:
    """Return the multisets of ``count`` alleles a parent can transmit

    A gamete is a sub-multiset of the parent's genotype: every transmitted
    allele must have a copy in the parent, and no allele can be transmitted
    more often than the parent carries it.  This is the standard meiosis
    constraint without phase.  It also covers polyploid gametes, for example a
    diploid parent producing a diploid gamete for a tetraploid child.

    If ``count`` exceeds the parent's called ploidy, no gamete is possible at
    that ploidy; callers record the assumption and the enumeration yields no
    completion rather than an invented allele.

    Using multisets rather than sequences is sufficient because child
    genotypes are unphased.
    """
    if count == 0:
        return {()}
    if not genotype or count > len(genotype):
        return set()
    return {tuple(sorted(gamete)) for gamete in itertools.combinations(sorted(genotype), count)}


def _contains_multiset(container: Sequence[int], sub: Sequence[int]) -> bool:
    """Return whether ``container`` contains every copy in ``sub``"""
    counts = dict()
    for allele in container:
        counts[allele] = counts.get(allele, 0) + 1
    for allele in sub:
        if counts.get(allele, 0) == 0:
            return False
        counts[allele] -= 1
    return True


def _autosomal_compatible(
    child: tuple[int, ...],
    father: tuple[int, ...] | None,
    mother: tuple[int, ...] | None,
) -> bool:
    """Return whether the child genotype can be formed from the parents

    ``None`` denotes an unconstrained parent whose genotype is not known.
    """
    child_ploidy = len(child)
    splits = _contribution_splits(child_ploidy)
    for father_count, mother_count in splits:
        if father is None and mother is None:
            return True
        if father is None:
            for gamete in _gametes(mother or (), mother_count):
                if _contains_multiset(child, gamete):
                    return True
            continue
        if mother is None:
            for gamete in _gametes(father, father_count):
                if _contains_multiset(child, gamete):
                    return True
            continue
        for paternal in _gametes(father, father_count):
            for maternal in _gametes(mother, mother_count):
                if tuple(sorted(paternal + maternal)) == tuple(sorted(child)):
                    return True
    return False


def _contribution_splits(child_ploidy: int) -> list[tuple[int, int]]:
    """Return possible (paternal, maternal) gamete sizes

    Even ploidy is split evenly.  Odd ploidy is split ``floor``/``ceil`` and
    the reverse, because the parent of origin of the extra allele is not
    known.  A haploid child is modelled as receiving its allele from either
    parent.
    """
    if child_ploidy <= 0:
        return [(0, 0)]
    if child_ploidy == 1:
        return [(1, 0), (0, 1)]
    half = child_ploidy // 2
    if child_ploidy % 2 == 0:
        return [(half, half)]
    return [(half, half + 1), (half + 1, half)]


def _x_linked_model_compatible(
    child: tuple[int, ...],
    father: tuple[int, ...] | None,
    mother: tuple[int, ...] | None,
    sex: str | None,
) -> bool:
    """Return whether the child genotype fits one explicit X-linked model"""
    if sex == "male":
        # A male receives X alleles from the mother only.  The model also
        # works for an XXY male, hence mother_count == child ploidy.
        if mother is None:
            return True
        return any(
            tuple(sorted(gamete)) == tuple(sorted(child)) for gamete in _gametes(mother, len(child))
        )
    # Female: one X from the father and the remainder from the mother.
    maternal_count = max(len(child) - 1, 0)
    if father is None and mother is None:
        return True
    if father is None:
        # The paternal X is unconstrained, so it can supply any allele.  The
        # maternal contribution still has to be contained in the child.
        return any(_contains_multiset(child, gamete) for gamete in _gametes(mother or (), maternal_count))
    if mother is None:
        for paternal in _gametes(father, 1):
            if _contains_multiset(child, paternal):
                return True
        return False
    for paternal in _gametes(father, 1):
        for maternal in _gametes(mother, maternal_count):
            if tuple(sorted(paternal + maternal)) == tuple(sorted(child)):
                return True
    return False


def _de_novo_compatible(
    child: tuple[int, ...],
    father: tuple[int, ...] | None,
    mother: tuple[int, ...] | None,
) -> bool:
    """Return whether the child has at least one allele absent from both parents

    An unconstrained parent (``None``) could carry any allele, so it cannot
    prove a de novo event.  It is modelled conservatively: the known parent's
    alleles are removed, and a novel allele must remain after allowing the
    unknown parent to carry anything.  Since the unknown parent can always
    carry the child allele in question, an unconstrained parent makes a
    de novo proof impossible.
    """
    if father is None or mother is None:
        return False
    parental = set(father) | set(mother)
    return bool(set(child) - parental)


# ---------------------------------------------------------------------------
# Screening internals
# ---------------------------------------------------------------------------


def _model_ploidy(role: str, member: PedigreeMember, inheritance: str) -> int:
    """Ploidy to assume for a completely missing call"""
    if inheritance.startswith("x_linked"):
        if role == "father":
            return 1
        if role == "mother":
            return 2
        if member.sex == "male":
            return 1
    return 2


def _sex_hypotheses(member: PedigreeMember, inheritance: str) -> tuple[str | None, ...]:
    """Return the sex models that have to be considered"""
    if not inheritance.startswith("x_linked"):
        return (None,)
    if member.sex in ("male", "female"):
        return (member.sex,)
    return ("male", "female")


def _absent_parent_matters(member: PedigreeMember, inheritance: str) -> bool:
    """Return whether an absent parent could change the verdict

    For an X-linked male the father contributes no X allele, so his absence
    does not prevent a proof.  In every other supported model both parents
    can contribute, so an absent parent leaves an unquantified degree of
    freedom.
    """
    if inheritance.startswith("x_linked"):
        if member.sex == "male":
            return member.mother is None
        if member.sex == "female":
            return member.father is None or member.mother is None
        return member.father is None or member.mother is None
    return True


def _pedigree_description(member: PedigreeMember) -> str:
    parents = [parent for parent in (member.father, member.mother) if parent is not None]
    if not parents:
        return member.name
    return f"{member.name} (parents: {', '.join(parents)})"


def _witness(
    child_sample: str,
    child_gt: tuple[int, ...],
    parent_gts: dict[str, tuple[int, ...]],
    child_sex: str | None = None,
) -> tuple[str, ...]:
    lines = [f"child {child_sample}: {_format_completed(child_gt)}"]
    if child_sex is not None:
        lines.append(f"child sex model: {child_sex}")
    for name, genotype in parent_gts.items():
        lines.append(f"parent {name}: {_format_completed(genotype)}")
    return tuple(lines)


def _missing_genotype_assumption(
    state: _CallState,
    role: str,
    member: PedigreeMember,
    inheritance: str,
    universe: tuple[int, ...],
) -> str:
    ploidy = _model_ploidy(role, member, inheritance)
    alleles = ", ".join(str(allele) for allele in universe)
    return (
        f"parent {state.sample} has no genotype call and is treated as an unknown "
        f"ploidy-{ploidy} genotype over the known allele set {{{alleles}}}"
    )


def _assumptions_for_completion(
    child: _CallState,
    parent_states: list[tuple[str, _CallState]],
    universe: tuple[int, ...],
    member: PedigreeMember,
    inheritance: str,
) -> list[str]:
    assumptions: list[str] = []
    any_missing = (
        child.partial
        or not child.has_gt
        or any(parent.partial or not parent.has_gt for _, parent in parent_states)
    )
    if any_missing:
        assumptions.append(
            "missing alleles are restricted to the known allele set "
            "{" + ", ".join(str(allele) for allele in universe) + "}"
        )
    if child.partial:
        assumptions.append(
            f"child {child.sample} is partially called as {_format_gt(child.alleles)}; "
            "its missing allele is not treated as a conflict"
        )
    for name, parent in parent_states:
        role = "father" if member.father == name else "mother"
        if not parent.has_gt:
            assumptions.append(_missing_genotype_assumption(parent, role, member, inheritance, universe))
        elif parent.partial:
            assumptions.append(
                f"parent {name} is partially called as {_format_gt(parent.alleles)}; "
                "the missing allele is completed from the known allele set"
            )
    return assumptions


def _ploidy_assumptions(
    child: _CallState,
    parent_states: list[tuple[str, _CallState]],
    member: PedigreeMember,
    inheritance: str,
    child_ploidy: int,
) -> list[str]:
    assumptions: list[str] = []
    if child_ploidy == 1:
        assumptions.append(
            f"child {member.name} is called as haploid; under autosomal inheritance it "
            "is modelled as receiving its allele from either parent"
        )
    elif child_ploidy > 2:
        splits = _contribution_splits(child_ploidy)
        split_text = ", ".join(f"{father}+{mother}" for father, mother in splits)
        assumptions.append(
            f"child {member.name} is polyploid (ploidy {child_ploidy}); "
            f"parental contribution split(s) considered: {split_text} (paternal+maternal)"
        )
        if max(max(father, mother) for father, mother in splits) > 1:
            assumptions.append(
                "at least one parent is assumed to transmit a polyploid gamete "
                "(more than one allele), which is only possible if that parent "
                "actually carries enough copies"
            )
    if inheritance.startswith("x_linked") and member.sex not in ("male", "female"):
        assumptions.append(
            "child sex is unknown; the X-linked check accepts either a male "
            "(maternal-only) or a female (paternal and maternal) model"
        )
    for name, parent in parent_states:
        if parent.ploidy is not None and parent.ploidy > 2:
            assumptions.append(f"parent {name} is polyploid (ploidy {parent.ploidy})")
    return assumptions


def _conflict_explanation(
    member: PedigreeMember,
    child: _CallState,
    parent_states: list[tuple[str, _CallState]],
    inheritance: str,
    total: int,
) -> str:
    if total > 1:
        return (
            f"none of the {total} completions of the missing alleles is compatible with "
            f"{inheritance} inheritance"
        )
    child_observed = set(child.observed)
    parent_observed = {name: set(parent.observed) for name, parent in parent_states}
    union = set().union(*parent_observed.values()) if parent_observed else set()
    unexplained = sorted(child_observed - union)
    if unexplained:
        return (
            f"child {member.name} carries allele(s) {', '.join(str(a) for a in unexplained)} "
            f"that are absent from every parent call"
        )
    if inheritance == "de_novo":
        return (
            f"child {member.name} genotype {_format_gt(child.alleles)} is fully explained by "
            "the parental alleles; no de novo allele is present"
        )
    if inheritance.startswith("x_linked"):
        if member.sex in ("male", "female"):
            model = f"the {member.sex} model"
        else:
            model = "either the male or the female model"
        return (
            f"child {member.name} genotype {_format_gt(child.alleles)} cannot be transmitted "
            f"under {model}"
        )
    return (
        f"child {member.name} genotype {_format_gt(child.alleles)} cannot be formed from a "
        f"paternal and a maternal gamete under {inheritance} inheritance"
    )


def _compatibility(
    child_gt: tuple[int, ...],
    parent_gts: dict[str, tuple[int, ...]],
    member: PedigreeMember,
    inheritance: str,
    child_sex: str | None = None,
) -> bool:
    father = parent_gts.get(member.father) if member.father is not None else None
    mother = parent_gts.get(member.mother) if member.mother is not None else None
    if inheritance in ("autosomal_dominant", "autosomal_recessive"):
        return _autosomal_compatible(child_gt, father, mother)
    if inheritance in ("x_linked_dominant", "x_linked_recessive"):
        return _x_linked_model_compatible(child_gt, father, mother, child_sex)
    if inheritance == "de_novo":
        return _de_novo_compatible(child_gt, father, mother)
    raise ValueError(f"Unknown inheritance mode {inheritance!r}")  # pragma: no cover


def _screen_one(
    member: PedigreeMember,
    calls: dict[str, Call | None],
    inheritance: str,
    *,
    known_alleles: tuple[int, ...],
    father_policy: AllelePolicy,
    mother_policy: AllelePolicy,
    child_policy: AllelePolicy,
    max_completions: int,
) -> ScreeningResult:
    description = _pedigree_description(member)

    def result(status: Status, explanation: str, **kwargs: Any) -> ScreeningResult:
        return ScreeningResult(
            status=status,
            sample=member.name,
            inheritance=inheritance,
            pedigree=description,
            explanation=explanation,
            **kwargs,
        )

    if child_policy == "ignore":
        return result(INSUFFICIENT, f"child {member.name!r} is configured to be ignored", informative=False)
    child = _resolve_call(calls.get(member.name), member.name, child_policy)
    if not child.has_gt:
        return result(
            INSUFFICIENT,
            f"child {member.name!r} has no genotype call",
            assumptions=(f"sample {member.name} has no GT field",),
            informative=False,
        )

    parent_states: list[tuple[str, _CallState]] = []
    issues: list[str] = []
    ignored_parent = False
    absent_parent = False
    for name, policy in ((member.father, father_policy), (member.mother, mother_policy)):
        if name is None:
            absent_parent = True
            issues.append("one parent is absent from the pedigree and is modelled as unconstrained")
            continue
        if policy == "ignore":
            ignored_parent = True
            issues.append(f"parent {name!r} is excluded from the screen by configuration")
            continue
        state = _resolve_call(calls.get(name), name, policy)
        if not state.has_gt:
            # A missing call is not a conflict and not a free pass: it is
            # enumerated below.  "optional" only records that the absence is
            # acceptable for the caller; it does not weaken the enumeration.
            suffix = " (optional)" if policy == "optional" else ""
            issues.append(f"parent {name!r} has no genotype call{suffix}")
        parent_states.append((name, state))

    if not parent_states:
        return result(
            INSUFFICIENT,
            "no parent genotype is available for screening",
            assumptions=tuple(issues),
            informative=bool(child.observed),
        )
    if ignored_parent:
        # Excluding a parent removes half of the transmission model; even a
        # fully compatible remaining parent cannot prove the full pedigree.
        return result(
            INSUFFICIENT,
            "a parent is excluded by configuration, so the pedigree check is only partial",
            assumptions=tuple(issues),
            informative=bool(child.observed),
        )
    if not child.observed:
        return result(
            INSUFFICIENT,
            f"child {member.name!r} has no observed allele; a fully missing call proves nothing",
            assumptions=tuple(issues),
            informative=False,
        )
    child_ploidy = child.ploidy if child.ploidy else _model_ploidy("child", member, inheritance)
    if inheritance == "de_novo" and (
        len(parent_states) < 2 or any(not state.has_gt or state.partial for _, state in parent_states)
    ):
        return result(
            INSUFFICIENT,
            "a de novo screen requires both parents to be fully determined; "
            "an unknown parental allele could explain the child allele",
            assumptions=tuple(issues),
            informative=bool(child.observed),
        )

    # Allele universe: known alleles plus everything actually observed.
    universe = tuple(sorted(set(known_alleles) | set(child.observed) | {
        allele for _, parent in parent_states for allele in parent.observed
    }))
    if not universe:
        return result(
            INSUFFICIENT,
            "no known alleles and no observed alleles are available for completion",
            assumptions=tuple(issues),
            informative=False,
        )

    child_ploidy = len(child.alleles) if child.alleles else _model_ploidy("child", member, inheritance)
    child_options = _genotype_options(child, universe, _model_ploidy("child", member, inheritance))
    hypothesis_values = _sex_hypotheses(member, inheritance)
    if len(hypothesis_values) > 1:
        # Unknown sex is a hypothesis, not a free choice per completion.  Each
        # completion must be judged under one consistent sex assignment.
        child_options = {(gt, sex) for gt in child_options for sex in hypothesis_values}
    else:
        child_options = {(gt, hypothesis_values[0]) for gt in child_options}

    parent_options: list[tuple[str, _CallState, set[tuple[int, ...]]]] = []
    for name, state in parent_states:
        role = "father" if member.father == name else "mother"
        options = _genotype_options(state, universe, _model_ploidy(role, member, inheritance))
        parent_options.append((name, state, options))

    total = len(child_options)
    for _, _, options in parent_options:
        total *= len(options)
    if total == 0:
        return result(
            INSUFFICIENT,
            "no genotype completion is possible from the available allele universe",
            assumptions=tuple(issues),
        )
    if total > max_completions:
        return result(
            INSUFFICIENT,
            f"{total} completions exceed the configured limit of {max_completions}; "
            "refusing to claim a verdict without enumerating all completions",
            assumptions=tuple(issues),
        )

    assumptions = list(issues)
    assumptions.extend(_assumptions_for_completion(child, parent_states, universe, member, inheritance))
    assumptions.extend(_ploidy_assumptions(child, parent_states, member, inheritance, child_ploidy))

    parent_names = [name for name, _, _ in parent_options]
    option_lists = [sorted(options) for _, _, options in parent_options]
    compatible_count = 0
    witness: tuple[str, ...] = ()
    counter_witness: tuple[str, ...] = ()
    for child_gt, child_sex in sorted(child_options, key=lambda item: (item[0], str(item[1]))):
        for combination in itertools.product(*option_lists) if option_lists else [()]:
            parent_gts = dict(zip(parent_names, combination))
            if _compatibility(child_gt, parent_gts, member, inheritance, child_sex):
                compatible_count += 1
                if not witness:
                    witness = _witness(member.name, child_gt, parent_gts, child_sex)
            elif not counter_witness:
                counter_witness = _witness(member.name, child_gt, parent_gts, child_sex)

    if compatible_count == total:
        if absent_parent and _absent_parent_matters(member, inheritance):
            # The absent parent is not part of the enumeration.  A compatible
            # child therefore only shows that the known parent is consistent,
            # not that the pedigree is: an unseen parent could be
            # incompatible.  Do not upgrade an existence witness to a proof.
            return result(
                INSUFFICIENT,
                "all enumerated completions are compatible, but a parent is absent "
                "from the pedigree; consistency of the full pedigree is not proven",
                assumptions=tuple(assumptions),
                witnesses=witness,
                compatible_completions=compatible_count,
                total_completions=total,
            )
        if len(hypothesis_values) > 1:
            # Every completion is compatible under both sex models only if
            # both models agree.  This is still a conditional statement.
            explanation = (
                f"all {total} completions are compatible under both possible child-sex "
                "models; consistency holds only under the recorded assumptions"
            )
        elif total == 1 and all(state.fully_called for _, state in parent_states) and child.fully_called:
            explanation = "all calls are complete and compatible with the pedigree"
        else:
            explanation = (
                f"all {total} completions of the missing alleles are compatible; "
                "consistency holds only under the recorded assumptions"
            )
        return result(
            CONSISTENT,
            explanation,
            assumptions=tuple(assumptions),
            witnesses=witness,
            compatible_completions=compatible_count,
            total_completions=total,
            informative=True,
            allele_universe=universe,
            sex_models=tuple(sex for sex in hypothesis_values if sex is not None),
        )

    if compatible_count == 0:
        return result(
            CONFLICT,
            _conflict_explanation(member, child, parent_states, inheritance, total),
            assumptions=tuple(assumptions),
            counter_witnesses=counter_witness or _observed_witness(member, child, parent_states),
            compatible_completions=0,
            total_completions=total,
            allele_universe=universe,
            sex_models=tuple(sex for sex in hypothesis_values if sex is not None),
        )

    return result(
        INSUFFICIENT,
        f"only {compatible_count} of {total} completions are compatible; "
        "the missing allele is material to the verdict",
        assumptions=tuple(assumptions),
        witnesses=witness,
        counter_witnesses=counter_witness,
        compatible_completions=compatible_count,
        total_completions=total,
        allele_universe=universe,
        sex_models=tuple(sex for sex in hypothesis_values if sex is not None),
    )


def _observed_witness(
    member: PedigreeMember,
    child: _CallState,
    parent_states: list[tuple[str, _CallState]],
) -> tuple[str, ...]:
    lines = [f"child {member.name}: {_format_gt(child.alleles)}"]
    for name, state in parent_states:
        lines.append(f"parent {name}: {_format_gt(state.alleles)}")
    return tuple(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def screen_calls(
    calls: dict[str, Call | None],
    pedigree: Pedigree,
    inheritance: str,
    *,
    known_alleles: Sequence[int] | None = None,
    samples: Iterable[str] | None = None,
    father_alleles: AllelePolicy = "required",
    mother_alleles: AllelePolicy = "required",
    child_alleles: AllelePolicy = "required",
    max_completions: int = 100_000,
) -> list[ScreeningResult]:
    """Screen genotype calls for consistency with a pedigree

    :param calls: mapping from sample name to :py:class:`vcfpy.Call`; a
        missing sample may be omitted or mapped to ``None``
    :param pedigree: :py:class:`Pedigree` with the relationships
    :param inheritance: one of :py:data:`INHERITANCE_MODES`
    :param known_alleles: allele indices that may be used to complete a
        missing allele.  Observed alleles are always added.  If omitted, the
        observed alleles plus the reference allele ``0`` form the universe.
    :param samples: children to screen; defaults to every pedigree member
        that has at least one parent
    :param father_alleles: how to treat a father call

        - ``"required"`` (default): an absent call is still enumerated as an
          unknown genotype, and its absence is recorded in the assumptions
        - ``"optional"``: the same enumeration, but the absence is recorded
          as acceptable rather than as a problem
        - ``"ignore"``: exclude the parent; a full pedigree proof is then
          impossible and the result is ``INSUFFICIENT``

        In none of these modes is an absent allele treated as a conflict by
        itself; conversely, an absent call never silently disappears from the
        quantification.
    :param mother_alleles: like ``father_alleles`` for the mother
    :param child_alleles: like ``father_alleles`` for the child; a missing or
        ignored child call always yields ``INSUFFICIENT``
    :param max_completions: safety limit for the enumeration; exceeding it
        yields ``INSUFFICIENT`` rather than an unchecked claim
    :return: one :py:class:`ScreeningResult` per screened child
    :raises KeyError: if an explicitly requested sample is not in the pedigree
    """
    if inheritance not in INHERITANCE_MODES:
        raise ValueError(f"Unknown inheritance mode {inheritance!r}, expected one of {', '.join(INHERITANCE_MODES)}")
    for name, policy in (
        ("father_alleles", father_alleles),
        ("mother_alleles", mother_alleles),
        ("child_alleles", child_alleles),
    ):
        if policy not in ("required", "optional", "ignore"):
            raise ValueError(f'Invalid {name} policy {policy!r}, must be "required", "optional", or "ignore"')
    observed: set[int] = set()
    for call in calls.values():
        if call is not None and getattr(call, "gt_alleles", None):
            observed |= {allele for allele in call.gt_alleles if allele is not None}
    if known_alleles is None:
        # Allele 0 is the VCF reference allele and always exists at a variant
        # site.  Omitting it would make completions artificially narrow and
        # could turn an unproven relationship into a false proof.
        universe = tuple(sorted(observed | {0}))
    else:
        universe = tuple(sorted(set(known_alleles) | observed))
    if samples is None:
        samples = [member.name for member in pedigree if member.father or member.mother]
    results: list[ScreeningResult] = []
    for sample in samples:
        if sample not in pedigree:
            raise KeyError(f"Sample {sample!r} is not in the pedigree")
        member = pedigree.get(sample)
        results.append(
            _screen_one(
                member,
                calls,
                inheritance,
                known_alleles=universe,
                father_policy=father_alleles,
                mother_policy=mother_alleles,
                child_policy=child_alleles,
                max_completions=max_completions,
            )
        )
    return results


def screen_record(
    record: Record,
    pedigree: Pedigree,
    inheritance: str,
    *,
    samples: Iterable[str] | None = None,
    father_alleles: AllelePolicy = "required",
    mother_alleles: AllelePolicy = "required",
    child_alleles: AllelePolicy = "required",
    max_completions: int = 100_000,
) -> list[ScreeningResult]:
    """Screen one VCF record

    The allele universe is ``0 .. len(record.ALT)``, so a missing allele may
    be completed with any allele present at this site.  Records with
    symbolic or structural alleles are not meaningful for this screen; use
    :py:func:`screen_records` and filter such records first if necessary.
    """
    calls: dict[str, Call | None] = {}
    for call in record.calls:
        calls[getattr(call, "sample", None)] = call  # type: ignore[index]
    known_alleles = tuple(range(len(record.ALT) + 1))
    return screen_calls(
        calls,
        pedigree,
        inheritance,
        known_alleles=known_alleles,
        samples=samples,
        father_alleles=father_alleles,
        mother_alleles=mother_alleles,
        child_alleles=child_alleles,
        max_completions=max_completions,
    )


def screen_records(
    records: Iterable[Record],
    pedigree: Pedigree,
    inheritance: str,
    *,
    samples: Iterable[str] | None = None,
    father_alleles: AllelePolicy = "required",
    mother_alleles: AllelePolicy = "required",
    child_alleles: AllelePolicy = "required",
    max_completions: int = 100_000,
) -> list[tuple[Record, list[ScreeningResult]]]:
    """Screen a stream of records

    :return: list of ``(record, results)`` pairs, preserving input order
    """
    return [
        (
            record,
            screen_record(
                record,
                pedigree,
                inheritance,
                samples=samples,
                father_alleles=father_alleles,
                mother_alleles=mother_alleles,
                child_alleles=child_alleles,
                max_completions=max_completions,
            ),
        )
        for record in records
    ]


def screen_reader(
    reader: Any,
    inheritance: str,
    *,
    pedigree: Pedigree | None = None,
    samples: Iterable[str] | None = None,
    father_alleles: AllelePolicy = "required",
    mother_alleles: AllelePolicy = "required",
    child_alleles: AllelePolicy = "required",
    max_completions: int = 100_000,
) -> Iterator[tuple[Record, list[ScreeningResult]]]:
    """Screen every record of a :py:class:`vcfpy.Reader`

    The pedigree defaults to the ``##PEDIGREE`` header lines.  Records are
    yielded lazily as ``(record, results)`` pairs, so this can be used on
    large files without materializing all results.

    :param reader: an open :py:class:`vcfpy.Reader`
    :param inheritance: one of :py:data:`INHERITANCE_MODES`
    :param pedigree: optional pedigree; defaults to
        :py:meth:`Pedigree.from_header`
    """
    if pedigree is None:
        pedigree = Pedigree.from_header(reader.header)
        if not pedigree:
            raise ValueError("No pedigree given and no ##PEDIGREE header lines found")
    for record in reader:
        yield record, screen_record(
            record,
            pedigree,
            inheritance,
            samples=samples,
            father_alleles=father_alleles,
            mother_alleles=mother_alleles,
            child_alleles=child_alleles,
            max_completions=max_completions,
        )
