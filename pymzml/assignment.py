#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Precursor-to-molecule assignment with uncertain charge state.

The charge state of an isolated precursor is often not known a priori. This
module therefore enumerates every feasible interpretation of a precursor over
the full charge x adduct grid before any attribution decision is made. A
feasible interpretation links one precursor to one candidate molecule through
a concrete charge state, adduct convention and mass error, provided the
theoretical m/z falls inside the molecule's declared mass window.

Attribution then operates on the bipartite precursor <-> molecule graph:

* a connected component containing exactly one molecule is a *unique claim*
  (every precursor in it is explained by that molecule, even when several
  charge/adduct variants remain feasible);
* a component containing several molecules is an *ambiguous group* and is
  reported in full instead of being silently resolved;
* precursors without any feasible interpretation are reported as *unassigned*.

Whether a single isolation event may host several molecules is declared by
the caller via ``allow_multiple_molecules_per_event``. When it is ``False``
(the default), isolation events act as hyper-nodes that merge every molecule
matched by their precursors into one component, so two molecules competing
within one event can never both be uniquely claimed.
"""

# Python mzML module - pymzml
# Copyright (C) 2010-2019 M. Kösters, C. Fufezan
#     The MIT License (MIT)

#     Permission is hereby granted, free of charge, to any person obtaining a copy
#     of this software and associated documentation files (the "Software"), to deal
#     in the Software without restriction, including without limitation the rights
#     to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#     copies of the Software, and to permit persons to whom the Software is
#     furnished to do so, subject to the following conditions:

#     The above copyright notice and this permission notice shall be included in all
#     copies or substantial portions of the Software.

#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
#     IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
#     CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
#     TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
#     SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

PROTON_MASS = 1.00727646677
ELECTRON_MASS = 0.000548579909065

CONVENTION_PROTON = "proton"
CONVENTION_ELECTRON = "electron"
_VALID_CONVENTIONS = (CONVENTION_PROTON, CONVENTION_ELECTRON)

TOLERANCE_PPM = "ppm"
TOLERANCE_DA = "da"
_VALID_TOLERANCE_MODES = (TOLERANCE_PPM, TOLERANCE_DA)


@dataclass(frozen=True)
class Adduct:
    """
    An ionisation adduct convention.

    Arguments:
        name (str): human readable label, e.g. ``"[M+H]+"``.
        charge (int): signed charge state of the ionised molecule.
        mass_shift (float): neutral mass added or removed by the adduct in Da.
            Protons are included explicitly in the proton convention.
        convention (str): ``"proton"`` or ``"electron"``.
    """

    name: str
    charge: int
    mass_shift: float
    convention: str = CONVENTION_PROTON

    def __post_init__(self):
        if self.charge == 0:
            raise ValueError("Adduct charge must be non-zero")
        if self.convention not in _VALID_CONVENTIONS:
            raise ValueError(
                "Unknown adduct convention '{0}'".format(self.convention)
            )


@dataclass(frozen=True)
class CandidateMolecule:
    """
    A candidate molecule described by its neutral monoisotopic mass and the
    mass window around a theoretical m/z in which a match is accepted.

    Arguments:
        molecule_id (str): unique identifier.
        neutral_mass (float): neutral monoisotopic mass in Da.
        tolerance (float or None): window half-width; ``None`` falls back to
            the engine default.
        tolerance_mode (str or None): ``"ppm"`` or ``"da"``; ``None`` falls
            back to the engine default.
    """

    molecule_id: str
    neutral_mass: float
    tolerance: Optional[float] = None
    tolerance_mode: Optional[str] = None

    def __post_init__(self):
        if self.neutral_mass <= 0:
            raise ValueError(
                "Neutral mass of molecule '{0}' must be positive".format(
                    self.molecule_id
                )
            )
        if self.tolerance is not None and self.tolerance < 0:
            raise ValueError(
                "Tolerance of molecule '{0}' must be non-negative".format(
                    self.molecule_id
                )
            )
        if (
            self.tolerance_mode is not None
            and self.tolerance_mode not in _VALID_TOLERANCE_MODES
        ):
            raise ValueError(
                "Unknown tolerance mode '{0}'".format(self.tolerance_mode)
            )


@dataclass(frozen=True)
class Precursor:
    """
    An isolated precursor ion with an uncertain charge state.

    Arguments:
        precursor_id (str): unique precursor identifier.
        isolation_event_id (str): identifier of the isolation event the
            precursor belongs to. Precursors sharing an event are subject to
            the engine's multi-molecule policy.
        mz (float): observed precursor m/z (always positive).
        charges (tuple of int): candidate signed charge states to enumerate.
            A known charge is expressed as a one-element tuple.
        polarity (int): ``1`` for positive, ``-1`` for negative mode.
    """

    precursor_id: str
    isolation_event_id: str
    mz: float
    charges: tuple[int, ...]
    polarity: int = 1

    def __post_init__(self):
        if self.mz <= 0:
            raise ValueError(
                "Precursor '{0}' m/z must be positive".format(self.precursor_id)
            )
        if self.polarity not in (1, -1):
            raise ValueError(
                "Precursor '{0}' polarity must be 1 or -1".format(
                    self.precursor_id
                )
            )
        if not self.charges:
            raise ValueError(
                "Precursor '{0}' needs at least one candidate charge".format(
                    self.precursor_id
                )
            )
        for charge in self.charges:
            if charge == 0 or (charge > 0) != (self.polarity > 0):
                raise ValueError(
                    "Charge {0} of precursor '{1}' is inconsistent with "
                    "polarity {2}".format(charge, self.precursor_id, self.polarity)
                )


@dataclass(frozen=True)
class FeasibleInterpretation:
    """A single feasible precursor <-> molecule explanation."""

    precursor_id: str
    isolation_event_id: str
    molecule_id: str
    charge: int
    adduct: str
    convention: str
    observed_mz: float
    theoretical_mz: float
    mass_error_da: float
    mass_error_ppm: float
    tolerance: float
    tolerance_mode: str

    def as_dict(self) -> dict:
        return {
            "precursor_id": self.precursor_id,
            "isolation_event_id": self.isolation_event_id,
            "molecule_id": self.molecule_id,
            "charge": self.charge,
            "adduct": self.adduct,
            "convention": self.convention,
            "observed_mz": self.observed_mz,
            "theoretical_mz": self.theoretical_mz,
            "mass_error_da": self.mass_error_da,
            "mass_error_ppm": self.mass_error_ppm,
            "tolerance": self.tolerance,
            "tolerance_mode": self.tolerance_mode,
        }


@dataclass
class UniqueClaim:
    """
    An unambiguous molecule attribution.

    All precursors in :attr:`precursor_ids` are explained by exactly one
    molecule. ``charge_adduct_ambiguous`` flags claims where more than one
    charge/adduct variant remains feasible, so the molecule is unique even
    though its ionisation interpretation is not.
    """

    molecule: CandidateMolecule
    precursor_ids: list[str]
    isolation_event_ids: list[str]
    interpretations: list[FeasibleInterpretation]
    charge_adduct_ambiguous: bool

    def as_dict(self) -> dict:
        return {
            "molecule_id": self.molecule.molecule_id,
            "precursor_ids": list(self.precursor_ids),
            "isolation_event_ids": list(self.isolation_event_ids),
            "charge_adduct_ambiguous": self.charge_adduct_ambiguous,
            "interpretations": [item.as_dict() for item in self.interpretations],
        }


@dataclass
class AmbiguousGroup:
    """A connected set of precursors and molecules that cannot be disentangled."""

    group_id: int
    molecule_ids: list[str]
    precursor_ids: list[str]
    isolation_event_ids: list[str]
    interpretations: list[FeasibleInterpretation]
    reasons: list[str]

    def as_dict(self) -> dict:
        return {
            "group_id": self.group_id,
            "molecule_ids": list(self.molecule_ids),
            "precursor_ids": list(self.precursor_ids),
            "isolation_event_ids": list(self.isolation_event_ids),
            "reasons": list(self.reasons),
            "interpretations": [item.as_dict() for item in self.interpretations],
        }


@dataclass
class UnassignedPrecursor:
    """A precursor for which no charge/adduct/molecule combination is feasible."""

    precursor: Precursor
    reason: str

    def as_dict(self) -> dict:
        return {
            "precursor_id": self.precursor.precursor_id,
            "isolation_event_id": self.precursor.isolation_event_id,
            "mz": self.precursor.mz,
            "charges": list(self.precursor.charges),
            "polarity": self.precursor.polarity,
            "reason": self.reason,
        }


@dataclass
class AssignmentResult:
    """Full outcome of an assignment run."""

    unique: list[UniqueClaim] = field(default_factory=list)
    ambiguous_groups: list[AmbiguousGroup] = field(default_factory=list)
    unassigned: list[UnassignedPrecursor] = field(default_factory=list)
    interpretations: list[FeasibleInterpretation] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "unique": [claim.as_dict() for claim in self.unique],
            "ambiguous_groups": [group.as_dict() for group in self.ambiguous_groups],
            "unassigned": [item.as_dict() for item in self.unassigned],
            "interpretations": [item.as_dict() for item in self.interpretations],
        }


def protonation_ladder(
    charges: Iterable[int], convention: str = CONVENTION_PROTON
) -> list[Adduct]:
    """
    Build the default protonation (or radical electron) adduct ladder.

    Arguments:
        charges (iterable of int): signed candidate charges.
        convention (str): ``"proton"`` yields ``[M+nH]n+`` / ``[M-nH]n-``
            style adducts, ``"electron"`` yields radical ``M+`` / ``M-``
            adducts carrying the electron mass correction.

    Returns:
        list(Adduct): one adduct per distinct absolute charge state.
    """
    if convention not in _VALID_CONVENTIONS:
        raise ValueError("Unknown adduct convention '{0}'".format(convention))
    adducts = []
    for charge in sorted(set(charges), key=abs):
        if convention == CONVENTION_PROTON:
            sign = "+" if charge > 0 else "-"
            proton_count = abs(charge)
            if proton_count == 1:
                name = "[M{0}H]{0}".format(sign)
            else:
                name = "[M{0}{1}H]{1}{0}".format(sign, proton_count)
            adducts.append(
                Adduct(
                    name=name,
                    charge=charge,
                    mass_shift=proton_count
                    * PROTON_MASS
                    * (1 if charge > 0 else -1),
                    convention=convention,
                )
            )
        else:
            name = "M{0}".format("+" if charge > 0 else "-")
            adducts.append(
                Adduct(
                    name=name,
                    charge=charge,
                    mass_shift=0.0,
                    convention=convention,
                )
            )
    return adducts


# Common singly charged adducts offered for custom adduct lists.
COMMON_ADDUCTS = [
    Adduct("[M+H]+", 1, PROTON_MASS),
    Adduct("[M+Na]+", 1, 22.989218),
    Adduct("[M+NH4]+", 1, 18.033823),
    Adduct("[M+K]+", 1, 38.963158),
    Adduct("[M-H]-", -1, -PROTON_MASS),
    Adduct("[M+Cl]-", -1, 34.969401),
    Adduct("[M+HCOO]-", -1, 44.998201),
    Adduct("M+", 1, 0.0, convention=CONVENTION_ELECTRON),
    Adduct("M-", -1, 0.0, convention=CONVENTION_ELECTRON),
]


def theoretical_mz(
    neutral_mass: float, adduct: Adduct
) -> float:
    """
    Calculate the theoretical m/z of a molecule ionised with a given adduct.

    Proton convention: ``m/z = (M + mass_shift) / |z|``.
    Electron convention: the electron mass is removed (positive ion) or added
    (negative ion) on top of the adduct shift.
    """
    if adduct.convention == CONVENTION_PROTON:
        ion_mass = neutral_mass + adduct.mass_shift
    else:
        ion_mass = neutral_mass + adduct.mass_shift - adduct.charge * ELECTRON_MASS
    return ion_mass / abs(adduct.charge)


def _within_window(
    observed_mz: float, expected_mz: float, tolerance: float, mode: str
) -> tuple[bool, float, float]:
    error_da = observed_mz - expected_mz
    error_ppm = error_da / expected_mz * 1e6
    if mode == TOLERANCE_PPM:
        feasible = abs(error_ppm) <= tolerance
    else:
        feasible = abs(error_da) <= tolerance
    return feasible, error_da, error_ppm


class AssignmentEngine:
    """
    Enumerate feasible precursor interpretations and attribute molecules.

    Arguments:
        molecules (list of CandidateMolecule): candidate molecule catalogue.
        adduct_convention (str): default convention used when no explicit
            adduct list is supplied.
        adducts (list, callable or None): fixed list of :class:`Adduct`
            objects, a callable mapping a precursor to its adduct list, or
            ``None`` to derive the default ladder from each precursor's
            candidate charges.
        default_tolerance (float): window half-width used for molecules that
            do not declare their own.
        default_tolerance_mode (str): ``"ppm"`` or ``"da"``.
        allow_multiple_molecules_per_event (bool): whether one isolation
            event is allowed to host more than one claimed molecule.
    """

    def __init__(
        self,
        molecules: Iterable[CandidateMolecule],
        adduct_convention: str = CONVENTION_PROTON,
        adducts: Optional[
            Iterable[Adduct] | Callable[[Precursor], list[Adduct]]
        ] = None,
        default_tolerance: float = 5.0,
        default_tolerance_mode: str = TOLERANCE_PPM,
        allow_multiple_molecules_per_event: bool = False,
    ):
        if adduct_convention not in _VALID_CONVENTIONS:
            raise ValueError(
                "Unknown adduct convention '{0}'".format(adduct_convention)
            )
        if default_tolerance_mode not in _VALID_TOLERANCE_MODES:
            raise ValueError(
                "Unknown tolerance mode '{0}'".format(default_tolerance_mode)
            )
        if default_tolerance < 0:
            raise ValueError("default_tolerance must be non-negative")
        self.molecules = list(molecules)
        self._check_molecule_ids()
        self.adduct_convention = adduct_convention
        if adducts is None or isinstance(adducts, Iterable):
            self._fixed_adducts = None if adducts is None else list(adducts)
            self._adduct_callable = None
        else:
            self._fixed_adducts = None
            self._adduct_callable = adducts
        self.default_tolerance = default_tolerance
        self.default_tolerance_mode = default_tolerance_mode
        self.allow_multiple_molecules_per_event = (
            allow_multiple_molecules_per_event
        )

    def _check_molecule_ids(self):
        seen = set()
        for molecule in self.molecules:
            if molecule.molecule_id in seen:
                raise ValueError(
                    "Duplicate molecule id '{0}'".format(molecule.molecule_id)
                )
            seen.add(molecule.molecule_id)

    def _adducts_for(self, precursor: Precursor) -> list[Adduct]:
        if self._adduct_callable is not None:
            adducts = list(self._adduct_callable(precursor))
        elif self._fixed_adducts is not None:
            adducts = list(self._fixed_adducts)
        else:
            adducts = protonation_ladder(
                precursor.charges, convention=self.adduct_convention
            )
        candidate_charges = set(precursor.charges)
        selected = []
        for adduct in adducts:
            if adduct.charge not in candidate_charges:
                continue
            if (adduct.charge > 0) != (precursor.polarity > 0):
                continue
            selected.append(adduct)
        return selected

    def _tolerance_for(self, molecule: CandidateMolecule) -> tuple[float, str]:
        tolerance = (
            molecule.tolerance
            if molecule.tolerance is not None
            else self.default_tolerance
        )
        mode = (
            molecule.tolerance_mode
            if molecule.tolerance_mode is not None
            else self.default_tolerance_mode
        )
        return tolerance, mode

    def enumerate_interpretations(
        self, precursors: Iterable[Precursor]
    ) -> list[FeasibleInterpretation]:
        """Enumerate every feasible interpretation across the full grid."""
        interpretations = []
        for precursor in precursors:
            for adduct in self._adducts_for(precursor):
                for molecule in self.molecules:
                    expected_mz = theoretical_mz(
                        molecule.neutral_mass, adduct
                    )
                    tolerance, mode = self._tolerance_for(molecule)
                    feasible, error_da, error_ppm = _within_window(
                        precursor.mz, expected_mz, tolerance, mode
                    )
                    if not feasible:
                        continue
                    interpretations.append(
                        FeasibleInterpretation(
                            precursor_id=precursor.precursor_id,
                            isolation_event_id=precursor.isolation_event_id,
                            molecule_id=molecule.molecule_id,
                            charge=adduct.charge,
                            adduct=adduct.name,
                            convention=adduct.convention,
                            observed_mz=precursor.mz,
                            theoretical_mz=expected_mz,
                            mass_error_da=error_da,
                            mass_error_ppm=error_ppm,
                            tolerance=tolerance,
                            tolerance_mode=mode,
                        )
                    )
        return interpretations

    def assign(self, precursors: Iterable[Precursor]) -> AssignmentResult:
        """Run the full enumeration and attribution workflow."""
        precursors = list(precursors)
        self._check_precursor_ids(precursors)
        interpretations = self.enumerate_interpretations(precursors)

        molecule_by_id = {
            item.molecule_id: item for item in self.molecules
        }

        edges: dict[str, set[str]] = {}
        matched_precursor_ids = set()
        for interpretation in interpretations:
            precursor_node = "p::{0}".format(interpretation.precursor_id)
            molecule_node = "m::{0}".format(interpretation.molecule_id)
            edges.setdefault(precursor_node, set()).add(molecule_node)
            edges.setdefault(molecule_node, set()).add(precursor_node)
            matched_precursor_ids.add(interpretation.precursor_id)

        if not self.allow_multiple_molecules_per_event:
            event_members: dict[str, list[str]] = {}
            for precursor in precursors:
                if precursor.precursor_id not in matched_precursor_ids:
                    continue
                event_members.setdefault(
                    precursor.isolation_event_id, []
                ).append(precursor.precursor_id)
            for event_id, member_ids in event_members.items():
                if len(member_ids) < 2:
                    continue
                event_node = "e::{0}".format(event_id)
                for precursor_id in member_ids:
                    precursor_node = "p::{0}".format(precursor_id)
                    edges.setdefault(event_node, set()).add(precursor_node)
                    edges.setdefault(precursor_node, set()).add(event_node)

        components = self._connected_components(edges)

        result = AssignmentResult(interpretations=interpretations)
        group_counter = 0
        for component in components:
            precursor_ids = sorted(
                node.split("::", 1)[1]
                for node in component
                if node.startswith("p::")
            )
            molecule_ids = sorted(
                node.split("::", 1)[1]
                for node in component
                if node.startswith("m::")
            )
            event_ids = sorted(
                node.split("::", 1)[1]
                for node in component
                if node.startswith("e::")
            )
            component_interpretations = [
                interpretation
                for interpretation in interpretations
                if interpretation.precursor_id in precursor_ids
                and interpretation.molecule_id in molecule_ids
            ]
            if len(molecule_ids) == 1:
                variants = {
                    (item.charge, item.adduct)
                    for item in component_interpretations
                }
                result.unique.append(
                    UniqueClaim(
                        molecule=molecule_by_id[molecule_ids[0]],
                        precursor_ids=precursor_ids,
                        isolation_event_ids=sorted(
                            {
                                item.isolation_event_id
                                for item in component_interpretations
                            }
                        ),
                        interpretations=component_interpretations,
                        charge_adduct_ambiguous=len(variants) > 1,
                    )
                )
            else:
                group_counter += 1
                result.ambiguous_groups.append(
                    AmbiguousGroup(
                        group_id=group_counter,
                        molecule_ids=molecule_ids,
                        precursor_ids=precursor_ids,
                        isolation_event_ids=event_ids
                        or sorted(
                            {
                                item.isolation_event_id
                                for item in component_interpretations
                            }
                        ),
                        interpretations=component_interpretations,
                        reasons=self._ambiguity_reasons(
                            component_interpretations
                        ),
                    )
                )

        for precursor in precursors:
            if precursor.precursor_id not in matched_precursor_ids:
                result.unassigned.append(
                    UnassignedPrecursor(
                        precursor=precursor,
                        reason=(
                            "no feasible charge/adduct/molecule combination "
                            "within the declared mass windows"
                        ),
                    )
                )
        return result

    @staticmethod
    def _check_precursor_ids(precursors: list[Precursor]):
        seen = set()
        for precursor in precursors:
            if precursor.precursor_id in seen:
                raise ValueError(
                    "Duplicate precursor id '{0}'".format(
                        precursor.precursor_id
                    )
                )
            seen.add(precursor.precursor_id)

    @staticmethod
    def _connected_components(edges: dict[str, set[str]]) -> list[set[str]]:
        visited: set[str] = set()
        components = []
        for start in edges:
            if start in visited:
                continue
            stack = [start]
            component = set()
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)
                stack.extend(edges.get(node, set()) - visited)
            components.append(component)
        return components

    @staticmethod
    def _ambiguity_reasons(
        interpretations: list[FeasibleInterpretation],
    ) -> list[str]:
        reasons = set()
        molecules_per_precursor: dict[str, set[str]] = {}
        molecules_per_event: dict[str, set[str]] = {}
        for item in interpretations:
            molecules_per_precursor.setdefault(
                item.precursor_id, set()
            ).add(item.molecule_id)
            molecules_per_event.setdefault(
                item.isolation_event_id, set()
            ).add(item.molecule_id)
        if any(len(ids) > 1 for ids in molecules_per_precursor.values()):
            reasons.add("precursor matches multiple candidate molecules")
        if any(len(ids) > 1 for ids in molecules_per_event.values()):
            reasons.add(
                "multiple molecules matched within one isolation event"
            )
        if not reasons:
            reasons.add("precursors share candidate molecules")
        return sorted(reasons)


def assign(
    precursors: Iterable[Precursor],
    molecules: Iterable[CandidateMolecule],
    adduct_convention: str = CONVENTION_PROTON,
    adducts: Optional[
        Iterable[Adduct] | Callable[[Precursor], list[Adduct]]
    ] = None,
    default_tolerance: float = 5.0,
    default_tolerance_mode: str = TOLERANCE_PPM,
    allow_multiple_molecules_per_event: bool = False,
) -> AssignmentResult:
    """Functional convenience wrapper around :class:`AssignmentEngine`."""
    engine = AssignmentEngine(
        molecules=molecules,
        adduct_convention=adduct_convention,
        adducts=adducts,
        default_tolerance=default_tolerance,
        default_tolerance_mode=default_tolerance_mode,
        allow_multiple_molecules_per_event=(
            allow_multiple_molecules_per_event
        ),
    )
    return engine.assign(precursors)
